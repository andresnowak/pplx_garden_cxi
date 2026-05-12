import dataclasses
import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch.distributed as dist

from pplx_garden.distributed import ParallelGroup, ParallelLaunch
from pplx_garden.kernels.p2p_all_to_all import P2PAllToAll
from pplx_garden.native.cumem import CUMemMapping
from pplx_garden.utils import logging_utils
from pplx_garden.utils.math import round_up
from pplx_garden.utils.torch import has_tp
from tests.fabric import get_nets_per_gpu
from tests.markers import gpu_only, mark_ci_2gpu, mark_ci_4gpu, mark_fabric, mark_kernel
from tests.p2p_all_to_all.data import RankTestData

logger = logging_utils.get_logger(__name__)


def _parallel_launch_kwargs(world_size: int) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    local_device_count = torch.cuda.device_count()
    if world_size <= local_device_count:
        return kwargs

    init_method = os.environ.get("PPLX_TEST_INIT_METHOD")
    if init_method:
        kwargs["init_method"] = init_method
    node_rank = os.environ.get("PPLX_TEST_NODE_RANK")
    if node_rank is not None:
        kwargs["node_rank"] = int(node_rank)
    return kwargs


def require_nets_per_gpu(n: int) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        get_nets_per_gpu() < n,
        reason=f"requires {n} NICs per GPU, got {get_nets_per_gpu()}",
    )


@dataclass
class _Config:
    world_size: int
    dp_size: int
    nets_per_gpu: int
    max_num_tokens: int
    num_experts: int
    hidden_dim: int
    hidden_dim_scale: Optional[int]
    max_private_tokens: Optional[int]
    num_experts_per_token: int
    in_dtype: torch.dtype
    out_dtype: torch.dtype
    scale_dtype: Optional[torch.dtype]
    expert_padding: int
    nvlink_group: Optional[int]
    restrict_to_dp_group: bool = False
    restrict_to_local_experts: bool = False
    exclude_dp_group: bool = False
    repetitions: int = 4
    id: str = ""


def _act(x: torch.Tensor, x_scale: Optional[torch.Tensor]) -> torch.Tensor:
    if x_scale is None:
        return x * 2

    _, hidden_dim = x.shape
    _, hidden_dim_scale = x_scale.shape
    return x.to(torch.float32) * x_scale.repeat(1, hidden_dim // hidden_dim_scale) * 2


def _generator(device: torch.device, rank: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(rank)
    return generator


def _decode_dispatch_trailers(
    all_to_all: P2PAllToAll,
    rank: int,
    indices: torch.Tensor,
    log: logging.Logger,
) -> None:
    hidden_dim = all_to_all._hidden_dim
    in_dtype = all_to_all._in_dtype
    hidden_dim_scale = all_to_all._hidden_dim_scale
    scale_dtype = all_to_all._scale_dtype
    num_local_experts = all_to_all._num_local_experts
    experts_per_rank = num_local_experts
    world_size = all_to_all._global_group.size
    num_experts = world_size * num_local_experts

    token_dim = round_up(hidden_dim * in_dtype.itemsize, 16)
    if hidden_dim_scale is not None and scale_dtype is not None:
        token_dim += round_up(hidden_dim_scale * scale_dtype.itemsize, 16)
    token_stride = token_dim + 16

    # Build position -> (token_idx, k) from local indices
    indices_cpu = indices.cpu()
    expert_counts = torch.zeros(num_experts, dtype=torch.int32)
    for tok in range(indices_cpu.shape[0]):
        for k in range(indices_cpu.shape[1]):
            expert_counts[indices_cpu[tok, k].item()] += 1
    expert_offsets = torch.cumsum(expert_counts, dim=0)
    position_to_token: dict[int, tuple[int, int]] = {}
    expert_seen = torch.zeros(num_experts, dtype=torch.int32)
    for tok in range(indices_cpu.shape[0]):
        for k in range(indices_cpu.shape[1]):
            expert = indices_cpu[tok, k].item()
            base = expert_offsets[expert - 1].item() if expert > 0 else 0
            pos = base + expert_seen[expert].item()
            position_to_token[pos] = (tok, k)
            expert_seen[expert] += 1

    def decode_buf(mapping: CUMemMapping, label: str) -> None:
        buf = mapping.to_tensor((mapping.size,), torch.uint8).cpu()
        total_slots = mapping.size // token_stride
        for pos in range(total_slots):
            base = pos * token_stride
            trailer = buf[base + token_dim : base + token_dim + 16]
            t = trailer.view(torch.uint32)
            expert = t[0].item()
            offset = t[1].item()
            position = t[2].item()
            weight = trailer[12:16].view(torch.float32)[0].item()
            if weight == 0.0 and expert == 0 and offset == 0:
                continue
            dst_rank = expert // experts_per_rank
            token_info = position_to_token.get(position, (-1, -1))
            tok_idx, expert_choice = token_info
            log.warning(
                "[rank=%d][%s] dispatch trailer buf_pos=%d -> expert=%d, dst_rank=%d, expert_offset=%d (unreliable), position=%d (unreliable), weight=%.3f, src_token_idx=%d (unreliable), expert_choice_k=%d (unreliable)",
                rank,
                label,
                pos,
                expert,
                dst_rank,
                offset,
                position,
                weight,
                tok_idx,
                expert_choice,
            )

    decode_buf(all_to_all._send_buffer_mapping, "send")
    decode_buf(all_to_all._recv_buffer_mapping, "recv")


def _debug_tensor_stats(rank: int, label: str, tensor: torch.Tensor) -> None:
    detached = tensor.detach()
    logger.warning(
        "[pplx-test-debug][rank=%d] %s shape=%s dtype=%s min=%s max=%s mean=%s",
        rank,
        label,
        tuple(detached.shape),
        detached.dtype,
        detached.min().item(),
        detached.max().item(),
        detached.float().mean().item(),
    )


def _dump_kernel_debug_state(
    all_to_all: P2PAllToAll,
    rank: int,
    label: str,
    *,
    max_token_offsets: int = 64,
    max_recv_entries: int = 128,
) -> None:
    state = all_to_all.debug_state(
        max_token_offsets=max_token_offsets,
        max_recv_entries=max_recv_entries,
    )
    logger.warning(
        "[pplx-test-debug][rank=%d] %s kernel debug_state "
        "num_recv_tokens=%s expert_offsets=%s token_offset=%s padded_index=%s "
        "combine_send_offset=%s source_dispatch_offset=%s source_rank=%s "
        "tokens_per_expert=%s ",
        # "sum_tokens_per_expert=%s "
        # "num_recv_tokens_main=%s num_recv_efa_tokens=%s total_padded_tokens=%s "
        # "max_padded_index=%s padded_index_out_of_bounds=%s",
        rank,
        label,
        state["num_recv_tokens"],
        state["expert_offsets"],
        state["token_offset"],
        state["padded_index"],
        state["combine_send_offset"],
        state["source_dispatch_offset"],
        state["source_rank"],
        state["tokens_per_expert"],
        # state["sum_tokens_per_expert"],
        # state["num_recv_tokens_main"],
        # state["num_recv_efa_tokens"],
        # state["total_padded_tokens"],
        # state["max_padded_index"],
        # state["padded_index_out_of_bounds"],
    )


def _dump_combine_recv_buffer(
    all_to_all: P2PAllToAll,
    rank: int,
    label: str,
    *,
    max_slots: int = 36,
) -> None:
    slot_width = all_to_all._hidden_dim
    flat = all_to_all._recv_buffer_mapping.to_tensor(
        (all_to_all._recv_buffer_mapping.size // all_to_all._out_dtype.itemsize,),
        all_to_all._out_dtype,
    )
    num_slots = min(max_slots, flat.numel() // slot_width)
    if num_slots == 0:
        logger.warning(
            "[pplx-test-debug][rank=%d] %s recv_buffer is empty",
            rank,
            label,
        )
        return

    slots = flat[: num_slots * slot_width].view(num_slots, slot_width).cpu()
    logger.warning(
        "[pplx-test-debug][rank=%d] %s recv_buffer_slots=%s",
        rank,
        label,
        slots.tolist(),
    )


def _test_p2p_all_to_all_worker(
    device: torch.device,
    tp_group: Optional[ParallelGroup],
    global_group: Optional[ParallelGroup],
    config: _Config,
) -> None:
    assert tp_group is not None
    assert global_group is not None

    dp_rank = global_group.rank // tp_group.size
    num_dp_groups = global_group.size // tp_group.size

    max_num_tokens = config.max_num_tokens
    num_experts = config.num_experts
    hidden_dim = config.hidden_dim
    hidden_dim_scale = config.hidden_dim_scale
    num_experts_per_token = config.num_experts_per_token
    in_dtype = config.in_dtype
    out_dtype = config.out_dtype
    scale_dtype = config.scale_dtype
    repetitions = config.repetitions

    num_local_experts = num_experts // global_group.size
    first_expert = global_group.rank * num_local_experts
    last_expert = min(first_expert + num_local_experts, num_experts)

    max_recv_tokens = max_num_tokens * num_local_experts * num_dp_groups

    # Set up dummy test data.
    rank_data = [
        RankTestData.create(
            dp_rank=rank_data,
            dp_size=config.dp_size,
            world_size=global_group.size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            max_num_tokens=max_num_tokens,
            hidden_dim=hidden_dim,
            hidden_dim_scale=hidden_dim_scale,
            in_dtype=in_dtype,
            scale_dtype=scale_dtype,
            generator=_generator(
                device,
                rank_data,  # The dp_rank of that group
            ),
            device=device,
            restrict_to_dp_group=config.restrict_to_dp_group,
            restrict_to_local_experts=config.restrict_to_local_experts,
            exclude_dp_group=config.exclude_dp_group,
        )
        for rank_data in range(
            num_dp_groups
        )  # Only create data for each different dp_group
    ]
    local_rank = rank_data[dp_rank]  # we grab the data from our rank's dp_group
    ref_out_tokens = _act(local_rank.dp_x, local_rank.dp_x_scale).to(out_dtype)
    print(f"[rank{global_group.rank}, local_rank.dp_x: {local_rank.dp_x.tolist()}, ref_out_tokens: {ref_out_tokens.tolist()}")

    # Sanity check that the local input tokens are the same across TP ranks since we rely on that for correctness. If this fails, it means there's a bug in our test data generation logic for TP.
    tp_local_tokens = [
        torch.empty((max_num_tokens, hidden_dim), dtype=in_dtype, device=device)
        for _ in range(tp_group.size)
    ]
    dist.all_gather(tp_local_tokens, local_rank.dp_x, group=tp_group._device_group)
    if tp_group.size == 2:
        print("Gathered local dp_x across TP ranks for debugging...")
        assert torch.allclose(
            tp_local_tokens[0], tp_local_tokens[1]
        ), f"dp_x mismatch across tp ranks: {tp_local_tokens[0]} vs {tp_local_tokens[1]}"

    tp_local_weights = [
        torch.empty((max_num_tokens, num_experts_per_token), dtype=torch.float32, device=device)
        for _ in range(tp_group.size)
    ]
    dist.all_gather(tp_local_weights, local_rank.weights, group=tp_group._device_group)
    if tp_group.size == 2:
        print("Gathered local weights across TP ranks for debugging...")
        assert torch.allclose(
            tp_local_weights[0], tp_local_weights[1]
        ), f"weights mismatch across tp ranks: {tp_local_weights[0]} vs {tp_local_weights[1]}"

    tp_local_indices = [
         torch.empty((max_num_tokens, num_experts_per_token), dtype=torch.int32, device=device)
         for _ in range(tp_group.size)
     ]
    dist.all_gather(tp_local_indices, local_rank.indices.to(torch.int32), group=tp_group._device_group)
    if tp_group.size == 2:
        print("Gathered local indices across TP ranks for debugging...")
        assert torch.allclose(
            tp_local_indices[0], tp_local_indices[1]
        ), f"indices mismatch across tp ranks: {tp_local_indices[0]} vs {tp_local_indices[1]}"

    # –---------------------

    node_group: Optional[ParallelGroup]
    if config.nvlink_group is not None:
        assert config.nvlink_group > 0
        assert global_group.size % config.nvlink_group == 0
        node_group = global_group.slice_by_count(
            global_group.size // config.nvlink_group,
        )
    else:
        node_group = None

    # Instantiate the all-to-all kernel.
    all_to_all = P2PAllToAll(
        max_num_tokens=max_num_tokens,
        num_experts=num_experts,
        expert_padding=config.expert_padding,
        hidden_dim=hidden_dim,
        hidden_dim_scale=hidden_dim_scale,
        max_private_tokens=config.max_private_tokens,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        scale_dtype=scale_dtype,
        num_experts_per_token=num_experts_per_token,
        nets_per_gpu=config.nets_per_gpu,
        device=device,
        dp_group=tp_group,
        node_group=node_group,
        global_group=global_group,
    )

    print(
        f"[rank={global_group.rank}] Starting all-to-all with config: {config}",
        flush=True,
    )

    try:
        for rep in range(repetitions):
            logger.info("Starting all-to-all repetition %d/%d", rep + 1, repetitions)

            # all_to_all = P2PAllToAll(
            #     max_num_tokens=max_num_tokens,
            #     num_experts=num_experts,
            #     expert_padding=config.expert_padding,
            #     hidden_dim=hidden_dim,
            #     hidden_dim_scale=hidden_dim_scale,
            #     max_private_tokens=config.max_private_tokens,
            #     in_dtype=in_dtype,
            #     out_dtype=out_dtype,
            #     scale_dtype=scale_dtype,
            #     num_experts_per_token=num_experts_per_token,
            #     nets_per_gpu=config.nets_per_gpu,
            #     device=device,
            #     dp_group=tp_group,
            #     node_group=node_group,
            #     global_group=global_group,
            # )
            all_to_all.debug_poison_transport_buffers(value=0)
            # torch.cuda.synchronize()

            expected_num_tokens = torch.sum(
                torch.stack(
                    [data.expected_num_tokens for data in rank_data],
                    dim=0,
                ),
                dim=0,
                dtype=torch.int32,
            ).to("cpu")

            # Dispatch.
            expert_num_tokens = torch.empty(
                (num_local_experts,),
                dtype=torch.int32,
                device=device,
            )
            out_expert_x = torch.empty(
                (max_recv_tokens, hidden_dim),
                dtype=in_dtype,
                device=device,
            )
            out_expert_prob = torch.empty(
                (max_recv_tokens,),
                dtype=torch.float32,
                device=device,
            )
            out_tokens = torch.empty(
                (max_num_tokens, hidden_dim),
                dtype=out_dtype,
                device=device,
            )

            if hidden_dim_scale is not None or scale_dtype is not None:
                assert scale_dtype is not None
                assert hidden_dim_scale is not None
                out_expert_x_scale = torch.empty(
                    (max_recv_tokens, hidden_dim_scale),
                    dtype=scale_dtype,
                    device=device,
                )
            else:
                out_expert_x_scale = None

            # Test run.
            all_to_all.dispatch(
                out_expert_num_tokens=expert_num_tokens,
                out_expert_x=out_expert_x,
                out_expert_x_scale=out_expert_x_scale,
                dp_x=local_rank.dp_x,
                dp_x_scale=local_rank.dp_x_scale,
                indices=local_rank.indices,
                weights=local_rank.weights,
                bound_m=None,
                out_expert_prob=out_expert_prob,
            )

            # ---------------------

            if os.environ.get("PPLX_TEST_DEBUG") == "1":
                _decode_dispatch_trailers(
                    all_to_all, global_group.rank, local_rank.indices, logger
                )

                state = all_to_all.debug_state(
                    max_token_offsets=max_num_tokens,
                    max_recv_entries=max_recv_tokens,
                )
                print(
                    f"rank {torch.distributed.get_rank()} repetition {rep + 1} kernel debug_state "
                    f"num_recv_tokens={state['num_recv_tokens']} "
                    f"expert_offsets={state['expert_offsets']} "
                    f"token_offset={state['token_offset']} "
                    f"padded_index={state['padded_index']} "
                    f"combine_send_offset={state['combine_send_offset']} "
                    f"source_dispatch_offset={state['source_dispatch_offset']} "
                    f"source_rank={state['source_rank']}"
                    f"tokens_per_expert={state['tokens_per_expert']} "
                    f"sum_tokens_per_expert={state['sum_tokens_per_expert']} "
                    f"num_recv_tokens_main={state['num_recv_tokens_main']} "
                    f"num_recv_efa_tokens={state['num_recv_efa_tokens']} "
                    f"total_padded_tokens={state['total_padded_tokens']} "
                    f"max_padded_index={state['max_padded_index']} "
                    f"padded_index_out_of_bounds={state['padded_index_out_of_bounds']} ",
                    flush=True,
                )


            expert_y = _act(out_expert_x, out_expert_x_scale).to(out_dtype)
            if os.environ.get("PPLX_TEST_DEBUG_RECV_BUFFER") == "1":
                all_to_all.combine(
                    out_tokens=out_tokens,
                    indices=local_rank.indices,
                    weights=local_rank.weights,
                    expert_y=expert_y,
                    bound_m=local_rank.bound_m,
                    do_send=True,
                    do_recv=False,
                )
                torch.cuda.synchronize()
                _dump_combine_recv_buffer(
                    all_to_all,
                    global_group.rank,
                    f"after-combine-send rep={rep + 1}",
                )
                all_to_all.combine(
                    out_tokens=out_tokens,
                    indices=local_rank.indices,
                    weights=local_rank.weights,
                    expert_y=expert_y,
                    bound_m=local_rank.bound_m,
                    do_send=False,
                    do_recv=True,
                )
            else:
                all_to_all.combine(
                    out_tokens=out_tokens,
                    indices=local_rank.indices,
                    weights=local_rank.weights,
                    expert_y=expert_y,
                    bound_m=local_rank.bound_m,
                )
            torch.cuda.synchronize()

            # Verify the token counts.
            expected_local_tokens = expected_num_tokens[first_expert:last_expert]
            torch.testing.assert_close(
                expected_local_tokens, expert_num_tokens.to("cpu")
            )

            # Verify the tokens.
            def hash_token(x: torch.Tensor, decimals: int = 6) -> str:
                return ",".join(f"{v:.{decimals}f}" for v in x.tolist())

            tokens_on_rank = set()
            index = 0
            for n in expected_local_tokens.tolist():
                for token in out_expert_x[index : index + n]:
                    tokens_on_rank.add(hash_token(token))
                index = round_up(index + n, config.expert_padding)

            # Verify the tokens on the rank after dispatch if we are missing any.
            num_missing = 0
            for i, (token, routes) in enumerate(
                zip(list(local_rank.dp_x), local_rank.indices.tolist())
            ):
                if not any(first_expert <= route < last_expert for route in routes):
                    continue
                key = hash_token(token)
                if key not in tokens_on_rank:
                    num_missing += 1
                    logger.error(
                        "Token %i: %s not found in output on rank %i (routed to %s)",
                        i,
                        key,
                        dp_rank,
                        ", ".join(str(route) for route in routes),
                    )
            assert num_missing == 0, f"Missing {num_missing} tokens on rank {dp_rank}"

            # Verify routed probabilities align with the dispatched tokens.
            def hash_prob(prob: float) -> str:
                return f"{prob:.6f}"

            expected_token_probs = Counter()
            for rank in rank_data:
                for token, routes, weights in zip(
                    rank.dp_x.tolist(), rank.indices.tolist(), rank.weights.tolist()
                ):
                    token_key = hash_token(torch.tensor(token))
                    for route, weight in zip(routes, weights):
                        if first_expert <= route < last_expert:
                            expected_token_probs[(token_key, hash_prob(weight))] += 1

            received_token_probs = Counter()
            index = 0
            for n in expected_local_tokens.tolist():
                for token, prob in zip(
                    out_expert_x[index : index + n], out_expert_prob[index : index + n]
                ):
                    received_token_probs[
                        (hash_token(token), hash_prob(float(prob.item())))
                    ] += 1
                index = round_up(index + n, config.expert_padding)

            # If the dispatch was wrong (we didn't get the expected tokens) this should raise
            if received_token_probs != expected_token_probs:
                extra = received_token_probs - expected_token_probs
                missing = expected_token_probs - received_token_probs
                msg_parts = [
                    f"Token-prob mismatch on rank {dp_rank} "
                    f"(first_expert={first_expert}, last_expert={last_expert}):"
                ]
                if missing:
                    msg_parts.append(f"  MISSING ({len(missing)} entries):")
                    for (tok, prob), cnt in list(missing.items())[:20]:
                        msg_parts.append(
                            f"    token={tok[:40]}  prob={prob}  count={cnt}"
                        )
                if extra:
                    msg_parts.append(f"  EXTRA ({len(extra)} entries):")
                    for (tok, prob), cnt in list(extra.items())[:20]:
                        msg_parts.append(
                            f"    token={tok[:40]}  prob={prob}  count={cnt}"
                        )

                # Find tokens that have both routes landing on this rank (multi-route tokens).
                msg_parts.append("  Tokens with multiple routes to this rank:")
                for rank_d in rank_data:
                    for token, routes, weights in zip(
                        rank_d.dp_x.tolist(),
                        rank_d.indices.tolist(),
                        rank_d.weights.tolist(),
                    ):
                        local_routes = [
                            (r, w)
                            for r, w in zip(routes, weights)
                            if first_expert <= r < last_expert
                        ]
                        if len(local_routes) > 1:
                            tok_key = hash_token(torch.tensor(token))
                            msg_parts.append(
                                f"    token={tok_key[:40]}  routes+weights={local_routes}"
                            )

                # Dump the raw received (token, prob) pairs at each expert slot.
                msg_parts.append("  Received per-expert slots:")
                idx = 0
                for expert_i, n in enumerate(expected_local_tokens.tolist()):
                    for slot in range(n):
                        tok = out_expert_x[idx + slot]
                        prob = out_expert_prob[idx + slot]
                        msg_parts.append(
                            f"    expert={first_expert + expert_i} slot={slot} "
                            f"token={hash_token(tok)[:40]}  prob={hash_prob(float(prob.item()))}"
                        )
                    idx = round_up(idx + n, config.expert_padding)
                raise AssertionError("\n".join(msg_parts))

            if received_token_probs != expected_token_probs:
                logger.error(
                    "Token-prob mismatch on rank %d (first_expert=%d, last_expert=%d) at repetition %d/%d",
                    dp_rank,
                    first_expert,
                    last_expert,
                    rep + 1,
                    repetitions,
                )
                assert received_token_probs == expected_token_probs

            # The next assert can fail sometimes even though the result was correct
            # # Verify that the multiset of output tokens matches reference,
            # # regardless of ordering. If this passes but assert_close fails,
            # # the correct tokens are present but in the wrong positions.
            # out_token_counts = Counter(hash_token(row, decimals=3) for row in out_tokens)
            # ref_token_counts = Counter(hash_token(row, decimals=3) for row in ref_out_tokens)
            # if out_token_counts != ref_token_counts:
            #     missing_toks = ref_token_counts - out_token_counts
            #     extra_toks = out_token_counts - ref_token_counts
            #     msg_parts = [f"Combine output token multiset mismatch on rank: {global_group.rank}, dp_rank: {dp_rank}, at repetition {rep + 1}/{repetitions}:"]
            #     if missing_toks:
            #         msg_parts.append(f"  MISSING ({len(missing_toks)} distinct tokens):")
            #         for tok, cnt in list(missing_toks.items())[:20]:
            #             msg_parts.append(f"    token={tok[:40]}  count={cnt}")
            #     if extra_toks:
            #         msg_parts.append(f"  EXTRA ({len(extra_toks)} distinct tokens):")
            #         for tok, cnt in list(extra_toks.items())[:20]:
            #             msg_parts.append(f"    token={tok[:40]}  count={cnt}")
            #     # Check if extra tokens belong to another dp rank's reference output.
            #     other_rank_token_sets = {
            #         other_dp_rank: Counter(
            #             hash_token(_act(rank_data[other_dp_rank].dp_x, rank_data[other_dp_rank].dp_x_scale).to(out_dtype)[i], decimals=3)
            #             for i in range(rank_data[other_dp_rank].dp_x.shape[0])
            #         )
            #         for other_dp_rank in range(num_dp_groups)
            #         if other_dp_rank != dp_rank
            #     }
            #     for other_dp_rank, other_counts in other_rank_token_sets.items():
            #         overlap = extra_toks & other_counts
            #         if overlap:
            #             msg_parts.append(
            #                 f"  EXTRA tokens match dp_rank={other_dp_rank} reference ({len(overlap)} tokens) — combine returned wrong rank's data!"
            #             )
            #     raise AssertionError("\n".join(msg_parts))

            # Verify the combine output.
            try:
                torch.testing.assert_close(out_tokens, ref_out_tokens)
                print(
                    f"[rank={global_group.rank}] All-to-all test passed for repetition {rep + 1}/{repetitions}"
                )
            except AssertionError as e:
                out_cpu = out_tokens.cpu()
                ref_cpu = ref_out_tokens.cpu()
                bad_rows = (
                    (out_cpu - ref_cpu).abs().any(dim=1).nonzero(as_tuple=True)[0]
                )
                for row in bad_rows.tolist():
                    num_diff = (out_cpu[row] != ref_cpu[row]).sum().item()
                    logger.error(
                        "[rank=%d] rep=%d bad row=%d num_diff=%d/%d",
                        global_group.rank,
                        rep + 1,
                        row,
                        num_diff,
                        out_cpu.shape[1],
                    )
                raise AssertionError(
                    f"[rank={global_group.rank}] repetition {rep + 1}/{repetitions} failed {config.id}: "
                    f"out_tokens mean={out_tokens.mean().item():.4f} std={out_tokens.std().item():.4f} "
                    f"min={out_tokens.min().item():.4f} max={out_tokens.max().item():.4f} mode={out_tokens.mode().values} \n | ref mean={ref_out_tokens.mean().item():.4f} std={ref_out_tokens.std().item():.4f} "
                    f"min={ref_out_tokens.min().item():.4f} max={ref_out_tokens.max().item():.4f} mode={ref_out_tokens.mode().values}\n | local_rank indices: {local_rank.indices.tolist()} out_tokens={out_tokens.tolist()} ref_out_tokens={ref_out_tokens.tolist()}"
                    f"{e}"
                ) from None

            # print(f"[rank={global_group.rank}] Completed all-to-all repetition {rep + 1}/{repetitions} out_tokens={out_tokens.tolist()}, ref_out_tokens={ref_out_tokens.tolist()} local_rank indices: {local_rank.indices.tolist()}", flush=True)

            print(
                f"[rank={global_group.rank}] Completed all-to-all repetition {rep + 1}/{repetitions}",
                flush=True,
            )

            # global_group.barrier()
            # # all_to_all.destroy() # NOTE: Fixes the problem if we also create a new all_to_all at the beginning of the loop

            # # TODO: Still don't know if this is necessary
            assert all_to_all._all_to_all is not None
            # all_to_all._all_to_all.wait_ready()
            all_to_all._global_group.barrier() # TODO: you need a barrier because we need to wait for all ranks to have recv their tokens in the combine and read them for the test before starting a new dispatch that can rewrite the buffer
            # all_to_all._all_to_all.reset_counters()

    except Exception:
        logger.exception("All-to-all failed")
        raise
    finally:
        logger.info("Stopping all-to-all")
        if all_to_all is not None:
            all_to_all.destroy()


def _test_p2p_all_to_all_moe_roundtrip_worker(
    device: torch.device,
    tp_group: Optional[ParallelGroup],
    global_group: Optional[ParallelGroup],
    config: _Config,
) -> None:
    assert tp_group is not None
    assert global_group is not None

    dp_rank = global_group.rank // tp_group.size
    num_dp_groups = global_group.size // tp_group.size
    num_local_experts = config.num_experts // global_group.size
    max_recv_tokens = config.max_num_tokens * num_local_experts * num_dp_groups

    local_rank = RankTestData.create(
        dp_rank=dp_rank,
        dp_size=config.dp_size,
        world_size=global_group.size,
        num_experts=config.num_experts,
        num_experts_per_token=config.num_experts_per_token,
        max_num_tokens=config.max_num_tokens,
        hidden_dim=config.hidden_dim,
        hidden_dim_scale=config.hidden_dim_scale,
        in_dtype=config.in_dtype,
        scale_dtype=config.scale_dtype,
        generator=_generator(
            device,
            dp_rank,
        ),
        device=device,
        restrict_to_dp_group=config.restrict_to_dp_group,
        restrict_to_local_experts=config.restrict_to_local_experts,
    )

    node_group: Optional[ParallelGroup]
    if config.nvlink_group is not None:
        assert config.nvlink_group > 0
        assert global_group.size % config.nvlink_group == 0
        node_group = global_group.slice_by_count(
            global_group.size // config.nvlink_group
        )
    else:
        node_group = None

    all_to_all = P2PAllToAll(
        max_num_tokens=config.max_num_tokens,
        num_experts=config.num_experts,
        expert_padding=config.expert_padding,
        hidden_dim=config.hidden_dim,
        hidden_dim_scale=config.hidden_dim_scale,
        max_private_tokens=config.max_private_tokens,
        in_dtype=config.in_dtype,
        out_dtype=config.out_dtype,
        scale_dtype=config.scale_dtype,
        num_experts_per_token=config.num_experts_per_token,
        nets_per_gpu=config.nets_per_gpu,
        device=device,
        dp_group=tp_group,
        node_group=node_group,
        global_group=global_group,
    )

    try:
        expert_num_tokens = torch.zeros(
            (num_local_experts,),
            dtype=torch.int32,
            device=device,
        )
        out_expert_x = torch.zeros(
            (max_recv_tokens, config.hidden_dim),
            dtype=config.in_dtype,
            device=device,
        )
        out_expert_prob = torch.zeros(
            (max_recv_tokens,),
            dtype=torch.float32,
            device=device,
        )
        out_tokens = torch.zeros(
            (config.max_num_tokens, config.hidden_dim),
            dtype=config.out_dtype,
            device=device,
        )

        if config.hidden_dim_scale is not None or config.scale_dtype is not None:
            assert config.scale_dtype is not None
            assert config.hidden_dim_scale is not None
            out_expert_x_scale = torch.empty(
                (max_recv_tokens, config.hidden_dim_scale),
                dtype=config.scale_dtype,
                device=device,
            )
        else:
            out_expert_x_scale = None

        all_to_all.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=out_expert_x,
            out_expert_x_scale=out_expert_x_scale,
            dp_x=local_rank.dp_x,
            dp_x_scale=local_rank.dp_x_scale,
            indices=local_rank.indices,
            weights=local_rank.weights,
            out_expert_prob=out_expert_prob,
        )

        valid_recv_tokens = int(expert_num_tokens.sum().item())
        logger.warning(
            "[pplx-test-debug][rank=%d] dispatch end valid_recv_tokens=%d tokens_per_expert=%s out_expert_x_shape=%s",
            global_group.rank,
            valid_recv_tokens,
            expert_num_tokens.tolist(),
            tuple(out_expert_x.shape),
        )
        _debug_tensor_stats(global_group.rank, "dispatch out_expert_x", out_expert_x)
        _debug_tensor_stats(
            global_group.rank, "dispatch out_expert_prob", out_expert_prob
        )
        logger.warning(
            "[pplx-test-debug][rank=%d] dispatch tail_probs_nonzero=%d",
            global_group.rank,
            int(torch.count_nonzero(out_expert_prob[valid_recv_tokens:]).item()),
        )
        _dump_kernel_debug_state(all_to_all, global_group.rank, "baseline")

        expert_y = out_expert_x * out_expert_prob.unsqueeze(-1).to(out_expert_x.dtype)
        _debug_tensor_stats(global_group.rank, "combine expert_y", expert_y)
        all_to_all.combine(
            out_tokens=out_tokens,
            indices=local_rank.indices,
            weights=torch.ones_like(local_rank.weights),
            expert_y=expert_y.to(config.out_dtype),
            bound_m=local_rank.bound_m,
        )
        torch.cuda.synchronize()
    finally:
        all_to_all.destroy()

    torch.testing.assert_close(out_tokens, local_rank.dp_x.to(config.out_dtype))


@mark_fabric
@mark_kernel
@gpu_only
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=128,
                num_experts=16,
                hidden_dim=128,
                hidden_dim_scale=None,
                num_experts_per_token=2,
                max_private_tokens=None,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_2gpu,
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
            ],
            id="TP2-NIC1-FP32",
        ),
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=128,
                num_experts=16,
                hidden_dim=128,
                hidden_dim_scale=None,
                num_experts_per_token=2,
                max_private_tokens=8,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_2gpu,
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
            ],
            id="TP2-NIC1-FP32-MIXED",
        ),
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=1024,
                num_experts=16,
                hidden_dim=128,
                hidden_dim_scale=None,
                max_private_tokens=8,
                num_experts_per_token=2,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=2,
            ),
            marks=[
                mark_ci_2gpu,
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
            ],
            id="TP2-NIC1-FP32-NVL",
        ),
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=1024,
                num_experts=16,
                hidden_dim=128,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=2,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_2gpu,
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
            ],
            id="TP2-NIC1-BF16",
        ),
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=8,
                num_experts=16,
                hidden_dim=16,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=2,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=16,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_2gpu,
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
            ],
            id="TP2-NIC1-BF16-PADDED",
        ),
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=2,
                num_experts=16,
                hidden_dim=128,
                hidden_dim_scale=16,
                max_private_tokens=None,
                num_experts_per_token=2,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=torch.float32,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_2gpu,
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
            ],
            id="TP2-NIC1-FP8",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=128,
                num_experts=128,
                hidden_dim=128,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=8,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_4gpu,
                pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices"),
            ],
            id="TP4-NIC1-FP32",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=1,
                nets_per_gpu=2,
                max_num_tokens=128,
                num_experts=256,
                hidden_dim=7168,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=8,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_4gpu,
                pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices"),
                require_nets_per_gpu(2),
            ],
            id="TP4-NIC2-BF16",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=2,
                nets_per_gpu=1,
                max_num_tokens=1,
                num_experts=4,
                hidden_dim=8,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=1,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_4gpu,
                pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices"),
            ],
            id="TP4-DP2-NIC1-BF16",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=2,
                nets_per_gpu=1,
                max_num_tokens=1024,
                num_experts=4,
                hidden_dim=8,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=1,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
                restrict_to_dp_group=False,
            ),
            marks=[
                mark_ci_4gpu,
                pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices"),
            ],
            id="TP4-DP2-NIC1-BF16-T1024",
        ),
        pytest.param(
            _Config(
                world_size=8,
                dp_size=1,
                nets_per_gpu=get_nets_per_gpu(),
                max_num_tokens=128,
                num_experts=128,
                hidden_dim=7168,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=8,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[pytest.mark.skipif(not has_tp(8), reason="Requires 8 devices")],
            id="TP8-BF16",
        ),
        pytest.param(
            _Config(
                world_size=8,
                dp_size=1,
                nets_per_gpu=get_nets_per_gpu(),
                max_num_tokens=256,
                num_experts=128,
                hidden_dim=7168,
                hidden_dim_scale=56,
                max_private_tokens=None,
                num_experts_per_token=8,
                in_dtype=torch.float8_e4m3fn,
                out_dtype=torch.bfloat16,
                scale_dtype=torch.float32,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[pytest.mark.skipif(not has_tp(8), reason="Requires 8 devices")],
            id="TP8-FP8",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=1,
                nets_per_gpu=get_nets_per_gpu(),
                max_num_tokens=128,
                num_experts=128,
                hidden_dim=7168,
                hidden_dim_scale=56,
                max_private_tokens=None,
                num_experts_per_token=8,
                in_dtype=torch.float8_e4m3fn,
                out_dtype=torch.bfloat16,
                scale_dtype=torch.float32,
                expert_padding=1,
                nvlink_group=2,
            ),
            marks=[pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices")],
            id="TP4-FP8-NVL2",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=2,
                nets_per_gpu=get_nets_per_gpu(),
                max_num_tokens=8,
                num_experts=4,
                hidden_dim=4,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=4,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=2,
            ),
            marks=[
                pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices"),
            ],
            id="TP4-DP2-NVL2",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=2,
                nets_per_gpu=get_nets_per_gpu(),
                max_num_tokens=256,
                num_experts=4,
                hidden_dim=4,
                hidden_dim_scale=None,
                max_private_tokens=256,
                num_experts_per_token=4,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=2,
                # restrict_to_dp_group=True,
                # exclude_dp_group=True,
            ),
            marks=[
                pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices"),
            ],
            id="TP4-DP2-NVL2-T256",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=1,
                nets_per_gpu=get_nets_per_gpu(),
                max_num_tokens=1024,
                num_experts=4,
                hidden_dim=4,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=4,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=4,
            ),
            marks=[pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices")],
            id="TP4-DP1-NVL4",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=2,
                nets_per_gpu=get_nets_per_gpu(),
                max_num_tokens=256,
                num_experts=4,
                hidden_dim=4,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=4,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=4,
            ),
            marks=[
                pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices"),
            ],
            id="TP4-DP2-NVL4",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=2,
                nets_per_gpu=get_nets_per_gpu(),
                max_num_tokens=1024,
                num_experts=4,
                hidden_dim=4,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=4,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=4,
            ),
            marks=[
                pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices"),
            ],
            id="TP4-DP2-NVL4-T1024",
        ),
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=256,
                num_experts=8,
                hidden_dim=256,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=2,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_2gpu,
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
            ],
            id="TP2-NIC1-BF16-LARGE",
        ),
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=1,
                num_experts=2,
                hidden_dim=4,
                hidden_dim_scale=None,
                max_private_tokens=32,
                num_experts_per_token=1,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
                pytest.mark.skip(
                    reason="This configuration seems to be invalid or unstable, needs investigation"
                ),
            ],
            id="TP2-EMPTY",
        ),
    ],
)
def test_p2p_all_to_all(config: _Config, request: pytest.FixtureRequest) -> None:
    config = dataclasses.replace(config, id=request.node.name)
    ParallelLaunch(
        world_size=config.world_size,
        dp_size=config.dp_size,
        **_parallel_launch_kwargs(config.world_size),
    ).run(
        _test_p2p_all_to_all_worker,
        config,
    )


@mark_fabric
@mark_kernel
@gpu_only
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            _Config(
                world_size=2,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=32,
                num_experts=16,
                hidden_dim=32,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=2,
                in_dtype=torch.float32,
                out_dtype=torch.float32,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_2gpu,
                pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices"),
            ],
            id="TP2-MoE-Roundtrip-FP32",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=32,
                num_experts=16,
                hidden_dim=32,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=2,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_4gpu,
                pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices"),
            ],
            id="TP4-MoE-Roundtrip-BF16",
        ),
        pytest.param(
            _Config(
                world_size=4,
                dp_size=1,
                nets_per_gpu=1,
                max_num_tokens=256,
                num_experts=8,
                hidden_dim=32,
                hidden_dim_scale=None,
                max_private_tokens=None,
                num_experts_per_token=2,
                in_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                scale_dtype=None,
                expert_padding=1,
                nvlink_group=None,
            ),
            marks=[
                mark_ci_4gpu,
                pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices"),
            ],
            id="TP4-MoE-Roundtrip-BF16-LARGE",
        ),
    ],
)
def test_p2p_all_to_all_moe_roundtrip(config: _Config) -> None:
    ParallelLaunch(
        world_size=config.world_size,
        dp_size=config.dp_size,
        **_parallel_launch_kwargs(config.world_size),
    ).run(
        _test_p2p_all_to_all_moe_roundtrip_worker,
        config,
    )
