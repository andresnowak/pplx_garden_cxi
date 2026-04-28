import dataclasses
import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

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
                "[rank=%d][%s] dispatch trailer buf_pos=%d -> expert=%d, dst_rank=%d, expert_offset=%d, weight=%.3f, src_token_idx=%d, expert_choice_k=%d",
                rank, label, pos, expert, dst_rank, offset, weight, tok_idx, expert_choice,
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
        "tokens_per_expert=%s sum_tokens_per_expert=%s "
        "num_recv_tokens_main=%s num_recv_efa_tokens=%s total_padded_tokens=%s "
        "max_padded_index=%s padded_index_out_of_bounds=%s",
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
        state["sum_tokens_per_expert"],
        state["num_recv_tokens_main"],
        state["num_recv_efa_tokens"],
        state["total_padded_tokens"],
        state["max_padded_index"],
        state["padded_index_out_of_bounds"],
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
                rank_data, # The dp_rank of that group
            ),
            device=device,
            restrict_to_dp_group=config.restrict_to_dp_group,
            restrict_to_local_experts=config.restrict_to_local_experts,
        )
        for rank_data in range(num_dp_groups) # Only create data for each different dp_group
    ]
    local_rank = rank_data[dp_rank] # we grab the data from our rank's dp_group
    ref_out_tokens = _act(local_rank.dp_x, local_rank.dp_x_scale).to(out_dtype)

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

    print(f"[rank={global_group.rank}] Starting all-to-all with config: {config}", flush=True)

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
            torch.cuda.synchronize()

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

            # _decode_dispatch_trailers(all_to_all, global_group.rank, local_rank.indices, logger)

            # state = all_to_all.debug_state(
            #     max_token_offsets=max_num_tokens,
            #     max_recv_entries=max_recv_tokens,
            # )
            # print(
            #     f"rank {torch.distributed.get_rank()} repetition {rep + 1} kernel debug_state "
            #     f"num_recv_tokens={state['num_recv_tokens']} "
            #     f"expert_offsets={state['expert_offsets']} "
            #     f"token_offset={state['token_offset']} "
            #     f"padded_index={state['padded_index']} "
            #     f"combine_send_offset={state['combine_send_offset']} "
            #     f"source_dispatch_offset={state['source_dispatch_offset']} "
            #     f"source_rank={state['source_rank']}"
            #     f"tokens_per_expert={state['tokens_per_expert']} "
            #     f"sum_tokens_per_expert={state['sum_tokens_per_expert']} "
            #     f"num_recv_tokens_main={state['num_recv_tokens_main']} "
            #     f"num_recv_efa_tokens={state['num_recv_efa_tokens']} "
            #     f"total_padded_tokens={state['total_padded_tokens']} "
            #     f"max_padded_index={state['max_padded_index']} "
            #     f"padded_index_out_of_bounds={state['padded_index_out_of_bounds']} ",
            #     flush=True,
            # )

            # ---------------------

            local_num_tokens = expert_num_tokens.sum().item()

            # This test for TP is not correct
            # if tp_group != None and tp_group.size > 1 and local_num_tokens > 0:
            #     try:
            #         print("Gathering out_expert_x for debugging...")
            #         all_out_expert_x = [torch.empty_like(out_expert_x) for _ in range(tp_group.size)]
            #         torch.distributed.all_gather(all_out_expert_x, out_expert_x, group=tp_group._device_group)

            #         for i in range(tp_group.size):
            #             print(
            #                 f"rank {torch.distributed.get_rank()} Comparing gathered out_expert_x from rank {i} on dp_rank {dp_rank}...\n out_expert_x (shape: {out_expert_x.shape}) mean={out_expert_x.mean().item():.4f} std={out_expert_x.std().item():.4f} min={out_expert_x.min().item():.4f} max={out_expert_x.max().item():.4f} values={out_expert_x.tolist()}\n vs \n all_out_expert_x[{i}] (shape: {all_out_expert_x[i].shape}) mean={all_out_expert_x[i].mean().item():.4f} std={all_out_expert_x[i].std().item():.4f} min={all_out_expert_x[i].min().item():.4f} max={all_out_expert_x[i].max().item():.4f} values={all_out_expert_x[i].tolist()}",
            #             )

            #             (
            #                 torch.testing.assert_close(all_out_expert_x[i], out_expert_x),
            #                 f"Mismatch in gathered out_expert_x on rank {dp_rank} from rank {i}",
            #             )
            #     except AssertionError as e:
            #         print(f"rank {torch.distributed.get_rank()} Error in gathered out_expert_x on dp_rank {dp_rank} from rank {i} at repetition {rep + 1}/{repetitions}. \n Error: {e}.", flush=True)

            expert_y = _act(out_expert_x, out_expert_x_scale).to(out_dtype)
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
            torch.testing.assert_close(expected_local_tokens, expert_num_tokens.to("cpu"))

            # Verify the tokens.
            def hash_token(x: torch.Tensor) -> str:
                return ",".join(f"{v:.2f}" for v in x.tolist())

            tokens_on_rank = set()
            index = 0
            for n in expected_local_tokens.tolist():
                for token in out_expert_x[index : index + n]:
                    tokens_on_rank.add(hash_token(token))
                index = round_up(index + n, config.expert_padding)

            # Verify the tokens on the rank.
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
                        msg_parts.append(f"    token={tok[:40]}  prob={prob}  count={cnt}")
                if extra:
                    msg_parts.append(f"  EXTRA ({len(extra)} entries):")
                    for (tok, prob), cnt in list(extra.items())[:20]:
                        msg_parts.append(f"    token={tok[:40]}  prob={prob}  count={cnt}")

                # Find tokens that have both routes landing on this rank (multi-route tokens).
                msg_parts.append("  Tokens with multiple routes to this rank:")
                for rank_d in rank_data:
                    for token, routes, weights in zip(
                        rank_d.dp_x.tolist(), rank_d.indices.tolist(), rank_d.weights.tolist()
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
                    repetitions
                )
                assert received_token_probs == expected_token_probs

            # Verify the combine output.
            try:
                torch.testing.assert_close(out_tokens, ref_out_tokens)
                print(f"[rank={global_group.rank}] All-to-all test passed for repetition {rep + 1}/{repetitions}")
            except AssertionError as e:
                out_cpu = out_tokens.cpu()
                ref_cpu = ref_out_tokens.cpu()
                bad_rows = (out_cpu - ref_cpu).abs().any(dim=1).nonzero(as_tuple=True)[0]
                for row in bad_rows.tolist():
                    num_diff = (out_cpu[row] != ref_cpu[row]).sum().item()
                    logger.error(
                        "[rank=%d] rep=%d bad row=%d num_diff=%d/%d",
                        global_group.rank, rep + 1, row, num_diff, out_cpu.shape[1],
                    )
                raise AssertionError(
                    f"[rank={global_group.rank}] repetition {rep + 1}/{repetitions} failed {config.id}: "
                    f"out_tokens mean={out_tokens.mean().item():.4f} std={out_tokens.std().item():.4f} "
                    f"min={out_tokens.min().item():.4f} max={out_tokens.max().item():.4f} mode={out_tokens.mode().values} \n | ref mean={ref_out_tokens.mean().item():.4f} std={ref_out_tokens.std().item():.4f} "
                    f"min={ref_out_tokens.min().item():.4f} max={ref_out_tokens.max().item():.4f} mode={ref_out_tokens.mode().values}\n | local_rank indices: {local_rank.indices.tolist()} out_tokens={out_tokens.tolist()} ref_out_tokens={ref_out_tokens.tolist()}"
                    f"{e}"
                ) from None

            # print(f"[rank={global_group.rank}] Completed all-to-all repetition {rep + 1}/{repetitions} out_tokens={out_tokens.tolist()}, ref_out_tokens={ref_out_tokens.tolist()} local_rank indices: {local_rank.indices.tolist()}", flush=True)

            print(f"[rank={global_group.rank}] Completed all-to-all repetition {rep + 1}/{repetitions}", flush=True)
    
            # global_group.barrier()
            # all_to_all.destroy() # NOTE: Fixes the problem if we also create a new all_to_all at the beginning of the loop
            # assert all_to_all._all_to_all is not None
            # all_to_all._all_to_all.wait_ready()
            # all_to_all._global_group.barrier()
            # all_to_all._all_to_all.reset_counters()


    except Exception:
        logger.exception("All-to-all failed")
        raise
    finally:
        logger.info("Stopping all-to-all")
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
        expert_num_tokens = torch.empty(
            (num_local_experts,),
            dtype=torch.int32,
            device=device,
        )
        out_expert_x = torch.empty(
            (max_recv_tokens, config.hidden_dim),
            dtype=config.in_dtype,
            device=device,
        )
        out_expert_prob = torch.empty(
            (max_recv_tokens,),
            dtype=torch.float32,
            device=device,
        )
        out_tokens = torch.empty(
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
        _debug_tensor_stats(global_group.rank, "dispatch out_expert_prob", out_expert_prob)
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

        if valid_recv_tokens < expert_y.shape[0]:
            poisoned_expert_y = expert_y.to(config.out_dtype).clone()
            poison_value = torch.tensor(1024.0, dtype=config.out_dtype, device=device)
            poisoned_expert_y[valid_recv_tokens:] = poison_value
            poisoned_out_tokens = torch.empty_like(out_tokens)
            _dump_kernel_debug_state(all_to_all, global_group.rank, "poisoned_before_combine")
            all_to_all.combine(
                out_tokens=poisoned_out_tokens,
                indices=local_rank.indices,
                weights=torch.ones_like(local_rank.weights),
                expert_y=poisoned_expert_y,
                bound_m=local_rank.bound_m,
            )
            torch.cuda.synchronize()
            poison_diff = (poisoned_out_tokens - out_tokens).abs()
            logger.warning(
                "[pplx-test-debug][rank=%d] tail poison check max_diff=%s changed_positions=%d poisoned_tail_rows=%d",
                global_group.rank,
                float(poison_diff.max().item()),
                int(torch.count_nonzero(poison_diff).item()),
                poisoned_expert_y.shape[0] - valid_recv_tokens,
            )
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
                max_num_tokens=32,
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
            marks=[pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices")],
            id="TP4-DP2-NVL2",
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
        # NOTE: It seems it doesn't like NVL = DP_size, but this should be valid because we want the ETP and EP in that node to be NVLink, so I'm not sure what this is doing
        # pytest.param(
        #     _Config(
        #         world_size=4,
        #         dp_size=2,
        #         nets_per_gpu=get_nets_per_gpu(),
        #         max_num_tokens=8,
        #         num_experts=4,
        #         hidden_dim=4,
        #         hidden_dim_scale=None,
        #         max_private_tokens=None,
        #         num_experts_per_token=4,
        #         in_dtype=torch.float32,
        #         out_dtype=torch.float32,
        #         scale_dtype=None,
        #         expert_padding=1,
        #         nvlink_group=4,
        #     ),
        #     marks=[pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices")],
        #     id="TP4-DP2-NVL4",
        # ),
        # pytest.param(
        #     _Config(
        #         world_size=4,
        #         dp_size=2,
        #         nets_per_gpu=get_nets_per_gpu(),
        #         max_num_tokens=32,
        #         num_experts=4,
        #         hidden_dim=4,
        #         hidden_dim_scale=None,
        #         max_private_tokens=None,
        #         num_experts_per_token=4,
        #         in_dtype=torch.float32,
        #         out_dtype=torch.float32,
        #         scale_dtype=None,
        #         expert_padding=1,
        #         nvlink_group=4,
        #     ),
        #     marks=[pytest.mark.skipif(not has_tp(4), reason="Requires 4 devices")],
        #     id="TP4-DP2-NVL4-T1024",
        # ),
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
        # pytest.param(
        #     _Config(
        #         world_size=2,
        #         dp_size=1,
        #         nets_per_gpu=1,
        #         max_num_tokens=1,
        #         num_experts=2,
        #         hidden_dim=4,
        #         hidden_dim_scale=None,
        #         max_private_tokens=32,
        #         num_experts_per_token=1,
        #         in_dtype=torch.float32,
        #         out_dtype=torch.float32,
        #         scale_dtype=None,
        #         expert_padding=1,
        #         nvlink_group=None,
        #     ),
        #     marks=[pytest.mark.skipif(not has_tp(2), reason="Requires 2 devices")],
        #     id="TP2-EMPTY",
        # ),
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
