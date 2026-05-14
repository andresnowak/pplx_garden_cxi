# ruff: noqa: B023

import argparse
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from pplx_garden.distributed import ParallelGroup, ParallelLaunch
from pplx_garden.fabric_lib import (
    MemoryRegionDescriptor,
    ScatterTarget,
    TransferEngine,
)
from pplx_garden.utils import logging_utils
from pplx_garden.utils.math import Statistics
from pplx_garden.utils.torch import str_to_dtype

logger = logging_utils.get_logger("bench_scatter_write")

PAGE_SIZE = 4096


def round_up_page(n: int) -> int:
    return (n + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE


@dataclass(slots=True)
class ScatterWriteConfig:
    nets_per_gpu: int
    max_num_tokens: int
    num_experts: int
    hidden_dim: int
    num_experts_per_token: int
    in_dtype: torch.dtype

    @property
    def dispatch_bytes(self) -> int:
        """Total bytes dispatched per rank — matches AllToAllConfig.dispatch_bytes."""
        return (
            self.max_num_tokens
            * self.num_experts_per_token
            * self.hidden_dim
            * self.in_dtype.itemsize
        )


def _worker(
    device: torch.device,
    dp_group: Optional[ParallelGroup],
    global_group: Optional[ParallelGroup],
    cfg: ScatterWriteConfig,
    num_warmup: int,
    num_repeats: int,
    output: Path,
) -> None:
    assert global_group is not None

    if global_group.rank == 0:
        logging_utils.setup(level="DEBUG")

    try:
        _benchmark(device, global_group, cfg, num_warmup, num_repeats, output)
    finally:
        global_group.barrier()


def _benchmark(
    device: torch.device,
    group: ParallelGroup,
    cfg: ScatterWriteConfig,
    num_warmup: int,
    num_repeats: int,
    output: Path,
) -> None:
    rank = group.rank
    world_size = group.size
    gpu_id = device.index
    node_size = 4 # Fixed value for clariden
    groups = world_size // node_size
    assert groups * node_size == world_size, "world_size must be divisible by node_size"
    node_rank = rank // node_size # which node we are in

    engine = TransferEngine(nets_per_gpu=cfg.nets_per_gpu, cuda_devices=[gpu_id])

    # Build remote rank list early so per_dest_bytes uses the actual destination count.
    remote_ranks = [r for r in range(world_size) if r // node_size != node_rank]
    num_remote = len(remote_ranks)

    # Each rank sends dispatch_bytes total, split evenly across remote destinations.
    total_send_bytes = cfg.dispatch_bytes
    per_dest_bytes = total_send_bytes // max(num_remote, 1)
    per_dest_slot = round_up_page(per_dest_bytes)

    # Source buffer (what we scatter-write from).
    src_buf = torch.empty(per_dest_slot, dtype=torch.uint8, device=device)
    src_mr_handle, _ = engine.register_tensor(src_buf)

    # Destination buffer: one slot per remote rank, indexed by sender rank.
    dst_size = per_dest_slot * world_size
    dst_buf = torch.zeros(dst_size, dtype=torch.uint8, device=device)
    _, dst_mr_desc = engine.register_tensor(dst_buf)

    # Exchange destination descriptors across all ranks.
    all_dst_descs: list[MemoryRegionDescriptor] = group.all_gather_object(dst_mr_desc)

    # Build scatter targets: only ranks on OTHER nodes (intra-node uses NVLink, not fabric).
    dsts = [
        ScatterTarget(
            dst_mr=all_dst_descs[r],
            length=per_dest_bytes,
            src_offset=0,
            dst_offset=rank * per_dest_slot,
        )
        for r in remote_ranks
    ]

    send_times_us: list[float] = []
    rtt_times_us: list[float] = []

    IMM_BASE = 0x5C47

    for i in range(num_warmup + num_repeats):
        if i == num_warmup:
            torch.cuda.profiler.start()

        now = time.time()
        logger.info("Iteration %d/%d", i + 1, num_warmup + num_repeats)

        imm = (IMM_BASE + i) & 0xFFFFFFFF

        recv_done = threading.Event()
        send_done = threading.Event()

        # Arm IMM counter before barrier so no incoming write is missed.
        engine.set_imm_count_expected(imm, num_remote, recv_done.set)

        group.barrier()

        t_submit = time.perf_counter()

        engine.submit_scatter_writes(
            src_mr=src_mr_handle,
            dsts=dsts,
            imm_data=imm,
            on_done=send_done.set,
            on_error=lambda err: (_ for _ in ()).throw(RuntimeError(f"fabric-lib: {err}")),
        )

        send_done.wait()
        t_send_done = time.perf_counter()

        recv_done.wait()
        t_rtt_done = time.perf_counter()

        if i >= num_warmup:
            send_times_us.append((t_send_done - t_submit) * 1e6)
            rtt_times_us.append((t_rtt_done - t_submit) * 1e6)

    torch.cuda.profiler.stop()

    all_send_times: list[float] = sum(group.all_gather_object(send_times_us), [])
    all_rtt_times: list[float] = sum(group.all_gather_object(rtt_times_us), [])

    if rank == 0:
        stat_send = Statistics.create(all_send_times)
        stat_rtt = Statistics.create(all_rtt_times)

        # Bandwidth: total bytes injected per rank / latency
        actual_send_bytes = per_dest_bytes * num_remote
        send_bw_gbs = actual_send_bytes / (stat_send.p50 * 1e-6) / 1e9
        rtt_bw_gbs = actual_send_bytes / (stat_rtt.p50 * 1e-6) / 1e9

        logger.info(
            "dispatch_bytes=%.1f MB  per_dest=%.1f MB  world_size=%d",
            cfg.dispatch_bytes / 1e6,
            per_dest_bytes / 1e6,
            world_size,
        )
        logger.info(
            "Send (submit→NIC-done): %s  %.2f GB/s",
            stat_send,
            send_bw_gbs,
        )
        logger.info(
            "RTT  (submit→all-recv): %s  %.2f GB/s",
            stat_rtt,
            rtt_bw_gbs,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "scatter_write": {
                "send": {**asdict(stat_send), "bandwidth_gbs": send_bw_gbs},
                "rtt": {**asdict(stat_rtt), "bandwidth_gbs": rtt_bw_gbs},
            },
            "config": {
                "dispatch_bytes": cfg.dispatch_bytes,
                "per_dest_bytes": per_dest_bytes,
                "world_size": world_size,
                "max_num_tokens": cfg.max_num_tokens,
                "hidden_dim": cfg.hidden_dim,
                "num_experts_per_token": cfg.num_experts_per_token,
                "in_dtype": str(cfg.in_dtype),
            },
        }
        with output.open("w") as f:
            f.write(json.dumps(data))

    engine.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Scatter Write Benchmark")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--init-method", type=str, default=None)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--nets-per-gpu", type=int, default=2)
    parser.add_argument("--max-num-tokens", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-experts-per-token", type=int, default=8)
    parser.add_argument("--in-dtype", type=str_to_dtype, default=torch.float16)
    parser.add_argument("--num-warmup", type=int, default=20)
    parser.add_argument("--num-repeats", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("log_output_scatter_write/scatter_write.json"),
    )
    args = parser.parse_args()

    cfg = ScatterWriteConfig(
        nets_per_gpu=args.nets_per_gpu,
        max_num_tokens=args.max_num_tokens,
        num_experts=args.num_experts,
        hidden_dim=args.hidden_dim,
        num_experts_per_token=args.num_experts_per_token,
        in_dtype=args.in_dtype,
    )

    ParallelLaunch(
        world_size=args.world_size,
        init_method=args.init_method,
        dp_size=1,
        node_rank=args.node_rank,
    ).run(
        _worker,
        cfg,
        args.num_warmup,
        args.num_repeats,
        args.output,
    )


if __name__ == "__main__":
    main()
