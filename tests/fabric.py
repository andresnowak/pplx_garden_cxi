import os
from functools import cache
from pathlib import Path

import torch


def _count_path_entries(path: str, pattern: str) -> int:
    sysfs_path = Path(path)
    if not sysfs_path.exists():
        return 0
    return len(list(sysfs_path.glob(pattern)))


@cache
def count_visible_gpus() -> int:
    return torch.cuda.device_count()


@cache
def count_sys_nvidia() -> int:
    return max(
        count_visible_gpus(), _count_path_entries("/sys/bus/pci/drivers/nvidia/", "*:*")
    )


@cache
def count_sys_cxi() -> int:
    return _count_path_entries("/sys/class/cxi", "cxi*")


@cache
def count_sys_infiniband_verbs() -> int:
    return _count_path_entries("/sys/class/infiniband_verbs/", "uverbs*")


@cache
def count_network_endpoints() -> int:
    env_override = os.environ.get("PPLX_TEST_NETS_PER_GPU")
    if env_override is not None:
        gpu_count = count_visible_gpus()
        if gpu_count == 0:
            return 0
        return int(env_override) * gpu_count

    cxi_count = count_sys_cxi()
    if cxi_count > 0:
        return cxi_count

    return count_sys_infiniband_verbs()


def get_nets_per_gpu() -> int:
    gpu_count = count_sys_nvidia()
    if gpu_count == 0:
        return 0
    return count_network_endpoints() // gpu_count
