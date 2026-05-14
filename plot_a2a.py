import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

LOG_DIR = Path("log_output_a2a")

# ------------------------------------------------------------------
# Load all JSON files and group by prefix (everything before _world)
# ------------------------------------------------------------------
groups = {}  # prefix -> list of (world_size, data_dict)

for p in sorted(LOG_DIR.glob("*.json")):
    m = re.match(r"(.+)_world(\d+)\.json$", p.name)
    if not m:
        continue
    prefix, world = m.group(1), int(m.group(2))
    with open(p) as f:
        d = json.load(f)
    groups.setdefault(prefix, []).append((world, d))

# Sort each group by world size
for prefix in groups:
    groups[prefix].sort(key=lambda x: x[0])

SUBKEYS = ["both", "send", "recv"]
PHASES = ["dispatch", "combine"]
COLOR_CYCLE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
METRICS = ["mean", "p50", "p95", "p99"]


def get_series(data: dict, method: str, phase: str, subkey: str) -> dict | None:
    if method in data:
        return data.get(method, {}).get(phase, {}).get(subkey)
    return data.get(phase, {}).get(subkey)


def available_methods(entries: list[tuple[int, dict]]) -> list[str]:
    methods = []
    for _, data in entries:
        if "a2a" in data:
            methods.append("a2a")
        if "baseline" in data:
            methods.append("baseline")
    if methods:
        ordered = []
        for name in ["a2a", "baseline"]:
            if name in methods:
                ordered.append(name)
        return ordered
    return ["a2a"]

def plot_entries(prefix: str, entries: list[tuple[int, dict]], output_name: str, title: str) -> None:
    methods = available_methods(entries)
    fig, axes = plt.subplots(
        len(methods),
        len(PHASES),
        figsize=(16, 5 * len(methods)),
    )
    if len(methods) == 1:
        axes = np.array([axes])

    colors = COLOR_CYCLE[: len(entries)]
    x = np.arange(len(SUBKEYS) * len(METRICS))
    width = 0.8 / max(len(entries), 1)
    xtick_labels = [f"{subkey}\n{metric}" for subkey in SUBKEYS for metric in METRICS]

    for row, method in enumerate(methods):
        for col, phase in enumerate(PHASES):
            ax = axes[row, col]
            for i, (world, data) in enumerate(entries):
                stats = []
                for subkey in SUBKEYS:
                    stat = get_series(data, method, phase, subkey)
                    for metric in METRICS:
                        stats.append((subkey, metric, stat))
                valid = [
                    (idx, metric, stat)
                    for idx, (_, metric, stat) in enumerate(stats)
                    if stat is not None
                ]
                if not valid:
                    continue

                pos = np.array([x[idx] for idx, _, _ in valid], dtype=float)
                vals = [stat[metric] for _, metric, stat in valid]

                offset = (i - (len(entries) - 1) / 2) * width
                ax.bar(
                    pos + offset,
                    vals,
                    width,
                    label=f"world{world}",
                    color=colors[i],
                    alpha=0.8,
                    zorder=3,
                )

            title_prefix = (
                "A2A" if method == "a2a" else "Megatron-style AllGather/ReduceScatter"
            )
            ax.set_title(f"{title_prefix} {phase.capitalize()}", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(xtick_labels, fontsize=11)
            ax.set_ylabel("Latency (µs)", fontsize=11)
            ax.set_xlabel("Component / Metric", fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
            ax.legend(fontsize=10)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = LOG_DIR / output_name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


for prefix, entries in groups.items():
    plot_entries(
        prefix,
        entries,
        f"{prefix}_latency.png",
        prefix,
    )
    for world, data in entries:
        plot_entries(
            prefix,
            [(world, data)],
            f"{prefix}_world{world}_latency.png",
            f"{prefix} world{world}",
        )
