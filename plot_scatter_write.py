import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

LOG_DIR = Path("log_output_scatter_write")

# ------------------------------------------------------------------
# Load all JSON files, group by (prefix_without_world, world_size)
# and index by max_num_tokens for the x-axis
# ------------------------------------------------------------------
# filename pattern: scatter_write_t{tokens}_e{experts}_h{hidden}_k{topk}_{dtype}_world{W}.json
records = []  # list of dicts with parsed fields + data

for p in sorted(LOG_DIR.glob("*.json")):
    m = re.match(
        r"(.+)_t(\d+)_e(\d+)_h(\d+)_k(\d+)_(\w+)_world(\d+)\.json$", p.name
    )
    if not m:
        continue
    prefix = m.group(1)
    tokens = int(m.group(2))
    world = int(m.group(7))
    with open(p) as f:
        d = json.load(f)
    records.append(
        dict(prefix=prefix, tokens=tokens, world=world, data=d)
    )

if not records:
    print(f"No JSON files found in {LOG_DIR}")
    raise SystemExit(1)

# Group by prefix, then plot one figure per prefix
prefixes = sorted({r["prefix"] for r in records})

METRICS = ["p50", "p95", "p99"]
SUBKEYS = ["send", "rtt"]
COLOR_CYCLE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def make_figure(prefix: str) -> None:
    subset = [r for r in records if r["prefix"] == prefix]
    world_sizes = sorted({r["world"] for r in subset})
    token_sizes = sorted({r["tokens"] for r in subset})

    colors = {w: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, w in enumerate(world_sizes)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_lat, ax_bw_send, ax_bw_rtt = axes

    # ---- Latency (p50/p95/p99) grouped by token count, colored by world size ----
    x = np.arange(len(token_sizes))
    width = 0.8 / max(len(world_sizes), 1)

    for i, world in enumerate(world_sizes):
        p50_send, p95_send, p99_send = [], [], []
        for t in token_sizes:
            row = next((r for r in subset if r["world"] == world and r["tokens"] == t), None)
            if row:
                s = row["data"]["scatter_write"]["send"]
                p50_send.append(s["p50"] / 1e3)
                p95_send.append(s["p95"] / 1e3)
                p99_send.append(s["p99"] / 1e3)
            else:
                p50_send.append(0); p95_send.append(0); p99_send.append(0)

        offset = (i - (len(world_sizes) - 1) / 2) * width
        bars = ax_lat.bar(
            x + offset, p50_send, width,
            label=f"world{world}", color=colors[world], alpha=0.85, zorder=3,
        )
        # p95 and p99 as error lines on top
        for j, (p50, p95, p99) in enumerate(zip(p50_send, p95_send, p99_send)):
            bx = x[j] + offset + width / 2
            ax_lat.plot([bx, bx], [p50, p99], color="black", linewidth=1.2, zorder=4)
            ax_lat.plot([bx - width * 0.3, bx + width * 0.3], [p95, p95],
                        color="black", linewidth=1.2, zorder=4)

    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels([f"{t//1024}K" if t >= 1024 else str(t) for t in token_sizes])
    ax_lat.set_xlabel("max_num_tokens")
    ax_lat.set_ylabel("Send latency p50 (ms)")
    ax_lat.set_title("Send Latency (bar=p50, line=p95–p99)")
    ax_lat.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax_lat.legend()

    # ---- Bandwidth: send ----
    for i, world in enumerate(world_sizes):
        bws = []
        for t in token_sizes:
            row = next((r for r in subset if r["world"] == world and r["tokens"] == t), None)
            bws.append(row["data"]["scatter_write"]["send"]["bandwidth_gbs"] if row else 0)
        offset = (i - (len(world_sizes) - 1) / 2) * width
        ax_bw_send.bar(x + offset, bws, width, label=f"world{world}",
                       color=colors[world], alpha=0.85, zorder=3)

    ax_bw_send.set_xticks(x)
    ax_bw_send.set_xticklabels([f"{t//1024}K" if t >= 1024 else str(t) for t in token_sizes])
    ax_bw_send.set_xlabel("max_num_tokens")
    ax_bw_send.set_ylabel("Bandwidth (GB/s)")
    ax_bw_send.set_title("Send Bandwidth (GB/s)")
    ax_bw_send.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax_bw_send.legend()

    # ---- Bandwidth: rtt ----
    for i, world in enumerate(world_sizes):
        bws = []
        for t in token_sizes:
            row = next((r for r in subset if r["world"] == world and r["tokens"] == t), None)
            bws.append(row["data"]["scatter_write"]["rtt"]["bandwidth_gbs"] if row else 0)
        offset = (i - (len(world_sizes) - 1) / 2) * width
        ax_bw_rtt.bar(x + offset, bws, width, label=f"world{world}",
                      color=colors[world], alpha=0.85, zorder=3)

    ax_bw_rtt.set_xticks(x)
    ax_bw_rtt.set_xticklabels([f"{t//1024}K" if t >= 1024 else str(t) for t in token_sizes])
    ax_bw_rtt.set_xlabel("max_num_tokens")
    ax_bw_rtt.set_ylabel("Bandwidth (GB/s)")
    ax_bw_rtt.set_title("RTT Bandwidth (GB/s)")
    ax_bw_rtt.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax_bw_rtt.legend()

    fig.suptitle(prefix, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = LOG_DIR / f"{prefix}_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


for prefix in prefixes:
    make_figure(prefix)
