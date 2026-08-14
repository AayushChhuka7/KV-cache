"""Generate the six paper figures from the CSV block in figures.md (Section 1)."""
import csv
import io
import re
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG_MD = HERE / "figures.md"
DPI = 150

# Palette and technique order from figures.md Section 2
PALETTE = ["#1F3A5F", "#C8553D", "#588B8B", "#F28F3B", "#7C6F9C",
           "#3A6B35", "#A66C29", "#586F7D"]
TECH_ORDER = [
    "Baseline FP16",
    "Quantized INT8",
    "Quantized FP8 (E5M2)",
    "GQA",
    "Low-Rank Compression",
    "PagedAttention (vLLM)",
    "CPU Offloading",
    "NVFP4 (simulated)",
    "MiniKV (simulated)",
    "xKV (simulated)",
]
SIMULATED = {"NVFP4 (simulated)", "MiniKV (simulated)", "xKV (simulated)"}
# PagedAttention is real data but rendered with x marker / simulated styling.
PAGED = "PagedAttention (vLLM)"

# Column names from CSV header (figures.md line 25)
COLS = ["Technique", "Category", "Context", "Batch", "Peak GPU MB",
        "CPU RSS MB", "KV Cache MB", "Latency (s)", "Tokens/s", "TTFT (s)",
        "Memory Saving (%)", "Throughput Delta (%)", "Notes"]


def load_rows():
    """Parse the CSV code block in figures.md into a list of dicts."""
    text = FIG_MD.read_text(encoding="utf-8")
    # Find the ```csv ... ``` block
    m = re.search(r"```csv\n(.*?)```", text, re.S)
    if not m:
        raise RuntimeError("CSV block not found in figures.md")
    # Inject the canonical header (the figures.md CSV block omits the header row)
    csv_text = ",".join(COLS) + "\n" + m.group(1)
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = []
    for r in reader:
        # Cast numeric fields
        for k in ("Context", "Batch", "Peak GPU MB", "CPU RSS MB",
                  "KV Cache MB", "Latency (s)", "Tokens/s", "TTFT (s)",
                  "Memory Saving (%)", "Throughput Delta (%)"):
            r[k] = float(r[k])
        r["Context"] = int(r["Context"])
        r["Batch"] = int(r["Batch"])
        rows.append(r)
    return rows


def color_for(tech, idx):
    return PALETTE[idx % len(PALETTE)]


def tech_idx(tech):
    return TECH_ORDER.index(tech)


def is_simulated_styling(tech):
    return tech in SIMULATED or tech == PAGED


def plot_memory_vs_context(rows, out, jitter=False, jitter_pct=0.18):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    included = [r for r in rows if r["Batch"] == 1 and r["Technique"] not in SIMULATED]
    # Group by technique, sort by context
    by_tech = {}
    for r in included:
        by_tech.setdefault(r["Technique"], []).append(r)
    overlapping = [
        "Quantized INT8", "Quantized FP8 (E5M2)",
        "Low-Rank Compression", "PagedAttention (vLLM)",
        "CPU Offloading",
    ]
    for tech in TECH_ORDER:
        if tech not in by_tech:
            continue
        data = sorted(by_tech[tech], key=lambda r: r["Context"])
        ys = [d["Peak GPU MB"] for d in data]
        if jitter and tech in overlapping:
            slot = overlapping.index(tech)
            center = (len(overlapping) - 1) / 2.0
            # Multiplicative offset on the log2 x-axis (4% per slot)
            offset = (slot - center) * jitter_pct
            xs = [d["Context"] * (1.0 + offset) for d in data]
        else:
            xs = [d["Context"] for d in data]
        idx = tech_idx(tech)
        marker = "x" if is_simulated_styling(tech) else "o"
        alpha = 0.55 if is_simulated_styling(tech) else 1.0
        ax.plot(xs, ys, marker=marker, color=color_for(tech, idx),
                label=tech, linewidth=1.6, markersize=7, alpha=alpha)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 7300)
    ax.set_title("Peak GPU memory vs context length (batch=1)")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Peak GPU memory (MB)")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_throughput_vs_context(rows, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    included = [r for r in rows if r["Batch"] == 1 and r["Technique"] not in SIMULATED]
    by_tech = {}
    for r in included:
        by_tech.setdefault(r["Technique"], []).append(r)
    for tech in TECH_ORDER:
        if tech not in by_tech:
            continue
        data = sorted(by_tech[tech], key=lambda r: r["Context"])
        xs = [d["Context"] for d in data]
        ys = [d["Tokens/s"] for d in data]
        idx = tech_idx(tech)
        marker = "x" if is_simulated_styling(tech) else "o"
        alpha = 0.55 if is_simulated_styling(tech) else 1.0
        ax.plot(xs, ys, marker=marker, color=color_for(tech, idx),
                label=tech, linewidth=1.6, markersize=7, alpha=alpha)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 300)
    ax.set_title("Throughput vs context length (batch=1)")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Tokens / second")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              ncol=1, fontsize=8, frameon=True)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_latency_vs_context(rows, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    included = [r for r in rows if r["Batch"] == 1 and r["Technique"] not in SIMULATED]
    by_tech = {}
    for r in included:
        by_tech.setdefault(r["Technique"], []).append(r)
    for tech in TECH_ORDER:
        if tech not in by_tech:
            continue
        data = sorted(by_tech[tech], key=lambda r: r["Context"])
        xs = [d["Context"] for d in data]
        ys = [d["Latency (s)"] for d in data]
        idx = tech_idx(tech)
        marker = "x" if is_simulated_styling(tech) else "o"
        alpha = 0.55 if is_simulated_styling(tech) else 1.0
        ax.plot(xs, ys, marker=marker, color=color_for(tech, idx),
                label=tech, linewidth=1.6, markersize=7, alpha=alpha)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("End-to-end latency vs context length (batch=1)")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Wall time (s)")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_memory_vs_batch(rows, out, jitter=False, jitter_width=0.06):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    included = [r for r in rows if r["Context"] == 512 and r["Technique"] not in SIMULATED]
    by_tech = {}
    for r in included:
        by_tech.setdefault(r["Technique"], []).append(r)
    # Plot 5 overlapping tiny-gpt2 techniques last so the leftmost ones are not
    # hidden behind the others. Real order is preserved in the legend.
    draw_order = TECH_ORDER
    for tech in draw_order:
        if tech not in by_tech:
            continue
        data = sorted(by_tech[tech], key=lambda r: r["Batch"])
        ys = [d["Peak GPU MB"] for d in data]
        if jitter:
            # Spread the 5 tiny-gpt2 techniques that share identical values
            # across a small horizontal window per batch point. Map each tech
            # to a slot in [-w, +w] so the lines fan out.
            overlapping = [
                "Quantized INT8", "Quantized FP8 (E5M2)",
                "Low-Rank Compression", "PagedAttention (vLLM)",
                "CPU Offloading",
            ]
            if tech in overlapping:
                slot = overlapping.index(tech)
                center = (len(overlapping) - 1) / 2.0
                offset = (slot - center) * jitter_width
            else:
                offset = 0.0
            xs = [d["Batch"] + offset for d in data]
        else:
            xs = [d["Batch"] for d in data]
        idx = tech_idx(tech)
        marker = "x" if is_simulated_styling(tech) else "o"
        alpha = 0.55 if is_simulated_styling(tech) else 1.0
        ax.plot(xs, ys, marker=marker, color=color_for(tech, idx),
                label=tech, linewidth=1.6, markersize=7, alpha=alpha)
    ax.set_title("Peak GPU memory vs batch size (ctx=512)")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak GPU memory (MB)")
    if jitter:
        ax.set_xticks([1, 2, 4])
        ax.set_xlim(0.5, 4.5)
    else:
        ax.set_xticks([1, 2, 4])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              ncol=1, fontsize=8, frameon=True)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_summary_tokens_per_s(rows, out):
    # 7 real techniques only
    real = [t for t in TECH_ORDER if t not in SIMULATED]
    means = []
    for tech in real:
        vals = [r["Tokens/s"] for r in rows if r["Technique"] == tech]
        means.append((tech, sum(vals) / len(vals)))
    # Largest on top -> invert y; matplotlib draws first item at bottom, so reverse
    means = means[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (tech, m) in enumerate(means):
        idx = tech_idx(tech)
        hatch = "//" if tech == PAGED else None
        alpha = 0.55 if tech == PAGED else 1.0
        ax.barh(i, m, color=color_for(tech, idx), edgecolor="black",
                hatch=hatch, alpha=alpha)
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels([t for t, _ in means])
    ax.set_title("Average tokens per second across all configurations")
    ax.set_xlabel("Tokens / second")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def plot_summary_kv_mb(rows, out):
    # All 10 techniques; largest on top
    means = []
    for tech in TECH_ORDER:
        vals = [r["KV Cache MB"] for r in rows if r["Technique"] == tech]
        means.append((tech, sum(vals) / len(vals)))
    means = means[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (tech, m) in enumerate(means):
        idx = tech_idx(tech)
        is_sim = tech in SIMULATED
        hatch = "//" if is_sim else None
        alpha = 0.55 if is_sim else 1.0
        # If value is 0, log scale can't show it; nudge to a tiny floor
        plot_val = m if m > 0 else 1e-5
        ax.barh(i, plot_val, color=color_for(tech, idx), edgecolor="black",
                hatch=hatch, alpha=alpha)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e2)
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels([t for t, _ in means])
    ax.set_title("Average analytical KV cache size (MB) across all configurations")
    ax.set_xlabel("KV cache size (MB)")
    ax.grid(True, axis="x", which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jitter-batch", action="store_true",
                    help="Spread the 5 overlapping tiny-gpt2 techniques on the "
                         "memory_vs_batch chart so each line is visible.")
    ap.add_argument("--jitter-context", action="store_true",
                    help="Spread the 5 overlapping tiny-gpt2 techniques on the "
                         "memory_vs_context chart so each line is visible.")
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated list of figure basenames to (re)plot.")
    args = ap.parse_args()
    rows = load_rows()
    print(f"Loaded {len(rows)} rows from figures.md")
    all_files = {
        "memory_vs_context.png": (plot_memory_vs_context,
                                  {"jitter": args.jitter_context}),
        "throughput_vs_context.png": (plot_throughput_vs_context, {}),
        "latency_vs_context.png": (plot_latency_vs_context, {}),
        "memory_vs_batch.png": (plot_memory_vs_batch,
                                {"jitter": args.jitter_batch}),
        "summary_tokens_per_s.png": (plot_summary_tokens_per_s, {}),
        "summary_kv_mb.png": (plot_summary_kv_mb, {}),
    }
    targets = (
        [n.strip() for n in args.only.split(",") if n.strip()]
        if args.only
        else list(all_files.keys())
    )
    for name in targets:
        if name not in all_files:
            print(f"  skip {name} (unknown figure)")
            continue
        fn, kwargs = all_files[name]
        out = HERE / name
        fn(rows, out, **kwargs)
        tag = ""
        if kwargs.get("jitter"):
            tag = " (jittered)"
        print(f"  wrote {out}{tag}")


if __name__ == "__main__":
    main()
