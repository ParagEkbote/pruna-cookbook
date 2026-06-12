"""
Memory Benchmark Plots — Pruna vs HQQ
======================================

Generates three focused memory-centric plots:

1. kv_cache_growth.png         — KV cache growth (GB/token) across generation lengths
2. peak_memory_scaling.png     — Peak memory (GB) across generation lengths
3. memory_stability_envelope.png — Peak memory mean ± 1 SD envelope

Input
-----
/workspaces/pruna-cookbook/benchmark/eda_outputs/combined_cleaned_results.csv

Outputs
-------
memory_plot_outputs/
├── kv_cache_growth.png
├── peak_memory_scaling.png
└── memory_stability_envelope.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = (
    "/workspaces/pruna-cookbook/benchmark/eda_outputs/"
    "combined_cleaned_results.csv"
)

OUTPUT_DIR = Path(
    "/workspaces/pruna-cookbook/benchmark/memory_plot_outputs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAILED_THRESHOLD = 1000


# ============================================================
# DESIGN SYSTEM
# ============================================================

COLORS = {
    "HQQ":   "#E05C2A",  # warm amber-red
    "Pruna": "#2A7AE0",  # cool blue
}

MARKERS = {
    "HQQ":   "o",
    "Pruna": "s",
}

plt.rcParams.update({
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "font.family":          "DejaVu Sans",
    "font.size":            11,
    "axes.titlesize":       13,
    "axes.titleweight":     "bold",
    "axes.labelsize":       11,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            False,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "#cccccc",
    "legend.fontsize":      10,
    "figure.facecolor":     "white",
    "axes.facecolor":       "#fafafa",
    "lines.linewidth":      2.2,
    "lines.markersize":     7,
})


# ============================================================
# HELPERS
# ============================================================

def style_ax(ax, xgrid=True, ygrid=True):
    """Apply consistent axis styling."""
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#444444", length=4)
    if ygrid:
        ax.yaxis.grid(True, which="major", color="#dddddd",
                      linestyle="--", linewidth=0.7)
    if xgrid:
        ax.xaxis.grid(True, which="major", color="#dddddd",
                      linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)


def framework_legend(ax, frameworks, extra_handles=None, loc="best"):
    """Build a consistent per-framework legend."""
    handles = [
        Line2D([0], [0],
               color=COLORS[fw],
               marker=MARKERS[fw],
               linewidth=2.2,
               markersize=7,
               label=fw)
        for fw in frameworks
    ]
    if extra_handles:
        handles += extra_handles
    ax.legend(handles=handles, loc=loc, framealpha=0.92)


def save_plot(name):
    plt.savefig(OUTPUT_DIR / name, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {name}")


# ============================================================
# LOAD & CLEAN
# ============================================================

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"  Raw rows: {len(df)}")

df = df[df["avg_decode_latency_per_token_ms"] <= FAILED_THRESHOLD].copy()
print(f"  After removing failed runs: {len(df)}")

# Derived metric — MB for human-readable y-axis (avoids 1e-5 scientific notation)
df["kv_cache_growth_mb_per_token"] = (
    df["decode_memory_growth_gb"] / df["generation_length"] * 1024
)

FRAMEWORKS = sorted(df["framework"].unique())


# ============================================================
# PLOT 1 — KV CACHE GROWTH PER TOKEN
# ============================================================

print("\nPlot 1: KV cache growth per token")

fig, ax = plt.subplots(figsize=(9, 5))

for fw in FRAMEWORKS:
    subset = (
        df[df["framework"] == fw]
        .groupby("generation_length")["kv_cache_growth_mb_per_token"]
        .mean()
        .reset_index()
        .sort_values("generation_length")
    )
    ax.plot(
        subset["generation_length"],
        subset["kv_cache_growth_mb_per_token"],
        marker=MARKERS[fw],
        color=COLORS[fw],
        label=fw,
    )

style_ax(ax)
ax.set_xlabel("Generation Length (tokens)")
ax.set_ylabel("KV Cache Growth (MB / token)")
ax.set_title("KV Cache Growth Efficiency Across Generation Lengths")
framework_legend(ax, FRAMEWORKS)
plt.tight_layout()
save_plot("kv_cache_growth.png")


# ============================================================
# PLOT 2 — PEAK MEMORY SCALING
# ============================================================

print("Plot 2: Peak memory scaling")

fig, ax = plt.subplots(figsize=(9, 5))

for fw in FRAMEWORKS:
    subset = (
        df[df["framework"] == fw]
        .groupby("generation_length")["peak_memory_gb"]
        .mean()
        .reset_index()
        .sort_values("generation_length")
    )
    ax.plot(
        subset["generation_length"],
        subset["peak_memory_gb"],
        marker=MARKERS[fw],
        color=COLORS[fw],
        label=fw,
    )

style_ax(ax)
ax.set_xlabel("Generation Length (tokens)")
ax.set_ylabel("Peak Memory (GB)")
ax.set_title("Peak Memory Scaling Across Generation Lengths")
framework_legend(ax, FRAMEWORKS)
plt.tight_layout()
save_plot("peak_memory_scaling.png")


# ============================================================
# PLOT 3 — MEMORY STABILITY ENVELOPE  (mean ± 1 SD)
# ============================================================

print("Plot 3: Memory stability envelope")

fig, ax = plt.subplots(figsize=(9, 5))

for fw in FRAMEWORKS:
    subset = (
        df[df["framework"] == fw]
        .groupby("generation_length")["peak_memory_gb"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("generation_length")
    )
    x    = subset["generation_length"].values
    mean = subset["mean"].values
    std  = subset["std"].values

    ax.plot(x, mean, marker=MARKERS[fw], color=COLORS[fw], zorder=4)
    ax.fill_between(x, mean - std, mean + std,
                    alpha=0.12, color=COLORS[fw], linewidth=0)

style_ax(ax)
ax.set_xlabel("Generation Length (tokens)")
ax.set_ylabel("Peak Memory (GB)")
ax.set_title(
    "Memory Stability Envelope — Mean ± 1 SD\n"
    "Narrower band = more consistent memory usage"
)

framework_legend(ax, FRAMEWORKS)
plt.tight_layout()
save_plot("memory_stability_envelope.png")


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("Memory plots complete.")
print(f"Outputs saved to: {OUTPUT_DIR}")
print("=" * 60)