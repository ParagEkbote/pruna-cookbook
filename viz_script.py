"""
Focused Inference Benchmark Analysis
===================================

Purpose
-------
Head-to-head systems benchmarking comparison between:
- Pruna
- HQQ

Features
--------
- Failed HQQ run removal
- Scaling-aware systems plots
- Stability envelopes (log-safe)
- KV cache growth normalization
- Prefill vs decode benchmark isolation
- Publication-quality visualizations

Input
-----
/workspaces/pruna-cookbook/benchmark/eda_outputs/combined_cleaned_results.csv

Outputs
-------
focused_benchmark_outputs/
├── filtered_results.csv
├── benchmark_summary.csv
├── percentile_summary.csv
├── stability_summary.csv
└── plots/
    ├── kv_cache_growth.png
    ├── prefill_latency_scaling.png
    ├── prefill_stability.png
    ├── throughput_stability.png
    ├── throughput_stability_envelope.png
    ├── prefill_stability_envelope.png
    ├── peak_memory_scaling.png
    ├── efficiency_frontier.png
    ├── throughput_vs_latency.png
    └── prompt_length_prefill_latency.png
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
    "/workspaces/pruna-cookbook/benchmark/focused_benchmark_outputs"
)

PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DESIGN SYSTEM
# ============================================================

# Framework colors — consistent across every plot
COLORS = {
    "HQQ":   "#E05C2A",   # warm amber-red
    "Pruna": "#2A7AE0",   # cool blue
}

MARKERS = {
    "HQQ":   "o",
    "Pruna": "s",
}

# Base rcParams — grid OFF by default; we enable per-axis below
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
    "axes.grid":            False,   # ← disabled globally; set per axis
    "grid.alpha":           0.35,
    "grid.linestyle":       "--",
    "grid.linewidth":       0.6,
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

def style_ax(ax, xgrid=True, ygrid=True, log_y=False, log_x=False):
    """Apply consistent axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#444444", length=4)

    # Only major gridlines — avoids clutter on log axes
    if ygrid:
        ax.yaxis.grid(True, which="major", color="#dddddd",
                      linestyle="--", linewidth=0.7)
    if xgrid:
        ax.xaxis.grid(True, which="major", color="#dddddd",
                      linestyle="--", linewidth=0.7)

    ax.set_axisbelow(True)


def framework_legend(ax, frameworks, loc="best"):
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
    ax.legend(handles=handles, loc=loc, framealpha=0.92)


def save_plot(name):
    plt.savefig(PLOTS_DIR / name, bbox_inches="tight")
    plt.close()


def safe_fill(ax, x, mean, std, color, log_scale=False):
    """
    Fill ±1 std band.  On log axes clips lower bound to a small
    positive value so log rendering doesn't break.
    """
    lo = mean - std
    hi = mean + std
    if log_scale:
        lo = np.maximum(lo, mean * 0.01)   # never negative on log axis
    ax.fill_between(x, lo, hi, alpha=0.18, color=color, linewidth=0)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("Loading cleaned benchmark dataset...")
print("=" * 80)

df = pd.read_csv(CSV_PATH)

print(f"\nOriginal Rows: {len(df)}")


# ============================================================
# REMOVE FAILED HQQ RUNS
# ============================================================

FAILED_THRESHOLD = 1000

failed_mask = df["avg_decode_latency_per_token_ms"] > FAILED_THRESHOLD
failed_rows = df[failed_mask]

print("\nRemoved Failed Runs:")
print(
    failed_rows[[
        "framework",
        "generation_length",
        "avg_decode_latency_per_token_ms",
        "decode_tokens_per_sec",
    ]]
)

df = df[~failed_mask].copy()
print(f"\nRemaining Rows: {len(df)}")


# ============================================================
# NORMALIZED METRICS
# ============================================================

df["kv_cache_growth_gb_per_token"] = (
    df["decode_memory_growth_gb"] / df["generation_length"]
)


# ============================================================
# SAVE FILTERED DATASET
# ============================================================

df.to_csv(OUTPUT_DIR / "filtered_results.csv", index=False)


# ============================================================
# BENCHMARK SPLITS
# ============================================================

prefill_df = df[df["generation_length"] == df["generation_length"].median()]
decode_df  = df[df["actual_prompt_length"] == df["actual_prompt_length"].median()]


# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary_metrics = [
    "prefill_latency_s",
    "decode_tokens_per_sec",
    "avg_decode_latency_per_token_ms",
    "peak_memory_gb",
    "decode_memory_growth_gb",
]

summary_df = df.groupby("framework")[summary_metrics].agg(
    ["mean", "median", "std", "min", "max"]
)
summary_df.to_csv(OUTPUT_DIR / "benchmark_summary.csv")

print("\nBenchmark Summary:")
print(summary_df)


# ============================================================
# PERCENTILE ANALYSIS
# ============================================================

percentile_rows = []
for framework in df["framework"].unique():
    subset = df[df["framework"] == framework]
    row = {"framework": framework}
    for metric in ["prefill_latency_s", "avg_decode_latency_per_token_ms"]:
        for p in [50, 90, 95, 99]:
            row[f"{metric}_p{p}"] = np.percentile(subset[metric], p)
    percentile_rows.append(row)

percentile_df = pd.DataFrame(percentile_rows)
percentile_df.to_csv(OUTPUT_DIR / "percentile_summary.csv", index=False)


# ============================================================
# STABILITY ANALYSIS
# ============================================================

stability_df = (
    df.groupby(["framework", "generation_length"])
    .agg({
        "decode_tokens_per_sec": ["mean", "std"],
        "prefill_latency_s":     ["mean", "std"],
    })
)
stability_df.columns = ["_".join(col) for col in stability_df.columns]
stability_df = stability_df.reset_index()

stability_df["throughput_cv"] = (
    stability_df["decode_tokens_per_sec_std"]
    / stability_df["decode_tokens_per_sec_mean"]
)
stability_df["prefill_cv"] = (
    stability_df["prefill_latency_s_std"]
    / stability_df["prefill_latency_s_mean"]
)

stability_df.to_csv(OUTPUT_DIR / "stability_summary.csv", index=False)

FRAMEWORKS = sorted(df["framework"].unique())


# ============================================================
# PLOT 1 — KV CACHE GROWTH
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

for fw in FRAMEWORKS:
    subset = (
        df[df["framework"] == fw]
        .groupby("generation_length")["kv_cache_growth_gb_per_token"]
        .mean()
        .reset_index()
        .sort_values("generation_length")
    )
    ax.plot(
        subset["generation_length"],
        subset["kv_cache_growth_gb_per_token"],
        marker=MARKERS[fw],
        color=COLORS[fw],
        label=fw,
    )

style_ax(ax)
ax.set_xlabel("Generation Length (tokens)")
ax.set_ylabel("KV Cache Growth (GB / token)")
ax.set_title("KV Cache Growth Across Generation Lengths")
framework_legend(ax, FRAMEWORKS)
plt.tight_layout()
save_plot("kv_cache_growth.png")


# ============================================================
# PLOT 2 — PREFILL LATENCY SCALING
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

for fw in FRAMEWORKS:
    subset = (
        df[df["framework"] == fw]
        .groupby("generation_length")["prefill_latency_s"]
        .mean()
        .reset_index()
        .sort_values("generation_length")
    )
    ax.plot(
        subset["generation_length"],
        subset["prefill_latency_s"],
        marker=MARKERS[fw],
        color=COLORS[fw],
        label=fw,
    )

ax.set_yscale("log")
style_ax(ax, ygrid=True)           # major log gridlines only
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
ax.set_xlabel("Generation Length (tokens)")
ax.set_ylabel("Prefill Latency (s, log scale)")
ax.set_title("Prefill Latency Scaling Across Generation Lengths")
framework_legend(ax, FRAMEWORKS, loc="upper left")
plt.tight_layout()
save_plot("prefill_latency_scaling.png")


# ============================================================
# PLOT 3 — PREFILL STABILITY (CV)
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

for fw in FRAMEWORKS:
    subset = (
        stability_df[stability_df["framework"] == fw]
        .sort_values("generation_length")
    )
    ax.plot(
        subset["generation_length"],
        subset["prefill_cv"],
        marker=MARKERS[fw],
        color=COLORS[fw],
        label=fw,
    )

ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")
style_ax(ax)
ax.set_xlabel("Generation Length (tokens)")
ax.set_ylabel("Coefficient of Variation (std / mean)")
ax.set_title("Prefill Latency Stability — Lower CV = More Consistent")
framework_legend(ax, FRAMEWORKS)
plt.tight_layout()
save_plot("prefill_stability.png")


# ============================================================
# PLOT 4 — THROUGHPUT STABILITY (CV)
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

for fw in FRAMEWORKS:
    subset = (
        stability_df[stability_df["framework"] == fw]
        .sort_values("generation_length")
    )
    ax.plot(
        subset["generation_length"],
        subset["throughput_cv"],
        marker=MARKERS[fw],
        color=COLORS[fw],
        label=fw,
    )

ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")
style_ax(ax)
ax.set_xlabel("Generation Length (tokens)")
ax.set_ylabel("Coefficient of Variation (std / mean)")
ax.set_title("Decode Throughput Stability — Lower CV = More Consistent")
framework_legend(ax, FRAMEWORKS)
plt.tight_layout()
save_plot("throughput_stability.png")


# ============================================================
# PLOT 5 — THROUGHPUT STABILITY ENVELOPE
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

for fw in FRAMEWORKS:
    subset = (
        stability_df[stability_df["framework"] == fw]
        .sort_values("generation_length")
    )
    x    = subset["generation_length"].values
    mean = subset["decode_tokens_per_sec_mean"].values
    std  = subset["decode_tokens_per_sec_std"].values

    ax.plot(x, mean, marker=MARKERS[fw], color=COLORS[fw], label=fw)
    safe_fill(ax, x, mean, std, color=COLORS[fw], log_scale=False)

style_ax(ax)
ax.set_xlabel("Generation Length (tokens)")
ax.set_ylabel("Decode Tokens / sec")
ax.set_title("Decode Throughput — Mean ± 1 SD Envelope")
framework_legend(ax, FRAMEWORKS)
plt.tight_layout()
save_plot("throughput_stability_envelope.png")


# ============================================================
# PLOT 6 — PREFILL STABILITY ENVELOPE  (log-safe)
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

for fw in FRAMEWORKS:
    subset = (
        stability_df[stability_df["framework"] == fw]
        .sort_values("generation_length")
    )
    x    = subset["generation_length"].values
    mean = subset["prefill_latency_s_mean"].values
    std  = subset["prefill_latency_s_std"].values

    ax.plot(x, mean, marker=MARKERS[fw], color=COLORS[fw], label=fw)
    safe_fill(ax, x, mean, std, color=COLORS[fw], log_scale=True)

ax.set_yscale("log")
style_ax(ax, ygrid=True)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
ax.set_xlabel("Generation Length (tokens)")
ax.set_ylabel("Prefill Latency (s, log scale)")
ax.set_title("Prefill Latency — Mean ± 1 SD Envelope (log scale)")
# Note: on log scale the band appears asymmetric — that's correct/honest
ax.annotate(
    "Band clipped to avoid\nnegative log values",
    xy=(0.97, 0.04), xycoords="axes fraction",
    ha="right", va="bottom", fontsize=8, color="#888888",
)
framework_legend(ax, FRAMEWORKS, loc="upper left")
plt.tight_layout()
save_plot("prefill_stability_envelope.png")


# ============================================================
# PLOT 7 — PEAK MEMORY SCALING
# ============================================================

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
# PLOT 8 — EFFICIENCY FRONTIER
# Scatter with log x-axis to handle HQQ outliers cleanly
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

for fw in FRAMEWORKS:
    subset = df[df["framework"] == fw]
    ax.scatter(
        subset["peak_memory_gb"],
        subset["decode_tokens_per_sec"],
        s=55,
        alpha=0.65,
        color=COLORS[fw],
        marker=MARKERS[fw],
        label=fw,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

style_ax(ax)
ax.set_xlabel("Peak Memory (GB)")
ax.set_ylabel("Decode Tokens / sec")
ax.set_title("Efficiency Frontier — Throughput vs Memory Footprint\n"
             "(upper-left = more efficient)")
framework_legend(ax, FRAMEWORKS)

# Annotate direction arrow
ax.annotate(
    "← less memory\n↑ faster decoding",
    xy=(0.02, 0.97), xycoords="axes fraction",
    va="top", ha="left", fontsize=8.5, color="#666666",
    style="italic",
)
plt.tight_layout()
save_plot("efficiency_frontier.png")


# ============================================================
# PLOT 9 — THROUGHPUT VS LATENCY  (annotated trade-off curve)
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

for fw in FRAMEWORKS:
    subset = (
        df[df["framework"] == fw]
        .groupby("generation_length")
        .agg({
            "avg_decode_latency_per_token_ms": "mean",
            "decode_tokens_per_sec":           "mean",
        })
        .reset_index()
        .sort_values("generation_length")
    )

    x_vals = subset["avg_decode_latency_per_token_ms"].values
    y_vals = subset["decode_tokens_per_sec"].values
    labels = subset["generation_length"].values

    ax.plot(x_vals, y_vals, marker=MARKERS[fw],
            color=COLORS[fw], label=fw, zorder=3)

    # Offset annotations to avoid overlap
    for i, (x, y, lbl) in enumerate(zip(x_vals, y_vals, labels)):
        offset = (6, 5) if i % 2 == 0 else (6, -12)
        ax.annotate(
            str(int(lbl)),
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            color=COLORS[fw],
            alpha=0.85,
        )

ax.set_xscale("log")
style_ax(ax, xgrid=True, ygrid=True)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
ax.set_xlabel("Avg Decode Latency per Token (ms, log scale)")
ax.set_ylabel("Decode Tokens / sec")
ax.set_title("Throughput vs Latency Trade-off\n"
             "(labels = generation length in tokens)")
framework_legend(ax, FRAMEWORKS, loc="upper right")
plt.tight_layout()
save_plot("throughput_vs_latency.png")


# ============================================================
# PLOT 10 — PROMPT LENGTH × PREFILL LATENCY  (error bars)
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

prompt_summary = (
    prefill_df.groupby(["framework", "prompt_target_length"])
    .agg({"prefill_latency_s": ["mean", "std"]})
)
prompt_summary.columns = ["_".join(col) for col in prompt_summary.columns]
prompt_summary = prompt_summary.reset_index()

for fw in FRAMEWORKS:
    subset = (
        prompt_summary[prompt_summary["framework"] == fw]
        .sort_values("prompt_target_length")
    )
    ax.errorbar(
        subset["prompt_target_length"],
        subset["prefill_latency_s_mean"],
        yerr=subset["prefill_latency_s_std"],
        marker=MARKERS[fw],
        color=COLORS[fw],
        capsize=5,
        capthick=1.4,
        elinewidth=1.2,
        label=fw,
    )

ax.set_yscale("log")
style_ax(ax, ygrid=True)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
ax.set_xlabel("Prompt Length (tokens)")
ax.set_ylabel("Prefill Latency (s, log scale)")
ax.set_title("Prefill Latency Across Prompt Lengths\n"
             "(error bars = ±1 SD)")
framework_legend(ax, FRAMEWORKS, loc="upper left")
plt.tight_layout()
save_plot("prompt_length_prefill_latency.png")


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("FOCUSED BENCHMARK ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nOutputs saved to:\n{OUTPUT_DIR}")
print("\nGenerated Plots:")
for f in sorted(PLOTS_DIR.glob("*.png")):
    print(f"  - {f.name}")
print("\nKey plots:")
print("  - throughput_stability_envelope.png")
print("  - prefill_stability_envelope.png")
print("  - efficiency_frontier.png")
print("  - throughput_vs_latency.png")
print("\nDone.")