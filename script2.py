"""
Pruna vs HQQ — Variance Story
==============================

Purpose
-------
Five targeted plots that demonstrate Pruna's low variance advantage
over HQQ + torch.compile across throughput and latency dimensions.

Plots
-----
1. throughput_envelope.png       — mean ± 1 SD band per generation length
2. prefill_envelope.png          — same, prefill latency (log scale)
3. cv_comparison.png             — coefficient of variation, both metrics
4. latency_percentile_fan.png    — p50 / p90 / p95 / p99 decode latency
5. raw_scatter_strip.png         — every run, jittered, showing spread

Input
-----
/workspaces/pruna-cookbook/benchmark/eda_outputs/combined_cleaned_results.csv

Output
------
variance_story_outputs/plots/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = (
    "/workspaces/pruna-cookbook/benchmark/eda_outputs/"
    "combined_cleaned_results.csv"
)

OUTPUT_DIR = Path(
    "/workspaces/pruna-cookbook/benchmark/variance_story_outputs"
)
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DESIGN SYSTEM
# ============================================================

COLORS  = {"HQQ": "#E05C2A", "Pruna": "#2A7AE0"}
MARKERS = {"HQQ": "o",       "Pruna": "s"}

PERCENTILE_ALPHAS = {
    "p50": 1.0,
    "p90": 0.55,
    "p95": 0.35,
    "p99": 0.18,
}

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "grid.alpha":        0.35,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#cccccc",
    "legend.fontsize":   10,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#fafafa",
    "lines.linewidth":   2.2,
    "lines.markersize":  7,
})


# ============================================================
# HELPERS
# ============================================================

def style_ax(ax, xgrid=False, ygrid=True):
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#444444", length=4)
    if ygrid:
        ax.yaxis.grid(True, which="major", color="#e0e0e0",
                      linestyle="--", linewidth=0.7)
    if xgrid:
        ax.xaxis.grid(True, which="major", color="#e0e0e0",
                      linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)


def safe_fill(ax, x, mean, std, color, log_scale=False):
    lo = mean - std
    hi = mean + std
    if log_scale:
        lo = np.maximum(lo, mean * 0.01)
    ax.fill_between(x, lo, hi, alpha=0.18, color=color, linewidth=0)


def fw_legend(ax, frameworks, extra_handles=None, loc="best"):
    handles = [
        Line2D([0], [0], color=COLORS[fw], marker=MARKERS[fw],
               linewidth=2.2, markersize=7, label=fw)
        for fw in frameworks
    ]
    if extra_handles:
        handles += extra_handles
    ax.legend(handles=handles, loc=loc, framealpha=0.92)


def save(name):
    plt.savefig(PLOTS_DIR / name, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {name}")


# ============================================================
# LOAD & CLEAN
# ============================================================

print("Loading data...")
df = pd.read_csv(CSV_PATH)
print(f"  Raw rows: {len(df)}")

FAILED_THRESHOLD = 1000
df = df[df["avg_decode_latency_per_token_ms"] <= FAILED_THRESHOLD].copy()
print(f"  After removing failed runs: {len(df)}")

FRAMEWORKS = sorted(df["framework"].unique())   # ["HQQ", "Pruna"]


# ============================================================
# STABILITY TABLE
# ============================================================

stability_df = (
    df.groupby(["framework", "generation_length"])
    .agg({
        "decode_tokens_per_sec": ["mean", "std"],
        "prefill_latency_s":     ["mean", "std"],
    })
)
stability_df.columns = ["_".join(c) for c in stability_df.columns]
stability_df = stability_df.reset_index()

stability_df["throughput_cv"] = (
    stability_df["decode_tokens_per_sec_std"]
    / stability_df["decode_tokens_per_sec_mean"]
)
stability_df["prefill_cv"] = (
    stability_df["prefill_latency_s_std"]
    / stability_df["prefill_latency_s_mean"]
)


# ============================================================
# PLOT 1 — THROUGHPUT STABILITY ENVELOPE
# Story: Pruna's band is a thin ribbon; HQQ's is a wide canyon
# ============================================================

print("\nPlot 1: Throughput envelope")

fig, ax = plt.subplots(figsize=(9, 5))

for fw in FRAMEWORKS:
    sub = stability_df[stability_df["framework"] == fw].sort_values("generation_length")
    x    = sub["generation_length"].values
    mean = sub["decode_tokens_per_sec_mean"].values
    std  = sub["decode_tokens_per_sec_std"].values

    ax.plot(x, mean, marker=MARKERS[fw], color=COLORS[fw], zorder=4)
    safe_fill(ax, x, mean, std, color=COLORS[fw])

style_ax(ax, ygrid=True)
ax.set_xlabel("Generation Length (tokens)")
ax.set_ylabel("Decode Throughput (tokens / sec)")
ax.set_title("Decode Throughput — Mean ± 1 SD\nPruna is stable; HQQ variance grows with sequence length")

band_handles = [
    Patch(facecolor=COLORS[fw], alpha=0.25, label=f"{fw} ± 1 SD")
    for fw in FRAMEWORKS
]
fw_legend(ax, FRAMEWORKS, extra_handles=band_handles, loc="lower left")

# Annotate the spread contrast
ax.annotate("wide HQQ band\n= high run-to-run variance",
            xy=(0.97, 0.60), xycoords="axes fraction",
            ha="right", fontsize=8.5, color=COLORS["HQQ"],
            style="italic")
ax.annotate("narrow Pruna band\n= consistent throughput",
            xy=(0.97, 0.12), xycoords="axes fraction",
            ha="right", fontsize=8.5, color=COLORS["Pruna"],
            style="italic")

plt.tight_layout()
save("throughput_envelope.png")


# ============================================================
# PLOT 2 — PREFILL LATENCY STABILITY ENVELOPE  (log-safe)
# Story: HQQ prefill blows up unpredictably; Pruna is flat
# ============================================================

print("Plot 2: Prefill envelope")

fig, ax = plt.subplots(figsize=(9, 5))

for fw in FRAMEWORKS:
    sub = stability_df[stability_df["framework"] == fw].sort_values("generation_length")
    x    = sub["generation_length"].values
    mean = sub["prefill_latency_s_mean"].values
    std  = sub["prefill_latency_s_std"].values

    ax.plot(x, mean, marker=MARKERS[fw], color=COLORS[fw], zorder=4)
    safe_fill(ax, x, mean, std, color=COLORS[fw], log_scale=True)

ax.set_yscale("log")
style_ax(ax, ygrid=True)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))

ax.set_xlabel("Generation Length (tokens)")
ax.set_ylabel("Prefill Latency (s, log scale)")
ax.set_title("Prefill Latency — Mean ± 1 SD (log scale)\nHQQ has extreme outlier runs; Pruna is tight across all lengths")

band_handles = [
    Patch(facecolor=COLORS[fw], alpha=0.25, label=f"{fw} ± 1 SD")
    for fw in FRAMEWORKS
]
fw_legend(ax, FRAMEWORKS, extra_handles=band_handles, loc="upper left")

ax.annotate("Band clipped at mean×0.01\nto keep log axis valid",
            xy=(0.97, 0.04), xycoords="axes fraction",
            ha="right", fontsize=7.5, color="#999999", style="italic")

plt.tight_layout()
save("prefill_envelope.png")


# ============================================================
# PLOT 3 — CV COMPARISON  (dual-panel)
# Story: direct numeric comparison of instability
# ============================================================

print("Plot 3: CV comparison")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle(
    "Coefficient of Variation (CV = std / mean) — Lower is More Stable\n"
    "Pruna hovers near zero; HQQ spikes unpredictably",
    fontsize=12, fontweight="bold", y=1.01,
)

panel_cfg = [
    ("throughput_cv",  "Decode Throughput CV", axes[0]),
    ("prefill_cv",     "Prefill Latency CV",   axes[1]),
]

for metric, ylabel, ax in panel_cfg:
    for fw in FRAMEWORKS:
        sub = stability_df[stability_df["framework"] == fw].sort_values("generation_length")
        ax.plot(
            sub["generation_length"],
            sub[metric],
            marker=MARKERS[fw],
            color=COLORS[fw],
            label=fw,
        )
        # Shade area under the curve to emphasise magnitude
        ax.fill_between(
            sub["generation_length"],
            0,
            sub[metric],
            alpha=0.08,
            color=COLORS[fw],
        )

    ax.axhline(0, color="#aaaaaa", linewidth=0.9, linestyle=":")
    style_ax(ax, ygrid=True)
    ax.set_xlabel("Generation Length (tokens)")
    ax.set_ylabel(ylabel)
    # Reference band for "good" stability (CV < 0.05)
    ax.axhspan(0, 0.05, color="#2A7AE0", alpha=0.06, zorder=0)
    ax.annotate("CV < 0.05\n(stable zone)",
                xy=(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else
                    stability_df["generation_length"].min(),
                    0.025),
                xycoords="data",
                fontsize=7.5, color="#2A7AE0", alpha=0.7)
    fw_legend(ax, FRAMEWORKS, loc="upper left")

plt.tight_layout()
save("cv_comparison.png")


# ============================================================
# PLOT 4 — PERCENTILE FAN  (p50 / p90 / p95 / p99)
# Story: Pruna's tail is near its median; HQQ's tail explodes
# ============================================================

print("Plot 4: Percentile fan")

PERCENTILES = [50, 90, 95, 99]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle(
    "Decode Latency per Token — Percentile Fan (p50 → p99)\n"
    "A wide fan = long-tail risk. Pruna's percentiles stack tightly.",
    fontsize=12, fontweight="bold", y=1.01,
)

for ax, fw in zip(axes, FRAMEWORKS):
    sub = df[df["framework"] == fw].sort_values("generation_length")
    gen_lengths = sorted(sub["generation_length"].unique())

    pct_data = {}
    for p in PERCENTILES:
        pct_data[p] = [
            np.percentile(
                sub[sub["generation_length"] == g]["avg_decode_latency_per_token_ms"],
                p,
            )
            for g in gen_lengths
        ]

    # Fill between successive percentile bands
    band_colors = ["#aaaaaa", "#888888", "#555555"]
    band_labels = ["p50–p90", "p90–p95", "p95–p99"]
    for i, (lo_p, hi_p, bc, bl) in enumerate(zip(
        PERCENTILES[:-1], PERCENTILES[1:], band_colors, band_labels
    )):
        ax.fill_between(gen_lengths, pct_data[lo_p], pct_data[hi_p],
                        alpha=0.18 + i * 0.08, color=COLORS[fw],
                        label=bl, zorder=2)

    # Draw each percentile line
    line_styles = ["-", "--", "-.", ":"]
    for p, ls in zip(PERCENTILES, line_styles):
        ax.plot(gen_lengths, pct_data[p],
                color=COLORS[fw], linestyle=ls, linewidth=1.8,
                label=f"p{p}", zorder=3)

    style_ax(ax, ygrid=True)
    ax.set_xlabel("Generation Length (tokens)")
    ax.set_ylabel("Avg Decode Latency per Token (ms)")
    ax.set_title(fw, color=COLORS[fw], fontweight="bold")

    # Build tidy legend
    from matplotlib.lines import Line2D as L2D
    line_handles = [
        L2D([0], [0], color=COLORS[fw], linestyle=ls, linewidth=1.8, label=f"p{p}")
        for p, ls in zip(PERCENTILES, line_styles)
    ]
    band_handles = [
        Patch(facecolor=COLORS[fw], alpha=0.25, label=bl)
        for bl in band_labels
    ]
    ax.legend(handles=line_handles + band_handles,
              fontsize=8.5, loc="upper left", ncol=2)

plt.tight_layout()
save("latency_percentile_fan.png")


# ============================================================
# PLOT 5 — RAW SCATTER STRIP
# Story: every single run plotted; visual proof of spread
# ============================================================

print("Plot 5: Raw scatter strip")

STRIP_METRICS = [
    ("decode_tokens_per_sec",           "Decode Throughput (tokens / sec)"),
    ("avg_decode_latency_per_token_ms", "Avg Decode Latency / Token (ms)"),
    ("prefill_latency_s",               "Prefill Latency (s)"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    "Raw Run-Level Scatter — Every Observation per Framework\n"
    "Pruna clusters tightly; HQQ spreads across a wide range",
    fontsize=12, fontweight="bold", y=1.01,
)

rng = np.random.default_rng(42)

for ax, (metric, ylabel) in zip(axes, STRIP_METRICS):
    for i, fw in enumerate(FRAMEWORKS):
        vals = df[df["framework"] == fw][metric].values
        # Jitter on the x-axis within the framework's column
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color=COLORS[fw],
            alpha=0.45,
            s=22,
            edgecolors="none",
            zorder=3,
        )
        # Overlay median line
        med = np.median(vals)
        ax.plot([i - 0.25, i + 0.25], [med, med],
                color=COLORS[fw], linewidth=2.5, zorder=4, solid_capstyle="round")
        # Overlay IQR bar
        q25, q75 = np.percentile(vals, [25, 75])
        ax.plot([i, i], [q25, q75],
                color=COLORS[fw], linewidth=5, alpha=0.35, zorder=4)

    style_ax(ax, ygrid=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(FRAMEWORKS)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, 1.5)

    # Highlight spread with brace-style annotation
    for i, fw in enumerate(FRAMEWORKS):
        vals = df[df["framework"] == fw][metric].values
        cv = vals.std() / vals.mean()
        ax.annotate(f"CV={cv:.2f}",
                    xy=(i, vals.max()),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8, color=COLORS[fw], fontweight="bold")

axes[0].set_title("Throughput")
axes[1].set_title("Decode Latency")
axes[2].set_title("Prefill Latency")

plt.tight_layout()
save("raw_scatter_strip.png")


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("Variance story plots complete.")
print(f"Outputs: {PLOTS_DIR}")
print("=" * 60)