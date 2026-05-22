"""
Focused Benchmark Comparison
============================

Purpose
-------
Head-to-head comparison between:
- Pruna
- HQQ

Using cleaned benchmark results with:
- failed HQQ run removed
- scaled visualizations
- log-safe plots
- publication-quality comparisons

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
    ├── prefill_latency_comparison.png
    ├── decode_throughput_comparison.png
    ├── throughput_stability.png
    ├── prefill_stability.png
    ├── kv_cache_growth.png
    ├── peak_memory_scaling.png
    ├── efficiency_frontier.png
    ├── latency_percentiles.png
    └── throughput_vs_latency.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
# MATPLOTLIB SETTINGS
# ============================================================

plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("Loading cleaned benchmark dataset...")
print("=" * 80)

df = pd.read_csv(CSV_PATH)

print(f"\nOriginal Rows: {len(df)}")


# ============================================================
# REMOVE FAILED HQQ RUN
# ============================================================

# catastrophic failed run
FAILED_THRESHOLD = 1000

failed_mask = (
    df["avg_decode_latency_per_token_ms"]
    > FAILED_THRESHOLD
)

failed_rows = df[failed_mask]

print("\nRemoved Failed Runs:")
print(failed_rows[
    [
        "framework",
        "generation_length",
        "avg_decode_latency_per_token_ms",
        "decode_tokens_per_sec",
    ]
])

df = df[~failed_mask].copy()

print(f"\nRemaining Rows: {len(df)}")


# ============================================================
# SAVE FILTERED DATASET
# ============================================================

df.to_csv(
    OUTPUT_DIR / "filtered_results.csv",
    index=False,
)


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

summary_df = (
    df.groupby("framework")[summary_metrics]
    .agg(["mean", "median", "std", "min", "max"])
)

summary_df.to_csv(
    OUTPUT_DIR / "benchmark_summary.csv"
)

print("\nBenchmark Summary:")
print(summary_df)


# ============================================================
# PERCENTILE ANALYSIS
# ============================================================

percentile_rows = []

for framework in df["framework"].unique():

    subset = df[df["framework"] == framework]

    row = {
        "framework": framework,
    }

    for metric in [
        "prefill_latency_s",
        "avg_decode_latency_per_token_ms",
    ]:

        row[f"{metric}_p50"] = np.percentile(
            subset[metric],
            50,
        )

        row[f"{metric}_p90"] = np.percentile(
            subset[metric],
            90,
        )

        row[f"{metric}_p95"] = np.percentile(
            subset[metric],
            95,
        )

        row[f"{metric}_p99"] = np.percentile(
            subset[metric],
            99,
        )

    percentile_rows.append(row)

percentile_df = pd.DataFrame(percentile_rows)

percentile_df.to_csv(
    OUTPUT_DIR / "percentile_summary.csv",
    index=False,
)


# ============================================================
# STABILITY ANALYSIS
# ============================================================

stability_rows = []

for framework in df["framework"].unique():

    subset = df[df["framework"] == framework]

    throughput_mean = subset[
        "decode_tokens_per_sec"
    ].mean()

    throughput_std = subset[
        "decode_tokens_per_sec"
    ].std()

    prefill_mean = subset[
        "prefill_latency_s"
    ].mean()

    prefill_std = subset[
        "prefill_latency_s"
    ].std()

    stability_rows.append({
        "framework": framework,

        "throughput_cv":
            throughput_std / throughput_mean,

        "prefill_cv":
            prefill_std / prefill_mean,
    })

stability_df = pd.DataFrame(stability_rows)

stability_df.to_csv(
    OUTPUT_DIR / "stability_summary.csv",
    index=False,
)


# ============================================================
# HELPER
# ============================================================

def save_plot(name):

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / name,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# PREFILL LATENCY COMPARISON
# ============================================================

plt.figure(figsize=(8, 6))

data = []

labels = []

for framework in df["framework"].unique():

    subset = df[
        df["framework"] == framework
    ]["prefill_latency_s"]

    data.append(subset)
    labels.append(framework)

plt.boxplot(
    data,
    tick_labels=labels,
)

plt.yscale("log")

plt.ylabel("Prefill Latency (s)")
plt.title("Prefill Latency Comparison")

save_plot("prefill_latency_comparison.png")


# ============================================================
# DECODE THROUGHPUT COMPARISON
# ============================================================

throughput_mean = (
    df.groupby("framework")[
        "decode_tokens_per_sec"
    ]
    .mean()
)

throughput_std = (
    df.groupby("framework")[
        "decode_tokens_per_sec"
    ]
    .std()
)

plt.figure(figsize=(8, 6))

plt.bar(
    throughput_mean.index,
    throughput_mean.values,
    yerr=throughput_std.values,
    capsize=8,
)

plt.ylabel("Decode Tokens/sec")
plt.title("Decode Throughput Comparison")

save_plot("decode_throughput_comparison.png")


# ============================================================
# THROUGHPUT STABILITY
# ============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    stability_df["framework"],
    stability_df["throughput_cv"],
)

plt.ylabel("Coefficient of Variation")
plt.title("Throughput Stability")

save_plot("throughput_stability.png")


# ============================================================
# PREFILL STABILITY
# ============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    stability_df["framework"],
    stability_df["prefill_cv"],
)

plt.ylabel("Coefficient of Variation")
plt.title("Prefill Stability")

save_plot("prefill_stability.png")


# ============================================================
# KV CACHE GROWTH
# ============================================================

plt.figure(figsize=(10, 7))

for framework in df["framework"].unique():

    subset = (
        df[df["framework"] == framework]
        .groupby("generation_length")[
            "decode_memory_growth_gb"
        ]
        .mean()
        .reset_index()
        .sort_values("generation_length")
    )

    plt.plot(
        subset["generation_length"],
        subset["decode_memory_growth_gb"],
        marker="o",
        linewidth=3,
        markersize=8,
        label=framework,
    )

plt.xlabel("Generation Length")
plt.ylabel("Decode Memory Growth (GB)")
plt.title("KV Cache Growth")

plt.legend()

save_plot("kv_cache_growth.png")

# ============================================================
# PEAK MEMORY SCALING
# ============================================================

plt.figure(figsize=(10, 7))

for framework in df["framework"].unique():

    subset = df[
        df["framework"] == framework
    ]

    plt.plot(
        subset["generation_length"],
        subset["peak_memory_gb"],
        marker="o",
        linewidth=2,
        label=framework,
    )

plt.xlabel("Generation Length")
plt.ylabel("Peak Memory (GB)")
plt.title("Peak Memory Scaling")
plt.legend()

save_plot("peak_memory_scaling.png")


# ============================================================
# EFFICIENCY FRONTIER
# ============================================================

plt.figure(figsize=(10, 7))

for framework in df["framework"].unique():

    subset = df[
        df["framework"] == framework
    ]

    plt.scatter(
        subset["peak_memory_gb"],
        subset["decode_tokens_per_sec"],
        s=140,
        alpha=0.8,
        label=framework,
    )

plt.xlabel("Peak Memory (GB)")
plt.ylabel("Decode Tokens/sec")
plt.title("Efficiency Frontier")
plt.legend()

save_plot("efficiency_frontier.png")


# ============================================================
# LATENCY PERCENTILES
# ============================================================

metrics = [
    "prefill_latency_s_p50",
    "prefill_latency_s_p95",
]

x = np.arange(len(percentile_df))

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    percentile_df[
        "prefill_latency_s_p50"
    ],
    width,
    label="P50",
)

plt.bar(
    x + width / 2,
    percentile_df[
        "prefill_latency_s_p95"
    ],
    width,
    label="P95",
)

plt.xticks(
    x,
    percentile_df["framework"],
)

plt.yscale("log")

plt.ylabel("Prefill Latency (s)")
plt.title("Latency Percentiles")
plt.legend()

save_plot("latency_percentiles.png")


# ============================================================
# THROUGHPUT VS LATENCY
# ============================================================

plt.figure(figsize=(10, 7))

for framework in df["framework"].unique():

    subset = (
        df[df["framework"] == framework]
        .groupby("generation_length")
        .agg({
            "avg_decode_latency_per_token_ms": "mean",
            "decode_tokens_per_sec": "mean",
        })
        .reset_index()
        .sort_values("generation_length")
    )

    plt.plot(
        subset["avg_decode_latency_per_token_ms"],
        subset["decode_tokens_per_sec"],
        marker="o",
        linewidth=3,
        markersize=9,
        label=framework,
    )

    # annotate generation length
    for _, row in subset.iterrows():

        plt.annotate(
            int(row["generation_length"]),
            (
                row["avg_decode_latency_per_token_ms"],
                row["decode_tokens_per_sec"],
            ),
            fontsize=9,
            alpha=0.8,
        )

plt.xscale("log")

plt.xlabel("Avg Decode Latency Per Token (ms)")
plt.ylabel("Decode Tokens/sec")

plt.title("Throughput vs Latency Scaling")

plt.legend()

save_plot("throughput_vs_latency.png")


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("FOCUSED BENCHMARK ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nOutputs saved to:\n{OUTPUT_DIR}")

print("\nGenerated Plots:")

for file in sorted(PLOTS_DIR.glob("*.png")):
    print(f" - {file.name}")

print("\nMost Important Plots:")
print(" - prefill_latency_comparison.png")
print(" - decode_throughput_comparison.png")
print(" - efficiency_frontier.png")
print(" - throughput_vs_latency.png")
print(" - kv_cache_growth.png")

print("\nDone.")