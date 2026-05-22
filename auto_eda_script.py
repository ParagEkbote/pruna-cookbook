"""
Simplified AutoEDA for Pruna vs HQQ Benchmarks
==============================================

Focus:
- Clean benchmark comparison
- High-quality visualizations
- Minimal preprocessing
- Publication-ready outputs

Expected Columns
----------------
run
model_id
prompt_target_length
actual_prompt_length
generation_length
prefill_latency_s
prefill_peak_memory_gb
decode_time_s
decode_tokens_per_sec
avg_decode_latency_per_token_ms
peak_memory_gb
decode_memory_growth_gb
memory_per_generated_token_mb
compile_backend
compile_mode
quantization
dtype
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

HQQ_PATH = (
    "/workspaces/pruna-cookbook/benchmark/raw_benchmark_results_hqq.csv"
)

PRUNA_PATH = (
    "/workspaces/pruna-cookbook/benchmark/raw_benchmark_results_pruna.csv"
)

OUTPUT_DIR = Path(
    "/workspaces/pruna-cookbook/benchmark/eda_outputs"
)

PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MATPLOTLIB SETTINGS
# ============================================================

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 180
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("Loading benchmark datasets...")
print("=" * 80)

hqq_df = pd.read_csv(HQQ_PATH)
pruna_df = pd.read_csv(PRUNA_PATH)

hqq_df["framework"] = "HQQ"
pruna_df["framework"] = "Pruna"

df = pd.concat(
    [hqq_df, pruna_df],
    ignore_index=True,
)

print(f"HQQ Rows    : {len(hqq_df)}")
print(f"Pruna Rows  : {len(pruna_df)}")
print(f"Total Rows  : {len(df)}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# NUMERIC CLEANING
# ============================================================

numeric_cols = [
    "prompt_target_length",
    "actual_prompt_length",
    "generation_length",
    "prefill_latency_s",
    "prefill_peak_memory_gb",
    "decode_time_s",
    "decode_tokens_per_sec",
    "avg_decode_latency_per_token_ms",
    "peak_memory_gb",
    "decode_memory_growth_gb",
    "memory_per_generated_token_mb",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

df = df.drop_duplicates()

df.to_csv(
    OUTPUT_DIR / "combined_cleaned_results.csv",
    index=False,
)

print("\nDataset cleaned and saved.")


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_metrics = [
    "prefill_latency_s",
    "decode_tokens_per_sec",
    "avg_decode_latency_per_token_ms",
    "peak_memory_gb",
    "memory_per_generated_token_mb",
]

summary_df = (
    df.groupby("framework")[summary_metrics]
    .agg(["mean", "median", "std", "min", "max"])
)

summary_df.to_csv(
    OUTPUT_DIR / "aggregate_summary.csv"
)

print("\nAggregate Summary:")
print(summary_df)


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
# THROUGHPUT DISTRIBUTION
# ============================================================

plt.figure()

for framework in df["framework"].unique():

    subset = df[df["framework"] == framework]

    plt.hist(
        subset["decode_tokens_per_sec"].dropna(),
        bins=25,
        alpha=0.6,
        label=framework,
    )

plt.xlabel("Decode Tokens/sec")
plt.ylabel("Frequency")
plt.title("Throughput Distribution")
plt.legend()

save_plot("throughput_distribution.png")


# ============================================================
# PREFILL LATENCY DISTRIBUTION
# ============================================================

plt.figure()

for framework in df["framework"].unique():

    subset = df[df["framework"] == framework]

    plt.hist(
        subset["prefill_latency_s"].dropna(),
        bins=25,
        alpha=0.6,
        label=framework,
    )

plt.xlabel("Prefill Latency (s)")
plt.ylabel("Frequency")
plt.title("Prefill Latency Distribution")
plt.legend()

save_plot("prefill_latency_distribution.png")


# ============================================================
# MEMORY DISTRIBUTION
# ============================================================

plt.figure()

for framework in df["framework"].unique():

    subset = df[df["framework"] == framework]

    plt.hist(
        subset["peak_memory_gb"].dropna(),
        bins=25,
        alpha=0.6,
        label=framework,
    )

plt.xlabel("Peak Memory (GB)")
plt.ylabel("Frequency")
plt.title("Peak Memory Distribution")
plt.legend()

save_plot("peak_memory_distribution.png")


# ============================================================
# EFFICIENCY FRONTIER
# ============================================================

plt.figure(figsize=(10, 7))

for framework in df["framework"].unique():

    subset = df[df["framework"] == framework]

    plt.scatter(
        subset["peak_memory_gb"],
        subset["decode_tokens_per_sec"],
        s=90,
        alpha=0.75,
        label=framework,
    )

plt.xlabel("Peak Memory (GB)")
plt.ylabel("Decode Tokens/sec")
plt.title("Efficiency Frontier")
plt.legend()

save_plot("efficiency_frontier.png")


# ============================================================
# LATENCY VS THROUGHPUT
# ============================================================

plt.figure(figsize=(10, 7))

for framework in df["framework"].unique():

    subset = df[df["framework"] == framework]

    plt.scatter(
        subset["avg_decode_latency_per_token_ms"],
        subset["decode_tokens_per_sec"],
        s=90,
        alpha=0.75,
        label=framework,
    )

plt.xlabel("Avg Decode Latency Per Token (ms)")
plt.ylabel("Decode Tokens/sec")
plt.title("Latency vs Throughput")
plt.legend()

save_plot("latency_vs_throughput.png")


# ============================================================
# MEMORY EFFICIENCY
# ============================================================

plt.figure(figsize=(10, 7))

for framework in df["framework"].unique():

    subset = df[df["framework"] == framework]

    plt.scatter(
        subset["memory_per_generated_token_mb"],
        subset["decode_tokens_per_sec"],
        s=90,
        alpha=0.75,
        label=framework,
    )

plt.xlabel("Memory Per Generated Token (MB)")
plt.ylabel("Decode Tokens/sec")
plt.title("Memory Efficiency")
plt.legend()

save_plot("memory_efficiency.png")


# ============================================================
# BOXPLOTS
# ============================================================

metrics = [
    "decode_tokens_per_sec",
    "avg_decode_latency_per_token_ms",
    "peak_memory_gb",
]

fig, axes = plt.subplots(
    1,
    len(metrics),
    figsize=(18, 6),
)

for ax, metric in zip(axes, metrics):

    data = []

    labels = []

    for framework in df["framework"].unique():

        subset = df[
            df["framework"] == framework
        ][metric].dropna()

        data.append(subset)
        labels.append(framework)

    ax.boxplot(
        data,
        tick_labels=labels,
    )

    ax.set_title(metric)

save_plot("framework_boxplots.png")


# ============================================================
# COMPILE MODE ANALYSIS
# ============================================================

if "compile_mode" in df.columns:

    grouped = (
        df.groupby(["framework", "compile_mode"])[
            "decode_tokens_per_sec"
        ]
        .mean()
        .unstack(0)
    )

    grouped.plot(
        kind="bar",
        figsize=(10, 6),
    )

    plt.ylabel("Mean Decode Tokens/sec")
    plt.title("Compile Mode Performance")

    save_plot("compile_mode_performance.png")


# ============================================================
# QUANTIZATION ANALYSIS
# ============================================================

if "quantization" in df.columns:

    grouped = (
        df.groupby(["framework", "quantization"])[
            "decode_tokens_per_sec"
        ]
        .mean()
        .unstack(0)
    )

    grouped.plot(
        kind="bar",
        figsize=(10, 6),
    )

    plt.ylabel("Mean Decode Tokens/sec")
    plt.title("Quantization Performance")

    save_plot("quantization_performance.png")


# ============================================================
# CORRELATION HEATMAP
# ============================================================

corr_cols = [
    "prefill_latency_s",
    "decode_tokens_per_sec",
    "avg_decode_latency_per_token_ms",
    "peak_memory_gb",
    "decode_memory_growth_gb",
    "memory_per_generated_token_mb",
    "generation_length",
]

corr_df = df[corr_cols].corr()

corr_df.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)

plt.figure(figsize=(10, 8))

plt.imshow(
    corr_df,
    aspect="auto",
)

plt.colorbar()

plt.xticks(
    range(len(corr_df.columns)),
    corr_df.columns,
    rotation=45,
)

plt.yticks(
    range(len(corr_df.columns)),
    corr_df.columns,
)

plt.title("Correlation Heatmap")

save_plot("correlation_heatmap.png")


# ============================================================
# STABILITY ANALYSIS
# ============================================================

stability = (
    df.groupby("framework")[
        "decode_tokens_per_sec"
    ]
    .agg(["mean", "std"])
)

stability["cv"] = (
    stability["std"] / stability["mean"]
)

stability.to_csv(
    OUTPUT_DIR / "stability_analysis.csv"
)

plt.figure(figsize=(8, 6))

plt.bar(
    stability.index,
    stability["cv"],
)

plt.ylabel("Coefficient of Variation")
plt.title("Throughput Stability")

save_plot("throughput_stability.png")


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)

print(f"\nOutputs saved to:\n{OUTPUT_DIR}")

print("\nGenerated Plots:")

for file in sorted(PLOTS_DIR.glob("*.png")):
    print(f" - {file.name}")

print("\nMost Important Visualizations:")
print(" - efficiency_frontier.png")
print(" - latency_vs_throughput.png")
print(" - memory_efficiency.png")
print(" - framework_boxplots.png")
print(" - throughput_stability.png")

print("\nDone.")