# ============================================================
# PRUNA STRENGTH VISUALIZATION SUITE
# ============================================================
#
# Purpose
# -------
# Generate plots emphasizing where Pruna shines:
#
# - Memory efficiency
# - KV cache scaling
# - Resource stability
# - Systems efficiency
# - Scaling efficiency
#
# Input
# -----
# combined_cleaned_results.csv
#
# Outputs
# -------
# pruna_strength_plots/
# ├── kv_cache_growth_per_token.png
# ├── peak_memory_scaling.png
# ├── tokens_per_sec_per_gb.png
# ├── memory_per_generated_token.png
# ├── scaling_gradient_memory.png
# ├── normalized_efficiency_frontier.png
# ├── memory_stability_envelope.png
# └── summary.csv
#
# ============================================================

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
    "/workspaces/pruna-cookbook/benchmark/pruna_strength_plots"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MATPLOTLIB SETTINGS
# ============================================================

plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.grid"] = True


# ============================================================
# HELPER
# ============================================================

def save_plot(name):

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / name,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("Loading dataset...")
print("=" * 80)

df = pd.read_csv(CSV_PATH)

print(f"\nRows: {len(df)}")


# ============================================================
# REMOVE FAILED RUNS
# ============================================================

FAILED_THRESHOLD = 1000

df = df[
    df["avg_decode_latency_per_token_ms"]
    < FAILED_THRESHOLD
].copy()

print(f"Rows after filtering: {len(df)}")


# ============================================================
# NORMALIZED METRICS
# ============================================================

df["kv_cache_growth_gb_per_token"] = (
    df["decode_memory_growth_gb"]
    / df["generation_length"]
)

df["tokens_per_sec_per_gb"] = (
    df["decode_tokens_per_sec"]
    / df["peak_memory_gb"]
)

df["peak_memory_per_token_mb"] = (
    (df["peak_memory_gb"] * 1024)
    / df["generation_length"]
)

df["latency_per_token_normalized"] = (
    df["avg_decode_latency_per_token_ms"]
    / df["generation_length"]
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = (
    df.groupby("framework")
    .agg({
        "kv_cache_growth_gb_per_token": [
            "mean",
            "std",
        ],
        "tokens_per_sec_per_gb": [
            "mean",
            "std",
        ],
        "peak_memory_per_token_mb": [
            "mean",
            "std",
        ],
    })
)

summary.to_csv(
    OUTPUT_DIR / "summary.csv"
)

print("\nSummary:")
print(summary)


# ============================================================
# 1. KV CACHE GROWTH PER TOKEN
# ============================================================

plt.figure(figsize=(10, 7))

for framework in df["framework"].unique():

    subset = (
        df[df["framework"] == framework]
        .groupby("generation_length")[
            "kv_cache_growth_gb_per_token"
        ]
        .mean()
        .reset_index()
        .sort_values("generation_length")
    )

    plt.plot(
        subset["generation_length"],
        subset["kv_cache_growth_gb_per_token"],
        marker="o",
        linewidth=3,
        markersize=8,
        label=framework,
    )

plt.xlabel("Generation Length (tokens)")

plt.ylabel(
    "KV Cache Growth (GB/token)"
)

plt.title(
    "KV Cache Growth Efficiency Across Generation Lengths"
)

plt.legend()

save_plot("kv_cache_growth_per_token.png")


# ============================================================
# 2. PEAK MEMORY SCALING
# ============================================================

plt.figure(figsize=(10, 7))

for framework in df["framework"].unique():

    subset = (
        df[df["framework"] == framework]
        .groupby("generation_length")[
            "peak_memory_gb"
        ]
        .mean()
        .reset_index()
        .sort_values("generation_length")
    )

    plt.plot(
        subset["generation_length"],
        subset["peak_memory_gb"],
        marker="o",
        linewidth=3,
        markersize=8,
        label=framework,
    )

plt.xlabel("Generation Length (tokens)")

plt.ylabel("Peak Memory (GB)")

plt.title(
    "Peak Memory Scaling Across Generation Lengths"
)

plt.legend()

save_plot("peak_memory_scaling.png")


# ============================================================
# 3. TOKENS/SEC PER GB
# ============================================================

plt.figure(figsize=(10, 7))

for framework in df["framework"].unique():

    subset = (
        df[df["framework"] == framework]
        .groupby("generation_length")[
            "tokens_per_sec_per_gb"
        ]
        .mean()
        .reset_index()
        .sort_values("generation_length")
    )

    plt.plot(
        subset["generation_length"],
        subset["tokens_per_sec_per_gb"],
        marker="o",
        linewidth=3,
        markersize=8,
        label=framework,
    )

plt.xlabel("Generation Length (tokens)")

plt.ylabel("Decode Tokens/sec per GB")

plt.title(
    "Normalized Throughput Efficiency"
)

plt.legend()

save_plot("tokens_per_sec_per_gb.png")


# ============================================================
# 4. MEMORY PER GENERATED TOKEN
# ============================================================

plt.figure(figsize=(10, 7))

for framework in df["framework"].unique():

    subset = (
        df[df["framework"] == framework]
        .groupby("generation_length")[
            "memory_per_generated_token_mb"
        ]
        .mean()
        .reset_index()
        .sort_values("generation_length")
    )

    plt.plot(
        subset["generation_length"],
        subset["memory_per_generated_token_mb"],
        marker="o",
        linewidth=3,
        markersize=8,
        label=framework,
    )

plt.xlabel("Generation Length (tokens)")

plt.ylabel(
    "Memory per Generated Token (MB)"
)

plt.title(
    "Memory Efficiency Across Generation Lengths"
)

plt.legend()

save_plot("memory_per_generated_token.png")


# ============================================================
# 5. SCALING GRADIENT (dMEM/dTOKEN)
# ============================================================

plt.figure(figsize=(10, 7))

for framework in df["framework"].unique():

    subset = (
        df[df["framework"] == framework]
        .groupby("generation_length")[
            "peak_memory_gb"
        ]
        .mean()
        .reset_index()
        .sort_values("generation_length")
    )

    x = subset["generation_length"].values
    y = subset["peak_memory_gb"].values

    gradients = np.gradient(y, x)

    plt.plot(
        x,
        gradients,
        marker="o",
        linewidth=3,
        markersize=8,
        label=framework,
    )

plt.xlabel("Generation Length (tokens)")

plt.ylabel(
    "Memory Scaling Gradient d(GB)/d(token)"
)

plt.title(
    "Marginal Memory Growth Across Generation Lengths"
)

plt.legend()

save_plot("scaling_gradient_memory.png")


# ============================================================
# 6. NORMALIZED EFFICIENCY FRONTIER
# ============================================================

plt.figure(figsize=(10, 7))

for framework in df["framework"].unique():

    subset = df[
        df["framework"] == framework
    ]

    plt.scatter(
        subset["peak_memory_gb"],
        subset["tokens_per_sec_per_gb"],
        s=160,
        alpha=0.8,
        label=framework,
    )

plt.xlabel("Peak Memory (GB)")

plt.ylabel(
    "Decode Tokens/sec per GB"
)

plt.title(
    "Normalized Efficiency Frontier"
)

plt.legend()

save_plot("normalized_efficiency_frontier.png")


# ============================================================
# 7. MEMORY STABILITY ENVELOPE
# ============================================================

plt.figure(figsize=(10, 7))

for framework in df["framework"].unique():

    subset = (
        df[df["framework"] == framework]
        .groupby("generation_length")[
            "peak_memory_gb"
        ]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("generation_length")
    )

    x = subset["generation_length"]

    mean = subset["mean"]

    std = subset["std"]

    plt.plot(
        x,
        mean,
        marker="o",
        linewidth=3,
        label=framework,
    )

    plt.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.2,
    )

plt.xlabel("Generation Length (tokens)")

plt.ylabel("Peak Memory (GB)")

plt.title(
    "Memory Stability Envelope"
)

plt.legend()

save_plot("memory_stability_envelope.png")


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("PRUNA STRENGTH VISUALIZATION COMPLETE")
print("=" * 80)

print(f"\nOutputs saved to:\n{OUTPUT_DIR}")

print("\nGenerated Plots:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(f" - {file.name}")

print("\nMost Important Pruna-Centric Plots:")
print(" - kv_cache_growth_per_token.png")
print(" - tokens_per_sec_per_gb.png")
print(" - scaling_gradient_memory.png")
print(" - normalized_efficiency_frontier.png")

print("\nDone.")