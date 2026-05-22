"""
Validation + Sanity Analysis Script
==================================

Purpose
-------
Validate whether the cleaned benchmark dataset is trustworthy
for visualization and analysis.

Input
-----
/workspaces/pruna-cookbook/benchmark/eda_outputs/combined_cleaned_results.csv

Outputs
-------
validation_outputs/
├── validation_summary.txt
├── numeric_summary.csv
├── missing_values.csv
├── outlier_summary.csv
├── plots/
│   ├── missing_values.png
│   ├── generation_vs_memory.png
│   ├── throughput_vs_latency.png
│   ├── outlier_boxplots.png
│   ├── framework_distribution.png
│   └── correlation_heatmap.png

Usage
-----
python validate_cleaned_results.py
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
    "/workspaces/pruna-cookbook/benchmark/validation_outputs"
)

PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MATPLOTLIB SETTINGS
# ============================================================

plt.rcParams["figure.dpi"] = 180
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("Loading cleaned dataset...")
print("=" * 80)

df = pd.read_csv(CSV_PATH)

print(f"\nRows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# NUMERIC COLUMNS
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

numeric_cols = [
    col for col in numeric_cols
    if col in df.columns
]


# ============================================================
# VALIDATION REPORT
# ============================================================

report_lines = []

report_lines.append("=" * 80)
report_lines.append("VALIDATION REPORT")
report_lines.append("=" * 80)

report_lines.append(f"\nTotal Rows: {len(df)}")
report_lines.append(f"Total Columns: {len(df.columns)}")


# ============================================================
# MISSING VALUES
# ============================================================

missing_df = (
    df.isnull().sum()
    .to_frame("missing_count")
)

missing_df["missing_percent"] = (
    missing_df["missing_count"] / len(df)
) * 100

missing_df = missing_df.sort_values(
    "missing_percent",
    ascending=False,
)

missing_df.to_csv(
    OUTPUT_DIR / "missing_values.csv"
)

report_lines.append("\nMissing Values:")
report_lines.append(str(missing_df))


# ============================================================
# DUPLICATES
# ============================================================

duplicates = df.duplicated().sum()

report_lines.append(
    f"\nDuplicate Rows: {duplicates}"
)


# ============================================================
# NUMERIC SUMMARY
# ============================================================

numeric_summary = (
    df[numeric_cols]
    .describe()
    .T
)

numeric_summary.to_csv(
    OUTPUT_DIR / "numeric_summary.csv"
)

report_lines.append("\nNumeric Summary:")
report_lines.append(str(numeric_summary))


# ============================================================
# NEGATIVE VALUE CHECK
# ============================================================

report_lines.append("\nNegative Values:")

for col in numeric_cols:

    negatives = (df[col] < 0).sum()

    report_lines.append(
        f"{col}: {negatives}"
    )


# ============================================================
# INFINITE VALUE CHECK
# ============================================================

report_lines.append("\nInfinite Values:")

for col in numeric_cols:

    infs = np.isinf(df[col]).sum()

    report_lines.append(
        f"{col}: {infs}"
    )


# ============================================================
# UNIQUENESS CHECK
# ============================================================

report_lines.append("\nUnique Value Counts:")

for col in numeric_cols:

    nunique = df[col].nunique()

    report_lines.append(
        f"{col}: {nunique}"
    )


# ============================================================
# OUTLIER DETECTION
# ============================================================

outlier_rows = []

report_lines.append("\nOutlier Detection:")

for col in numeric_cols:

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = (
        (df[col] < lower) |
        (df[col] > upper)
    )

    count = outliers.sum()

    outlier_rows.append({
        "column": col,
        "outlier_count": count,
        "outlier_percent": (
            count / len(df)
        ) * 100,
    })

    report_lines.append(
        f"{col}: {count}"
    )

outlier_df = pd.DataFrame(outlier_rows)

outlier_df.to_csv(
    OUTPUT_DIR / "outlier_summary.csv",
    index=False,
)


# ============================================================
# FRAMEWORK BALANCE
# ============================================================

if "framework" in df.columns:

    framework_counts = (
        df["framework"]
        .value_counts()
    )

    report_lines.append("\nFramework Counts:")
    report_lines.append(str(framework_counts))


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    OUTPUT_DIR / "validation_summary.txt",
    "w",
) as f:

    f.write("\n".join(report_lines))

print("\nValidation report saved.")


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
# MISSING VALUES PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    missing_df.index,
    missing_df["missing_percent"],
)

plt.xticks(rotation=90)

plt.ylabel("Missing %")
plt.title("Missing Values Percentage")

save_plot("missing_values.png")


# ============================================================
# GENERATION LENGTH VS MEMORY
# ============================================================

if (
    "generation_length" in df.columns
    and "peak_memory_gb" in df.columns
):

    plt.figure(figsize=(10, 6))

    for framework in df["framework"].unique():

        subset = df[
            df["framework"] == framework
        ]

        plt.scatter(
            subset["generation_length"],
            subset["peak_memory_gb"],
            alpha=0.7,
            s=70,
            label=framework,
        )

    plt.xlabel("Generation Length")
    plt.ylabel("Peak Memory (GB)")
    plt.title("Generation Length vs Peak Memory")
    plt.legend()

    save_plot("generation_vs_memory.png")


# ============================================================
# THROUGHPUT VS LATENCY
# ============================================================

if (
    "decode_tokens_per_sec" in df.columns
    and "avg_decode_latency_per_token_ms" in df.columns
):

    plt.figure(figsize=(10, 6))

    for framework in df["framework"].unique():

        subset = df[
            df["framework"] == framework
        ]

        plt.scatter(
            subset["avg_decode_latency_per_token_ms"],
            subset["decode_tokens_per_sec"],
            alpha=0.7,
            s=70,
            label=framework,
        )

    plt.xlabel("Avg Decode Latency Per Token (ms)")
    plt.ylabel("Decode Tokens/sec")
    plt.title("Throughput vs Latency")
    plt.legend()

    save_plot("throughput_vs_latency.png")


# ============================================================
# OUTLIER BOXPLOTS
# ============================================================

plot_cols = [
    "prefill_latency_s",
    "decode_tokens_per_sec",
    "peak_memory_gb",
]

plot_cols = [
    col for col in plot_cols
    if col in df.columns
]

fig, axes = plt.subplots(
    1,
    len(plot_cols),
    figsize=(6 * len(plot_cols), 6),
)

if len(plot_cols) == 1:
    axes = [axes]

for ax, col in zip(axes, plot_cols):

    ax.boxplot(
        df[col].dropna(),
    )

    ax.set_title(col)

save_plot("outlier_boxplots.png")


# ============================================================
# FRAMEWORK DISTRIBUTION
# ============================================================

if "framework" in df.columns:

    plt.figure(figsize=(7, 5))

    df["framework"].value_counts().plot(
        kind="bar"
    )

    plt.ylabel("Count")
    plt.title("Framework Distribution")

    save_plot("framework_distribution.png")


# ============================================================
# CORRELATION HEATMAP
# ============================================================

corr_df = (
    df[numeric_cols]
    .corr()
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
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)

print(f"\nValidation outputs saved to:\n{OUTPUT_DIR}")

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.glob("*")):
    print(f" - {file.name}")

print("\nGenerated plots:")

for file in sorted(PLOTS_DIR.glob("*.png")):
    print(f" - {file.name}")

print("\nDone.")