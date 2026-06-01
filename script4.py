import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "/workspaces/pruna-cookbook/benchmark/eda_outputs/aggregate_summary.csv"
OUTPUT_DIR = "/workspaces/pruna-cookbook/benchmark/eda_outputs"

# ============================================================
# DESIGN SYSTEM
# ============================================================

PRUNA_COLOR   = "#6A0DAD"   # purple  — Pruna
HQQ_COLOR     = "#E05C2A"   # amber-red — HQQ + torch.compile

PRUNA_LIGHT   = "#F3E8FF"   # soft purple tint for alternating rows
HQQ_LIGHT     = "#FFF0E8"   # soft amber tint for alternating rows
HEADER_BG     = "#1A1A2E"   # near-black header band
ROW_LABEL_BG  = "#F5F5F5"

# ============================================================
# LOAD CSV
# ============================================================

df_raw = pd.read_csv(CSV_PATH, header=[0, 1], index_col=0)

df_raw.columns = [
    f"{metric}_{stat}"
    for metric, stat in df_raw.columns
]
df_raw = df_raw.reset_index()

# ============================================================
# FORMAT VALUES
# ============================================================

def format_value(col, val):
    if pd.isna(val):
        return ""
    if "tokens_per_sec" in col:
        return f"{val:.2f}"
    return f"{val:.3f}"

formatted_df = df_raw.copy()
for col in formatted_df.columns:
    if col != "framework":
        formatted_df[col] = formatted_df[col].apply(
            lambda x: format_value(col, x)
        )

# ============================================================
# METRIC MAPPING
# ============================================================

metric_mapping = {
    "framework": "Framework",

    "prefill_latency_s_mean":   "Prefill Mean (s)",
    "prefill_latency_s_median": "Prefill Median (s)",
    "prefill_latency_s_std":    "Prefill Std (s)",
    "prefill_latency_s_max":    "Prefill Max (s)",

    "decode_tokens_per_sec_mean":   "Decode TPS Mean",
    "decode_tokens_per_sec_median": "Decode TPS Median",
    "decode_tokens_per_sec_std":    "Decode TPS Std",

    "avg_decode_latency_per_token_ms_mean":   "Decode Lat Mean (ms)",
    "avg_decode_latency_per_token_ms_median": "Decode Lat Median (ms)",
    "avg_decode_latency_per_token_ms_std":    "Decode Lat Std (ms)",

    "peak_memory_gb_mean": "Peak Mem Mean (GB)",
    "peak_memory_gb_std":  "Peak Mem Std (GB)",

    "memory_per_generated_token_mb_mean": "Mem / Token (MB)",
}

# ============================================================
# PREP DISPLAY DATAFRAME
# Column order: Pruna first, HQQ second
# ============================================================

display_df = (
    formatted_df[list(metric_mapping.keys())]
    .rename(columns=metric_mapping)
    .set_index("Framework")
    .T
)

# Ensure column order: Pruna | HQQ
col_order = [c for c in ["Pruna", "HQQ"] if c in display_df.columns]
display_df = display_df[col_order]

# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.axis("off")

# Column headers use full framework names
col_labels = ["Pruna (HQQ + torch.compile)", "Base (HQQ + torch.compile)"]

table = ax.table(
    cellText=display_df.values,
    rowLabels=display_df.index,
    colLabels=col_labels,
    cellLoc="center",
    rowLoc="right",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.15, 1.85)

# ============================================================
# CELL STYLING — no bolding of data values
# ============================================================

n_rows = len(display_df)

for (row, col), cell in table.get_celld().items():

    cell.set_linewidth(0.4)
    cell.set_edgecolor("#dddddd")

    # ── Column header row ──────────────────────────────────
    if row == 0:
        cell.set_height(0.082)
        cell.set_text_props(weight="bold", color="white", fontsize=10)

        if col == 0:          # Pruna header
            cell.set_facecolor(PRUNA_COLOR)
        elif col == 1:        # HQQ header
            cell.set_facecolor(HQQ_COLOR)
        else:                 # corner cell (col == -1 doesn't exist at row 0)
            cell.set_facecolor(HEADER_BG)

    # ── Row label column ───────────────────────────────────
    elif col == -1:
        cell.set_text_props(weight="bold", fontsize=9, color="#333333")
        cell.set_facecolor(ROW_LABEL_BG)
        cell.set_edgecolor("#cccccc")

    # ── Data cells — alternating tinted rows, plain text ──
    else:
        data_row = row - 1   # 0-indexed within data
        if data_row % 2 == 0:
            bg = PRUNA_LIGHT if col == 0 else HQQ_LIGHT
        else:
            bg = "white"

        cell.set_facecolor(bg)
        # Plain weight, framework color for easy column scanning
        cell.set_text_props(
            color=PRUNA_COLOR if col == 0 else HQQ_COLOR,
            fontsize=9.5,
        )

# ============================================================
# TITLE
# ============================================================

plt.title(
    "Aggregate Benchmark Summary\nPruna  vs  HQQ + torch.compile",
    fontsize=15,
    weight="bold",
    pad=20,
    color="#1A1A2E",
)

# ============================================================
# FOOTNOTE
# ============================================================

plt.figtext(
    0.5, 0.015,
    "Values are mean / median / std across all benchmark runs after removing failed HQQ runs (latency > 1000 ms).",
    ha="center",
    fontsize=8.5,
    color="#666666",
)

# ============================================================
# SAVE
# ============================================================

output_path = Path(OUTPUT_DIR) / "aggregate_summary_table_styled.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved table figure to: {output_path}")