import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- Paths & Config ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_METRICS_CSV = BASE_DIR / "stage2_analysis" / "preprocessed_metrics.csv"
RANKING_CSV = BASE_DIR / "stage2_analysis" / "preprocessing_method_ranking.csv"
PLOTS_DIR = BASE_DIR / "stage2_analysis" / "plots"

METRICS = ["contrast", "snr", "entropy", "sharpness", "edge_strength", "noise_estimate"]
METHODS = ["Raw", "HE", "AHE", "CLAHE"]

METRIC_LABELS = {
    "contrast": "RMS Contrast",
    "snr": "Signal-to-Noise Ratio (SNR)",
    "entropy": "Shannon Entropy (bits)",
    "sharpness": "Sharpness (Laplacian Var)",
    "edge_strength": "Edge Strength (Sobel Mean)",
    "noise_estimate": "Noise Estimate (Sigma)"
}


def generate_publication_plots():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading metrics and ranking data for publication plot generation...")

    if not INPUT_METRICS_CSV.exists():
        print(f"Error: Missing input metrics CSV at {INPUT_METRICS_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_METRICS_CSV)
    df = df[df["modality"].str.lower() != "seg"].copy()
    
    # Ensure ordered category
    df["enhancement_method"] = pd.Categorical(df["enhancement_method"], categories=METHODS, ordered=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plot_count = 0

    # --- 1. Violin Plots: Method Comparison across all 6 metrics ---
    print("\nGenerating Violin Plots (Method Comparison)...")
    for metric in METRICS:
        plt.figure(figsize=(9, 6))
        ax = sns.violinplot(
            data=df,
            x="enhancement_method",
            y=metric,
            hue="enhancement_method",
            palette="Set2",
            inner="quartile",
            legend=False
        )
        label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        plt.title(f"Stage 2 Preprocessing Comparison: {label}", fontsize=14, pad=12, fontweight="bold")
        plt.xlabel("Preprocessing / Enhancement Method", fontsize=12)
        plt.ylabel(label, fontsize=12)
        plt.tight_layout()

        fname = PLOTS_DIR / f"violin_{metric}_method_comparison.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        plot_count += 1
        print(f"[{plot_count}] Saved: {fname.name}")

    # --- 2. Organ-Wise Boxplots (Brain vs Spine) ---
    print("\nGenerating Organ-Wise Boxplots...")
    for metric in ["contrast", "edge_strength", "entropy"]:
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(
            data=df,
            x="enhancement_method",
            y=metric,
            hue="organ",
            palette="Dark2"
        )
        label = METRIC_LABELS.get(metric, metric)
        plt.title(f"Organ-Wise Comparison (Brain vs Spine): {label}", fontsize=14, pad=12, fontweight="bold")
        plt.xlabel("Enhancement Method", fontsize=12)
        plt.ylabel(label, fontsize=12)
        plt.legend(title="Organ", loc="upper left")
        plt.tight_layout()

        fname = PLOTS_DIR / f"organ_comparison_{metric}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        plot_count += 1
        print(f"[{plot_count}] Saved: {fname.name}")

    # --- 3. Pathology-Wise Boxplots (Normal vs Pathological) ---
    print("\nGenerating Pathology-Wise Boxplots...")
    for metric in ["contrast", "edge_strength"]:
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(
            data=df,
            x="enhancement_method",
            y=metric,
            hue="pathological",
            palette="Set1"
        )
        label = METRIC_LABELS.get(metric, metric)
        plt.title(f"Pathology Analysis (Normal vs Pathological): {label}", fontsize=14, pad=12, fontweight="bold")
        plt.xlabel("Enhancement Method", fontsize=12)
        plt.ylabel(label, fontsize=12)
        handles, _ = ax.get_legend_handles_labels()
        plt.legend(handles=handles, labels=["Normal (False)", "Pathological (True)"], title="Pathology Status")
        plt.tight_layout()

        fname = PLOTS_DIR / f"pathology_comparison_{metric}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        plot_count += 1
        print(f"[{plot_count}] Saved: {fname.name}")

    # --- 4. Correlation Heatmap ---
    print("\nGenerating Image Quality Metric Correlation Heatmap...")
    plt.figure(figsize=(8, 7))
    corr_matrix = df[METRICS].corr()
    
    # Clean label names for heatmap
    clean_labels = [METRIC_LABELS.get(m, m) for m in METRICS]
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        linewidths=0.5
    )
    plt.title("Correlation Matrix of MRI Quality & Structure Metrics", fontsize=14, pad=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fname = PLOTS_DIR / "correlation_heatmap_metrics.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    plot_count += 1
    print(f"[{plot_count}] Saved: {fname.name}")

    # --- 5. KDE Distribution Curves for Contrast & Edge Strength ---
    print("\nGenerating KDE Distribution Plots...")
    for metric in ["contrast", "edge_strength"]:
        plt.figure(figsize=(9, 6))
        for method in METHODS:
            sns.kdeplot(
                data=df[df["enhancement_method"] == method],
                x=metric,
                label=method,
                linewidth=2
            )
        label = METRIC_LABELS.get(metric, metric)
        plt.title(f"Distribution Curves across Enhancement Pipelines: {label}", fontsize=14, pad=12, fontweight="bold")
        plt.xlabel(label, fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(title="Method")
        plt.tight_layout()

        fname = PLOTS_DIR / f"kde_distribution_{metric}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        plot_count += 1
        print(f"[{plot_count}] Saved: {fname.name}")

    # --- 6. Composite Quality Index Method Ranking Bar Chart ---
    print("\nGenerating Method Ranking Bar Chart...")
    if RANKING_CSV.exists():
        df_rank = pd.read_csv(RANKING_CSV)
        plt.figure(figsize=(8, 5))
        bars = sns.barplot(
            data=df_rank,
            x="Method",
            y="Composite_Quality_Score",
            palette="viridis",
            hue="Method",
            legend=False
        )
        plt.title("Optimal Preprocessing Selection (Composite Quality Score)", fontsize=14, pad=12, fontweight="bold")
        plt.xlabel("Preprocessing Pipeline", fontsize=12)
        plt.ylabel("Composite Quality Score (CQI)", fontsize=12)
        
        # Add values on top of bars
        for bar in bars.patches:
            val = bar.get_height()
            bars.annotate(
                f"{val:+.4f}",
                (bar.get_x() + bar.get_width() / 2., val),
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=11,
                fontweight="bold",
                xytext=(0, 4),
                textcoords="offset points"
            )

        plt.tight_layout()
        fname = PLOTS_DIR / "method_ranking_bar_chart.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        plot_count += 1
        print(f"[{plot_count}] Saved: {fname.name}")

    print("\n" + "="*60)
    print(" STAGE 2 VISUALIZATION SUITE COMPLETE")
    print("="*60)
    print(f"Total publication-quality figures saved: {plot_count} PNG files (150 DPI)")
    print(f"Output directory: {PLOTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    generate_publication_plots()
