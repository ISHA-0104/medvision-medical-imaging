import os
import sys

# Configure matplotlib to use non-interactive Agg backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Configuration ---
HACKATHON_CSV = "hackathon_stats.csv"
BRATS_CSV = "brats_stats.csv"
OUTPUT_DIR = "."  # Directory to save generated plot PNGs

METRICS = [
    "mean_intensity",
    "std_intensity",
    "contrast",
    "sharpness",
    "edge_strength",
    "noise_estimate",
]


def format_title(metric_name):
    """Utility to convert snake_case metric names into clean titles."""
    return metric_name.replace("_", " ").title()


def generate_plots():
    print("Loading quality metric CSV files...")
    
    if not os.path.exists(HACKATHON_CSV):
        print(f"Error: Could not find {HACKATHON_CSV}")
        sys.exit(1)
        
    if not os.path.exists(BRATS_CSV):
        print(f"Error: Could not find {BRATS_CSV}")
        sys.exit(1)

    df_hackathon = pd.read_csv(HACKATHON_CSV)
    df_brats = pd.read_csv(BRATS_CSV)

    # 1. Drop segmentation mask rows (modality == "seg")
    df_hackathon = df_hackathon[df_hackathon["modality"].str.lower() != "seg"].copy()
    df_brats = df_brats[df_brats["modality"].str.lower() != "seg"].copy()

    # 2. Add combined "organ - modality" column for Hackathon dataset
    df_hackathon["organ_modality"] = df_hackathon["organ"].str.capitalize() + " - " + df_hackathon["modality"]

    # 3. Prepare combined Brain dataset for Hackathon vs BraTS comparison
    df_hack_brain = df_hackathon[df_hackathon["organ"].str.lower() == "brain"].copy()
    df_hack_brain["source"] = "Hackathon Test/Val"
    
    df_brats_brain = df_brats.copy()
    df_brats_brain["source"] = "BraTS Training"

    df_brain_combined = pd.concat([df_hack_brain, df_brats_brain], ignore_index=True)

    # Styling configuration
    sns.set_theme(style="whitegrid")
    plot_count = 0

    print(f"\nStarting plot generation for {len(METRICS)} metrics across 3 analysis categories...")

    # --- Plot Set A: Hackathon Metrics by Combined (Organ - Modality) ---
    print("\n--- Generating Set A: Hackathon Metrics by Organ & Modality ---")
    for metric in METRICS:
        plt.figure(figsize=(10, 6))
        
        # Sort category order for consistent plotting
        order = sorted(df_hackathon["organ_modality"].unique())
        
        ax = sns.boxplot(
            data=df_hackathon,
            x="organ_modality",
            y=metric,
            order=order,
            hue="organ_modality",
            palette="Set2",
            legend=False
        )
        
        title_str = format_title(metric)
        plt.title(f"Hackathon Offline MRI: {title_str} by Organ & Modality", fontsize=14, pad=12)
        plt.xlabel("Organ - Modality", fontsize=12)
        plt.ylabel(title_str, fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        filename = f"hackathon_{metric}_by_organ_modality.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        plot_count += 1
        print(f"[{plot_count}/18] Saved: {filename}")

    # --- Plot Set B: Hackathon Normal vs Pathological per Organ ---
    print("\n--- Generating Set B: Hackathon Normal vs Pathological by Organ ---")
    for metric in METRICS:
        plt.figure(figsize=(8, 6))
        
        ax = sns.boxplot(
            data=df_hackathon,
            x="organ",
            y=metric,
            hue="pathological",
            palette="Set1"
        )
        
        title_str = format_title(metric)
        plt.title(f"Hackathon Offline MRI: {title_str} (Normal vs Pathological)", fontsize=14, pad=12)
        plt.xlabel("Organ", fontsize=12)
        plt.ylabel(title_str, fontsize=12)
        
        # Customize legend titles
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=["Normal (False)", "Pathological (True)"], title="Pathology Status")
        
        plt.tight_layout()

        filename = f"hackathon_{metric}_normal_vs_pathological.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        plot_count += 1
        print(f"[{plot_count}/18] Saved: {filename}")

    # --- Plot Set C: Brain MRI - Hackathon Test/Val vs BraTS Training ---
    print("\n--- Generating Set C: Brain MRI (Hackathon Test/Val vs BraTS Training) ---")
    for metric in METRICS:
        plt.figure(figsize=(9, 6))
        
        ax = sns.boxplot(
            data=df_brain_combined,
            x="modality",
            y=metric,
            hue="source",
            palette="Dark2"
        )
        
        title_str = format_title(metric)
        plt.title(f"Brain MRI: {title_str} by Modality (Hackathon vs BraTS)", fontsize=14, pad=12)
        plt.xlabel("MRI Modality", fontsize=12)
        plt.ylabel(title_str, fontsize=12)
        plt.legend(title="Dataset Source")
        
        plt.tight_layout()

        filename = f"brain_{metric}_hackathon_vs_brats.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        plot_count += 1
        print(f"[{plot_count}/18] Saved: {filename}")

    print("\n" + "="*50)
    print(" MRI METRIC PLOTTING COMPLETE")
    print("="*50)
    print(f"Total plots generated & saved: {plot_count} PNG files (150 DPI)")
    print("="*50)


if __name__ == "__main__":
    generate_plots()
