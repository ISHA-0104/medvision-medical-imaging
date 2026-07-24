import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_METRICS_CSV = BASE_DIR / "stage2_analysis" / "preprocessed_metrics.csv"
OUTPUT_DIR = BASE_DIR / "stage2_analysis"

METRICS = ["contrast", "snr", "entropy", "sharpness", "edge_strength", "noise_estimate"]
METHODS = ["Raw", "HE", "AHE", "CLAHE"]


def run_statistical_analysis():
    print("Loading preprocessed metric database...")
    if not INPUT_METRICS_CSV.exists():
        print(f"Error: Could not find {INPUT_METRICS_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_METRICS_CSV)

    # 1. Filter out segmentation mask rows
    df = df[df["modality"].str.lower() != "seg"].copy()
    print(f"Total valid volume metric records: {len(df)}")

    # Ensure enhancement_method order
    df["enhancement_method"] = pd.Categorical(df["enhancement_method"], categories=METHODS, ordered=True)

    # --- 1. Overall Method Comparison Summary (Mean ± Std) ---
    print("\nCalculating Method Comparison Summary...")
    method_stats = []
    for method in METHODS:
        m_df = df[df["enhancement_method"] == method]
        row = {"Enhancement_Method": method, "Count": len(m_df)}
        for metric in METRICS:
            mean = m_df[metric].mean()
            std = m_df[metric].std()
            row[f"{metric}_mean"] = round(mean, 4)
            row[f"{metric}_std"] = round(std, 4)
            row[f"{metric}_formatted"] = f"{mean:.4f} ± {std:.4f}"
        method_stats.append(row)

    df_method_summary = pd.DataFrame(method_stats)
    df_method_summary.to_csv(OUTPUT_DIR / "method_comparison_summary.csv", index=False)

    # --- 2. Statistical Significance Tests (ANOVA & Kruskal-Wallis) ---
    print("Performing Statistical Hypothesis Testing across methods...")
    test_results = []
    for metric in METRICS:
        groups = [df[df["enhancement_method"] == m][metric].dropna().values for m in METHODS]
        
        # One-way ANOVA
        f_stat, p_val_anova = stats.f_oneway(*groups)
        # Kruskal-Wallis (Non-parametric)
        kw_stat, p_val_kw = stats.kruskal(*groups)
        
        test_results.append({
            "Metric": metric,
            "ANOVA_F_Statistic": round(f_stat, 4),
            "ANOVA_p_value": p_val_anova,
            "ANOVA_Significant": p_val_anova < 0.05,
            "Kruskal_Statistic": round(kw_stat, 4),
            "Kruskal_p_value": p_val_kw,
            "Kruskal_Significant": p_val_kw < 0.05
        })

    df_tests = pd.DataFrame(test_results)
    df_tests.to_csv(OUTPUT_DIR / "statistical_test_results.csv", index=False)

    # --- 3. Organ-Wise Analysis (Brain vs Spine) ---
    print("Calculating Organ-Wise Summary (Brain vs Spine)...")
    organ_summary = df.groupby(["organ", "enhancement_method"])[METRICS].agg(["mean", "std"]).round(4)
    organ_summary.columns = [f"{col[0]}_{col[1]}" for col in organ_summary.columns]
    organ_summary.reset_index().to_csv(OUTPUT_DIR / "organ_wise_summary.csv", index=False)

    # --- 4. Pathology-Wise Analysis (Normal vs Pathological) ---
    print("Calculating Pathology-Wise Summary (Normal vs Pathological)...")
    path_summary = df.groupby(["pathological", "enhancement_method"])[METRICS].agg(["mean", "std"]).round(4)
    path_summary.columns = [f"{col[0]}_{col[1]}" for col in path_summary.columns]
    path_summary.reset_index().to_csv(OUTPUT_DIR / "pathology_wise_summary.csv", index=False)

    # --- 5. Quantitative Method Ranking & Selection ---
    print("Computing Quantitative Method Rankings...")
    # Standardize metrics so higher is better for Contrast, SNR, Entropy, Sharpness, Edge Strength
    # and lower is better for Noise
    ranking_scores = {m: 0.0 for m in METHODS if m != "Raw"}
    
    # Calculate Z-score means for enhanced methods relative to Raw baseline
    raw_df = df[df["enhancement_method"] == "Raw"]
    
    score_details = []
    for method in ["HE", "AHE", "CLAHE"]:
        m_df = df[df["enhancement_method"] == method]
        
        # Gain over Raw
        c_gain = (m_df["contrast"].mean() - raw_df["contrast"].mean()) / raw_df["contrast"].mean()
        snr_gain = (m_df["snr"].mean() - raw_df["snr"].mean()) / raw_df["snr"].mean()
        ent_gain = (m_df["entropy"].mean() - raw_df["entropy"].mean()) / raw_df["entropy"].mean()
        sharp_gain = (m_df["sharpness"].mean() - raw_df["sharpness"].mean()) / raw_df["sharpness"].mean()
        edge_gain = (m_df["edge_strength"].mean() - raw_df["edge_strength"].mean()) / raw_df["edge_strength"].mean()
        noise_penalty = (m_df["noise_estimate"].mean() - raw_df["noise_estimate"].mean()) / raw_df["noise_estimate"].mean()
        
        # Composite Quality Index (CQI)
        cqi = (c_gain * 0.25) + (snr_gain * 0.20) + (ent_gain * 0.15) + (sharp_gain * 0.20) + (edge_gain * 0.20) - (noise_penalty * 0.10)
        
        score_details.append({
            "Method": method,
            "Contrast_Gain_%": round(c_gain * 100, 2),
            "SNR_Gain_%": round(snr_gain * 100, 2),
            "Entropy_Gain_%": round(ent_gain * 100, 2),
            "Sharpness_Gain_%": round(sharp_gain * 100, 2),
            "Edge_Strength_Gain_%": round(edge_gain * 100, 2),
            "Noise_Increase_%": round(noise_penalty * 100, 2),
            "Composite_Quality_Score": round(cqi, 4)
        })

    df_ranking = pd.DataFrame(score_details).sort_values("Composite_Quality_Score", ascending=False)
    df_ranking.to_csv(OUTPUT_DIR / "preprocessing_method_ranking.csv", index=False)

    best_method = df_ranking.iloc[0]["Method"]
    best_score = df_ranking.iloc[0]["Composite_Quality_Score"]

    # --- Print Summary Output ---
    print("\n" + "="*60)
    print(" STAGE 2 STATISTICAL ANALYSIS & METHOD SELECTION SUMMARY")
    print("="*60)
    print("\n--- Overall Method Means ---")
    for row in method_stats:
        print(f"  {row['Enhancement_Method']:<6} | Contrast: {row['contrast_mean']:.4f} | SNR: {row['snr_mean']:.4f} | Entropy: {row['entropy_mean']:.4f} | Sharpness: {row['sharpness_mean']:.6f} | Edge: {row['edge_strength_mean']:.6f}")

    print("\n--- Quantitative Method Ranking (Composite Quality Score) ---")
    for _, r in df_ranking.iterrows():
        print(f"  Rank #{_ + 1}: {r['Method']:<6} | Composite Score: {r['Composite_Quality_Score']:+.4f} | Contrast Gain: {r['Contrast_Gain_%']:+6.2f}% | Edge Gain: {r['Edge_Strength_Gain_%']:+6.2f}%")

    print("\n" + "*"*60)
    print(f" OPTIMAL PREPROCESSING METHOD SELECTED: {best_method} (Score: {best_score:+.4f})")
    print("*"*60)
    print(f"\nGenerated CSV Summaries in {OUTPUT_DIR}:")
    print("  - method_comparison_summary.csv")
    print("  - statistical_test_results.csv")
    print("  - organ_wise_summary.csv")
    print("  - pathology_wise_summary.csv")
    print("  - preprocessing_method_ranking.csv")
    print("="*60)


if __name__ == "__main__":
    run_statistical_analysis()
