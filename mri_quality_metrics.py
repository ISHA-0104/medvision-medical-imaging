import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

# Graceful import check for required imaging/scientific packages
missing_packages = []
try:
    import cv2
except ImportError:
    missing_packages.append("opencv-python")

try:
    from skimage.filters import sobel
    from skimage.restoration import estimate_sigma
except ImportError:
    missing_packages.append("scikit-image")

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    missing_packages.append("scipy")

if missing_packages:
    print(f"Error: Missing required packages: {', '.join(missing_packages)}")
    print(f"Please install them via: pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Check PyWavelets for skimage.restoration.estimate_sigma
HAS_PYWT = True
try:
    import pywt
except ImportError:
    HAS_PYWT = False

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent
HACKATHON_INDEX_CSV = PROJECT_ROOT / "dataset_index.csv"
BRATS_INDEX_CSV = PROJECT_ROOT / "brats_index.csv"

HACKATHON_OUTPUT_CSV = PROJECT_ROOT / "hackathon_stats.csv"
BRATS_OUTPUT_CSV = PROJECT_ROOT / "brats_stats.csv"


def resolve_csv_path(filepath):
    p = Path(filepath)
    return p if p.is_absolute() else PROJECT_ROOT / p


def to_relative_path(path):
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(os.path.relpath(p, PROJECT_ROOT)).as_posix()


def estimate_noise_level(slice_2d):
    """
    Estimates noise level of a 2D slice.
    Uses skimage.restoration.estimate_sigma if PyWavelets is present,
    otherwise falls back to high-frequency residual standard deviation.
    """
    if HAS_PYWT:
        try:
            return float(estimate_sigma(slice_2d, average_sigmas=True))
        except Exception:
            pass
            
    # Fallback noise estimate: standard deviation of high-frequency residual
    blurred = gaussian_filter(slice_2d, sigma=1.0)
    residual = slice_2d - blurred
    return float(np.std(residual))


def compute_slice_metrics(filepath):
    """
    Loads NIfTI volume, extracts middle 2D slice along the last axis,
    and computes image quality metrics (RMS contrast on foreground,
    and min-max normalized sharpness and edge strength).
    """
    img = nib.load(filepath)
    data = img.get_fdata()

    # Extract 2D slice from middle along slice dimension (axis 2 or last spatial axis)
    if data.ndim == 3:
        mid_idx = data.shape[2] // 2
        slice_2d = data[:, :, mid_idx]
    elif data.ndim == 4:
        mid_idx = data.shape[2] // 2
        slice_2d = data[:, :, mid_idx, 0]
    elif data.ndim == 2:
        slice_2d = data
    else:
        slice_2d = data.take(indices=data.shape[-1] // 2, axis=-1)

    slice_2d = np.ascontiguousarray(slice_2d, dtype=np.float64)

    # 1. Mean intensity & 2. Standard deviation (computed on full slice)
    mean_val = float(np.mean(slice_2d))
    std_val = float(np.std(slice_2d))

    # Foreground Mask (exclude background air voxels at/near intensity 0)
    fg_mask = slice_2d > 1.0
    fg_pixels = slice_2d[fg_mask]

    # --- Fix 1: RMS Contrast on Foreground ---
    if fg_pixels.size > 0:
        fg_mean = float(np.mean(fg_pixels))
        fg_std = float(np.std(fg_pixels))
        contrast_val = float(fg_std / fg_mean) if fg_mean > 0 else 0.0
    else:
        contrast_val = 0.0

    # --- Fix 2: Normalized Sharpness & Edge Strength ---
    if fg_pixels.size > 0:
        fg_min = float(np.min(fg_pixels))
        fg_max = float(np.max(fg_pixels))
        
        if fg_max > fg_min:
            norm_slice = (slice_2d - fg_min) / (fg_max - fg_min)
            norm_slice = np.clip(norm_slice, 0.0, 1.0)
        else:
            norm_slice = np.zeros_like(slice_2d)
    else:
        norm_slice = np.zeros_like(slice_2d)

    # 4. Sharpness (Variance of Laplacian on 0-1 normalized slice)
    sharpness_val = float(cv2.Laplacian(norm_slice, cv2.CV_64F).var())

    # 5. Edge Strength (Mean Sobel Gradient on 0-1 normalized slice)
    edge_val = float(np.mean(sobel(norm_slice)))

    # 6. Noise Level Estimate (unchanged)
    noise_val = estimate_noise_level(slice_2d)

    return {
        "mean_intensity": round(mean_val, 4),
        "std_intensity": round(std_val, 4),
        "contrast": round(contrast_val, 4),
        "sharpness": round(sharpness_val, 4),
        "edge_strength": round(edge_val, 4),
        "noise_estimate": round(noise_val, 4),
    }


def main():
    print("Loading dataset index CSVs...")

    # Load dataset_index.csv (Hackathon offline)
    if not os.path.exists(HACKATHON_INDEX_CSV):
        print(f"Error: Could not find {HACKATHON_INDEX_CSV}")
        sys.exit(1)
    df_hackathon = pd.read_csv(HACKATHON_INDEX_CSV)
    if "dataset_source" not in df_hackathon.columns:
        df_hackathon["dataset_source"] = "hackathon_offline"

    # Load brats_index.csv (BraTS 2020)
    if not os.path.exists(BRATS_INDEX_CSV):
        print(f"Error: Could not find {BRATS_INDEX_CSV}")
        sys.exit(1)
    df_brats = pd.read_csv(BRATS_INDEX_CSV)
    if "dataset_source" not in df_brats.columns:
        df_brats["dataset_source"] = "brats_training"

    # Concatenate into single combined DataFrame
    combined_df = pd.concat([df_hackathon, df_brats], ignore_index=True)
    total_files = len(combined_df)
    print(f"Total files to process: {total_files} ({len(df_hackathon)} hackathon_offline, {len(df_brats)} brats_training)")

    # Prepare storage for metrics
    metrics_list = []
    failures = []
    success_count = 0
    skipped_seg_count = 0

    print("\nStarting slice metrics computation (with updated foreground RMS & min-max normalized edge metrics)...")
    for idx, row in combined_df.iterrows():
        filepath = resolve_csv_path(row["filepath"])
        modality = str(row["modality"])

        # Skip metrics computation for segmentation masks, but keep the row
        if modality.lower() in ["seg", "mask", "label"]:
            metrics_list.append({
                "mean_intensity": np.nan,
                "std_intensity": np.nan,
                "contrast": np.nan,
                "sharpness": np.nan,
                "edge_strength": np.nan,
                "noise_estimate": np.nan,
            })
            skipped_seg_count += 1
            success_count += 1
        else:
            try:
                metrics = compute_slice_metrics(filepath)
                metrics_list.append(metrics)
                success_count += 1
            except Exception as e:
                failures.append((filepath, str(e)))
                metrics_list.append({
                    "mean_intensity": np.nan,
                    "std_intensity": np.nan,
                    "contrast": np.nan,
                    "sharpness": np.nan,
                    "edge_strength": np.nan,
                    "noise_estimate": np.nan,
                })

        # Progress reporting every 20 files
        if (idx + 1) % 20 == 0 or (idx + 1) == total_files:
            print(f"Progress: Processed {idx + 1}/{total_files} files...")

    # Attach computed metrics to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    result_df = pd.concat([combined_df, metrics_df], axis=1)

    # Split back into separate datasets
    df_hackathon_out = result_df[result_df["dataset_source"] == "hackathon_offline"].copy()
    df_brats_out = result_df[result_df["dataset_source"] == "brats_training"].copy()

    # Ensure output CSVs keep relative paths for portability
    df_hackathon_out["filepath"] = df_hackathon_out["filepath"].apply(to_relative_path)
    df_brats_out["filepath"] = df_brats_out["filepath"].apply(to_relative_path)

    # Save CSVs
    df_hackathon_out.to_csv(HACKATHON_OUTPUT_CSV, index=False)
    df_brats_out.to_csv(BRATS_OUTPUT_CSV, index=False)

    print("\n" + "="*50)
    print(" MRI IMAGE QUALITY METRICS SUMMARY (UPDATED)")
    print("="*50)
    print(f"Total files in combined index: {total_files}")
    print(f"Successfully processed:       {success_count} (including {skipped_seg_count} segmentation masks)")
    print(f"Failed / unreadable files:     {len(failures)}")

    if failures:
        print("\n--- Failed Files ---")
        for path, err in failures:
            print(f"  - {path}: {err}")

    # Groupby summary table (mean metrics by organ & modality)
    metric_cols = ["mean_intensity", "std_intensity", "contrast", "sharpness", "edge_strength", "noise_estimate"]
    summary_table = result_df.groupby(["organ", "modality"])[metric_cols].mean()

    print("\n--- Mean Metrics by Organ & Modality ---")
    print(summary_table.to_string())

    print("\nSaved output files:")
    print(f"  - {HACKATHON_OUTPUT_CSV} ({len(df_hackathon_out)} rows)")
    print(f"  - {BRATS_OUTPUT_CSV} ({len(df_brats_out)} rows)")
    print("="*50)


if __name__ == "__main__":
    main()
