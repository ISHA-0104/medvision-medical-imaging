import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import nibabel as nib

# Import imaging and metric libraries
try:
    import cv2
    from skimage.filters import sobel
    from skimage.measure import shannon_entropy
    from skimage.restoration import estimate_sigma
    from scipy.ndimage import gaussian_filter
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please ensure opencv-python, scikit-image, and scipy are installed.")
    sys.exit(1)

# PyWavelets check for estimate_sigma
HAS_PYWT = True
try:
    import pywt
except ImportError:
    HAS_PYWT = False

# --- Paths & Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT
PREPROCESSED_INDEX_CSV = BASE_DIR / "preprocessed_index.csv"
RAW_INDEX_CSV = BASE_DIR / "dataset_index.csv"
BRATS_INDEX_CSV = BASE_DIR / "brats_index.csv"

OUTPUT_DIR = BASE_DIR / "stage2_analysis"
OUTPUT_METRICS_CSV = OUTPUT_DIR / "preprocessed_metrics.csv"

MAX_WORKERS = min(8, os.cpu_count() or 4)


def estimate_noise_level(slice_2d):
    """Estimates noise level sigma of a 2D slice."""
    if HAS_PYWT:
        try:
            return float(estimate_sigma(slice_2d, average_sigmas=True))
        except Exception:
            pass
    blurred = gaussian_filter(slice_2d, sigma=1.0)
    residual = slice_2d - blurred
    return float(np.std(residual))


def compute_comprehensive_metrics(filepath):
    """
    Loads NIfTI volume, extracts 2D middle slice, and computes:
    - Mean & Std intensity
    - RMS Foreground Contrast
    - Shannon Entropy
    - Signal-to-Noise Ratio (SNR)
    - Min-max normalized Sharpness (Laplacian variance)
    - Min-max normalized Edge Strength (Sobel mean)
    - Noise Level Estimate
    """
    img = nib.load(filepath)
    data = img.get_fdata()

    # Extract 2D middle slice along axis 2
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

    # 1. Mean & 2. Std Intensity (Full slice)
    mean_val = float(np.mean(slice_2d))
    std_val = float(np.std(slice_2d))

    # Foreground Mask (exclude background air voxels at/near 0)
    fg_mask = slice_2d > 1.0 if np.max(slice_2d) > 1.0 else slice_2d > 0.01
    fg_pixels = slice_2d[fg_mask]

    # 3. RMS Foreground Contrast
    if fg_pixels.size > 0:
        fg_mean = float(np.mean(fg_pixels))
        fg_std = float(np.std(fg_pixels))
        contrast_val = float(fg_std / fg_mean) if fg_mean > 0 else 0.0
        # 5. SNR = fg_mean / fg_std
        snr_val = float(fg_mean / fg_std) if fg_std > 0 else 0.0
    else:
        contrast_val = 0.0
        snr_val = 0.0

    # 4. Shannon Entropy
    try:
        entropy_val = float(shannon_entropy(slice_2d))
    except Exception:
        entropy_val = 0.0

    # Normalization over foreground for scale-invariant Sharpness & Edge Strength
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

    # 6. Sharpness (Laplacian Variance)
    sharpness_val = float(cv2.Laplacian(norm_slice, cv2.CV_64F).var())

    # 7. Edge Strength (Sobel Gradient Mean)
    edge_val = float(np.mean(sobel(norm_slice)))

    # 8. Noise Estimate
    noise_val = estimate_noise_level(slice_2d)

    return {
        "mean_intensity": round(mean_val, 4),
        "std_intensity": round(std_val, 4),
        "contrast": round(contrast_val, 4),
        "entropy": round(entropy_val, 4),
        "snr": round(snr_val, 4),
        "sharpness": round(sharpness_val, 6),
        "edge_strength": round(edge_val, 6),
        "noise_estimate": round(noise_val, 4),
    }


def resolve_csv_path(filepath):
    p = Path(filepath)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def to_relative_path(path):
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(os.path.relpath(p, PROJECT_ROOT)).as_posix()


def process_row_worker(row):
    """
    Worker task for a single row from index dataframe.
    """
    patient_id = str(row["patient_id"])
    organ = str(row["organ"])
    modality = str(row["modality"])
    pathological = row["pathological"]
    dataset_source = str(row["dataset_source"])
    method = str(row["enhancement_method"])
    filepath = resolve_csv_path(row["filepath"])

    # Skip metrics for segmentation masks
    if modality.lower() in ["seg", "mask", "label"]:
        return {
            "patient_id": patient_id,
            "organ": organ,
            "modality": modality,
            "pathological": pathological,
            "dataset_source": dataset_source,
            "enhancement_method": method,
                "filepath": to_relative_path(filepath),
        }, None

    try:
        metrics = compute_comprehensive_metrics(filepath)
        rel_filepath = to_relative_path(filepath)
        res = {
            "patient_id": patient_id,
            "organ": organ,
            "modality": modality,
            "pathological": pathological,
            "dataset_source": dataset_source,
            "enhancement_method": method,
            "filepath": rel_filepath,
        }
        res.update(metrics)
        return res, None
    except Exception as e:
        return None, (filepath, str(e))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building comprehensive index for metrics calculation...")
    if not PREPROCESSED_INDEX_CSV.exists():
        print(f"Error: Missing preprocessed index CSV at {PREPROCESSED_INDEX_CSV}")
        sys.exit(1)

    df_prep = pd.read_csv(PREPROCESSED_INDEX_CSV)
    df_prep = df_prep.rename(columns={"new_filepath": "filepath"})
    if "filepath" in df_prep.columns:
        df_prep["filepath"] = df_prep["filepath"].apply(lambda p: resolve_csv_path(p).as_posix())

    # Also build Raw rows from dataset_index.csv and brats_index.csv for Raw vs Enhanced baseline comparison
    raw_rows = []
    if RAW_INDEX_CSV.exists():
        df_raw1 = pd.read_csv(RAW_INDEX_CSV)
        if "dataset_source" not in df_raw1.columns:
            df_raw1["dataset_source"] = "hackathon_offline"
        df_raw1["enhancement_method"] = "Raw"
        if "filepath" in df_raw1.columns:
            df_raw1["filepath"] = df_raw1["filepath"].apply(lambda p: resolve_csv_path(p).as_posix())
        raw_rows.append(df_raw1)

    if BRATS_INDEX_CSV.exists():
        df_raw2 = pd.read_csv(BRATS_INDEX_CSV)
        if "dataset_source" not in df_raw2.columns:
            df_raw2["dataset_source"] = "brats_training"
        df_raw2["enhancement_method"] = "Raw"
        if "filepath" in df_raw2.columns:
            df_raw2["filepath"] = df_raw2["filepath"].apply(lambda p: resolve_csv_path(p).as_posix())
        raw_rows.append(df_raw2)

    if raw_rows:
        df_raw = pd.concat(raw_rows, ignore_index=True)
        # Select common columns
        cols = ["patient_id", "organ", "modality", "pathological", "dataset_source", "enhancement_method", "filepath"]
        df_raw_clean = df_raw[[c for c in cols if c in df_raw.columns]]
        df_prep_clean = df_prep[[c for c in cols if c in df_prep.columns]]
        combined_df = pd.concat([df_raw_clean, df_prep_clean], ignore_index=True)
    else:
        combined_df = df_prep

    # Drop duplicate entries if any
    combined_df = combined_df.drop_duplicates(subset=["filepath", "enhancement_method"])
    total_files = len(combined_df)
    print(f"Total entries to compute (Raw + HE + AHE + CLAHE): {total_files}")
    print(f"Parallel Execution enabled: {MAX_WORKERS} workers across CPU cores.")

    results = []
    failures = []
    completed = 0
    start_time = time.time()

    print("\nExtracting Stage 2 Image Quality Metrics...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row_worker, row): idx for idx, row in combined_df.iterrows()}

        for future in as_completed(futures):
            completed += 1
            res, err = future.result()
            if res:
                results.append(res)
            if err:
                failures.append(err)

            if completed % 50 == 0 or completed == total_files:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"Progress: Processed {completed}/{total_files} entries ({rate:.1f} entries/sec)...")

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_METRICS_CSV, index=False)

    print("\n" + "="*60)
    print(" STAGE 2 METRIC EXTRACTION COMPLETE")
    print("="*60)
    print(f"Total entries processed: {len(df_out)}")
    print(f"Failures / errors:       {len(failures)}")
    print(f"Saved metric database:   {OUTPUT_METRICS_CSV}")
    print("="*60)


if __name__ == "__main__":
    main()
