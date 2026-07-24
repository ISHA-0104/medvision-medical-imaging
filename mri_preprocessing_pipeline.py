import sys
import os
import shutil
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import nibabel as nib

PROJECT_ROOT = Path(__file__).resolve().parent
# Force temporary files to be created in the project workspace
TEMP_DIR = PROJECT_ROOT / "tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TMPDIR"] = str(TEMP_DIR)
os.environ["TEMP"] = str(TEMP_DIR)
os.environ["TMP"] = str(TEMP_DIR)

# Check required third-party libraries gracefully
missing = []
try:
    import SimpleITK as sitk
except ImportError:
    missing.append("SimpleITK")

try:
    from skimage.restoration import denoise_nl_means, estimate_sigma
    from skimage.exposure import equalize_hist, equalize_adapthist
except ImportError:
    missing.append("scikit-image")

try:
    from scipy import ndimage
except ImportError:
    missing.append("scipy")

if missing:
    print(f"Error: Missing required packages: {', '.join(missing)}")
    print(f"Please install them using: pip install {' '.join(missing)}")
    sys.exit(1)

# --- Configuration (100% Strict D: Drive Paths) ---
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT

HACKATHON_INDEX_CSV = BASE_DIR / "dataset_index.csv"
BRATS_INDEX_CSV = BASE_DIR / "brats_index.csv"

OUTPUT_ROOT_HE = BASE_DIR / "preprocessed_HE"
OUTPUT_ROOT_AHE = BASE_DIR / "preprocessed_AHE"
OUTPUT_ROOT_CLAHE = BASE_DIR / "preprocessed_CLAHE"

PREPROCESSED_INDEX_CSV = BASE_DIR / "preprocessed_index.csv"
MAX_WORKERS = min(8, os.cpu_count() or 4)

TARGET_SIZE = (256, 256)  # Standardized in-plane size (see resample_image docstring)


def load_sitk_image(filepath, is_label=False):
    """
    Load a NIfTI volume via SimpleITK so spacing, origin, and direction
    are taken directly from the file header (not nibabel zooms alone).
    """
    sitk_img = sitk.ReadImage(str(filepath))
    if is_label:
        return sitk.Cast(sitk_img, sitk.sitkUInt8)
    return sitk.Cast(sitk_img, sitk.sitkFloat32)


def resample_image(sitk_img, target_size=(256, 256), is_label=False):
    """
    Resample a 3D volume to a fixed in-plane matrix size using SimpleITK.

    Strategy: fixed image dimensions (256 x 256 in-plane), not fixed voxel spacing.
    Heterogeneous sources (e.g. hackathon 256x256, BraTS 240x240) need a common
    slice matrix for slice-wise NLM and histogram enhancement; depth (slice count)
    is kept unchanged. In-plane spacing is scaled to preserve field of view while
    origin and direction are carried through from the source image.
    """
    original_size = sitk_img.GetSize()
    original_spacing = sitk_img.GetSpacing()
    
    if len(original_size) == 3:
        target_size_3d = (target_size[0], target_size[1], original_size[2])
        # Compute new spacing to preserve physical field of view (extent)
        new_spacing = (
            original_spacing[0] * (original_size[0] / target_size[0]),
            original_spacing[1] * (original_size[1] / target_size[1]),
            original_spacing[2]
        )
    else:
        target_size_3d = target_size
        new_spacing = original_spacing

    resample = sitk.ResampleImageFilter()
    resample.SetSize(target_size_3d)
    resample.SetOutputSpacing(new_spacing)
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetOutputDirection(sitk_img.GetDirection())
    
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
        
    return resample.Execute(sitk_img)


def n4_bias_correction(vol_data):
    """
    Applies accelerated SimpleITK N4BiasFieldCorrection to a 3D volume.
    Uses 2x downsampling for log-bias field computation for speed.
    """
    vol_float = vol_data.astype(np.float32)
    sitk_img = sitk.GetImageFromArray(vol_float)
    
    try:
        shrink_img = sitk.Shrink(sitk_img, [2, 2, 2])
        mask = sitk.OtsuThreshold(shrink_img, 0, 1, 200)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([10, 10, 10])
        corrector.Execute(shrink_img, mask)

        log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_img)
        bias_field = sitk.Exp(log_bias_field)
        corrected_sitk = sitk_img / bias_field
        
        return sitk.GetArrayFromImage(corrected_sitk)
    except Exception:
        return vol_float


def denoise_volume(vol_data):
    """
    Denoises 3D volume slice-by-slice using Non-Local Means (skimage.restoration.denoise_nl_means).
    """
    denoised = np.zeros_like(vol_data, dtype=np.float32)
    num_slices = vol_data.shape[-1]
    
    for s in range(num_slices):
        if vol_data.ndim == 3:
            slice_2d = vol_data[:, :, s]
        elif vol_data.ndim == 4:
            slice_2d = vol_data[:, :, s, 0]
            
        try:
            sigma = float(estimate_sigma(slice_2d, average_sigmas=True))
        except Exception:
            sigma = float(np.std(slice_2d)) if np.std(slice_2d) > 0 else 1.0
            
        if sigma > 0 and np.max(slice_2d) > 0:
            denoised_slice = denoise_nl_means(
                slice_2d,
                h=0.8 * sigma,
                patch_size=5,
                patch_distance=3,
                fast_mode=True
            )
        else:
            denoised_slice = slice_2d
            
        if vol_data.ndim == 3:
            denoised[:, :, s] = denoised_slice
        elif vol_data.ndim == 4:
            denoised[:, :, s, 0] = denoised_slice
            
    return denoised


def normalize_intensity(vol_data):
    """Min-max normalizes volume to [0, 1] using foreground (non-zero) voxels only."""
    fg_mask = vol_data > 0
    fg_pixels = vol_data[fg_mask]
    
    if fg_pixels.size > 0:
        fg_min = float(np.min(fg_pixels))
        fg_max = float(np.max(fg_pixels))
        
        if fg_max > fg_min:
            norm_vol = (vol_data - fg_min) / (fg_max - fg_min)
            norm_vol = np.clip(norm_vol, 0.0, 1.0)
            return norm_vol.astype(np.float32)
            
    return np.zeros_like(vol_data, dtype=np.float32)


def enhance_contrast_slice_by_slice(norm_vol):
    """
    Applies 3 contrast enhancement methods slice-by-slice:
    1. HE: Histogram Equalization (equalize_hist)
    2. AHE: Adaptive Histogram Equalization (equalize_adapthist, clip_limit=0.01)
    3. CLAHE: Contrast Limited AHE (equalize_adapthist, clip_limit=0.03)
    """
    he_vol = np.zeros_like(norm_vol, dtype=np.float32)
    ahe_vol = np.zeros_like(norm_vol, dtype=np.float32)
    clahe_vol = np.zeros_like(norm_vol, dtype=np.float32)
    
    num_slices = norm_vol.shape[-1]
    
    for s in range(num_slices):
        if norm_vol.ndim == 3:
            slice_2d = norm_vol[:, :, s]
        elif norm_vol.ndim == 4:
            slice_2d = norm_vol[:, :, s, 0]
            
        if np.max(slice_2d) > 0:
            he_slice = equalize_hist(slice_2d)
            ahe_slice = equalize_adapthist(slice_2d, clip_limit=0.01)
            clahe_slice = equalize_adapthist(slice_2d, clip_limit=0.03)
        else:
            he_slice = slice_2d
            ahe_slice = slice_2d
            clahe_slice = slice_2d
            
        if norm_vol.ndim == 3:
            he_vol[:, :, s] = he_slice
            ahe_vol[:, :, s] = ahe_slice
            clahe_vol[:, :, s] = clahe_slice
        elif norm_vol.ndim == 4:
            he_vol[:, :, s, 0] = he_slice
            ahe_vol[:, :, s, 0] = ahe_slice
            clahe_vol[:, :, s, 0] = clahe_slice
            
    return he_vol, ahe_vol, clahe_vol


def save_nifti(vol_data, orig_img, dest_path):
    """Saves numpy volume data back into a .nii.gz file using nibabel."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    new_img = nib.Nifti1Image(vol_data.astype(np.float32), affine=orig_img.affine, header=orig_img.header)
    new_img.to_filename(str(dest_path))


def process_single_file(row):
    """
    Worker task function for processing a single MRI volume or segmentation mask.
    """
    patient_id = str(row["patient_id"])
    organ = str(row["organ"])
    modality = str(row["modality"])
    pathological = row["pathological"]
    dataset_source = str(row["dataset_source"])
    orig_filepath = Path(row["filepath"])
    if not orig_filepath.is_absolute():
        orig_filepath = PROJECT_ROOT / orig_filepath
    
    orig_fname = orig_filepath.name
    if orig_fname.endswith(".nii") and not orig_fname.endswith(".nii.gz"):
        dest_fname = orig_fname + ".gz"
    else:
        dest_fname = orig_fname

    rel_subpath = Path(organ) / dataset_source / patient_id / dest_fname
    path_he = OUTPUT_ROOT_HE / rel_subpath
    path_ahe = OUTPUT_ROOT_AHE / rel_subpath
    path_clahe = OUTPUT_ROOT_CLAHE / rel_subpath

    rows_to_add = []

    # Handle segmentation mask files: resample and copy
    if modality.lower() in ["seg", "mask", "label"]:
        try:
            sitk_img = load_sitk_image(orig_filepath, is_label=True)

            # Resample with Nearest Neighbor to preserve discrete labels
            resampled_sitk = resample_image(sitk_img, target_size=TARGET_SIZE, is_label=True)
            
            for dest_p in [path_he, path_ahe, path_clahe]:
                dest_p.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(resampled_sitk, str(dest_p))

            rows_to_add.append({
                "patient_id": patient_id,
                "organ": organ,
                "modality": modality,
                "pathological": pathological,
                "dataset_source": dataset_source,
                "enhancement_method": "none_seg_mask",
                "original_filepath": orig_filepath.relative_to(PROJECT_ROOT).as_posix() if orig_filepath.is_absolute() else orig_filepath.as_posix(),
                "new_filepath": path_he.relative_to(PROJECT_ROOT).as_posix(),
            })
            return True, rows_to_add, None, True
        except Exception as e:
            return False, [], (str(orig_filepath), f"Mask resample error: {e}"), True
    else:
        try:
            sitk_img = load_sitk_image(orig_filepath, is_label=False)

            # 1. Resample at the very beginning of the pipeline
            resized_sitk = resample_image(sitk_img, target_size=TARGET_SIZE, is_label=False)
            resized_vol = sitk.GetArrayFromImage(resized_sitk)

            # 2. N4 Bias Field Correction
            bias_corrected = n4_bias_correction(resized_vol)

            # 3. Denoising
            denoised = denoise_volume(bias_corrected)

            # 4. Intensity Normalization
            norm_vol = normalize_intensity(denoised)

            # 5. Contrast Enhancement
            he_vol, ahe_vol, clahe_vol = enhance_contrast_slice_by_slice(norm_vol)

            # Convert back to SimpleITK & save with metadata preservation
            for vol_data_out, dest_p in [(he_vol, path_he), (ahe_vol, path_ahe), (clahe_vol, path_clahe)]:
                out_sitk = sitk.GetImageFromArray(vol_data_out.astype(np.float32))
                out_sitk.CopyInformation(resized_sitk)
                dest_p.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(out_sitk, str(dest_p))

            for method_name, dest_p in [("HE", path_he), ("AHE", path_ahe), ("CLAHE", path_clahe)]:
                rows_to_add.append({
                    "patient_id": patient_id,
                    "organ": organ,
                    "modality": modality,
                    "pathological": pathological,
                    "dataset_source": dataset_source,
                    "enhancement_method": method_name,
                    "original_filepath": orig_filepath.relative_to(PROJECT_ROOT).as_posix() if orig_filepath.is_absolute() else orig_filepath.as_posix(),
                    "new_filepath": dest_p.relative_to(PROJECT_ROOT).as_posix(),
                })

            return True, rows_to_add, None, False
        except Exception as e:
            return False, [], (str(orig_filepath), str(e)), False


def process_pipeline():
    print(f"Strict D: Drive Working Directory: {BASE_DIR}")
    print(f"Temporary Files Directory: {TEMP_DIR}")
    print("Loading dataset index CSV files...")
    
    if not HACKATHON_INDEX_CSV.exists() or not BRATS_INDEX_CSV.exists():
        print(f"Error: Missing index CSVs in {BASE_DIR}")
        sys.exit(1)

    df_hackathon = pd.read_csv(HACKATHON_INDEX_CSV)
    if "dataset_source" not in df_hackathon.columns:
        df_hackathon["dataset_source"] = "hackathon_offline"

    df_brats = pd.read_csv(BRATS_INDEX_CSV)
    if "dataset_source" not in df_brats.columns:
        df_brats["dataset_source"] = "brats_training"

    combined_df = pd.concat([df_hackathon, df_brats], ignore_index=True)
    total_files = len(combined_df)

    print(f"Total files in index: {total_files}")
    print(f"Parallel Execution enabled: {MAX_WORKERS} worker processes across CPU cores.")

    preprocessed_index_rows = []
    failures = []
    success_count = 0
    seg_mask_count = 0
    completed_count = 0
    start_time = time.time()

    print("\nStarting Parallel Resizing + MRI Preprocessing Pipeline on D: Drive...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_file, row): idx for idx, row in combined_df.iterrows()}

        for future in as_completed(futures):
            completed_count += 1
            is_success, rows, err, is_seg = future.result()

            if is_success:
                success_count += 1
                if is_seg:
                    seg_mask_count += 1
                preprocessed_index_rows.extend(rows)
            else:
                if err:
                    failures.append(err)

            if completed_count % 10 == 0 or completed_count == total_files:
                elapsed = time.time() - start_time
                rate = completed_count / elapsed if elapsed > 0 else 0
                print(f"Progress: Processed {completed_count}/{total_files} files ({rate:.2f} files/sec)...")

    # Save preprocessed_index.csv to D: drive
    df_index_out = pd.DataFrame(preprocessed_index_rows)
    
    # Try writing index safely to avoid Errno 13 permission lockouts
    try:
        df_index_out.to_csv(PREPROCESSED_INDEX_CSV, index=False)
        print(f"Saved index to: {PREPROCESSED_INDEX_CSV}")
    except PermissionError:
        fallback = PREPROCESSED_INDEX_CSV.parent / (PREPROCESSED_INDEX_CSV.stem + "_split.csv")
        df_index_out.to_csv(fallback, index=False)
        print(f"Warning: {PREPROCESSED_INDEX_CSV.name} was locked. Saved to fallback: {fallback}")

    total_time = time.time() - start_time
    print("\n" + "="*55)
    print(" MRI PREPROCESSING PIPELINE SUMMARY (STRICT D: DRIVE)")
    print("="*55)
    print(f"Total execution time:          {total_time / 60:.2f} minutes")
    print(f"Total files in index:           {total_files}")
    print(f"Successfully processed:         {success_count}")
    print(f"  - Segmentation masks resampled: {seg_mask_count}")
    print(f"  - MRI volumes enhanced:         {success_count - seg_mask_count}")
    print(f"Failed / unreadable files:       {len(failures)}")

    if failures:
        print("\n--- Failed Files ---")
        for path, err_msg in failures[:15]:
            print(f"  - {path}: {err_msg}")
        if len(failures) > 15:
            print(f"  ... and {len(failures) - 15} more")

    print("\nGenerated Output Folders on D: Drive:")
    print(f"  1. HE Root:    {OUTPUT_ROOT_HE.resolve()}")
    print(f"  2. AHE Root:   {OUTPUT_ROOT_AHE.resolve()}")
    print(f"  3. CLAHE Root: {OUTPUT_ROOT_CLAHE.resolve()}")
    print("="*55)


if __name__ == "__main__":
    process_pipeline()
