import os
import random
from pathlib import Path
import nibabel as nib
import pandas as pd

# --- Configuration ---
# Path to extracted BraTS 2020 dataset
BRATS_DIR = r"D:\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
OUTPUT_CSV = "brats_index.csv"

PROJECT_ROOT = Path(__file__).resolve().parent

# Resource-aware patient subsetting
# Set to an integer (e.g. 80) to limit processing for lightweight GPUs, or None to process all.
MAX_PATIENTS = 80
RANDOM_SEED = 42

NII_EXTS = (".nii", ".nii.gz")


def detect_brats_modality(filename):
    """
    Extract modality for BraTS 2020 naming convention:
    <patient_id>_t1ce.nii(.gz) -> T1c
    <patient_id>_t1.nii(.gz)   -> T1
    <patient_id>_t2.nii(.gz)   -> T2
    <patient_id>_flair.nii(.gz)-> FLAIR
    <patient_id>_seg.nii(.gz)  -> seg
    """
    f = filename.lower()
    if "_t1ce." in f or f.endswith("_t1ce.nii") or f.endswith("_t1ce.nii.gz"):
        return "T1c"
    elif "_t1." in f or f.endswith("_t1.nii") or f.endswith("_t1.nii.gz"):
        return "T1"
    elif "_t2." in f or f.endswith("_t2.nii") or f.endswith("_t2.nii.gz"):
        return "T2"
    elif "_flair." in f or f.endswith("_flair.nii") or f.endswith("_flair.nii.gz"):
        return "FLAIR"
    elif "_seg." in f or f.endswith("_seg.nii") or f.endswith("_seg.nii.gz"):
        return "seg"
    return "unmatched"


def to_relative_path(path):
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(os.path.relpath(p, PROJECT_ROOT)).as_posix()


def index_brats_dataset(root_dir, max_patients=None, seed=42):
    root = Path(root_dir)
    if not root.exists():
        print(f"Error: Directory not found - {root_dir}")
        return [], []

    # 1. Discover all patient folders
    patient_dirs = []
    for item in os.listdir(root):
        item_path = root / item
        if item_path.is_dir():
            # Verify folder contains at least one NIfTI file
            nii_files = [f for f in os.listdir(item_path) if f.lower().endswith(NII_EXTS)]
            if nii_files:
                patient_dirs.append(item_path)

    patient_dirs = sorted(patient_dirs, key=lambda p: p.name)
    total_patients_found = len(patient_dirs)

    # 2. Apply random subsetting if MAX_PATIENTS is specified
    if max_patients is not None and max_patients < total_patients_found:
        rng = random.Random(seed)
        selected_patient_dirs = sorted(rng.sample(patient_dirs, max_patients), key=lambda p: p.name)
        print(f"Patient Subsetting: Total found = {total_patients_found}, Selected = {len(selected_patient_dirs)} (Seed = {seed})")
    else:
        selected_patient_dirs = patient_dirs
        print(f"Processing all {total_patients_found} patient directories...")

    rows = []
    unmatched = []

    # 3. Process selected patient directories
    for pdir in selected_patient_dirs:
        patient_id = pdir.name
        
        for fname in os.listdir(pdir):
            if not fname.lower().endswith(NII_EXTS):
                continue

            full_path = pdir / fname
            modality = detect_brats_modality(fname)

            if modality == "unmatched":
                unmatched.append((str(full_path), "unrecognized_brats_filename"))
                continue

            try:
                img = nib.load(str(full_path))
                shape = tuple(img.shape)
                zooms = img.header.get_zooms()
                spacing = tuple(round(float(z), 3) for z in zooms) if zooms else None
            except Exception as e:
                unmatched.append((str(full_path), f"nibabel_load_error: {e}"))
                continue

            rows.append({
                "patient_id": patient_id,
                "organ": "brain",
                "modality": modality,
                "pathological": True,
                "dataset_source": "brats_training",
                "filepath": to_relative_path(full_path),
                "shape": shape,
                "voxel_spacing": spacing
            })

    return rows, unmatched, total_patients_found, len(selected_patient_dirs)


def main():
    print("Starting BraTS 2020 dataset indexing...")
    print(f"Dataset root: {BRATS_DIR}")

    rows, unmatched, total_found, total_selected = index_brats_dataset(
        BRATS_DIR, max_patients=MAX_PATIENTS, seed=RANDOM_SEED
    )

    if not rows:
        print("No valid BraTS files were processed.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "="*45)
    print(" BRATS 2020 DATASET INDEX SUMMARY")
    print("="*45)
    print(f"Total patient folders found:    {total_found}")
    print(f"Patient folders selected:       {total_selected} (MAX_PATIENTS = {MAX_PATIENTS})")
    print(f"Total NIfTI files indexed:       {len(df)}")
    print(f"Dataset Source identifier:       {df['dataset_source'].iloc[0]}")
    
    print("\n--- Files per Modality ---")
    print(df["modality"].value_counts().to_string())

    print("\n--- Sample Patient Shape & Spacing ---")
    sample_patient = df["patient_id"].iloc[0]
    sample_df = df[df["patient_id"] == sample_patient][["modality", "shape", "voxel_spacing"]]
    print(f"Patient: {sample_patient}")
    print(sample_df.to_string(index=False))

    if unmatched:
        print(f"\n--- Warning: {len(unmatched)} Unmatched / Corrupted Files ---")
        for path, reason in unmatched[:15]:
            print(f"  - [{reason}] {path}")
        if len(unmatched) > 15:
            print(f"  ... and {len(unmatched) - 15} more")
    else:
        print("\nUnmatched/Corrupted files: 0 (100% clean parse)")

    print(f"\nSuccessfully saved BraTS index to {OUTPUT_CSV}")
    print("="*45)


if __name__ == "__main__":
    main()
