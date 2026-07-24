import os
from pathlib import Path
import nibabel as nib
import pandas as pd

# --- Configuration ---
# Update these paths to point to your actual dataset locations
BRAIN_MRI_DIR = r"D:\shivamogga_hackathon\Brain_JN_1\Brain DATASETS"
SPINE_MRI_DIR = r"D:\shivamogga_hackathon\Spine DATASETS\Spine DATASETS"
OUTPUT_CSV = "dataset_index.csv"
PROJECT_ROOT = Path(__file__).resolve().parent

NII_EXTS = (".nii", ".nii.gz")


def get_pathological_status(path_parts):
    """Search every folder name in the path for normal/pathological keywords."""
    for part in path_parts:
        p = part.lower()
        if any(k in p for k in ["patholog", "abnormal", "tumor", "hgg", "lgg"]):
            return True
        if any(k in p for k in ["normal", "healthy"]):
            return False
    return "unknown"


def get_patient_id(path_parts):
    """
    Patient ID = folder name immediately AFTER the category folder (containing normal/pathological).
    Falls back to the direct parent folder if no category keyword is found.
    """
    for i, part in enumerate(path_parts):
        p = part.lower()
        if "normal" in p or "patholog" in p:
            if i + 1 < len(path_parts) - 1:  # -1 because last part is the filename
                return path_parts[i + 1]
    # Fallback to direct parent folder of the file
    return path_parts[-2] if len(path_parts) >= 2 else "unknown_patient"


def detect_modality(filename):
    """
    Extract modality based on filename keywords.
    Order matters: check specific/contrast tags before generic T1/T2 tags.
    """
    f = filename.lower()
    
    # Check contrast / post-contrast T1 first -> T1c
    if any(k in f for k in ["t1ce", "t1c", "gado", "post", "contrast"]):
        return "T1c"
    # Check FLAIR
    if "flair" in f:
        return "FLAIR"
    # Check STIR / Fat-suppressed sequences (STIR, SPAIR, SPIR)
    if any(k in f for k in ["stir", "spair", "spir"]):
        return "STIR"
    # Check Segmentation masks
    if any(k in f for k in ["seg", "label", "mask"]):
        return "seg"
    # Check generic T2 & T1
    if "t2" in f:
        return "T2"
    if "t1" in f:
        return "T1"
        
    return "unmatched"


def to_relative_path(path):
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(os.path.relpath(p, PROJECT_ROOT)).as_posix()


def index_dataset(root_dir, organ):
    rows = []
    unmatched = []
    root = Path(root_dir)

    if not root.exists():
        print(f"Warning: Directory not found - {root_dir}")
        return rows, unmatched

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith(NII_EXTS):
                continue

            full_path = Path(dirpath) / fname
            path_parts = full_path.parts

            modality = detect_modality(fname)
            pathological = get_pathological_status(path_parts)
            patient_id = get_patient_id(path_parts)

            try:
                img = nib.load(str(full_path))
                shape = tuple(img.shape)
                # Cast numpy float values to clean Python floats rounded to 3 decimals
                zooms = img.header.get_zooms()
                spacing = tuple(round(float(z), 3) for z in zooms) if zooms else None
            except Exception as e:
                unmatched.append((str(full_path), f"load_error: {e}"))
                continue

            if modality == "unmatched":
                unmatched.append((str(full_path), "modality_not_recognized"))
                continue

            rows.append({
                "patient_id": patient_id,
                "organ": organ,
                "modality": modality,
                "pathological": pathological,
                "filepath": to_relative_path(full_path),
                "shape": shape,
                "voxel_spacing": spacing,
            })

    return rows, unmatched


def main():
    print("Starting dataset indexing (v2)...")
    all_rows = []
    all_unmatched = []

    print(f"Processing Brain MRI dataset at {BRAIN_MRI_DIR}...")
    rows, unmatched = index_dataset(BRAIN_MRI_DIR, "brain")
    all_rows.extend(rows)
    all_unmatched.extend(unmatched)

    print(f"Processing Spine MRI dataset at {SPINE_MRI_DIR}...")
    rows, unmatched = index_dataset(SPINE_MRI_DIR, "spine")
    all_rows.extend(rows)
    all_unmatched.extend(unmatched)

    if not all_rows:
        print("No valid files found in the specified directories. Please check the paths.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "="*40)
    print(" DATASET INDEX SUMMARY (v2)")
    print("="*40)
    print(f"Total files indexed: {len(df)}")
    
    print("\n--- Patients per Organ ---")
    print(df.groupby("organ")["patient_id"].nunique())
    
    print("\n--- Files per Modality ---")
    print(df["modality"].value_counts())
    
    print("\n--- Pathological Status Counts ---")
    print(df["pathological"].value_counts())

    if all_unmatched:
        print(f"\n--- Warning: {len(all_unmatched)} Unmatched/Corrupted Files ---")
        for path, reason in all_unmatched[:15]:
            print(f"  - [{reason}] {path}")
        if len(all_unmatched) > 15:
            print(f"  ... and {len(all_unmatched) - 15} more")

    print(f"\nSuccessfully saved dataset index to {OUTPUT_CSV}")
    print("="*40)


if __name__ == "__main__":
    main()
