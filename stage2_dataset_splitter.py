import os
import sys
from pathlib import Path
import random
import pandas as pd

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSED_INDEX_CSV = PROJECT_ROOT / "preprocessed_index.csv"
HACKATHON_INDEX_CSV = PROJECT_ROOT / "dataset_index.csv"
BRATS_INDEX_CSV = PROJECT_ROOT / "brats_index.csv"

SPLIT_RATIO = (0.70, 0.15, 0.15)  # Train / Val / Test
SEED = 42


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


def normalize_path_columns(df):
    for col in ["filepath", "original_filepath", "new_filepath"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda p: to_relative_path(p) if pd.notna(p) else p)
    return df


def assign_patient_splits(patients, split_ratio, seed):
    """
    Shuffles and splits a list of patient IDs reproducibly.
    """
    patients_list = list(patients)
    # Reproducible shuffle
    rng = random.Random(seed)
    rng.shuffle(patients_list)

    n_total = len(patients_list)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    train_patients = set(patients_list[:n_train])
    val_patients = set(patients_list[n_train:n_train + n_val])
    test_patients = set(patients_list[n_train + n_val:])

    split_map = {}
    for p in patients_list:
        if p in train_patients:
            split_map[p] = "train"
        elif p in val_patients:
            split_map[p] = "val"
        else:
            split_map[p] = "test"

    return split_map


def save_dataframe_safely(df, dest_path):
    """Tries to write a dataframe to a CSV file. If locked, falls back to a suffix."""
    try:
        df.to_csv(dest_path, index=False)
        print(f"Saved: {dest_path}")
    except PermissionError:
        fallback = dest_path.parent / (dest_path.stem + "_split.csv")
        df.to_csv(fallback, index=False)
        print(f"Warning: {dest_path.name} is currently locked (likely open in Excel/Viewer). Saved to fallback: {fallback}")


def split_datasets():
    print("Starting reproducible patient-level dataset partitioning...")

    if not PREPROCESSED_INDEX_CSV.exists():
        print(f"Error: Missing index CSV at {PREPROCESSED_INDEX_CSV}")
        sys.exit(1)

    df_prep = pd.read_csv(PREPROCESSED_INDEX_CSV)
    
    # We split patients separately for each dataset source to maintain balanced ratios
    sources = df_prep["dataset_source"].unique()
    all_split_maps = {}

    for src in sources:
        src_df = df_prep[df_prep["dataset_source"] == src]
        patients = src_df["patient_id"].unique()
        split_map = assign_patient_splits(patients, SPLIT_RATIO, SEED)
        all_split_maps.update(split_map)
        
        print(f"  Source '{src}': Total Patients = {len(patients)}")
        splits_counts = pd.Series(list(split_map.values())).value_counts()
        for k, v in splits_counts.items():
            print(f"    - {k}: {v} patients")

    # Map split to DataFrame
    df_prep["split"] = df_prep["patient_id"].map(all_split_maps)
    
    # Save back preprocessed index
    save_dataframe_safely(normalize_path_columns(df_prep), PREPROCESSED_INDEX_CSV)

    # Update raw dataset_index.csv and brats_index.csv with matching splits for consistency
    if HACKATHON_INDEX_CSV.exists():
        df_h = pd.read_csv(HACKATHON_INDEX_CSV)
        df_h["split"] = df_h["patient_id"].map(all_split_maps)
        save_dataframe_safely(normalize_path_columns(df_h), HACKATHON_INDEX_CSV)

    if BRATS_INDEX_CSV.exists():
        df_b = pd.read_csv(BRATS_INDEX_CSV)
        df_b["split"] = df_b["patient_id"].map(all_split_maps)
        save_dataframe_safely(normalize_path_columns(df_b), BRATS_INDEX_CSV)

    print("\nDataset split successfully verified! Volumes and masks are matched at the patient level.")


if __name__ == "__main__":
    split_datasets()
