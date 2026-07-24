import csv
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

# Project root for relative path conversion.
PROJECT_ROOT = Path(__file__).resolve().parent


def is_absolute_project_path(value: Any) -> bool:
    """Return True if the value is an absolute filesystem path under the project root."""
    if not isinstance(value, str):
        return False

    try:
        path = Path(value)
    except Exception:
        return False

    return path.is_absolute() and PROJECT_ROOT in path.parents or path == PROJECT_ROOT


def convert_path_value(value: Any) -> Any:
    """Convert an absolute project-root path string to a relative POSIX-style path."""
    if not isinstance(value, str):
        return value

    path = Path(value)
    if path.is_absolute():
        try:
            relative_path = path.relative_to(PROJECT_ROOT)
            return relative_path.as_posix()
        except ValueError:
            # Path is absolute but not under project root, so leave unchanged.
            return value

    return value


def process_csv_file(csv_path: Path) -> None:
    """Read a CSV, convert absolute project-root paths to relative, and overwrite the file."""
    print(f"Processing CSV: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        print(f"  Skipping {csv_path.name}: {exc}")
        return
    except pd.errors.ParserError as exc:
        print(f"  Skipping {csv_path.name}: {exc}")
        return
    except Exception as exc:
        print(f"  Failed to read {csv_path.name}: {exc}")
        return

    columns_with_paths = []
    converted_count = 0

    for column in df.columns:
        if df[column].dtype == object:
            column_converted = 0
            new_values = []

            for cell_value in df[column].tolist():
                new_value = convert_path_value(cell_value)
                if new_value != cell_value:
                    column_converted += 1
                new_values.append(new_value)

            if column_converted > 0:
                columns_with_paths.append(column)
                converted_count += column_converted
                df[column] = new_values

    if converted_count > 0:
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=str(csv_path.parent),
                prefix=f"{csv_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as temp_handle:
                temp_file = Path(temp_handle.name)
                df.to_csv(temp_handle, index=False, quoting=csv.QUOTE_MINIMAL)

            if temp_file is None or not temp_file.exists():
                raise OSError(f"Temporary file for {csv_path.name} was not created successfully.")

            os.replace(temp_file, csv_path)
            print(f"  Converted {converted_count} path(s) in columns: {', '.join(columns_with_paths)}")
        except PermissionError:
            if temp_file is not None and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            print(
                f"Could not replace {csv_path.name} because it is currently in use. "
                "Please close any editor, Excel, or Python process using the file and run the script again."
            )
        except Exception as exc:
            if temp_file is not None and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            print(f"  Failed to update {csv_path.name}: {exc}")
    else:
        print("  No absolute project-root paths found; file left unchanged.")


def main() -> None:
    """Recursively search for CSV files and convert absolute project-root paths to relative."""
    csv_files = sorted(PROJECT_ROOT.rglob("*.csv"))

    if not csv_files:
        print("No CSV files found in the project.")
        return

    print(f"Found {len(csv_files)} CSV file(s) under {PROJECT_ROOT}")

    for csv_file in csv_files:
        process_csv_file(csv_file)


if __name__ == "__main__":
    main()
