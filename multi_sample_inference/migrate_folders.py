import argparse
import json
import sys
from pathlib import Path

# We try to import the naming function from utils.
# This requires this script to be in the same folder as utils.py
from .utils import get_folder_name


def main():
    parser = argparse.ArgumentParser(
        description="Migration script: Removes 'repeat' from config.json and renames folders."
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="The parent directory containing the generated video folders",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would happen without modifying files"
    )

    args = parser.parse_args()
    root_path = args.output_path

    if not root_path.exists():
        print(f"Error: Directory '{root_path}' does not exist.")
        sys.exit(1)

    print(f"Scanning {root_path}...")

    renamed_count = 0
    updated_only_count = 0
    skipped_count = 0
    errors = 0

    # Iterate over all items in the output path
    for folder in root_path.iterdir():
        if not folder.is_dir():
            continue

        config_path = folder / "config.json"

        # Skip folders that don't look like generation outputs
        if not config_path.exists():
            continue

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            print(f"[ERROR] Could not read JSON in {folder.name}: {e}")
            errors += 1
            continue

        # Check if 'repeat' exists in the config
        if "repeat" in config:
            # 1. Modify the config object in memory
            config.pop("repeat")

            # 2. Calculate the new name based on the cleaned config
            try:
                new_folder_name = get_folder_name(config)
            except Exception as e:
                print(f"[ERROR] Failed to generate folder name for {folder.name}: {e}")
                errors += 1
                continue

            new_folder_path = root_path / new_folder_name

            # Scenario A: The name doesn't change
            if new_folder_path == folder:
                print(f"[UPDATE ONLY] {folder.name} (Name matches new config)")
                if not args.dry_run:
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=2)
                updated_only_count += 1

            # Scenario B: The new name already exists on disk
            elif new_folder_path.exists():
                print(f"[SKIP] Target exists: {folder.name} -> {new_folder_name}")
                print("       (Manual intervention required to merge/delete)")
                skipped_count += 1

            # Scenario C: Rename and Update
            else:
                print(f"[RENAME] {folder.name} -> {new_folder_name}")

                if not args.dry_run:
                    # Update the config file inside the current folder FIRST
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=2)

                    # Rename the folder
                    try:
                        folder.rename(new_folder_path)
                        renamed_count += 1
                    except OSError as e:
                        print(f"       [OS ERROR] Could not rename: {e}")
                        errors += 1
                else:
                    renamed_count += 1

    print("-" * 30)
    print("Migration Complete.")
    if args.dry_run:
        print("(DRY RUN - No files changed)")
    print(f"Renamed folders: {renamed_count}")
    print(f"Config updated (no rename): {updated_only_count}")
    print(f"Skipped (collision): {skipped_count}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
