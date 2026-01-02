import argparse
import json
import shutil
import sys
from pathlib import Path

# Try to import the naming function from utils.
from .utils import get_folder_name


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def merge_directories(src_dir: Path, dst_dir: Path, dry_run: bool = False) -> bool:
    """
    Merges src_dir into dst_dir.
    1. Checks if config.json matches.
    2. Moves files from src to dst.
    3. If collision, keeps the latest file (mtime).
    4. Removes src_dir if empty.
    Returns True if merge happened, False if aborted (mismatched config).
    """

    # 1. Compare Configs
    src_config_path = src_dir / "config.json"
    dst_config_path = dst_dir / "config.json"

    if not src_config_path.exists() or not dst_config_path.exists():
        print("       [MERGE ERROR] Missing config.json in one of the folders. Skipping.")
        return False

    try:
        src_conf = load_json(src_config_path)
        dst_conf = load_json(dst_config_path)
    except Exception as e:
        print(f"       [MERGE ERROR] JSON Error: {e}")
        return False

    # Normalize by converting back to string to ignore formatting diffs,
    # but strictly compare dictionaries
    if src_conf != dst_conf:
        print("       [MERGE CONFLICT] Configs do not match. Cannot merge.")
        print(f"       Source: {src_dir.name}")
        print(f"       Target: {dst_dir.name}")
        return False

    print(f"       [MERGE] Configs match. Merging '{src_dir.name}' into '{dst_dir.name}'...")

    # 2. Iterate and Move/Merge Files
    # We iterate over the source directory
    for src_file in src_dir.iterdir():
        dst_file = dst_dir / src_file.name

        if src_file.is_dir():
            print(f"       [SKIP SUBDIR] Cannot merge subdirectories: {src_file.name}")
            continue

        if not dst_file.exists():
            # No collision, just move
            if not dry_run:
                shutil.move(str(src_file), str(dst_file))
            print(f"           Moved: {src_file.name}")
        else:
            # Collision: Check timestamps
            src_mtime = src_file.stat().st_mtime
            dst_mtime = dst_file.stat().st_mtime

            if src_mtime > dst_mtime:
                # Source is newer, overwrite destination
                print(f"           Overwrite (Newer): {src_file.name}")
                if not dry_run:
                    dst_file.unlink()  # Remove old dest
                    shutil.move(str(src_file), str(dst_file))
            else:
                # Destination is newer or equal, delete source
                print(f"           Discard (Older/Eq): {src_file.name}")
                if not dry_run:
                    src_file.unlink()

    # 3. Clean up source directory
    if not dry_run:
        try:
            src_dir.rmdir()
            print(f"       [CLEANUP] Removed empty source folder: {src_dir.name}")
        except OSError:
            print(f"       [WARNING] Could not remove {src_dir.name} (not empty?).")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify folder names against config.json, rename or merge if necessary."
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="The parent directory containing the generated video folders",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print actions without modifying files"
    )

    args = parser.parse_args()
    root_path = args.output_path

    if not root_path.exists():
        print(f"Error: Directory '{root_path}' does not exist.")
        sys.exit(1)

    print(f"Scanning {root_path}...")

    stats = {"verified": 0, "renamed": 0, "merged": 0, "errors": 0, "skipped": 0}

    # Iterate over all directories
    # We list() it to avoid modifying the iterator while looping if we rename things
    folders = [f for f in root_path.iterdir() if f.is_dir()]

    for folder in folders:
        config_path = folder / "config.json"

        if not config_path.exists():
            continue

        try:
            config = load_json(config_path)
            # Generate the strictly correct name based on current logic
            expected_name = get_folder_name(config)
        except Exception as e:
            print(f"[ERROR] processing {folder.name}: {e}")
            stats["errors"] += 1
            continue

        expected_folder = root_path / expected_name

        # Case 1: Name is already correct
        if folder.name == expected_name:
            # verbose check: print(f"[OK] {folder.name}")
            stats["verified"] += 1
            continue

        # Case 2: Mismatch
        print(f"[MISMATCH] Found: {folder.name}")
        print(f"           Expect: {expected_name}")

        # Check if target already exists
        if not expected_folder.exists():
            # Simple Rename
            print(f"       Action: RENAME -> {expected_name}")
            if not args.dry_run:
                try:
                    folder.rename(expected_folder)
                    stats["renamed"] += 1
                except OSError as e:
                    print(f"       [OS ERROR] Rename failed: {e}")
                    stats["errors"] += 1
            else:
                stats["renamed"] += 1

        else:
            # Target exists - MERGE required
            print("       Action: MERGE into existing folder")
            success = merge_directories(folder, expected_folder, args.dry_run)
            if success:
                stats["merged"] += 1
            else:
                stats["skipped"] += 1

    print("-" * 30)
    print("Verification Complete.")
    if args.dry_run:
        print("(DRY RUN MODE)")
    print(f"Verified (No change): {stats['verified']}")
    print(f"Renamed:              {stats['renamed']}")
    print(f"Merged:               {stats['merged']}")
    print(f"Skipped (Conflict):   {stats['skipped']}")
    print(f"Errors:               {stats['errors']}")


if __name__ == "__main__":
    main()
