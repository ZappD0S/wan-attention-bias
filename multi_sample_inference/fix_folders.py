import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Try to import the specific hashing function from your codebase
from .utils import get_folder_name


def main():
    parser = argparse.ArgumentParser(
        description="Fix folder names by removing 'repeat' from config hash."
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="The folder containing the generated results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without moving files",
    )
    args = parser.parse_args()

    root_path = args.output_path
    if not root_path.exists():
        print(f"Path {root_path} does not exist.")
        sys.exit(1)

    # 1. Map Target Folder -> List of Source Folders
    merge_plan = defaultdict(list)
    target_configs = {}

    subfolders = [f for f in root_path.iterdir() if f.is_dir()]
    print(f"Found {len(subfolders)} folders to analyze...")

    for folder in subfolders:
        config_path = folder / "config.json"

        if not config_path.exists():
            continue

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping corrupted config: {folder}")
            continue

        # Create a deep copy for the "clean" config
        clean_config = json.loads(json.dumps(config))

        # Remove 'repeat' from the clean config if it exists inside params
        if "repeat" in clean_config.get("params", {}):
            clean_config["params"].pop("repeat")

        # Calculate what the folder name SHOULD be
        new_folder_name = get_folder_name(clean_config)
        target_path = root_path / new_folder_name

        merge_plan[target_path].append(folder)
        target_configs[target_path] = clean_config

    # 2. Execute the Merge
    print(f"\nIdentified {len(merge_plan)} unique target configurations.")

    for target_path, source_folders in merge_plan.items():
        # Sort source folders to maintain deterministic order
        source_folders.sort(key=lambda p: p.name)

        if not args.dry_run:
            target_path.mkdir(exist_ok=True)
            with open(target_path / "config.json", "w") as f:
                json.dump(target_configs[target_path], f, indent=2)

        # Determine the starting index for new videos
        current_idx = 0
        if target_path.exists():
            existing_videos = list(target_path.glob("video_*.mp4"))
            indices = []
            for v in existing_videos:
                # Expecting format video_{N}.mp4
                if "with_masks" in v.name:
                    continue
                try:
                    idx = int(v.stem.split("_")[-1])
                    indices.append(idx)
                except ValueError:
                    pass
            if indices:
                current_idx = max(indices) + 1

        print(f"\nProcessing Target: {target_path.name}")

        for source in source_folders:
            is_same_folder = source == target_path

            # Find all video files in the source
            source_videos = sorted(list(source.glob("video_*.mp4")))

            if not source_videos and not is_same_folder:
                # If source folder is empty and it's not the target, delete it
                if args.dry_run:
                    print(f"  [Dry Run] Would delete empty source: {source.name}")
                else:
                    shutil.rmtree(source)
                continue

            for vid_file in source_videos:
                if "with_masks" in vid_file.name:
                    continue

                # Identify the mask file associated with this video
                old_idx_str = vid_file.stem.split("_")[-1]  # video_0 -> 0
                mask_file = source / f"video_with_masks_{old_idx_str}.mp4"

                new_video_name = f"video_{current_idx}.mp4"
                new_mask_name = f"video_with_masks_{current_idx}.mp4"

                if is_same_folder:
                    # File is already in the correct folder; skip moving.
                    pass
                else:
                    if args.dry_run:
                        print(
                            f"  [Dry Run] Move {source.name}/{vid_file.name} -> {target_path.name}/{new_video_name}"
                        )
                    else:
                        shutil.move(str(vid_file), str(target_path / new_video_name))
                        if mask_file.exists():
                            shutil.move(str(mask_file), str(target_path / new_mask_name))

                # Increment index only if we are moving files in from outside.
                # Files already inside (is_same_folder) were accounted for in the initial current_idx calculation.
                if not is_same_folder:
                    current_idx += 1

            # Clean up source folder if it isn't the target
            if not is_same_folder:
                if args.dry_run:
                    print(f"  [Dry Run] Would remove source folder: {source.name}")
                else:
                    shutil.rmtree(source)

        if is_same_folder and len(source_folders) > 1:
            print("  (Note: Merged other folders into this existing one)")


if __name__ == "__main__":
    main()
