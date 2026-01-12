import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict

from nicegui import ui

# --- CONFIGURATION ---
VIDEOS_PATH = Path("multi_sample_inference/output/")
SCORES_FILE = VIDEOS_PATH / "video_scores.json"

# --- STATE ---
video_scores: Dict[str, int] = {}


def load_scores():
    """Loads scores into the global video_scores dictionary."""
    global video_scores
    if SCORES_FILE.exists():
        try:
            with open(SCORES_FILE, "r") as f:
                video_scores.update(json.load(f))
        except Exception as e:
            print(f"Error loading scores: {e}")


def save_score(unique_key: str, score: int):
    """Updates state and persists to JSON."""
    video_scores[unique_key] = score
    try:
        with open(SCORES_FILE, "w") as f:
            json.dump(video_scores, f, indent=2)
    except Exception as e:
        ui.notify(f"Failed to save score: {e}", type="negative")


def get_group_id(target_data, registry):
    for index, existing_data in enumerate(registry):
        if target_data == existing_data:
            return index
    registry.append(target_data)
    return len(registry) - 1


def get_video_index(filename):
    match = re.search(r"video_(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0


def discover_groups(root_path: Path):
    registry = []
    groups = defaultdict(lambda: defaultdict(list))

    if not root_path.exists():
        print(f"Warning: Path '{root_path}' does not exist.")
        return groups

    for config_path in root_path.rglob("config.json"):
        try:
            with open(config_path) as f:
                cfg = json.load(f)

            p_data = cfg.get("prompt_data", {})
            p_type = cfg.get("prompt_type", "unknown")
            gid = get_group_id(p_data, registry)

            folder = config_path.parent
            folder_name = folder.name

            video_candidates = folder.glob("video_*.mp4")
            valid_videos = [f for f in video_candidates if re.match(r"^video_\d+\.mp4$", f.name)]
            video_files = sorted(valid_videos, key=lambda x: get_video_index(x.name))

            cfg["_folder_name"] = folder_name
            cfg["_gid"] = gid

            cfg["_videos"] = []
            for v in video_files:
                v_idx = get_video_index(v.name)
                url = f"/videos/{folder_name}/{v.name}"
                cfg["_videos"].append((url, v.name, v_idx))

            groups[gid][p_type].append(cfg)

        except (json.JSONDecodeError, KeyError, IOError):
            continue

    return groups


def generate_dataset_json(groups_data):
    records = []
    for gid, types in groups_data.items():
        for p_type, configs in types.items():
            for cfg in configs:
                folder_name = cfg.get("_folder_name", "")
                params = cfg.get("params", {})
                for url, filename, v_idx in cfg.get("_videos", []):
                    score_key = f"{folder_name}/{v_idx}"
                    score = video_scores.get(score_key, None)
                    record = {
                        "group_id": gid,
                        "prompt_type": p_type,
                        "folder_name": folder_name,
                        "video_file": filename,
                        "video_index": v_idx,
                        "score": score,
                    }
                    record.update(params)
                    records.append(record)
    return json.dumps(records, indent=2)
