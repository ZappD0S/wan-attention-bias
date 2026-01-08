import json
import re

from collections import defaultdict
from pathlib import Path
from typing import Dict

from nicegui import app, ui


# --- 1. CONFIGURATION & PERSISTENCE ---

VIDEOS_PATH = Path("multi_sample_inference/output/")
SCORES_FILE = VIDEOS_PATH / "video_scores.json"

app.add_static_files("/videos", VIDEOS_PATH)

# Global memory for scores
video_scores: Dict[str, int] = {}


def load_scores():
    global video_scores
    if SCORES_FILE.exists():
        try:
            with open(SCORES_FILE, "r") as f:
                video_scores = json.load(f)
        except Exception as e:
            print(f"Error loading scores: {e}")
            video_scores = {}


def save_score(unique_key: str, score: int):
    video_scores[unique_key] = score
    try:
        with open(SCORES_FILE, "w") as f:
            json.dump(video_scores, f, indent=2)
    except Exception as e:
        ui.notify(f"Failed to save score: {e}", type="negative")


# --- 2. DATA DISCOVERY ---

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


def discover_groups(root_path):
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


# --- 3. UI COMPONENTS ---

def render_sidebar_link(index, target_id):
    with ui.link(target=f"#{target_id}").classes(
        "w-full flex items-center gap-2 group no-underline mb-0.5 py-1"
    ):
        ui.element("div").classes("w-1.5 h-1.5 rounded-full bg-slate-500 shrink-0").classes(
            "group-hover:bg-blue-400 transition-colors"
        )
        ui.label(f"Prompt Group #{index}").classes(
            "text-sm text-slate-300 group-hover:text-blue-400 font-medium"
        )


def render_group_header(index):
    row_cls = (
        "w-full bg-slate-800 p-4 border-l-8 border-blue-500 "
        "items-center sticky top-0 shadow-md z-50 flex-nowrap"
    )
    with ui.row().classes(row_cls):
        ui.icon("auto_awesome_motion", color="blue-500").classes("text-3xl")
        ui.label(f"Prompt Group #{index}").classes("text-2xl font-bold")


def render_video_card(config, folder_name):
    """
    Renders the card.
    Uses ui.refreshable to update the control bar (outside the carousel)
    when the carousel slide changes.
    """
    with (
        ui.card()
        .classes("p-0 rounded-lg overflow-hidden bg-slate-800 shadow-lg")
        .classes("w-full max-w-4xl mx-auto border border-slate-700")
        .classes("flex flex-col h-full")
    ):
        video_list = config.get("_videos", [])

        if video_list:
            # 1. State: Track which slide is active.
            # We use a dictionary so we can mutate it inside callbacks easily.
            state = {'slide': '0'}

            # 2. Control Bar: Defined as refreshable so it can rebuild with new data
            @ui.refreshable
            def render_controls():
                # Get the current index from the state
                try:
                    idx = int(state['slide'])
                except (ValueError, TypeError):
                    idx = 0

                # Safety check
                if idx >= len(video_list):
                    return

                url, filename, v_idx = video_list[idx]
                unique_key = f"{folder_name}/{v_idx}"
                current_score = video_scores.get(unique_key, None)

                # Render the Bar
                with ui.row().classes(
                    "w-full bg-slate-900 px-4 py-3 items-center justify-between border-t border-slate-700"
                ):
                    # Left: File Info
                    with ui.column().classes("gap-0"):
                        ui.label(f"Video {idx + 1} of {len(video_list)}").classes("text-xs text-blue-400 font-bold")
                        ui.label(filename).classes("text-[10px] text-slate-500 font-mono")

                    # Right: Score Toggle
                    with ui.row().classes("items-center gap-2"):
                        ui.label("Score:").classes("text-xs text-slate-400 font-bold")

                        ui.toggle(
                            options=[1, 2, 3, 4, 5],
                            value=current_score,
                            on_change=lambda e, k=unique_key: save_score(k, e.value),
                            clearable=True
                        ).props(
                            'unelevated dense '
                            'text-color="slate-300" '  # Original text color
                            'color="slate-800" '       # Lighter background (was slate-900)
                            'toggle-color="blue-600"'  # Active Blue
                        ).classes(
                            'text-xs rounded overflow-hidden '
                            'gap-px '
                            'bg-slate-700/70 '
                            'border border-slate-700/70'
                        )
            # 3. Carousel: Contains ONLY videos
            # We bind value to state['slide'] and call refresh on change
            with (
                ui.carousel(
                    value='0',
                    on_value_change=lambda e: render_controls.refresh()
                )
                .bind_value(state, 'slide')
                .props('height="auto" control-color="blue-500" arrows navigation')
                .classes("w-full aspect-video bg-black") # Standard 16:9 Aspect Ratio
            ):
                for i, (url, filename, v_idx) in enumerate(video_list):
                    # Slide name must match the string version of the index ('0', '1'...)
                    with ui.carousel_slide(name=str(i)).classes("p-0"):
                         ui.video(url).props("controls muted").classes("w-full h-full object-contain")

            # 4. Render the initial control bar (outside carousel)
            render_controls()

        else:
            ui.label("No videos found").classes("p-8 text-center text-slate-500")

        # --- FOOTER: PARAMETERS ---
        with ui.element("div").classes("p-4 bg-slate-900 w-full grow border-t border-slate-700"):
            with ui.row().classes("gap-2 flex-wrap"):
                params = config.get("params", {})
                for k, v in sorted(params.items()):
                    render_badge(k, v)
                render_badge("folder", folder_name)


def render_badge(k, v):
    with ui.element("div").classes(
        "bg-slate-800 border border-slate-600 rounded px-2 py-1 flex gap-1"
    ):
        ui.label(k).classes("text-[10px] text-slate-400")
        ui.label(str(v)).classes("text-[10px] text-blue-300 font-bold")


# --- 4. MAIN LAYOUT ---

load_scores()
video_groups = discover_groups(VIDEOS_PATH)

ui.query("body").classes("bg-slate-900 text-slate-200 p-0")

with ui.left_drawer(value=True, fixed=True).classes("bg-slate-800 p-4 border-r border-slate-700"):
    ui.label("Gallery Index").classes("text-xl font-bold text-blue-500 mb-6")

    def on_download_click():
        json_str = generate_dataset_json(video_groups)
        ui.download(json_str.encode("utf-8"), "scores.json")

    ui.button("Download Scores", icon="file_download", on_click=on_download_click).classes(
        "w-full mb-6 bg-blue-600 hover:bg-blue-500 shadow-lg"
    )
    ui.separator().classes("mb-4 bg-slate-600")
    with ui.scroll_area().classes("h-full"):
        for gid in sorted(video_groups.keys()):
            render_sidebar_link(gid + 1, f"group-{gid}")

with ui.column().classes("p-8 w-full"):
    ui.label("Video Comparison Tool").classes(
        "text-4xl font-bold mb-10 text-center text-blue-500 w-full"
    )

    if not video_groups:
        ui.label("No data found in paths.").classes("text-xl text-red-400 text-center w-full")

    for gid in sorted(video_groups.keys()):
        types_dict = video_groups[gid]
        with ui.element("div").props(f'id="group-{gid}"').classes("w-full mb-24"):
            render_group_header(gid + 1)
            with (
                ui.element("div")
                .classes("w-full bg-slate-800 rounded-lg border border-slate-700")
                .classes("overflow-hidden mt-4 shadow-md")
            ):
                with ui.tabs().classes("w-full bg-slate-800 text-slate-400") as t:
                    first_tab = None
                    tab_list = []
                    for p_type, configs in sorted(types_dict.items(), key=lambda x: x[0]):
                        tab_obj = ui.tab(p_type)
                        if first_tab is None:
                            first_tab = tab_obj
                        tab_list.append((tab_obj, p_type, configs))
                ui.separator().classes("bg-slate-700")
                with (
                    ui.tab_panels(t, value=first_tab, animated=False)
                    .props("keep-alive")
                    .classes("w-full bg-transparent p-0")
                ):
                    for tab_obj, p_type, configs in tab_list:
                        with ui.tab_panel(tab_obj).classes("p-0"):
                            try:
                                p_data = configs[0]["prompt_data"]
                                segments = p_data["action_prompts"][p_type]["segments"]
                                full_prompt = " ".join(segments)
                            except Exception:
                                full_prompt = "Prompt data unavailable"

                            with ui.element("div").classes("w-full p-6 bg-slate-900/30 border-b border-slate-700"):
                                ui.label("Active Prompt:").classes("text-xs text-blue-400 font-bold uppercase")
                                ui.label(full_prompt).classes("text-lg italic text-slate-300")

                            with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 gap-8 p-8 xl:grid-cols-3"):
                                for cfg in configs:
                                    folder_name = cfg["_folder_name"]
                                    render_video_card(cfg, folder_name)

ui.run(title="Video Gallery", dark=True, show=False)
