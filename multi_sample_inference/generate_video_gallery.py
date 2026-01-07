import json
import re

from collections import defaultdict
from pathlib import Path

from nicegui import app, ui


# --- 1. CONFIGURATION ---
VIDEOS_PATH = Path("multi_sample_inference/output/")
app.add_static_files("/videos", VIDEOS_PATH)


def get_group_id(target_data, registry):
    """Matches a dict against a list to return a stable integer ID."""
    for index, existing_data in enumerate(registry):
        if target_data == existing_data:
            return index

    registry.append(target_data)
    return len(registry) - 1


def get_video_index(path):
    """Safely extracts the integer index from filenames like video_1.mp4."""
    match = re.search(r"video_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return 0


def discover_groups(root_path):
    """Groups folders by prompt_data (ID) and prompt_type (Tabs)."""
    registry = []
    groups = defaultdict(lambda: defaultdict(list))

    for config_path in root_path.rglob("config.json"):
        try:
            with open(config_path) as f:
                cfg = json.load(f)

            # Metadata for organization
            p_data = cfg["prompt_data"]
            p_type = cfg["prompt_type"]

            gid = get_group_id(p_data, registry)

            folder = config_path.parent
            video_candidates = folder.glob("video_*.mp4")
            valid_videos = [f for f in video_candidates if re.match(r"^video_\d+\.mp4$", f.name)]
            video_files = sorted(valid_videos, key=get_video_index)

            cfg["_folder_name"] = config_path.parent.name
            cfg["_videos"] = [f"/videos/{folder.name}/{v.name}" for v in video_files]

            groups[gid][p_type].append(cfg)
        except (json.JSONDecodeError, KeyError, IOError):
            continue
    return groups


# --- 2. UI COMPONENTS ---


def render_sidebar_link(index, target_id):
    with ui.link(target=f"#{target_id}").classes(
        "w-full flex items-center gap-2 group no-underline mb-0.5 py-1"
    ):
        # The Bullet: A generic div shaped into a circle
        ui.element("div").classes("w-1.5 h-1.5 rounded-full bg-slate-500 shrink-0").classes(
            "group-hover:bg-blue-400 transition-colors"
        )

        # The Text
        ui.label(f"Prompt Group #{index}").classes(
            "text-sm text-slate-300 group-hover:text-blue-400 font-medium"
        )


def render_group_header(index, section_id):
    row_cls = (
        "w-full bg-slate-800 p-4 border-l-8 border-blue-500 "
        "items-center sticky top-0 shadow-md z-50 flex-nowrap"
    )
    with ui.row().props(f'id="{section_id}"').classes(row_cls):
        ui.icon("auto_awesome_motion", color="blue-500").classes("text-3xl")
        ui.label(f"Prompt Group #{index}").classes("text-2xl font-bold")


def render_video_card(config, folder_name):
    with (
        ui.card()
        .classes("p-0 rounded-lg overflow-hidden bg-slate-800 shadow-lg")
        .classes("w-full max-w-4xl mx-auto border border-slate-700")
        .classes("flex flex-col h-full")
    ):
        video_urls = config.get("_videos", [])

        if video_urls:
            with (
                ui.carousel(animated=True, arrows=True, navigation=True)
                .props('height="auto" control-color="blue-500"')
                .classes("w-full aspect-video bg-black")
            ):
                for url in video_urls:
                    with ui.carousel_slide().classes("p-0"):
                        ui.video(url).props("controls muted").classes("w-full")
        else:
            ui.label("No videos").classes("p-8 text-center text-slate-500")

        with ui.element("div").classes("p-4 bg-slate-900 w-full grow"):
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


# --- 3. MAIN PAGE LAYOUT ---

ui.query("body").classes("bg-slate-900 text-slate-200 p-0")
video_groups = discover_groups(VIDEOS_PATH)

# Sidebar
with ui.left_drawer(value=True, fixed=True).classes("bg-slate-800 p-4"):
    ui.label("Gallery Index").classes("text-xl font-bold text-blue-500 mb-4")
    for gid in sorted(video_groups.keys()):
        render_sidebar_link(gid + 1, f"group-{gid}")

# Main Content
with ui.column().classes("p-8 w-full"):
    ui.label("Video Comparison Tool").classes(
        "text-4xl font-bold mb-10 text-center text-blue-500 w-full"
    )

    for gid in sorted(video_groups.keys()):
        types_dict = video_groups[gid]

        with ui.element("div").classes("w-full mb-24"):
            render_group_header(gid + 1, f"group-{gid}")
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
                            p_data = configs[0]["prompt_data"]
                            a_prompts = p_data["action_prompts"]
                            segments = a_prompts[p_type]["segments"]
                            full_prompt = " ".join(segments)
                            # folder_name = configs.get("_folder_name", "Unknown")

                            with (
                                ui.element("div")
                                .classes("w-full p-6 bg-slate-900/30 border-b")
                                .classes("border-slate-700")
                            ):
                                ui.label("Active Prompt:").classes(
                                    "text-xs text-blue-400 font-bold uppercase"
                                )
                                ui.label(full_prompt).classes("text-lg italic text-slate-300")

                            with (
                                ui.element("div")
                                .classes("grid grid-cols-1 md:grid-cols-2 gap-8 p-8")
                                .classes("xl:grid-cols-3")
                            ):
                                for cfg in configs:
                                    folder_name = cfg["_folder_name"]
                                    render_video_card(cfg, folder_name)

ui.run(title="Video Gallery", dark=True, show=False)
