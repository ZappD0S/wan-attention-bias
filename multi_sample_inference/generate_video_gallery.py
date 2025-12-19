import json
from pathlib import Path
from nicegui import ui, app
from multi_sample_inference.utils import get_folder_name
from sklearn.model_selection import ParameterGrid

# --- 1. CONFIGURATION & DATA LOADING ---
VIDEOS_PATH = Path("multi_sample_inference/output/")
app.add_static_files("/videos", VIDEOS_PATH)

with open("multi_sample_inference/param_grid.json") as f:
    param_grid = json.load(f)

with open("image_prompt_generation/prompts_modified.json") as f:
    prompts_data_list = json.load(f)

# --- 2. GLOBAL STYLES ---
ui.query("body").classes("bg-slate-900 text-slate-200 p-0")

# --- 3. SIDEBAR NAVIGATION ---
with ui.left_drawer(value=True, fixed=True).classes(
    "bg-slate-800 border-r border-slate-700 p-4"
) as sidebar:
    ui.label("Navigation").classes("text-xl font-bold text-blue-500 mb-4")
    ui.label("Prompts:").classes("text-xs text-slate-400 uppercase tracking-wider mb-2")

# --- 4. MAIN GALLERY ---
with ui.column().classes("p-8 w-full"):
    ui.label("Video Gallery").classes("text-5xl font-bold mb-10 text-center text-blue-500 w-full")

    for i, prompt_data in enumerate(prompts_data_list):
        prompt_text = " ".join(prompt_data["action_prompt"]["segments"])
        section_id = f"prompt-{i}"

        # A. Sidebar Link (Truncates automatically via CSS)
        with sidebar:
            ui.link(f"{i + 1}. {prompt_text}", target=f"#{section_id}").classes(
                "text-sm text-slate-300 hover:text-blue-400 no-underline mb-2 block truncate w-full"
            ).props(f'title="{prompt_text}"')

        # B. Prompt Section Wrapper
        # Wraps Header + Grid. When this scrolls off, it takes the sticky header with it.
        with ui.element("div").classes("w-full mb-24"):
            # --- Sticky Header ---
            with (
                ui.row()
                .props(f'id="{section_id}"')
                .classes(
                    "w-full bg-slate-800 p-4 border-l-8 border-blue-500 items-center sticky top-0 shadow-md z-50 flex-nowrap"
                )
            ):
                # Icon (shrink-0 prevents squishing)
                ui.icon("movie_filter", color="blue-500").classes("text-3xl mr-3 shrink-0")

                # Text (min-w-0 forces text wrapping inside flex container)
                with ui.column().classes("gap-0 grow min-w-0"):
                    ui.label(f"Prompt #{i + 1}").classes(
                        "text-xs text-blue-400 font-bold uppercase"
                    )
                    ui.label(prompt_text).classes(
                        "text-xl font-bold text-white leading-tight break-words"
                    )

            # --- Video Grid ---
            with ui.element("div").classes(
                "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 p-6"
            ):
                for config in ParameterGrid(param_grid):
                    # Prepare Config
                    cur_config = config.copy()
                    cur_config["prompt"] = prompt_data
                    folder_name = get_folder_name(cur_config)
                    video_url = f"/videos/{folder_name}/video.mp4"

                    # Video Card
                    with ui.card().classes(
                        "p-0 rounded-lg overflow-hidden bg-slate-800 shadow-lg hover:scale-105 transition-transform duration-300 border border-slate-700"
                    ):
                        ui.video(video_url).props("controls muted").classes(
                            "w-full h-auto aspect-video"
                        )

                        # Configuration Badges
                        with ui.element("div").classes(
                            "w-full p-3 bg-slate-900 border-t border-slate-700 h-full"
                        ):
                            ui.label("Configuration:").classes(
                                "text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-2"
                            )

                            with ui.row().classes("gap-2 w-full wrap"):
                                for k, v in sorted(config.items()):
                                    with ui.element("div").classes(
                                        "bg-slate-800 border border-slate-600 rounded px-2 py-1 flex items-center gap-1"
                                    ):
                                        ui.label(k).classes("text-xs text-slate-400")
                                        ui.label("=").classes("text-xs text-slate-600")
                                        ui.label(str(v)).classes(
                                            "text-xs text-blue-300 font-bold font-mono"
                                        )

ui.run(title="Video Gallery", dark=True, show=False)
