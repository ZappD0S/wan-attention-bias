import json
from pathlib import Path
from nicegui import ui, app
from multi_sample_inference.utils import get_folder_name
from sklearn.model_selection import ParameterGrid

# tell explicitly NiceGUI to serve the video folder
# so the browser can access the files via URL.
VIDEOS_PATH = Path("multi_sample_inference/output/")
app.add_static_files("/videos", VIDEOS_PATH)

with open("multi_sample_inference/param_grid.json") as f:
    param_grid = json.load(f)

with open("image_prompt_generation/prompts_modified.json") as f:
    prompts_data_list = json.load(f)

ui.query("body").classes("bg-slate-900 text-slate-200 p-0")

# -- create the Sidebar --
# value=True means open by default. fixed=True means it stays while scrolling.
with ui.left_drawer(value=True, fixed=True).classes(
    "bg-slate-800 border-r border-slate-700 p-4"
) as sidebar:
    ui.label("Navigation").classes("text-xl font-bold text-blue-500 mb-4")
    ui.label("Configurations:").classes("text-xs text-slate-400 uppercase tracking-wider mb-2")

with ui.column().classes("p-8 w-full"):
    ui.label("Video Gallery").classes("text-5xl font-bold mb-10 text-center text-blue-500 w-full")

    for i, config in enumerate(ParameterGrid(param_grid)):
        params_str = ", ".join(f"{k}={v}" for k, v in config.items() if k not in ("bias_method",))
        group_name = f"{config['bias_method']}, {params_str}"

        section_id = f"section-{i}"

        # add link to sidebar
        with sidebar:
            # scroll to the specific section ID when clicked
            # classes: styling for the link (hover effects, etc)
            ui.link(group_name, target=f"#{section_id}").classes(
                "text-sm text-slate-300 hover:text-blue-400 no-underline mb-2 block truncate"
            ).props(f'title="{group_name}"')  # tooltip for long names

        # --- section header  ---
        # We add the .props(id=...) here so the link finds this spot
        with (
            ui.row()
            .props(f'id="{section_id}"')
            .classes(
                "w-full bg-slate-800 p-4 border-l-8 border-blue-500 mb-4 items-center sticky top-0 z-10 shadow-md"
            )
        ):
            ui.icon("category", color="blue-500").classes("text-3xl mr-3")
            ui.label(group_name).classes("text-2xl font-bold text-white")
            ui.label(f"({len(prompts_data_list)} videos)").classes(
                "text-slate-400 text-sm ml-2 mt-1"
            )

        # --- video grid ---
        with ui.element("div").classes(
            "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 p-6"
        ):
            for prompt_data in prompts_data_list:
                cur_config = config.copy()
                cur_config["prompt"] = prompt_data

                folder_name = get_folder_name(cur_config)

                # use the mounted static path '/videos/' instead of local path
                # this ensures the browser can actually load the file
                video_url = f"/videos/{folder_name}/video.mp4"

                with ui.card().classes(
                    "p-0 rounded-lg overflow-hidden bg-slate-800 shadow-lg hover:scale-105 transition-transform duration-300 border border-slate-700"
                ):
                    ui.video(video_url).props("controls muted").classes(
                        "w-full h-auto aspect-video"
                    )

                    caption = " ".join(prompt_data["action_prompt"]["segments"])
                    with ui.element("div").classes("p-4 h-full bg-slate-900"):
                        ui.label(caption).classes("text-sm text-slate-400 font-mono leading-tight")

        # just a spacer
        ui.element("div").classes("h-12")

ui.run(title="Video Gallery", dark=True, show=False)
