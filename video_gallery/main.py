from nicegui import app, ui

import data_manager
import gallery_view
import table_view

# --- INITIALIZATION ---

app.add_static_files("/videos", data_manager.VIDEOS_PATH)

data_manager.load_scores()
video_groups = data_manager.discover_groups(data_manager.VIDEOS_PATH)

# --- APP LAYOUT ---

ui.query("body").classes("bg-slate-900 text-slate-200 p-0 m-0")
ui.query('.nicegui-content').classes('w-full h-screen p-0 m-0')

with ui.column().classes("h-screen w-full p-0 gap-0 overflow-hidden"):

    # --- HEADER ---
    with ui.row().classes("w-full bg-slate-900 border-b border-slate-700 px-8 py-0 items-end gap-6 shrink-0"):
        ui.label("Video Evaluation").classes("text-2xl font-bold text-slate-100 py-4 mr-4")

        with ui.tabs().classes("bg-transparent text-slate-400") \
            .props("indicator-color='blue-500' active-color='blue-400'") as main_tabs:
            tab_gallery = ui.tab("Gallery").props("icon=perm_media")
            tab_table = ui.tab("Score Matrix").props("icon=table_chart")

    # --- TAB CONTENT ---
    with ui.tab_panels(main_tabs, value=tab_gallery).classes("w-full flex-grow bg-slate-900 p-0 overflow-hidden"):

        # TAB 1: GALLERY
        with ui.tab_panel(tab_gallery).classes("p-0 w-full h-full overflow-hidden"):
            gallery_view.render_gallery_view(video_groups)

        # TAB 2: TABLE
        with ui.tab_panel(tab_table).classes("p-0 w-full h-full overflow-hidden"):
             table_view.render_table_tab(video_groups, data_manager.video_scores)

ui.run(title="Video Gallery", dark=True, show=False)
