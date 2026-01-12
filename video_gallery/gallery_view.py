import data_manager
from nicegui import ui

# --- UI HELPERS ---


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


def render_badge(k, v):
    with ui.element("div").classes(
        "bg-slate-800 border border-slate-600 rounded px-2 py-1 flex gap-1"
    ):
        ui.label(k).classes("text-[10px] text-slate-400")
        ui.label(str(v)).classes("text-[10px] text-blue-300 font-bold")


def render_video_card(config, folder_name):
    with (
        ui.card()
        .classes("p-0 rounded-lg overflow-hidden bg-slate-800 shadow-lg")
        .classes("w-full max-w-4xl mx-auto border border-slate-700")
        .classes("flex flex-col h-full")
    ):
        video_list = config.get("_videos", [])

        if video_list:
            state = {"slide": "0"}

            @ui.refreshable
            def render_controls():
                try:
                    idx = int(state["slide"])
                except (ValueError, TypeError):
                    idx = 0

                if idx >= len(video_list):
                    return

                url, filename, v_idx = video_list[idx]
                unique_key = f"{folder_name}/{v_idx}"
                current_score = data_manager.video_scores.get(unique_key, None)

                with ui.row().classes(
                    "w-full bg-slate-900 px-4 py-3 items-center justify-between border-t border-slate-700"
                ):
                    with ui.column().classes("gap-0"):
                        ui.label(f"Video {idx + 1} of {len(video_list)}").classes(
                            "text-xs text-blue-400 font-bold"
                        )
                        ui.label(filename).classes("text-[10px] text-slate-500 font-mono")

                    with ui.row().classes("items-center gap-2"):
                        ui.label("Score:").classes("text-xs text-slate-400 font-bold")
                        ui.toggle(
                            options=[1, 2, 3, 4, 5],
                            value=current_score,
                            on_change=lambda e, k=unique_key: data_manager.save_score(k, e.value),
                            clearable=True,
                        ).props(
                            'unelevated dense text-color="slate-300" color="slate-800" toggle-color="blue-600"'
                        ).classes(
                            "text-xs rounded overflow-hidden gap-px bg-slate-700/70 border border-slate-700/70"
                        )

            with (
                ui.carousel(value="0", on_value_change=lambda e: render_controls.refresh())
                .bind_value(state, "slide")
                .props('height="auto" control-color="blue-500" arrows navigation')
                .classes("w-full aspect-video bg-black")
            ):
                for i, (url, filename, v_idx) in enumerate(video_list):
                    with ui.carousel_slide(name=str(i)).classes("p-0"):
                        ui.video(url).props("controls muted").classes(
                            "w-full h-full object-contain"
                        )

            render_controls()

        else:
            ui.label("No videos found").classes("p-8 text-center text-slate-500")

        with ui.element("div").classes("p-4 bg-slate-900 w-full grow border-t border-slate-700"):
            with ui.row().classes("gap-2 flex-wrap"):
                params = config.get("params", {})
                for k, v in sorted(params.items()):
                    render_badge(k, v)
                render_badge("folder", folder_name)


# --- MAIN RENDERER ---


def render_gallery_view(video_groups):
    if not video_groups:
        ui.label("No data found in paths.").classes("text-xl text-red-400 text-center w-full")
        return

    # Main container (split layout)
    with ui.row().classes("w-full h-full flex-nowrap items-start gap-0 overflow-hidden"):
        # --- LEFT COLUMN: NAVIGATION ---
        with ui.column().classes(
            "w-64 shrink-0 bg-slate-800 h-full p-4 border-r border-slate-700 flex flex-col"
        ):
            ui.label("Dataset").classes("text-xl font-bold text-blue-500 mb-6")

            def on_download_click():
                json_str = data_manager.generate_dataset_json(video_groups)
                ui.download(json_str.encode("utf-8"), "scores.json")

            ui.button("Download Scores", icon="file_download", on_click=on_download_click).classes(
                "w-full mb-6 bg-blue-600 hover:bg-blue-500 shadow-lg"
            )
            ui.separator().classes("mb-4 bg-slate-600")

            ui.label("Gallery Index").classes("text-xs font-bold text-slate-500 uppercase mb-2")

            with ui.scroll_area().classes("w-full flex-grow"):
                for gid in sorted(video_groups.keys()):
                    render_sidebar_link(gid + 1, f"group-{gid}")

        # --- RIGHT COLUMN: CONTENT ---
        with ui.column().classes("grow h-full overflow-y-auto bg-slate-900 min-w-0"):
            for gid in sorted(video_groups.keys()):
                types_dict = video_groups[gid]
                with ui.element("div").props(f'id="group-{gid}"').classes("w-full mb-24"):
                    render_group_header(gid + 1)
                    with (
                        ui.element("div")
                        .classes("mx-8 mt-4 bg-slate-800 rounded-lg border border-slate-700")
                        .classes("overflow-hidden shadow-md")
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
                                        if (
                                            "action_prompts" in p_data
                                            and p_type in p_data["action_prompts"]
                                        ):
                                            segments = p_data["action_prompts"][p_type]["segments"]
                                            full_prompt = " ".join(segments)
                                        else:
                                            full_prompt = str(p_data)
                                    except Exception:
                                        full_prompt = "Prompt data unavailable"

                                    with ui.element("div").classes(
                                        "w-full p-6 bg-slate-900/30 border-b border-slate-700"
                                    ):
                                        ui.label("Active Prompt:").classes(
                                            "text-xs text-blue-400 font-bold uppercase"
                                        )
                                        ui.label(full_prompt).classes(
                                            "text-lg italic text-slate-300"
                                        )

                                    with ui.element("div").classes(
                                        "grid grid-cols-1 md:grid-cols-2 gap-8 p-8 xl:grid-cols-3"
                                    ):
                                        for cfg in configs:
                                            folder_name = cfg["_folder_name"]
                                            render_video_card(cfg, folder_name)
