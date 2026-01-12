import pandas as pd
from great_tables import GT, html
from nicegui import ui


def process_data_for_table(video_groups, video_scores, selected_prompt_type):
    # (Same data processing logic as before)
    raw_data = []

    for gid, types in video_groups.items():
        if selected_prompt_type not in types:
            continue

        for cfg in types[selected_prompt_type]:
            params = cfg.get("params", {})
            folder = cfg.get("_folder_name", "")

            for _, _, v_idx in cfg.get("_videos", []):
                raw_data.append({"gid": gid, "params": params, "score_key": f"{folder}/{v_idx}"})

    if not raw_data:
        return pd.DataFrame()

    df = pd.DataFrame(raw_data)
    df["score"] = df["score_key"].map(video_scores)

    df["Configuration"] = df["params"].apply(
        lambda p: "\n".join(
            f"{k.replace('_', ' ')}={v}" for k, v in sorted(p.items(), key=lambda item: item[0])
        )
        if p
        else "Default"
    )

    pivot_df = df.pivot_table(index="gid", columns="Configuration", values="score", aggfunc="mean")

    pivot_df = pivot_df.sort_index()
    pivot_df.index = "Prompt #" + (pivot_df.index + 1).astype(str)

    return pivot_df.reset_index(names="Prompt")


def render_table_tab(video_groups, video_scores):
    """
    Renders the Analysis/Table Tab with a Sidebar Layout.
    """
    all_types = sorted({p_type for types in video_groups.values() for p_type in types.keys()})

    state = {"p_type": all_types[0] if all_types else None}

    # --- UI COMPONENTS ---

    def set_type(p_type):
        state["p_type"] = p_type
        sidebar_menu.refresh()
        table_content.refresh()

    @ui.refreshable
    def sidebar_menu():
        ui.label("Prompt Types").classes("text-xs font-bold text-slate-500 uppercase mb-2")

        with ui.scroll_area().classes("w-full flex-grow"):
            with ui.column().classes("w-full gap-1"):
                for p_type in all_types:
                    is_active = p_type == state["p_type"]

                    bg_class = (
                        "bg-blue-500/10 text-blue-400"
                        if is_active
                        else "text-slate-400 hover:text-slate-200 hover:bg-slate-700/50"
                    )
                    indicator_color = "bg-blue-500" if is_active else "bg-slate-600"

                    with (
                        ui.row()
                        .classes(
                            f"w-full cursor-pointer items-center gap-2 px-3 py-1 rounded transition-colors {bg_class}"
                        )
                        .on("click", lambda _, pt=p_type: set_type(pt))
                    ):
                        ui.element("div").classes(
                            f"w-1.5 h-1.5 rounded-full {indicator_color} shrink-0"
                        )
                        ui.label(p_type).classes("text-sm font-medium truncate")

    @ui.refreshable
    def table_content():
        current_type = state["p_type"]

        if not current_type:
            ui.label("No data available.").classes("text-slate-400")
            return

        df = process_data_for_table(video_groups, video_scores, current_type)

        if df.empty:
            ui.label(f"No results found for {current_type}").classes("text-slate-400")
            return

        config_cols = [c for c in df.columns if c != "Prompt"]

        gt_tbl = (
            GT(df)
            .tab_header(
                title=f"Results: {current_type}", subtitle="Average scores per configuration"
            )
            .cols_align(align="center", columns=config_cols)
            .data_color(
                columns=config_cols,
                palette=["#ef4444", "#eab308", "#22c55e"],
                domain=[1, 5],
                na_color="#1e293b",
            )
            .fmt_number(columns=config_cols, decimals=1)
            .tab_options(
                table_background_color="#1e293b",
                table_font_color="#e2e8f0",
                heading_title_font_size="24px",
                table_border_top_color="#334155",
                table_border_bottom_color="#334155",
                heading_background_color="#0f172a",
                column_labels_background_color="#334155",
                # Compact Styling
                column_labels_font_weight="bold",
                column_labels_font_size="10px",
                column_labels_padding="4px",
                table_font_size="13px",
                data_row_padding="5px",
            )
            .cols_label({c: html(c.replace("\n", "<br>")) for c in config_cols})
        )

        with ui.column().classes("w-full items-center"):
            ui.html(gt_tbl.as_raw_html(), sanitize=False).classes(
                "w-fit max-w-full overflow-x-auto rounded-lg shadow-lg border border-slate-700 p-6 bg-slate-800"
            )

    # --- MAIN SPLIT LAYOUT ---

    # 1. Wrapper: w-full h-full, no padding
    with ui.row().classes("w-full h-full flex-nowrap items-start gap-0 overflow-hidden"):
        # 2. Sidebar: Matches Gallery View perfectly
        with ui.column().classes(
            "w-64 shrink-0 bg-slate-800 h-full p-4 border-r border-slate-700 flex flex-col"
        ):
            ui.label("Analysis").classes("text-xl font-bold text-blue-500 mb-6")

            ui.label("Select Matrix").classes("text-xs font-bold text-slate-500 uppercase mb-2")
            ui.separator().classes("mb-4 bg-slate-600")

            sidebar_menu()

        # 3. Content: Added 'p-8' here to replace the padding removed from main.py
        with ui.column().classes("grow h-full overflow-y-auto bg-slate-900 p-8 min-w-0"):
            table_content()
