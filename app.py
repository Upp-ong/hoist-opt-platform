from __future__ import annotations

import hashlib
import html
import math
import time
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components

from hoisting.charts import (
    build_component_mix,
    build_convergence_figure,
    build_cost_figure,
    build_gantt_figure,
    build_pareto_figure,
    build_plan_radar,
    build_risk_breakdown_figure,
    build_risk_figure,
)
from hoisting.domain import (
    Component,
    STATION_PRESETS,
    TYPE_SPECS,
    build_station,
    components_to_frame,
    validate_components,
)
from hoisting.importer import (
    build_excel_template,
    normalize_project_metadata,
    read_import_file,
    template_frame,
    validate_import_bundle,
    validate_import_frame,
)
from hoisting.optimizer import OptimizationResult, optimize_hoisting_sequence
from hoisting.expert_ui import (
    render_algorithm_audit,
    render_expert_metrics,
    render_mode_comparison,
    render_objective_model,
    render_pareto_pages,
    render_risk_analysis,
    render_sensitivity_analysis,
)
from hoisting.presentation import (
    APP_CSS,
    decision_summary,
    hero_html,
    highlight_high_risk_rows,
    plan_tiles_html,
    schedule_frame,
    sidebar_brand_html,
    sidebar_readout,
)
from hoisting.projects import (
    delete_project,
    list_saved_projects,
    load_project,
    save_project,
)
from hoisting.three_viewer import build_three_viewer_html


st.set_page_config(
    page_title="装配式地铁车站吊装调度与可视化决策平台",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(APP_CSS, unsafe_allow_html=True)


def build_cranes(
    station_name: str,
    crane_count: int,
    base_capacity_t: float,
    length_m: float | None = None,
    height_m: float | None = None,
    width_m: float | None = None,
    components: Sequence[Component] | None = None,
) -> list[dict]:
    preset = STATION_PRESETS[station_name]
    if components:
        minimum_x = min(
            component.x - component.length_m * 0.5 for component in components
        )
        maximum_x = max(
            component.x + component.length_m * 0.5 for component in components
        )
        minimum_y = min(
            component.y - component.width_m * 0.5 for component in components
        )
        maximum_y = max(
            component.y + component.width_m * 0.5 for component in components
        )
        maximum_z = max(
            component.z + component.height_m * 0.5 for component in components
        )
        if length_m is not None:
            maximum_x = minimum_x + float(length_m)
        if width_m is not None:
            maximum_y = minimum_y + float(width_m)
        if height_m is not None:
            maximum_z = max(maximum_z, float(height_m))
    else:
        minimum_x = 0.0
        minimum_y = 0.0
        length_m = float(length_m if length_m is not None else preset["length_m"])
        width_m = float(width_m if width_m is not None else preset["width_m"])
        maximum_x = minimum_x + length_m
        maximum_y = minimum_y + width_m
        maximum_z = float(height_m if height_m is not None else preset["height_m"])
    rail_start_x = minimum_x - 8.0
    rail_end_x = maximum_x + 8.0
    trolley_min_y = minimum_y - 1.5
    trolley_max_y = maximum_y + 1.5
    rail_length = rail_end_x - rail_start_x
    base_capacity_t = float(base_capacity_t)
    capacity_factors = [1.0, 0.84, 0.68]
    return [
        {
            "id": f"GM-{index + 1:02d}",
            "type": "龙门吊",
            "capacity_t": round(base_capacity_t * capacity_factors[index], 1),
            "rate_yuan_h": round(
                260.0 + base_capacity_t * capacity_factors[index] * 1.95,
                0,
            ),
            "position": (
                rail_start_x + rail_length * (index + 1) / (crane_count + 1),
                (trolley_min_y + trolley_max_y) * 0.5,
            ),
            "initial_x": rail_start_x
            + rail_length * (index + 1) / (crane_count + 1),
            "initial_y": (trolley_min_y + trolley_max_y) * 0.5,
            "rail_start_x": rail_start_x,
            "rail_end_x": rail_end_x,
            "trolley_min_y": trolley_min_y,
            "trolley_max_y": trolley_max_y,
            "max_lift_height": round(maximum_z + 6.0, 2),
            "long_speed_m_min": 20.0,
            "trolley_speed_m_min": 12.0,
            "hoist_speed_m_min": 8.0,
            "minimum_spacing_m": 9.0,
        }
        for index in range(crane_count)
    ]


def config_tuple(config: dict[str, Any]) -> tuple:
    return (
        config["station_name"],
        int(config["component_count"]),
        float(config["length_m"]),
        float(config["height_m"]),
        float(config["width_m"]),
        int(config["crane_count"]),
        float(config["base_capacity_t"]),
        int(config["population_size"]),
        int(config["generations"]),
        int(config["seed"]),
        round(float(config["time_weight"]), 3),
        round(float(config["cost_weight"]), 3),
        round(float(config["risk_weight"]), 3),
    )


def runtime_config_tuple(config: dict[str, Any]) -> tuple:
    return config_tuple(config) + (
        str(config.get("project_source", "")),
        str(config.get("data_source", "")),
        str(config.get("project_name", "")),
        str(config.get("dataset_label", "")),
    )


def numeric_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def display_text(value: Any, fallback: str = "-") -> str:
    text = str(value).strip()
    if not text or "\ufffd" in text:
        return fallback
    question_marks = text.count("?")
    if question_marks >= 2 and question_marks >= len(text) // 3:
        return fallback
    return text


def optimization_overlay_html(value: float, text: str) -> str:
    percent = max(0.0, min(100.0, float(value) * 100.0))
    return f"""
<div class="optimization-overlay" role="status" aria-live="polite">
  <div class="optimization-overlay-backdrop"></div>
  <div class="optimization-overlay-panel">
    <div class="optimization-overlay-kicker">HOIST-OPT</div>
    <div class="optimization-overlay-title">优化计算中</div>
    <div class="optimization-overlay-text">{html.escape(str(text))}</div>
    <div class="optimization-overlay-track">
      <div class="optimization-overlay-fill" style="width:{percent:.2f}%"></div>
    </div>
    <div class="optimization-overlay-footer">
      <span>正在求解吊装调度方案</span>
      <strong>{percent:.0f}%</strong>
    </div>
  </div>
</div>
"""


class OverlayProgress:
    def __init__(self, text: str = "正在处理...") -> None:
        self.placeholder = st.empty()
        self.progress(0.0, text)

    def progress(self, value: float, text: str | None = None) -> None:
        self.placeholder.markdown(
            optimization_overlay_html(
                value,
                text or "正在处理...",
            ),
            unsafe_allow_html=True,
        )

    def empty(self) -> None:
        self.placeholder.empty()


@lru_cache(maxsize=6)
def run_cached_optimization(serialized_config: tuple) -> tuple[OptimizationResult, list[dict]]:
    (
        station_name,
        component_count,
        length_m,
        height_m,
        width_m,
        crane_count,
        base_capacity_t,
        population_size,
        generations,
        seed,
        time_weight,
        cost_weight,
        risk_weight,
    ) = serialized_config
    components = build_station(
        station_name=station_name,
        component_count=component_count,
        length_m=length_m,
        height_m=height_m,
        width_m=width_m,
        seed=seed,
    )
    cranes = build_cranes(
        station_name,
        crane_count,
        base_capacity_t,
        length_m=length_m,
        width_m=width_m,
    )
    result = optimize_hoisting_sequence(
        components,
        cranes,
        population_size=population_size,
        generations=generations,
        seed=seed,
        decision_weights=(time_weight, cost_weight, risk_weight),
    )
    return result, cranes


def run_optimization_with_progress(
    config: dict[str, Any],
    progress_bar,
    components: Sequence[Component] | None = None,
    equipment_config: Sequence[dict] | None = None,
) -> tuple[OptimizationResult, list[dict], float]:
    components = list(components) if components is not None else build_station(
        station_name=config["station_name"],
        component_count=config["component_count"],
        length_m=config["length_m"],
        height_m=config["height_m"],
        width_m=config["width_m"],
        seed=config["seed"],
    )
    cranes = (
        list(equipment_config)
        if equipment_config
        else build_cranes(
            config["station_name"],
            config["crane_count"],
            config["base_capacity_t"],
            length_m=config["length_m"],
            height_m=config["height_m"],
            width_m=config["width_m"],
            components=components if config.get("data_source") == "导入数据" else None,
        )
    )

    def update_progress(generation: int, total: int, objectives: np.ndarray) -> None:
        best_time = objectives[:, 0].min()
        progress_bar.progress(
            generation / total,
            text=f"迭代 {generation}/{total} · 当前最短工期 {best_time / 60.0:.1f} h",
        )

    started = time.perf_counter()
    result = optimize_hoisting_sequence(
        components,
        cranes,
        population_size=config["population_size"],
        generations=config["generations"],
        seed=config["seed"],
        decision_weights=(
            config["time_weight"],
            config["cost_weight"],
            config["risk_weight"],
        ),
        progress_callback=update_progress,
    )
    elapsed = time.perf_counter() - started
    progress_bar.progress(1.0, text="优化完成")
    time.sleep(0.25)
    return result, cranes, elapsed


def component_signature(components: Sequence[Component]) -> tuple:
    return tuple(
        (
            component.id,
            component.type,
            round(component.x, 3),
            round(component.y, 3),
            round(component.z, 3),
            round(component.weight_t, 3),
            component.predecessors,
        )
        for component in components
    )


def equipment_signature(equipment: Sequence[dict]) -> tuple:
    return tuple(
        (
            item["id"],
            round(float(item["capacity_t"]), 3),
            round(float(item.get("rail_start_x", 0.0)), 3),
            round(float(item.get("rail_end_x", 0.0)), 3),
            round(float(item.get("trolley_min_y", 0.0)), 3),
            round(float(item.get("trolley_max_y", 0.0)), 3),
        )
        for item in equipment
    )


def run_expert_optimization_with_progress(
    config: dict[str, Any],
    components: Sequence[Component],
    cranes: Sequence[dict],
    reference_divisions: int,
    generations: int,
    progress_bar,
) -> tuple[OptimizationResult, float]:
    reference_count = math.comb(
        5 + reference_divisions - 1,
        reference_divisions,
    )
    population_size = max(35, reference_count, int(config["population_size"]))

    def update_progress(generation: int, total: int, objectives: np.ndarray) -> None:
        progress_bar.progress(
            generation / total,
            text=(
                f"NSGA-III 迭代 {generation}/{total}"
                f" · 前沿规模 {len(_fast_front_indices(objectives))}"
            ),
        )

    started = time.perf_counter()
    result = optimize_hoisting_sequence(
        components,
        cranes,
        population_size=population_size,
        generations=generations,
        seed=config["seed"],
        decision_weights=(
            config["time_weight"],
            config["cost_weight"],
            config["risk_weight"],
        ),
        algorithm="nsga3",
        objective_mode="expert",
        reference_divisions=reference_divisions,
        progress_callback=update_progress,
    )
    elapsed = time.perf_counter() - started
    progress_bar.progress(1.0, text="优化完成")
    time.sleep(0.25)
    return result, elapsed


def _fast_front_indices(objectives: np.ndarray) -> list[int]:
    if len(objectives) <= 1:
        return list(range(len(objectives)))
    indices = np.arange(len(objectives))
    dominated = np.zeros(len(objectives), dtype=bool)
    for index in indices:
        for other in indices:
            if index == other:
                continue
            if np.all(objectives[other] <= objectives[index]) and np.any(
                objectives[other] < objectives[index]
            ):
                dominated[index] = True
                break
    return [int(index) for index in indices[~dominated]]


default_station_name = next(iter(STATION_PRESETS))
default_preset = STATION_PRESETS[default_station_name]

if "active_config" not in st.session_state:
    st.session_state.active_config = {
        "station_name": default_station_name,
        "component_count": default_preset["component_count"],
        "length_m": default_preset["length_m"],
        "height_m": default_preset["height_m"],
        "width_m": default_preset["width_m"],
        "crane_count": len(default_preset["cranes"]),
        "base_capacity_t": default_preset["cranes"][0]["capacity_t"],
        "population_size": 32,
        "generations": 20,
        "seed": 20250925,
        "time_weight": 0.40,
        "cost_weight": 0.30,
        "risk_weight": 0.30,
        "data_source": "参数化场景",
        "project_source": "示例项目",
        "project_name": default_station_name,
        "dataset_label": default_station_name,
        "data_revision": 0,
    }

st.session_state.active_config.setdefault("data_source", "参数化场景")
st.session_state.active_config.setdefault("project_source", "示例项目")
st.session_state.active_config.setdefault(
    "project_name",
    st.session_state.active_config["dataset_label"],
)
st.session_state.active_config.setdefault(
    "dataset_label",
    st.session_state.active_config["station_name"],
)
st.session_state.active_config.setdefault("data_revision", 0)
st.session_state.setdefault("imported_components", None)
st.session_state.setdefault("imported_validation", None)
st.session_state.setdefault("imported_source_name", "")
st.session_state.setdefault("imported_equipment", [])
st.session_state.setdefault("imported_project", {})
st.session_state.setdefault("imported_relations", pd.DataFrame())

if "result" not in st.session_state:
    with st.spinner("正在构建参数化模型并求解 Pareto 前沿，请稍候..."):
        initial_result, initial_cranes = run_cached_optimization(
            config_tuple(st.session_state.active_config)
        )
        st.session_state.result = initial_result
        st.session_state.cranes = initial_cranes
        st.session_state.elapsed_seconds = float("nan")

st.session_state.setdefault(
    "selected_project_source",
    st.session_state.active_config.get("project_source", "示例项目"),
)
st.session_state.setdefault(
    "applied_config",
    dict(st.session_state.active_config),
)
st.session_state.setdefault("last_run_summary", None)
st.session_state.setdefault("applied_progress", 0)


with st.sidebar:
    st.markdown(sidebar_brand_html(), unsafe_allow_html=True)
    presentation_mode = st.checkbox(
        "演示模式（一屏展示）",
        key="presentation_mode",
    )
    with st.container():
        imported_components = st.session_state.get("imported_components")
        active_config = st.session_state.active_config
        project_source = st.session_state.get(
            "selected_project_source",
            active_config.get("project_source", "示例项目"),
        )
        using_imported_data = (
            project_source == "导入项目" and imported_components is not None
        )
        example_options = list(STATION_PRESETS.keys())
        if using_imported_data:
            example_selection = active_config["station_name"]
        else:
            example_index = (
                example_options.index(active_config["station_name"])
                if active_config.get("station_name") in example_options
                else 0
            )
            example_selection = st.selectbox(
                "典型示例",
                example_options,
                index=example_index,
                key=f"example_selector_{active_config.get('data_revision', 0)}",
            )
        if project_source == "导入项目" and imported_components is None:
            st.info("请先在“项目与数据”页导入项目文件。")
        if project_source == "导入项目":
            selected_station = active_config["station_name"]
            if using_imported_data:
                project_name = display_text(
                    normalize_project_metadata(
                        st.session_state.get("imported_project", {})
                    ).get("项目名称"),
                    active_config.get(
                        "dataset_label",
                        active_config["station_name"],
                    ),
                )
            else:
                project_name = display_text(
                    active_config.get("project_name"),
                    "导入项目",
                )
            switch_requested = False
            if using_imported_data:
                with st.popover("切换典型示例", use_container_width=True):
                    switch_example = st.radio(
                        "选择示例项目",
                        example_options,
                        key=(
                            "switch_example_"
                            f"{active_config.get('data_revision', 0)}"
                        ),
                    )
                    switch_requested = st.button(
                        "载入并运行示例",
                        use_container_width=True,
                        key=(
                            "switch_example_confirm_"
                            f"{active_config.get('data_revision', 0)}"
                        ),
                    )
                    if switch_requested:
                        example_selection = switch_example
            if switch_requested:
                selected_preset = STATION_PRESETS[example_selection]
                example_config = st.session_state.active_config.copy()
                example_config.update(
                    {
                        "station_name": example_selection,
                        "component_count": int(
                            selected_preset["component_count"]
                        ),
                        "length_m": float(selected_preset["length_m"]),
                        "height_m": float(selected_preset["height_m"]),
                        "width_m": float(selected_preset["width_m"]),
                        "crane_count": len(selected_preset["cranes"]),
                        "base_capacity_t": float(
                            selected_preset["cranes"][0]["capacity_t"]
                        ),
                        "project_source": "示例项目",
                        "data_source": "参数化场景",
                        "project_name": example_selection,
                        "dataset_label": example_selection,
                        "data_revision": int(
                            st.session_state.active_config.get(
                                "data_revision",
                                0,
                            )
                        )
                        + 1,
                    }
                )
                progress = OverlayProgress("正在载入示例项目并重新优化...")
                example_result, example_cranes, example_elapsed = (
                    run_optimization_with_progress(
                        example_config,
                        progress,
                    )
                )
                progress.empty()
                st.session_state.result = example_result
                st.session_state.cranes = example_cranes
                st.session_state.active_config = example_config
                st.session_state.applied_config = dict(example_config)
                st.session_state.selected_project_source = "示例项目"
                st.session_state.project_name_override = None
                st.session_state.last_run_summary = {
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_seconds": float(example_elapsed),
                    "pareto_size": int(
                        example_result.metadata["pareto_size"]
                    ),
                    "project_name": example_selection,
                    "component_count": len(example_result.components),
                    "crane_count": len(example_cranes),
                }
                st.rerun()
        else:
            selected_station = example_selection
            if project_source == "示例项目":
                project_name = selected_station
            else:
                project_name = display_text(
                    active_config.get("project_name"),
                    "参数化新建项目",
                )
        selected_preset = STATION_PRESETS[selected_station]
        selected_code = selected_preset["code"]
        imported_project_metadata = (
            normalize_project_metadata(
                st.session_state.get("imported_project", {})
            )
            if using_imported_data
            else {}
        )
        imported_length_m = numeric_value(
            imported_project_metadata.get("长度"),
            active_config["length_m"] if using_imported_data else selected_preset["length_m"],
        )
        imported_height_m = numeric_value(
            imported_project_metadata.get("高度"),
            active_config["height_m"] if using_imported_data else selected_preset["height_m"],
        )
        imported_width_m = numeric_value(
            imported_project_metadata.get("宽度"),
            active_config["width_m"] if using_imported_data else selected_preset["width_m"],
        )
        default_project_name = project_name
        rename_context = "|".join(
            (
                str(project_source),
                str(selected_station),
                str(active_config.get("data_revision", 0)),
                str(default_project_name),
            )
        )
        project_name_override = st.session_state.get("project_name_override")
        if (
            isinstance(project_name_override, dict)
            and project_name_override.get("context") == rename_context
        ):
            project_name = display_text(
                project_name_override.get("name"),
                default_project_name,
            )
        current_project_label = display_text(
            project_name,
            selected_station,
        )
        current_component_count = (
            len(imported_components or [])
            if using_imported_data
            else int(selected_preset["component_count"])
        )
        st.markdown(
            sidebar_readout("当前项目", current_project_label),
            unsafe_allow_html=True,
        )
        rename_key = hashlib.sha256(
            rename_context.encode("utf-8")
        ).hexdigest()[:12]
        with st.popover("重命名项目", use_container_width=True):
            renamed_project_name = st.text_input(
                "新项目名称",
                value=current_project_label,
                key=f"rename_project_{rename_key}",
            )
            if st.button(
                "确认重命名",
                type="primary",
                use_container_width=True,
                key=f"rename_project_confirm_{rename_key}",
            ):
                clean_project_name = display_text(
                    renamed_project_name,
                    current_project_label,
                )
                st.session_state.project_name_override = {
                    "context": rename_context,
                    "name": clean_project_name,
                }
                st.session_state.active_config["project_name"] = clean_project_name
                st.session_state.active_config["dataset_label"] = clean_project_name
                applied_config = st.session_state.get("applied_config")
                if isinstance(applied_config, dict):
                    applied_config["project_name"] = clean_project_name
                    applied_config["dataset_label"] = clean_project_name
                last_run_summary = st.session_state.get("last_run_summary")
                if isinstance(last_run_summary, dict):
                    last_run_summary["project_name"] = clean_project_name
                if using_imported_data:
                    updated_imported_project = normalize_project_metadata(
                        st.session_state.get("imported_project", {})
                    )
                    updated_imported_project["项目名称"] = clean_project_name
                    st.session_state.imported_project = updated_imported_project
                st.rerun()
        st.markdown(
            sidebar_readout(
                "构件总数",
                f"{current_component_count} 件",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            sidebar_readout(
                "构件总重",
                (
                    f"{sum(component.weight_t for component in imported_components):.2f} t"
                    if using_imported_data and imported_components
                    else "按示例数据计算"
                ),
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            sidebar_readout(
                "站台尺寸",
                f"长 {imported_length_m:g} · 宽 {imported_width_m:g} · 高 {imported_height_m:g}",
            ),
            unsafe_allow_html=True,
        )

        component_count = int(current_component_count)
        length_m = st.slider(
            "站台长度 (m)",
            min_value=50.0,
            max_value=200.0,
            value=(
                float(imported_length_m)
                if using_imported_data
                else (
                    float(st.session_state.active_config["length_m"])
                    if selected_station
                    == st.session_state.active_config["station_name"]
                    else float(selected_preset["length_m"])
                )
            ),
            step=1.0,
            key=f"length_{selected_code}",
            disabled=project_source == "导入项目" and imported_components is None,
        )
        height_m = st.slider(
            "结构高度 (m)",
            min_value=5.0,
            max_value=15.0,
            value=(
                float(imported_height_m)
                if using_imported_data
                else (
                    float(st.session_state.active_config["height_m"])
                    if selected_station
                    == st.session_state.active_config["station_name"]
                    else float(selected_preset["height_m"])
                )
            ),
            step=0.2,
            key=f"height_{selected_code}",
            disabled=project_source == "导入项目" and imported_components is None,
        )
        width_m = st.slider(
            "车站宽度 (m)",
            min_value=18.0,
            max_value=32.0,
            value=(
                float(imported_width_m)
                if using_imported_data
                else (
                    float(st.session_state.active_config["width_m"])
                    if selected_station
                    == st.session_state.active_config["station_name"]
                    else float(selected_preset["width_m"])
                )
            ),
            step=0.2,
            key=f"width_{selected_code}",
            disabled=project_source == "导入项目" and imported_components is None,
        )

        st.markdown('<div class="section-label">设备配置</div>', unsafe_allow_html=True)
        imported_equipment = st.session_state.get("imported_equipment", [])
        equipment_source_options = ["面板参数"]
        if using_imported_data and imported_equipment:
            equipment_source_options = ["导入设备表", "面板参数"]
        equipment_source = st.radio(
            "设备来源",
            equipment_source_options,
            horizontal=True,
        )
        use_imported_equipment = (
            using_imported_data
            and bool(imported_equipment)
            and equipment_source == "导入设备表"
        )
        crane_count = st.slider(
            "龙门吊数量",
            min_value=1,
            max_value=3,
            value=(
                len(imported_equipment)
                if use_imported_equipment
                else (
                    int(st.session_state.active_config["crane_count"])
                    if selected_station == st.session_state.active_config["station_name"]
                    else len(selected_preset["cranes"])
                )
            ),
            key=f"crane_count_{selected_code}",
            disabled=use_imported_equipment,
        )
        base_capacity_t = st.slider(
            "单机额定能力 (t)",
            min_value=80.0,
            max_value=220.0,
            value=(
                numeric_value(
                    imported_equipment[0].get("capacity_t")
                    if use_imported_equipment and imported_equipment
                    else None,
                    (
                        active_config["base_capacity_t"]
                        if selected_station == active_config["station_name"]
                        else selected_preset["cranes"][0]["capacity_t"]
                    ),
                )
            ),
            step=5.0,
            key=f"capacity_{selected_code}",
            disabled=use_imported_equipment,
        )

        with st.expander("求解参数与决策权重", expanded=False):
            population_size = st.slider(
                "种群规模",
                min_value=16,
                max_value=80,
                value=int(st.session_state.active_config["population_size"]),
                step=4,
            )
            generations = st.slider(
                "迭代代数",
                min_value=10,
                max_value=80,
                value=int(st.session_state.active_config["generations"]),
                step=2,
            )
            seed = st.number_input(
                "随机种子",
                min_value=1,
                max_value=99999999,
                value=int(st.session_state.active_config["seed"]),
                step=1,
            )
            time_weight = st.slider(
                "工期权重",
                min_value=0.05,
                max_value=0.80,
                value=float(st.session_state.active_config["time_weight"]),
                step=0.05,
            )
            remaining_weight = max(0.10, 1.0 - time_weight)
            cost_share = st.slider(
                "成本权重占比",
                min_value=0.10,
                max_value=0.90,
                value=(
                    float(st.session_state.active_config["cost_weight"]) / remaining_weight
                    if remaining_weight > 0
                    else 0.50
                ),
                step=0.05,
            )
            cost_weight = remaining_weight * cost_share
            risk_weight = remaining_weight * (1.0 - cost_share)
            st.caption(f"成本 {cost_weight:.2f} · 安全 {risk_weight:.2f}")

        draft_project_name = display_text(
            project_name,
            active_config.get(
                "dataset_label",
                active_config["station_name"],
            ),
        )
        source_label = {
            "示例项目": "示例数据",
            "参数化新建": "参数化数据",
            "导入项目": "导入数据",
        }.get(project_source, project_source)
        if project_source == "导入项目" and imported_components is None:
            data_status = "未导入"
        elif project_source == "导入项目":
            data_status = "校验通过"
        else:
            data_status = "可计算"
        draft_config = {
            "station_name": selected_station,
            "component_count": int(component_count),
            "length_m": float(length_m),
            "height_m": float(height_m),
            "width_m": float(width_m),
            "crane_count": int(crane_count),
            "base_capacity_t": float(base_capacity_t),
            "population_size": int(population_size),
            "generations": int(generations),
            "seed": int(seed),
            "time_weight": float(time_weight),
            "cost_weight": float(cost_weight),
            "risk_weight": float(risk_weight),
            "project_source": project_source,
            "data_source": (
                "导入数据" if project_source == "导入项目" else "参数化场景"
            ),
            "project_name": draft_project_name,
            "dataset_label": draft_project_name,
            "data_revision": int(active_config.get("data_revision", 0)),
        }
        config_changed = runtime_config_tuple(draft_config) != runtime_config_tuple(
            st.session_state.applied_config
        )
        st.markdown(
            sidebar_readout("数据来源", source_label),
            unsafe_allow_html=True,
        )
        st.markdown(
            sidebar_readout("数据状态", data_status),
            unsafe_allow_html=True,
        )
        st.markdown(
            sidebar_readout(
                "运行状态",
                "配置已修改，待运行" if config_changed else "与最近一次运行一致",
            ),
            unsafe_allow_html=True,
        )
        submitted = st.button(
            "应用修改并重新优化" if config_changed else "重新运行优化",
            type="primary",
            use_container_width=True,
            disabled=project_source == "导入项目" and imported_components is None,
        )
        last_run_summary = st.session_state.get("last_run_summary")
        if last_run_summary:
            st.caption(
                f"最近运行：{last_run_summary['finished_at']} · "
                f"{last_run_summary['elapsed_seconds']:.1f} 秒 · "
                f"{last_run_summary['pareto_size']} 个 Pareto 解"
            )

    st.markdown(
        sidebar_readout(
            "模型状态",
            f"{len(st.session_state.result.components)} 构件 · LOD 300",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        sidebar_readout(
            "Pareto 前沿",
            f"{st.session_state.result.metadata['pareto_size']} 个非支配解",
        ),
        unsafe_allow_html=True,
    )


if submitted:
    submitted_project_name = str(project_name).strip()
    if not submitted_project_name:
        submitted_project_name = (
            str(
                normalize_project_metadata(
                    st.session_state.get("imported_project", {})
                ).get("项目名称")
            ).strip()
            if using_imported_data
            else str(selected_station)
        )
    if using_imported_data:
        updated_imported_project = normalize_project_metadata(
            st.session_state.get("imported_project", {})
        )
        updated_imported_project["项目名称"] = submitted_project_name
        st.session_state.imported_project = updated_imported_project
    new_config = {
        "station_name": selected_station,
        "component_count": int(component_count),
        "length_m": float(length_m),
        "height_m": float(height_m),
        "width_m": float(width_m),
        "crane_count": int(crane_count),
        "base_capacity_t": float(base_capacity_t),
        "population_size": int(population_size),
        "generations": int(generations),
        "seed": int(seed),
        "time_weight": float(time_weight),
        "cost_weight": float(cost_weight),
        "risk_weight": float(risk_weight),
        "project_source": project_source,
        "data_source": "导入数据" if using_imported_data else "参数化场景",
        "project_name": submitted_project_name,
        "dataset_label": submitted_project_name,
        "data_revision": int(st.session_state.active_config.get("data_revision", 0)),
    }
    progress = OverlayProgress("初始化种群...")
    optimization_components = imported_components if using_imported_data else None
    result, cranes, elapsed = run_optimization_with_progress(
        new_config,
        progress,
        components=optimization_components,
        equipment_config=(
            st.session_state.get("imported_equipment")
            if use_imported_equipment
            else None
        ),
    )
    progress.empty()
    st.session_state.result = result
    st.session_state.cranes = cranes
    st.session_state.active_config = new_config
    st.session_state.applied_config = dict(new_config)
    st.session_state.selected_project_source = project_source
    st.session_state.last_run_summary = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": float(elapsed),
        "pareto_size": int(result.metadata["pareto_size"]),
        "project_name": submitted_project_name,
        "component_count": len(result.components),
        "crane_count": len(cranes),
    }
    st.session_state.elapsed_seconds = elapsed
    st.rerun()


if presentation_mode:
    st.markdown(
        """
<style>
[data-testid="stAppViewContainer"] {
  zoom: .78;
}
.block-container {
  max-width: 1760px !important;
  padding-top: .35rem !important;
  padding-bottom: .55rem !important;
}
.app-hero {
  display: none !important;
}
[data-testid="stMetric"] {
  min-height: 84px !important;
  padding: 8px 10px !important;
}
[data-testid="stMetricValue"] {
  font-size: 22px !important;
}
.subsection-heading {
  font-size: 15px !important;
  margin: 4px 0 1px !important;
}
.subsection-note {
  display: none !important;
}
[data-testid="stTabs"] [role="tabpanel"] {
  padding-top: 6px !important;
}
[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"] {
  margin-top: 0 !important;
  margin-bottom: 4px !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

result: OptimizationResult = st.session_state.result
components = result.components
cranes = st.session_state.cranes
active_config = st.session_state.active_config
recommended = result.plans["综合推荐"]
active_imported_project = (
    normalize_project_metadata(st.session_state.get("imported_project", {}))
    if active_config.get("data_source") == "导入数据"
    else {}
)
active_project_name = display_text(
    active_config.get("project_name")
    or active_config.get("dataset_label")
    or active_imported_project.get("项目名称"),
    display_text(active_config["station_name"]),
)
active_structure_template = display_text(
    active_imported_project.get("结构形式")
    or (
        "预制装配式地铁车站结构"
        if active_config.get("data_source") == "参数化场景"
        else "自定义装配式结构"
    ),
    "预制装配式结构",
)

st.markdown(
    hero_html(
        project_name=active_project_name,
        component_count=len(components),
        cranes=cranes,
        algorithm_label="NSGA-II / Pareto 求解",
    ),
    unsafe_allow_html=True,
)

st.session_state.setdefault("expert_result", None)
st.session_state.setdefault("expert_cache_key", None)
analysis_mode = st.radio(
    "分析模式",
    ["决策模式", "专业模式"],
    horizontal=True,
    label_visibility="collapsed",
)

if analysis_mode == "专业模式":
    with st.expander("NSGA-III 参数", expanded=False):
        reference_divisions = st.select_slider(
            "参考点划分数",
            options=[2, 3, 4],
            value=3,
        )
        expert_generations = st.slider(
            "专业模式迭代代数",
            min_value=10,
            max_value=60,
            value=max(20, int(active_config["generations"])),
            step=2,
        )
    expert_cache_key = (
        config_tuple(active_config),
        component_signature(components),
        equipment_signature(cranes),
        int(reference_divisions),
        int(expert_generations),
    )
    if (
        st.session_state.expert_result is None
        or st.session_state.expert_cache_key != expert_cache_key
    ):
        st.markdown(
            '<div class="decision-band">'
            '<h3>五目标 NSGA-III 专业分析</h3>'
            '<p>'
            "目标为工期、成本、空间碰撞、作业环境和设备均衡。"
            "该计算量高于决策模式，仅在需要算法分析时运行。"
            "</p></div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "运行五目标 NSGA-III",
            type="primary",
            use_container_width=True,
        ):
            progress = OverlayProgress("初始化 NSGA-III 种群...")
            expert_result, expert_elapsed = run_expert_optimization_with_progress(
                active_config,
                components,
                cranes,
                reference_divisions=int(reference_divisions),
                generations=int(expert_generations),
                progress_bar=progress,
            )
            progress.empty()
            st.session_state.expert_result = expert_result
            st.session_state.expert_cache_key = expert_cache_key
            st.session_state.expert_elapsed_seconds = expert_elapsed
            st.rerun()
        st.stop()

    expert_result: OptimizationResult = st.session_state.expert_result
    render_expert_metrics(expert_result)
    st.caption(
        f"NSGA-III · 参考方向 {expert_result.metadata['reference_points']} · "
        f"求解 {st.session_state.get('expert_elapsed_seconds', 0.0):.1f} 秒"
    )
    expert_objective_tab, expert_pareto_tab, expert_risk_tab, expert_sensitivity_tab, expert_audit_tab, expert_compare_tab = st.tabs(
        ["目标体系", "NSGA-III 前沿", "风险分解", "权重敏感", "算法审计", "模式对比"]
    )
    with expert_objective_tab:
        render_objective_model(expert_result)
    with expert_pareto_tab:
        render_pareto_pages(expert_result)
    with expert_risk_tab:
        render_risk_analysis(expert_result)
    with expert_sensitivity_tab:
        render_sensitivity_analysis(expert_result)
    with expert_audit_tab:
        render_algorithm_audit(expert_result)
    with expert_compare_tab:
        render_mode_comparison(expert_result, result)
    st.stop()

metric_columns = st.columns(4)
metric_columns[0].metric(
    "计划完成周期",
    f"{recommended['makespan_min'] / (8.0 * 60.0):.1f} 工日",
    f"较最高方案 -{100 * (1 - recommended['makespan_min'] / max(plan['makespan_min'] for plan in result.plans.values())):.0f}%",
)
metric_columns[1].metric(
    "吊装综合成本",
    f"{recommended['cost_yuan'] / 10000.0:.1f} 万元",
    f"较最高方案 -{100 * (1 - recommended['cost_yuan'] / max(plan['cost_yuan'] for plan in result.plans.values())):.0f}%",
)
metric_columns[2].metric(
    "安全风险指数",
    f"{recommended['risk_score']:.1f}",
    f"较最高方案 -{100 * (1 - recommended['risk_score'] / max(plan['risk_score'] for plan in result.plans.values())):.0f}%",
)
metric_columns[3].metric(
    "最优前沿规模",
    f"{result.metadata['pareto_size']} 解",
    f"评估 {result.metadata['evaluations']} 个序列",
)

st.markdown('<div class="section-label">方案视图</div>', unsafe_allow_html=True)
selected_plan_name = st.radio(
    "当前方案",
    list(result.plans.keys()),
    index=list(result.plans.keys()).index("综合推荐"),
    horizontal=True,
    label_visibility="collapsed",
)
selected_plan = result.plans[selected_plan_name]

# Importing a differently sized dataset changes every downstream table shape.
if active_config.get("data_revision", 0):
    st.empty()

project_data_tab, overview_tab, sequence_tab, tradeoff_tab, decision_tab = st.tabs(
    ["项目与数据", "三维总览", "优化序列", "目标权衡", "方案决策"]
)


with project_data_tab:
    current_project = (
        normalize_project_metadata(
            dict(st.session_state.get("imported_project", {}))
        )
        if active_config.get("data_source") == "导入数据"
        else {
            "项目编号": active_config.get("project_id", ""),
            "项目名称": active_config.get(
                "project_name",
                active_config.get("dataset_label", active_config["station_name"]),
            ),
            "工程场景": active_config["station_name"],
        }
    )
    current_project.setdefault(
        "项目名称",
        active_config.get("dataset_label", active_config["station_name"]),
    )
    current_project.setdefault("构件数量", len(components))
    current_project.setdefault("设备数量", len(cranes))

    save_column = st.container()
    with save_column:
        st.markdown(
            '<div class="subsection-heading">保存项目</div>'
            '<div class="subsection-note">'
            "保存构件、关系、设备和项目参数，不保存算法运行明细。"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "保存当前项目",
            type="secondary",
            use_container_width=True,
        ):
            imported_relations = st.session_state.get("imported_relations")
            relation_records = (
                imported_relations.to_dict("records")
                if isinstance(imported_relations, pd.DataFrame)
                and not imported_relations.empty
                else []
            )
            saved = save_project(
                project=current_project,
                components=components,
                equipment=cranes,
                relations=relation_records,
                project_id=st.session_state.get("active_project_id"),
            )
            st.session_state.active_project_id = saved["id"]
            st.session_state.active_config["project_id"] = saved["id"]
            st.session_state.active_config["project_name"] = saved["project"].get(
                "项目名称",
                active_config.get("project_name"),
            )
            st.success(f"项目已保存：{saved['project'].get('项目名称', saved['id'])}")

    st.markdown(
        '<div class="subsection-heading">已保存项目</div>',
        unsafe_allow_html=True,
    )
    saved_projects = list_saved_projects()
    if not saved_projects:
        st.info("还没有保存项目。当前示例项目或参数化项目可以直接保存。")
    else:
        delete_notice = st.session_state.pop("project_delete_notice", None)
        if delete_notice:
            st.success(delete_notice)
        load_column, open_column, delete_column = st.columns(
            [2.6, 1.0, 1.0],
            gap="medium",
        )
        with load_column:
            selected_project_id = st.selectbox(
                "选择项目",
                [item["id"] for item in saved_projects],
                format_func=lambda project_id: next(
                    (
                        f"{item['name']} · {item['component_count']} 构件 · "
                        f"{item['updated_at']}"
                        for item in saved_projects
                        if item["id"] == project_id
                    ),
                    project_id,
                ),
            )
        with open_column:
            st.markdown(
                '<div class="project-action-spacer"></div>',
                unsafe_allow_html=True,
            )
            if st.button("打开所选项目", use_container_width=True):
                loaded = load_project(selected_project_id)
                loaded_components = loaded["components"]
                loaded_equipment = loaded.get("equipment", [])
                loaded_project = normalize_project_metadata(
                    loaded.get("project", {})
                )
                loaded_name = display_text(
                    loaded_project.get("项目名称"),
                    selected_project_id,
                )
                new_config = st.session_state.active_config.copy()
                new_config.update(
                    {
                        "project_source": "导入项目",
                        "data_source": "导入数据",
                        "project_name": loaded_name,
                        "dataset_label": loaded_name,
                        "project_id": selected_project_id,
                        "component_count": len(loaded_components),
                        "length_m": numeric_value(
                            loaded_project.get("长度"),
                            st.session_state.active_config["length_m"],
                        ),
                        "height_m": numeric_value(
                            loaded_project.get("高度"),
                            st.session_state.active_config["height_m"],
                        ),
                        "width_m": numeric_value(
                            loaded_project.get("宽度"),
                            st.session_state.active_config["width_m"],
                        ),
                        "crane_count": len(loaded_equipment)
                        or int(st.session_state.active_config["crane_count"]),
                        "base_capacity_t": numeric_value(
                            loaded_equipment[0].get("capacity_t")
                            if loaded_equipment
                            else None,
                            st.session_state.active_config["base_capacity_t"],
                        ),
                        "data_revision": int(
                            st.session_state.active_config.get("data_revision", 0)
                        )
                        + 1,
                    }
                )
                progress = OverlayProgress("正在打开项目并重新优化...")
                loaded_result, loaded_cranes, elapsed = run_optimization_with_progress(
                    new_config,
                    progress,
                    components=loaded_components,
                    equipment_config=loaded_equipment or None,
                )
                progress.empty()
                st.session_state.imported_components = loaded_components
                st.session_state.imported_equipment = loaded_equipment
                st.session_state.imported_project = loaded_project
                st.session_state.selected_project_source = "导入项目"
                st.session_state.imported_relations = pd.DataFrame(
                    loaded.get("relations", [])
                )
                st.session_state.imported_source_name = f"{loaded_name}.json"
                st.session_state.imported_validation = None
                st.session_state.import_signature = None
                st.session_state.result = loaded_result
                st.session_state.cranes = loaded_cranes
                st.session_state.active_config = new_config
                st.session_state.applied_config = dict(new_config)
                st.session_state.last_run_summary = {
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_seconds": float(elapsed),
                    "pareto_size": int(loaded_result.metadata["pareto_size"]),
                    "project_name": loaded_name,
                    "component_count": len(loaded_components),
                    "crane_count": len(loaded_cranes),
                }
                st.session_state.elapsed_seconds = elapsed
                st.session_state.applied_progress = 0
                st.session_state.expert_result = None
                st.session_state.expert_cache_key = None
                st.rerun()
        with delete_column:
            st.markdown(
                '<div class="project-action-spacer"></div>',
                unsafe_allow_html=True,
            )
            if st.button("删除所选项目", use_container_width=True):
                st.session_state.pending_delete_project_id = selected_project_id
                st.rerun()

        pending_project_id = st.session_state.get("pending_delete_project_id")
        if pending_project_id:
            pending_item = next(
                (
                    item
                    for item in saved_projects
                    if item["id"] == pending_project_id
                ),
                None,
            )
            if pending_item is None:
                st.session_state.pending_delete_project_id = None
            else:
                st.warning(
                    "确认删除已保存项目："
                    f"{pending_item['name']}（{pending_item['id']}）？"
                    "删除后无法恢复。"
                )
                confirm_column, cancel_column = st.columns(2)
                with confirm_column:
                    if st.button(
                        "确认删除",
                        type="primary",
                        use_container_width=True,
                    ):
                        try:
                            deleted = delete_project(pending_project_id)
                        except (OSError, ValueError) as error:
                            st.error(f"删除失败：{error}")
                        else:
                            if deleted:
                                if (
                                    st.session_state.get("active_project_id")
                                    == pending_project_id
                                ):
                                    st.session_state.pop(
                                        "active_project_id",
                                        None,
                                    )
                                if (
                                    st.session_state.active_config.get("project_id")
                                    == pending_project_id
                                ):
                                    st.session_state.active_config.pop(
                                        "project_id",
                                        None,
                                    )
                                st.session_state.project_delete_notice = (
                                    f"已删除保存项目：{pending_item['name']}"
                                )
                            else:
                                st.session_state.project_delete_notice = (
                                    "项目不存在或已经被删除。"
                                )
                            st.session_state.pending_delete_project_id = None
                            st.rerun()
                with cancel_column:
                    if st.button("取消删除", use_container_width=True):
                        st.session_state.pending_delete_project_id = None
                        st.rerun()

with overview_tab:
    visual_column, insight_column = st.columns([2.15, 1.0], gap="medium")
    with visual_column:
        with st.spinner("正在生成 Three.js 三维场景..."):
            st_components.html(
                build_three_viewer_html(
                    components,
                    selected_plan,
                    cranes,
                ),
                height=700,
                scrolling=False,
            )
    with insight_column:
        st.plotly_chart(
            build_cost_figure(selected_plan),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.plotly_chart(
            build_component_mix(components),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    workload_rows = []
    total_workload = sum(item["active_minutes"] for item in selected_plan["crane_workload"])
    for item in selected_plan["crane_workload"]:
        share = item["active_minutes"] / total_workload if total_workload else 0.0
        workload_rows.append(
            {
                "吊装设备": item["crane"],
                "有效作业(min)": round(item["active_minutes"], 1),
                "任务数": item["moves"],
                "负荷占比": f"{share:.1%}",
            }
        )
    left_workload, right_workload = st.columns([1.0, 1.0], gap="medium")
    with left_workload:
        st.markdown(
            '<div class="subsection-heading">设备负荷分配</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(workload_rows),
            use_container_width=True,
            hide_index=True,
        )
    with right_workload:
        st.markdown(
            '<div class="subsection-heading">场景校验</div>',
            unsafe_allow_html=True,
        )
        validation = validate_components(components)
        validation_frame = pd.DataFrame(
            [
                {"校验项": "构件总数", "结果": validation["component_count"]},
                {"校验项": "唯一构件编号", "结果": validation["unique_ids"]},
                {"校验项": "重复编号", "结果": validation["duplicate_ids"]},
                {"校验项": "总起吊重量(t)", "结果": validation["total_weight_t"]},
                {"校验项": "最大单件重量(t)", "结果": validation["maximum_weight_t"]},
                {"校验项": "前置约束数量", "结果": validation["dependency_count"]},
            ]
        )
        st.dataframe(validation_frame, use_container_width=True, hide_index=True)


with sequence_tab:
    schedule_data = schedule_frame(selected_plan["schedule"])
    schedule_csv = schedule_data.to_csv(index=False).encode("utf-8-sig")
    toolbar_left, toolbar_right = st.columns([3.0, 1.0])
    with toolbar_left:
        st.markdown(
            '<div class="subsection-heading">吊装任务时间窗</div>'
            '<div class="subsection-note">按双机任务分配展示前 30 个吊装节拍，颜色对应吊装设备。</div>',
            unsafe_allow_html=True,
        )
    with toolbar_right:
        st.download_button(
            "⇩ 导出吊装调度方案",
            data=schedule_csv,
            file_name=f"{active_config['station_name']}_{selected_plan_name}_吊装调度方案.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
    st.plotly_chart(
        build_gantt_figure(selected_plan["schedule"], max_tasks=30),
        use_container_width=True,
        config={"displaylogo": False},
    )
    sequence_table_column, risk_column = st.columns([1.55, 1.0], gap="medium")
    with sequence_table_column:
        st.markdown(
            '<div class="subsection-heading">顺序任务清单</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            highlight_high_risk_rows(
                schedule_data.head(12 if presentation_mode else 18)
            ),
            use_container_width=True,
            hide_index=True,
            height=240 if presentation_mode else 500,
        )
    with risk_column:
        st.markdown(
            '<div class="subsection-heading">高风险构件定位</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_risk_figure(components, selected_plan),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.plotly_chart(
            build_risk_breakdown_figure(selected_plan),
            use_container_width=True,
            config={"displayModeBar": False},
        )


with tradeoff_tab:
    st.markdown(
        '<div class="subsection-heading">Pareto 非支配前沿</div>'
        '<div class="subsection-note">菱形标记为进度、成本、安全三类决策方案及综合推荐方案。</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_pareto_figure(result.pareto_objectives, result.plans),
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.markdown(
        '<div class="subsection-heading">迭代收敛过程</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_convergence_figure(result.history),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    front_frame = pd.DataFrame(
        result.pareto_objectives,
        columns=["工期(min)", "成本(元)", "安全风险指数"],
    )
    front_frame.insert(0, "Pareto序号", range(1, len(front_frame) + 1))
    front_frame["工期(工日)"] = (front_frame["工期(min)"] / (8.0 * 60.0)).round(1)
    front_frame["成本(万元)"] = (front_frame["成本(元)"] / 10000.0).round(1)
    front_frame = front_frame[
        ["Pareto序号", "工期(工日)", "成本(万元)", "安全风险指数"]
    ].round(2)
    st.dataframe(
        front_frame,
        use_container_width=True,
        hide_index=True,
        height=170 if presentation_mode else 280,
    )


with decision_tab:
    st.markdown(
        plan_tiles_html(result.plans),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="decision-band">
  <h3>决策结论 · {selected_plan_name}</h3>
  <p>{decision_summary(selected_plan, len(components))}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    radar_column, comparison_column = st.columns([1.0, 1.0], gap="medium")
    with radar_column:
        st.plotly_chart(
            build_plan_radar(result.plans),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with comparison_column:
        comparison = pd.DataFrame(
            {
                name: {
                    "工期(工日)": plan["makespan_min"] / (8.0 * 60.0),
                    "成本(万元)": plan["cost_yuan"] / 10000.0,
                    "风险指数": plan["risk_score"],
                    "碰撞敏感点": plan["adjacent_crossings"],
                }
                for name, plan in result.plans.items()
            }
        ).T.round(2)
        st.markdown(
            '<div class="subsection-heading">多维指标对照</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            comparison,
            use_container_width=True,
            height=220 if presentation_mode else None,
        )
    st.markdown(
        '<div class="risk-note">'
        "方案推演中的碰撞风险来自吊装顺序、构件空间邻近度和高度变化，"
        "施工前仍需结合实测坐标、吊臂姿态与现场工况完成专业复核。"
        "</div>",
        unsafe_allow_html=True,
    )
    decision_csv = comparison.to_csv().encode("utf-8-sig")
    st.download_button(
        "⇩ 导出决策对照表",
        data=decision_csv,
        file_name=f"{active_config['station_name']}_吊装方案决策对照.csv",
        mime="text/csv",
    )


with project_data_tab:
    import_title, import_actions = st.columns([2.0, 1.35], gap="medium")
    with import_title:
        st.markdown(
            '<div class="subsection-heading">标准化构件数据导入</div>'
            '<div class="subsection-note">'
            "每行对应一件预制构件；校验通过后可直接参与吊装调度优化。"
            "</div>",
            unsafe_allow_html=True,
        )
    with import_actions:
        st.markdown(
            '<div class="template-action-spacer"></div>',
            unsafe_allow_html=True,
        )
        template_columns = st.columns(2)
        with template_columns[0]:
            st.download_button(
                "⇩ Excel模板",
                data=build_excel_template().getvalue(),
                file_name="装配式地铁车站_构件导入模板.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with template_columns[1]:
            st.download_button(
                "⇩ CSV模板",
                data=template_frame().to_csv(index=False).encode("utf-8-sig"),
                file_name="装配式地铁车站_构件导入模板.csv",
                mime="text/csv",
                use_container_width=True,
            )

    uploaded_file = st.file_uploader(
        "导入构件数据",
        type=["csv", "xlsx"],
        key="component_import_file",
    )

    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        signature = hashlib.sha256(
            uploaded_file.name.encode("utf-8") + raw_bytes
        ).hexdigest()
        if (
            st.session_state.get("import_signature") != signature
            or st.session_state.get("imported_validation") is None
        ):
            try:
                uploaded_file.seek(0)
                import_bundle = read_import_file(uploaded_file)
                validation = validate_import_bundle(import_bundle)
                st.session_state.import_signature = signature
                st.session_state.imported_validation = validation
            except Exception as error:
                st.error(f"数据文件读取失败：{error}")
                validation = None
        else:
            validation = st.session_state.get("imported_validation")

        if validation is not None:
            summary = validation.summary
            summary_columns = st.columns(4)
            summary_columns[0].metric("数据行数", summary["total_rows"])
            summary_columns[1].metric("有效构件", summary["valid_rows"])
            summary_columns[2].metric("错误项", summary["error_count"])
            summary_columns[3].metric("警告项", summary["warning_count"])

            if not validation.errors.empty:
                st.error("数据校验未通过，请按错误报告修正后重新上传。")
                st.dataframe(
                    validation.errors,
                    use_container_width=True,
                    hide_index=True,
                    height=min(420, 42 + len(validation.errors) * 36),
                )
            if not validation.warnings.empty:
                with st.expander(
                    f"警告信息（{len(validation.warnings)} 项）",
                    expanded=False,
                ):
                    st.dataframe(
                        validation.warnings,
                        use_container_width=True,
                        hide_index=True,
                    )

            if validation.is_valid and validation.components is not None:
                normalized_view = components_to_frame(validation.components)
                st.success(
                    f"校验通过：{len(validation.components)} 件构件，"
                    f"总起吊重量 {summary['total_weight_t']:,.2f} t，"
                    f"前置约束 {summary['dependency_count']} 条。"
                )
                if validation.project:
                    clean_project_parameters = normalize_project_metadata(
                        validation.project
                    )
                    project_frame = pd.DataFrame(
                        [
                            {
                                "参数": display_text(key),
                                "值": display_text(value),
                            }
                            for key, value in clean_project_parameters.items()
                            if not str(key).startswith("_")
                        ]
                    )
                    with st.expander("项目参数", expanded=False):
                        st.dataframe(
                            project_frame,
                            use_container_width=True,
                            hide_index=True,
                        )
                if not validation.relations.empty:
                    with st.expander(
                        f"构件关系（{len(validation.relations)} 条）",
                        expanded=False,
                    ):
                        st.dataframe(
                            validation.relations,
                            use_container_width=True,
                            hide_index=True,
                        )
                if validation.equipment:
                    equipment_frame = pd.DataFrame(
                        [
                            {
                                "设备编号": equipment["id"],
                                "设备类型": equipment["type"],
                                "额定能力(t)": equipment["capacity_t"],
                                "轨道起点X": equipment["rail_start_x"],
                                "轨道终点X": equipment["rail_end_x"],
                                "小车最小Y": equipment["trolley_min_y"],
                                "小车最大Y": equipment["trolley_max_y"],
                                "最大起升高度(m)": equipment["max_lift_height"],
                                "台班费率(元/h)": equipment["rate_yuan_h"],
                            }
                            for equipment in validation.equipment
                        ]
                    )
                    with st.expander(
                        f"设备参数（{len(validation.equipment)} 台）",
                        expanded=False,
                    ):
                        st.dataframe(
                            equipment_frame,
                            use_container_width=True,
                            hide_index=True,
                        )
                st.dataframe(
                    normalized_view.head(80),
                    use_container_width=True,
                    hide_index=True,
                    height=360,
                )
                if st.button(
                    "应用导入数据并重新优化",
                    type="primary",
                    use_container_width=True,
                ):
                    imported_project_name = display_text(
                        validation.project.get("项目名称"),
                        uploaded_file.name,
                    )
                    new_config = st.session_state.active_config.copy()
                    new_config.update(
                        {
                            "project_source": "导入项目",
                            "data_source": "导入数据",
                            "project_name": imported_project_name,
                            "dataset_label": imported_project_name,
                            "component_count": len(validation.components or []),
                            "length_m": numeric_value(
                                validation.project.get("长度"),
                                st.session_state.active_config["length_m"],
                            ),
                            "height_m": numeric_value(
                                validation.project.get("高度"),
                                st.session_state.active_config["height_m"],
                            ),
                            "width_m": numeric_value(
                                validation.project.get("宽度"),
                                st.session_state.active_config["width_m"],
                            ),
                            "crane_count": len(validation.equipment)
                            or int(
                                st.session_state.active_config["crane_count"]
                            ),
                            "base_capacity_t": numeric_value(
                                validation.equipment[0].get("capacity_t")
                                if validation.equipment
                                else None,
                                st.session_state.active_config[
                                    "base_capacity_t"
                                ],
                            ),
                            "data_revision": int(
                                st.session_state.active_config.get("data_revision", 0)
                            )
                            + 1,
                        }
                    )
                    st.session_state.imported_components = validation.components
                    st.session_state.imported_source_name = uploaded_file.name
                    st.session_state.imported_equipment = validation.equipment
                    st.session_state.imported_project = normalize_project_metadata(
                        validation.project
                    )
                    st.session_state.selected_project_source = "导入项目"
                    st.session_state.imported_relations = validation.relations
                    progress = OverlayProgress("正在使用导入构件初始化种群...")
                    result, cranes, elapsed = run_optimization_with_progress(
                        new_config,
                        progress,
                        components=validation.components,
                        equipment_config=validation.equipment or None,
                    )
                    progress.empty()
                    st.session_state.result = result
                    st.session_state.cranes = cranes
                    st.session_state.active_config = new_config
                    st.session_state.applied_config = dict(new_config)
                    st.session_state.last_run_summary = {
                        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "elapsed_seconds": float(elapsed),
                        "pareto_size": int(result.metadata["pareto_size"]),
                        "project_name": imported_project_name,
                        "component_count": len(validation.components or []),
                        "crane_count": len(cranes),
                    }
                    st.session_state.elapsed_seconds = elapsed
                    st.rerun()

    elif st.session_state.get("imported_components"):
        imported_components = st.session_state.imported_components
        st.info(
            f"已载入导入数据：{st.session_state.imported_source_name}，"
            f"共 {len(imported_components)} 件构件。"
        )

    st.markdown(
        '<div class="section-label">当前活动模型构件</div>',
        unsafe_allow_html=True,
    )
    component_frame = components_to_frame(components)
    left_filter, right_filter = st.columns([1.0, 2.0])
    with left_filter:
        type_options = [spec["label"] for spec in TYPE_SPECS.values()]
        selected_types = st.multiselect(
            "几何族",
            type_options,
            default=type_options,
        )
    with right_filter:
        source_label = (
            "导入数据"
            if active_config.get("data_source") == "导入数据"
            else "参数化场景"
        )
        st.text_input("当前数据源", value=source_label, disabled=True)
    filtered_frame = component_frame[
        component_frame["几何族"].isin(selected_types)
    ].copy()
    st.dataframe(
        filtered_frame,
        use_container_width=True,
        hide_index=True,
            height=260 if presentation_mode else 470,
    )
    bom_csv = component_frame.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⇩ 导出当前 BOM 清单",
        data=bom_csv,
        file_name=f"{active_config.get('dataset_label', active_config['station_name'])}_BOM清单.csv",
        mime="text/csv",
    )
