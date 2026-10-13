from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .domain import Component, TYPE_SPECS


COLORS = {
    "ink": "#cbd5e1",
    "muted": "#94a3b8",
    "paper": "#1e2534",
    "grid": "#334155",
    "orange": "#f97316",
    "amber": "#eab308",
    "teal": "#2563eb",
    "green": "#22c55e",
    "red": "#ef4444",
    "blue": "#2563eb",
    "violet": "#a855f7",
}

TYPE_COLORS = {
    "bottom_slab": "#2563eb",
    "column": "#f97316",
    "side_wall": "#22c55e",
    "beam": "#a855f7",
    "middle_slab": "#0ea5e9",
    "platform_slab": "#ec4899",
    "roof_slab": "#eab308",
    "stair": "#94a3b8",
}

PLAN_COLORS = {
    "A": "#2563eb",
    "B": "#f97316",
    "C": "#22c55e",
    "D": "#a855f7",
}


def _plan_color(plan_name: str, fallback: str = "#2563eb") -> str:
    prefix = str(plan_name).strip()[:1].upper()
    return PLAN_COLORS.get(prefix, fallback)


def _base_layout(height: int = 430, margin: dict | None = None) -> dict:
    return {
        "height": height,
        "margin": margin or {"l": 18, "r": 18, "t": 42, "b": 18},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Microsoft YaHei, Noto Sans CJK SC, sans-serif", "color": COLORS["ink"]},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11},
            "bgcolor": "rgba(30,37,52,0.88)",
            "bordercolor": "rgba(51,65,85,0.9)",
            "borderwidth": 1,
        },
        "hoverlabel": {
            "bgcolor": "#1e2534",
            "font": {"color": "#ffffff", "size": 12},
            "bordercolor": "#334155",
        },
    }


BOX_VERTEX_SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=float,
)

BOX_FACES = np.asarray(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ],
    dtype=int,
)

LAYER_FILTERS = {
    "全部构件": set(TYPE_SPECS),
    "基础与底板": {"bottom_slab"},
    "中部结构与中板": {"column", "side_wall", "beam", "middle_slab", "platform_slab"},
    "上部结构": {"roof_slab", "stair"},
}

STATUS_STYLES = {
    "已完成": {"color": "#2563eb", "opacity": 1.0},
    "当前吊装": {"color": "#f97316", "opacity": 1.0},
    "下一件": {"color": "#22c55e", "opacity": 0.96},
    "未安装": {"color": "#94a3b8", "opacity": 0.68},
}


def build_site_figure(
    components: Sequence[Component],
    plan: dict,
    cranes: Sequence[dict],
    color_mode: str = "安装状态",
    layer_filter: str = "全部构件",
    progress_percent: float = 0.0,
    camera_preset: str = "45°轴测",
) -> go.Figure:
    schedule = plan["schedule"]
    order_map = {row["component_index"]: row for row in schedule}
    allowed_types = LAYER_FILTERS.get(layer_filter, set(TYPE_SPECS))
    visible_indices = [
        index
        for index, component in enumerate(components)
        if component.type in allowed_types
    ]
    if not visible_indices:
        visible_indices = list(range(len(components)))

    clamped_progress = float(np.clip(progress_percent, 0.0, 100.0))
    completed_count = int(clamped_progress / 100.0 * len(schedule))
    status_by_index: dict[int, str] = {}
    for index in visible_indices:
        if clamped_progress <= 0.0:
            status_by_index[index] = "隐藏"
            continue
        order = int(order_map.get(index, {}).get("order", len(schedule) + 1))
        if order <= completed_count:
            status_by_index[index] = "已完成"
        elif order == completed_count + 1:
            status_by_index[index] = "当前吊装"
        elif order == completed_count + 2:
            status_by_index[index] = "下一件"
        else:
            status_by_index[index] = "未安装"

    rendered_indices = [
        index
        for index in visible_indices
        if status_by_index[index] in {"已完成", "当前吊装", "下一件"}
    ]
    grouped: dict[tuple[str, str], dict] = {}
    for index in rendered_indices:
        component = components[index]
        status = status_by_index[index]
        label, color, opacity = _component_visual_style(
            component=component,
            status=status,
            task=order_map.get(index, {}),
            color_mode=color_mode,
        )
        group = grouped.setdefault(
            (label, color),
            {"indices": [], "opacity": opacity, "label": label, "color": color},
        )
        group["indices"].append(index)

    figure = go.Figure()
    min_x, max_x, min_y, max_y = _component_bounds(components)
    min_z = min(component.z - component.height_m * 0.5 for component in components)
    max_z = max(component.z + component.height_m * 0.5 for component in components)
    section_context = _station_section_context(
        components=components,
        min_y=min_y,
        max_y=max_y,
        max_z=max_z,
    )
    figure.add_trace(
        go.Surface(
            x=[min_x - 4.0, max_x + 4.0],
            y=[min_y - 3.0, max_y + 3.0],
            z=[[min_z, min_z], [min_z, min_z]],
            colorscale=[[0.0, "#d8d3c6"], [1.0, "#d8d3c6"]],
            opacity=0.16,
            showscale=False,
            hoverinfo="skip",
            name="施工场地",
        )
    )

    for group_key, group in grouped.items():
        label, color = group_key
        figure.add_trace(
            _component_mesh_trace(
                components=components,
                indices=group["indices"],
                order_map=order_map,
                status_by_index=status_by_index,
                color=color,
                opacity=group["opacity"],
                name=label,
                section_context=section_context,
            )
        )

    marker_indices = [
        index
        for index in visible_indices
        if index not in rendered_indices
    ]
    if marker_indices:
        figure.add_trace(
            _component_position_trace(
                components=components,
                indices=marker_indices,
                order_map=order_map,
            )
        )

    arch_guide = (
        _arch_guide_trace(components, section_context)
        if "roof_slab" in allowed_types
        else None
    )
    if arch_guide is not None:
        figure.add_trace(arch_guide)

    stair_detail = (
        _stair_detail_trace(components)
        if "stair" in allowed_types
        and any(components[index].type == "stair" for index in rendered_indices)
        else None
    )
    if stair_detail is not None:
        figure.add_trace(stair_detail)

    for index in rendered_indices:
        status = status_by_index[index]
        if status in {"当前吊装", "下一件"}:
            figure.add_trace(
                _component_outline_trace(
                    components[index],
                    color=COLORS["orange"] if status == "当前吊装" else COLORS["amber"],
                    name=status,
                    showlegend=False,
                    section_context=section_context,
                )
            )

    completed_indices = [
        index
        for index in rendered_indices
        if status_by_index[index] == "已完成"
    ]
    recent_completed_indices = sorted(
        completed_indices,
        key=lambda index: int(order_map.get(index, {}).get("order", 0)),
        reverse=True,
    )[:12]
    if recent_completed_indices:
        figure.add_trace(
            _component_group_outline_trace(
                components=components,
                indices=recent_completed_indices,
                color="#0b6764",
                name="最近安装构件轮廓",
                section_context=section_context,
                line_width=2,
                opacity=0.62,
            )
        )

    current_index = next(
        (
            index
            for index in rendered_indices
            if status_by_index[index] == "当前吊装"
        ),
        None,
    )
    for crane in cranes:
        crane_x = float(crane.get("initial_x", crane.get("position", (0.0, 0.0))[0]))
        rail_start = float(crane.get("rail_start_x", min_x - 8.0))
        rail_end = float(crane.get("rail_end_x", max_x + 8.0))
        trolley_min = float(crane.get("trolley_min_y", min_y - 2.0))
        trolley_max = float(crane.get("trolley_max_y", max_y + 2.0))
        gantry_height = max(8.0, min(max_z + 5.0, 18.0))
        figure.add_trace(
            go.Scatter3d(
                x=[rail_start, rail_end, None, rail_start, rail_end],
                y=[
                    trolley_min,
                    trolley_min,
                    None,
                    trolley_max,
                    trolley_max,
                ],
                z=[0.08, 0.08, None, 0.08, 0.08],
                mode="lines",
                line={"color": "#5b6761", "width": 3},
                hoverinfo="skip",
                showlegend=False,
                name=f"{crane['id']} 轨道",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=[
                    crane_x,
                    crane_x,
                    None,
                    crane_x,
                    crane_x,
                    None,
                    crane_x,
                    crane_x,
                ],
                y=[
                    trolley_min,
                    trolley_min,
                    None,
                    trolley_max,
                    trolley_max,
                    None,
                    trolley_min,
                    trolley_max,
                ],
                z=[
                    0.0,
                    gantry_height,
                    None,
                    0.0,
                    gantry_height,
                    None,
                    gantry_height,
                    gantry_height,
                ],
                mode="lines",
                line={"color": "#3f4c47", "width": 5},
                hoverinfo="skip",
                showlegend=False,
                name=f"{crane['id']} 门架",
            )
        )

    if current_index is not None:
        current_task = order_map[current_index]
        current_component = components[current_index]
        current_crane = next(
            (
                crane
                for crane in cranes
                if crane["id"] == current_task.get("crane")
            ),
            None,
        )
        if current_crane is not None:
            figure.add_trace(
                go.Scatter3d(
                    x=[current_component.x, current_component.x],
                    y=[current_component.y, current_component.y],
                    z=[12.0, current_component.z],
                    mode="lines+markers",
                    line={"color": COLORS["orange"], "width": 6, "dash": "dash"},
                    marker={"size": [4, 7], "color": COLORS["orange"], "symbol": "diamond"},
                    hovertemplate=(
                        f"<b>{current_component.id}</b><br>"
                        f"当前吊装构件<br>设备 {current_crane['id']}<extra></extra>"
                    ),
                    showlegend=False,
                    name="当前吊装路径",
                )
            )

    figure.add_trace(
        go.Scatter3d(
            x=[
                float(crane.get("initial_x", crane.get("position", (0.0, 0.0))[0]))
                for crane in cranes
            ],
            y=[
                float(crane.get("initial_y", crane.get("position", (0.0, 0.0))[1]))
                for crane in cranes
            ],
            z=[0.7 for _ in cranes],
            mode="markers+text",
            marker={
                "size": 8,
                "symbol": "diamond",
                "color": "#3f4c47",
                "line": {"color": COLORS["amber"], "width": 2},
            },
            text=[crane["id"] for crane in cranes],
            textposition="top center",
            textfont={"color": COLORS["ink"], "size": 11},
            hovertemplate=(
                "<b>%{text}</b><br>"
                "额定能力 %{customdata[0]:.0f} t<br>"
                "台时单价 %{customdata[1]:.0f} 元/h<br>"
                "轨道长度 %{customdata[2]:.1f} m<br>"
                "小车行程 %{customdata[3]:.1f} m<extra></extra>"
            ),
            customdata=[
                [
                    crane["capacity_t"],
                    crane["rate_yuan_h"],
                    crane.get("rail_end_x", max_x + 8.0)
                    - crane.get("rail_start_x", min_x - 8.0),
                    crane.get("trolley_max_y", max_y + 2.0)
                    - crane.get("trolley_min_y", min_y - 2.0),
                ]
                for crane in cranes
            ],
            name="龙门吊",
        )
    )

    camera_eye = {
        "45°轴测": {"x": 0.90, "y": -1.80, "z": 0.85},
        "俯视": {"x": 0.01, "y": -0.01, "z": 2.15},
        "端部视图": {"x": 2.05, "y": 0.0, "z": 0.72},
        "侧视": {"x": 0.0, "y": -2.05, "z": 0.72},
    }.get(camera_preset, {"x": 1.45, "y": -1.55, "z": 1.05})
    camera_aspect = {
        "45°轴测": {"x": 1.00, "y": 0.72, "z": 0.52},
        "俯视": {"x": 1.00, "y": 0.90, "z": 0.24},
        "端部视图": {"x": 0.44, "y": 1.00, "z": 0.76},
        "侧视": {"x": 1.00, "y": 0.48, "z": 0.58},
    }.get(camera_preset, {"x": 1.00, "y": 0.72, "z": 0.52})
    crane_x_values = [
        value
        for crane in cranes
        for value in (
            float(crane.get("initial_x", crane.get("position", (0.0, 0.0))[0])),
            float(crane.get("rail_start_x", min_x - 8.0)),
            float(crane.get("rail_end_x", max_x + 8.0)),
        )
    ]
    crane_y_values = [
        value
        for crane in cranes
        for value in (
            float(crane.get("trolley_min_y", min_y - 2.0)),
            float(crane.get("trolley_max_y", max_y + 2.0)),
        )
    ]
    view_min_x = min([min_x, *crane_x_values]) - 5.0
    view_max_x = max([max_x, *crane_x_values]) + 5.0
    view_min_y = min([min_y, *crane_y_values]) - 4.0
    view_max_y = max([max_y, *crane_y_values]) + 4.0
    figure.update_layout(
        **_base_layout(height=600),
        scene={
            "xaxis": {
                "title": "纵向里程 (m)",
                "gridcolor": "rgba(51,65,85,0.55)",
                "backgroundcolor": "rgba(30,37,52,0.65)",
                "showbackground": True,
                "range": [view_min_x, view_max_x],
            },
            "yaxis": {
                "title": "横向轴线 (m)",
                "gridcolor": "rgba(51,65,85,0.55)",
                "backgroundcolor": "rgba(30,37,52,0.65)",
                "showbackground": True,
                "range": [view_min_y, view_max_y],
            },
            "zaxis": {
                "title": "安装标高 (m)",
                "gridcolor": "rgba(51,65,85,0.55)",
                "backgroundcolor": "rgba(30,37,52,0.35)",
                "showbackground": True,
                "range": [min(-1.0, min_z - 1.0), max_z + 4.0],
            },
            "bgcolor": "rgba(20,24,34,0.65)",
            "camera": {"eye": camera_eye, "up": {"x": 0.0, "y": 0.0, "z": 1.0}},
            "aspectmode": "manual",
            "aspectratio": camera_aspect,
        },
    )
    return figure


def _arch_guide_trace(
    components: Sequence[Component],
    section_context: dict[str, float],
) -> go.Scatter3d | None:
    roof_components = [
        component for component in components if component.type == "roof_slab"
    ]
    if not roof_components:
        return None
    roof_x_values = sorted({round(component.x, 5) for component in roof_components})
    if len(roof_x_values) > 8:
        selected_indices = np.linspace(0, len(roof_x_values) - 1, 8, dtype=int)
        roof_x_values = [roof_x_values[index] for index in selected_indices]
    station_width = section_context["station_width"]
    center_y = section_context["center_y"]
    arch_base_z = section_context["arch_base_z"]
    arch_rise = section_context["arch_rise"]
    y_values = np.linspace(center_y - station_width * 0.5, center_y + station_width * 0.5, 31)
    normalized_y = np.clip(
        (y_values - center_y) / max(station_width * 0.5, 0.1),
        -1.0,
        1.0,
    )
    z_values = arch_base_z + arch_rise * (1.0 - normalized_y**2)
    x_line: list[float | None] = []
    y_line: list[float | None] = []
    z_line: list[float | None] = []
    for x_value in roof_x_values:
        x_line.extend([x_value] * len(y_values))
        x_line.append(None)
        y_line.extend(y_values.tolist())
        y_line.append(None)
        z_line.extend(z_values.tolist())
        z_line.append(None)
    return go.Scatter3d(
        x=x_line,
        y=y_line,
        z=z_line,
        mode="lines",
        line={"color": "#56625c", "width": 4},
        opacity=0.68,
        hoverinfo="skip",
        name="拱形控制线",
        showlegend=False,
    )


def _stair_detail_trace(
    components: Sequence[Component],
) -> go.Scatter3d | None:
    stairs = [component for component in components if component.type == "stair"]
    if not stairs:
        return None
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for stair in stairs:
        step_count = max(12, min(36, int(np.ceil(stair.height_m / 0.18))))
        run = stair.length_m / step_count
        rise = stair.height_m / step_count
        bottom_z = stair.z - stair.height_m * 0.5
        left_y = stair.y - stair.width_m * 0.5
        right_y = stair.y + stair.width_m * 0.5
        for step_index in range(step_count + 1):
            x_value = stair.x - stair.length_m * 0.5 + step_index * run
            z_value = bottom_z + step_index * rise
            x_values.extend([x_value, x_value, None])
            y_values.extend([left_y, right_y, None])
            z_values.extend([z_value, z_value, None])
        side_path_x: list[float | None] = []
        side_path_z: list[float | None] = []
        for step_index in range(step_count):
            start_x = stair.x - stair.length_m * 0.5 + step_index * run
            end_x = start_x + run
            start_z = bottom_z + step_index * rise
            end_z = bottom_z + (step_index + 1) * rise
            side_path_x.extend([start_x, start_x, end_x, None])
            side_path_z.extend([start_z, end_z, end_z, None])
        for side_y in (left_y, right_y):
            x_values.extend(side_path_x)
            y_values.extend(
                [
                    side_y if value is not None else None
                    for value in side_path_x
                ]
            )
            z_values.extend(side_path_z)
    return go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="lines",
        line={"color": "#67716b", "width": 3},
        opacity=0.74,
        hoverinfo="skip",
        name="楼梯踏步控制线",
        showlegend=False,
    )


def _component_bounds(components: Sequence[Component]) -> tuple[float, float, float, float]:
    return (
        min(component.x - component.length_m * 0.5 for component in components),
        max(component.x + component.length_m * 0.5 for component in components),
        min(component.y - component.width_m * 0.5 for component in components),
        max(component.y + component.width_m * 0.5 for component in components),
    )


def _component_visual_style(
    component: Component,
    status: str,
    task: dict,
    color_mode: str,
) -> tuple[str, str, float]:
    if color_mode == "构件类型":
        label = component.semantic_type or TYPE_SPECS[component.type]["label"]
        color = TYPE_COLORS[component.type]
        opacity = 1.0 if status == "当前吊装" else (0.96 if status != "未安装" else 0.62)
        if status == "下一件":
            color = COLORS["amber"]
        return label, color, opacity
    if color_mode == "风险等级":
        risk = float(task.get("risk", component.risk * 100.0))
        if risk >= 50.0:
            label, color = "高风险", COLORS["red"]
        elif risk >= 35.0:
            label, color = "较高风险", COLORS["orange"]
        elif risk >= 20.0:
            label, color = "中等风险", COLORS["amber"]
        else:
            label, color = "低风险", COLORS["green"]
        opacity = 1.0 if status == "当前吊装" else (0.96 if status != "未安装" else 0.62)
        return label, color, opacity
    style = STATUS_STYLES[status]
    return status, style["color"], style["opacity"]


def _component_position_trace(
    components: Sequence[Component],
    indices: Sequence[int],
    order_map: dict[int, dict],
) -> go.Scatter3d:
    colors = [TYPE_COLORS[components[index].type] for index in indices]
    sizes = [
        max(3.5, min(7.0, components[index].weight_t * 0.28))
        for index in indices
    ]
    hover = []
    for index in indices:
        component = components[index]
        task = order_map.get(index, {})
        hover.append(
            f"<b>{component.id}</b><br>"
            f"{component.semantic_type or TYPE_SPECS[component.type]['label']} · {component.name}<br>"
            f"计划吊装序号 {int(task.get('order', 0))}<br>"
            f"重量 {component.weight_t:.1f} t<br>"
            f"定位 {component.x:.1f}, {component.y:.1f}, {component.z:.1f} m"
        )
    return go.Scatter3d(
        x=[components[index].x for index in indices],
        y=[components[index].y for index in indices],
        z=[components[index].z for index in indices],
        mode="markers",
        marker={
            "size": sizes,
            "color": colors,
            "symbol": "diamond",
            "opacity": 0.60,
            "line": {"color": "#ffffff", "width": 0.7},
        },
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name="待安装定位",
        showlegend=True,
    )


def _component_mesh_trace(
    components: Sequence[Component],
    indices: Sequence[int],
    order_map: dict[int, dict],
    status_by_index: dict[int, str],
    color: str,
    opacity: float,
    name: str,
    section_context: dict[str, float],
) -> go.Mesh3d:
    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []
    i_values: list[int] = []
    j_values: list[int] = []
    k_values: list[int] = []
    face_colors: list[str] = []
    hover_values: list[str] = []

    for index in indices:
        component = components[index]
        vertices, geometry_faces = _component_geometry(component, section_context)
        offset = len(x_values)
        component_color = _shade_hex_color(
            color,
            factor=0.90 + ((index * 37) % 7) * 0.025,
        )
        x_values.extend(vertices[:, 0])
        y_values.extend(vertices[:, 1])
        z_values.extend(vertices[:, 2])
        i_values.extend(geometry_faces[:, 0] + offset)
        j_values.extend(geometry_faces[:, 1] + offset)
        k_values.extend(geometry_faces[:, 2] + offset)
        face_colors.extend([component_color] * len(geometry_faces))
        task = order_map.get(index, {})
        hover = (
            f"<b>{component.id}</b><br>"
            f"{TYPE_SPECS[component.type]['label']} · {component.name}<br>"
            f"吊装序号 {int(task.get('order', 0))}<br>"
            f"状态 {status_by_index.get(index, '未安装')}<br>"
            f"尺寸 {component.length_m:.2f} × {component.width_m:.2f} × {component.height_m:.2f} m<br>"
            f"重量 {component.weight_t:.1f} t<br>"
            f"设备 {task.get('crane', '待分配')}<br>"
            f"综合风险 {float(task.get('risk', component.risk * 100.0)):.1f}"
        )
        hover_values.extend([hover] * len(vertices))

    return go.Mesh3d(
        x=x_values,
        y=y_values,
        z=z_values,
        i=i_values,
        j=j_values,
        k=k_values,
        color=color,
        facecolor=face_colors,
        opacity=opacity,
        flatshading=True,
        text=hover_values,
        hovertemplate="%{text}<extra></extra>",
        lighting={
            "ambient": 0.88,
            "diffuse": 0.68,
            "specular": 0.10,
            "roughness": 0.92,
            "fresnel": 0.05,
        },
        lightposition={"x": 100.0, "y": -180.0, "z": 180.0},
        name=name,
        showlegend=True,
    )


def _component_outline_trace(
    component: Component,
    color: str,
    name: str,
    showlegend: bool,
    section_context: dict[str, float],
) -> go.Scatter3d:
    vertices, faces = _component_geometry(component, section_context)
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for first, second in _geometry_edges(faces):
        x_values.extend([vertices[first, 0], vertices[second, 0], None])
        y_values.extend([vertices[first, 1], vertices[second, 1], None])
        z_values.extend([vertices[first, 2], vertices[second, 2], None])
    return go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="lines",
        line={"color": color, "width": 6},
        hoverinfo="skip",
        name=name,
        showlegend=showlegend,
    )


def _component_group_outline_trace(
    components: Sequence[Component],
    indices: Sequence[int],
    color: str,
    name: str,
    section_context: dict[str, float],
    line_width: int,
    opacity: float,
) -> go.Scatter3d:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for index in indices:
        vertices, faces = _component_geometry(components[index], section_context)
        for first, second in _geometry_edges(faces):
            x_values.extend([vertices[first, 0], vertices[second, 0], None])
            y_values.extend([vertices[first, 1], vertices[second, 1], None])
            z_values.extend([vertices[first, 2], vertices[second, 2], None])
    return go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="lines",
        line={"color": color, "width": line_width},
        opacity=opacity,
        hoverinfo="skip",
        name=name,
        showlegend=False,
    )


def _station_section_context(
    components: Sequence[Component],
    min_y: float,
    max_y: float,
    max_z: float,
) -> dict[str, float]:
    station_width = max(max_y - min_y, 1.0)
    center_y = (min_y + max_y) * 0.5
    roof_components = [
        component for component in components if component.type == "roof_slab"
    ]
    roof_center_z = (
        float(np.median([component.z for component in roof_components]))
        if roof_components
        else max_z
    )
    side_wall_components = [
        component for component in components if component.type == "side_wall"
    ]
    side_wall_top = (
        max(
            component.z + component.height_m * 0.5
            for component in side_wall_components
        )
        if side_wall_components
        else roof_center_z - max(1.0, station_width * 0.10)
    )
    bottom_components = [
        component for component in components if component.type == "bottom_slab"
    ]
    platform_components = [
        component for component in components if component.type == "platform_slab"
    ]
    floor_base_z = (
        float(np.median([component.z for component in bottom_components]))
        if bottom_components
        else 0.3
    )
    platform_center_z = (
        float(np.median([component.z for component in platform_components]))
        if platform_components
        else floor_base_z + max(0.65, station_width * 0.035)
    )
    nominal_arch_rise = max(
        roof_center_z - side_wall_top,
        station_width * 0.14,
    )
    arch_rise = float(
        np.clip(
            nominal_arch_rise,
            1.2,
            max(1.4, station_width * 0.22),
        )
    )
    return {
        "station_width": station_width,
        "center_y": center_y,
        "roof_center_z": roof_center_z,
        "arch_base_z": side_wall_top,
        "arch_crown_z": side_wall_top + arch_rise,
        "arch_rise": arch_rise,
        "wall_curve": max(0.25, min(station_width * 0.035, 1.1)),
        "floor_base_z": floor_base_z,
        "floor_rise": max(0.18, min(station_width * 0.028, 0.85)),
        "platform_center_z": platform_center_z,
    }


def _component_geometry(
    component: Component,
    section_context: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = _component_geometry_raw(component, section_context)
    center = np.asarray([component.x, component.y, component.z], dtype=float)
    scale = np.asarray([0.985, 0.985, 1.0], dtype=float)
    return center + (vertices - center) * scale, faces


def _component_geometry_raw(
    component: Component,
    section_context: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    length_m = max(component.length_m, 0.05)
    width_m = max(component.width_m, 0.05)
    height_m = max(component.height_m, 0.05)
    center_x, center_y, center_z = component.x, component.y, component.z
    section_context = section_context or {
        "station_width": width_m,
        "center_y": center_y,
        "roof_center_z": center_z,
        "arch_base_z": center_z - max(height_m, width_m * 0.18),
        "arch_crown_z": center_z,
        "arch_rise": max(height_m, width_m * 0.18),
        "wall_curve": 0.25,
        "floor_base_z": center_z,
        "floor_rise": max(0.10, min(width_m * 0.025, 0.40)),
        "platform_center_z": center_z + max(0.55, height_m),
    }

    if component.type == "bottom_slab":
        return _curved_floor_geometry(
            component,
            section_context,
            downward_convex=True,
        )

    if component.type == "column":
        chamfer = min(length_m, width_m) * 0.18
        profile = np.asarray(
            [
                [-length_m * 0.5 + chamfer, -width_m * 0.5],
                [length_m * 0.5 - chamfer, -width_m * 0.5],
                [length_m * 0.5, -width_m * 0.5 + chamfer],
                [length_m * 0.5, width_m * 0.5 - chamfer],
                [length_m * 0.5 - chamfer, width_m * 0.5],
                [-length_m * 0.5 + chamfer, width_m * 0.5],
                [-length_m * 0.5, width_m * 0.5 - chamfer],
                [-length_m * 0.5, -width_m * 0.5 + chamfer],
            ],
            dtype=float,
        )
        shaft = _extrude_profile(
            profile,
            axis=2,
            start=-height_m * 0.5,
            end=height_m * 0.5,
            origin=(center_x, center_y, center_z),
        )
        corbel = _box_geometry(
            center_x,
            center_y,
            center_z + height_m * 0.48,
            length_m * 1.22,
            width_m * 1.22,
            height_m * 0.06,
        )
        return _merge_geometries(shaft, corbel)

    if component.type == "side_wall":
        wall = _curved_side_wall_geometry(component, section_context)
        haunch = _box_geometry(
            center_x,
            center_y + width_m * 0.20,
            center_z - height_m * 0.34,
            length_m * 0.96,
            width_m * 1.22,
            height_m * 0.32,
        )
        return _merge_geometries(wall, haunch)

    if component.type == "beam":
        top_flange = _box_geometry(
            center_x,
            center_y,
            center_z - height_m * 0.38,
            length_m,
            width_m,
            height_m * 0.24,
        )
        web = _box_geometry(
            center_x,
            center_y,
            center_z - height_m * 0.03,
            length_m * 0.98,
            width_m * 0.34,
            height_m * 0.70,
        )
        return _merge_geometries(top_flange, web)

    if component.type == "middle_slab":
        top_plate = _box_geometry(
            center_x,
            center_y,
            center_z + height_m * 0.24,
            length_m,
            width_m,
            height_m * 0.46,
        )
        rib_geometries = [
            _box_geometry(
                center_x,
                center_y + offset_y,
                center_z - height_m * 0.17,
                length_m * 0.96,
                width_m * 0.10,
                height_m * 0.78,
            )
            for offset_y in (-width_m * 0.34, 0.0, width_m * 0.34)
        ]
        return _merge_geometries(top_plate, *rib_geometries)

    if component.type == "platform_slab":
        platform = _box_geometry(
            center_x,
            center_y,
            center_z,
            length_m,
            width_m,
            height_m,
        )
        center_y_reference = section_context.get("center_y", center_y)
        side_direction = -1.0 if center_y < center_y_reference else 1.0
        edge_width = max(0.18, width_m * 0.14)
        edge_y = center_y + side_direction * width_m * 0.42
        safety_edge = _box_geometry(
            center_x,
            edge_y,
            center_z + height_m * 0.62,
            length_m,
            edge_width,
            height_m * 0.72,
        )
        nosing = _box_geometry(
            center_x,
            edge_y + side_direction * edge_width * 0.72,
            center_z + height_m * 0.88,
            length_m,
            edge_width * 0.82,
            height_m * 0.34,
        )
        return _merge_geometries(platform, safety_edge, nosing)

    if component.type == "roof_slab":
        return _arched_roof_geometry(component, section_context)

    if component.type == "stair":
        return _stair_geometry(component)

    return _box_geometry(
        center_x,
        center_y,
        center_z,
        length_m,
        width_m,
        height_m,
    )


def _arched_roof_geometry(
    component: Component,
    section_context: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    length_m = max(component.length_m, 0.05)
    width_m = max(component.width_m, 0.05)
    thickness = max(component.height_m, 0.08)
    arch_rise = section_context["arch_rise"]
    center_y = section_context["center_y"]
    half_station_width = max(section_context["station_width"] * 0.5, width_m * 0.5)
    roof_center_z = section_context["roof_center_z"]
    point_count = 11
    cross_axis = np.linspace(-width_m * 0.5, width_m * 0.5, point_count)
    global_y = component.y + cross_axis
    normalized_y = np.clip(
        (global_y - center_y) / half_station_width,
        -1.0,
        1.0,
    )
    arch_curve = (
        section_context["arch_crown_z"]
        - arch_rise
        + arch_rise * (1.0 - normalized_y**2)
        + component.z
        - roof_center_z
    )
    return _curved_shell_geometry(
        component=component,
        y_values=global_y,
        center_z_values=arch_curve,
        thickness=thickness,
    )


def _stair_geometry(component: Component) -> tuple[np.ndarray, np.ndarray]:
    length_m = max(component.length_m, 0.5)
    width_m = max(component.width_m, 0.5)
    height_m = max(component.height_m, 0.5)
    step_count = max(12, min(36, int(np.ceil(height_m / 0.18))))
    run = length_m / step_count
    rise = height_m / step_count
    bottom_z = component.z - height_m * 0.5
    tread_thickness = min(0.08, rise * 0.34)
    riser_thickness = min(run * 0.16, 0.07)
    step_geometries: list[tuple[np.ndarray, np.ndarray]] = []

    for step_index in range(step_count):
        start_x = component.x - length_m * 0.5 + step_index * run
        top_z = bottom_z + (step_index + 1) * rise
        step_geometries.append(
            _box_geometry(
                start_x + run * 0.5,
                component.y,
                top_z - tread_thickness * 0.5,
                run * 1.04,
                width_m,
                tread_thickness,
            )
        )
        if step_index > 0:
            step_geometries.append(
                _box_geometry(
                    start_x,
                    component.y,
                    top_z - rise * 0.5,
                    riser_thickness,
                    width_m,
                    rise * 0.96,
                )
            )

    return _merge_geometries(*step_geometries)


def _curved_floor_geometry(
    component: Component,
    section_context: dict[str, float],
    downward_convex: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    width_m = max(component.width_m, 0.05)
    thickness = max(component.height_m, 0.08)
    point_count = 13
    cross_axis = np.linspace(-width_m * 0.5, width_m * 0.5, point_count)
    global_y = component.y + cross_axis
    half_station_width = max(section_context["station_width"] * 0.5, width_m * 0.5)
    normalized_y = np.clip(
        (global_y - section_context["center_y"]) / half_station_width,
        -1.0,
        1.0,
    )
    base_curve = component.z + section_context["floor_rise"] * (1.0 - normalized_y**2)
    if downward_convex:
        base_curve = (
            component.z
            - section_context["floor_rise"] * (1.0 - normalized_y**2)
        )
    return _curved_shell_geometry(
        component=component,
        y_values=global_y,
        center_z_values=base_curve,
        thickness=thickness,
    )


def _curved_shell_geometry(
    component: Component,
    y_values: np.ndarray,
    center_z_values: np.ndarray,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    length_m = max(component.length_m, 0.05)
    point_count = len(y_values)
    lower_curve = np.column_stack(
        [y_values, center_z_values - thickness * 0.5]
    )
    upper_curve = np.column_stack(
        [y_values, center_z_values + thickness * 0.5]
    )
    front_lower = np.column_stack(
        [
            np.full(point_count, component.x - length_m * 0.5),
            lower_curve,
        ]
    )
    front_upper = np.column_stack(
        [
            np.full(point_count, component.x - length_m * 0.5),
            upper_curve,
        ]
    )
    back_lower = np.column_stack(
        [
            np.full(point_count, component.x + length_m * 0.5),
            lower_curve,
        ]
    )
    back_upper = np.column_stack(
        [
            np.full(point_count, component.x + length_m * 0.5),
            upper_curve,
        ]
    )
    vertices = np.vstack([front_lower, front_upper, back_lower, back_upper])
    lower_front_offset = 0
    upper_front_offset = point_count
    lower_back_offset = point_count * 2
    upper_back_offset = point_count * 3
    faces: list[list[int]] = []
    for index in range(point_count - 1):
        faces.extend(
            [
                [
                    lower_front_offset + index,
                    lower_front_offset + index + 1,
                    upper_front_offset + index + 1,
                ],
                [
                    lower_front_offset + index,
                    upper_front_offset + index + 1,
                    upper_front_offset + index,
                ],
                [
                    lower_back_offset + index,
                    upper_back_offset + index + 1,
                    lower_back_offset + index + 1,
                ],
                [
                    lower_back_offset + index,
                    upper_back_offset + index,
                    upper_back_offset + index + 1,
                ],
                [
                    lower_front_offset + index,
                    lower_back_offset + index,
                    lower_back_offset + index + 1,
                ],
                [
                    lower_front_offset + index,
                    lower_back_offset + index + 1,
                    lower_front_offset + index + 1,
                ],
                [
                    upper_front_offset + index,
                    upper_front_offset + index + 1,
                    upper_back_offset + index + 1,
                ],
                [
                    upper_front_offset + index,
                    upper_back_offset + index + 1,
                    upper_back_offset + index,
                ],
            ]
        )
    faces.extend(
        [
            [
                lower_front_offset,
                upper_front_offset,
                upper_back_offset,
            ],
            [
                lower_front_offset,
                upper_back_offset,
                lower_back_offset,
            ],
            [
                lower_front_offset + point_count - 1,
                lower_back_offset + point_count - 1,
                upper_back_offset + point_count - 1,
            ],
            [
                lower_front_offset + point_count - 1,
                upper_back_offset + point_count - 1,
                upper_front_offset + point_count - 1,
            ],
        ]
    )
    return vertices, np.asarray(faces, dtype=int)


def _curved_side_wall_geometry(
    component: Component,
    section_context: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    length_m = max(component.length_m, 0.05)
    thickness = max(component.width_m, 0.05)
    height_m = max(component.height_m, 0.05)
    side_direction = 1.0 if component.y >= section_context["center_y"] else -1.0
    curvature = min(section_context["wall_curve"], height_m * 0.22)
    point_count = 9
    height_levels = np.linspace(0.0, 1.0, point_count)
    inward_shift = curvature * height_levels**1.65
    outer_y = (
        component.y
        + side_direction * thickness * 0.5
        - side_direction * inward_shift
    )
    inner_y = (
        component.y
        - side_direction * thickness * 0.5
        - side_direction * inward_shift
    )
    z_levels = component.z - height_m * 0.5 + height_levels * height_m

    front_outer = np.column_stack(
        [
            np.full(point_count, component.x - length_m * 0.5),
            outer_y,
            z_levels,
        ]
    )
    front_inner = np.column_stack(
        [
            np.full(point_count, component.x - length_m * 0.5),
            inner_y,
            z_levels,
        ]
    )
    back_outer = np.column_stack(
        [
            np.full(point_count, component.x + length_m * 0.5),
            outer_y,
            z_levels,
        ]
    )
    back_inner = np.column_stack(
        [
            np.full(point_count, component.x + length_m * 0.5),
            inner_y,
            z_levels,
        ]
    )
    vertices = np.vstack([front_outer, front_inner, back_outer, back_inner])
    front_outer_offset = 0
    front_inner_offset = point_count
    back_outer_offset = point_count * 2
    back_inner_offset = point_count * 3
    faces: list[list[int]] = []
    for index in range(point_count - 1):
        faces.extend(
            [
                [
                    front_outer_offset + index,
                    front_outer_offset + index + 1,
                    front_inner_offset + index + 1,
                ],
                [
                    front_outer_offset + index,
                    front_inner_offset + index + 1,
                    front_inner_offset + index,
                ],
                [
                    back_outer_offset + index,
                    back_inner_offset + index + 1,
                    back_outer_offset + index + 1,
                ],
                [
                    back_outer_offset + index,
                    back_inner_offset + index,
                    back_inner_offset + index + 1,
                ],
                [
                    front_outer_offset + index,
                    back_outer_offset + index,
                    back_outer_offset + index + 1,
                ],
                [
                    front_outer_offset + index,
                    back_outer_offset + index + 1,
                    front_outer_offset + index + 1,
                ],
                [
                    front_inner_offset + index,
                    front_inner_offset + index + 1,
                    back_inner_offset + index + 1,
                ],
                [
                    front_inner_offset + index,
                    back_inner_offset + index + 1,
                    back_inner_offset + index,
                ],
            ]
        )
    faces.extend(
        [
            [
                front_outer_offset,
                front_inner_offset,
                back_inner_offset,
            ],
            [
                front_outer_offset,
                back_inner_offset,
                back_outer_offset,
            ],
            [
                front_outer_offset + point_count - 1,
                back_outer_offset + point_count - 1,
                back_inner_offset + point_count - 1,
            ],
            [
                front_outer_offset + point_count - 1,
                back_inner_offset + point_count - 1,
                front_inner_offset + point_count - 1,
            ],
        ]
    )
    return vertices, np.asarray(faces, dtype=int)


def _box_geometry(
    center_x: float,
    center_y: float,
    center_z: float,
    length_m: float,
    width_m: float,
    height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray([center_x, center_y, center_z], dtype=float)
    half_dimensions = np.asarray(
        [length_m * 0.5, width_m * 0.5, height_m * 0.5],
        dtype=float,
    )
    return center + BOX_VERTEX_SIGNS * half_dimensions, BOX_FACES.copy()


def _extrude_profile(
    profile: np.ndarray,
    axis: int,
    start: float,
    end: float,
    origin: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    origin_array = np.asarray(origin, dtype=float)
    profile_count = len(profile)
    start_vertices = np.zeros((profile_count, 3), dtype=float)
    end_vertices = np.zeros((profile_count, 3), dtype=float)
    if axis == 0:
        start_vertices[:, 0] = start
        end_vertices[:, 0] = end
        start_vertices[:, 1] = profile[:, 0]
        end_vertices[:, 1] = profile[:, 0]
        start_vertices[:, 2] = profile[:, 1]
        end_vertices[:, 2] = profile[:, 1]
    elif axis == 1:
        start_vertices[:, 1] = start
        end_vertices[:, 1] = end
        start_vertices[:, 0] = profile[:, 0]
        end_vertices[:, 0] = profile[:, 0]
        start_vertices[:, 2] = profile[:, 1]
        end_vertices[:, 2] = profile[:, 1]
    else:
        start_vertices[:, 2] = start
        end_vertices[:, 2] = end
        start_vertices[:, 0] = profile[:, 0]
        end_vertices[:, 0] = profile[:, 0]
        start_vertices[:, 1] = profile[:, 1]
        end_vertices[:, 1] = profile[:, 1]
    vertices = np.vstack([start_vertices, end_vertices]) + origin_array
    faces: list[list[int]] = []
    for index in range(profile_count):
        next_index = (index + 1) % profile_count
        faces.extend(
            [
                [index, next_index, profile_count + next_index],
                [index, profile_count + next_index, profile_count + index],
            ]
        )
    for index in range(1, profile_count - 1):
        faces.append([0, index + 1, index])
        faces.append(
            [
                profile_count,
                profile_count + index,
                profile_count + index + 1,
            ]
        )
    return vertices, np.asarray(faces, dtype=int)


def _merge_geometries(
    *geometries: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for geometry_vertices, geometry_faces in geometries:
        vertices.append(geometry_vertices)
        faces.append(geometry_faces + vertex_offset)
        vertex_offset += len(geometry_vertices)
    if not vertices:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)
    return np.vstack(vertices), np.vstack(faces)


def _geometry_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        for first, second in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edges.add((min(first, second), max(first, second)))
    return sorted(edges)


def build_gantt_figure(schedule: Sequence[dict], max_tasks: int = 30) -> go.Figure:
    frame = pd.DataFrame(schedule)
    if frame.empty:
        return go.Figure()
    frame = frame.sort_values("start").head(max_tasks).sort_values("start", ascending=False)
    crane_colors = {
        crane: color
        for crane, color in zip(
            sorted(frame["crane"].unique()),
            [
                COLORS["blue"],
                COLORS["orange"],
                COLORS["violet"],
                COLORS["green"],
            ],
        )
    }
    figure = go.Figure()
    for crane, group in frame.groupby("crane", sort=True):
        figure.add_trace(
            go.Bar(
                x=group["duration"],
                y=group["component_id"],
                base=group["start"],
                orientation="h",
                marker={
                    "color": crane_colors.get(crane, COLORS["orange"]),
                    "line": {"color": "rgba(255,255,255,0.75)", "width": 0.5},
                },
                text=[f"{value:.0f} min" for value in group["duration"]],
                textposition="inside",
                insidetextanchor="middle",
                textfont={"size": 9, "color": "#ffffff"},
                customdata=np.stack(
                    [
                        group["component_type"],
                        group["weight_t"],
                        group["start"],
                        group["end"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>%{customdata[0]}<br>"
                    "重量 %{customdata[1]:.1f} t<br>"
                    "开始 %{customdata[2]:.1f} min<br>"
                    "完成 %{customdata[3]:.1f} min<extra></extra>"
                ),
                name=crane,
            )
        )
    figure.update_layout(
        **_base_layout(height=570, margin={"l": 112, "r": 24, "t": 48, "b": 40}),
        barmode="overlay",
        bargap=0.28,
        xaxis={
            "title": "作业时间 (min)",
            "gridcolor": COLORS["grid"],
            "zeroline": False,
        },
        yaxis={
            "title": "",
            "tickfont": {"size": 10},
            "automargin": True,
        },
        showlegend=True,
    )
    return figure


def build_pareto_figure(
    objectives: np.ndarray,
    plans: dict[str, dict],
) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.08,
        subplot_titles=("工期 - 成本", "工期 - 安全风险", "成本 - 安全风险"),
    )
    objective_frame = pd.DataFrame(objectives, columns=["工期", "成本", "风险"])
    normalized = (
        objective_frame - objective_frame.min()
    ) / (objective_frame.max() - objective_frame.min()).replace(0, 1)
    compromise = 0.4 * normalized["工期"] + 0.3 * normalized["成本"] + 0.3 * normalized["风险"]
    projections = [(0, 1), (0, 2), (1, 2)]
    axis_titles = [
        ("工期 (min)", "成本 (元)"),
        ("工期 (min)", "安全风险指数"),
        ("成本 (元)", "安全风险指数"),
    ]
    for position, ((horizontal, vertical), titles) in enumerate(
        zip(projections, axis_titles),
        start=1,
    ):
        figure.add_trace(
            go.Scatter(
                x=objectives[:, horizontal],
                y=objectives[:, vertical],
                mode="markers",
                marker={
                    "size": 5,
                    "color": "#94a3b8",
                    "opacity": 0.72,
                    "line": {"color": "#1e2534", "width": 0.35},
                },
                name="Pareto 解",
                showlegend=position == 1,
                hovertemplate=(
                    f"工期 {objectives[:, 0].min():.0f}-{objectives[:, 0].max():.0f} min<br>"
                    f"{titles[0]} %{{x:.2f}}<br>{titles[1]} %{{y:.2f}}<extra></extra>"
                ),
            ),
            row=1,
            col=position,
        )
        for plan_name, plan in plans.items():
            plan_objective = plan["objective"]
            figure.add_trace(
                go.Scatter(
                    x=[plan_objective[horizontal]],
                    y=[plan_objective[vertical]],
                    mode="markers+text",
                    marker={
                        "size": 14,
                        "symbol": "diamond",
                        "color": _plan_color(plan_name),
                        "line": {"color": "#ffffff", "width": 1.5},
                    },
                    text=[plan_name.split(" · ")[0]],
                    textposition="top center",
                    textfont={"size": 10, "color": "#f8fafc"},
                    name=plan_name,
                    showlegend=position == 1,
                    hovertemplate=(
                        f"<b>{plan_name}</b><br>"
                        f"工期 {{x:.1f}} min<br>成本 {{customdata[0]:,.0f}} 元<br>"
                        f"风险 {{customdata[1]:.2f}}<extra></extra>"
                    ),
                    customdata=[[plan_objective[1], plan_objective[2]]],
                ),
                row=1,
                col=position,
            )
        figure.update_xaxes(
            title_text=titles[0],
            gridcolor=COLORS["grid"],
            zeroline=False,
            row=1,
            col=position,
        )
        figure.update_yaxes(
            title_text=titles[1],
            gridcolor=COLORS["grid"],
            zeroline=False,
            row=1,
            col=position,
        )
    pareto_layout = _base_layout(
        height=460,
        margin={"l": 54, "r": 56, "t": 104, "b": 48},
    )
    pareto_layout["legend"] = {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.12,
        "xanchor": "center",
        "x": 0.5,
        "font": {"size": 11},
        "bgcolor": "rgba(30,37,52,0.88)",
        "bordercolor": "rgba(51,65,85,0.9)",
        "borderwidth": 1,
    }
    figure.update_layout(**pareto_layout)
    return figure


def build_convergence_figure(history: dict[str, list[float]]) -> go.Figure:
    generations = history["generation"]
    figure = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.10,
        subplot_titles=("最短工期", "最低成本", "最低安全风险"),
    )
    series = [
        ("best_time", "工期 (min)", COLORS["teal"]),
        ("best_cost", "成本 (元)", COLORS["orange"]),
        ("best_risk", "风险指数", COLORS["red"]),
    ]
    for position, (key, axis_title, color) in enumerate(series, start=1):
        figure.add_trace(
            go.Scatter(
                x=generations,
                y=history[key],
                mode="lines+markers",
                line={"color": color, "width": 2.6},
                marker={"size": 4},
                fill="tozeroy",
                fillcolor="rgba(37,99,235,0.12)" if position == 1 else "rgba(249,115,22,0.10)",
                hovertemplate="第 %{x} 代<br>%{y:.3f}<extra></extra>",
                name=axis_title,
                showlegend=False,
            ),
            row=1,
            col=position,
        )
        figure.update_xaxes(
            title_text="迭代代数",
            gridcolor=COLORS["grid"],
            zeroline=False,
            row=1,
            col=position,
        )
        figure.update_yaxes(
            title_text=axis_title,
            gridcolor=COLORS["grid"],
            zeroline=False,
            row=1,
            col=position,
        )
    figure.update_layout(**_base_layout(height=320, margin={"l": 52, "r": 22, "t": 48, "b": 42}))
    return figure


def build_plan_radar(plans: dict[str, dict]) -> go.Figure:
    frame = pd.DataFrame(
        {
            name: {
                "工期效率": plan["makespan_min"],
                "成本表现": plan["cost_yuan"],
                "安全表现": plan["risk_score"],
                "设备均衡": _crane_balance(plan),
                "执行稳定": plan["adjacent_crossings"],
            }
            for name, plan in plans.items()
        }
    ).T
    normalised = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        minimum, maximum = float(values.min()), float(values.max())
        if maximum <= minimum:
            normalised[column] = 100.0
        else:
            normalised[column] = 100.0 * (maximum - values) / (maximum - minimum)
    categories = list(normalised.columns)
    colors = [
        PLAN_COLORS["A"],
        PLAN_COLORS["B"],
        PLAN_COLORS["C"],
        PLAN_COLORS["D"],
    ]
    figure = go.Figure()
    for index, (plan_name, row) in enumerate(normalised.iterrows()):
        values = row.tolist()
        plan_color = _plan_color(plan_name, colors[index % len(colors)])
        figure.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor=_rgba(plan_color, 0.13),
                line={"color": plan_color, "width": 2.4},
                name=plan_name,
                hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
            )
        )
    figure.update_layout(
        **_base_layout(height=420, margin={"l": 44, "r": 44, "t": 62, "b": 30}),
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "gridcolor": COLORS["grid"],
                "tickfont": {"size": 9},
            },
            "angularaxis": {"gridcolor": COLORS["grid"]},
        },
    )
    return figure


def build_cost_figure(plan: dict) -> go.Figure:
    breakdown = plan["cost_breakdown"]
    labels = list(breakdown.keys())
    values = list(breakdown.values())
    figure = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.58,
            marker={
                "colors": [COLORS["blue"], COLORS["orange"], COLORS["green"]],
                "line": {"color": "#1e2534", "width": 2},
            },
            textinfo="label+percent",
            textfont={"size": 11},
            hovertemplate="%{label}<br>%{value:,.0f} 元<br>%{percent}<extra></extra>",
        )
    )
    figure.update_layout(**_base_layout(height=320, margin={"l": 16, "r": 16, "t": 24, "b": 16}))
    return figure


def build_risk_figure(components: Sequence[Component], plan: dict) -> go.Figure:
    schedule = plan["schedule"]
    top_items = sorted(schedule, key=lambda row: row["risk"] * row["weight_t"], reverse=True)[:12]
    labels = [item["component_id"] for item in reversed(top_items)]
    values = [item["risk"] for item in reversed(top_items)]
    colors = [COLORS["red"] if value >= 42 else COLORS["amber"] for value in values]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colors},
            text=[f"{value:.1f}" for value in values],
            textposition="outside",
            hovertemplate="%{y}<br>风险指数 %{x:.1f}<extra></extra>",
        )
    )
    figure.update_layout(
        **_base_layout(height=360, margin={"l": 112, "r": 32, "t": 24, "b": 34}),
        xaxis={"title": "综合风险指数", "range": [0, max(values) * 1.22], "gridcolor": COLORS["grid"]},
        yaxis={"title": ""},
        showlegend=False,
    )
    return figure


def build_risk_breakdown_figure(plan: dict) -> go.Figure:
    breakdown = plan.get("risk_breakdown", {})
    ordered = sorted(breakdown.items(), key=lambda item: item[1])
    labels = [item[0] for item in ordered]
    values = [item[1] for item in ordered]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={
                "color": [
                    COLORS["red"] if value >= 12 else COLORS["amber"]
                    for value in values
                ]
            },
            text=[f"{value:.1f}" for value in values],
            textposition="outside",
            hovertemplate="%{y}<br>风险贡献 %{x:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        **_base_layout(height=360, margin={"l": 92, "r": 36, "t": 30, "b": 34}),
        xaxis={
            "title": "风险贡献（满分 100）",
            "range": [0, max(max(values, default=1.0) * 1.30, 1.0)],
            "gridcolor": COLORS["grid"],
        },
        yaxis={"title": ""},
        showlegend=False,
    )
    return figure


def build_component_mix(components: Sequence[Component]) -> go.Figure:
    counts = pd.Series([component.type for component in components]).value_counts()
    labels = [TYPE_SPECS[key]["label"] for key in counts.index]
    colors = [TYPE_COLORS[key] for key in counts.index]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=counts.values,
            marker={"color": colors},
            text=counts.values,
            textposition="outside",
            hovertemplate="%{x}<br>%{y} 件<extra></extra>",
        )
    )
    figure.update_layout(
        **_base_layout(height=330, margin={"l": 32, "r": 22, "t": 24, "b": 50}),
        xaxis={"title": "", "tickangle": -20, "gridcolor": COLORS["grid"]},
        yaxis={"title": "构件数量", "gridcolor": COLORS["grid"]},
        showlegend=False,
    )
    return figure


def _crane_balance(plan: dict) -> float:
    workloads = np.asarray(
        [item["active_minutes"] for item in plan["crane_workload"]],
        dtype=float,
    )
    if len(workloads) == 0 or float(workloads.mean()) <= 0:
        return 0.0
    return float(workloads.std() / workloads.mean())


def _rgba(hex_color: str, alpha: float) -> str:
    clean = hex_color.lstrip("#")
    red, green, blue = (int(clean[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def _shade_hex_color(hex_color: str, factor: float) -> str:
    clean = hex_color.lstrip("#")
    red, green, blue = (int(clean[index : index + 2], 16) for index in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(
        int(np.clip(red * factor, 0, 255)),
        int(np.clip(green * factor, 0, 255)),
        int(np.clip(blue * factor, 0, 255)),
    )
