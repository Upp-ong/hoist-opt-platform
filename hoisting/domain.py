from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence
import heapq

import numpy as np
import pandas as pd


STATION_PRESETS = {
    "深圳地铁 14 号线示范站": {
        "code": "SZ14",
        "component_count": 204,
        "length_m": 126.0,
        "height_m": 12.4,
        "width_m": 24.8,
        "cranes": [
            {"id": "TC-01", "capacity_t": 160.0, "rate_yuan_h": 520.0, "position": (-12.0, 4.8)},
            {"id": "TC-02", "capacity_t": 130.0, "rate_yuan_h": 445.0, "position": (138.0, 20.0)},
        ],
        "target_days": 36.0,
    },
    "青岛地铁 6 号线可洛石站": {
        "code": "QD6",
        "component_count": 168,
        "length_m": 108.0,
        "height_m": 11.2,
        "width_m": 22.4,
        "cranes": [
            {"id": "TC-01", "capacity_t": 150.0, "rate_yuan_h": 500.0, "position": (-10.0, 4.0)},
            {"id": "TC-02", "capacity_t": 120.0, "rate_yuan_h": 420.0, "position": (118.0, 18.4)},
        ],
        "target_days": 32.0,
    },
    "长春地铁 2 号线袁家店站": {
        "code": "CC2",
        "component_count": 152,
        "length_m": 96.0,
        "height_m": 10.8,
        "width_m": 21.6,
        "cranes": [
            {"id": "TC-01", "capacity_t": 140.0, "rate_yuan_h": 475.0, "position": (-9.0, 3.6)},
            {"id": "TC-02", "capacity_t": 110.0, "rate_yuan_h": 405.0, "position": (105.0, 18.0)},
        ],
        "target_days": 30.0,
    },
}


TYPE_SPECS = {
    "bottom_slab": {
        "label": "底板",
        "share": 0.18,
        "weight_t": 18.4,
        "duration_min": 42.0,
        "risk": 0.16,
        "layer": 0,
    },
    "column": {
        "label": "预制立柱",
        "share": 0.20,
        "weight_t": 12.8,
        "duration_min": 35.0,
        "risk": 0.30,
        "layer": 1,
    },
    "side_wall": {
        "label": "侧墙",
        "share": 0.18,
        "weight_t": 14.2,
        "duration_min": 38.0,
        "risk": 0.34,
        "layer": 1,
    },
    "beam": {
        "label": "中纵梁",
        "share": 0.12,
        "weight_t": 8.6,
        "duration_min": 28.0,
        "risk": 0.22,
        "layer": 2,
    },
    "middle_slab": {
        "label": "中板",
        "share": 0.14,
        "weight_t": 11.6,
        "duration_min": 32.0,
        "risk": 0.20,
        "layer": 2,
    },
    "platform_slab": {
        "label": "站台板",
        "share": 0.076,
        "weight_t": 10.2,
        "duration_min": 31.0,
        "risk": 0.18,
        "layer": 1,
    },
    "roof_slab": {
        "label": "顶板",
        "share": 0.10,
        "weight_t": 16.7,
        "duration_min": 40.0,
        "risk": 0.28,
        "layer": 3,
    },
    "stair": {
        "label": "楼梯段",
        "share": 0.004,
        "weight_t": 6.8,
        "duration_min": 34.0,
        "risk": 0.36,
        "layer": 3,
    },
}


@dataclass(frozen=True)
class Component:
    id: str
    type: str
    name: str
    x: float
    y: float
    z: float
    weight_t: float
    duration_min: float
    risk: float
    semantic_type: str = ""
    length_m: float = 1.0
    width_m: float = 1.0
    height_m: float = 1.0
    obstacle_level: int = 0
    visibility_level: int = 0
    ground_level: int = 0
    center_of_gravity_m: float | None = None
    rotation_required: bool = False
    predecessors: tuple[int, ...] = ()


def _allocate_counts(total: int) -> dict[str, int]:
    raw = {key: total * spec["share"] for key, spec in TYPE_SPECS.items()}
    counts = {key: max(1, int(np.floor(value))) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    fractions = sorted(
        ((raw[key] - np.floor(raw[key]), key) for key in TYPE_SPECS),
        reverse=True,
    )
    index = 0
    while remainder > 0:
        counts[fractions[index % len(fractions)][1]] += 1
        remainder -= 1
        index += 1
    while remainder < 0:
        key = max(counts, key=lambda item: counts[item])
        if counts[key] > 1:
            counts[key] -= 1
            remainder += 1
        else:
            break
    for key, multiple in (
        ("bottom_slab", 3),
        ("middle_slab", 3),
        ("roof_slab", 2),
        ("platform_slab", 2),
    ):
        counts[key] = max(
            multiple,
            int(round(counts[key] / multiple)) * multiple,
        )
    balance = total - sum(counts.values())
    flexible_key = "side_wall"
    if counts[flexible_key] + balance < 1:
        flexible_key = "beam"
    counts[flexible_key] += balance
    return counts


def _layout_positions(
    component_type: str,
    count: int,
    length_m: float,
    width_m: float,
    height_m: float,
) -> list[tuple[float, float, float]]:
    layer = TYPE_SPECS[component_type]["layer"]
    if component_type == "beam":
        y_count = 1
        x_count = max(1, count)
    elif component_type == "column":
        y_count = 1
        x_count = max(1, count)
    elif component_type == "platform_slab":
        x_count = max(1, int(np.ceil(count / 2)))
        y_count = min(count, 2)
    elif component_type == "bottom_slab":
        y_count = 3
        x_count = max(1, int(np.ceil(count / y_count)))
    elif component_type == "middle_slab":
        y_count = 3
        x_count = max(1, int(np.ceil(count / y_count)))
    elif component_type == "roof_slab":
        y_count = 2
        x_count = max(1, int(np.ceil(count / y_count)))
    elif component_type == "stair":
        x_count = 1
        y_count = min(count, 2)
    elif component_type in {"side_wall"}:
        x_count = max(2, int(np.ceil(count / 2)))
        y_count = 2
    else:
        x_count = max(2, int(np.ceil(np.sqrt(count * length_m / max(width_m, 1.0)))))
        y_count = max(1, int(np.ceil(count / x_count)))

    positions: list[tuple[float, float, float]] = []
    x_span = max(length_m - 8.0, length_m * 0.86)
    y_span = max(width_m - 4.0, width_m * 0.82)
    x_offset = (length_m - x_span) / 2.0
    y_offset = (width_m - y_span) / 2.0
    for index in range(count):
        ix = index % x_count
        iy = (index // x_count) % y_count
        x = x_offset + (ix / max(x_count - 1, 1)) * x_span
        if component_type == "side_wall":
            y = y_offset if iy == 0 else (length_m * 0.0 + y_offset + y_span)
        elif component_type == "platform_slab":
            y = y_offset + y_span * (0.375 if iy == 0 else 0.625)
        elif component_type == "stair":
            y = y_offset + y_span * 0.66
        elif component_type in {"beam", "column"}:
            y = y_offset + y_span * 0.50
        else:
            y = y_offset + ((iy + 0.5) / max(y_count, 1)) * y_span

        z = _component_level_z(component_type, height_m, width_m)
        positions.append((float(x), float(y), float(z)))
    return positions


def _component_level_z(
    component_type: str,
    height_m: float,
    width_m: float,
) -> float:
    middle_center = height_m * 0.50
    platform_center = max(0.85, min(1.45, height_m * 0.10))
    platform_top = platform_center + 0.15
    middle_bottom = middle_center - 0.15
    beam_height = 0.60
    beam_bottom = middle_bottom - beam_height
    floor_rise = max(0.18, min(width_m * 0.028, 0.85))
    bottom_connection_z = 0.28 - floor_rise + 0.05
    roof_center = height_m * 0.78
    if component_type == "bottom_slab":
        return 0.28
    if component_type == "platform_slab":
        return platform_center
    if component_type == "column":
        return (bottom_connection_z + beam_bottom) * 0.5
    if component_type == "side_wall":
        return (0.48 + height_m * 0.64) * 0.5
    if component_type == "beam":
        return beam_bottom + beam_height * 0.5
    if component_type == "middle_slab":
        return middle_center
    if component_type == "roof_slab":
        return roof_center
    if component_type == "stair":
        return (platform_top + middle_bottom) * 0.5
    return height_m * 0.5


def _apply_parameterized_layout_corrections(
    components: Sequence[Component],
    length_m: float,
    width_m: float,
) -> list[Component]:
    center_y = width_m * 0.5
    platform_components = [
        component for component in components if component.type == "platform_slab"
    ]
    right_platform_y = (
        max(component.y for component in platform_components)
        if platform_components
        else center_y + width_m * 0.16
    )
    stairs = [component for component in components if component.type == "stair"]
    if not stairs:
        return list(components)

    stair_x = length_m * 0.50
    selected_stair = replace(
        stairs[0],
        x=stair_x,
        y=right_platform_y,
    )
    opening_radius = max(6.0, selected_stair.length_m * 0.85)

    corrected: list[Component] = []
    stair_kept = False
    for component in components:
        if component.type == "stair":
            if stair_kept:
                continue
            corrected.append(selected_stair)
            stair_kept = True
            continue
        if (
            component.type in {"column", "middle_slab", "beam"}
            and component.y > center_y
            and abs(component.x - stair_x) <= opening_radius
        ):
            continue
        corrected.append(component)
    return corrected


def _attach_predecessors(
    components: list[Component],
    length_m: float,
) -> list[Component]:
    by_id = {index: component for index, component in enumerate(components)}
    result: list[Component] = []
    for index, component in enumerate(components):
        layer = TYPE_SPECS[component.type]["layer"]
        predecessors: list[int] = []
        if layer > 0:
            candidates = [
                other_index
                for other_index, other in by_id.items()
                if TYPE_SPECS[other.type]["layer"] < layer
            ]
            candidates = [
                candidate
                for candidate in candidates
                if abs(by_id[candidate].x - component.x) <= max(8.0, length_m / 13.0)
                and abs(by_id[candidate].y - component.y) <= 7.5
            ]
            if candidates:
                candidates.sort(
                    key=lambda candidate: (
                        abs(by_id[candidate].x - component.x)
                        + 0.8 * abs(by_id[candidate].y - component.y)
                    )
                )
                predecessor_count = 1 if component.type not in {"roof_slab", "platform_slab"} else 2
                predecessors = candidates[:predecessor_count]
        result.append(
            Component(
                id=component.id,
                type=component.type,
                name=component.name,
                semantic_type=component.semantic_type,
                x=component.x,
                y=component.y,
                z=component.z,
                weight_t=component.weight_t,
                duration_min=component.duration_min,
                risk=component.risk,
                length_m=component.length_m,
                width_m=component.width_m,
                height_m=component.height_m,
                obstacle_level=component.obstacle_level,
                visibility_level=component.visibility_level,
                ground_level=component.ground_level,
                center_of_gravity_m=component.center_of_gravity_m,
                rotation_required=component.rotation_required,
                predecessors=tuple(predecessors),
            )
        )
    return result


def build_station(
    station_name: str,
    component_count: int | None = None,
    length_m: float | None = None,
    height_m: float | None = None,
    width_m: float | None = None,
    seed: int = 20250925,
) -> list[Component]:
    preset = STATION_PRESETS[station_name]
    component_count = int(component_count or preset["component_count"])
    length_m = float(length_m or preset["length_m"])
    height_m = float(height_m or preset["height_m"])
    width_m = float(width_m or preset["width_m"])
    rng = np.random.default_rng(seed)
    counts = _allocate_counts(component_count)

    components: list[Component] = []
    sequence_number = 1
    for component_type, count in counts.items():
        spec = TYPE_SPECS[component_type]
        positions = _layout_positions(
            component_type,
            count,
            length_m=length_m,
            width_m=width_m,
            height_m=height_m,
        )
        for index, (x, y, z) in enumerate(positions):
            serial = f"{preset['code']}-{component_type[:2].upper()}-{sequence_number:04d}"
            dimensions, obstacle_level, visibility_level, ground_level, rotation_required = (
                _generated_component_risk_attributes(component_type, rng)
            )
            dimensions = _adjust_component_dimensions(
                component_type,
                dimensions,
                positions,
                height_m,
                width_m,
            )
            weight_t = float(
                max(2.2, rng.normal(spec["weight_t"], spec["weight_t"] * 0.08))
            )
            component_risk = estimate_static_risk(
                length_m=dimensions[0],
                width_m=dimensions[1],
                height_m=dimensions[2],
                weight_t=weight_t,
                obstacle_level=obstacle_level,
                visibility_level=visibility_level,
                ground_level=ground_level,
                rotation_required=rotation_required,
            )
            components.append(
                Component(
                    id=serial,
                    type=component_type,
                    name=f"{spec['label']}-{index + 1:02d}",
                    semantic_type=spec["label"],
                    x=x,
                    y=y,
                    z=z,
                    weight_t=weight_t,
                    duration_min=float(
                        max(16.0, rng.normal(spec["duration_min"], spec["duration_min"] * 0.10))
                    ),
                    risk=component_risk,
                    length_m=dimensions[0],
                    width_m=dimensions[1],
                    height_m=dimensions[2],
                    obstacle_level=obstacle_level,
                    visibility_level=visibility_level,
                    ground_level=ground_level,
                    center_of_gravity_m=dimensions[2] * 0.5,
                    rotation_required=rotation_required,
                )
            )
            sequence_number += 1

    components = _align_beams_with_columns(components)
    components = _apply_parameterized_layout_corrections(
        components,
        length_m=length_m,
        width_m=width_m,
    )
    components = _attach_predecessors(components, length_m)
    return components


def _align_beams_with_columns(
    components: Sequence[Component],
) -> list[Component]:
    columns = [component for component in components if component.type == "column"]
    beams = [component for component in components if component.type == "beam"]
    if not columns or not beams:
        return list(components)

    column_rows: dict[float, list[Component]] = {}
    for column in columns:
        row_key = round(column.y, 3)
        column_rows.setdefault(row_key, []).append(column)
    row_keys = sorted(column_rows)
    for row_key in row_keys:
        column_rows[row_key].sort(key=lambda component: component.x)

    beam_rows: dict[float, list[Component]] = {}
    for beam in beams:
        row_key = min(row_keys, key=lambda candidate: abs(candidate - beam.y))
        beam_rows.setdefault(row_key, []).append(beam)

    replacements: dict[str, Component] = {}
    for row_key, row_beams in beam_rows.items():
        row_columns = column_rows[row_key]
        if len(row_columns) < 2:
            continue
        pair_count = len(row_columns) - 1
        pair_indices = np.linspace(0, pair_count - 1, len(row_beams), dtype=int)
        for beam, pair_index in zip(sorted(row_beams, key=lambda item: item.x), pair_indices):
            first_column = row_columns[int(pair_index)]
            second_column = row_columns[int(pair_index) + 1]
            midpoint = (first_column.x + second_column.x) * 0.5
            beam_length = (
                abs(second_column.x - first_column.x)
                + max(first_column.length_m, second_column.length_m) * 0.45
            )
            replacements[beam.id] = replace(
                beam,
                x=midpoint,
                y=(first_column.y + second_column.y) * 0.5,
                length_m=beam_length,
                width_m=max(beam.width_m, min(first_column.width_m, second_column.width_m)),
            )
    return [replacements.get(component.id, component) for component in components]


def component_successors(components: Sequence[Component]) -> list[list[int]]:
    successors = [[] for _ in components]
    for component_index, component in enumerate(components):
        for predecessor in component.predecessors:
            successors[predecessor].append(component_index)
    return successors


def component_indegrees(components: Sequence[Component]) -> np.ndarray:
    return np.asarray([len(component.predecessors) for component in components], dtype=np.int16)


def decode_priority_vector(
    priority: Sequence[float],
    components: Sequence[Component],
    successors: Sequence[Sequence[int]] | None = None,
) -> list[int]:
    successors = successors or component_successors(components)
    indegree = component_indegrees(components).astype(np.int16)
    available: list[tuple[float, int]] = []
    for index, degree in enumerate(indegree):
        if degree == 0:
            heapq.heappush(available, (float(priority[index]), index))

    sequence: list[int] = []
    while available:
        _, current = heapq.heappop(available)
        sequence.append(current)
        for successor in successors[current]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(available, (float(priority[successor]), successor))
    if len(sequence) != len(components):
        raise ValueError("Precedence graph contains a cycle.")
    return sequence


def components_to_frame(components: Sequence[Component]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "构件编号": [component.id for component in components],
            "语义类型": [
                component.semantic_type or TYPE_SPECS[component.type]["label"]
                for component in components
            ],
            "几何族": [TYPE_SPECS[component.type]["label"] for component in components],
            "构件名称": [component.name for component in components],
            "纵向里程(m)": [round(component.x, 2) for component in components],
            "横向轴线(m)": [round(component.y, 2) for component in components],
            "安装标高(m)": [round(component.z, 2) for component in components],
            "重量(t)": [round(component.weight_t, 2) for component in components],
            "预计安装(min)": [round(component.duration_min, 1) for component in components],
            "基础风险指数": [round(component.risk * 100.0, 1) for component in components],
            "构件长(m)": [round(component.length_m, 2) for component in components],
            "构件宽(m)": [round(component.width_m, 2) for component in components],
            "构件高(m)": [round(component.height_m, 2) for component in components],
            "重心高度(m)": [
                round(
                    component.center_of_gravity_m
                    if component.center_of_gravity_m is not None
                    else component.height_m * 0.5,
                    2,
                )
                for component in components
            ],
            "障碍等级": [component.obstacle_level for component in components],
            "作业可视性": [component.visibility_level for component in components],
            "地基条件": [component.ground_level for component in components],
            "旋转要求": [
                "是" if component.rotation_required else "否" for component in components
            ],
            "前置构件数": [len(component.predecessors) for component in components],
        }
    )


def validate_components(components: Sequence[Component]) -> dict[str, int | float]:
    ids = [component.id for component in components]
    unique_count = len(set(ids))
    maximum_weight = max(component.weight_t for component in components)
    total_weight = sum(component.weight_t for component in components)
    dependency_count = sum(len(component.predecessors) for component in components)
    return {
        "component_count": len(components),
        "unique_ids": unique_count,
        "duplicate_ids": len(ids) - unique_count,
        "total_weight_t": round(total_weight, 2),
        "maximum_weight_t": round(maximum_weight, 2),
        "dependency_count": dependency_count,
    }


def estimate_static_risk(
    length_m: float,
    width_m: float,
    height_m: float,
    weight_t: float,
    obstacle_level: int,
    visibility_level: int,
    ground_level: int,
    rotation_required: bool,
) -> float:
    wind_risk = estimate_wind_risk(
        length_m=length_m,
        width_m=width_m,
        height_m=height_m,
        weight_t=weight_t,
    )
    weighted_static = (
        0.13 * np.clip(obstacle_level / 3.0, 0.0, 1.0)
        + 0.10 * np.clip(visibility_level / 2.0, 0.0, 1.0)
        + 0.10 * np.clip(ground_level / 2.0, 0.0, 1.0)
        + 0.10 * wind_risk
        + 0.05 * (1.0 if rotation_required else 0.0)
    )
    return float(np.clip(weighted_static / 0.48, 0.0, 1.0))


def estimate_wind_risk(
    length_m: float,
    width_m: float,
    height_m: float,
    weight_t: float,
) -> float:
    side_area = max(length_m * height_m, width_m * height_m)
    area_weight_ratio = side_area / max(weight_t, 0.1)
    return float(np.clip((area_weight_ratio - 0.45) / 4.2, 0.0, 1.0))


def _generated_component_risk_attributes(
    component_type: str,
    rng: np.random.Generator,
) -> tuple[tuple[float, float, float], int, int, int, bool]:
    dimensions = {
        "bottom_slab": (12.0, 2.4, 0.20),
        "column": (1.2, 1.2, 5.80),
        "side_wall": (12.0, 0.60, 5.80),
        "beam": (6.0, 0.80, 0.60),
        "middle_slab": (12.0, 2.4, 0.30),
        "platform_slab": (12.0, 2.4, 0.25),
        "roof_slab": (12.0, 2.4, 0.30),
        "stair": (4.0, 2.0, 3.00),
    }[component_type]
    obstacle_level = int(rng.integers(0, 3))
    visibility_level = int(rng.integers(0, 2 if component_type not in {"column"} else 3))
    ground_level = int(rng.integers(0, 2))
    rotation_required = component_type in {"column", "side_wall", "stair"}
    return dimensions, obstacle_level, visibility_level, ground_level, rotation_required


def _adjust_component_dimensions(
    component_type: str,
    dimensions: tuple[float, float, float],
    positions: Sequence[tuple[float, float, float]],
    station_height_m: float,
    station_width_m: float,
) -> tuple[float, float, float]:
    if component_type in {"bottom_slab", "middle_slab", "roof_slab"}:
        x_values = sorted({round(position[0], 6) for position in positions})
        y_values = sorted({round(position[1], 6) for position in positions})
        if len(x_values) > 1:
            x_spacing = (x_values[-1] - x_values[0]) / (len(x_values) - 1)
            dimensions = (max(dimensions[0], x_spacing), dimensions[1], dimensions[2])
        if len(y_values) > 1:
            y_spacing = (y_values[-1] - y_values[0]) / (len(y_values) - 1)
            dimensions = (dimensions[0], max(dimensions[1], y_spacing), dimensions[2])
    elif component_type == "platform_slab":
        x_values = sorted({round(position[0], 6) for position in positions})
        if len(x_values) > 1:
            x_spacing = (x_values[-1] - x_values[0]) / (len(x_values) - 1)
            dimensions = (max(dimensions[0], x_spacing), dimensions[1], dimensions[2])
        dimensions = (
            dimensions[0],
            max(dimensions[1], station_width_m * 0.15),
            dimensions[2],
        )

    platform_center = max(0.85, min(1.45, station_height_m * 0.10))
    platform_top = platform_center + 0.15
    middle_bottom = station_height_m * 0.50 - 0.15
    beam_height = 0.60
    beam_bottom = middle_bottom - beam_height
    floor_rise = max(0.18, min(station_width_m * 0.028, 0.85))
    bottom_connection_z = 0.28 - floor_rise + 0.05
    if component_type == "column":
        clear_height = max(1.0, beam_bottom - bottom_connection_z)
        dimensions = (dimensions[0], dimensions[1], clear_height)
    elif component_type == "side_wall":
        wall_height = max(1.0, station_height_m * 0.64 - 0.48)
        dimensions = (dimensions[0], dimensions[1], wall_height)
    elif component_type == "beam":
        dimensions = (dimensions[0], dimensions[1], beam_height)
    elif component_type == "stair":
        stair_height = max(1.0, middle_bottom - platform_top)
        dimensions = (
            max(dimensions[0], min(12.0, station_height_m * 0.96)),
            max(dimensions[1], station_width_m * 0.12),
            stair_height,
        )
    return dimensions
