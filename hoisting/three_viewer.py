from __future__ import annotations

import base64
from functools import lru_cache
import json
from pathlib import Path
from typing import Sequence

from .charts import (
    TYPE_COLORS,
    _component_bounds,
    _component_geometry,
    _station_section_context,
)
from .domain import Component, TYPE_SPECS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIEWER_ROOT = PROJECT_ROOT / "assets" / "three_viewer"
THREE_SOURCE = PROJECT_ROOT / "assets" / "vendor" / "three.module.min.js"
THREE_CORE_SOURCE = PROJECT_ROOT / "assets" / "vendor" / "three.core.min.js"


@lru_cache(maxsize=1)
def _viewer_assets() -> tuple[str, str, str]:
    return (
        (VIEWER_ROOT / "index.html").read_text(encoding="utf-8"),
        (VIEWER_ROOT / "viewer.css").read_text(encoding="utf-8"),
        (VIEWER_ROOT / "viewer.js").read_text(encoding="utf-8"),
    )


@lru_cache(maxsize=1)
def _three_source_base64() -> str:
    return base64.b64encode(THREE_SOURCE.read_bytes()).decode("ascii")


@lru_cache(maxsize=1)
def _three_core_source_base64() -> str:
    return base64.b64encode(THREE_CORE_SOURCE.read_bytes()).decode("ascii")


def build_three_viewer_html(
    components: Sequence[Component],
    plan: dict,
    cranes: Sequence[dict],
) -> str:
    schedule = plan["schedule"]
    order_map = {row["component_index"]: row for row in schedule}
    min_x, max_x, min_y, max_y = _component_bounds(components)
    min_z = min(component.z - component.height_m * 0.5 for component in components)
    max_z = max(component.z + component.height_m * 0.5 for component in components)
    section_context = _station_section_context(
        components=components,
        min_y=min_y,
        max_y=max_y,
        max_z=max_z,
    )

    serialized_components: list[dict] = []
    geometry_min_z = float("inf")
    geometry_max_z = float("-inf")
    for index, component in enumerate(components):
        vertices, faces = _component_geometry(component, section_context)
        geometry_min_z = min(geometry_min_z, float(vertices[:, 2].min()))
        geometry_max_z = max(geometry_max_z, float(vertices[:, 2].max()))
        task = order_map.get(index, {})
        serialized_components.append(
            {
                "id": component.id,
                "name": component.name,
                "type": component.type,
                "typeLabel": (
                    component.semantic_type or TYPE_SPECS[component.type]["label"]
                ),
                "color": TYPE_COLORS[component.type],
                "x": round(component.x, 3),
                "y": round(component.y, 3),
                "z": round(component.z, 3),
                "weight": round(component.weight_t, 3),
                "risk": round(float(task.get("risk", component.risk * 100.0)), 2),
                "order": int(task.get("order", len(schedule) + index + 1)),
                "crane": str(task.get("crane", "")),
                "start": round(float(task.get("start", 0.0)), 3),
                "end": round(float(task.get("end", 0.0)), 3),
                "vertices": [
                    [round(float(value), 3) for value in vertex]
                    for vertex in vertices
                ],
                "faces": [
                    [int(face[0]), int(face[1]), int(face[2])]
                    for face in faces
                ],
            }
        )

    payload = {
        "components": serialized_components,
        "componentCount": len(schedule),
        "typeLabels": {
            component_type: spec["label"]
            for component_type, spec in TYPE_SPECS.items()
        },
        "typeColors": {
            component_type: TYPE_COLORS[component_type]
            for component_type in TYPE_SPECS
        },
        "dimensions": {
            "minX": min_x,
            "maxX": max_x,
            "minY": min_y,
            "maxY": max_y,
            "minZ": geometry_min_z,
            "maxZ": geometry_max_z,
            "centerX": (min_x + max_x) * 0.5,
            "centerY": (min_y + max_y) * 0.5,
            "spanX": max(max_x - min_x, 1.0),
            "spanY": max(max_y - min_y, 1.0),
            "height": max(geometry_max_z - geometry_min_z, 1.0),
        },
        "cranes": [
            {
                "id": crane["id"],
                "x": float(crane.get("initial_x", crane["position"][0])),
                "y": float(crane.get("initial_y", crane["position"][1])),
                "capacity": float(crane["capacity_t"]),
                "railStartX": float(
                    crane.get("rail_start_x", min_x - 8.0)
                ),
                "railEndX": float(
                    crane.get("rail_end_x", max_x + 8.0)
                ),
                "trolleyMinY": float(
                    crane.get("trolley_min_y", min_y - 2.0)
                ),
                "trolleyMaxY": float(
                    crane.get("trolley_max_y", max_y + 2.0)
                ),
                "maxLiftHeight": float(
                    crane.get("max_lift_height", geometry_max_z + 6.0)
                ),
            }
            for crane in cranes
        ],
    }
    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    template, css, viewer_js = _viewer_assets()
    return (
        template.replace("__VIEWER_CSS__", css)
        .replace("__VIEWER_PAYLOAD__", payload_json)
        .replace("__THREE_SOURCE_BASE64__", _three_source_base64())
        .replace("__THREE_CORE_SOURCE_BASE64__", _three_core_source_base64())
        .replace("__VIEWER_JS__", viewer_js)
    )
