from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Sequence
import uuid

from .domain import Component


PROJECT_ROOT = Path(__file__).resolve().parent.parent / "projects"


def list_saved_projects() -> list[dict[str, Any]]:
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    projects: list[dict[str, Any]] = []
    for path in PROJECT_ROOT.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            projects.append(
                {
                    "id": payload.get("id", path.stem),
                    "name": payload.get("project", {}).get("项目名称", path.stem),
                    "updated_at": payload.get("updated_at", ""),
                    "path": str(path),
                    "project": payload.get("project", {}),
                    "component_count": len(payload.get("components", [])),
                    "equipment_count": len(payload.get("equipment", [])),
                }
            )
        except (OSError, ValueError, TypeError):
            continue
    return sorted(projects, key=lambda item: item["updated_at"], reverse=True)


def save_project(
    *,
    project: dict[str, Any],
    components: Sequence[Component],
    equipment: Sequence[dict[str, Any]],
    relations: Sequence[dict[str, Any]] | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    project_id = project_id or uuid.uuid4().hex[:12]
    payload = {
        "schema_version": "1.0",
        "id": project_id,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "project": dict(project),
        "components": [_component_to_dict(component) for component in components],
        "relations": list(relations or []),
        "equipment": [dict(item) for item in equipment],
    }
    path = PROJECT_ROOT / f"{project_id}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def load_project(project_id: str) -> dict[str, Any]:
    path = PROJECT_ROOT / f"{project_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["components"] = [
        _component_from_dict(item) for item in payload.get("components", [])
    ]
    return payload


def delete_project(project_id: str) -> bool:
    project_id = str(project_id).strip()
    if (
        not project_id
        or project_id in {".", ".."}
        or Path(project_id).name != project_id
    ):
        raise ValueError("非法项目编号")
    root = PROJECT_ROOT.resolve()
    path = (root / f"{project_id}.json").resolve()
    if path.parent != root:
        raise ValueError("项目路径越界")
    if not path.is_file():
        return False
    path.unlink()
    return True


def _component_to_dict(component: Component) -> dict[str, Any]:
    payload = asdict(component)
    payload["predecessors"] = list(component.predecessors)
    return payload


def _component_from_dict(payload: dict[str, Any]) -> Component:
    data = dict(payload)
    data["predecessors"] = tuple(int(value) for value in data.get("predecessors", []))
    return Component(**data)
