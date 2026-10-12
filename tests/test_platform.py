from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from hoisting.domain import (
    STATION_PRESETS,
    build_station,
    decode_priority_vector,
    validate_components,
)
from hoisting.optimizer import (
    _choose_plan,
    _generate_reference_points,
    optimize_hoisting_sequence,
    schedule_sequence,
)
from hoisting.charts import (
    _component_bounds,
    _component_geometry,
    _station_section_context,
    build_site_figure,
)
from hoisting.presentation import hero_html
from hoisting.three_viewer import build_three_viewer_html
import hoisting.projects as projects_module
from hoisting.importer import (
    ImportBundle,
    build_excel_template,
    normalize_project_metadata,
    read_import_file,
    template_frame,
    validate_import_bundle,
    validate_import_frame,
)


class DomainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.station_name = next(iter(STATION_PRESETS))
        self.components = build_station(
            self.station_name,
            component_count=96,
            seed=17,
        )

    def test_component_generation_is_complete_and_unique(self) -> None:
        validation = validate_components(self.components)
        self.assertEqual(validation["component_count"], 96)
        self.assertEqual(validation["unique_ids"], 96)
        self.assertEqual(validation["duplicate_ids"], 0)
        self.assertGreater(validation["dependency_count"], 0)

    def test_hero_uses_current_project_metadata(self) -> None:
        markup = hero_html(
            project_name="\u6211\u7684\u8f6c\u4f53\u9879\u76ee",
            component_count=180,
            cranes=[{}, {}],
            algorithm_label="NSGA-II / Pareto \u6c42\u89e3",
        )
        self.assertIn("\u5f53\u524d\u9879\u76ee \u6211\u7684\u8f6c\u4f53\u9879\u76ee", markup)
        self.assertIn("\u6784\u4ef6 180 \u4ef6", markup)
        self.assertNotIn("\u6df1\u5733", markup)

    def test_priority_decode_respects_predecessors(self) -> None:
        priority = np.random.default_rng(3).normal(size=len(self.components))
        sequence = decode_priority_vector(priority, self.components)
        positions = {component_index: order for order, component_index in enumerate(sequence)}
        for component_index, component in enumerate(self.components):
            for predecessor in component.predecessors:
                self.assertLess(
                    positions[predecessor],
                    positions[component_index],
                    msg=f"{predecessor} must precede {component_index}",
                )

    def test_schedule_uses_all_components(self) -> None:
        cranes = STATION_PRESETS[self.station_name]["cranes"]
        metrics = schedule_sequence(
            list(range(len(self.components))),
            self.components,
            cranes,
        )
        self.assertEqual(len(metrics["schedule"]), len(self.components))
        self.assertEqual(metrics["unassigned"], [])
        self.assertGreater(metrics["makespan_min"], 0.0)
        self.assertGreater(metrics["cost_yuan"], 0.0)
        self.assertGreaterEqual(metrics["risk_score"], 0.0)
        self.assertLessEqual(metrics["risk_score"], 100.0)
        self.assertTrue(all(0.0 <= row["risk"] <= 100.0 for row in metrics["schedule"]))
        self.assertAlmostEqual(
            metrics["risk_score"],
            sum(metrics["risk_breakdown"].values()),
            places=6,
        )

    def test_parameterized_station_column_beam_and_slab_levels(self) -> None:
        columns = [component for component in self.components if component.type == "column"]
        beams = [component for component in self.components if component.type == "beam"]
        middle_slabs = [
            component for component in self.components if component.type == "middle_slab"
        ]
        platform_slabs = [
            component for component in self.components if component.type == "platform_slab"
        ]
        column_rows = {round(component.y, 4) for component in columns}
        beam_rows = {round(component.y, 4) for component in beams}
        self.assertEqual(len(column_rows), 1)
        self.assertEqual(column_rows, beam_rows)
        self.assertAlmostEqual(
            columns[0].z + columns[0].height_m * 0.5,
            beams[0].z - beams[0].height_m * 0.5,
            places=4,
        )
        self.assertAlmostEqual(
            beams[0].z + beams[0].height_m * 0.5,
            middle_slabs[0].z - middle_slabs[0].height_m * 0.5,
            places=4,
        )
        self.assertEqual(
            len({round(component.z, 4) for component in platform_slabs}),
            1,
        )
        platform_centers = sorted({round(component.y, 4) for component in platform_slabs})
        self.assertEqual(len(platform_centers), 2)
        platform_width = platform_slabs[0].width_m
        platform_gap = platform_centers[1] - platform_centers[0] - platform_width
        self.assertGreaterEqual(platform_gap, 0.8)
        self.assertLessEqual(platform_gap, 1.7)
        min_x, max_x, min_y, max_y = _component_bounds(self.components)
        max_z = max(
            component.z + component.height_m * 0.5
            for component in self.components
        )
        context = _station_section_context(
            self.components,
            min_y,
            max_y,
            max_z,
        )
        left_platform = min(platform_slabs, key=lambda component: component.y)
        right_platform = max(platform_slabs, key=lambda component: component.y)
        left_vertices, _ = _component_geometry(left_platform, context)
        right_vertices, _ = _component_geometry(right_platform, context)
        self.assertLess(
            left_vertices[:, 1].min(),
            left_platform.y - left_platform.width_m * 0.5,
        )
        self.assertGreater(
            right_vertices[:, 1].max(),
            right_platform.y + right_platform.width_m * 0.5,
        )
        bottom_slabs = [
            component for component in self.components if component.type == "bottom_slab"
        ]
        bottom_rows = {}
        for bottom_slab in bottom_slabs:
            key = round(bottom_slab.x, 4)
            bottom_rows[key] = bottom_rows.get(key, 0) + 1
        self.assertTrue(bottom_rows)
        self.assertEqual(set(bottom_rows.values()), {3})
        roof_slabs = [
            component for component in self.components if component.type == "roof_slab"
        ]
        roof_rows = {}
        for roof_slab in roof_slabs:
            key = round(roof_slab.x, 4)
            roof_rows[key] = roof_rows.get(key, 0) + 1
        self.assertTrue(roof_rows)
        self.assertEqual(set(roof_rows.values()), {2})
        for beam in beams:
            supporting_columns = [
                column
                for column in columns
                if abs(column.y - beam.y) < 0.01
                and beam.x - beam.length_m * 0.5 - 0.35
                <= column.x
                <= beam.x + beam.length_m * 0.5 + 0.35
            ]
            self.assertGreaterEqual(
                len(supporting_columns),
                2,
                msg=f"{beam.id} must be supported by two columns",
            )


class OptimizerTests(unittest.TestCase):
    def test_project_round_trip_preserves_components_and_equipment(self) -> None:
        station_name = next(iter(STATION_PRESETS))
        components = build_station(station_name, component_count=40, seed=44)
        equipment = STATION_PRESETS[station_name]["cranes"]
        with TemporaryDirectory() as directory:
            projects_module.PROJECT_ROOT = Path(directory)
            saved = projects_module.save_project(
                project={"项目编号": "TEST-01", "项目名称": "测试项目"},
                components=components,
                equipment=equipment,
                relations=[{"当前构件编号": "C2", "目标构件编号": "C1"}],
            )
            loaded = projects_module.load_project(saved["id"])
        self.assertEqual(loaded["project"]["项目名称"], "测试项目")
        self.assertEqual(len(loaded["components"]), len(components))
        self.assertEqual(len(loaded["equipment"]), len(equipment))
        self.assertEqual(loaded["components"][0].semantic_type, components[0].semantic_type)

    def test_delete_saved_project_removes_file(self) -> None:
        station_name = next(iter(STATION_PRESETS))
        components = build_station(station_name, component_count=24, seed=45)
        equipment = STATION_PRESETS[station_name]["cranes"]
        with TemporaryDirectory() as directory:
            projects_module.PROJECT_ROOT = Path(directory)
            saved = projects_module.save_project(
                project={
                    "\u9879\u76ee\u7f16\u53f7": "TEST-DELETE",
                    "\u9879\u76ee\u540d\u79f0": "\u5f85\u5220\u9664\u9879\u76ee",
                },
                components=components,
                equipment=equipment,
            )
            path = Path(directory) / f"{saved['id']}.json"
            self.assertTrue(path.is_file())
            self.assertTrue(projects_module.delete_project(saved["id"]))
            self.assertFalse(path.exists())
            self.assertFalse(projects_module.delete_project(saved["id"]))
            with self.assertRaises(ValueError):
                projects_module.delete_project("../outside")

    def test_nsga3_reference_points_and_expert_objectives(self) -> None:
        reference_points = _generate_reference_points(5, 1)
        self.assertEqual(reference_points.shape, (5, 5))
        self.assertTrue(
            np.allclose(reference_points.sum(axis=1), np.ones(5))
        )
        station_name = next(iter(STATION_PRESETS))
        components = build_station(station_name, component_count=48, seed=33)
        result = optimize_hoisting_sequence(
            components,
            STATION_PRESETS[station_name]["cranes"],
            population_size=10,
            generations=3,
            seed=33,
            algorithm="nsga3",
            objective_mode="expert",
            reference_divisions=1,
        )
        self.assertEqual(result.pareto_objectives.shape[1], 5)
        self.assertEqual(result.metadata["algorithm"], "NSGA3")
        self.assertEqual(result.metadata["reference_points"], 5)
        self.assertEqual(
            set(result.plans),
            {"工期优先", "成本优先", "空间安全", "环境安全", "设备均衡", "综合推荐"},
        )

    def test_decision_plans_select_distinct_pareto_solutions(self) -> None:
        objectives = np.asarray(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.4, 0.4, 0.4],
            ]
        )
        selected = set()
        for weights in (
            (0.70, 0.20, 0.10),
            (0.15, 0.75, 0.10),
            (0.10, 0.15, 0.75),
        ):
            choice = _choose_plan(
                objectives,
                weights,
                excluded_indices=selected,
            )
            self.assertNotIn(choice, selected)
            selected.add(choice)

    def test_gantry_schedule_uses_three_axis_motion(self) -> None:
        station_name = next(iter(STATION_PRESETS))
        components = build_station(station_name, component_count=48, seed=23)
        gantries = [
            {
                "id": "GM-01",
                "type": "龙门吊",
                "capacity_t": 120.0,
                "rate_yuan_h": 520.0,
                "position": (-8.0, 12.0),
                "initial_x": -8.0,
                "initial_y": 12.0,
                "rail_start_x": -15.0,
                "rail_end_x": 150.0,
                "trolley_min_y": -3.0,
                "trolley_max_y": 28.0,
                "max_lift_height": 24.0,
            }
        ]
        metrics = schedule_sequence(
            list(range(len(components))),
            components,
            gantries,
        )
        self.assertEqual(metrics["unassigned"], [])
        self.assertEqual(len(metrics["schedule"]), len(components))
        for row in metrics["schedule"]:
            self.assertGreaterEqual(row["long_travel_m"], 0.0)
            self.assertGreaterEqual(row["trolley_travel_m"], 0.0)
            self.assertGreaterEqual(row["hoist_travel_m"], 0.0)
            self.assertGreaterEqual(row["movement_minutes"], 0.0)
            self.assertGreaterEqual(row["trolley_offset_m"], 0.0)

    def test_two_gantries_keep_home_side_work_zones(self) -> None:
        station_name = next(iter(STATION_PRESETS))
        components = build_station(station_name, component_count=72, seed=11)
        cranes = STATION_PRESETS[station_name]["cranes"]
        metrics = schedule_sequence(
            list(range(len(components))),
            components,
            cranes,
            prefer_home_zone=True,
        )
        assignments = {
            crane["id"]: [
                row["x"]
                for row in metrics["schedule"]
                if row["crane"] == crane["id"]
            ]
            for crane in cranes
        }
        self.assertTrue(all(values for values in assignments.values()))
        left_values = assignments[cranes[0]["id"]]
        right_values = assignments[cranes[1]["id"]]
        self.assertLessEqual(max(left_values), max(right_values))
        self.assertLessEqual(max(left_values), min(right_values) + 12.0)

    def test_optimizer_returns_pareto_front_and_decisions(self) -> None:
        station_name = next(iter(STATION_PRESETS))
        components = build_station(station_name, component_count=72, seed=31)
        result = optimize_hoisting_sequence(
            components,
            STATION_PRESETS[station_name]["cranes"],
            population_size=16,
            generations=8,
            seed=31,
        )
        self.assertEqual(result.pareto_objectives.shape[1], 3)
        self.assertGreaterEqual(result.pareto_objectives.shape[0], 1)
        self.assertEqual(
            set(result.plans),
            {"A · 进度优先", "B · 成本优先", "C · 安全优先", "综合推荐"},
        )
        for plan in result.plans.values():
            self.assertEqual(len(plan["schedule"]), len(components))
            self.assertTrue(np.isfinite(plan["makespan_min"]))
            self.assertTrue(np.isfinite(plan["cost_yuan"]))
            self.assertTrue(np.isfinite(plan["risk_score"]))


class ImportTests(unittest.TestCase):
    def test_gb18030_csv_import_is_decoded(self) -> None:
        raw = template_frame().to_csv(index=False).encode("gb18030")
        uploaded = BytesIO(raw)
        uploaded.name = "gb18030.csv"  # type: ignore[attr-defined]
        result = validate_import_bundle(read_import_file(uploaded))
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.components or []), 3)

    def test_project_metadata_repairs_mojibake_and_internal_keys(self) -> None:
        broken_key = "项目名称".encode("utf-8").decode("gb18030")
        broken_value = "测试车站".encode("utf-8").decode("gb18030")
        cleaned = normalize_project_metadata(
            {
                f"\ufeff{broken_key}": broken_value,
                "_source": "internal",
                "_units": {broken_key: "m"},
            }
        )
        self.assertEqual(cleaned["项目名称"], "测试车站")
        self.assertNotIn("_source", cleaned)
        self.assertEqual(cleaned["_units"]["项目名称"], "m")

    def test_phase_one_excel_bundle_reads_all_optional_sheets(self) -> None:
        template = build_excel_template()
        sheets = pd.read_excel(template, sheet_name=None)
        bundle = ImportBundle(
            component_frame=sheets["构件数据"],
            project_frame=sheets["项目参数"],
            relations_frame=sheets["构件关系"],
            equipment_frame=sheets["设备数据"],
            source_name="phase1.xlsx",
        )
        result = validate_import_bundle(bundle)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.project["项目编号"], "SZ-METRO-001")
        self.assertEqual(len(result.relations), 2)
        self.assertEqual(len(result.equipment), 2)
        self.assertEqual(result.components[0].semantic_type, "底板")  # type: ignore[index]

    def test_template_import_builds_valid_components(self) -> None:
        result = validate_import_frame(template_frame(), "template.xlsx")
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.components or []), 3)
        self.assertEqual(result.summary["dependency_count"], 2)
        self.assertEqual(result.errors.shape[0], 0)

    def test_missing_required_column_is_reported(self) -> None:
        frame = template_frame().drop(columns=["重量(t)"])
        result = validate_import_frame(frame, "missing.xlsx")
        self.assertFalse(result.is_valid)
        self.assertIn("重量(t)", set(result.errors["字段"]))

    def test_duplicate_and_missing_predecessor_are_reported(self) -> None:
        frame = template_frame()
        frame.loc[1, "构件编号"] = "C001"
        frame.loc[2, "前置构件编号"] = "NOT-FOUND"
        result = validate_import_frame(frame, "invalid.xlsx")
        self.assertFalse(result.is_valid)
        problems = set(result.errors["问题"])
        self.assertIn("构件编号重复", problems)
        self.assertIn("引用了不存在的构件", problems)

    def test_cycle_is_reported(self) -> None:
        frame = template_frame()
        frame.loc[0, "前置构件编号"] = "C003"
        result = validate_import_frame(frame, "cycle.xlsx")
        self.assertFalse(result.is_valid)
        self.assertIn("前置关系形成循环", set(result.errors["问题"]))

    def test_excel_template_round_trip_and_risk_calculation(self) -> None:
        template = build_excel_template()
        frame = pd.read_excel(template, sheet_name="构件数据")
        result = validate_import_frame(frame, "template.xlsx")
        self.assertTrue(result.is_valid)
        component = result.components[0]  # type: ignore[index]
        self.assertGreater(component.risk, 0.0)
        self.assertLessEqual(component.risk, 1.0)
        self.assertEqual(component.length_m, 12.0)
        self.assertEqual(component.obstacle_level, 1)

    def test_legacy_manual_risk_column_is_ignored_with_warning(self) -> None:
        frame = template_frame()
        frame["风险指数"] = 0.8
        result = validate_import_frame(frame, "legacy.xlsx")
        self.assertTrue(result.is_valid)
        self.assertIn("人工风险指数", " ".join(result.warnings["问题"].tolist()))

    def test_invalid_risk_input_levels_are_reported(self) -> None:
        frame = template_frame()
        frame.loc[0, "障碍等级"] = 9
        frame.loc[1, "作业可视性"] = 5
        frame.loc[2, "地基条件"] = -1
        result = validate_import_frame(frame, "levels.xlsx")
        self.assertFalse(result.is_valid)
        fields = set(result.errors["字段"])
        self.assertIn("障碍等级", fields)
        self.assertIn("作业可视性", fields)
        self.assertIn("地基条件", fields)

    def test_three_dimensional_view_uses_parametric_component_solids(self) -> None:
        station_name = next(iter(STATION_PRESETS))
        components = build_station(station_name, component_count=32, seed=9)
        cranes = STATION_PRESETS[station_name]["cranes"]
        plan = schedule_sequence(list(range(len(components))), components, cranes)
        figure = build_site_figure(
            components,
            plan,
            cranes,
            color_mode="安装状态",
            progress_percent=35.0,
        )
        meshes = [trace for trace in figure.data if trace.type == "mesh3d"]
        self.assertGreaterEqual(len(meshes), 1)
        self.assertGreaterEqual(sum(len(trace.x) for trace in meshes), len(components) * 8)
        self.assertGreaterEqual(sum(len(trace.i) for trace in meshes), len(components) * 12)
        for trace in meshes:
            self.assertEqual(len(trace.facecolor), len(trace.i))

        initial_figure = build_site_figure(
            components,
            plan,
            cranes,
            color_mode="构件类型",
            progress_percent=0.0,
        )
        progressed_figure = build_site_figure(
            components,
            plan,
            cranes,
            color_mode="构件类型",
            progress_percent=50.0,
        )
        initial_vertices = sum(
            len(trace.x)
            for trace in initial_figure.data
            if trace.type == "mesh3d"
        )
        progressed_vertices = sum(
            len(trace.x)
            for trace in progressed_figure.data
            if trace.type == "mesh3d"
        )
        self.assertEqual(initial_vertices, 0)
        initial_markers = next(
            trace
            for trace in initial_figure.data
            if trace.name == "待安装定位"
        )
        self.assertEqual(len(initial_markers.x), len(components))
        self.assertGreater(progressed_vertices, 0)
        progressed_markers = next(
            trace
            for trace in progressed_figure.data
            if trace.name == "待安装定位"
        )
        self.assertLess(len(progressed_markers.x), len(components))
        self.assertNotIn(
            "楼梯踏步控制线",
            {trace.name for trace in initial_figure.data},
        )
        self.assertIn(
            "最近安装构件轮廓",
            {trace.name for trace in progressed_figure.data},
        )

        completed_figure = build_site_figure(
            components,
            plan,
            cranes,
            color_mode="构件类型",
            progress_percent=100.0,
        )
        completed_vertices = sum(
            len(trace.x)
            for trace in completed_figure.data
            if trace.type == "mesh3d"
        )
        self.assertGreaterEqual(completed_vertices, len(components) * 8)
        self.assertIn(
            "楼梯踏步控制线",
            {trace.name for trace in completed_figure.data},
        )

    def test_three_viewer_bundle_contains_local_engine_and_payload(self) -> None:
        station_name = next(iter(STATION_PRESETS))
        components = build_station(station_name, component_count=48, seed=12)
        cranes = STATION_PRESETS[station_name]["cranes"]
        plan = schedule_sequence(list(range(len(components))), components, cranes)
        viewer_html = build_three_viewer_html(components, plan, cranes)
        self.assertIn('id="viewer-payload"', viewer_html)
        self.assertIn('id="three-source"', viewer_html)
        self.assertIn("viewerStats", viewer_html)
        self.assertIn('value="crane"', viewer_html)
        self.assertIn('"start":', viewer_html)
        self.assertIn('"end":', viewer_html)
        self.assertLess(len(viewer_html), 1_500_000)


if __name__ == "__main__":
    unittest.main()
