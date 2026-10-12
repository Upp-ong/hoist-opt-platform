from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .domain import (
    Component,
    TYPE_SPECS,
    component_successors,
    estimate_static_risk,
)


REQUIRED_COLUMNS = [
    "构件编号",
    "语义类型",
    "构件名称",
    "X坐标(m)",
    "Y坐标(m)",
    "Z坐标(m)",
    "重量(t)",
    "安装时长(min)",
    "构件长(m)",
    "构件宽(m)",
    "构件高(m)",
    "障碍等级",
    "作业可视性",
    "地基条件",
]

OPTIONAL_COLUMNS = [
    "几何族",
    "前置构件编号",
    "重心高度(m)",
    "旋转要求",
    "风险指数",
]

COLUMN_ALIASES = {
    "构件编号": ["构件编号", "构件ID", "构件 Id", "编号", "component_id", "component id", "id"],
    "语义类型": [
        "语义类型",
        "构件类型",
        "类型",
        "semantic_type",
        "component_type",
        "type",
    ],
    "几何族": ["几何族", "形状族", "shape_family", "geometry_family"],
    "构件名称": ["构件名称", "名称", "component_name", "component name", "name"],
    "X坐标(m)": ["X坐标(m)", "X坐标", "纵向里程(m)", "纵向坐标(m)", "x", "x坐标"],
    "Y坐标(m)": ["Y坐标(m)", "Y坐标", "横向轴线(m)", "横向坐标(m)", "y", "y坐标"],
    "Z坐标(m)": ["Z坐标(m)", "Z坐标", "安装标高(m)", "标高(m)", "z", "z坐标"],
    "重量(t)": ["重量(t)", "重量", "构件重量(t)", "weight_t", "weight", "tonnage"],
    "安装时长(min)": [
        "安装时长(min)",
        "安装时长",
        "预计安装(min)",
        "吊装工时(min)",
        "duration_min",
        "duration",
    ],
    "构件长(m)": ["构件长(m)", "构件长", "长度(m)", "length_m", "length"],
    "构件宽(m)": ["构件宽(m)", "构件宽", "宽度(m)", "width_m", "width"],
    "构件高(m)": ["构件高(m)", "构件高", "高度(m)", "height_m", "height"],
    "重心高度(m)": ["重心高度(m)", "重心高度", "center_of_gravity_m", "cog_z"],
    "障碍等级": ["障碍等级", "障碍复杂度", "obstacle_level", "obstacle"],
    "作业可视性": ["作业可视性", "可视条件", "visibility_level", "visibility"],
    "地基条件": ["地基条件", "地基等级", "ground_level", "ground"],
    "旋转要求": ["旋转要求", "是否翻身", "rotation_required", "rotation"],
    "风险指数": ["风险指数", "风险值", "risk", "risk_index"],
    "前置构件编号": [
        "前置构件编号",
        "前置构件",
        "前置构件ID",
        "predecessors",
        "predecessor_ids",
        "dependency",
    ],
}

TYPE_ALIASES = {
    "底板": "bottom_slab",
    "底板构件": "bottom_slab",
    "bottom_slab": "bottom_slab",
    "预制立柱": "column",
    "立柱": "column",
    "column": "column",
    "侧墙": "side_wall",
    "预制侧墙": "side_wall",
    "side_wall": "side_wall",
    "中纵梁": "beam",
    "纵梁": "beam",
    "beam": "beam",
    "中板": "middle_slab",
    "中层板": "middle_slab",
    "middle_slab": "middle_slab",
    "站台板": "platform_slab",
    "platform_slab": "platform_slab",
    "顶板": "roof_slab",
    "顶板构件": "roof_slab",
    "roof_slab": "roof_slab",
    "楼梯段": "stair",
    "楼梯": "stair",
    "stair": "stair",
}

GEOMETRY_ALIASES = {
    "板": "middle_slab",
    "平板": "middle_slab",
    "密肋板": "middle_slab",
    "梁": "beam",
    "纵梁": "beam",
    "横梁": "beam",
    "柱": "column",
    "立柱": "column",
    "墙": "side_wall",
    "侧墙": "side_wall",
    "拱": "roof_slab",
    "拱板": "roof_slab",
    "顶板": "roof_slab",
    "楼梯": "stair",
    "楼梯段": "stair",
    "底板": "bottom_slab",
    "站台板": "platform_slab",
    "自定义": "middle_slab",
    **{key: key for key in set(TYPE_ALIASES.values())},
}


@dataclass
class ImportValidationResult:
    components: list[Component] | None
    normalized_frame: pd.DataFrame
    errors: pd.DataFrame
    warnings: pd.DataFrame
    summary: dict[str, Any]
    project: dict[str, Any] = field(default_factory=dict)
    relations: pd.DataFrame = field(default_factory=pd.DataFrame)
    equipment: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.components is not None and self.errors.empty


@dataclass
class ImportBundle:
    component_frame: pd.DataFrame
    project_frame: pd.DataFrame | None = None
    relations_frame: pd.DataFrame | None = None
    equipment_frame: pd.DataFrame | None = None
    source_name: str = ""


def template_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "构件编号": "C001",
                "语义类型": "底板",
                "几何族": "底板",
                "构件名称": "底板-01",
                "X坐标(m)": 0.0,
                "Y坐标(m)": 4.0,
                "Z坐标(m)": 0.25,
                "重量(t)": 18.4,
                "安装时长(min)": 42.0,
                "构件长(m)": 12.0,
                "构件宽(m)": 2.4,
                "构件高(m)": 0.2,
                "重心高度(m)": 0.1,
                "障碍等级": 1,
                "作业可视性": 0,
                "地基条件": 0,
                "旋转要求": 0,
                "前置构件编号": "",
            },
            {
                "构件编号": "C002",
                "语义类型": "预制立柱",
                "几何族": "柱",
                "构件名称": "立柱-01",
                "X坐标(m)": 0.0,
                "Y坐标(m)": 4.0,
                "Z坐标(m)": 5.2,
                "重量(t)": 12.8,
                "安装时长(min)": 35.0,
                "构件长(m)": 1.2,
                "构件宽(m)": 1.2,
                "构件高(m)": 5.8,
                "重心高度(m)": 2.9,
                "障碍等级": 2,
                "作业可视性": 1,
                "地基条件": 0,
                "旋转要求": 1,
                "前置构件编号": "C001",
            },
            {
                "构件编号": "C003",
                "语义类型": "顶板",
                "几何族": "拱板",
                "构件名称": "顶板-01",
                "X坐标(m)": 0.0,
                "Y坐标(m)": 4.0,
                "Z坐标(m)": 9.7,
                "重量(t)": 16.7,
                "安装时长(min)": 40.0,
                "构件长(m)": 12.0,
                "构件宽(m)": 2.4,
                "构件高(m)": 0.3,
                "重心高度(m)": 0.15,
                "障碍等级": 2,
                "作业可视性": 1,
                "地基条件": 0,
                "旋转要求": 0,
                "前置构件编号": "C002",
            },
        ]
    )


def template_instructions_frame() -> pd.DataFrame:
    rows = [
        {
            "字段": "构件编号",
            "必填": "是",
            "格式": "文本，全局唯一",
            "示例": "C001",
            "说明": "用于吊装调度、前置关系和结果回写",
        },
        {
            "字段": "语义类型",
            "必填": "是",
            "格式": "项目实际构件名称",
            "示例": "顶纵梁、中纵梁、仰拱、PC立柱",
            "说明": "允许不同项目使用自己的构件名称",
        },
        {
            "字段": "几何族",
            "必填": "否",
            "格式": "板、梁、柱、墙、拱板、楼梯、自定义",
            "示例": "梁",
            "说明": "不填写时系统根据语义类型自动推断",
        },
        {
            "字段": "构件名称",
            "必填": "是",
            "格式": "文本",
            "示例": "底板-01",
            "说明": "面向现场识别的名称",
        },
        {
            "字段": "X坐标(m)、Y坐标(m)、Z坐标(m)",
            "必填": "是",
            "格式": "数值，单位 m",
            "示例": "0、4、0.25",
            "说明": "统一使用同一工程坐标系和标高基准",
        },
        {
            "字段": "重量(t)",
            "必填": "是",
            "格式": "正数，单位 t",
            "示例": "18.4",
            "说明": "用于吊车能力校核",
        },
        {
            "字段": "安装时长(min)",
            "必填": "是",
            "格式": "正数，单位 min",
            "示例": "42",
            "说明": "包含对位、就位和临时固定时间",
        },
        {
            "字段": "构件长(m)、构件宽(m)、构件高(m)",
            "必填": "是",
            "格式": "正数，单位 m",
            "示例": "12.0、2.4、0.2",
            "说明": "用于空间碰撞、净空和迎风面积计算",
        },
        {
            "字段": "障碍等级",
            "必填": "是",
            "格式": "整数 0-3",
            "示例": "2",
            "说明": "0 无显著障碍，1 一般，2 较复杂，3 高风险受限空间",
        },
        {
            "字段": "作业可视性",
            "必填": "是",
            "格式": "整数 0-2",
            "示例": "1",
            "说明": "0 直接可见，1 部分遮挡，2 盲吊",
        },
        {
            "字段": "地基条件",
            "必填": "是",
            "格式": "整数 0-2",
            "示例": "0",
            "说明": "0 良好，1 一般，2 较差",
        },
        {
            "字段": "重心高度(m)",
            "必填": "否",
            "格式": "数值，单位 m",
            "示例": "2.9",
            "说明": "异形构件建议填写，不填时按构件高度的一半计算",
        },
        {
            "字段": "旋转要求",
            "必填": "否",
            "格式": "0 或 1",
            "示例": "1",
            "说明": "0 不需要翻身，1 需要旋转或翻身",
        },
        {
            "字段": "风险指数",
            "必填": "否",
            "格式": "系统计算字段",
            "示例": "不建议填写",
            "说明": "如保留该列，系统会提示并忽略人工值，改用原始参数自动计算",
        },
        {
            "字段": "前置构件编号",
            "必填": "否",
            "格式": "多个编号用分号分隔",
            "示例": "C001;C002",
            "说明": "节点必须先于当前构件安装，系统会检查缺失引用和循环依赖",
        },
    ]
    return pd.DataFrame(rows)


def project_template_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"参数": "项目编号", "值": "SZ-METRO-001", "单位": "", "说明": "项目唯一编号"},
            {"参数": "项目名称", "值": "示范装配式地铁车站", "单位": "", "说明": "项目名称"},
            {"参数": "车站形式", "值": "岛式", "单位": "", "说明": "岛式、侧式或混合"},
            {"参数": "施工方法", "值": "明挖法", "单位": "", "说明": "明挖、盖挖或暗挖"},
            {"参数": "结构形式", "值": "拱形装配式", "单位": "", "说明": "结构体系"},
            {"参数": "长度", "值": "126", "单位": "m", "说明": "车站设计长度"},
            {"参数": "宽度", "值": "24.8", "单位": "m", "说明": "车站设计宽度"},
            {"参数": "高度", "值": "12.4", "单位": "m", "说明": "结构总高度"},
            {"参数": "坐标原点X", "值": "0", "单位": "m", "说明": "模型坐标系原点"},
            {"参数": "坐标原点Y", "值": "0", "单位": "m", "说明": "模型坐标系原点"},
            {"参数": "坐标原点Z", "值": "0", "单位": "m", "说明": "模型坐标零点"},
            {"参数": "施工环数量", "值": "10", "单位": "环", "说明": "可选"},
            {"参数": "施工仓段数量", "值": "3", "单位": "段", "说明": "可选"},
        ]
    )


def relations_template_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "当前构件编号": "C002",
                "关系类型": "支承",
                "目标构件编号": "C001",
                "最小等待时间(min)": 0,
                "是否强制": 1,
            },
            {
                "当前构件编号": "C003",
                "关系类型": "前置",
                "目标构件编号": "C002",
                "最小等待时间(min)": 60,
                "是否强制": 1,
            },
        ]
    )


def equipment_template_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "设备编号": "GM-01",
                "设备类型": "龙门吊",
                "额定能力(t)": 160,
                "轨道起点X": -10,
                "轨道终点X": 136,
                "小车最小Y": -2,
                "小车最大Y": 27,
                "最大起升高度(m)": 20,
                "大车速度(m/min)": 20,
                "小车速度(m/min)": 12,
                "吊钩速度(m/min)": 8,
                "台班费率(元/h)": 520,
                "最小安全间距(m)": 9,
            },
            {
                "设备编号": "GM-02",
                "设备类型": "龙门吊",
                "额定能力(t)": 130,
                "轨道起点X": -10,
                "轨道终点X": 136,
                "小车最小Y": -2,
                "小车最大Y": 27,
                "最大起升高度(m)": 20,
                "大车速度(m/min)": 20,
                "小车速度(m/min)": 12,
                "吊钩速度(m/min)": 8,
                "台班费率(元/h)": 445,
                "最小安全间距(m)": 9,
            },
        ]
    )


def build_excel_template() -> BytesIO:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        project_template_frame().to_excel(writer, sheet_name="项目参数", index=False)
        template_frame().to_excel(writer, sheet_name="构件数据", index=False)
        relations_template_frame().to_excel(writer, sheet_name="构件关系", index=False)
        equipment_template_frame().to_excel(writer, sheet_name="设备数据", index=False)
        template_instructions_frame().to_excel(writer, sheet_name="填写说明", index=False)
        worksheet = writer.sheets["构件数据"]
        widths = {
            "A": 16,
            "B": 14,
            "C": 16,
            "D": 13,
            "E": 13,
            "F": 13,
            "G": 12,
            "H": 16,
            "I": 12,
            "J": 24,
            "K": 13,
            "L": 13,
            "M": 13,
            "N": 14,
            "O": 13,
            "P": 13,
            "Q": 12,
            "R": 12,
            "S": 12,
        }
        for column, width in widths.items():
            worksheet.column_dimensions[column].width = width
    buffer.seek(0)
    return buffer


def validate_import_frame(
    frame: pd.DataFrame,
    source_name: str = "上传文件",
    project_frame: pd.DataFrame | None = None,
    relations_frame: pd.DataFrame | None = None,
    equipment_frame: pd.DataFrame | None = None,
) -> ImportValidationResult:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    project = _parse_project_frame(project_frame, warnings)
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    normalized = normalized.dropna(how="all").reset_index(drop=True)
    normalized.insert(0, "_原表行号", normalized.index + 2)

    alias_lookup: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_lookup[_normalize_header(alias)] = canonical

    rename_map: dict[str, str] = {}
    unknown_columns: list[str] = []
    for column in normalized.columns:
        if column == "_原表行号":
            continue
        canonical = alias_lookup.get(_normalize_header(column))
        if canonical:
            if canonical in rename_map.values():
                errors.append(
                    _issue(
                        1,
                        canonical,
                        "字段重复",
                        column,
                        "删除或合并重复列后重新导入",
                    )
                )
            else:
                rename_map[column] = canonical
        else:
            unknown_columns.append(column)
    normalized = normalized.rename(columns=rename_map)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing_columns:
        for column in missing_columns:
            errors.append(
                _issue(
                    1,
                    column,
                    "缺少必填列",
                    "",
                    "按模板补充该列后重新导入",
                )
            )
        return _result(
            components=None,
            normalized=normalized,
            errors=errors,
            warnings=warnings,
            source_name=source_name,
            total_rows=len(normalized),
            project=project,
        )

    if unknown_columns:
        warnings.append(
            _issue(
                1,
                "扩展字段",
                "存在未识别列，分析时将忽略",
                "; ".join(unknown_columns),
                "如需纳入计算，请先映射到标准字段",
            )
        )

    for column in REQUIRED_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = np.nan
    for column, default_value in (
        ("前置构件编号", ""),
        ("重心高度(m)", np.nan),
        ("旋转要求", 0),
    ):
        if column not in normalized.columns:
            normalized[column] = default_value

    if "风险指数" in normalized.columns:
        warnings.append(
            _issue(
                1,
                "风险指数",
                "检测到人工风险指数，系统将忽略并依据原始参数重新计算",
                "整列",
                "删除该列可直接使用新版模板",
            )
        )

    required_numeric_columns = [
        "X坐标(m)",
        "Y坐标(m)",
        "Z坐标(m)",
        "重量(t)",
        "安装时长(min)",
        "构件长(m)",
        "构件宽(m)",
        "构件高(m)",
        "障碍等级",
        "作业可视性",
        "地基条件",
    ]
    for column in required_numeric_columns:
        converted = pd.to_numeric(normalized[column], errors="coerce")
        invalid_mask = normalized[column].notna() & converted.isna()
        for row_index in normalized.index[invalid_mask]:
            errors.append(
                _issue(
                    int(normalized.at[row_index, "_原表行号"]),
                    column,
                    "不是有效数值",
                    normalized.at[row_index, column],
                    "改为数字格式",
                )
            )
        missing_mask = converted.isna()
        for row_index in normalized.index[missing_mask]:
            errors.append(
                _issue(
                    int(normalized.at[row_index, "_原表行号"]),
                    column,
                    "不能为空",
                    normalized.at[row_index, column],
                    "填写有效数值",
                )
            )
        normalized[column] = converted

    for column in ("重心高度(m)", "旋转要求"):
        converted = pd.to_numeric(normalized[column], errors="coerce")
        invalid_mask = normalized[column].notna() & converted.isna()
        for row_index in normalized.index[invalid_mask]:
            errors.append(
                _issue(
                    int(normalized.at[row_index, "_原表行号"]),
                    column,
                    "不是有效数值",
                    normalized.at[row_index, column],
                    "改为数字格式或留空",
                )
            )
        normalized[column] = converted

    id_values: list[str] = []
    predecessor_values: list[list[str]] = []
    type_values: list[str] = []
    semantic_values: list[str] = []
    for row_index in normalized.index:
        row_number = int(normalized.at[row_index, "_原表行号"])
        component_id = _normalize_identifier(normalized.at[row_index, "构件编号"])
        component_name = _normalize_text(normalized.at[row_index, "构件名称"])
        raw_semantic_type = _normalize_text(
            normalized.at[row_index, "语义类型"]
        )
        raw_geometry_family = _normalize_text(
            normalized.get(
                "几何族",
                pd.Series("", index=normalized.index),
            ).at[row_index]
        )
        semantic_type = raw_semantic_type
        inferred_family = TYPE_ALIASES.get(
            raw_semantic_type.lower(),
            TYPE_ALIASES.get(raw_semantic_type),
        )
        component_type = (
            GEOMETRY_ALIASES.get(
                raw_geometry_family.lower(),
                GEOMETRY_ALIASES.get(raw_geometry_family),
            )
            if raw_geometry_family
            else inferred_family
        )
        predecessors = _split_identifiers(normalized.at[row_index, "前置构件编号"])

        if not component_id:
            errors.append(
                _issue(row_number, "构件编号", "不能为空", "", "填写唯一构件编号")
            )
        if not component_name:
            errors.append(
                _issue(row_number, "构件名称", "不能为空", "", "填写构件名称")
            )
        if not raw_semantic_type:
            errors.append(
                _issue(row_number, "语义类型", "不能为空", "", "填写项目构件名称")
            )
        if raw_geometry_family and component_type is None:
            errors.append(
                _issue(
                    row_number,
                    "几何族",
                    "几何族未识别",
                    raw_geometry_family,
                    "填写板、梁、柱、墙、拱板、楼梯或自定义",
                )
            )
        elif not raw_geometry_family and component_type is None:
            errors.append(
                _issue(
                    row_number,
                    "语义类型",
                    "无法自动判断几何族",
                    raw_semantic_type,
                    "补充几何族字段",
                )
            )

        weight = normalized.at[row_index, "重量(t)"]
        duration = normalized.at[row_index, "安装时长(min)"]
        x_value = normalized.at[row_index, "X坐标(m)"]
        y_value = normalized.at[row_index, "Y坐标(m)"]
        z_value = normalized.at[row_index, "Z坐标(m)"]
        length_value = normalized.at[row_index, "构件长(m)"]
        width_value = normalized.at[row_index, "构件宽(m)"]
        height_value = normalized.at[row_index, "构件高(m)"]
        center_of_gravity = normalized.at[row_index, "重心高度(m)"]
        obstacle_level = normalized.at[row_index, "障碍等级"]
        visibility_level = normalized.at[row_index, "作业可视性"]
        ground_level = normalized.at[row_index, "地基条件"]
        rotation_value = normalized.at[row_index, "旋转要求"]

        if pd.notna(weight) and not 0.1 <= float(weight) <= 500.0:
            errors.append(
                _issue(row_number, "重量(t)", "超出允许范围 0.1-500", weight, "复核构件重量")
            )
        if pd.notna(duration) and not 1.0 <= float(duration) <= 1440.0:
            errors.append(
                _issue(
                    row_number,
                    "安装时长(min)",
                    "超出允许范围 1-1440",
                    duration,
                    "复核单件安装时长",
                )
            )
        if pd.notna(z_value) and float(z_value) < 0.0:
            errors.append(
                _issue(row_number, "Z坐标(m)", "不得小于 0", z_value, "检查标高基准")
            )
        for coordinate_name, coordinate_value in (
            ("X坐标(m)", x_value),
            ("Y坐标(m)", y_value),
        ):
            if pd.notna(coordinate_value) and not -10000.0 <= float(coordinate_value) <= 10000.0:
                errors.append(
                    _issue(
                        row_number,
                        coordinate_name,
                        "坐标超出允许范围",
                        coordinate_value,
                        "核对工程坐标系",
                    )
                )

        for dimension_name, dimension_value in (
            ("构件长(m)", length_value),
            ("构件宽(m)", width_value),
            ("构件高(m)", height_value),
        ):
            if pd.notna(dimension_value) and not 0.05 <= float(dimension_value) <= 200.0:
                errors.append(
                    _issue(
                        row_number,
                        dimension_name,
                        "超出允许范围 0.05-200",
                        dimension_value,
                        "按构件实际外廓尺寸填写",
                    )
                )
        for level_name, level_value, maximum in (
            ("障碍等级", obstacle_level, 3),
            ("作业可视性", visibility_level, 2),
            ("地基条件", ground_level, 2),
        ):
            if pd.notna(level_value) and (
                float(level_value) != int(float(level_value))
                or not 0 <= int(float(level_value)) <= maximum
            ):
                errors.append(
                    _issue(
                        row_number,
                        level_name,
                        f"超出允许范围 0-{maximum}",
                        level_value,
                        "填写模板规定的整数等级",
                    )
                )
        if pd.notna(center_of_gravity) and pd.notna(height_value):
            if not 0.0 <= float(center_of_gravity) <= float(height_value) * 1.5:
                errors.append(
                    _issue(
                        row_number,
                        "重心高度(m)",
                        "超出构件高度允许范围",
                        center_of_gravity,
                        "按构件底部起算填写重心高度",
                    )
                )
        if pd.notna(rotation_value) and float(rotation_value) not in (0.0, 1.0):
            errors.append(
                _issue(
                    row_number,
                    "旋转要求",
                    "只能填写 0 或 1",
                    rotation_value,
                    "0 表示不需要，1 表示需要",
                )
            )

        id_values.append(component_id)
        predecessor_values.append(predecessors)
        type_values.append(component_type or "")
        semantic_values.append(semantic_type)

    id_to_indices: dict[str, list[int]] = {}
    for index, component_id in enumerate(id_values):
        id_to_indices.setdefault(component_id, []).append(index)
    duplicate_ids = {
        component_id: indices
        for component_id, indices in id_to_indices.items()
        if component_id and len(indices) > 1
    }
    for component_id, indices in duplicate_ids.items():
        for index in indices:
            row_number = int(normalized.at[index, "_原表行号"])
            errors.append(
                _issue(
                    row_number,
                    "构件编号",
                    "构件编号重复",
                    component_id,
                    "确保每个构件编号全局唯一",
                )
            )

    known_ids = set(id_to_indices)
    predecessor_indices: list[list[int]] = []
    for index, predecessor_ids in enumerate(predecessor_values):
        row_number = int(normalized.at[index, "_原表行号"])
        current_id = id_values[index]
        current_predecessors: list[int] = []
        for predecessor_id in predecessor_ids:
            if predecessor_id not in known_ids:
                errors.append(
                    _issue(
                        row_number,
                        "前置构件编号",
                        "引用了不存在的构件",
                        predecessor_id,
                        "改用同一文件中的有效构件编号",
                    )
                )
            elif predecessor_id == current_id:
                errors.append(
                    _issue(
                        row_number,
                        "前置构件编号",
                        "构件不能依赖自身",
                        predecessor_id,
                        "删除自引用",
                    )
                )
            else:
                current_predecessors.append(id_to_indices[predecessor_id][0])
        predecessor_indices.append(list(dict.fromkeys(current_predecessors)))

    relation_predecessors, normalized_relations, relation_errors, relation_warnings = (
        _validate_relations_frame(relations_frame, id_to_indices)
    )
    errors.extend(relation_errors)
    warnings.extend(relation_warnings)
    for index, extra_predecessors in enumerate(relation_predecessors):
        if index >= len(predecessor_indices):
            break
        predecessor_indices[index] = list(
            dict.fromkeys(predecessor_indices[index] + extra_predecessors)
        )

    equipment, equipment_errors, equipment_warnings = _validate_equipment_frame(
        equipment_frame,
        component_frame=normalized,
    )
    errors.extend(equipment_errors)
    warnings.extend(equipment_warnings)

    if errors:
        return _result(
            components=None,
            normalized=normalized,
            errors=errors,
            warnings=warnings,
            source_name=source_name,
            total_rows=len(normalized),
            project=project,
            relations=normalized_relations,
            equipment=equipment,
        )

    components: list[Component] = []
    for index in range(len(normalized)):
        weight_t = float(normalized.at[index, "重量(t)"])
        length_m = float(normalized.at[index, "构件长(m)"])
        width_m = float(normalized.at[index, "构件宽(m)"])
        height_m = float(normalized.at[index, "构件高(m)"])
        center_of_gravity = normalized.at[index, "重心高度(m)"]
        center_of_gravity_m = (
            float(center_of_gravity)
            if pd.notna(center_of_gravity)
            else height_m * 0.5
        )
        obstacle_level = int(normalized.at[index, "障碍等级"])
        visibility_level = int(normalized.at[index, "作业可视性"])
        ground_level = int(normalized.at[index, "地基条件"])
        rotation_value = normalized.at[index, "旋转要求"]
        rotation_required = bool(
            int(float(rotation_value)) if pd.notna(rotation_value) else 0
        )
        components.append(
            Component(
                id=id_values[index],
                type=type_values[index],
                name=_normalize_text(normalized.at[index, "构件名称"]),
                semantic_type=semantic_values[index],
                x=float(normalized.at[index, "X坐标(m)"]),
                y=float(normalized.at[index, "Y坐标(m)"]),
                z=float(normalized.at[index, "Z坐标(m)"]),
                weight_t=weight_t,
                duration_min=float(normalized.at[index, "安装时长(min)"]),
                risk=estimate_static_risk(
                    length_m=length_m,
                    width_m=width_m,
                    height_m=height_m,
                    weight_t=weight_t,
                    obstacle_level=obstacle_level,
                    visibility_level=visibility_level,
                    ground_level=ground_level,
                    rotation_required=rotation_required,
                ),
                length_m=length_m,
                width_m=width_m,
                height_m=height_m,
                obstacle_level=obstacle_level,
                visibility_level=visibility_level,
                ground_level=ground_level,
                center_of_gravity_m=center_of_gravity_m,
                rotation_required=rotation_required,
                predecessors=tuple(predecessor_indices[index]),
            )
        )

    cycle_indices = _find_cycle_nodes(components)
    if cycle_indices:
        for index in cycle_indices:
            errors.append(
                _issue(
                    int(normalized.at[index, "_原表行号"]),
                    "前置构件编号",
                    "前置关系形成循环",
                    "",
                    "整理依赖关系，确保所有前置约束可以拓扑排序",
                )
            )
        return _result(
            components=None,
            normalized=normalized,
            errors=errors,
            warnings=warnings,
            source_name=source_name,
            total_rows=len(normalized),
            project=project,
            relations=normalized_relations,
            equipment=equipment,
        )

    return _result(
        components=components,
        normalized=normalized,
        errors=errors,
        warnings=warnings,
        source_name=source_name,
        total_rows=len(normalized),
        project=project,
        relations=normalized_relations,
        equipment=equipment,
    )


def read_import_file(uploaded_file) -> ImportBundle:
    file_name = uploaded_file.name.lower()
    raw_bytes = (
        uploaded_file.getvalue()
        if hasattr(uploaded_file, "getvalue")
        else uploaded_file.read()
    )
    if file_name.endswith(".csv"):
        component_frame = _read_csv_frame(raw_bytes)
        return ImportBundle(
            component_frame=component_frame,
            source_name=uploaded_file.name,
        )
    if file_name.endswith(".xlsx"):
        sheets = pd.read_excel(
            BytesIO(raw_bytes),
            sheet_name=None,
            engine="openpyxl",
        )
        return ImportBundle(
            component_frame=_find_sheet(
                sheets,
                ["构件数据", "构件", "Components"],
            ),
            project_frame=_find_sheet(
                sheets,
                ["项目参数", "项目信息", "Project"],
                required=False,
            ),
            relations_frame=_find_sheet(
                sheets,
                ["构件关系", "关系数据", "Relations"],
                required=False,
            ),
            equipment_frame=_find_sheet(
                sheets,
                ["设备数据", "吊装设备", "Equipment"],
                required=False,
            ),
            source_name=uploaded_file.name,
        )
    raise ValueError("仅支持 CSV 或 XLSX 文件")


def _read_csv_frame(raw_bytes: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "utf-16"):
        try:
            return pd.read_csv(BytesIO(raw_bytes), encoding=encoding)
        except (UnicodeDecodeError, UnicodeError) as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise ValueError("无法识别 CSV 文件编码")


def validate_import_bundle(
    bundle: ImportBundle,
) -> ImportValidationResult:
    return validate_import_frame(
        bundle.component_frame,
        source_name=bundle.source_name,
        project_frame=bundle.project_frame,
        relations_frame=bundle.relations_frame,
        equipment_frame=bundle.equipment_frame,
    )


def _find_sheet(
    sheets: dict[str, pd.DataFrame],
    aliases: Sequence[str],
    required: bool = True,
) -> pd.DataFrame | None:
    for alias in aliases:
        for sheet_name, frame in sheets.items():
            if str(sheet_name).strip().lower() == alias.lower():
                return frame
    if required:
        raise ValueError(f"缺少工作表：{aliases[0]}")
    return None


def _parse_project_frame(
    frame: pd.DataFrame | None,
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    normalized_columns = {_normalize_header(column): column for column in frame.columns}
    parameter_column = _first_matching_column(
        normalized_columns,
        ["参数", "参数名称", "项目参数", "parameter", "key", "name"],
    )
    value_column = _first_matching_column(
        normalized_columns,
        ["值", "参数值", "value"],
    )
    unit_column = _first_matching_column(
        normalized_columns,
        ["单位", "unit"],
    )
    if parameter_column is None or value_column is None:
        warnings.append(
            _issue(
                1,
                "项目参数",
                "参数表缺少“参数”或“值”列，已忽略项目参数",
                "",
                "使用模板中的项目参数表格式",
            )
        )
        return {}
    project: dict[str, Any] = {"_units": {}}
    for row_index, row in frame.iterrows():
        key = _clean_project_key(row.get(parameter_column))
        if not key:
            continue
        value = _clean_project_value(row.get(value_column))
        project[key] = value
        if unit_column is not None:
            project["_units"][key] = _normalize_text(row.get(unit_column))
    return normalize_project_metadata(project)


def normalize_project_metadata(project: dict[str, Any]) -> dict[str, Any]:
    """Clean project metadata imported from files or older saved sessions."""
    normalized: dict[str, Any] = {}
    units: dict[str, str] = {}
    for key, value in project.items():
        if str(key) == "_units":
            continue
        clean_key = _clean_project_key(key)
        if not clean_key or clean_key.startswith("_"):
            continue
        normalized[clean_key] = _clean_project_value(value)
    raw_units = project.get("_units")
    if isinstance(raw_units, dict):
        for key, value in raw_units.items():
            clean_key = _clean_project_key(key)
            if clean_key:
                units[clean_key] = _normalize_text(value)
    if units:
        normalized["_units"] = units
    return normalized


def _validate_relations_frame(
    frame: pd.DataFrame | None,
    id_to_indices: dict[str, list[int]],
) -> tuple[list[list[int]], pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    predecessors: list[list[int]] = [[] for _ in id_to_indices]
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    columns = [
        "当前构件编号",
        "关系类型",
        "目标构件编号",
        "最小等待时间(min)",
        "是否强制",
    ]
    if frame is None or frame.empty:
        return predecessors, pd.DataFrame(columns=columns), errors, warnings
    normalized_columns = {_normalize_header(column): column for column in frame.columns}
    current_column = _first_matching_column(
        normalized_columns,
        ["当前构件编号", "构件编号", "component_id"],
    )
    relation_column = _first_matching_column(
        normalized_columns,
        ["关系类型", "relation_type", "relation"],
    )
    target_column = _first_matching_column(
        normalized_columns,
        ["目标构件编号", "目标构件", "target_id"],
    )
    wait_column = _first_matching_column(
        normalized_columns,
        ["最小等待时间(min)", "等待时间", "wait_minutes"],
    )
    mandatory_column = _first_matching_column(
        normalized_columns,
        ["是否强制", "强制", "mandatory"],
    )
    if current_column is None or relation_column is None or target_column is None:
        errors.append(
            _issue(
                1,
                "构件关系",
                "缺少必需列",
                "",
                "需要当前构件编号、关系类型和目标构件编号",
            )
        )
        return predecessors, pd.DataFrame(columns=columns), errors, warnings

    normalized_rows: list[dict[str, Any]] = []
    known_ids = set(id_to_indices)
    for row_index, row in frame.iterrows():
        row_number = int(row_index) + 2
        current_id = _normalize_identifier(row.get(current_column))
        target_id = _normalize_identifier(row.get(target_column))
        relation_type = _normalize_text(row.get(relation_column))
        wait_value = (
            pd.to_numeric(row.get(wait_column), errors="coerce")
            if wait_column is not None
            else 0
        )
        mandatory_value = (
            _normalize_text(row.get(mandatory_column))
            if mandatory_column is not None
            else "1"
        )
        mandatory = mandatory_value.lower() not in {"0", "false", "否", "no"}
        if current_id not in known_ids:
            errors.append(
                _issue(
                    row_number,
                    "当前构件编号",
                    "构件不存在",
                    current_id,
                    "改为构件数据表中的有效编号",
                )
            )
            continue
        if target_id not in known_ids:
            errors.append(
                _issue(
                    row_number,
                    "目标构件编号",
                    "构件不存在",
                    target_id,
                    "改为构件数据表中的有效编号",
                )
            )
            continue
        if relation_type in {"前置", "支承", "连接", "闭合"} and mandatory:
            predecessors[id_to_indices[current_id][0]].append(
                id_to_indices[target_id][0]
            )
        elif relation_type == "对称":
            warnings.append(
                _issue(
                    row_number,
                    "关系类型",
                    "对称关系已记录，当前求解器暂不强制左右同步",
                    relation_type,
                    "后续可升级为对称约束",
                )
            )
        elif mandatory:
            warnings.append(
                _issue(
                    row_number,
                    "关系类型",
                    "关系类型未参与算法约束",
                    relation_type,
                    "可改用前置、支承、连接或闭合",
                )
            )
        if pd.notna(wait_value) and float(wait_value) > 0:
            warnings.append(
                _issue(
                    row_number,
                    "最小等待时间(min)",
                    "等待时间已记录，当前仅作为拓扑依赖",
                    wait_value,
                    "后续可增加强度和养护等待调度",
                )
            )
        normalized_rows.append(
            {
                "当前构件编号": current_id,
                "关系类型": relation_type,
                "目标构件编号": target_id,
                "最小等待时间(min)": wait_value,
                "是否强制": 1 if mandatory else 0,
            }
        )
    return (
        [list(dict.fromkeys(values)) for values in predecessors],
        pd.DataFrame(normalized_rows, columns=columns),
        errors,
        warnings,
    )


def _validate_equipment_frame(
    frame: pd.DataFrame | None,
    component_frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if frame is None or frame.empty:
        return [], errors, warnings
    normalized_columns = {_normalize_header(column): column for column in frame.columns}
    id_column = _first_matching_column(normalized_columns, ["设备编号", "equipment_id"])
    capacity_column = _first_matching_column(
        normalized_columns,
        ["额定能力(t)", "额定能力", "capacity_t"],
    )
    if id_column is None or capacity_column is None:
        errors.append(
            _issue(
                1,
                "设备数据",
                "缺少设备编号或额定能力",
                "",
                "至少填写设备编号和额定能力",
            )
        )
        return [], errors, warnings

    minimum_x = float(
        pd.to_numeric(component_frame.get("X坐标(m)", pd.Series()), errors="coerce").min()
    )
    maximum_x = float(
        pd.to_numeric(component_frame.get("X坐标(m)", pd.Series()), errors="coerce").max()
    )
    minimum_y = float(
        pd.to_numeric(component_frame.get("Y坐标(m)", pd.Series()), errors="coerce").min()
    )
    maximum_y = float(
        pd.to_numeric(component_frame.get("Y坐标(m)", pd.Series()), errors="coerce").max()
    )
    maximum_z = float(
        (
            pd.to_numeric(component_frame.get("Z坐标(m)", pd.Series()), errors="coerce")
            + pd.to_numeric(
                component_frame.get("构件高(m)", pd.Series()),
                errors="coerce",
            )
            * 0.5
        ).max()
    )
    defaults = {
        "轨道起点X": minimum_x - 8.0,
        "轨道终点X": maximum_x + 8.0,
        "小车最小Y": minimum_y - 2.0,
        "小车最大Y": maximum_y + 2.0,
        "最大起升高度(m)": maximum_z + 6.0,
        "大车速度(m/min)": 20.0,
        "小车速度(m/min)": 12.0,
        "吊钩速度(m/min)": 8.0,
        "台班费率(元/h)": 500.0,
        "最小安全间距(m)": 9.0,
    }
    equipment: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        row_number = int(row_index) + 2
        equipment_id = _normalize_identifier(row.get(id_column))
        if not equipment_id:
            errors.append(
                _issue(row_number, "设备编号", "不能为空", "", "填写唯一设备编号")
            )
            continue
        capacity = pd.to_numeric(row.get(capacity_column), errors="coerce")
        if pd.isna(capacity) or float(capacity) <= 0:
            errors.append(
                _issue(
                    row_number,
                    "额定能力(t)",
                    "必须为正数",
                    row.get(capacity_column),
                    "填写设备额定起重量",
                )
            )
            continue
        values: dict[str, float] = {}
        for key, default in defaults.items():
            column = _first_matching_column(normalized_columns, [key])
            raw_value = row.get(column) if column is not None else default
            numeric = pd.to_numeric(raw_value, errors="coerce")
            if pd.isna(numeric):
                numeric = default
            values[key] = float(numeric)
        equipment.append(
            {
                "id": equipment_id,
                "type": "龙门吊",
                "capacity_t": float(capacity),
                "rate_yuan_h": values["台班费率(元/h)"],
                "position": (
                    values["轨道起点X"],
                    (values["小车最小Y"] + values["小车最大Y"]) * 0.5,
                ),
                "initial_x": values["轨道起点X"],
                "initial_y": (values["小车最小Y"] + values["小车最大Y"]) * 0.5,
                "rail_start_x": values["轨道起点X"],
                "rail_end_x": values["轨道终点X"],
                "trolley_min_y": values["小车最小Y"],
                "trolley_max_y": values["小车最大Y"],
                "max_lift_height": values["最大起升高度(m)"],
                "long_speed_m_min": values["大车速度(m/min)"],
                "trolley_speed_m_min": values["小车速度(m/min)"],
                "hoist_speed_m_min": values["吊钩速度(m/min)"],
                "minimum_spacing_m": values["最小安全间距(m)"],
            }
        )
    for index, item in enumerate(equipment):
        rail_length = item["rail_end_x"] - item["rail_start_x"]
        item["initial_x"] = (
            item["rail_start_x"]
            + rail_length * (index + 1) / (len(equipment) + 1)
        )
        item["initial_y"] = (
            item["trolley_min_y"] + item["trolley_max_y"]
        ) * 0.5
        item["position"] = (item["initial_x"], item["initial_y"])
    if not equipment:
        warnings.append(
            _issue(
                1,
                "设备数据",
                "未解析到有效设备，将使用界面参数自动生成",
                "",
                "可删除设备表或补齐设备数据",
            )
        )
    return equipment, errors, warnings


def _first_matching_column(
    normalized_columns: dict[str, str],
    aliases: Sequence[str],
) -> str | None:
    for alias in aliases:
        match = normalized_columns.get(_normalize_header(alias))
        if match is not None:
            return match
    return None


def _find_cycle_nodes(components: Sequence[Component]) -> list[int]:
    successors = component_successors(components)
    indegree = np.asarray([len(component.predecessors) for component in components], dtype=int)
    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    visited: list[int] = []
    while ready:
        current = ready.pop()
        visited.append(current)
        for successor in successors[current]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(visited) == len(components):
        return []
    visited_set = set(visited)
    return [index for index in range(len(components)) if index not in visited_set]


def _result(
    components: list[Component] | None,
    normalized: pd.DataFrame,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    source_name: str,
    total_rows: int,
    project: dict[str, Any] | None = None,
    relations: pd.DataFrame | None = None,
    equipment: list[dict[str, Any]] | None = None,
) -> ImportValidationResult:
    error_frame = pd.DataFrame(
        errors,
        columns=["行号", "字段", "问题", "原值", "处理建议"],
    )
    warning_frame = pd.DataFrame(
        warnings,
        columns=["行号", "字段", "问题", "原值", "处理建议"],
    )
    if components is None:
        total_weight = float(
            pd.to_numeric(normalized.get("重量(t)", pd.Series(dtype=float)), errors="coerce").sum()
        )
        maximum_weight = float(
            pd.to_numeric(normalized.get("重量(t)", pd.Series(dtype=float)), errors="coerce").max()
        ) if len(normalized) else 0.0
        dependency_count = sum(
            len(_split_identifiers(value))
            for value in normalized.get("前置构件编号", pd.Series(dtype=str))
        )
    else:
        total_weight = sum(component.weight_t for component in components)
        maximum_weight = max(component.weight_t for component in components)
        dependency_count = sum(len(component.predecessors) for component in components)
    return ImportValidationResult(
        components=components,
        normalized_frame=normalized,
        errors=error_frame,
        warnings=warning_frame,
        summary={
            "source_name": source_name,
            "total_rows": total_rows,
            "valid_rows": len(components) if components is not None else 0,
            "error_count": len(error_frame),
            "warning_count": len(warning_frame),
            "total_weight_t": round(total_weight, 2),
            "maximum_weight_t": round(maximum_weight, 2),
            "dependency_count": int(dependency_count),
        },
        project=project or {},
        relations=relations if relations is not None else pd.DataFrame(),
        equipment=equipment or [],
    )


def _issue(
    row_number: int,
    field: str,
    problem: str,
    value: Any,
    suggestion: str,
) -> dict[str, Any]:
    if pd.isna(value):
        display_value = ""
    else:
        display_value = str(value)
    return {
        "行号": row_number,
        "字段": field,
        "问题": problem,
        "原值": display_value,
        "处理建议": suggestion,
    }


def _normalize_header(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("（", "(")
        .replace("）", ")")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def _normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lstrip("\ufeff").strip()
    return _repair_mojibake_text(text)


def _clean_project_key(value: Any) -> str:
    key = _normalize_text(value)
    return key.lstrip("\ufeff").strip().strip(":：")


def _clean_project_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, str):
        text = _normalize_text(value)
        try:
            numeric = float(text)
            if numeric.is_integer():
                return int(numeric)
            return numeric
        except ValueError:
            return text
    return value


def _repair_mojibake_text(value: str) -> str:
    if not value or not any(ord(character) > 127 for character in value):
        return value
    best = value
    best_score = _text_quality(value)
    for encoding in ("gb18030", "latin1", "cp1252"):
        try:
            candidate = value.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if _contains_cjk(value) and _contains_cjk(candidate):
            return candidate
        candidate_score = _text_quality(candidate)
        if candidate_score > best_score + 0.5:
            best = candidate
            best_score = candidate_score
    return best


def _contains_cjk(value: str) -> bool:
    return any(0x4E00 <= ord(character) <= 0x9FFF for character in value)


def _text_quality(value: str) -> float:
    score = 0.0
    for character in value:
        codepoint = ord(character)
        if character == "\ufffd":
            score -= 20.0
        elif codepoint < 32 and character not in "\t\r\n":
            score -= 10.0
        elif 0x4E00 <= codepoint <= 0x9FFF:
            score += 3.0
        elif character.isascii():
            score += 0.25
        elif 0x80 <= codepoint <= 0x24F:
            score += 0.1
        else:
            score += 0.5
    return score


def _normalize_identifier(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return text


def _split_identifiers(value: Any) -> list[str]:
    text = _normalize_text(value)
    if not text:
        return []
    for separator in ("；", "，", ",", "|", "/"):
        text = text.replace(separator, ";")
    return [
        normalized
        for item in text.split(";")
        if (normalized := _normalize_identifier(item))
    ]
