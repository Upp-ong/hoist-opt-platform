from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .charts import (
    COLORS,
    build_gantt_figure,
    build_risk_breakdown_figure,
    build_risk_figure,
)
from .optimizer import (
    OptimizationResult,
    _associate_reference_points,
    _generate_reference_points,
)
from .presentation import schedule_frame


EXPERT_OBJECTIVE_NAMES = ["工期", "成本", "空间碰撞", "作业环境", "设备均衡"]
EXPERT_OBJECTIVE_UNITS = ["min", "元", "指数", "指数", "指数"]


def render_expert_metrics(result: OptimizationResult) -> None:
    plan = result.plans["综合推荐"]
    columns = st.columns(5)
    columns[0].metric("计划完成周期", f"{plan['makespan_min'] / 480.0:.1f} 工日")
    columns[1].metric("吊装综合成本", f"{plan['cost_yuan'] / 10000.0:.1f} 万元")
    columns[2].metric("空间碰撞风险", f"{plan['spatial_risk_score']:.1f}")
    columns[3].metric("作业环境风险", f"{plan['environment_risk_score']:.1f}")
    columns[4].metric("设备均衡偏差", f"{plan['equipment_balance_score']:.1f}")


def render_objective_model(result: OptimizationResult) -> None:
    st.markdown(
        '<div class="subsection-heading">专业模式目标体系</div>'
        '<div class="subsection-note">'
        "安全风险拆分为空间碰撞和作业环境，设备负荷作为独立目标。"
        "</div>",
        unsafe_allow_html=True,
    )
    objective_frame = pd.DataFrame(
        [
            {
                "目标": "总工期",
                "含义": "所有预制构件完成安装的最晚时间",
                "单位": "min",
                "方向": "越小越好",
            },
            {
                "目标": "总成本",
                "含义": "龙门吊台班、三轴行走定位和重载附加费用",
                "单位": "元",
                "方向": "越小越好",
            },
            {
                "目标": "空间碰撞风险",
                "含义": "构件邻近、高度变化和安装顺序造成的空间冲突",
                "单位": "0-100",
                "方向": "越小越好",
            },
            {
                "目标": "作业环境风险",
                "含义": "起升高度、障碍、盲吊、地基、风荷载和翻身要求",
                "单位": "0-100",
                "方向": "越小越好",
            },
            {
                "目标": "设备均衡",
                "含义": "多台龙门吊作业时长和行走负荷的不均衡程度",
                "单位": "0-100",
                "方向": "越小越好",
            },
        ]
    )
    st.dataframe(objective_frame, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="subsection-heading">硬约束</div>',
        unsafe_allow_html=True,
    )
    constraints = pd.DataFrame(
        [
            {"约束": "起重能力", "判断": "构件重量 ≤ 龙门吊额定能力"},
            {"约束": "轨道范围", "判断": "构件 X 坐标位于轨道有效范围"},
            {"约束": "小车行程", "判断": "构件 Y 坐标位于小车横向行程"},
            {"约束": "起升高度", "判断": "构件安装高度 ≤ 最大吊钩高度"},
            {"约束": "结构层序", "判断": "前置构件必须先完成"},
            {"约束": "双机安全", "判断": "相邻龙门吊保持最小纵向间距"},
        ]
    )
    st.dataframe(constraints, use_container_width=True, hide_index=True)


def render_pareto_pages(result: OptimizationResult) -> None:
    objectives = result.pareto_objectives
    st.markdown(
        '<div class="subsection-heading">五目标平行坐标</div>'
        '<div class="subsection-note">每条线代表一个非支配吊装调度方案，颜色表示工期。</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _build_parallel_coordinates(objectives, result.plans),
        use_container_width=True,
        config={"displaylogo": False},
    )

    st.markdown(
        '<div class="subsection-heading">主要目标投影</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _build_objective_projections(objectives, result.plans),
        use_container_width=True,
        config={"displaylogo": False},
    )

    st.markdown(
        '<div class="subsection-heading">Pareto 解明细</div>',
        unsafe_allow_html=True,
    )
    frame = pd.DataFrame(objectives, columns=EXPERT_OBJECTIVE_NAMES)
    frame.insert(0, "Pareto序号", range(1, len(frame) + 1))
    frame["工期(工日)"] = (frame["工期"] / 480.0).round(2)
    frame["成本(万元)"] = (frame["成本"] / 10000.0).round(2)
    frame = frame[
        ["Pareto序号", "工期(工日)", "成本(万元)", "空间碰撞", "作业环境", "设备均衡"]
    ].round(2)
    st.dataframe(frame, use_container_width=True, hide_index=True, height=360)


def render_risk_analysis(result: OptimizationResult) -> None:
    plan_name = st.selectbox(
        "分析方案",
        list(result.plans.keys()),
        index=list(result.plans.keys()).index("综合推荐"),
    )
    plan = result.plans[plan_name]
    left, right = st.columns([1.0, 1.0], gap="medium")
    with left:
        st.markdown(
            '<div class="subsection-heading">综合风险组成</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_risk_breakdown_figure(plan),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with right:
        st.markdown(
            '<div class="subsection-heading">高风险构件</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_risk_figure(result.components, plan),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    st.plotly_chart(
        build_gantt_figure(plan["schedule"], max_tasks=36),
        use_container_width=True,
        config={"displaylogo": False},
    )


def render_sensitivity_analysis(result: OptimizationResult) -> None:
    sensitivity, frequency = _weight_sensitivity(result.pareto_objectives)
    left, right = st.columns([1.0, 1.2], gap="medium")
    with left:
        st.markdown(
            '<div class="subsection-heading">权重敏感性</div>'
            '<div class="subsection-note">随机权重下 Pareto 解成为最优方案的比例。</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            go.Figure(
                go.Bar(
                    x=frequency["Pareto序号"].astype(str),
                    y=frequency["入选比例"],
                    marker={"color": COLORS["teal"]},
                    text=[f"{value:.0%}" for value in frequency["入选比例"]],
                    textposition="outside",
                    hovertemplate="Pareto %{x}<br>入选比例 %{y:.1%}<extra></extra>",
                )
            ).update_layout(
                height=350,
                margin={"l": 34, "r": 20, "t": 28, "b": 42},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis={"title": "Pareto 解"},
                yaxis={"title": "入选比例", "tickformat": ".0%", "gridcolor": COLORS["grid"]},
                showlegend=False,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with right:
        st.markdown(
            '<div class="subsection-heading">稳定性结果</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(sensitivity, use_container_width=True, hide_index=True, height=350)


def render_algorithm_audit(result: OptimizationResult) -> None:
    metadata = result.metadata
    columns = st.columns(4)
    columns[0].metric("求解算法", metadata.get("algorithm", "NSGA2"))
    columns[1].metric("种群规模", metadata.get("population_size", 0))
    columns[2].metric("迭代代数", metadata.get("generations", 0))
    columns[3].metric("参考方向", metadata.get("reference_points", 0))

    st.markdown(
        '<div class="subsection-heading">每代最优目标</div>',
        unsafe_allow_html=True,
    )
    history = result.history
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("最短工期", "最低成本", "最低空间碰撞"),
        horizontal_spacing=0.08,
    )
    for column, (key, title, color) in enumerate(
        (
            ("best_time", "工期 (min)", COLORS["teal"]),
            ("best_cost", "成本 (元)", COLORS["orange"]),
            ("best_spatial_risk", "空间碰撞", COLORS["red"]),
        ),
        start=1,
    ):
        figure.add_trace(
            go.Scatter(
                x=history["generation"],
                y=history.get(key, history.get("best_risk", [])),
                mode="lines+markers",
                line={"color": color, "width": 2.4},
                marker={"size": 4},
                hovertemplate="第 %{x} 代<br>%{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(title_text="迭代代数", gridcolor=COLORS["grid"], row=1, col=column)
        figure.update_yaxes(title_text=title, gridcolor=COLORS["grid"], row=1, col=column)
    figure.update_layout(
        height=330,
        margin={"l": 46, "r": 20, "t": 48, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

    reference_points = _generate_reference_points(
        5,
        int(metadata.get("reference_divisions", 3)),
    )
    associations, _ = _associate_reference_points(
        _normalize(result.pareto_objectives),
        reference_points,
    )
    counts = pd.Series(associations).value_counts().sort_index()
    reference_frame = pd.DataFrame(
        {
            "参考方向": [f"R{index + 1:02d}" for index in counts.index],
            "命中解数量": counts.values,
        }
    )
    st.markdown(
        '<div class="subsection-heading">参考方向命中分布</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(reference_frame, use_container_width=True, hide_index=True)


def render_mode_comparison(
    expert_result: OptimizationResult,
    decision_result: OptimizationResult,
) -> None:
    expert_plan = expert_result.plans["综合推荐"]
    decision_plan = decision_result.plans["综合推荐"]
    same_sequence = expert_plan["sequence"] == decision_plan["sequence"]
    comparison = pd.DataFrame(
        [
            {
                "模式": "决策模式",
                "算法": decision_result.metadata.get("algorithm", "NSGA2"),
                "目标数量": 3,
                "工期(工日)": round(decision_plan["makespan_min"] / 480.0, 2),
                "成本(万元)": round(decision_plan["cost_yuan"] / 10000.0, 2),
                "综合风险": round(decision_plan["risk_score"], 2),
                "空间碰撞": round(decision_plan["spatial_risk_score"], 2),
                "作业环境": round(decision_plan["environment_risk_score"], 2),
                "设备均衡": round(decision_plan["equipment_balance_score"], 2),
            },
            {
                "模式": "专业模式",
                "算法": expert_result.metadata.get("algorithm", "NSGA3"),
                "目标数量": 5,
                "工期(工日)": round(expert_plan["makespan_min"] / 480.0, 2),
                "成本(万元)": round(expert_plan["cost_yuan"] / 10000.0, 2),
                "综合风险": round(expert_plan["risk_score"], 2),
                "空间碰撞": round(expert_plan["spatial_risk_score"], 2),
                "作业环境": round(expert_plan["environment_risk_score"], 2),
                "设备均衡": round(expert_plan["equipment_balance_score"], 2),
            },
        ]
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.info(
        "两种模式选中同一吊装调度方案。"
        if same_sequence
        else "两种模式选中的吊装调度方案不同，说明目标拆分改变了推荐结果。"
    )


def _build_parallel_coordinates(
    objectives: np.ndarray,
    plans: dict[str, dict],
) -> go.Figure:
    dimensions = [
        {
            "label": label,
            "values": objectives[:, index],
            "range": [float(objectives[:, index].min()), float(objectives[:, index].max())],
        }
        for index, label in enumerate(EXPERT_OBJECTIVE_NAMES)
    ]
    figure = go.Figure(
        go.Parcoords(
            line={
                "color": objectives[:, 0],
                "colorscale": [
                    [0.0, COLORS["teal"]],
                    [0.55, COLORS["amber"]],
                    [1.0, COLORS["red"]],
                ],
                "showscale": True,
                "colorbar": {"title": "工期"},
            },
            dimensions=dimensions,
        )
    )
    figure.update_layout(
        height=430,
        margin={"l": 58, "r": 44, "t": 42, "b": 42},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Microsoft YaHei, Noto Sans CJK SC, sans-serif"},
    )
    return figure


def _build_objective_projections(
    objectives: np.ndarray,
    plans: dict[str, dict],
) -> go.Figure:
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    figure = make_subplots(
        rows=2,
        cols=3,
        horizontal_spacing=0.08,
        vertical_spacing=0.18,
        subplot_titles=[f"{EXPERT_OBJECTIVE_NAMES[a]} - {EXPERT_OBJECTIVE_NAMES[b]}" for a, b in pairs],
    )
    for position, (horizontal, vertical) in enumerate(pairs, start=1):
        row = 1 if position <= 3 else 2
        column = position if position <= 3 else position - 3
        figure.add_trace(
            go.Scatter(
                x=objectives[:, horizontal],
                y=objectives[:, vertical],
                mode="markers",
                marker={
                    "size": 8,
                    "color": COLORS["teal"],
                    "opacity": 0.78,
                    "line": {"color": "#fffdf7", "width": 0.6},
                },
                hovertemplate=(
                    f"{EXPERT_OBJECTIVE_NAMES[horizontal]} %{{x:.2f}}<br>"
                    f"{EXPERT_OBJECTIVE_NAMES[vertical]} %{{y:.2f}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        figure.update_xaxes(
            title_text=EXPERT_OBJECTIVE_NAMES[horizontal],
            gridcolor=COLORS["grid"],
            row=row,
            col=column,
        )
        figure.update_yaxes(
            title_text=EXPERT_OBJECTIVE_NAMES[vertical],
            gridcolor=COLORS["grid"],
            row=row,
            col=column,
        )
    figure.update_layout(
        height=620,
        margin={"l": 48, "r": 24, "t": 52, "b": 46},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Microsoft YaHei, Noto Sans CJK SC, sans-serif"},
    )
    return figure


def _weight_sensitivity(
    objectives: np.ndarray,
    samples: int = 500,
    seed: int = 20250925,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    normalised = _normalize(objectives)
    counts = np.zeros(len(objectives), dtype=int)
    for _ in range(samples):
        weights = rng.dirichlet(np.ones(objectives.shape[1]))
        counts[int(np.argmin(normalised @ weights))] += 1
    selected_frequency = pd.DataFrame(
        {
            "Pareto序号": np.arange(1, len(objectives) + 1),
            "入选次数": counts,
            "入选比例": counts / samples,
            "工期": objectives[:, 0],
            "成本": objectives[:, 1],
            "空间碰撞": objectives[:, 2],
            "作业环境": objectives[:, 3],
            "设备均衡": objectives[:, 4],
        }
    )
    active = selected_frequency[selected_frequency["入选次数"] > 0]
    return (
        selected_frequency.sort_values("Pareto序号").reset_index(drop=True),
        active.sort_values("入选次数", ascending=False).reset_index(drop=True),
    )


def _normalize(objectives: np.ndarray) -> np.ndarray:
    minimum = objectives.min(axis=0)
    maximum = objectives.max(axis=0)
    span = np.where(maximum > minimum, maximum - minimum, 1.0)
    return (objectives - minimum) / span
