from __future__ import annotations

import html
from typing import Sequence

import pandas as pd


APP_CSS = """
<style>
:root {
  --ink: #141822;
  --ink-soft: #1e2534;
  --paper: #141822;
  --panel: #1e2534;
  --line: #334155;
  --muted: #94a3b8;
  --orange: #f97316;
  --teal: #2563eb;
  --amber: #eab308;
  --red: #ef4444;
}

html, body, [class*="css"] {
  font-family: "Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", sans-serif;
}

[data-testid="stAppViewContainer"] {
  background: var(--paper);
}

[data-testid="stSidebar"] {
  background: var(--ink);
  border-right: 1px solid #303b37;
}

[data-testid="stSidebar"] * {
  color: #f5f1e7;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
  background: #24302c;
  border-color: #40504a;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
  background: #24302c !important;
  border-color: #40504a !important;
  color: #f5f1e7 !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] input {
  color: #f5f1e7 !important;
  -webkit-text-fill-color: #f5f1e7 !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] svg {
  color: #b7c1bb !important;
  fill: #b7c1bb !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] label {
  color: #b7c1bb !important;
}

div[data-baseweb="popover"] [role="listbox"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] [data-testid="stSelectboxVirtualDropdown"] {
  background: #202b27 !important;
  color: #f5f1e7 !important;
  border: 1px solid #40504a !important;
  box-shadow: 0 12px 28px rgba(0, 0, 0, .28) !important;
}

div[data-baseweb="popover"] [role="option"] {
  background: #202b27 !important;
  color: #f5f1e7 !important;
}

div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
  background: #2e3d38 !important;
  color: #bce7e3 !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
  border-color: #40504a;
  background: #202b27;
}

[data-testid="stSidebar"] [data-testid="stPopover"] > div > button,
[data-testid="stSidebar"] [data-testid="stPopover"] button {
  background: #24302c !important;
  color: #f5f1e7 !important;
  border: 1px solid #40504a !important;
  min-height: 40px;
  box-shadow: none;
}

[data-testid="stSidebar"] [data-testid="stPopover"] > div > button:hover,
[data-testid="stSidebar"] [data-testid="stPopover"] button:hover {
  background: #2e3d38 !important;
  border-color: #5a6d65 !important;
}

[data-testid="stSidebar"] [data-testid="stPopover"] > div > button:focus-visible,
[data-testid="stSidebar"] [data-testid="stPopover"] button:focus-visible {
  outline: 2px solid var(--teal);
  outline-offset: 2px;
}

[data-testid="stSidebar"] [data-testid="stPopover"] svg {
  color: currentColor;
  fill: currentColor;
}

div[data-testid="stPopoverBody"],
div[data-baseweb="popover"] > div {
  background: #202b27 !important;
  color: #f5f1e7 !important;
  border: 1px solid #40504a !important;
  box-shadow: 0 12px 28px rgba(0, 0, 0, .28) !important;
}

div[data-testid="stPopoverBody"] *,
div[data-baseweb="popover"] > div * {
  color: #f5f1e7;
}

div[data-testid="stPopoverBody"] [data-baseweb="input"] > div {
  background: #24302c !important;
  border-color: #40504a !important;
}

div[data-testid="stPopoverBody"] input {
  color: #f5f1e7 !important;
  -webkit-text-fill-color: #f5f1e7 !important;
}

div[data-testid="stPopoverBody"] label {
  color: #b7c1bb !important;
}

div[data-testid="stPopoverBody"] .stButton button {
  background: #24302c !important;
  color: #bce7e3 !important;
  border-color: var(--teal) !important;
}

div[data-testid="stPopoverBody"] .stButton button:hover {
  background: #2e3d38 !important;
  color: #e4f4f2 !important;
  border-color: #22a39e !important;
}

[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
  background-color: var(--orange);
  border-color: var(--orange);
}

.block-container {
  max-width: 1560px;
  padding-top: 1.1rem;
  padding-bottom: 3rem;
}

.app-hero {
  position: relative;
  overflow: hidden;
  background: var(--ink);
  border: 1px solid #303b37;
  border-left: 7px solid var(--orange);
  padding: 24px 28px 22px;
  margin-bottom: 14px;
}

.app-hero::after {
  content: "";
  position: absolute;
  right: 26px;
  top: 24px;
  width: 92px;
  height: 7px;
  background: var(--amber);
  box-shadow: 0 14px 0 var(--teal), 0 28px 0 #59645e;
}

.app-kicker {
  color: #b7c1bb;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .11em;
  text-transform: uppercase;
  margin-bottom: 8px;
}

.app-title {
  color: #fffdf7 !important;
  -webkit-text-fill-color: #fffdf7 !important;
  font-size: clamp(25px, 2.25vw, 39px);
  line-height: 1.12;
  letter-spacing: 0;
  font-weight: 800;
  max-width: 1100px;
  margin: 0;
}

.app-subtitle {
  color: #c8d0cb;
  font-size: 13px;
  line-height: 1.7;
  margin-top: 9px;
  max-width: 980px;
}

.status-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 16px;
}

.status-chip {
  color: #f7f3e9;
  background: #27332f;
  border: 1px solid #46534d;
  padding: 5px 9px;
  font-size: 11px;
  line-height: 1;
}

.status-chip.ready {
  border-color: var(--teal);
  color: #bce7e3;
}

.status-chip.warn {
  border-color: var(--amber);
  color: #f3d590;
}

.section-label {
  color: var(--muted);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .09em;
  text-transform: uppercase;
  margin: 14px 0 -3px 0;
}

.sidebar-brand {
  padding: 8px 0 16px;
  border-bottom: 1px solid #3a4641;
  margin-bottom: 18px;
}

.sidebar-brand strong {
  display: block;
  font-size: 21px;
  color: #fffdf7;
  letter-spacing: .02em;
}

.sidebar-brand span {
  display: block;
  color: #99a49e;
  font-size: 11px;
  margin-top: 5px;
  letter-spacing: .08em;
}

.sidebar-readout {
  background: #24302c;
  border-left: 3px solid var(--teal);
  padding: 10px 11px;
  margin: 9px 0;
}

.sidebar-readout span {
  color: #99a49e;
  display: block;
  font-size: 10px;
  margin-bottom: 3px;
}

.sidebar-readout strong {
  color: #fffdf7;
  font-size: 14px;
}

[data-testid="stMetric"] {
  background: var(--panel);
  border-top: 3px solid var(--teal);
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  border-left: 1px solid var(--line);
  padding: 13px 15px 12px;
  min-height: 106px;
}

[data-testid="stMetric"] label {
  color: var(--muted);
  font-size: 11px;
}

[data-testid="stMetricValue"] {
  color: var(--ink);
  font-weight: 800;
  letter-spacing: 0;
}

[data-testid="stMetricDelta"] {
  font-size: 11px;
}

[data-testid="stAlert"] {
  overflow: visible !important;
}

[data-testid="stAlert"] p {
  overflow: visible !important;
  white-space: normal !important;
  line-height: 1.55 !important;
}

[data-testid="stSpinner"],
.stSpinner {
  background: transparent !important;
  color: var(--ink) !important;
  padding: 8px 0 !important;
  overflow: visible !important;
}

[data-testid="stSpinner"] p,
.stSpinner p {
  color: var(--ink) !important;
  overflow: visible !important;
  white-space: normal !important;
  line-height: 1.5 !important;
}

[data-testid="stProgress"],
.stProgress,
[data-testid="stProgress"] > div,
.stProgress > div,
[data-testid="stProgress"] [role="progressbar"],
.stProgress [role="progressbar"] {
  width: 100% !important;
}

[data-testid="stProgress"],
.stProgress {
  margin: 8px 0 14px;
  overflow: visible !important;
}

[data-testid="stProgress"] > div:first-child,
.stProgress > div:first-child {
  min-height: 22px;
  padding-bottom: 6px;
  overflow: visible !important;
  white-space: normal !important;
  line-height: 1.45 !important;
}

.optimization-overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  display: grid;
  place-items: center;
}

.optimization-overlay-backdrop {
  position: absolute;
  inset: 0;
  background: rgba(24, 33, 31, 0.66);
  backdrop-filter: blur(3px);
}

.optimization-overlay-panel {
  position: relative;
  z-index: 1;
  width: min(520px, calc(100vw - 40px));
  padding: 24px 26px 22px;
  background: #202b27;
  border: 1px solid #40504a;
  border-top: 4px solid var(--orange);
  box-shadow: 0 22px 60px rgba(0, 0, 0, .34);
}

.optimization-overlay-kicker {
  color: var(--teal);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .13em;
  margin-bottom: 7px;
}

.optimization-overlay-title {
  color: #fffdf7;
  font-size: 22px;
  font-weight: 800;
  margin-bottom: 7px;
}

.optimization-overlay-text {
  min-height: 24px;
  color: #c8d0cb;
  font-size: 13px;
  line-height: 1.6;
  margin-bottom: 15px;
}

.optimization-overlay-track {
  height: 10px;
  overflow: hidden;
  background: #111916;
  border: 1px solid #40504a;
}

.optimization-overlay-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--teal), #22a39e);
  transition: width .18s ease;
}

.optimization-overlay-footer {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  margin-top: 10px;
  color: #99a49e;
  font-size: 11px;
}

.optimization-overlay-footer strong {
  color: #bce7e3;
  font-size: 12px;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 2px;
  border-bottom: 1px solid var(--line);
}

[data-testid="stTabs"] [data-baseweb="tab"] {
  height: 42px;
  padding: 0 17px;
  background: transparent;
  color: var(--muted);
  border-radius: 0;
  font-weight: 700;
}

[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--ink);
  border-bottom: 3px solid var(--orange);
}

[data-testid="stPlotlyChart"] {
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 5px 7px 2px;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
}

.stButton button,
.stDownloadButton button,
[data-testid="stFormSubmitButton"] button {
  min-height: 42px;
  border-radius: 2px;
  border: 1px solid #33413c;
  font-weight: 750;
}

[data-testid="stFormSubmitButton"] button {
  width: 100%;
  background: var(--orange);
  color: #fffdf7;
  border-color: var(--orange);
}

[data-testid="stFormSubmitButton"] button:hover {
  background: #d85a26;
  border-color: #d85a26;
  color: #fffdf7;
}

.stDownloadButton button {
  background: var(--ink);
  color: #fffdf7;
}

.decision-band {
  background: var(--panel);
  border: 1px solid var(--line);
  border-top: 4px solid var(--orange);
  padding: 17px 18px 15px;
  margin-bottom: 10px;
}

.decision-band h3 {
  color: var(--ink);
  margin: 0 0 7px 0;
  font-size: 19px;
}

.decision-band p {
  color: #4f5752;
  margin: 0;
  font-size: 13px;
  line-height: 1.7;
}

.plan-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin: 8px 0 15px;
}

.plan-tile {
  background: var(--panel);
  border: 1px solid var(--line);
  border-top: 3px solid #a7aca7;
  padding: 12px;
  min-height: 126px;
}

.plan-tile.recommended {
  border-top-color: var(--orange);
  background: #fff8ef;
}

.plan-tile strong {
  display: block;
  color: var(--ink);
  font-size: 14px;
  margin-bottom: 9px;
}

.plan-tile span {
  display: block;
  color: var(--muted);
  font-size: 11px;
  margin-top: 5px;
}

.plan-tile b {
  color: var(--ink);
  font-size: 15px;
}

.risk-note {
  background: #fff8eb;
  border-left: 4px solid var(--amber);
  padding: 11px 13px;
  color: #5e4c20;
  font-size: 12px;
  line-height: 1.65;
}

.subsection-heading {
  color: var(--ink);
  font-size: 18px;
  font-weight: 800;
  margin: 12px 0 2px;
}

.subsection-note {
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 12px;
}

.project-action-spacer {
  height: 28px;
}

.template-action-spacer {
  height: 30px;
}

/* Dark engineering theme overrides */
:root {
  --ink: #141822;
  --ink-soft: #1e2534;
  --paper: #141822;
  --panel: #1e2534;
  --line: #334155;
  --muted: #94a3b8;
  --orange: #f97316;
  --teal: #2563eb;
  --amber: #eab308;
  --red: #ef4444;
}

[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stAppViewContainer"] > .main {
  background: #141822 !important;
  color: #cbd5e1 !important;
}

[data-testid="stSidebar"] {
  background: #0f131c !important;
}

[data-testid="stSidebar"] * {
  color: #cbd5e1;
}

[data-testid="stSidebar"] .sidebar-readout {
  background: #1e2534 !important;
  border-left-color: #2563eb !important;
}

[data-testid="stSidebar"] .sidebar-readout span {
  color: #94a3b8 !important;
}

.app-hero {
  background: #0b0f16 !important;
  border-color: #334155 !important;
  border-left-color: #2563eb !important;
}

.app-kicker {
  color: #94a3b8 !important;
}

.app-title {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

.app-subtitle {
  color: #cbd5e1 !important;
}

.status-chip {
  background: #1e2534 !important;
  border-color: #334155 !important;
  color: #cbd5e1 !important;
}

.status-chip.ready {
  border-color: #2563eb !important;
  color: #93c5fd !important;
}

.status-chip.warn {
  border-color: #f97316 !important;
  color: #fdba74 !important;
}

.section-label,
.subsection-note {
  color: #94a3b8 !important;
}

.subsection-heading {
  color: #f8fafc !important;
  position: relative;
  padding-left: 12px;
}

.subsection-heading::before {
  content: "";
  position: absolute;
  left: 0;
  top: 4px;
  bottom: 4px;
  width: 3px;
  background: #2563eb;
  border-radius: 2px;
}

[data-testid="stMetric"] {
  background: #1e2534 !important;
  border: 1px solid #334155 !important;
  border-top: 3px solid #2563eb !important;
  border-radius: 6px !important;
  padding: 18px 18px 16px !important;
  min-height: 118px;
  transition: background .16s ease, border-color .16s ease;
}

[data-testid="stMetric"]:hover {
  background: #242c3d !important;
  border-color: #475569 !important;
}

[data-testid="stMetric"] label,
[data-testid="stMetricLabel"] {
  color: #94a3b8 !important;
  font-size: 11px !important;
}

[data-testid="stMetricValue"] {
  color: #ffffff !important;
  font-size: 30px !important;
  font-weight: 800 !important;
}

[data-testid="stMetricDelta"] {
  color: #22c55e !important;
  font-size: 11px !important;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
  border-bottom: 1px solid #334155 !important;
  gap: 4px !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
  color: #94a3b8 !important;
  border-bottom: none !important;
  border-radius: 4px 4px 0 0;
}

[data-testid="stTabs"] [aria-selected="true"] {
  color: #60a5fa !important;
  border-bottom: none !important;
  background: rgba(37, 99, 235, .13) !important;
}

[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
[data-testid="stFileUploader"],
[data-testid="stForm"],
[data-testid="stCustomComponentV1"],
[data-testid="stVerticalBlockBorderWrapper"] {
  background: #1e2534 !important;
  border: 1px solid #334155 !important;
  border-radius: 6px !important;
  padding: 10px !important;
}

[data-testid="stCustomComponentV1"] iframe {
  background: #141822 !important;
}

.stButton button,
.stDownloadButton button {
  min-height: 42px;
  border-radius: 6px !important;
  border: 1px solid #334155 !important;
  background: #1e2534 !important;
  color: #cbd5e1 !important;
  font-weight: 700;
  transition: background .16s ease, border-color .16s ease;
}

.stButton button:hover,
.stDownloadButton button:hover {
  background: #283247 !important;
  border-color: #475569 !important;
  color: #ffffff !important;
}

.stButton button[kind="primary"],
[data-testid="stFormSubmitButton"] button {
  background: #2563eb !important;
  border-color: #2563eb !important;
  color: #ffffff !important;
}

.stButton button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] button:hover {
  background: #1d4ed8 !important;
  border-color: #1d4ed8 !important;
}

.decision-band,
.plan-tile,
.risk-note {
  background: #1e2534 !important;
  border-color: #334155 !important;
}

.decision-band h3,
.plan-tile strong,
.plan-tile b {
  color: #f8fafc !important;
}

.decision-band p,
.plan-tile span {
  color: #cbd5e1 !important;
}

.plan-tile.recommended {
  background: #202a3d !important;
  border-top-color: #2563eb !important;
}

.plan-tile.plan-a {
  border-top-color: #2563eb !important;
}

.plan-tile.plan-b {
  border-top-color: #f97316 !important;
}

.plan-tile.plan-c {
  border-top-color: #22c55e !important;
}

.plan-tile.plan-d {
  border-top-color: #a855f7 !important;
}

[data-testid="stTabs"] [role="tabpanel"] {
  padding-top: 14px !important;
}

[data-testid="stDataFrame"],
[data-testid="stPlotlyChart"] {
  overflow: hidden !important;
}

[data-testid="stDataFrame"] {
  padding: 0 !important;
}

[data-testid="stDataFrame"] > div {
  border-radius: 5px;
}

[data-testid="stFileUploader"] {
  padding: 16px !important;
}

[data-testid="stExpander"] {
  background: #1e2534 !important;
  border: 1px solid #334155 !important;
  border-radius: 6px !important;
}

[data-testid="stExpander"] summary {
  color: #cbd5e1 !important;
}

@media (max-width: 1100px) {
  .plan-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .block-container {
    padding-left: .7rem;
    padding-right: .7rem;
  }
  .app-hero {
    padding: 19px 18px 18px;
  }
  .app-hero::after {
    display: none;
  }
  .app-title {
    font-size: 25px;
  }
  .plan-grid {
    grid-template-columns: 1fr;
  }
  [data-testid="stMetric"] {
    min-height: 92px;
  }
}
</style>
"""


def hero_html(
    project_name: str,
    component_count: int,
    cranes: Sequence[dict],
    algorithm_label: str,
) -> str:
    return f"""
<div class="app-hero">
  <div class="app-kicker">Prefabricated Metro Station · Hoisting Decision System</div>
  <h1 class="app-title">装配式地铁车站吊装调度与可视化决策平台</h1>
  <div class="app-subtitle">
    面向预制构件吊装全流程，联动构件约束、设备能力、工期成本与安全风险，
    生成可解释、可比较、可导出的吊装决策方案。
  </div>
  <div class="status-row">
    <span class="status-chip ready">当前项目 {html.escape(project_name)}</span>
    <span class="status-chip">构件 {component_count} 件</span>
    <span class="status-chip">吊装设备 {len(cranes)} 台</span>
    <span class="status-chip warn">{html.escape(algorithm_label)}</span>
  </div>
</div>
"""


def sidebar_brand_html() -> str:
    return """
<div class="sidebar-brand">
  <strong>HOIST-OPT</strong>
  <span>吊装调度决策控制台 · V1.0</span>
</div>
"""


def project_center_visual_html() -> str:
    return """
<div class="project-visual">
  <div class="project-visual-head">
    <span>Prefabricated Metro Station</span>
    <strong>预制装配式车站 · 龙门吊吊装示意</strong>
  </div>
  <svg viewBox="0 0 1100 440" role="img" aria-label="装配式地铁车站龙门吊吊装示意">
    <defs>
      <pattern id="pc-grid" width="38" height="38" patternUnits="userSpaceOnUse">
        <path d="M 38 0 L 0 0 0 38" fill="none" stroke="#334155" stroke-width="1" opacity=".28"/>
      </pattern>
      <linearGradient id="pc-sky" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#1e2534"/>
        <stop offset="1" stop-color="#141822"/>
      </linearGradient>
      <linearGradient id="pc-slab" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0" stop-color="#eab308"/>
        <stop offset="1" stop-color="#f97316"/>
      </linearGradient>
    </defs>

    <rect width="1100" height="440" fill="url(#pc-sky)"/>
    <rect width="1100" height="440" fill="url(#pc-grid)"/>

    <path d="M0 352 H1100" stroke="#334155" stroke-width="2"/>
    <path d="M0 367 H1100" stroke="#1e2534" stroke-width="1"/>

    <g opacity=".92">
      <path d="M150 328 Q550 374 950 328" fill="none" stroke="#f97316" stroke-width="10" stroke-linecap="round"/>
      <path d="M174 316 Q550 356 926 316" fill="none" stroke="#eab308" stroke-width="3" opacity=".72"/>
      <rect x="196" y="286" width="184" height="18" rx="4" fill="#2563eb" opacity=".9"/>
      <rect x="720" y="286" width="184" height="18" rx="4" fill="#2563eb" opacity=".9"/>
    </g>

    <g>
      <rect x="272" y="194" width="556" height="19" rx="5" fill="url(#pc-slab)"/>
      <rect x="300" y="213" width="500" height="5" rx="2" fill="#a855f7" opacity=".65"/>

      <g stroke="#cbd5e1" stroke-width="8" stroke-linecap="round">
        <path d="M386 213 V315"/>
        <path d="M550 213 V315"/>
        <path d="M714 213 V315"/>
      </g>
      <g fill="#93c5fd">
        <rect x="374" y="207" width="24" height="9" rx="3"/>
        <rect x="538" y="207" width="24" height="9" rx="3"/>
        <rect x="702" y="207" width="24" height="9" rx="3"/>
        <rect x="374" y="313" width="24" height="9" rx="3"/>
        <rect x="538" y="313" width="24" height="9" rx="3"/>
        <rect x="702" y="313" width="24" height="9" rx="3"/>
      </g>

      <path d="M244 190 Q550 70 856 190" fill="none" stroke="#2563eb" stroke-width="11" stroke-linecap="round"/>
      <path d="M260 190 Q550 87 840 190" fill="none" stroke="#60a5fa" stroke-width="2" opacity=".62"/>
      <g stroke="#475569" stroke-width="2" opacity=".85">
        <path d="M310 172 L328 118"/>
        <path d="M410 142 L422 98"/>
        <path d="M550 120 V74"/>
        <path d="M690 142 L678 98"/>
        <path d="M790 172 L772 118"/>
      </g>
      <path d="M244 190 V318" stroke="#334155" stroke-width="5" stroke-dasharray="8 9"/>
      <path d="M856 190 V318" stroke="#334155" stroke-width="5" stroke-dasharray="8 9"/>
    </g>

    <g>
      <path d="M486 315 H512 V286 H534 V258 H556 V230 H578" fill="none" stroke="#cbd5e1" stroke-width="5" stroke-linejoin="round"/>
      <path d="M486 315 H512" stroke="#2563eb" stroke-width="9" stroke-linecap="round"/>
    </g>

    <g>
      <path d="M120 52 H978" stroke="#f97316" stroke-width="17" stroke-linecap="round"/>
      <path d="M122 75 H976" stroke="#a855f7" stroke-width="5" stroke-linecap="round" opacity=".75"/>
      <path d="M178 60 V348" stroke="#cbd5e1" stroke-width="13" stroke-linecap="round"/>
      <path d="M920 60 V348" stroke="#cbd5e1" stroke-width="13" stroke-linecap="round"/>
      <path d="M148 92 H208 M892 92 H952" stroke="#eab308" stroke-width="8" stroke-linecap="round"/>
      <path d="M151 330 H206 M894 330 H949" stroke="#334155" stroke-width="8" stroke-linecap="round"/>
      <rect x="520" y="62" width="92" height="23" rx="5" fill="#eab308"/>
      <rect x="534" y="67" width="28" height="13" rx="3" fill="#141822" opacity=".72"/>
      <path d="M566 85 V150" stroke="#cbd5e1" stroke-width="3" stroke-dasharray="7 7"/>
      <path d="M548 150 H584" stroke="#cbd5e1" stroke-width="4" stroke-linecap="round"/>
      <path d="M554 150 Q566 174 578 150" fill="none" stroke="#f97316" stroke-width="4"/>
      <rect x="514" y="177" width="104" height="29" rx="5" fill="#2563eb" stroke="#93c5fd" stroke-width="3"/>
      <path d="M530 191 H602 M530 198 H586" stroke="#bfdbfe" stroke-width="3" stroke-linecap="round" opacity=".8"/>
    </g>

    <g opacity=".9">
      <rect x="42" y="278" width="74" height="22" rx="3" fill="#2563eb"/>
      <rect x="52" y="258" width="54" height="20" rx="3" fill="#334155"/>
      <rect x="64" y="239" width="34" height="19" rx="3" fill="#eab308"/>
      <rect x="982" y="276" width="76" height="24" rx="3" fill="#2563eb"/>
      <rect x="994" y="254" width="52" height="22" rx="3" fill="#334155"/>
      <rect x="1006" y="234" width="30" height="20" rx="3" fill="#f97316"/>
    </g>

    <g fill="#94a3b8" font-family="Arial, sans-serif" font-size="12" font-weight="700" letter-spacing="2">
      <text x="42" y="404">PRECAST COMPONENTS</text>
      <text x="812" y="404">GANTRY / HOISTING</text>
    </g>
    <path d="M42 390 H204" stroke="#2563eb" stroke-width="3"/>
    <path d="M812 390 H974" stroke="#f97316" stroke-width="3"/>
  </svg>
</div>
"""


def sidebar_readout(label: str, value: str) -> str:
    return f"""
<div class="sidebar-readout">
  <span>{html.escape(label)}</span>
  <strong>{html.escape(value)}</strong>
</div>
"""


def plan_tiles_html(plans: dict[str, dict], recommended_name: str = "综合推荐") -> str:
    tiles: list[str] = []
    for plan_name, plan in plans.items():
        prefix = str(plan_name).strip()[:1].upper()
        color_class = {
            "A": "plan-a",
            "B": "plan-b",
            "C": "plan-c",
            "D": "plan-d",
        }.get(prefix, "")
        classes = "plan-tile"
        if color_class:
            classes += f" {color_class}"
        if plan_name == recommended_name:
            classes += " recommended"
        tiles.append(
            f"""
<div class="{classes}">
  <strong>{html.escape(plan_name)}</strong>
  <b>{format_duration(plan['makespan_min'])}</b>
  <span>成本 {plan['cost_yuan']:,.0f} 元</span>
  <span>风险指数 {plan['risk_score']:.1f}</span>
  <span>碰撞敏感点 {plan['adjacent_crossings']} 处</span>
</div>
"""
        )
    return f'<div class="plan-grid">{"".join(tiles)}</div>'


def schedule_frame(schedule: Sequence[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(schedule)
    if frame.empty:
        return frame
    frame = frame[
        [
            "order",
            "component_id",
            "component_name",
            "component_type",
            "crane",
            "start",
            "end",
            "duration",
            "weight_t",
            "long_travel_m",
            "trolley_travel_m",
            "hoist_travel_m",
            "trolley_offset_m",
            "capacity_utilization",
            "risk",
        ]
    ].copy()
    frame.columns = [
        "吊装顺序",
        "构件编号",
        "构件名称",
        "构件类型",
        "吊装设备",
        "开始(min)",
        "完成(min)",
        "耗时(min)",
        "重量(t)",
        "大车纵移(m)",
        "小车横移(m)",
        "吊钩行程(m)",
        "小车偏移(m)",
        "能力利用率",
        "风险指数",
    ]
    for column in [
        "开始(min)",
        "完成(min)",
        "耗时(min)",
        "重量(t)",
        "大车纵移(m)",
        "小车横移(m)",
        "吊钩行程(m)",
        "小车偏移(m)",
        "能力利用率",
        "风险指数",
    ]:
        frame[column] = frame[column].round(2)
    return frame


def highlight_high_risk_rows(
    frame: pd.DataFrame,
    threshold: float = 35.0,
) -> pd.io.formats.style.Styler:
    """Apply a restrained warning treatment to high-risk schedule rows."""
    if frame.empty:
        return frame.style

    def _style_row(row: pd.Series) -> list[str]:
        risk_value = pd.to_numeric(row.iloc[-1], errors="coerce")
        risk = 0.0 if pd.isna(risk_value) else float(risk_value)
        style = (
            "background-color: rgba(249, 115, 22, 0.16); "
            "color: #fdba74; font-weight: 600;"
            if risk >= threshold
            else ""
        )
        return [style for _ in row]

    return frame.style.apply(_style_row, axis=1)


def decision_summary(plan: dict, component_count: int) -> str:
    makespan_days = plan["makespan_min"] / (8.0 * 60.0)
    crane_count = len(plan["crane_workload"])
    return (
        f"建议采用“{plan['plan_name']}”。该方案计划使用 {crane_count} 台吊装设备，"
        f"完成 {component_count} 件预制构件吊装，设备作业窗口约 {makespan_days:.1f} 个工日，"
        f"估算吊装成本 {plan['cost_yuan']:,.0f} 元，安全风险指数 {plan['risk_score']:.1f}。"
    )


def format_duration(minutes: float) -> str:
    hours = minutes / 60.0
    if hours >= 24.0:
        return f"{hours / 8.0:.1f} 工日"
    return f"{hours:.1f} 小时"
