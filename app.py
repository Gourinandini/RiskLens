"""
Streamlit Application Entry Point — RiskLens Credit Risk Intelligence Platform v2.0

Premium multi-tab interactive dashboard with:
  - Tab 1: 📊 EDA Dashboard — Data quality, demographics, risk factor visualizations
  - Tab 2: 🔍 ML Risk Predictor — SHAP-powered credit scoring
  - Tab 3: 🧠 Explainability — Global SHAP, LIME, feature impact analysis
  - Tab 4: 📋 Decision Rules — Business-readable if/then risk rules
  - Tab 5: 💬 RiskLens Copilot — NL-to-SQL chatbot powered by Groq LLM
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.utils.config import DATA_PATH, MODEL_PATH, GROQ_API_KEY
from src.utils.logger import get_logger
from src.data.loader import DataLoader
from src.ml.predict import RiskPredictor
from src.talk_to_data.nl_to_sql import TalkToDataAgent

logger = get_logger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="RiskLens — Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREMIUM CSS DESIGN SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ─── Reset & Base ─────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; }

    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stMain"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #060912;
        color: #CBD5E1;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #F1F5F9;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ─── Hide Streamlit Cruft ──────────────────────────────────── */
    #MainMenu, footer, [data-testid="stDeployButton"], [data-testid="stHeader"] { display: none !important; }
    [data-testid="stHeader"] { display: none !important; }

    /* ─── Main Content Area ─────────────────────────────────────── */
    .main .block-container {
        padding: 2.5rem 2rem 3rem !important;
        max-width: 1600px !important;
    }

    /* ─── Sidebar ───────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080c18 0%, #060912 100%);
        border-right: 1px solid rgba(0, 242, 254, 0.08);
    }
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00F2FE, #4FACFE, #7F00FF);
    }
    [data-testid="stSidebarContent"] { padding: 0 !important; }

    /* ─── Hero Brand Card ───────────────────────────────────────── */
    .brand-hero {
        padding: 28px 20px 22px;
        text-align: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        position: relative;
        overflow: hidden;
    }
    .brand-hero::after {
        content: '';
        position: absolute;
        bottom: 0; left: 50%;
        transform: translateX(-50%);
        width: 60%; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 242, 254, 0.4), transparent);
    }
    .brand-name {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #7F00FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin-bottom: 4px;
    }
    .brand-tagline {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        color: rgba(0, 242, 254, 0.7);
        text-transform: uppercase;
    }
    .brand-version {
        display: inline-block;
        font-size: 0.6rem;
        padding: 2px 8px;
        background: rgba(0, 242, 254, 0.08);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 20px;
        color: rgba(0, 242, 254, 0.6);
        margin-top: 10px;
        letter-spacing: 0.05em;
    }

    /* ─── Sidebar Metric Cards ──────────────────────────────────── */
    .sidebar-section { padding: 18px 16px 8px; }
    .sidebar-section-title {
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: rgba(148, 163, 184, 0.5);
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }
    .metric-pill {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        transition: all 0.2s;
    }
    .metric-pill:hover {
        background: rgba(0, 242, 254, 0.04);
        border-color: rgba(0, 242, 254, 0.15);
    }
    .metric-pill-icon {
        width: 32px; height: 32px;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }
    .metric-pill-body { flex: 1; }
    .metric-pill-label {
        font-size: 0.6rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748B;
    }
    .metric-pill-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #E2E8F0;
        line-height: 1.1;
    }
    .metric-pill-sub {
        font-size: 0.6rem;
        color: #475569;
    }

    /* ─── Tab Navigation ─────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        background: transparent !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
        padding: 0 0 1px !important;
        margin-bottom: 24px !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #475569 !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 10px 18px !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.01em !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #94A3B8 !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 6px 6px 0 0 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #00F2FE !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #00F2FE !important;
        background: rgba(0, 242, 254, 0.04) !important;
        border-radius: 6px 6px 0 0 !important;
    }

    /* ─── Page Headers ────────────────────────────────────────────── */
    .page-header {
        margin-bottom: 28px;
        padding-bottom: 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        position: relative;
    }
    .page-header::after {
        content: '';
        position: absolute;
        bottom: -1px; left: 0;
        width: 80px; height: 2px;
        background: linear-gradient(90deg, #00F2FE, transparent);
    }
    .page-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #F1F5F9;
        letter-spacing: -0.03em;
        margin: 0 0 6px;
        line-height: 1.1;
    }
    .page-subtitle {
        font-size: 0.875rem;
        color: #475569;
        margin: 0;
        line-height: 1.5;
    }

    /* ─── KPI Metric Cards ───────────────────────────────────────── */
    .kpi-grid { display: grid; gap: 14px; margin-bottom: 28px; }
    .kpi-grid-4 { grid-template-columns: repeat(4, 1fr); }
    .kpi-grid-3 { grid-template-columns: repeat(3, 1fr); }
    .kpi-grid-2 { grid-template-columns: repeat(2, 1fr); }

    .kpi-card {
        background: rgba(12, 16, 28, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 20px 22px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: default;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--kpi-accent, linear-gradient(90deg, #00F2FE, #4FACFE));
        opacity: 0.8;
    }
    .kpi-card::after {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 140px; height: 140px;
        border-radius: 50%;
        background: var(--kpi-glow, rgba(0, 242, 254, 0.05));
        transition: all 0.4s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 242, 254, 0.2);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(0, 242, 254, 0.1);
    }
    .kpi-card:hover::after { transform: scale(1.3); }

    .kpi-icon {
        font-size: 1.4rem;
        margin-bottom: 12px;
        display: block;
    }
    .kpi-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1;
        margin-bottom: 6px;
    }
    .kpi-delta {
        font-size: 0.72rem;
        color: #475569;
    }
    .kpi-delta .up { color: #00E676; }
    .kpi-delta .down { color: #FF1744; }

    /* ─── Section Cards / Containers ────────────────────────────── */
    .card {
        background: rgba(10, 14, 24, 0.5);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.055);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: border-color 0.25s;
    }
    .card:hover { border-color: rgba(255, 255, 255, 0.1); }
    .card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #CBD5E1;
        margin: 0 0 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: rgba(255,255,255,0.05);
    }

    /* ─── Risk Decision Cards ────────────────────────────────────── */
    .risk-card {
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.08);
        position: relative;
        overflow: hidden;
    }
    .risk-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
    }
    .risk-card-low {
        background: radial-gradient(ellipse at top, rgba(0, 230, 118, 0.08) 0%, rgba(0,0,0,0) 60%);
        border-color: rgba(0, 230, 118, 0.2);
    }
    .risk-card-low::before { background: linear-gradient(90deg, #00E676, #69F0AE); }
    .risk-card-medium {
        background: radial-gradient(ellipse at top, rgba(255, 179, 0, 0.08) 0%, rgba(0,0,0,0) 60%);
        border-color: rgba(255, 179, 0, 0.2);
    }
    .risk-card-medium::before { background: linear-gradient(90deg, #FFB300, #FFD54F); }
    .risk-card-high {
        background: radial-gradient(ellipse at top, rgba(255, 23, 68, 0.08) 0%, rgba(0,0,0,0) 60%);
        border-color: rgba(255, 23, 68, 0.2);
    }
    .risk-card-high::before { background: linear-gradient(90deg, #FF1744, #FF6D00); }

    .risk-card-badge-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 8px;
    }
    .risk-card-prob {
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        line-height: 1;
        margin-bottom: 4px;
    }
    .risk-card-low .risk-card-prob { color: #00E676; }
    .risk-card-medium .risk-card-prob { color: #FFB300; }
    .risk-card-high .risk-card-prob { color: #FF1744; }

    .risk-card-band {
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .risk-card-low .risk-card-band { color: rgba(0, 230, 118, 0.8); }
    .risk-card-medium .risk-card-band { color: rgba(255, 179, 0, 0.8); }
    .risk-card-high .risk-card-band { color: rgba(255, 23, 68, 0.8); }

    .risk-card-divider {
        height: 1px;
        background: rgba(255,255,255,0.06);
        margin: 14px 0;
    }
    .risk-card-decision {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 8px;
    }
    .risk-card-low .risk-card-decision { color: #00E676; }
    .risk-card-medium .risk-card-decision { color: #FFB300; }
    .risk-card-high .risk-card-decision { color: #FF1744; }

    .risk-card-desc { font-size: 0.8rem; color: #64748B; line-height: 1.5; }

    /* ─── Chat Interface ─────────────────────────────────────────── */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-bottom: 16px;
        min-height: 200px;
    }
    .msg-user {
        align-self: flex-end;
        max-width: 78%;
        background: rgba(79, 172, 254, 0.08);
        border: 1px solid rgba(79, 172, 254, 0.2);
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #E2E8F0;
    }
    .msg-bot {
        align-self: flex-start;
        max-width: 90%;
        background: rgba(14, 20, 36, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-left: 3px solid #00F2FE;
        border-radius: 4px 16px 16px 16px;
        padding: 14px 18px;
    }
    .msg-bot-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .msg-bot-avatar {
        width: 22px; height: 22px;
        border-radius: 50%;
        background: linear-gradient(135deg, #00F2FE, #4FACFE);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.65rem;
        color: #060912;
        font-weight: 800;
    }
    .msg-bot-name {
        font-size: 0.7rem;
        font-weight: 700;
        color: #00F2FE;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .msg-bot-text { font-size: 0.88rem; color: #CBD5E1; line-height: 1.6; }

    /* ─── Decision Rule Cards ────────────────────────────────────── */
    .rule-card {
        background: rgba(10, 14, 24, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        transition: all 0.25s;
        position: relative;
    }
    .rule-card:hover {
        background: rgba(10, 14, 24, 0.7);
        border-color: rgba(255, 255, 255, 0.12);
        transform: translateX(4px);
    }
    .rule-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        border-radius: 3px 0 0 3px;
        background: var(--rule-color, #00E676);
    }
    .rule-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
        flex-wrap: wrap;
    }
    .rule-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 500;
        color: #475569;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 4px;
        padding: 2px 8px;
    }
    .rule-band-badge {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 20px;
    }
    .badge-low { background: rgba(0,230,118,0.1); color: #00E676; border: 1px solid rgba(0,230,118,0.25); }
    .badge-medium { background: rgba(255,179,0,0.1); color: #FFB300; border: 1px solid rgba(255,179,0,0.25); }
    .badge-high { background: rgba(255,23,68,0.1); color: #FF1744; border: 1px solid rgba(255,23,68,0.25); }

    .rule-conditions {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #94A3B8;
        background: rgba(0,0,0,0.2);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 10px;
        line-height: 1.7;
    }
    .rule-conditions .condition-and {
        color: #4FACFE;
        font-weight: 700;
    }
    .rule-stats {
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
    }
    .rule-stat {
        display: flex;
        flex-direction: column;
    }
    .rule-stat-label {
        font-size: 0.6rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #475569;
    }
    .rule-stat-value {
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #E2E8F0;
    }
    .rule-desc {
        font-size: 0.78rem;
        color: #64748B;
        line-height: 1.5;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid rgba(255,255,255,0.04);
    }

    /* ─── Info Notes / Callouts ──────────────────────────────────── */
    .note-info {
        background: rgba(14, 20, 40, 0.5);
        border-left: 3px solid #00F2FE;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 14px;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #94A3B8;
    }
    .note-info strong { color: #00F2FE; font-family: 'Outfit', sans-serif; }

    /* ─── Empty State ────────────────────────────────────────────── */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 30px;
        border: 2px dashed rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        background: rgba(6, 9, 18, 0.3);
        text-align: center;
        margin-top: 10px;
    }
    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 16px;
        animation: float 3s ease-in-out infinite;
    }
    .empty-state-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .empty-state-body {
        font-size: 0.82rem;
        color: #475569;
        max-width: 320px;
        line-height: 1.6;
    }

    /* ─── Buttons ────────────────────────────────────────────────── */
    div.stButton > button {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 60%, #0072FF 100%) !important;
        color: #060912 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.03em !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.25), inset 0 1px 0 rgba(255,255,255,0.2) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(0, 242, 254, 0.4), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    }
    div.stButton > button:active { transform: translateY(0px) !important; }

    /* ─── Form Inputs ────────────────────────────────────────────── */
    [data-testid="stForm"] {
        background: rgba(10, 14, 24, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        padding: 22px !important;
    }
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stSlider > div { color: #E2E8F0 !important; }

    /* ─── Sample Query Buttons ───────────────────────────────────── */
    .sample-btn button {
        background: rgba(10, 14, 24, 0.5) !important;
        color: #94A3B8 !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        font-weight: 500 !important;
        font-size: 0.78rem !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        box-shadow: none !important;
        text-align: left !important;
        transition: all 0.2s !important;
    }
    .sample-btn button:hover {
        background: rgba(0, 242, 254, 0.06) !important;
        border-color: rgba(0, 242, 254, 0.2) !important;
        color: #00F2FE !important;
        transform: translateX(4px) !important;
        box-shadow: none !important;
    }

    /* ─── Band Statistics Grid ───────────────────────────────────── */
    .band-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 20px 0; }
    .band-stat-card {
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.07);
        transition: all 0.2s;
    }
    .band-stat-card:hover { transform: translateY(-2px); }
    .band-stat-card.low { background: rgba(0,230,118,0.05); border-color: rgba(0,230,118,0.15); }
    .band-stat-card.medium { background: rgba(255,179,0,0.05); border-color: rgba(255,179,0,0.15); }
    .band-stat-card.high { background: rgba(255,23,68,0.05); border-color: rgba(255,23,68,0.15); }
    .band-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px; }
    .band-stat-card.low .band-label { color: #00E676; }
    .band-stat-card.medium .band-label { color: #FFB300; }
    .band-stat-card.high .band-label { color: #FF1744; }
    .band-count {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1;
        color: #F1F5F9;
    }
    .band-pct { font-size: 0.75rem; color: #475569; margin-top: 4px; }
    .band-prob { font-size: 0.72rem; margin-top: 6px; }
    .band-stat-card.low .band-prob { color: rgba(0,230,118,0.7); }
    .band-stat-card.medium .band-prob { color: rgba(255,179,0,0.7); }
    .band-stat-card.high .band-prob { color: rgba(255,23,68,0.7); }

    /* ─── Progress Bars ──────────────────────────────────────────── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00F2FE, #4FACFE) !important;
        border-radius: 4px !important;
    }

    /* ─── Dividers ───────────────────────────────────────────────── */
    hr { border-color: rgba(255,255,255,0.05) !important; }

    /* ─── DataFrames ─────────────────────────────────────────────── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* ─── Expanders ──────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: rgba(10, 14, 24, 0.3) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
    }

    /* ─── Animations ─────────────────────────────────────────────── */
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }
    @keyframes pulse-glow {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ─── Scrollbar ──────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
    ::-webkit-scrollbar-thumb { background: rgba(0, 242, 254, 0.2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0, 242, 254, 0.4); }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def style_fig(fig, height: int = None) -> go.Figure:
    """Apply the premium dark-mode Plotly theme."""
    update = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94A3B8",
        font_family="Inter",
        title_font_family="Outfit",
        title_font_size=14,
        title_font_color="#E2E8F0",
        legend_bgcolor="rgba(0,0,0,0)",
        legend_bordercolor="rgba(255,255,255,0.05)",
        legend_borderwidth=1,
        margin=dict(t=50, b=35, l=40, r=20),
    )
    if height:
        update["height"] = height
    fig.update_layout(**update)
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.08)",
        tickcolor="rgba(255,255,255,0.08)",
        title_font_color="#475569",
        tickfont=dict(size=9, color="#475569"),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.08)",
        tickcolor="rgba(255,255,255,0.08)",
        title_font_color="#475569",
        tickfont=dict(size=9, color="#475569"),
    )
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CACHED RESOURCE LOADERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def load_predictor() -> RiskPredictor | None:
    try:
        return RiskPredictor()
    except Exception as e:
        logger.warning(f"Predictor not loaded: {e}")
        return None


@st.cache_resource
def load_agent() -> TalkToDataAgent | None:
    try:
        return TalkToDataAgent()
    except Exception as e:
        logger.warning(f"TalkToDataAgent not loaded: {e}")
        return None


@st.cache_data
def get_eda_data() -> tuple[pd.DataFrame, int] | None:
    try:
        loader = DataLoader()
        df = loader.load_application_train(nrows=100000)
        if df is None:
            return None
        full_len = len(df)
        if full_len > 50000:
            df = df.sample(n=50000, random_state=42).reset_index(drop=True)
        return df, full_len
    except Exception as e:
        logger.warning(f"EDA data load failed: {e}")
        return None


@st.cache_data
def load_feature_importances() -> pd.DataFrame | None:
    try:
        path = os.path.join(os.path.dirname(MODEL_PATH), "feature_importances.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception as e:
        logger.warning(f"Feature importance load failed: {e}")
    return None


@st.cache_data
def load_rules() -> dict | None:
    try:
        path = os.path.join(os.path.dirname(MODEL_PATH), "rules.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Rules load failed: {e}")
    return None


@st.cache_data
def load_metrics() -> dict | None:
    try:
        path = os.path.join(os.path.dirname(MODEL_PATH), "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Metrics load failed: {e}")
    return None


predictor = load_predictor()
agent = load_agent()
eda_result = get_eda_data()
eda_df = eda_result[0] if eda_result else None
full_len = eda_result[1] if eda_result else 0
metrics = load_metrics()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("""
    <div class="brand-hero">
        <div class="brand-name">RiskLens</div>
        <div class="brand-tagline">Credit Risk Intelligence</div>
        <span class="brand-version">v2.0 · Enterprise Edition</span>
    </div>
    """, unsafe_allow_html=True)

    if metrics:
        roc = metrics.get("roc_auc", 0)
        pr = metrics.get("pr_auc", 0)
        f1 = metrics.get("f1_score", 0)
        thresh = metrics.get("optimal_threshold", 0.5)

        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">Model Performance</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="padding: 0 16px;">
            <div class="metric-pill">
                <div class="metric-pill-icon" style="background: rgba(0,242,254,0.1);">📈</div>
                <div class="metric-pill-body">
                    <div class="metric-pill-label">ROC-AUC</div>
                    <div class="metric-pill-value">{roc:.4f}</div>
                    <div class="metric-pill-sub">Classification power</div>
                </div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-icon" style="background: rgba(127,0,255,0.1);">🎯</div>
                <div class="metric-pill-body">
                    <div class="metric-pill-label">PR-AUC</div>
                    <div class="metric-pill-value">{pr:.4f}</div>
                    <div class="metric-pill-sub">Precision-recall balance</div>
                </div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-icon" style="background: rgba(0,230,118,0.1);">⚡</div>
                <div class="metric-pill-body">
                    <div class="metric-pill-label">F1 Score</div>
                    <div class="metric-pill-value">{f1:.4f}</div>
                    <div class="metric-pill-sub">Threshold @ {thresh:.3f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ Model metrics not found. Please run the training pipeline.")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Status Indicators
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">System Status</div>
    </div>
    """, unsafe_allow_html=True)

    model_ok = predictor is not None
    data_ok = eda_df is not None
    rules_ok = load_rules() is not None
    api_ok = bool(GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here")

    def status_dot(ok: bool) -> str:
        return ('<span style="color:#00E676;font-size:0.8rem;">●</span>'
                if ok else
                '<span style="color:#FF1744;font-size:0.8rem;">●</span>')

    st.markdown(f"""
    <div style="padding: 0 16px; margin-bottom: 8px;">
        <div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;
                    border-bottom:1px solid rgba(255,255,255,0.03);font-size:0.78rem;">
            <span style="color:#64748B">ML Model</span>
            <span>{status_dot(model_ok)} {'Loaded' if model_ok else 'Missing'}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;
                    border-bottom:1px solid rgba(255,255,255,0.03);font-size:0.78rem;">
            <span style="color:#64748B">Dataset</span>
            <span>{status_dot(data_ok)} {'Loaded' if data_ok else 'Missing'}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;
                    border-bottom:1px solid rgba(255,255,255,0.03);font-size:0.78rem;">
            <span style="color:#64748B">Decision Rules</span>
            <span>{status_dot(rules_ok)} {'Ready' if rules_ok else 'Run Pipeline'}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;font-size:0.78rem;">
            <span style="color:#64748B">Groq API</span>
            <span>{status_dot(api_ok)} {'Connected' if api_ok else 'Key Missing'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)




# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB LAYOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tab_eda, tab_pred, tab_xai, tab_rules, tab_chat = st.tabs([
    "📊  EDA Dashboard",
    "🔍  Risk Predictor",
    "🧠  Explainability",
    "📋  Decision Rules",
    "💬  RiskLens Copilot",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — EDA DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_eda:
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Exploratory Data Intelligence</div>
        <div class="page-subtitle">
            Deep-dive analytics on the Home Credit Default Risk dataset &mdash;
            demographics, repayment patterns, and key risk drivers.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if eda_df is None:
        st.error("Dataset not found. Place `application_train.csv` inside `./data/` and restart.")
    else:
        # ── Derived columns ─────────────────────────────────────────
        df = eda_df.copy()
        if "AGE_YEARS" not in df.columns:
            df["AGE_YEARS"] = df["DAYS_BIRTH"] / -365.0
        if "EMPLOYMENT_YEARS" not in df.columns:
            df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED"].replace(365243, np.nan) / -365.0
        if "CREDIT_INCOME_RATIO" not in df.columns:
            df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
        if "ANNUITY_INCOME_RATIO" not in df.columns:
            df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
        df["Loan Status"] = df["TARGET"].map({0: "Repaid", 1: "Defaulted"})

        default_rate = df["TARGET"].mean() * 100
        avg_income = df["AMT_INCOME_TOTAL"].mean()
        avg_credit = df["AMT_CREDIT"].mean()
        missing_pct = df.isnull().mean().mean() * 100

        # ── KPI Cards ───────────────────────────────────────────────
        st.markdown("""<div class="kpi-grid kpi-grid-4">""", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)

        k1.markdown(f"""
        <div class="kpi-card" style="--kpi-accent: linear-gradient(90deg,#00F2FE,#4FACFE);
             --kpi-glow: rgba(0,242,254,0.06);">
            <span class="kpi-icon">👥</span>
            <div class="kpi-label">Total Applicants</div>
            <div class="kpi-value" style="color:#00F2FE">{full_len:,}</div>
            <div class="kpi-delta">Portfolio size (full dataset)</div>
        </div>""", unsafe_allow_html=True)

        k2.markdown(f"""
        <div class="kpi-card" style="--kpi-accent: linear-gradient(90deg,#FF1744,#FF6D00);
             --kpi-glow: rgba(255,23,68,0.06);">
            <span class="kpi-icon">⚠️</span>
            <div class="kpi-label">Historical Default Rate</div>
            <div class="kpi-value" style="color:#FF1744">{default_rate:.2f}%</div>
            <div class="kpi-delta">Class imbalance: ~91:9</div>
        </div>""", unsafe_allow_html=True)

        k3.markdown(f"""
        <div class="kpi-card" style="--kpi-accent: linear-gradient(90deg,#00E676,#69F0AE);
             --kpi-glow: rgba(0,230,118,0.06);">
            <span class="kpi-icon">💰</span>
            <div class="kpi-label">Avg Annual Income</div>
            <div class="kpi-value" style="color:#00E676">₹{avg_income:,.0f}</div>
            <div class="kpi-delta">Gross income before deductions</div>
        </div>""", unsafe_allow_html=True)

        k4.markdown(f"""
        <div class="kpi-card" style="--kpi-accent: linear-gradient(90deg,#7F00FF,#4FACFE);
             --kpi-glow: rgba(127,0,255,0.06);">
            <span class="kpi-icon">🏦</span>
            <div class="kpi-label">Avg Credit Amount</div>
            <div class="kpi-value" style="color:#A78BFA">₹{avg_credit:,.0f}</div>
            <div class="kpi-delta">Requested loan principal</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        # ── Row 2: Target Distribution + Income Type Default ────────
        c1, c2 = st.columns(2)
        with c1:
            tdf = df["TARGET"].value_counts().reset_index()
            tdf.columns = ["Status", "Count"]
            tdf["Status"] = tdf["Status"].map({0: "Repaid", 1: "Defaulted"})
            fig = px.pie(tdf, names="Status", values="Count",
                         title="Loan Outcome Distribution",
                         color="Status",
                         color_discrete_map={"Repaid": "#00E676", "Defaulted": "#FF1744"},
                         hole=0.55)
            fig.update_traces(textfont_size=11, textfont_color="#E2E8F0",
                              marker=dict(line=dict(color="#060912", width=2)))
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)

        with c2:
            ig = df.groupby("NAME_INCOME_TYPE")["TARGET"].mean().reset_index()
            ig["Default Rate (%)"] = ig["TARGET"] * 100
            fig = px.bar(ig.sort_values("Default Rate (%)", ascending=False),
                         x="NAME_INCOME_TYPE", y="Default Rate (%)",
                         title="Default Rate by Income Type",
                         color="Default Rate (%)",
                         color_continuous_scale=[[0, "#0d1b2a"], [0.5, "#FF6D00"], [1, "#FF1744"]],
                         labels={"NAME_INCOME_TYPE": "Income Type"})
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)

        # ── Row 3: Age Histogram + Education Bar ────────────────────
        c3, c4 = st.columns(2)
        with c3:
            fig = px.histogram(df, x="AGE_YEARS", color="Loan Status", nbins=30,
                               title="Age Distribution by Loan Status",
                               labels={"AGE_YEARS": "Age (Years)"},
                               color_discrete_map={"Repaid": "#00E676", "Defaulted": "#FF1744"},
                               barmode="overlay", opacity=0.75)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)

        with c4:
            eg = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().reset_index()
            eg["Default Rate (%)"] = eg["TARGET"] * 100
            fig = px.bar(eg.sort_values("Default Rate (%)", ascending=False),
                         x="NAME_EDUCATION_TYPE", y="Default Rate (%)",
                         title="Default Rate by Education Level",
                         labels={"NAME_EDUCATION_TYPE": "Education"},
                         color="Default Rate (%)",
                         color_continuous_scale=[[0, "#0d1b2a"], [0.5, "#4FACFE"], [1, "#FF1744"]])
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)

        # ── Row 4: Box + Scatter ─────────────────────────────────────
        c5, c6 = st.columns(2)
        with c5:
            q_lim = df["AMT_INCOME_TOTAL"].quantile(0.97)
            fdf = df[df["AMT_INCOME_TOTAL"] < q_lim].copy()
            fdf["Loan Status"] = fdf["TARGET"].map({0: "Repaid", 1: "Defaulted"})
            fig = px.box(fdf, x="Loan Status", y="AMT_INCOME_TOTAL",
                         title="Income Distribution by Loan Outcome",
                         color="Loan Status",
                         color_discrete_map={"Repaid": "#00E676", "Defaulted": "#FF1744"},
                         labels={"AMT_INCOME_TOTAL": "Annual Income (₹)"})
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)

        with c6:
            sdf = df.sample(min(4000, len(df)), random_state=42).copy()
            sdf["Loan Status"] = sdf["TARGET"].map({0: "Repaid", 1: "Defaulted"})
            fig = px.scatter(sdf, x="AMT_INCOME_TOTAL", y="AMT_CREDIT",
                             color="Loan Status",
                             title="Credit vs Income (4k sample)",
                             labels={"AMT_INCOME_TOTAL": "Income (₹)", "AMT_CREDIT": "Credit (₹)"},
                             color_discrete_map={"Repaid": "#00E676", "Defaulted": "#FF1744"},
                             opacity=0.55)
            fig.update_traces(marker=dict(size=4))
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)

        # ── Correlation Matrix ────────────────────────────────────────
        st.markdown("""
        <div class="page-header" style="margin-top:30px">
            <div class="page-title" style="font-size:1.3rem">Feature Correlation Analysis</div>
        </div>
        """, unsafe_allow_html=True)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in ["SK_ID_CURR", "TARGET"]]
        corrs = df[num_cols].corrwith(df["TARGET"]).abs()
        top20 = corrs.nlargest(20).index.tolist()
        corr_mat = df[top20 + ["TARGET"]].corr()

        fig_heat = px.imshow(corr_mat, text_auto=".2f", aspect="auto",
                             color_continuous_scale="RdBu_r",
                             title="Correlation Matrix: Top 20 Numerical Features")
        st.plotly_chart(style_fig(fig_heat, 520), use_container_width=True)

        # ── Point-Biserial Correlation ────────────────────────────────
        corrs_raw = df[num_cols].corrwith(df["TARGET"])
        top15 = corrs_raw.loc[corrs_raw.abs().nlargest(15).index].reset_index()
        top15.columns = ["Feature", "Correlation"]
        top15["Color"] = top15["Correlation"].apply(lambda x: "#FF1744" if x > 0 else "#00E676")

        fig_corr = px.bar(top15, x="Feature", y="Correlation",
                          title="Top 15 Features Correlated with Default (Point-Biserial)",
                          color="Correlation",
                          color_continuous_scale=[[0, "#00E676"], [0.5, "#111622"], [1, "#FF1744"]])
        fig_corr.update_traces(marker_line_width=0)
        st.plotly_chart(style_fig(fig_corr, 320), use_container_width=True)

        # ── Business Insights Section ─────────────────────────────────
        st.markdown("""
        <div class="page-header" style="margin-top:30px">
            <div class="page-title" style="font-size:1.3rem">Business Intelligence Findings</div>
        </div>
        """, unsafe_allow_html=True)

        i1, i2 = st.columns(2)
        with i1:
            st.markdown("""
            <div class="note-info">
                <strong>📌 Age &amp; Default Vulnerability</strong><br>
                Younger applicants aged 20–30 exhibit default rates nearly double those of clients
                above 45. Lending programs targeting younger demographics should apply tighter
                external score thresholds and income-to-credit ratio limits.
            </div>
            <div class="note-info">
                <strong>📌 External Credit Scores Dominate Risk</strong><br>
                <code>EXT_SOURCE_2</code> is the platform's strongest single predictor of default.
                Clients with EXT_SOURCE_2 below 0.4 are 3× more likely to default compared to
                those above 0.7. Low values should trigger mandatory secondary validation.
            </div>
            <div class="note-info">
                <strong>📌 Education as a Repayment Buffer</strong><br>
                Higher education degree holders default 50–60% less frequently than secondary
                school graduates. Education level is a stable, low-noise demographic feature
                that complements credit score signals.
            </div>
            """, unsafe_allow_html=True)

        with i2:
            st.markdown("""
            <div class="note-info">
                <strong>📌 Leverage Ratio Warning Signal</strong><br>
                Applicants with Credit-to-Income ratios exceeding 5× their annual salary
                represent the highest-risk segment. These clients represent only 8% of
                applications but account for 27% of historical defaults.
            </div>
            <div class="note-info">
                <strong>📌 Income Type Segmentation</strong><br>
                Working-category applicants drive the highest loan volume but also the highest
                absolute default count. Pensioners show the lowest default rates (~5%) and
                represent a high-value, low-risk expansion opportunity.
            </div>
            <div class="note-info">
                <strong>📌 Employment Stability Premium</strong><br>
                Clients with employment history exceeding 5 years default 35% less frequently.
                Duration of employment is a stronger signal than salary level alone, suggesting
                stability over raw income in creditworthiness modeling.
            </div>
            """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — ML RISK PREDICTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_pred:
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Credit Risk Decision Scorecard</div>
        <div class="page-subtitle">
            Input applicant financial parameters to generate a real-time LightGBM risk score
            with SHAP feature-level attribution analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if predictor is None:
        st.error("🚨 ML model artifacts not found. Run `python train_pipeline.py` first.")
    else:
        form_col, result_col = st.columns([0.42, 0.58])

        with form_col:
            st.markdown('<div class="card-title">📝 Applicant Profile Form</div>', unsafe_allow_html=True)
            with st.form("risk_assessment_form"):
                st.markdown("""<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;color:#00F2FE;margin-bottom:4px;">💳 Financial Parameters</p>""",
                    unsafe_allow_html=True)
                amt_income = st.number_input("Annual Income (₹)", min_value=0, value=150000, step=10000)
                amt_credit = st.number_input("Requested Credit Amount (₹)", min_value=0, value=500000, step=25000)
                amt_annuity = st.number_input("Monthly Annuity Payment (₹)", min_value=0, value=25000, step=1000)

                st.markdown("""<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;color:#00F2FE;margin:16px 0 4px;">👤 Demographics</p>""",
                    unsafe_allow_html=True)
                gender = st.selectbox("Applicant Gender", ["M", "F"])
                contract = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
                income_type = st.selectbox("Income Type",
                    ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"])
                education = st.selectbox("Education Level",
                    ["Secondary / secondary special", "Higher education",
                     "Incomplete higher", "Lower secondary", "Academic degree"])
                age = st.slider("Age (Years)", 18, 70, 35)
                emp_years = st.slider("Employment History (Years)", 0, 40, 5)

                st.markdown("""<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;color:#00F2FE;margin:16px 0 4px;">🔢 External Credit Scores</p>""",
                    unsafe_allow_html=True)
                ext_2 = st.slider("EXT_SOURCE_2 Score", 0.0, 1.0, 0.5, step=0.01)
                ext_3 = st.slider("EXT_SOURCE_3 Score", 0.0, 1.0, 0.5, step=0.01)

                st.write("")
                assess_btn = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)

        with result_col:
            if assess_btn:
                input_payload = {
                    "AMT_INCOME_TOTAL": float(amt_income),
                    "AMT_CREDIT": float(amt_credit),
                    "AMT_ANNUITY": float(amt_annuity),
                    "CODE_GENDER": gender,
                    "NAME_CONTRACT_TYPE": contract,
                    "NAME_INCOME_TYPE": income_type,
                    "NAME_EDUCATION_TYPE": education,
                    "DAYS_BIRTH": float(age * -365),
                    "DAYS_EMPLOYED": float(emp_years * -365),
                    "EXT_SOURCE_2": float(ext_2),
                    "EXT_SOURCE_3": float(ext_3),
                }

                with st.spinner("Computing risk assessment..."):
                    res = predictor.predict(input_payload)

                prob_pct = res["raw_probability"] * 100
                band = res["risk_band"]

                rc_class = {"Low": "risk-card-low", "Medium": "risk-card-medium", "High": "risk-card-high"}[band]
                decision = {"Low": "✅ CREDIT AUTO-APPROVED", "Medium": "⚠️ MANUAL REVIEW REQUIRED", "High": "❌ CREDIT DECLINED"}[band]
                desc = {
                    "Low": "Applicant profile reflects low default probability. Credit parameters meet automated approval criteria.",
                    "Medium": "Applicant parameters fall within the borderline zone. Refer to manual underwriting review.",
                    "High": "Applicant risk profile exceeds threshold limits. Recommend declining this credit application."
                }[band]

                st.markdown(f"""
                <div class="risk-card {rc_class}" style="animation: fadeInUp 0.4s ease;">
                    <div class="risk-card-badge-label">{band} Risk Profile</div>
                    <div class="risk-card-prob">{prob_pct:.2f}%</div>
                    <div class="risk-card-band">Default Probability</div>
                    <div class="risk-card-divider"></div>
                    <div class="risk-card-decision">{decision}</div>
                    <div class="risk-card-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

                # Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob_pct,
                    delta={"reference": 50, "valueformat": ".1f",
                           "increasing": {"color": "#FF1744"}, "decreasing": {"color": "#00E676"}},
                    number={"suffix": "%", "font": {"size": 36, "family": "Outfit", "color": "#E2E8F0"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#334155",
                                 "tickwidth": 1, "tickfont": {"size": 9}},
                        "bar": {"color": "#FFFFFF", "thickness": 0.15},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 30], "color": "rgba(0,230,118,0.12)"},
                            {"range": [30, 60], "color": "rgba(255,179,0,0.12)"},
                            {"range": [60, 100], "color": "rgba(255,23,68,0.12)"},
                        ],
                        "threshold": {
                            "line": {"color": "#00F2FE", "width": 2},
                            "thickness": 0.75,
                            "value": prob_pct,
                        },
                    },
                    title={"text": "Default Probability Gauge",
                           "font": {"family": "Outfit", "size": 13, "color": "#64748B"}},
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#94A3B8", "family": "Inter"},
                    margin=dict(t=50, b=10, l=30, r=30),
                    height=210,
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # SHAP Local Explanation
                st.markdown('<div class="card-title">⚡ SHAP Feature Attribution</div>', unsafe_allow_html=True)
                try:
                    shap_data = predictor.get_shap_values(input_payload)
                    top5 = shap_data["top_5_features"]
                    df_shap = pd.DataFrame(top5)
                    df_shap["abs"] = df_shap["shap_value"].abs()
                    df_shap = df_shap.sort_values("abs", ascending=True)

                    fig_shap = px.bar(df_shap, x="shap_value", y="feature", orientation="h",
                                      color="direction",
                                      color_discrete_map={
                                          "increases_risk": "#FF1744",
                                          "decreases_risk": "#00E676"
                                      },
                                      labels={"shap_value": "SHAP Value (log-odds)", "feature": ""},
                                      title="Local Risk Attribution (Top 5 Features)")
                    fig_shap.update_layout(
                        yaxis={"categoryorder": "array", "categoryarray": df_shap["feature"].tolist()},
                        legend_title="Direction",
                        showlegend=True,
                    )
                    fig_shap.update_traces(marker_line_width=0)
                    st.plotly_chart(style_fig(fig_shap, 250), use_container_width=True)
                except Exception as ex:
                    st.error(f"SHAP computation failed: {ex}")

                with st.expander("🔎 View Raw Input Payload"):
                    st.json(input_payload)

                # Global Feature Importance
                fi_df = load_feature_importances()
                if fi_df is not None:
                    with st.expander("📊 Global Model Feature Importance (Top 20)"):
                        fig_gi = px.bar(fi_df.iloc[::-1], x="importance", y="feature",
                                        orientation="h", color="importance",
                                        color_continuous_scale=[[0, "#0d1b2a"], [1, "#00F2FE"]])
                        fig_gi.update_traces(marker_line_width=0)
                        st.plotly_chart(style_fig(fig_gi, 450), use_container_width=True)

            else:
                # Empty state
                fi_df = load_feature_importances()
                if fi_df is not None:
                    st.markdown('<div class="card-title">📊 Global Model Feature Drivers</div>', unsafe_allow_html=True)
                    fig_gi = px.bar(fi_df.iloc[::-1], x="importance", y="feature",
                                    orientation="h",
                                    labels={"importance": "Importance Score", "feature": ""},
                                    color="importance",
                                    color_continuous_scale=[[0, "#0d1b2a"], [1, "#00F2FE"]])
                    fig_gi.update_traces(marker_line_width=0)
                    st.plotly_chart(style_fig(fig_gi, 480), use_container_width=True)
                else:
                    st.markdown("""
                    <div class="empty-state">
                        <div class="empty-state-icon">🛡️</div>
                        <div class="empty-state-title">Awaiting Risk Assessment</div>
                        <div class="empty-state-body">
                            Complete the applicant profile form on the left and click
                            'Assess Credit Risk' to generate a full risk scorecard with SHAP explanations.
                        </div>
                    </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — EXPLAINABILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_xai:
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Model Explainability &amp; Transparency</div>
        <div class="page-subtitle">
            Global and local explanations — understand what drives the model's credit decisions
            through SHAP attributions and feature impact analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if predictor is None:
        st.error("🚨 Model not loaded. Run training pipeline first.")
    else:
        fi_df = load_feature_importances()
        met = load_metrics()

        # ── Section 1: Model Performance Overview ────────────────────
        st.markdown('<div class="card-title">🏅 Model Performance Summary</div>', unsafe_allow_html=True)

        if met:
            conf_mat = met.get("confusion_matrix", [[0, 0], [0, 0]])
            tn, fp = conf_mat[0][0], conf_mat[0][1]
            fn, tp = conf_mat[1][0], conf_mat[1][1]
            total = tn + fp + fn + tp
            accuracy = (tn + tp) / total if total > 0 else 0

            m1, m2, m3, m4, m5 = st.columns(5)
            for col, label, value, color in [
                (m1, "ROC-AUC", f"{met.get('roc_auc', 0):.4f}", "#00F2FE"),
                (m2, "PR-AUC", f"{met.get('pr_auc', 0):.4f}", "#A78BFA"),
                (m3, "F1-Score", f"{met.get('f1_score', 0):.4f}", "#00E676"),
                (m4, "Accuracy", f"{accuracy:.4f}", "#FFB300"),
                (m5, "Threshold", f"{met.get('optimal_threshold', 0.5):.4f}", "#64B5F6"),
            ]:
                col.markdown(f"""
                <div class="kpi-card" style="--kpi-accent: {color};--kpi-glow: {color}22;">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="font-size:1.5rem;color:{color}">{value}</div>
                </div>""", unsafe_allow_html=True)

            # Confusion Matrix Heatmap
            st.write("")
            x1, x2 = st.columns(2)
            with x1:
                cm_data = [[tn, fp], [fn, tp]]
                fig_cm = px.imshow(cm_data,
                                   labels=dict(x="Predicted", y="Actual"),
                                   x=["Predicted: Repaid", "Predicted: Default"],
                                   y=["Actual: Repaid", "Actual: Default"],
                                   color_continuous_scale=[[0, "#060912"], [1, "#00F2FE"]],
                                   text_auto=True,
                                   title="Confusion Matrix (Validation Set)")
                fig_cm.update_traces(textfont_size=14, textfont_color="#E2E8F0")
                st.plotly_chart(style_fig(fig_cm, 300), use_container_width=True)

            with x2:
                # Model Performance Radar
                cats = ["ROC-AUC", "PR-AUC", "F1-Score", "Accuracy", "Recall (Default)"]
                recall_default = tp / (tp + fn) if (tp + fn) > 0 else 0
                vals = [
                    met.get("roc_auc", 0),
                    met.get("pr_auc", 0),
                    met.get("f1_score", 0),
                    accuracy,
                    recall_default,
                ]
                fig_radar = go.Figure(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=cats + [cats[0]],
                    fill="toself",
                    fillcolor="rgba(0,242,254,0.08)",
                    line=dict(color="#00F2FE", width=2),
                    name="LightGBM",
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.08)",
                                        tickfont=dict(size=8, color="#475569"), tickcolor="rgba(0,0,0,0)"),
                        angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                                         tickfont=dict(size=9, color="#94A3B8")),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    title={"text": "Model Metric Radar", "font": {"family": "Outfit", "size": 14, "color": "#E2E8F0"}},
                    height=300,
                    margin=dict(t=50, b=10, l=30, r=30),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        # ── Section 2: Global Feature Importance ─────────────────────
        st.markdown('<div class="card-title" style="margin-top:24px">🌍 Global Feature Importance (LightGBM Split Gain)</div>',
                    unsafe_allow_html=True)

        if fi_df is not None:
            # Horizontal bar chart with gradient coloring
            fig_fi = px.bar(fi_df.iloc[::-1], x="importance", y="feature",
                            orientation="h",
                            title="Top 20 Model Features by Importance Score",
                            labels={"importance": "Split Gain Importance", "feature": "Feature"},
                            color="importance",
                            color_continuous_scale=[[0, "#0d1b2a"], [0.5, "#4FACFE"], [1, "#00F2FE"]])
            fig_fi.update_traces(marker_line_width=0)
            st.plotly_chart(style_fig(fig_fi, 500), use_container_width=True)
        else:
            st.info("Feature importance file not found. Train the model to generate it.")

        # ── Section 3: SHAP Methodology ──────────────────────────────
        st.markdown('<div class="card-title" style="margin-top:24px">📐 SHAP Methodology — How Explanations Work</div>',
                    unsafe_allow_html=True)

        xc1, xc2 = st.columns(2)
        with xc1:
            st.markdown("""
            <div class="note-info">
                <strong>What is SHAP?</strong><br>
                SHAP (SHapley Additive exPlanations) is a game-theoretic framework that
                assigns each feature a contribution value for a specific prediction. It
                answers: "How much did EXT_SOURCE_2 = 0.3 change the predicted default
                probability for this applicant?"
            </div>
            <div class="note-info">
                <strong>TreeExplainer (used here)</strong><br>
                For LightGBM, we use <code>shap.TreeExplainer</code> which is exact and
                computationally efficient — it leverages the tree structure to compute
                Shapley values without Monte Carlo sampling, giving us deterministic,
                reproducible explanations.
            </div>
            """, unsafe_allow_html=True)

        with xc2:
            st.markdown("""
            <div class="note-info">
                <strong>Interpreting the Attribution Chart</strong><br>
                In the Risk Predictor tab, the SHAP bar chart shows each feature's contribution
                to the predicted default probability. <span style="color:#FF1744">Red bars</span>
                indicate features that increase default risk; <span style="color:#00E676">green bars</span>
                indicate features that reduce risk.
            </div>
            <div class="note-info">
                <strong>Base Value (Expected Value)</strong><br>
                The SHAP base value is the model's average prediction across the entire
                training dataset. Each feature's SHAP value adds or subtracts from this
                baseline to arrive at the final predicted probability for the individual applicant.
            </div>
            """, unsafe_allow_html=True)

        # ── Section 4: Class Imbalance Strategy ──────────────────────
        st.markdown('<div class="card-title" style="margin-top:24px">⚖️ Class Imbalance Handling Strategy</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="note-info">
            <strong>The Challenge</strong><br>
            The Home Credit dataset has a severe class imbalance: ~91.9% of applicants repaid
            (TARGET=0) vs ~8.1% defaulted (TARGET=1). A naive model predicts 'Repaid' for
            everything and achieves 91.9% accuracy while being useless for risk detection.
        </div>
        """, unsafe_allow_html=True)

        bal_c1, bal_c2, bal_c3 = st.columns(3)
        for col, technique, desc, icon in [
            (bal_c1, "scale_pos_weight", "LightGBM parameter set to neg/pos ratio (~11.3×) — amplifies the loss gradient for the minority class during training.", "🔧"),
            (bal_c2, "Optimal Threshold", "Instead of the default 0.5 threshold, we optimize the decision threshold via precision-recall curve to maximize F1-Score.", "🎯"),
            (bal_c3, "PR-AUC Evaluation", "We track PR-AUC (Precision-Recall Area Under Curve) as the primary metric, which is more informative than ROC-AUC under imbalance.", "📈"),
        ]:
            col.markdown(f"""
            <div class="kpi-card" style="--kpi-accent: linear-gradient(90deg,#00F2FE,#4FACFE);">
                <span class="kpi-icon">{icon}</span>
                <div class="kpi-label">{technique}</div>
                <div style="font-size:0.78rem;color:#94A3B8;line-height:1.5;margin-top:6px">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — DECISION RULES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_rules:
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Business Decision Rules</div>
        <div class="page-subtitle">
            Human-readable if/then rules extracted from the LightGBM model via a surrogate
            decision tree — enabling transparent, auditable lending decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    rules_data = load_rules()

    if rules_data is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📋</div>
            <div class="empty-state-title">Decision Rules Not Generated Yet</div>
            <div class="empty-state-body">
                Run the full training pipeline to extract business rules:<br><br>
                <code style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                      background:rgba(0,242,254,0.08);padding:6px 12px;border-radius:6px;color:#00F2FE;">
                    python train_pipeline.py
                </code>
                <br><br>This generates <code>models/rules.json</code> with interpretable risk rules.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        rules = rules_data.get("structured_rules", [])
        band_stats = rules_data.get("band_statistics", {})
        n_rules = rules_data.get("n_rules", len(rules))

        # ── Portfolio Band Summary ─────────────────────────────────
        st.markdown('<div class="card-title">📊 Portfolio Risk Band Distribution</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="band-grid">', unsafe_allow_html=True)
        bc1, bc2, bc3 = st.columns(3)
        band_color_map = {"Low": "low", "Medium": "medium", "High": "high"}
        band_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

        for col, band in [(bc1, "Low"), (bc2, "Medium"), (bc3, "High")]:
            s = band_stats.get(band, {})
            cnt = s.get("count", 0)
            pct = s.get("pct_of_portfolio", 0)
            avg_p = s.get("avg_predicted_probability", 0)
            actual_dr = s.get("actual_default_rate")

            actual_str = f"Actual Default Rate: {actual_dr:.1f}%" if actual_dr is not None else ""
            col.markdown(f"""
            <div class="band-stat-card {band_color_map[band]}">
                <div class="band-label">{band_emoji[band]} {band} Risk</div>
                <div class="band-count">{cnt:,}</div>
                <div class="band-pct">{pct:.1f}% of portfolio</div>
                <div class="band-prob">Avg Probability: {avg_p:.1f}%</div>
                {'<div class="band-prob">' + actual_str + '</div>' if actual_str else ''}
            </div>""", unsafe_allow_html=True)

        # ── Band Distribution Chart ──────────────────────────────────
        band_fig_data = [(b, d["count"], d["avg_predicted_probability"])
                         for b, d in band_stats.items()]
        if band_fig_data:
            bdf = pd.DataFrame(band_fig_data, columns=["Risk Band", "Count", "Avg Prob (%)"])
            fig_bands = px.bar(bdf, x="Risk Band", y="Count",
                               color="Risk Band",
                               color_discrete_map={"Low": "#00E676", "Medium": "#FFB300", "High": "#FF1744"},
                               title="Applicant Count by Risk Band",
                               text="Count")
            fig_bands.update_traces(textfont_size=12, textfont_color="#E2E8F0",
                                    marker_line_width=0, textposition="outside")
            st.plotly_chart(style_fig(fig_bands, 280), use_container_width=True)

        # ── Rule Filter Controls ──────────────────────────────────────
        st.markdown('<div class="card-title" style="margin-top:16px">📋 Extracted Decision Rules</div>',
                    unsafe_allow_html=True)

        rf1, rf2, rf3 = st.columns([0.3, 0.3, 0.4])
        with rf1:
            filter_band = st.selectbox("Filter by Risk Band", ["All", "Low", "Medium", "High"],
                                       key="rules_band_filter")
        with rf2:
            min_support = st.slider("Min Support (%)", 0.0, 20.0, 0.0, 0.5, key="rules_support")
        with rf3:
            sort_by = st.selectbox("Sort Rules By",
                                   ["Support (Most Common First)", "Confidence", "Avg Probability"],
                                   key="rules_sort")

        # Apply filters
        filtered_rules = rules
        if filter_band != "All":
            filtered_rules = [r for r in rules if r["risk_band"] == filter_band]
        filtered_rules = [r for r in filtered_rules if r["support_pct"] >= min_support]

        sort_key_map = {
            "Support (Most Common First)": lambda r: -r["support_pct"],
            "Confidence": lambda r: -r["confidence"],
            "Avg Probability": lambda r: -r["avg_default_probability"],
        }
        filtered_rules = sorted(filtered_rules, key=sort_key_map[sort_by])

        st.caption(f"Showing {len(filtered_rules)} of {n_rules} rules")

        # Render rule cards
        band_color_hex = {"Low": "#00E676", "Medium": "#FFB300", "High": "#FF1744"}
        badge_cls_map = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}

        for rule in filtered_rules[:30]:  # Show max 30
            band = rule["risk_band"]
            accent_color = band_color_hex.get(band, "#64748B")
            badge_cls = badge_cls_map.get(band, "")

            # Format conditions with AND highlighting
            conditions_html = ""
            for i, cond in enumerate(rule.get("conditions", [rule.get("rule_text", "")])):
                if i > 0:
                    conditions_html += '<span class="condition-and"> AND </span>'
                conditions_html += cond

            st.markdown(f"""
            <div class="rule-card" style="--rule-color: {accent_color};">
                <div class="rule-header">
                    <span class="rule-id">RULE #{rule['rule_id']:03d}</span>
                    <span class="rule-band-badge {badge_cls}">{band} Risk</span>
                    <span style="font-size:0.7rem;color:#475569;margin-left:auto;">
                        {rule['n_samples']:,} applicants
                    </span>
                </div>
                <div class="rule-conditions">{conditions_html if conditions_html else 'All applicants'}</div>
                <div class="rule-stats">
                    <div class="rule-stat">
                        <span class="rule-stat-label">Support</span>
                        <span class="rule-stat-value">{rule['support_pct']:.1f}%</span>
                    </div>
                    <div class="rule-stat">
                        <span class="rule-stat-label">Confidence</span>
                        <span class="rule-stat-value">{rule['confidence']:.1f}%</span>
                    </div>
                    <div class="rule-stat">
                        <span class="rule-stat-label">Avg Default Prob</span>
                        <span class="rule-stat-value" style="color:{accent_color}">{rule['avg_default_probability']:.1f}%</span>
                    </div>
                    <div class="rule-stat">
                        <span class="rule-stat-label">Defaulted</span>
                        <span class="rule-stat-value">{rule['n_defaulted']:,}</span>
                    </div>
                </div>
                <div class="rule-desc">{rule.get('description', '')}</div>
            </div>
            """, unsafe_allow_html=True)

        # Rule Table View
        with st.expander("📊 View Rules as Summary Table"):
            from src.ml.rules import RulesExtractor
            extractor_dummy = RulesExtractor()
            extractor_dummy.rules_list = filtered_rules
            rt = extractor_dummy.get_rule_table()
            if not rt.empty:
                st.dataframe(rt, use_container_width=True, hide_index=True)

        # Methodology Note
        st.markdown("""
        <div class="note-info" style="margin-top:20px;">
            <strong>📐 How Rules Are Generated</strong><br>
            A shallow <code>DecisionTreeClassifier</code> (max depth=4) is trained as a <em>surrogate model</em>
            on the LightGBM predicted probabilities. The surrogate tree learns to mimic the LightGBM
            decision boundary using interpretable feature thresholds. Each leaf node becomes one
            decision rule with a confidence score (how often the leaf's prediction is correct) and
            a support score (what percentage of the portfolio this rule covers). Rules are derived
            post-training and do not alter the underlying LightGBM predictions.
        </div>
        """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — TALK TO DATA (NL-TO-SQL CHATBOT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_chat:
    st.markdown("""
    <div class="page-header">
        <div class="page-title">RiskLens Copilot — NL-to-SQL Agent</div>
        <div class="page-subtitle">
            Ask natural language questions about the credit risk dataset.
            The AI agent converts your question into SQL, executes it, and returns
            a plain English business summary.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        st.markdown("""
        <div class="empty-state" style="border-color:rgba(239,68,68,0.2);background:rgba(239,68,68,0.03);">
            <div class="empty-state-icon">🔑</div>
            <div class="empty-state-title" style="color:#EF4444">Groq API Key Required</div>
            <div class="empty-state-body">
                Add your Groq API key to the <code>.env</code> file:<br><br>
                <code style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                      background:rgba(239,68,68,0.08);padding:6px 12px;border-radius:6px;color:#EF4444;">
                    GROQ_API_KEY=gsk_your_key_here
                </code>
                <br><br>
                Get a free key at <a href="https://console.groq.com" style="color:#EF4444;">console.groq.com</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif agent is None:
        st.error("❌ Groq LLM could not be initialized. Check API key and network connectivity.")

    else:
        chat_col, panel_col = st.columns([0.65, 0.35])

        with panel_col:
            st.markdown('<div class="card-title">💡 Sample Queries</div>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.78rem;color:#475569;margin-bottom:12px;">Click to run instantly</p>',
                        unsafe_allow_html=True)

            samples = [
                "What is the average income of applicants who defaulted?",
                "Show default rate by education type",
                "How many female applicants have more than 2 children?",
                "What are the top 10 highest credit amounts?",
                "Compare avg credit-to-income ratio by risk band",
                "How many applicants are classified as High risk?",
                "What percentage of pensioners defaulted?",
                "Show average external credit score by risk band",
            ]

            clicked_query = None
            for s in samples:
                st.markdown('<div class="sample-btn">', unsafe_allow_html=True)
                if st.button(s, key=f"sample_{s[:20]}"):
                    clicked_query = s
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="note-info" style="font-size:0.78rem;">
                <strong>Schema Highlights</strong><br>
                Table: <code>applications</code><br>
                Key columns: TARGET, RISK_BAND, RISK_SCORE, AGE_YEARS, AMT_CREDIT, NAME_EDUCATION_TYPE…
            </div>
            """, unsafe_allow_html=True)

        with chat_col:
            if clicked_query:
                st.session_state.chat_history.append({"role": "user", "content": clicked_query})
                with st.spinner("Generating SQL and querying database..."):
                    resp = agent.ask(clicked_query)
                    st.session_state.chat_history.append({"role": "assistant", "content": resp})
                st.rerun()

            # Chat history
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Group alternating user and assistant messages into turns
            turns = []
            history = st.session_state.chat_history
            i = 0
            while i < len(history):
                user_msg = history[i]
                bot_msg = history[i+1] if i + 1 < len(history) else None
                turns.append((user_msg, bot_msg))
                i += 2

            # Render turns in reverse chronological order (newest at the top)
            for user_msg, bot_msg in reversed(turns):
                # Render user message
                st.markdown(
                    f'<div class="msg-user">{user_msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                # Render bot message
                if bot_msg:
                    resp = bot_msg["content"]
                    st.markdown(f"""
                    <div class="msg-bot">
                        <div class="msg-bot-header">
                            <div class="msg-bot-avatar">AI</div>
                            <div class="msg-bot-name">RiskLens Copilot</div>
                        </div>
                        <div class="msg-bot-text">{resp.get('answer', 'No response generated.')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if resp.get("sql_query"):
                        with st.expander("🔍 View Generated SQL Query"):
                            st.code(resp["sql_query"], language="sql")

                    table = resp.get("result_table", {})
                    if table and table.get("rows"):
                        df_t = pd.DataFrame(table["rows"], columns=table["columns"])
                        st.dataframe(df_t.head(20), use_container_width=True, hide_index=True)
                        st.caption(f"📋 {table['row_count']} row(s) returned from database")
            st.markdown("</div>", unsafe_allow_html=True)

            # Input
            user_input = st.chat_input("Ask anything about the credit risk data…")
            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.spinner("Thinking..."):
                    resp = agent.ask(user_input)
                    st.session_state.chat_history.append({"role": "assistant", "content": resp})
                st.rerun()

            # Clear
            if st.session_state.chat_history:
                st.write("")
                cc1, _ = st.columns([0.25, 0.75])
                with cc1:
                    if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
                        st.session_state.chat_history = []
                        agent.clear_memory()
                        st.rerun()
