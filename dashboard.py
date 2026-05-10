"""
ATHENA: AI Workforce Intelligence System
Block 5 - Streamlit Interactive Dashboard
Albert Sans font - Conscious Cybernetics brand consistency

Author: Aditya Roy - Conscious Cybernetics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
import os

# ─────────────────────────────────────────────────────────────────────
# PASSWORD GATE
# ─────────────────────────────────────────────────────────────────────

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown(
            """
            <style>
            .stApp {
                background: radial-gradient(ellipse at center,
                    #0d1117 0%, #0a0c0f 60%, #000000 100%);
            }
            .login-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 85vh;
                text-align: center;
            }
            .login-card {
                background: #111418;
                border: 1px solid #1e2530;
                border-top: 3px solid #f0a500;
                border-radius: 6px;
                padding: 56px 64px;
                width: 480px;
                max-width: 90vw;
            }
            .login-brand {
                color: #f0a500;
                font-family: monospace;
                font-size: 11px;
                letter-spacing: 4px;
                margin: 0 0 16px 0;
            }
            .login-title {
                color: #f0f2f5;
                font-size: 32px;
                font-weight: 300;
                letter-spacing: 2px;
                margin: 0 0 8px 0;
                font-family: 'Albert Sans', sans-serif;
            }
            .login-subtitle {
                color: #7a8799;
                font-size: 13px;
                margin: 0 0 40px 0;
                line-height: 1.7;
            }
            .login-divider {
                height: 1px;
                background: #1e2530;
                margin: 32px 0;
            }
            .login-footer {
                color: #3d4d5c;
                font-size: 10px;
                font-family: monospace;
                letter-spacing: 1px;
                margin-top: 12px;
            }
            </style>

            <div class="login-container">
                <div class="login-card">
                    <p class="login-brand">CONSCIOUS CYBERNETICS</p>
                    <h1 class="login-title">Athena</h1>
                    <p class="login-subtitle">
                        AI Workforce Intelligence System<br>
                        Aerospace Sector Model
                    </p>
                    <div class="login-divider"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([1.2, 1.6, 1.2])
        with col2:
            pwd = st.text_input(
                "",
                type="password",
                placeholder="Enter access code",
                label_visibility="collapsed"
            )
            st.markdown("<div style='height:8px'></div>",
                        unsafe_allow_html=True)
            btn = st.button(
                "Enter Athena",
                use_container_width=True
            )
            if btn:
                if pwd == "conscious":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect access code.")
            st.markdown(
                "<p style='color:#3d4d5c;font-size:10px;"
                "font-family:monospace;text-align:center;"
                "margin-top:16px;letter-spacing:1px'>"
                "consciousCybernetics.org</p>",
                unsafe_allow_html=True
            )
        return False
    return True

if not check_password():
    st.stop()

# ── Auto-setup for Streamlit Cloud deployment ──────────────────────
import subprocess
import os

os.makedirs("outputs", exist_ok=True)

if not os.path.exists("ge_aerosim_workforce.csv"):
    st.info("Setting up Athena for first run... this takes 3-4 minutes.")
    subprocess.run(["python3", "generate_dataset.py"], check=True)

if not os.path.exists("outputs/athena_classifier.pkl"):
    st.info("Training classifier...")
    subprocess.run(["python3", "train_classifier.py"], check=True)

if not os.path.exists("outputs/scenario_results.json"):
    st.info("Running Monte Carlo simulations...")
    subprocess.run(["python3", "monte_carlo.py"], check=True)

if not os.path.exists("outputs/chro_brief.txt"):
    st.info("Generating CHRO brief...")
    subprocess.run(
        ["python3", "generate_brief.py"],
        check=True,
        env={**os.environ, "ANTHROPIC_API_KEY": st.secrets["ANTHROPIC_API_KEY"]}
    )


st.set_page_config(
    page_title="Athena | AI Workforce Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Albert+Sans:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Albert Sans', sans-serif; }
    .stApp { background-color: #0a0c0f; color: #f0f2f5; }
    .main .block-container { padding: 2rem 3rem; }
    h1, h2, h3 { color: #f0f2f5; font-weight: 400; letter-spacing: 1px; }
    [data-testid="stMetricValue"] { color: #f0a500 !important; font-size: 28px !important; }
    [data-testid="stMetricLabel"] { color: #7a8799 !important; }
    div[data-testid="stSidebar"] { background-color: #111418; }
    </style>
    """,
    unsafe_allow_html=True
)

COLORS = {
    "Automate":       "#ff4d4d",
    "Augment":        "#f0a500",
    "Create":         "#00d68f",
    "Judgment-Heavy": "#c084fc"
}

# ─────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("ge_aerosim_workforce.csv")
    with open("outputs/scenario_results.json") as f:
        scenarios = json.load(f)
    brief = open("outputs/chro_brief.txt").read()
    return df, scenarios, brief

df, scenarios, chro_brief = load_data()

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<p style='color:#f0a500;letter-spacing:4px;font-size:10px;"
        "font-family:monospace;margin:0'>CONSCIOUS CYBERNETICS</p>",
        unsafe_allow_html=True
    )
    st.markdown("## Athena")
    st.markdown(
        "<p style='color:#7a8799;font-size:13px;margin-top:0'>"
        "AI Workforce Intelligence System<br>"
        "GE Aero-Sim - 50,000 Employees<br>"
        "7 Global Sites - 55-Skill Taxonomy</p>",
        unsafe_allow_html=True
    )
    st.divider()

    selected_segment = st.selectbox(
        "Segment Filter",
        ["All Segments"] + sorted(df["segment"].unique().tolist())
    )
    selected_site = st.selectbox(
        "Site Filter",
        ["All Sites"] + sorted(df["site"].unique().tolist())
    )
    selected_scenario = st.selectbox(
        "AI Adoption Scenario",
        list(scenarios.keys())
    )
    st.divider()

    st.markdown(
        "<p style='color:#3d4d5c;font-size:10px;font-family:monospace'>"
        "ADITYA ROY<br>"
        "Founder, Conscious Cybernetics<br>"
        "Author, RAG for HR<br>"
        "consciousCybernetics.org<br><br>"
        "All data synthetic.<br>"
        "Calibrated on public GE Aerospace<br>"
        "disclosures only.<br>"
        "NVIDIA DGX - Local inference.<br>"
        "No data left this machine.</p>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────────────────────────────

fdf = df.copy()
if selected_segment != "All Segments":
    fdf = fdf[fdf["segment"] == selected_segment]
if selected_site != "All Sites":
    fdf = fdf[fdf["site"] == selected_site]

# ─────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────

st.markdown(
    "<div style='margin-bottom:28px;border-bottom:1px solid #1e2530;"
    "padding-bottom:20px'>"
    "<p style='color:#f0a500;letter-spacing:4px;font-size:10px;"
    "font-family:monospace;margin:0'>"
    "ATHENA - AI WORKFORCE INTELLIGENCE - FLIGHT DECK ALIGNED</p>"
    "<h1 style='margin:6px 0 4px;font-size:28px'>"
    "AI Workforce Intelligence · Aerospace Sector Model</h1>"
    "<p style='color:#7a8799;font-size:13px;margin:0'>"
    "50,000 employees - 7 global sites - 55-skill taxonomy - "
    "XGBoost + SHAP - Monte Carlo - Claude Sonnet 4 - "
    "NVIDIA DGX local inference</p>"
    "</div>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Workforce Overview",
    "FLIGHT DECK Readiness",
    "SQDC Risk",
    "Transformation Scenarios",
    "CHRO Strategic Brief",
    "Meridian - Internal Talent"
])

# ─────────────────────────────────────────────────────────────────────
# TAB 1 - WORKFORCE OVERVIEW
# ─────────────────────────────────────────────────────────────────────

with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Employees", f"{len(fdf):,}")
    c2.metric(
        "Automation Risk",
        f"{(fdf['ai_impact_label'] == 'Automate').mean() * 100:.1f}%"
    )
    c3.metric(
        "Avg AI Readiness",
        f"{fdf['ai_readiness_score'].mean():.1f}/100"
    )
    c4.metric(
        "Safety x Automate",
        f"{len(fdf[(fdf['ai_impact_label'] == 'Automate') & (fdf['sqdc_safety'] > 0.85)]):,}"
    )
    c5.metric(
        "Judgment-Heavy",
        f"{(fdf['ai_impact_label'] == 'Judgment-Heavy').mean() * 100:.1f}%"
    )

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        ic = fdf["ai_impact_label"].value_counts().reset_index()
        ic.columns = ["Impact", "Count"]
        fig = px.pie(
            ic, values="Count", names="Impact",
            title="Role Impact Distribution - Today",
            color="Impact",
            color_discrete_map=COLORS,
            hole=0.45
        )
        fig.update_layout(
            paper_bgcolor="#111418",
            plot_bgcolor="#111418",
            font_color="#f0f2f5",
            title_font_color="#f0f2f5"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        seg_impact = (
            fdf.groupby("segment")["ai_impact_label"]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0) * 100
        )
        fig2 = go.Figure()
        for label in ["Automate", "Augment", "Create", "Judgment-Heavy"]:
            if label in seg_impact.columns:
                fig2.add_trace(go.Bar(
                    name=label,
                    x=seg_impact.index,
                    y=seg_impact[label],
                    marker_color=COLORS[label],
                    opacity=0.85
                ))
        fig2.update_layout(
            barmode="stack",
            title="Impact by Segment",
            paper_bgcolor="#111418",
            plot_bgcolor="#111418",
            font_color="#f0f2f5",
            title_font_color="#f0f2f5",
            legend=dict(bgcolor="#111418")
        )
        st.plotly_chart(fig2, use_container_width=True)

    site_auto = (
        fdf.groupby("site")
        .apply(lambda x: pd.Series({
            "Automate %": round(
                (x["ai_impact_label"] == "Automate").mean() * 100, 1),
            "Avg AI Readiness": round(
                x["ai_readiness_score"].mean(), 1),
            "Headcount": len(x)
        }))
        .reset_index()
        .sort_values("Automate %", ascending=False)
    )
    fig3 = px.bar(
        site_auto, x="site", y="Automate %",
        color="Avg AI Readiness",
        color_continuous_scale="RdYlGn",
        title="Automation Risk by Global Site",
        text="Automate %"
    )
    fig3.update_layout(
        paper_bgcolor="#111418",
        plot_bgcolor="#111418",
        font_color="#f0f2f5",
        title_font_color="#f0f2f5"
    )
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 2 - FLIGHT DECK READINESS
# ─────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### FLIGHT DECK Readiness Gap Analysis")
    st.markdown(
        "<p style='color:#7a8799;font-size:13px'>"
        "The FLIGHT DECK Readiness Gap measures the distance between "
        "each employee's current operating discipline and what "
        "AI-embedded FLIGHT DECK cadences will require. "
        "High gap = priority intervention target.</p>",
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns(2)

    with col_a:
        fig_fd = px.histogram(
            fdf, x="fd_readiness_gap",
            color="ai_impact_label",
            color_discrete_map=COLORS,
            nbins=40, opacity=0.75,
            barmode="overlay",
            title="FLIGHT DECK Gap Distribution by Impact Category"
        )
        fig_fd.add_vline(
            x=70, line_dash="dash",
            line_color="#ff4d4d",
            annotation_text="High Risk Threshold (70)",
            annotation_font_color="#ff4d4d"
        )
        fig_fd.update_layout(
            paper_bgcolor="#111418",
            plot_bgcolor="#111418",
            font_color="#f0f2f5",
            title_font_color="#f0f2f5"
        )
        st.plotly_chart(fig_fd, use_container_width=True)

    with col_b:
        seg_fd = fdf.groupby("segment").agg(
            avg_fd_gap=("fd_readiness_gap", "mean"),
            high_risk=("fd_readiness_gap", lambda x: (x > 70).sum()),
            headcount=("fd_readiness_gap", "count")
        ).round(1).reset_index()
        seg_fd["high_risk_pct"] = (
            seg_fd["high_risk"] / seg_fd["headcount"] * 100
        ).round(1)

        fig_seg = px.bar(
            seg_fd.sort_values("avg_fd_gap", ascending=True),
            x="avg_fd_gap", y="segment",
            orientation="h",
            color="avg_fd_gap",
            color_continuous_scale="RdYlGn_r",
            title="Avg FLIGHT DECK Gap by Segment",
            text="avg_fd_gap"
        )
        fig_seg.update_layout(
            paper_bgcolor="#111418",
            plot_bgcolor="#111418",
            font_color="#f0f2f5",
            title_font_color="#f0f2f5"
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    st.divider()
    st.markdown("### SHAP Explainability - What Drives Each Category")
    st.markdown(
        "<p style='color:#7a8799;font-size:13px'>"
        "Every prediction is explainable. No black box. "
        "In a safety-critical environment, every verdict "
        "has visible reasoning.</p>",
        unsafe_allow_html=True
    )

    sh1, sh2 = st.columns(2)
    with sh1:
        if os.path.exists("outputs/shap_automate_risk.png"):
            st.image(
                "outputs/shap_automate_risk.png",
                caption="Top Drivers of Automation Risk (SHAP)",
                use_column_width=True
            )
    with sh2:
        if os.path.exists("outputs/shap_judgment_heavy.png"):
            st.image(
                "outputs/shap_judgment_heavy.png",
                caption="Predictors of Judgment-Heavy Roles (SHAP)",
                use_column_width=True
            )


# ─────────────────────────────────────────────────────────────────────
# TAB 3 - SQDC RISK
# ─────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("### SQDC Risk Assessment")
    st.markdown(
        "<p style='color:#7a8799;font-size:13px'>"
        "Safety - Quality - Delivery - Cost. Always in that order. "
        "This view surfaces where AI transformation risk intersects "
        "with GE Aerospace's operational priorities.</p>",
        unsafe_allow_html=True
    )

    critical = fdf[
        (fdf["ai_impact_label"] == "Automate") &
        (fdf["sqdc_safety"] > 0.85)
    ].sort_values("fd_readiness_gap", ascending=False)

    st.markdown(
        "<div style='background:#111418;border:1px solid #ff4d4d;"
        "border-left:4px solid #ff4d4d;border-radius:4px;"
        "padding:16px;margin-bottom:20px'>"
        "<p style='color:#ff4d4d;font-family:monospace;font-size:11px;"
        "letter-spacing:3px;margin:0 0 6px'>CRITICAL INTERSECTION</p>"
        "<p style='color:#f0f2f5;font-size:15px;margin:0'>"
        f"<strong>{len(critical):,} employees</strong> carry both "
        "automation risk AND safety-critical responsibility. "
        "This is GE Aerospace's highest-priority intervention target. "
        "Safety leads SQDC - always.</p></div>",
        unsafe_allow_html=True
    )

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        sqdc_seg = fdf.groupby("segment").agg(
            safety_crit_pct=(
                "sqdc_safety",
                lambda x: (x > 0.85).mean() * 100
            ),
            automate_pct=(
                "ai_impact_label",
                lambda x: (x == "Automate").mean() * 100
            )
        ).round(1).reset_index()

        fig_sqdc = px.scatter(
            sqdc_seg,
            x="automate_pct",
            y="safety_crit_pct",
            color="segment",
            size="automate_pct",
            title="Safety Criticality vs Automation Risk by Segment",
            labels={
                "automate_pct":    "Automation Risk %",
                "safety_crit_pct": "Safety Critical %"
            },
            size_max=50
        )
        fig_sqdc.update_layout(
            paper_bgcolor="#111418",
            plot_bgcolor="#111418",
            font_color="#f0f2f5",
            title_font_color="#f0f2f5"
        )
        st.plotly_chart(fig_sqdc, use_container_width=True)

    with col_s2:
        st.markdown("**Top 10 - Highest FLIGHT DECK Gap in Critical Roles**")
        top_critical = critical.nlargest(10, "fd_readiness_gap")[
            ["employee_id", "segment", "site",
             "level_label", "tenure_years",
             "ai_readiness_score", "fd_readiness_gap"]
        ].round(1)
        st.dataframe(
            top_critical,
            use_container_width=True,
            hide_index=True
        )

    st.divider()
    if os.path.exists("outputs/segment_risk_heatmap.png"):
        st.image(
            "outputs/segment_risk_heatmap.png",
            caption="Segment Transformation Risk Matrix",
            use_column_width=True
        )


# ─────────────────────────────────────────────────────────────────────
# TAB 4 - TRANSFORMATION SCENARIOS
# ─────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("### Workforce Transformation Scenarios")
    st.markdown(
        "<p style='color:#7a8799;font-size:13px'>"
        "1,000 Monte Carlo simulations per scenario - "
        "3-year horizon - 90% confidence intervals - "
        "Select a scenario to explore in detail.</p>",
        unsafe_allow_html=True
    )

    SCENARIO_COLORS = {
        "Automate":       "#ff4d4d",
        "Augment":        "#f0a500",
        "Create":         "#00d68f",
        "Judgment-Heavy": "#c084fc"
    }

    # ── DYNAMIC CHART — updates with dropdown ────────────────────────
    # Build a grouped bar chart for the selected scenario
    # showing mean + 90% CI error bars

    current = scenarios.get(selected_scenario, {})

    if current:
        labels = ["Automate", "Augment", "Create", "Judgment-Heavy"]
        means  = [current[l]["mean"]  for l in labels]
        lowers = [current[l]["mean"]  - current[l]["lower"] for l in labels]
        uppers = [current[l]["upper"] - current[l]["mean"]  for l in labels]

        fig_scenario = go.Figure()

        fig_scenario.add_trace(go.Bar(
            x=labels,
            y=means,
            marker_color=[SCENARIO_COLORS[l] for l in labels],
            opacity=0.85,
            error_y=dict(
                type="data",
                symmetric=False,
                array=uppers,
                arrayminus=lowers,
                color="white",
                thickness=1.5,
                width=6
            ),
            text=[f"{m:.1f}%" for m in means],
            textposition="outside",
            textfont=dict(color="white", size=13)
        ))

        # Baseline reference line
        baseline_auto = (df["ai_impact_label"] == "Automate").mean() * 100
        fig_scenario.add_hline(
            y=baseline_auto,
            line_dash="dash",
            line_color="#ff4d4d",
            line_width=1.5,
            opacity=0.6,
            annotation_text=f"Automate today ({baseline_auto:.0f}%)",
            annotation_font_color="#ff4d4d",
            annotation_position="top right"
        )

        fig_scenario.update_layout(
            title=f"Athena - {selected_scenario} - 3-Year Workforce Projection",
            paper_bgcolor="#111418",
            plot_bgcolor="#111418",
            font_color="#f0f2f5",
            title_font_color="#f0f2f5",
            yaxis=dict(
                title="% of Workforce (50,000 employees)",
                ticksuffix="%",
                gridcolor="#1e2530",
                range=[0, max(means) * 1.35]
            ),
            xaxis=dict(gridcolor="#1e2530"),
            showlegend=False,
            height=480
        )

        st.plotly_chart(fig_scenario, use_container_width=True)

        # Selected scenario callout
        st.markdown(
            "<div style='background:#111418;border:1px solid #f0a500;"
            "border-left:4px solid #f0a500;border-radius:4px;"
            "padding:16px;margin-top:8px'>"
            "<p style='color:#f0a500;font-family:monospace;"
            "font-size:11px;letter-spacing:2px;margin:0 0 6px'>"
            f"SELECTED SCENARIO - {selected_scenario.upper()}</p>"
            "<p style='color:#f0f2f5;font-size:14px;margin:0 0 6px'>"
            f"Automate: <strong>{current['Automate']['mean']:.1f}%</strong>"
            f" - Augment: <strong>{current['Augment']['mean']:.1f}%</strong>"
            f" - Create: <strong>{current['Create']['mean']:.1f}%</strong>"
            f" - Judgment-Heavy: <strong>"
            f"{current['Judgment-Heavy']['mean']:.1f}%</strong></p>"
            "<p style='color:#7a8799;font-size:12px;margin:0'>"
            f"90% CI - Automate: "
            f"{current['Automate']['lower']:.1f}% to "
            f"{current['Automate']['upper']:.1f}% | "
            f"Create: "
            f"{current['Create']['lower']:.1f}% to "
            f"{current['Create']['upper']:.1f}%"
            "</p></div>",
            unsafe_allow_html=True
        )

    st.divider()

    # ── ALL SCENARIOS COMPARISON TABLE ───────────────────────────────
    st.markdown("#### All Scenarios - Side by Side")
    rows = []
    for name, res in scenarios.items():
        rows.append({
            "Scenario":        name,
            "Automate Mean":   f"{res['Automate']['mean']:.1f}%",
            "Automate 90% CI": (
                f"{res['Automate']['lower']:.1f}-"
                f"{res['Automate']['upper']:.1f}%"
            ),
            "Augment Mean":    f"{res['Augment']['mean']:.1f}%",
            "Create Mean":     f"{res['Create']['mean']:.1f}%",
            "Judgment-Heavy":  f"{res['Judgment-Heavy']['mean']:.1f}%"
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )

    # ── ALL SCENARIOS OVERLAY CHART ───────────────────────────────────
    st.markdown("#### All Scenarios - Visual Comparison")
    scenario_names = list(scenarios.keys())
    x = np.arange(len(scenario_names))
    width = 0.20

    fig_all = go.Figure()
    labels = ["Automate", "Augment", "Create", "Judgment-Heavy"]

    for i, label in enumerate(labels):
        means_all  = [scenarios[s][label]["mean"]  for s in scenario_names]
        lowers_all = [scenarios[s][label]["mean"]  -
                      scenarios[s][label]["lower"] for s in scenario_names]
        uppers_all = [scenarios[s][label]["upper"] -
                      scenarios[s][label]["mean"]  for s in scenario_names]

        fig_all.add_trace(go.Bar(
            name=label,
            x=[f"{s}<br>{label}" for s in scenario_names],
            y=means_all,
            marker_color=SCENARIO_COLORS[label],
            opacity=0.85,
            error_y=dict(
                type="data",
                symmetric=False,
                array=uppers_all,
                arrayminus=lowers_all,
                color="white",
                thickness=1.2,
                width=5
            )
        ))

    fig_all.update_layout(
        barmode="group",
        paper_bgcolor="#111418",
        plot_bgcolor="#111418",
        font_color="#f0f2f5",
        title_font_color="#f0f2f5",
        yaxis=dict(
            title="% of Workforce",
            ticksuffix="%",
            gridcolor="#1e2530"
        ),
        xaxis=dict(gridcolor="#1e2530"),
        legend=dict(bgcolor="#111418"),
        height=420
    )
    st.plotly_chart(fig_all, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 5 - CHRO STRATEGIC BRIEF
# ─────────────────────────────────────────────────────────────────────

with tab5:
    st.markdown("### CHRO Strategic Brief")
    st.markdown(
        "<p style='color:#7a8799;font-size:13px'>"
        "LLM-generated executive narrative - Claude Sonnet 4 - "
        "FLIGHT DECK vocabulary - Board-ready structure - "
        "Generated from Athena model outputs</p>",
        unsafe_allow_html=True
    )
    st.divider()

    brief_html = chro_brief.replace("\n", "<br>")
    st.markdown(
        "<div style='background:#111418;border:1px solid #1e2530;"
        "border-left:4px solid #f0a500;border-radius:4px;"
        "padding:32px;font-size:14px;line-height:2.0;"
        "color:#c8d0d8;font-family:Georgia,serif;'>"
        f"{brief_html}"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='color:#3d4d5c;font-size:10px;"
        "font-family:monospace;margin-top:16px'>"
        "ATHENA WORKFORCE INTELLIGENCE - CONSCIOUS CYBERNETICS - "
        "ADITYA ROY - consciousCybernetics.org - "
        "All data synthetic - Calibrated on public disclosures only</p>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────
# TAB 6 - MERIDIAN
# ─────────────────────────────────────────────────────────────────────

with tab6:
    st.markdown("### Meridian: Internal Talent Intelligence Layer")
    st.markdown(
        "<p style='color:#7a8799;font-size:13px'>"
        "Athena identifies which roles will change. "
        "Meridian finds who inside GE Aerospace is already on the "
        "trajectory to fill them - before they know it themselves. "
        "Four-agent reasoning pipeline - Trajectory inference - "
        "Not keyword matching.</p>",
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        "<div style='background:#111418;border:1px solid #f0a500;"
        "border-left:4px solid #f0a500;border-radius:4px;"
        "padding:20px;margin-bottom:24px'>"
        "<p style='color:#f0a500;font-family:monospace;font-size:11px;"
        "letter-spacing:3px;margin:0 0 8px'>MERIDIAN TRIGGER</p>"
        "<p style='color:#f0f2f5;font-size:14px;margin:0'>"
        "A new role has been created: <strong>FLIGHT DECK AI Integration"
        " Architect, Commercial Engines and Services.</strong> "
        "It does not exist on any internal career ladder. "
        "Standard HR keyword search returned zero results. "
        "Meridian was given one input: the job description in plain text."
        "</p></div>",
        unsafe_allow_html=True
    )

    candidates = [
        {
            "name":         "M. Okonkwo",
            "current_role": "Senior Quality and Safety Engineer - Evendale, OH",
            "signals": [
                "Completed Lean Six Sigma Black Belt and ML Foundations "
                "certification (last 14 months)",
                "Volunteered for 3 of 4 Digital FLIGHT DECK working "
                "group sessions - unprompted",
                "Performance review language shifted: "
                "process adherent to systems architect",
                "Cross-functional collaboration: "
                "Quality to Digital Engineering (new, 2024)"
            ],
            "case_for": (
                "Okonkwo's trajectory shows a consistent shift from "
                "execution-layer quality work toward governance and "
                "systems design. The convergence of ML certification, "
                "voluntary FLIGHT DECK digital participation, and a "
                "semantic shift in review language constitutes a "
                "coherent trajectory signal. The role requires someone "
                "who understands both lean operating discipline and AI "
                "integration - this aligns closely with the direction "
                "of travel, not merely the current position."
            ),
            "confidence":   "Medium-High",
            "uncertainty":  "Depth of Python and tooling proficiency. Probe directly.",
            "color":        "#f0a500"
        },
        {
            "name":         "S. Lindberg",
            "current_role": "Propulsion Engineer to Digital Engineering "
                            "Specialist (self-initiated, 16 months ago)",
            "signals": [
                "Self-initiated role change - strongest single "
                "trajectory signal available",
                "Attended 5 AI in Aerospace forums (external, "
                "voluntary, last 18 months)",
                "Contributing to Open Fan RISE digital twin workstream "
                "(non-mandatory)",
                "Review text: Lindberg consistently sees the system, "
                "not just the component"
            ],
            "case_for": (
                "The self-initiated transition from Propulsion Engineering "
                "to Digital Engineering is the clearest trajectory evidence "
                "Meridian can surface. It represents a deliberate, voluntary "
                "move toward the exact convergence this role requires. "
                "External forum attendance and digital twin contribution "
                "reinforce that this is direction, not drift."
            ),
            "confidence":   "High",
            "uncertainty":  "Depth of AI governance vs AI application knowledge. Verify.",
            "color":        "#00d68f"
        },
        {
            "name":         "P. Vasquez",
            "current_role": "Supply Chain Lead - Manufacturing and Operations "
                            "- Asheville, NC",
            "signals": [
                "Led cross-functional kaizen with Digital Engineering - "
                "first ops leader to do so",
                "Completed Value Stream Management and AI Tools Proficiency "
                "modules (Q1 2026)",
                "Persistent lateral connections with Transformation Office "
                "(2 years, growing)",
                "Review language: operationally rigorous to architecturally minded"
            ],
            "case_for": (
                "Vasquez represents the FLIGHT DECK trajectory signal GE "
                "values most: someone who has mastered operational discipline "
                "and is now reaching toward systems design. The kaizen "
                "cross-function initiative was self-proposed. The lateral "
                "Transformation Office connections are persistent and growing. "
                "This is a person becoming ready - not yet ready - with a "
                "6 to 12 month readiness horizon."
            ),
            "confidence":   "Medium",
            "uncertainty":  "AI technical depth emerging, not established. "
                            "High retention risk if not engaged now.",
            "color":        "#c084fc"
        }
    ]

    for cand in candidates:
        with st.expander(
            f"{cand['name']} - {cand['current_role']}",
            expanded=False
        ):
            col_l, col_r = st.columns([1, 1])
            with col_l:
                st.markdown(
                    "<p style='color:#7a8799;font-size:11px;"
                    "font-family:monospace;letter-spacing:2px'>"
                    "TRAJECTORY SIGNALS</p>",
                    unsafe_allow_html=True
                )
                for sig in cand["signals"]:
                    st.markdown(
                        f"<div style='display:flex;gap:10px;"
                        f"margin-bottom:10px'>"
                        f"<span style='color:{cand['color']};"
                        f"font-size:10px;margin-top:3px'>+</span>"
                        f"<span style='color:#c8d0d8;font-size:13px'>"
                        f"{sig}</span></div>",
                        unsafe_allow_html=True
                    )
            with col_r:
                st.markdown(
                    f"<div style='background:#0a0c0f;border-left:3px solid "
                    f"{cand['color']};padding:20px;border-radius:3px'>"
                    f"<p style='color:{cand['color']};font-family:monospace;"
                    f"font-size:10px;letter-spacing:2px;margin:0 0 10px'>"
                    f"MERIDIAN A4 - MATCH CASE</p>"
                    f"<p style='color:#c8d0d8;font-size:13px;"
                    f"line-height:1.9;font-family:Georgia,serif;"
                    f"margin:0 0 14px'>{cand['case_for']}</p>"
                    f"<p style='color:{cand['color']};font-size:12px;"
                    f"margin:0 0 4px'>"
                    f"Confidence: {cand['confidence']}</p>"
                    f"<p style='color:#7a8799;font-size:12px;"
                    f"font-style:italic;margin:0'>"
                    f"What we cannot know: {cand['uncertainty']}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    st.divider()
    st.markdown(
        "<div style='background:#111418;border:1px solid #1e2530;"
        "border-radius:4px;padding:20px'>"
        "<p style='color:#3d4d5c;font-size:11px;font-family:monospace;"
        "margin:0;line-height:1.8'>"
        "MERIDIAN produces arguments, not decisions. "
        "All talent decisions remain with qualified human professionals. "
        "No individual profiling for adverse action. "
        "Mandatory uncertainty disclosure on every match case. "
        "Human-in-the-loop by design. "
        "Conscious Cybernetics Ethical Architecture</p></div>",
        unsafe_allow_html=True
    )
