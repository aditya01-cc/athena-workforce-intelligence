"""
ATHENA: AI Workforce Intelligence System
Block 3 — Monte Carlo Workforce Transformation Scenarios (v2)

Fix: Scenario-level uncertainty + corrected classification logic.
Each simulation now varies the adoption rate itself — capturing
the real uncertainty of enterprise transformation: will leadership
stay committed? Will site managers adopt consistently? Will the
reskilling budget survive the next economic cycle?

Author: Aditya Roy · Conscious Cybernetics™
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

os.makedirs("outputs", exist_ok=True)
np.random.seed(42)

SCENARIOS = {
    "Conservative\n(30% Adoption)": {
        "adoption_mean":   0.30,
        "adoption_std":    0.10,   # high uncertainty — early stage
        "reskilling_mean": 0.25,
        "reskilling_std":  0.10,
        "color":           "#f0a500",
    },
    "Moderate\n(60% Adoption)": {
        "adoption_mean":   0.60,
        "adoption_std":    0.12,
        "reskilling_mean": 0.55,
        "reskilling_std":  0.12,
        "color":           "#00c9d4",
    },
    "Accelerated\n(90% Adoption)": {
        "adoption_mean":   0.90,
        "adoption_std":    0.08,   # tighter — committed programme
        "reskilling_mean": 0.80,
        "reskilling_std":  0.08,
        "color":           "#00d68f",
    }
}

IMPACT_COLORS = {
    "Automate":       "#ff4d4d",
    "Augment":        "#f0a500",
    "Create":         "#00d68f",
    "Judgment-Heavy": "#c084fc"
}

def run_simulation(df, adoption_mean, adoption_std,
                   reskilling_mean, reskilling_std,
                   n_simulations=1000, years=3):
    """
    v3 — Three fixes:
    1. JH gets small threshold noise per simulation → real CI width
    2. Create threshold loosened + grows with AI skill gains
    3. Automate floor: resistant population stays stuck regardless
       of reskilling — not everyone will adapt
    """

    results = {
        "Automate":       [],
        "Augment":        [],
        "Create":         [],
        "Judgment-Heavy": []
    }

    n = len(df)

    # Static arrays — these don't change with AI adoption
    tenure    = df["tenure_years"].values.copy()
    level     = df["level"].values.copy()
    sqdc_safe = df["sqdc_safety"].values.copy()
    dom_skills= df["avg_domain_skills"].values.copy()

    # Resistance flag — top 8% most tenure-heavy + lowest AI readiness
    # These employees will NOT reskill regardless of investment.
    # Every enterprise has them. Pretending otherwise is dishonest.
    resistance_threshold_ai  = np.percentile(
        df["ai_readiness_score"].values, 12)
    resistance_threshold_lv  = np.percentile(
        df["learning_velocity"].values, 12)
    resistant = (
        (df["ai_readiness_score"].values < resistance_threshold_ai) &
        (df["learning_velocity"].values < resistance_threshold_lv) &
        (tenure > 16)
    )

    # Starting arrays — evolve each simulation
    ai_base   = df["ai_readiness_score"].values.copy()
    lv_base   = df["learning_velocity"].values.copy()
    fd_base   = df["fd_readiness_gap"].values.copy()
    ask_base  = df["avg_ai_skills"].values.copy()

    for sim in range(n_simulations):

        # Simulation-level adoption and reskilling rates
        sim_adoption   = np.clip(
            np.random.normal(adoption_mean, adoption_std), 0.05, 1.0)
        sim_reskilling = np.clip(
            np.random.normal(reskilling_mean, reskilling_std), 0.05, 1.0)

        # Evolve workforce over years
        s_ai   = ai_base.copy()
        s_lv   = lv_base.copy()
        s_fd   = fd_base.copy()
        s_ask  = ask_base.copy()

        for year in range(years):
            gain_ai  = np.random.normal(sim_reskilling * 7, 4.5, n)
            gain_lv  = np.random.normal(sim_adoption   * 4, 3.0, n)
            gain_fd  = np.random.normal(sim_adoption   * 5, 3.0, n)
            gain_ask = np.random.normal(sim_reskilling * 0.22, 0.12, n)

            # Resistant employees gain at 15% of normal rate
            gain_ai[resistant]  *= 0.15
            gain_lv[resistant]  *= 0.15
            gain_ask[resistant] *= 0.15

            s_ai  = np.clip(s_ai  + gain_ai,  0, 100)
            s_lv  = np.clip(s_lv  + gain_lv,  0, 100)
            s_fd  = np.clip(s_fd  - gain_fd,   0, 100)
            s_ask = np.clip(s_ask + gain_ask,  0, 5)

        # ── CLASSIFICATION (v3) ──────────────────────────────────────

        # JUDGMENT-HEAVY: safety-critical domain experts.
        # Small threshold noise per simulation = realistic CI width.
        # The noise represents: will this manager flag this role?
        # Will this year's safety audit change the classification?
        jh_threshold_safe  = 0.88 + np.random.normal(0, 0.03)
        jh_threshold_dom   = 3.0  + np.random.normal(0, 0.15)
        jh_threshold_level = 3

        jh_mask = (
            (sqdc_safe  > jh_threshold_safe) &
            (dom_skills > jh_threshold_dom)  &
            (level      >= jh_threshold_level)
        )

        # AUTOMATE: resistant population + anyone still genuinely stuck.
        # Floor exists because some people don't reskill regardless.
        automate_mask = (
            resistant |
            (
                (s_ai < 38) &
                (tenure > 16) &
                (s_lv < 40)
            )
        ) & ~jh_mask

        # CREATE: grows with adoption.
        # Threshold eases as AI tools proliferate —
        # more people can design AI-native workflows
        # when the tools are everywhere around them.
        create_threshold_ai  = 62 - (sim_adoption * 8)  # eases with adoption
        create_threshold_ask = 2.8 - (sim_adoption * 0.3)
        create_threshold_lv  = 58

        create_mask = (
            (s_ai  > create_threshold_ai)  &
            (s_ask > create_threshold_ask) &
            (s_lv  > create_threshold_lv)  &
            (level >= 3)
        ) & ~jh_mask & ~automate_mask

        # AUGMENT: everyone else
        augment_mask = ~jh_mask & ~automate_mask & ~create_mask

        total = n
        results["Judgment-Heavy"].append(jh_mask.sum()   / total * 100)
        results["Automate"].append(automate_mask.sum()    / total * 100)
        results["Create"].append(create_mask.sum()        / total * 100)
        results["Augment"].append(augment_mask.sum()      / total * 100)

    summary = {}
    for label, values in results.items():
        arr = np.array(values)
        summary[label] = {
            "mean":   round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
            "lower":  round(float(np.percentile(arr, 5)), 2),
            "upper":  round(float(np.percentile(arr, 95)), 2),
            "std":    round(float(arr.std()), 2)
        }
    return summary


# ─────────────────────────────────────────────────────────────────────
# RUN ALL SCENARIOS
# ─────────────────────────────────────────────────────────────────────

print("Loading GE Aero-Sim dataset...")
df = pd.read_csv("ge_aerosim_workforce.csv")
print(f"  {len(df):,} employees loaded\n")

print("Baseline workforce state:")
baseline = df["ai_impact_label"].value_counts()
for label, count in baseline.items():
    print(f"  {label:<18} {count:>6,}  ({count/len(df)*100:.1f}%)")
print()

all_results = {}

for scenario_name, config in SCENARIOS.items():
    clean = scenario_name.replace("\n", " ")
    print(f"Running: {clean}")
    print(f"  Adoption: {config['adoption_mean']*100:.0f}% "
          f"±{config['adoption_std']*100:.0f}% · "
          f"Reskilling: {config['reskilling_mean']*100:.0f}% "
          f"±{config['reskilling_std']*100:.0f}%")
    print(f"  1,000 simulations × 3 years × 50,000 employees...")

    result = run_simulation(
        df,
        adoption_mean=config["adoption_mean"],
        adoption_std=config["adoption_std"],
        reskilling_mean=config["reskilling_mean"],
        reskilling_std=config["reskilling_std"],
        n_simulations=1000,
        years=3
    )
    all_results[clean] = result

    print(f"  Results (mean · 90% CI):")
    for label in ["Automate", "Augment", "Create", "Judgment-Heavy"]:
        r = result[label]
        print(f"    {label:<18} "
              f"{r['mean']:>5.1f}%  "
              f"[{r['lower']:.1f}–{r['upper']:.1f}%]")
    print()

with open("outputs/scenario_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("✓ Scenario results saved: outputs/scenario_results.json\n")


# ─────────────────────────────────────────────────────────────────────
# VISUALISATION 1: SCENARIO COMPARISON
# ─────────────────────────────────────────────────────────────────────

scenario_names = list(SCENARIOS.keys())
labels         = ["Automate", "Augment", "Create", "Judgment-Heavy"]
x              = np.arange(len(scenario_names))
width          = 0.20

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor('#0a0c0f')
ax.set_facecolor('#111418')

for i, label in enumerate(labels):
    means  = [all_results[sn.replace("\n"," ")][label]["mean"]
              for sn in scenario_names]
    lowers = [all_results[sn.replace("\n"," ")][label]["mean"] -
              all_results[sn.replace("\n"," ")][label]["lower"]
              for sn in scenario_names]
    uppers = [all_results[sn.replace("\n"," ")][label]["upper"] -
              all_results[sn.replace("\n"," ")][label]["mean"]
              for sn in scenario_names]

    bars = ax.bar(
        x + i * width, means,
        width * 0.85,
        label=label,
        color=IMPACT_COLORS[label],
        alpha=0.85
    )
    ax.errorbar(
        x + i * width, means,
        yerr=[lowers, uppers],
        fmt='none', color='white',
        capsize=5, linewidth=1.2, alpha=0.7
    )
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.5,
            f'{mean:.0f}%',
            ha='center', va='bottom',
            color='white', fontsize=9, fontweight='bold'
        )

# Baseline reference
baseline_auto = (df["ai_impact_label"] == "Automate").mean() * 100
ax.axhline(
    y=baseline_auto,
    color='#ff4d4d', linestyle='--',
    linewidth=1.2, alpha=0.5,
    label=f'Automate today ({baseline_auto:.0f}%)'
)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(
    [s.replace("\n", "\n") for s in scenario_names],
    color='#f0f2f5', fontsize=12
)
ax.set_ylabel(
    '% of Workforce (50,000 employees)',
    color='#7a8799', fontsize=11
)
ax.set_ylim(0, 75)
ax.tick_params(colors='#7a8799')
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f'{y:.0f}%')
)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('#1e2530')

ax.legend(
    facecolor='#111418', edgecolor='#1e2530',
    labelcolor='#f0f2f5', fontsize=10, loc='upper right'
)
ax.set_title(
    'Athena · GE Aero-Sim: Workforce Transformation Scenarios\n'
    '1,000 Monte Carlo Simulations · 3-Year Horizon · '
    '90% Confidence Intervals · Conscious Cybernetics™',
    color='#f0f2f5', fontsize=13, pad=20
)

plt.tight_layout()
plt.savefig(
    'outputs/monte_carlo_scenarios.png',
    dpi=150, bbox_inches='tight', facecolor='#0a0c0f'
)
plt.close()
print("✓ Scenario chart saved: outputs/monte_carlo_scenarios.png")


# ─────────────────────────────────────────────────────────────────────
# VISUALISATION 2: SEGMENT RISK HEAT MAP
# ─────────────────────────────────────────────────────────────────────

print("Generating segment risk heat map...")

segments = df["segment"].unique()
segment_metrics = []

for seg in segments:
    seg_df = df[df["segment"] == seg]
    segment_metrics.append({
        "Segment":           seg,
        "Automate %":        round((seg_df["ai_impact_label"] == "Automate").mean() * 100, 1),
        "Judgment-Heavy %":  round((seg_df["ai_impact_label"] == "Judgment-Heavy").mean() * 100, 1),
        "Avg AI Readiness":  round(seg_df["ai_readiness_score"].mean(), 1),
        "Avg FD Gap":        round(seg_df["fd_readiness_gap"].mean(), 1),
        "Safety Critical %": round((seg_df["sqdc_safety"] > 0.85).mean() * 100, 1),
        "High Risk Count":   int((seg_df["fd_readiness_gap"] > 70).sum())
    })

seg_summary = pd.DataFrame(segment_metrics).set_index("Segment")

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#0a0c0f')
ax.set_facecolor('#111418')

data_norm = (seg_summary - seg_summary.min()) / (
    seg_summary.max() - seg_summary.min() + 1e-9)

ax.imshow(
    data_norm.T.values,
    cmap='RdYlGn_r', aspect='auto', alpha=0.8
)
ax.set_xticks(range(len(seg_summary.index)))
ax.set_xticklabels(
    [s.replace(" & ", "\n& ") for s in seg_summary.index],
    color='#f0f2f5', fontsize=9
)
ax.set_yticks(range(len(seg_summary.columns)))
ax.set_yticklabels(seg_summary.columns, color='#f0f2f5', fontsize=10)

for i in range(len(seg_summary.columns)):
    for j in range(len(seg_summary.index)):
        ax.text(
            j, i, str(seg_summary.iloc[j, i]),
            ha='center', va='center',
            color='white', fontsize=9, fontweight='bold'
        )

ax.set_title(
    'Athena · GE Aero-Sim: Segment Transformation Risk Matrix\n'
    'Red = Higher Risk · Green = Lower Risk · Conscious Cybernetics™',
    color='#f0f2f5', fontsize=12, pad=16
)
for spine in ax.spines.values():
    spine.set_color('#1e2530')

plt.tight_layout()
plt.savefig(
    'outputs/segment_risk_heatmap.png',
    dpi=150, bbox_inches='tight', facecolor='#0a0c0f'
)
plt.close()
print("✓ Segment heat map saved: outputs/segment_risk_heatmap.png")


# ─────────────────────────────────────────────────────────────────────
# STRATEGIC SUMMARY
# ─────────────────────────────────────────────────────────────────────

print("\n── Strategic Summary ──────────────────────────────────────")
moderate = all_results["Moderate (60% Adoption)"]
accel    = all_results["Accelerated (90% Adoption)"]
conserv  = all_results["Conservative (30% Adoption)"]

b_auto   = (df["ai_impact_label"] == "Automate").mean() * 100
b_create = (df["ai_impact_label"] == "Create").mean()   * 100
b_jh     = (df["ai_impact_label"] == "Judgment-Heavy").mean() * 100

print(f"\n  Automate risk:  {b_auto:.1f}% today → "
      f"{moderate['Automate']['mean']:.1f}% (Moderate) → "
      f"{accel['Automate']['mean']:.1f}% (Accelerated)")
print(f"  Create roles:   {b_create:.1f}% today → "
      f"{moderate['Create']['mean']:.1f}% (Moderate) → "
      f"{accel['Create']['mean']:.1f}% (Accelerated)")
print(f"  Judgment-Heavy: {b_jh:.1f}% today → "
      f"{conserv['Judgment-Heavy']['mean']:.1f}% (Conservative) → "
      f"{accel['Judgment-Heavy']['mean']:.1f}% (Accelerated)")
print(f"\n  Key insight: Judgment-Heavy variance across scenarios: "
      f"{abs(accel['Judgment-Heavy']['mean'] - conserv['Judgment-Heavy']['mean']):.1f}pp")
print(f"  Human judgment in safety-critical aerospace is "
      f"structurally resilient to AI adoption level.")

print(f"\n✓ Block 3 complete (v2 — scenario-level uncertainty).")
print(f"  Confidence intervals now reflect real transformation uncertainty.")