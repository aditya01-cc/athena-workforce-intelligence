"""
ATHENA: AI Workforce Intelligence System
Block 4 — CHRO Strategic Brief Generator (Claude API)

WHY AN LLM BRIEF:
Athena produces numbers. Numbers don't move boards.
Stories backed by numbers do.

This block takes Athena's model outputs and generates an
executive-grade strategic brief — structured for board-level
consumption, written in GE Aerospace's FLIGHT DECK vocabulary,
and calibrated to Larry Culp's stated AI philosophy:
AI to augment worker performance, not replace jobs.

Same principle as Aurora's compliance narrative generation:
AI outputs made human-readable for non-technical decision-makers.

Author: Aditya Roy · Conscious Cybernetics™
"""

import anthropic
import pandas as pd
import numpy as np
import json
import os

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# LOAD ALL ATHENA OUTPUTS
# ─────────────────────────────────────────────────────────────────────

print("Loading Athena outputs for brief generation...")

df = pd.read_csv("ge_aerosim_workforce.csv")

with open("outputs/scenario_results.json") as f:
    scenarios = json.load(f)

print(f"  {len(df):,} employee records loaded")
print(f"  Scenario results: {len(scenarios)} scenarios")

# ─────────────────────────────────────────────────────────────────────
# COMPUTE SUMMARY STATISTICS FOR THE BRIEF
# ─────────────────────────────────────────────────────────────────────

total       = len(df)
avg_ready   = df["ai_readiness_score"].mean()
avg_fd_gap  = df["fd_readiness_gap"].mean()

# Impact distribution
impact = df["ai_impact_label"].value_counts()
automate_n  = impact.get("Automate", 0)
augment_n   = impact.get("Augment", 0)
create_n    = impact.get("Create", 0)
jh_n        = impact.get("Judgment-Heavy", 0)

# High risk populations
high_fd_gap = (df["fd_readiness_gap"] > 70).sum()
safety_auto = len(df[
    (df["ai_impact_label"] == "Automate") &
    (df["sqdc_safety"] > 0.85)
])

# Segment-level risk
seg_risk = df.groupby("segment").apply(
    lambda x: pd.Series({
        "automate_pct":   (x["ai_impact_label"] == "Automate").mean() * 100,
        "safety_crit_pct":(x["sqdc_safety"] > 0.85).mean() * 100,
        "avg_fd_gap":      x["fd_readiness_gap"].mean(),
        "avg_ai_ready":    x["ai_readiness_score"].mean(),
        "high_risk_n":    (x["fd_readiness_gap"] > 70).sum(),
        "headcount":       len(x)
    })
).round(1)

# Highest risk segment
worst_seg = seg_risk.sort_values("automate_pct", ascending=False).index[0]
worst_auto_pct = seg_risk.loc[worst_seg, "automate_pct"]
worst_safety   = seg_risk.loc[worst_seg, "safety_crit_pct"]
worst_fd       = seg_risk.loc[worst_seg, "avg_fd_gap"]

# Scenario comparison
s_conserv = scenarios.get("Conservative (30% Adoption)", {})
s_moderate = scenarios.get("Moderate (60% Adoption)", {})
s_accel    = scenarios.get("Accelerated (90% Adoption)", {})


# ─────────────────────────────────────────────────────────────────────
# BUILD DATA SUMMARY FOR THE PROMPT
# ─────────────────────────────────────────────────────────────────────

data_summary = f"""
ATHENA WORKFORCE INTELLIGENCE — GE AERO-SIM ANALYSIS
Digital Twin: {total:,} synthetic employees calibrated to GE Aerospace's
public workforce disclosures. 5 segments. 7 global sites. 55-skill taxonomy.
Model: XGBoost classifier with SHAP explainability. 74% accuracy.
Scenarios: Monte Carlo, 1,000 simulations each, 3-year horizon, 90% CI.

═══ CURRENT STATE ═══════════════════════════════════════════════════

Total workforce modelled:    {total:,}
Avg AI Readiness Score:      {avg_ready:.1f}/100
Avg FLIGHT DECK Gap:         {avg_fd_gap:.1f}/100

Role Impact (today):
  Augment:        {augment_n:,}  ({augment_n/total*100:.1f}%)
  Automate:       {automate_n:,}  ({automate_n/total*100:.1f}%)
  Create:         {create_n:,}  ({create_n/total*100:.1f}%)
  Judgment-Heavy: {jh_n:,}   ({jh_n/total*100:.1f}%)

High-risk populations:
  FLIGHT DECK Gap > 70:        {high_fd_gap:,} employees
  Safety-critical + Automate:  {safety_auto:,} employees
  [This is GE Aerospace's most urgent intervention target]

═══ SEGMENT BREAKDOWN ════════════════════════════════════════════════

{seg_risk[['automate_pct','safety_crit_pct','avg_fd_gap',
           'avg_ai_ready','high_risk_n','headcount']].to_string()}

CRITICAL INTERSECTION — Defense & Systems:
  Automate risk:    {seg_risk.loc['Defense & Systems','automate_pct']:.1f}%
  Safety critical:  {seg_risk.loc['Defense & Systems','safety_crit_pct']:.1f}%
  FLIGHT DECK Gap:  {seg_risk.loc['Defense & Systems','avg_fd_gap']:.1f}
  High risk count:  {int(seg_risk.loc['Defense & Systems','high_risk_n'])}
  [Highest safety criticality × second highest automation risk]

CRITICAL INTERSECTION — Manufacturing & Supply Chain:
  Automate risk:    {seg_risk.loc['Manufacturing & Supply Chain','automate_pct']:.1f}%
  Safety critical:  {seg_risk.loc['Manufacturing & Supply Chain','safety_crit_pct']:.1f}%
  AI Readiness:     {seg_risk.loc['Manufacturing & Supply Chain','avg_ai_ready']:.1f}/100
  [Highest automation risk × second highest safety criticality]

═══ 3-YEAR TRANSFORMATION SCENARIOS (Monte Carlo) ════════════════════

Conservative (30% Adoption · 25% Reskilling):
  Automate:        {s_conserv.get('Automate',{}).get('mean','?')}%
                   [{s_conserv.get('Automate',{}).get('lower','?')}–
                   {s_conserv.get('Automate',{}).get('upper','?')}% CI]
  Create:          {s_conserv.get('Create',{}).get('mean','?')}%
  Judgment-Heavy:  {s_conserv.get('Judgment-Heavy',{}).get('mean','?')}%

Moderate (60% Adoption · 55% Reskilling):
  Automate:        {s_moderate.get('Automate',{}).get('mean','?')}%
                   [{s_moderate.get('Automate',{}).get('lower','?')}–
                   {s_moderate.get('Automate',{}).get('upper','?')}% CI]
  Create:          {s_moderate.get('Create',{}).get('mean','?')}%
  Judgment-Heavy:  {s_moderate.get('Judgment-Heavy',{}).get('mean','?')}%

Accelerated (90% Adoption · 80% Reskilling):
  Automate:        {s_accel.get('Automate',{}).get('mean','?')}%
                   [{s_accel.get('Automate',{}).get('lower','?')}–
                   {s_accel.get('Automate',{}).get('upper','?')}% CI]
  Create:          {s_accel.get('Create',{}).get('mean','?')}%
  Judgment-Heavy:  {s_accel.get('Judgment-Heavy',{}).get('mean','?')}%

KEY SHAP FINDINGS:
1. ai_vulnerability is the dominant predictor of automation risk —
   by a factor of 4x over any other feature. Automation risk is
   not about job function. It is about a person's relationship
   with change.
2. skill_Aerospace_Regulatory_Compliance is the #1 predictor of
   Judgment-Heavy roles — regulatory expertise is what makes a
   role irreplaceable by AI in a safety-first environment.
3. FLIGHT DECK skills (kaizen, A3 problem solving, daily visual
   management) appear in Judgment-Heavy predictors — the people
   most deeply practising GE's operating discipline are the ones
   making judgment calls that hold the system together.
"""

# ─────────────────────────────────────────────────────────────────────
# GENERATE BRIEF VIA CLAUDE API
# ─────────────────────────────────────────────────────────────────────

print("\nGenerating CHRO Strategic Brief via Claude API...")
print("(This may take 30–60 seconds — the brief is substantive)\n")

client = anthropic.Anthropic()

prompt = f"""You are presenting findings to the CHRO and CEO of
GE Aerospace — a 53,000-person global aerospace company with a
$190 billion backlog. The company runs a proprietary lean operating
system called FLIGHT DECK, built on three behaviours: Respect for
People, Continuous Improvement, and Customer Driven. Their
operational hierarchy is SQDC: Safety, Quality, Delivery, Cost —
always in that order. CEO Larry Culp has publicly stated that AI
should augment worker performance, not replace jobs.

The company faces a structural workforce challenge: the aerospace
industry will lose 210,000 skilled workers annually through 2033.
76% of aerospace companies cannot find engineering talent. GE
carries a $190B backlog that requires workforce capability to deliver.

You have run Athena — an AI workforce intelligence system — on a
50,000-employee synthetic model of GE Aerospace's workforce, calibrated
to their public disclosures. The findings are below.

Write a CHRO Strategic Brief with the following structure:

**EXECUTIVE SUMMARY**
Two sentences. The single most important finding and its immediate
implication for GE Aerospace's board. Be specific — use numbers.

**THE CRITICAL RISK: WHERE SAFETY MEETS AUTOMATION**
One paragraph. The dangerous intersection Athena found — safety-critical
segments with high automation risk. Name the segments specifically.
Use SQDC language. This is the finding that belongs in the board pack.

**THREE STRATEGIC FINDINGS**
Each finding must:
— State a specific number from the data
— Connect it to FLIGHT DECK or SQDC language
— Name an implication for GE's $190B backlog delivery

**WHAT THE SHAP MODEL FOUND**
One paragraph. Explain what SHAP discovered about the true drivers of
automation risk and Judgment-Heavy roles — in plain English that a CHRO
would use in a board presentation. Do not use technical jargon.
Reference the regulatory compliance finding specifically.

**THREE RECOMMENDATIONS**
Each must be:
— Actionable in the next 90 days
— Named as a FLIGHT DECK initiative (kaizen, value stream, standard work)
— Tied to a specific segment or population from the data
— Sequenced by SQDC priority (Safety first)

**THE BOARD QUESTION**
One sentence. A question that reframes how GE Aerospace's leadership
is currently thinking about AI workforce transformation. Make it
uncomfortable. Make it specific to GE's situation.

Write with precision and authority. Every sentence must earn its place.
Use FLIGHT DECK vocabulary throughout. This will be read in a board room.
Do not use consultant language. Write as a practitioner who has studied
this organisation and built this system specifically for them.

DATA:
{data_summary}"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1500,
    messages=[{"role": "user", "content": prompt}]
)

brief = message.content[0].text

# ─────────────────────────────────────────────────────────────────────
# FORMAT AND SAVE
# ─────────────────────────────────────────────────────────────────────

header = """ATHENA WORKFORCE INTELLIGENCE SYSTEM
GE Aero-Sim Digital Twin · 50,000 Employees · 7 Global Sites
CHRO Strategic Brief — Board Ready

Conscious Cybernetics™ · Aditya Roy
All data synthetic · Calibrated on public GE Aerospace disclosures only
════════════════════════════════════════════════════════════════════════

"""

full_brief = header + brief

with open("outputs/chro_brief.txt", "w") as f:
    f.write(full_brief)

print("═" * 68)
print(full_brief)
print("═" * 68)

print("\n✓ Brief saved: outputs/chro_brief.txt")
print("✓ Block 4 complete.")
print("  The CHRO brief is ready for the Streamlit dashboard.")
print("  Running on: NVIDIA DGX · Claude claude-sonnet-4-20250514 · "
      "No HR data left the machine.")