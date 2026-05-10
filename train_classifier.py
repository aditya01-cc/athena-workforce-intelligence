"""
ATHENA: AI Workforce Intelligence System
Block 2 — XGBoost Role Impact Classifier + SHAP Explainability

WHY XGBOOST:
Think of 300 HR managers each seeing a slightly different slice of the
workforce data, each correcting the mistakes of the previous group.
XGBoost automates that collective judgment process — fast, accurate,
and fully compatible with SHAP explainability.

WHY SHAP:
Every prediction is decomposed into which factors drove it and by how
much. No black box. Every verdict has visible reasoning — essential
for a company like GE Aerospace where every decision must be defensible.

Author: Aditya Roy · Conscious Cybernetics™
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────

print("Loading GE Aero-Sim dataset...")
df = pd.read_csv("ge_aerosim_workforce.csv")
print(f"  {len(df):,} employees · {len(df.columns)} columns loaded\n")

# ─────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# We give the model composite scores in addition to raw skills.
# This mirrors how an experienced HR leader thinks:
# not just individual skills but patterns across skill domains.
# ─────────────────────────────────────────────────────────────────────

skill_cols = [c for c in df.columns if c.startswith("skill_")]

# Encode categorical variables
# (XGBoost needs numbers, not text)
le_segment  = LabelEncoder()
le_site     = LabelEncoder()
le_region   = LabelEncoder()
le_country  = LabelEncoder()

df["segment_enc"] = le_segment.fit_transform(df["segment"])
df["site_enc"]    = le_site.fit_transform(df["site"])
df["region_enc"]  = le_region.fit_transform(df["region"])
df["country_enc"] = le_country.fit_transform(df["country"])

# SQDC composite — overall operational criticality score
df["sqdc_composite"] = (
    df["sqdc_safety"]   * 0.35 +
    df["sqdc_quality"]  * 0.30 +
    df["sqdc_delivery"] * 0.20 +
    df["sqdc_cost"]     * 0.15
)

# AI vulnerability index
# High tenure + low AI readiness + low learning velocity = vulnerable
df["ai_vulnerability"] = (
    (df["tenure_years"] / 38) * 0.35 +
    (1 - df["ai_readiness_score"] / 100) * 0.40 +
    (1 - df["learning_velocity"] / 100) * 0.25
)

# FLIGHT DECK alignment score
df["fd_alignment"] = 1 - (df["fd_readiness_gap"] / 100)

# Define feature columns for the model
feature_cols = (
    skill_cols +
    [
        "level",
        "tenure_years",
        "ai_readiness_score",
        "learning_velocity",
        "fd_readiness_gap",
        "sqdc_safety",
        "sqdc_quality",
        "sqdc_delivery",
        "sqdc_cost",
        "sqdc_composite",
        "avg_ai_skills",
        "avg_fd_skills",
        "avg_tech_skills",
        "avg_domain_skills",
        "ai_vulnerability",
        "fd_alignment",
        "segment_enc",
        "site_enc",
        "region_enc",
        "country_enc"
    ]
)

X = df[feature_cols]
y = df["ai_impact_label"]

# Encode target labels
le_target = LabelEncoder()
y_enc = le_target.fit_transform(y)

print(f"Target classes: {list(le_target.classes_)}")
print(f"Feature count:  {len(feature_cols)}\n")

# ─────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# 80% training, 20% testing — standard practice.
# Stratified means all four impact categories are represented
# proportionally in both sets.
# ─────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.20,
    random_state=42,
    stratify=y_enc
)

print(f"Training set:  {len(X_train):,} employees")
print(f"Test set:      {len(X_test):,} employees\n")

# ─────────────────────────────────────────────────────────────────────
# TRAIN XGBOOST MODEL
# GPU-accelerated on your DGX Spark.
# 300 trees, learning rate 0.05 — a careful, precise model.
# ─────────────────────────────────────────────────────────────────────

print("Training XGBoost classifier on NVIDIA DGX GPU...")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",   # GPU-optimised
    device="cuda",        # Use your DGX GPU
    verbosity=0
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ─────────────────────────────────────────────────────────────────────
# EVALUATE
# A good model here should exceed 80% accuracy.
# More important than overall accuracy: the Judgment-Heavy category
# should be correctly identified — those are your safety-critical roles.
# ─────────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✓ Training complete")
print(f"\n── Model Performance ───────────────────────────")
print(f"  Overall Accuracy: {accuracy:.1%}")
print(f"\n── Per-Category Performance ────────────────────")
print(classification_report(
    y_test, y_pred,
    target_names=le_target.classes_
))

# ─────────────────────────────────────────────────────────────────────
# SHAP EXPLAINABILITY
#
# WHY THIS MATTERS FOR GE:
# GE Aerospace runs on SQDC — Safety first, always.
# Any AI system that cannot explain its reasoning is a liability
# in a safety-critical environment. SHAP makes every prediction
# auditable. A safety engineer can look at any employee's automation
# risk score and see exactly which factors drove it.
#
# We sample 2,000 employees for SHAP — enough to be statistically
# representative of 50,000 without unnecessary compute.
# ─────────────────────────────────────────────────────────────────────

print("\nComputing SHAP explainability values...")
print("(Sampling 2,000 employees for representative analysis)")

sample_idx  = np.random.choice(len(X_test), 2000, replace=False)
X_sample    = X_test.iloc[sample_idx]

explainer = shap.TreeExplainer(model)
shap_values_raw = explainer.shap_values(X_sample)

# ── Handle multi-class SHAP format ───────────────────────────────────
# WHY THIS FIX:
# For multi-class models, newer SHAP returns a single 3D array
# (samples × features × classes) instead of a list of 2D arrays.
# We slice it correctly per class below.

if isinstance(shap_values_raw, list):
    # Older SHAP format — list of 2D arrays, one per class
    shap_by_class = shap_values_raw
else:
    # Newer SHAP format — single 3D array, slice by class index
    shap_by_class = [shap_values_raw[:, :, i]
                     for i in range(shap_values_raw.shape[2])]

automate_idx      = list(le_target.classes_).index("Automate")
jh_idx            = list(le_target.classes_).index("Judgment-Heavy")

# ── Chart 1: Automation Risk Drivers ─────────────────────────────────
plt.figure(figsize=(13, 9))
plt.style.use('dark_background')
shap.summary_plot(
    shap_by_class[automate_idx],
    X_sample,
    feature_names=feature_cols,
    plot_type="bar",
    show=False,
    max_display=18,
    color="#ff4d4d"
)
plt.title(
    "Athena · GE Aero-Sim: Top Drivers of Automation Risk\n"
    "SHAP Feature Importance · 50,000 Employee Digital Twin · "
    "Conscious Cybernetics™",
    fontsize=12, pad=20, color="#f0f2f5"
)
plt.gca().set_facecolor('#111418')
plt.gcf().set_facecolor('#0a0c0f')
plt.tight_layout()
plt.savefig(
    "outputs/shap_automate_risk.png",
    dpi=150, bbox_inches="tight", facecolor='#0a0c0f'
)
plt.close()
print("✓ SHAP chart saved: outputs/shap_automate_risk.png")

# ── Chart 2: Judgment-Heavy Drivers ──────────────────────────────────
plt.figure(figsize=(13, 9))
plt.style.use('dark_background')
shap.summary_plot(
    shap_by_class[jh_idx],
    X_sample,
    feature_names=feature_cols,
    plot_type="bar",
    show=False,
    max_display=18,
    color="#f0a500"
)
plt.title(
    "Athena · GE Aero-Sim: Predictors of Judgment-Heavy Roles\n"
    "SHAP Feature Importance · Roles Where Human Judgment Remains "
    "Irreplaceable · Conscious Cybernetics™",
    fontsize=12, pad=20, color="#f0f2f5"
)
plt.gca().set_facecolor('#111418')
plt.gcf().set_facecolor('#0a0c0f')
plt.tight_layout()
plt.savefig(
    "outputs/shap_judgment_heavy.png",
    dpi=150, bbox_inches="tight", facecolor='#0a0c0f'
)
plt.close()
print("✓ SHAP chart saved: outputs/shap_judgment_heavy.png")

# ─────────────────────────────────────────────────────────────────────
# SAVE ALL MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────

pickle.dump(model,        open("outputs/athena_classifier.pkl", "wb"))
pickle.dump(le_target,    open("outputs/label_encoder.pkl",     "wb"))
pickle.dump(le_segment,   open("outputs/segment_encoder.pkl",   "wb"))
pickle.dump(feature_cols, open("outputs/feature_cols.pkl",      "wb"))

print("\n── Saved Artifacts ─────────────────────────────")
print("  outputs/athena_classifier.pkl")
print("  outputs/label_encoder.pkl")
print("  outputs/segment_encoder.pkl")
print("  outputs/feature_cols.pkl")
print("  outputs/shap_automate_risk.png")
print("  outputs/shap_judgment_heavy.png")

print(f"\n✓ Block 2 complete.")
print(f"  Model accuracy: {accuracy:.1%}")
print(f"  SHAP explainability: active on all {len(X_sample):,} sampled predictions")
print(f"  Running on: NVIDIA DGX · Local inference · No data left the machine.")