"""
ATHENA: AI Workforce Intelligence System
GE Aero-Sim Digital Twin — 50,000 Employee Dataset
Calibrated to publicly available GE Aerospace workforce disclosures.

Author: Aditya Roy · Conscious Cybernetics™
Note: All data is synthetic. Calibrated on public sources, not proprietary data.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────────────────────────────
# GE AEROSPACE SEGMENT STRUCTURE
# Calibrated from: GE Aerospace 2025 Annual Report,
# investor presentations, and public workforce disclosures.
# Total synthetic headcount: 50,000
# ─────────────────────────────────────────────────────────────────────

SEGMENTS = {
    "Commercial Engines & Services": {
        "sub_functions": [
            "Propulsion Engineering",
            "MRO & Field Services",
            "Customer Technical Support",
            "Fleet Management & Analytics",
            "Engine Programme Management"
        ],
        "weight": 0.38,           # largest segment by headcount
        "avg_tenure_years": 11.2, # experienced workforce
        "ai_readiness_base": 48,  # moderate AI fluency
        "sqdc_profile": {         # how critical to each SQDC pillar
            "safety": 0.90,
            "quality": 0.85,
            "delivery": 0.80,
            "cost": 0.65
        }
    },
    "Defense & Systems": {
        "sub_functions": [
            "Military Propulsion",
            "Avionics & Electronic Systems",
            "Advanced Development Programs",
            "Government & Export Compliance",
            "Classified Programs"
        ],
        "weight": 0.22,
        "avg_tenure_years": 13.5, # highest tenure — security clearances
        "ai_readiness_base": 42,  # lower — classified environment constraints
        "sqdc_profile": {
            "safety": 0.95,
            "quality": 0.90,
            "delivery": 0.85,
            "cost": 0.55
        }
    },
    "Engineering & Technology": {
        "sub_functions": [
            "Core Engine Design",
            "Digital Engineering & Simulation",
            "Additive Manufacturing",
            "Open Fan / RISE Programme",
            "AI & Advanced Analytics",
            "Materials & Composites"
        ],
        "weight": 0.25,
        "avg_tenure_years": 9.8,
        "ai_readiness_base": 68,  # highest AI fluency
        "sqdc_profile": {
            "safety": 0.80,
            "quality": 0.90,
            "delivery": 0.70,
            "cost": 0.60
        }
    },
    "Manufacturing & Supply Chain": {
        "sub_functions": [
            "Production Operations",
            "Quality Assurance & Safety",
            "Supply Chain Management",
            "Ceramic Matrix Composites",
            "Maintenance, Repair & Overhaul"
        ],
        "weight": 0.10,
        "avg_tenure_years": 14.1, # longest tenure — skilled trades
        "ai_readiness_base": 32,  # lowest — hands-on workforce
        "sqdc_profile": {
            "safety": 0.95,
            "quality": 0.95,
            "delivery": 0.90,
            "cost": 0.80
        }
    },
    "Business Operations": {
        "sub_functions": [
            "Finance & FP&A",
            "Human Resources",
            "Legal & Compliance",
            "Commercial & Contracts",
            "Transformation Office",
            "Communications & Brand"
        ],
        "weight": 0.05,
        "avg_tenure_years": 8.4,
        "ai_readiness_base": 55,
        "sqdc_profile": {
            "safety": 0.40,
            "quality": 0.65,
            "delivery": 0.70,
            "cost": 0.85
        }
    }
}

# ─────────────────────────────────────────────────────────────────────
# GE AEROSPACE GLOBAL SITES
# Calibrated from public GE Aerospace site disclosures
# ─────────────────────────────────────────────────────────────────────

SITES = {
    "Evendale, OH": {
        "region": "Americas",
        "weight": 0.28,    # HQ + largest manufacturing site
        "timezone": "EST",
        "country": "USA",
        "labor_market": "competitive",
        "primary_segments": [
            "Commercial Engines & Services",
            "Engineering & Technology"
        ]
    },
    "Lynn, MA": {
        "region": "Americas",
        "weight": 0.18,
        "timezone": "EST",
        "country": "USA",
        "labor_market": "very_competitive",
        "primary_segments": [
            "Defense & Systems"
        ]
    },
    "Asheville, NC": {
        "region": "Americas",
        "weight": 0.14,
        "timezone": "EST",
        "country": "USA",
        "labor_market": "moderate",
        "primary_segments": [
            "Manufacturing & Supply Chain",
            "Commercial Engines & Services"
        ]
    },
    "Grand Rapids, MI": {
        "region": "Americas",
        "weight": 0.12,
        "timezone": "EST",
        "country": "USA",
        "labor_market": "moderate",
        "primary_segments": [
            "Defense & Systems",
            "Engineering & Technology"
        ]
    },
    "Avio Aero, Italy": {
        "region": "Europe",
        "weight": 0.14,
        "timezone": "CET",
        "country": "Italy",
        "labor_market": "specialised",
        "primary_segments": [
            "Commercial Engines & Services",
            "Engineering & Technology"
        ]
    },
    "Rzeszow, Poland": {
        "region": "Europe",
        "weight": 0.08,
        "timezone": "CET",
        "country": "Poland",
        "labor_market": "growing",
        "primary_segments": [
            "Manufacturing & Supply Chain",
            "Defense & Systems"
        ]
    },
    "Singapore": {
        "region": "Asia Pacific",
        "weight": 0.06,
        "timezone": "SGT",
        "country": "Singapore",
        "labor_market": "very_competitive",
        "primary_segments": [
            "Commercial Engines & Services",
            "Business Operations"
        ]
    }
}

# ─────────────────────────────────────────────────────────────────────
# CAREER LEVELS
# Aligned to GE Aerospace's known band structure
# ─────────────────────────────────────────────────────────────────────

LEVELS = {
    1: {
        "label": "Entry / Graduate",
        "salary_range_usd": (52000, 78000),
        "weight": 0.18
    },
    2: {
        "label": "Specialist",
        "salary_range_usd": (72000, 105000),
        "weight": 0.24
    },
    3: {
        "label": "Senior Specialist",
        "salary_range_usd": (98000, 142000),
        "weight": 0.24
    },
    4: {
        "label": "Lead / Principal",
        "salary_range_usd": (132000, 182000),
        "weight": 0.16
    },
    5: {
        "label": "Manager",
        "salary_range_usd": (158000, 218000),
        "weight": 0.09
    },
    6: {
        "label": "Senior Manager",
        "salary_range_usd": (198000, 275000),
        "weight": 0.05
    },
    7: {
        "label": "Director",
        "salary_range_usd": (255000, 355000),
        "weight": 0.03
    },
    8: {
        "label": "VP / Executive",
        "salary_range_usd": (315000, 600000),
        "weight": 0.01
    }
}

# ─────────────────────────────────────────────────────────────────────
# 55-SKILL TAXONOMY
# Organised in 5 domains.
# Domain mix is calibrated per segment in generate_skills()
# ─────────────────────────────────────────────────────────────────────

SKILLS = {
    "Technical_AI": [
        "Python_Proficiency",
        "ML_Frameworks",
        "Data_Analysis_Tools",
        "AI_Tools_Usage",
        "Prompt_Engineering",
        "Statistical_Modeling",
        "Data_Visualization",
        "Cloud_Platforms",
        "API_Integration",
        "Digital_Twin_Literacy"
    ],
    "Domain_Aerospace": [
        "Propulsion_Systems_Knowledge",
        "Aerospace_Regulatory_Compliance",
        "Quality_Management_Systems",
        "Process_Design_Engineering",
        "Defence_Systems_Awareness",
        "Additive_Manufacturing",
        "Structural_Analysis",
        "Avionics_Systems",
        "MRO_Operations",
        "Supply_Chain_Engineering",
        "Ceramic_Matrix_Composites",
        "FMEA_Risk_Analysis"
    ],
    "FLIGHT_DECK_Readiness": [
        "Standard_Work_Adherence",
        "Daily_Visual_Management",
        "Kaizen_Participation",
        "Value_Stream_Thinking",
        "SQDC_Literacy",
        "Problem_Solving_A3",
        "Continuous_Improvement_Mindset",
        "Operating_Cadence_Discipline",
        "Action_Planning_Rigor",
        "Lean_Tool_Proficiency",
        "Gemba_Walk_Practice"
    ],
    "Leadership_Org": [
        "Strategic_Thinking",
        "Change_Leadership",
        "Stakeholder_Management",
        "Executive_Communication",
        "Team_Development",
        "CrossFunctional_Collaboration",
        "Decision_Making_Under_Uncertainty",
        "Conflict_Resolution",
        "Influence_Without_Authority",
        "Project_Management",
        "Global_Team_Leadership"
    ],
    "AI_Readiness": [
        "AI_Literacy",
        "Human_AI_Collaboration",
        "Prompt_Design",
        "AI_Tool_Adoption",
        "Data_Interpretation",
        "Algorithmic_Thinking",
        "AI_Ethics_Awareness",
        "Workflow_Redesign_for_AI",
        "AI_Output_Validation",
        "AI_Risk_Assessment",
        "AI_Governance_Awareness"
    ]
}

ALL_SKILLS = [s for domain in SKILLS.values() for s in domain]


# ─────────────────────────────────────────────────────────────────────
# SKILL GENERATION
# Each employee gets a skill score 0-5 per skill.
# Scores are shaped by: segment, level, site, AI readiness base
# ─────────────────────────────────────────────────────────────────────

def generate_skills(segment, level, ai_readiness, site):
    """
    Generate a realistic 55-skill profile for one employee.

    WHY THIS MATTERS:
    In real HR, skills are not uniformly distributed. A propulsion engineer
    in Evendale has deep aerospace domain knowledge but may have low AI
    literacy. A Digital Engineering specialist has the reverse profile.
    This function replicates that reality through domain-specific boosts.
    """

    segment_boosts = {
        "Commercial Engines & Services": {
            "Technical_AI": 1.1,
            "Domain_Aerospace": 1.5,
            "FLIGHT_DECK_Readiness": 1.3,
            "Leadership_Org": 1.1,
            "AI_Readiness": 1.0
        },
        "Defense & Systems": {
            "Technical_AI": 1.0,
            "Domain_Aerospace": 1.6,
            "FLIGHT_DECK_Readiness": 1.2,
            "Leadership_Org": 1.2,
            "AI_Readiness": 0.8  # constrained by clearance environment
        },
        "Engineering & Technology": {
            "Technical_AI": 1.5,
            "Domain_Aerospace": 1.4,
            "FLIGHT_DECK_Readiness": 1.1,
            "Leadership_Org": 1.0,
            "AI_Readiness": 1.5
        },
        "Manufacturing & Supply Chain": {
            "Technical_AI": 0.7,
            "Domain_Aerospace": 1.4,
            "FLIGHT_DECK_Readiness": 1.6, # highest FLIGHT DECK mastery
            "Leadership_Org": 1.0,
            "AI_Readiness": 0.7
        },
        "Business Operations": {
            "Technical_AI": 1.1,
            "Domain_Aerospace": 0.8,
            "FLIGHT_DECK_Readiness": 1.1,
            "Leadership_Org": 1.4,
            "AI_Readiness": 1.2
        }
    }

    # Site modifiers — European sites have slightly different skill profiles
    site_ai_modifier = {
        "Evendale, OH": 1.0,
        "Lynn, MA": 0.9,
        "Asheville, NC": 0.85,
        "Grand Rapids, MI": 0.9,
        "Avio Aero, Italy": 0.95,
        "Rzeszow, Poland": 0.88,
        "Singapore": 1.05
    }

    boosts = segment_boosts.get(segment, {})
    level_multiplier = 0.55 + (level * 0.09)
    site_modifier = site_ai_modifier.get(site, 1.0)

    skills = {}
    for domain, skill_list in SKILLS.items():
        domain_boost = boosts.get(domain, 1.0)
        for skill in skill_list:
            base = np.random.beta(2, 3) * 5
            adjusted = base * domain_boost * level_multiplier

            # AI Readiness skills further shaped by site and base score
            if domain in ["AI_Readiness", "Technical_AI"]:
                adjusted *= (0.6 + (ai_readiness / 100) * 0.8) * site_modifier

            # FLIGHT DECK skills peak in Manufacturing
            if domain == "FLIGHT_DECK_Readiness":
                adjusted *= (0.7 + min(level, 5) * 0.06)

            skills[skill] = min(5.0, max(0.0, round(
                adjusted + np.random.normal(0, 0.15), 2
            )))

    return skills


# ─────────────────────────────────────────────────────────────────────
# AI IMPACT LABEL LOGIC
# Four categories — not three.
# Judgment-Heavy is the addition that aligns to FLIGHT DECK philosophy.
#
# WHY FOUR CATEGORIES:
# Standard frameworks use Automate / Augment / Create.
# GE Aerospace's FLIGHT DECK builds on human judgment at every level.
# Safety decisions, regulatory calls, design innovation, ethical
# trade-offs in military programs — these cannot be automated or
# simply augmented. They require human judgment as the primary input,
# with AI as a supporting layer only. Judgment-Heavy captures this.
# ─────────────────────────────────────────────────────────────────────

def assign_impact_label(segment, level, ai_readiness,
                         learning_velocity, tenure, skills):

    fd_score = np.mean([skills.get(s, 2.5)
                        for s in SKILLS["FLIGHT_DECK_Readiness"]])
    ai_score = np.mean([skills.get(s, 2.5)
                        for s in SKILLS["AI_Readiness"]])
    tech_score = np.mean([skills.get(s, 2.5)
                          for s in SKILLS["Technical_AI"]])

    # Judgment-Heavy: senior roles in safety-critical or classified segments
    # Logic: high level + safety-critical segment + high domain expertise
    if (level >= 5 and
        segment in ["Defense & Systems",
                    "Commercial Engines & Services",
                    "Manufacturing & Supply Chain"] and
        skills.get("FMEA_Risk_Assessment", 0) > 3.0 or
        skills.get("Aerospace_Regulatory_Compliance", 0) > 3.5):
        if np.random.random() < 0.45:
            return "Judgment-Heavy"

    # Automate: low AI readiness + long tenure + low learning velocity
    automate_score = (
        (1 - ai_readiness / 100) * 38 +
        (1 - ai_score / 5)       * 28 +
        min(tenure, 20) / 20     * 18 +
        (1 - learning_velocity / 100) * 16 +
        np.random.normal(0, 6)
    )

    # Create: high AI readiness + high tech skills + high level
    create_score = (
        (ai_readiness / 100)    * 33 +
        (tech_score / 5)        * 25 +
        (level / 8)             * 24 +
        (learning_velocity / 100) * 18 +
        np.random.normal(0, 6)
    )

    if automate_score > 60:
        return "Automate"
    elif create_score > 57:
        return "Create"
    else:
        return "Augment"


# ─────────────────────────────────────────────────────────────────────
# FLIGHT DECK READINESS GAP
# Renamed from "transformation_risk" to align with GE vocabulary.
#
# WHY THIS NAME:
# GE doesn't think about risk the way a consultant does.
# They think about gaps relative to standard work.
# A high FLIGHT DECK Readiness Gap means the employee has not yet
# internalised the operating discipline GE requires — especially
# as AI tools get embedded into FLIGHT DECK cadences.
# ─────────────────────────────────────────────────────────────────────

def calculate_fd_readiness_gap(ai_readiness, learning_velocity,
                                tenure, skills):
    fd_skill_mean = np.mean([skills.get(s, 2.5)
                             for s in SKILLS["FLIGHT_DECK_Readiness"]])
    gap = (
        (1 - ai_readiness / 100) * 38 +
        (1 - learning_velocity / 100) * 28 +
        min(tenure, 25) / 25     * 18 +
        (1 - fd_skill_mean / 5)  * 16 +
        np.random.normal(0, 4)
    )
    return min(100, max(0, round(gap, 1)))


# ─────────────────────────────────────────────────────────────────────
# MAIN DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────

def generate_dataset(n=50000):
    print(f"Generating GE Aero-Sim Digital Twin ({n:,} employees)...")
    print("Calibrated on publicly available GE Aerospace workforce data.\n")

    records = []

    # Build weighted pools
    segment_pool = []
    for seg, cfg in SEGMENTS.items():
        segment_pool.extend([seg] * int(n * cfg["weight"]))
    while len(segment_pool) < n:
        segment_pool.append("Commercial Engines & Services")
    random.shuffle(segment_pool)

    site_pool = []
    for site, cfg in SITES.items():
        site_pool.extend([site] * int(n * cfg["weight"]))
    while len(site_pool) < n:
        site_pool.append("Evendale, OH")
    random.shuffle(site_pool)

    level_pool = []
    for level, cfg in LEVELS.items():
        level_pool.extend([level] * int(n * cfg["weight"]))
    while len(level_pool) < n:
        level_pool.append(2)
    random.shuffle(level_pool)

    for i in range(n):
        segment    = segment_pool[i]
        site       = site_pool[i]
        level      = level_pool[i]
        seg_cfg    = SEGMENTS[segment]
        site_cfg   = SITES[site]
        level_cfg  = LEVELS[level]

        # Sub-function
        sub_fn = random.choice(seg_cfg["sub_functions"])

        # Tenure — shaped by segment (Manufacturing longest, Ops shortest)
        avg_tenure = seg_cfg["avg_tenure_years"]
        tenure = min(38, max(0.5,
            np.random.gamma(shape=2.2, scale=avg_tenure / 2.2)
        ))

        # Salary — adjusted for site (Italian and Polish sites lower USD eq.)
        country_adj = {
            "USA": 1.00,
            "Italy": 0.78,
            "Poland": 0.62,
            "Singapore": 0.90
        }
        s_min, s_max = level_cfg["salary_range_usd"]
        adj = country_adj.get(site_cfg["country"], 1.0)
        salary = int((np.random.uniform(s_min, s_max) +
                      min(tenure * 750, 22000)) * adj)

        # AI Readiness — shaped by segment base + level + tenure penalty
        ai_base    = seg_cfg["ai_readiness_base"]
        level_boost = level * 4
        tenure_pen  = max(0, (tenure - 12) * 1.2)
        ai_readiness = max(5, min(98,
            np.random.normal(ai_base + level_boost - tenure_pen, 15)
        ))

        # Learning velocity — younger employees and digital segments higher
        lv_base = 65 - (tenure * 0.8) + (level * 2)
        if segment == "Engineering & Technology":
            lv_base += 10
        learning_velocity = max(5, min(98,
            np.random.normal(lv_base, 16)
        ))

        # Generate 55-skill profile
        skills = generate_skills(segment, level, ai_readiness, site)

        # Composite scores for reporting
        ai_skill_mean   = np.mean([skills[s]
                                   for s in SKILLS["AI_Readiness"]])
        fd_skill_mean   = np.mean([skills[s]
                                   for s in SKILLS["FLIGHT_DECK_Readiness"]])
        tech_skill_mean = np.mean([skills[s]
                                   for s in SKILLS["Technical_AI"]])
        domain_skill_mean = np.mean([skills[s]
                                     for s in SKILLS["Domain_Aerospace"]])

        # SQDC contribution scores (segment-calibrated)
        sqdc = seg_cfg["sqdc_profile"]
        sqdc_safety   = round(min(1.0, sqdc["safety"]   + np.random.normal(0, 0.05)), 2)
        sqdc_quality  = round(min(1.0, sqdc["quality"]  + np.random.normal(0, 0.05)), 2)
        sqdc_delivery = round(min(1.0, sqdc["delivery"] + np.random.normal(0, 0.05)), 2)
        sqdc_cost     = round(min(1.0, sqdc["cost"]     + np.random.normal(0, 0.05)), 2)

        # AI Impact Label (4 categories)
        ai_impact = assign_impact_label(
            segment, level, ai_readiness,
            learning_velocity, tenure, skills
        )

        # FLIGHT DECK Readiness Gap
        fd_gap = calculate_fd_readiness_gap(
            ai_readiness, learning_velocity, tenure, skills
        )

        record = {
            # Identity
            "employee_id":           f"GEA-{100000 + i}",
            "segment":                segment,
            "sub_function":           sub_fn,
            "site":                   site,
            "region":                 site_cfg["region"],
            "country":                site_cfg["country"],
            "timezone":               site_cfg["timezone"],
            "level":                  level,
            "level_label":            level_cfg["label"],

            # Employment
            "tenure_years":           round(tenure, 1),
            "annual_salary_usd":      salary,

            # AI & Learning
            "ai_readiness_score":     round(ai_readiness, 1),
            "learning_velocity":      round(learning_velocity, 1),

            # SQDC
            "sqdc_safety":            sqdc_safety,
            "sqdc_quality":           sqdc_quality,
            "sqdc_delivery":          sqdc_delivery,
            "sqdc_cost":              sqdc_cost,

            # Composites
            "avg_ai_skills":          round(ai_skill_mean, 2),
            "avg_fd_skills":          round(fd_skill_mean, 2),
            "avg_tech_skills":        round(tech_skill_mean, 2),
            "avg_domain_skills":      round(domain_skill_mean, 2),

            # Output labels
            "ai_impact_label":        ai_impact,
            "fd_readiness_gap":       fd_gap,

            # All 55 skills
            **{f"skill_{k}": v for k, v in skills.items()}
        }
        records.append(record)

        # Progress indicator every 10,000 rows
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1:,} employees generated...")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_dataset(50000)

    output_file = "ge_aerosim_workforce.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✓ Dataset saved: {output_file}")
    print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")

    print(f"\n── AI Impact Distribution ──────────────────────")
    print(df["ai_impact_label"].value_counts().to_string())

    print(f"\n── Segment Distribution ────────────────────────")
    print(df["segment"].value_counts().to_string())

    print(f"\n── Site Distribution ───────────────────────────")
    print(df["site"].value_counts().to_string())

    print(f"\n── Key Metrics ─────────────────────────────────")
    print(f"  Avg AI Readiness:       {df['ai_readiness_score'].mean():.1f}/100")
    print(f"  Avg FLIGHT DECK Gap:    {df['fd_readiness_gap'].mean():.1f}/100")
    print(f"  High FD Gap (>70):      {(df['fd_readiness_gap'] > 70).sum():,} employees")
    print(f"  Judgment-Heavy roles:   {(df['ai_impact_label'] == 'Judgment-Heavy').sum():,}")
    print(f"  Automate risk:          {(df['ai_impact_label'] == 'Automate').sum():,}")

    print(f"\n── SQDC Safety-Critical + Automate Risk ────────")
    safety_auto = df[
        (df["ai_impact_label"] == "Automate") &
        (df["sqdc_safety"] > 0.85)
    ]
    print(f"  Safety-critical roles at automation risk: "
          f"{len(safety_auto):,}")
    print(f"  This is GE Aerospace's highest-priority intervention.")

    print(f"\n✓ GE Aero-Sim Digital Twin generation complete.")
    print(f"  All data is synthetic. Calibrated on public disclosures only.")