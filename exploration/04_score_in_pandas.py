import pandas as pd
from pathlib import Path

df = pd.read_csv("outputs/master_joined.csv")
if "is_top" not in df.columns or "employee_id" not in df.columns:
    raise SystemExit("Run 02_build_master.py first.")

# Psych
TVS_COGNITIVE = ["iq","gtq"]   # from profiles_psych

# Competency pillars (exact labels from dim_competency_pillars)
PILLARS = [
    "Growth Drive & Resilience",
    "Curiosity & Experimentation",
    "Insight & Decision Sharpness",
    "Quality Delivery Discipline",
    "Synergy & Team Orientation"
]

# Optionally split pillars into groups (TGVs)
TVS_STRATEGY = ["Insight & Decision Sharpness", "Curiosity & Experimentation"]
TVS_EXECUTION = ["Quality Delivery Discipline", "Growth Drive & Resilience"]
TVS_TEAMWORK  = ["Synergy & Team Orientation"]

# Cast TVs to numeric (ignore if missing)
for col in set(TVS_COGNITIVE + PILLARS):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

bench = df[df["is_top"] == 1].copy()
if bench.empty:
    raise SystemExit("No top performers (rating==5). Check performance data.")

# Compute benchmark means
bench_means = {}
for col in set(TVS_COGNITIVE + PILLARS):
    if col in df.columns:
        bench_means[col] = bench[col].mean()

def tv_match(a, b):
    import math
    if b is None or pd.isna(b) or b == 0 or a is None or pd.isna(a):
        return None
    return min(100.0, 100.0 * a / b)

def tgv(row, cols):
    vals = []
    for c in cols:
        if c in row and c in bench_means:
            vals.append(tv_match(row[c], bench_means[c]))
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None

out = df.copy()

# TGVs
out["tgv_cognitive"] = out.apply(lambda r: tgv(r, TVS_COGNITIVE), axis=1)
out["tgv_strategy"]  = out.apply(lambda r: tgv(r, TVS_STRATEGY), axis=1)
out["tgv_execution"] = out.apply(lambda r: tgv(r, TVS_EXECUTION), axis=1)
out["tgv_teamwork"]  = out.apply(lambda r: tgv(r, TVS_TEAMWORK), axis=1)

def avg_nonnull(values):
    vals = [v for v in values if v is not None]
    return sum(vals)/len(vals) if vals else None

out["final_match_rate"] = out.apply(
    lambda r: avg_nonnull([r["tgv_cognitive"], r["tgv_strategy"], r["tgv_execution"], r["tgv_teamwork"]]),
    axis=1
)

name_col = "fullname" if "fullname" in out.columns else None
cols = ["employee_id"] + ([name_col] if name_col else []) + [
    "final_match_rate","tgv_cognitive","tgv_strategy","tgv_execution","tgv_teamwork"
]

Path("outputs").mkdir(exist_ok=True)
out[cols].sort_values("final_match_rate", ascending=False).to_csv("outputs/match_scores.csv", index=False)
print("Saved to outputs/match_scores.csv")
