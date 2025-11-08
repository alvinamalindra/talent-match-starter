
import pandas as pd
from pathlib import Path

excel = Path("exploration/Study Case DA.xlsx")

# Load exact sheets
employees            = pd.read_excel(excel, sheet_name="employees")
performance_yearly   = pd.read_excel(excel, sheet_name="performance_yearly")
profiles_psych       = pd.read_excel(excel, sheet_name="profiles_psych")
competencies_yearly  = pd.read_excel(excel, sheet_name="competencies_yearly")
dim_pillars          = pd.read_excel(excel, sheet_name="dim_competency_pillars")

try:
    papi_scores = pd.read_excel(excel, sheet_name="papi_scores")
except Exception:
    papi_scores = None

try:
    strengths = pd.read_excel(excel, sheet_name="strengths")
except Exception:
    strengths = None

# Latest performance per employee
perf_latest = (
    performance_yearly
    .sort_values(["employee_id","year"])
    .groupby("employee_id", as_index=False)
    .tail(1)
)

# Latest year competencies per employee, pivoted
comp_max_year = competencies_yearly.groupby("employee_id")["year"].max().reset_index(name="max_year")
comp_latest = competencies_yearly.merge(comp_max_year, on="employee_id", how="inner")
comp_latest = comp_latest[comp_latest["year"] == comp_latest["max_year"]].drop(columns=["max_year"])
comp_latest = comp_latest.merge(dim_pillars, on="pillar_code", how="left")

pivot_comp = (
    comp_latest
    .pivot_table(index="employee_id", columns="pillar_label", values="score", aggfunc="mean")
    .reset_index()
)

master = perf_latest.merge(
    employees.drop_duplicates(subset=["employee_id"]),
    on="employee_id", how="left"
)

master = master.merge(
    profiles_psych.drop_duplicates(subset=["employee_id"]),
    on="employee_id", how="left"
)

master = master.merge(pivot_comp, on="employee_id", how="left")

# Add PAPI
if papi_scores is not None and "employee_id" in papi_scores.columns:
    papi = papi_scores.copy()
    num_cols = papi.select_dtypes("number").columns.tolist()
    papi_agg = papi.groupby("employee_id")[num_cols].mean().reset_index()
    # prefix papi
    papi_agg = papi_agg.rename(columns={c: f"papi_{c}" for c in num_cols})
    master = master.merge(papi_agg, on="employee_id", how="left")

# Add strengths (aggregate; if numeric -> mean; if categorical -> counts) 
if strengths is not None and "employee_id" in strengths.columns:
    stg = strengths.copy()
    num_cols = stg.select_dtypes("number").columns.tolist()
    if num_cols:
        stg_agg = stg.groupby("employee_id")[num_cols].mean().reset_index()
    else:
        label_col = next((c for c in stg.columns if c.lower() in ("strength","theme","name","label")), None)
        if label_col:
            stg_agg = (
                stg.assign(val=1)
                   .pivot_table(index="employee_id", columns=label_col, values="val", aggfunc="sum", fill_value=0)
                   .reset_index()
            )
            # prefix strengths
            stg_agg = stg_agg.rename(columns={c: f"strengths_{c}" for c in stg_agg.columns if c != "employee_id"})
        else:
            # Just a count of rows per employee
            stg_agg = stg.groupby("employee_id").size().rename("strengths_count").reset_index()
    master = master.merge(stg_agg, on="employee_id", how="left")

# Business flags
master["is_top"] = (master["rating"] == 5).astype(int)

# Cast identifiers to string so they’re not treated as numeric metrics
id_like = [c for c in master.columns if c.endswith("_id")] + ["employee_id","nip"]
for c in id_like:
    if c in master.columns:
        master[c] = master[c].astype("string")

# Save (wide, enriched) 
Path("outputs").mkdir(exist_ok=True)
master.to_csv("outputs/master_joined.csv", index=False)
print("SAVED to outputs/master_joined.csv")
print("rows:", len(master), "| unique employee_id:", master["employee_id"].nunique())
