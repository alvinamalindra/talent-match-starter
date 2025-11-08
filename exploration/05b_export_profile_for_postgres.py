import pandas as pd
from pathlib import Path

df = pd.read_csv("outputs/master_joined.csv")

# Pick important profile columns
profile_cols = [
    # ids & names
    "employee_id", "fullname", "nip",
    # org & education
    "company_id","area_id","position_id","department_id","division_id","directorate_id",
    "grade_id","education_id","major_id","years_of_service_months",
    # core psych
    "pauli","faxtor","disc","disc_word","mbti","iq","gtq","tiki",
]

dyn_cols = [c for c in df.columns if c.startswith("papi_") or c.startswith("strengths_")]

keep = [c for c in profile_cols if c in df.columns] + dyn_cols
out = df[keep].copy()

for c in [k for k in keep if k.endswith("_id")] + ["employee_id","nip"]:
    if c in out.columns:
        out[c] = out[c].astype("string")

Path("outputs").mkdir(exist_ok=True)
out.to_csv("outputs/employee_profile_pg.csv", index=False)
print("Saved to outputs/employee_profile_pg.csv")
print("Rows:", len(out), "| Cols:", len(out.columns))
