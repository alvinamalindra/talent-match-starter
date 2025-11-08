import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("outputs/master_joined.csv")

for col in ["employee_id", "rating", "is_top"]:
    if col not in df.columns:
        raise SystemExit("Run 02_build_master.py first (needs employee_id, rating, is_top).")

# Exclude identifiers & *_id from numeric analysis
exclude = set(["employee_id","nip","is_top","rating","years_of_service_months"])
exclude |= {c for c in df.columns if c.endswith("_id")}

numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
# Keep columns with enough data
numeric_cols = [c for c in numeric_cols if df[c].notna().sum() >= max(10, int(0.2 * len(df)))]

rows = []
for col in numeric_cols:
    top = df.loc[df.is_top==1, col].mean()
    non = df.loc[df.is_top==0, col].mean()
    rows.append([col, top, non, (top - non)])

summary = pd.DataFrame(rows, columns=["metric","top_avg","non_top_avg","gap"]).sort_values("gap", ascending=False)
Path("outputs").mkdir(exist_ok=True)
summary.to_csv("outputs/top_vs_others_numeric_comparison.csv", index=False)
print("Saved -> outputs/top_vs_others_numeric_comparison.csv")

topN = summary.head(20)
plt.figure(figsize=(12,5))
plt.bar(topN["metric"], topN["gap"])
plt.xticks(rotation=60, ha="right")
plt.ylabel("Average gap (Top - Others)")
plt.title("Top Performers vs Others — Biggest Positive Gaps")
plt.tight_layout()
plt.savefig("outputs/pillar_gap.png", dpi=150)
print("Saved to outputs/pillar_gap.png") # Biar engga lupa nama folder
