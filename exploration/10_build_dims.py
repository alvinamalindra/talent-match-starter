import pandas as pd
from pathlib import Path

excel_candidates = [
    Path("exploration/Study Case DA.xlsx"),
    Path("Study Case DA.xlsx"),
]
dim_sheet_names = [
    "dim_companies",
    "dim_divisions",
    "dim_departments",
    "dim_directorates",
    "dim_grades",
    "dim_positions",
    "dim_areas",
    "dim_education",
    "dim_majors",
]
outdir = Path("outputs/dims")
outdir.mkdir(parents=True, exist_ok=True)

def find_excel(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("Study Case DA.xlsx not found in expected locations.")


xlsx_path = find_excel(excel_candidates)
xls = pd.ExcelFile(xlsx_path)
print("Sheets found:", xls.sheet_names)


exported = []
skipped = []

for sheet in dim_sheet_names:
    if sheet not in xls.sheet_names:
        skipped.append(sheet)
        continue

    df = xls.parse(sheet)  
    out_csv = outdir / f"{sheet}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    exported.append((sheet, len(df), list(df.columns)))
    print(f"[OK] {sheet} -> {out_csv} | rows={len(df)}")


print("\n=== Summary ===")
if exported:
    for s, n, cols in exported:
        print(f"{s}: {n} rows | cols={cols}")
else:
    print("No dim_* sheets were exported.")

if skipped:
    print("\nSkipped (not present in workbook):", ", ".join(skipped))
