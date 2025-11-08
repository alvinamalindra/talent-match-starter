import pandas as pd
from pathlib import Path

excel_path = Path('exploration/Study Case DA.xlsx')

xls = pd.ExcelFile(excel_path)
print('Sheets found:', xls.sheet_names)


dfs = {name: xls.parse(name) for name in xls.sheet_names}

for name, df in dfs.items():
    print(f'\n=== {name} ===')
    print(df.head(3))
    print('rows:', len(df), '| cols:', len(df.columns))
    print('missing ratio (top 10):')
    print(df.isna().mean().sort_values(ascending=False).head(10))

possible_perf_name = "performance_yearly"
possible_emp_name = "employees"

if (possible_perf_name in dfs) and (possible_emp_name in dfs):
    perf = dfs[possible_perf_name].copy()
    emp = dfs[possible_emp_name].copy()

    if "employee_id" in perf.columns and "year" in perf.columns:
        perf_sorted = perf.sort_values("year")
        perf_latest = perf_sorted.groupby("employee_id", as_index=False).tail(1)
    else:
        perf_latest = perf  # fallback if columns differ

    master_preview = perf_latest.merge(emp, on="employee_id", how="left")

    print("\n=== Master preview (latest perf + employees) ===")
    print(master_preview.head(5))
    print("Columns in master_preview:", master_preview.columns.tolist())

    master_preview.to_csv("outputs/master_preview.csv", index=False)

else:
    print("\n[!] Couldn't find expected sheets 'performance_yearly' and/or 'employees'.")
