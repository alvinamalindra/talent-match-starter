# Talent Match Intelligence 

This repo made for Case Study

## Folders
- `exploration/` — notebooks / scripts for data exploration and joining tables
- `sql/` — final SQL to compute match rates
- `app/` — a tiny Streamlit app to display results
- `report/` — PDF Final / Deck
- `outputs/` — CSVs exported from your analysis 

## Quick Start (Python)
1. Create a virtual env and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Open `exploration/01_exploration.ipynb` and run the first two cells to load the Excel and inspect sheets.
3. Update `sql/01_match_scores.sql` to reflect your Success Formula and run it against your DB (or implement in pandas and export `outputs/match_scores.csv` as a stand‑in).
4. Run the app:
   ```bash
   streamlit run app/app.py
   ```

## Notes
- The Streamlit app expects a CSV at `outputs/match_scores.csv` with columns like:
  `employee_id, fullname, final_match_rate, tgv_leadership, tgv_cognitive, ...`
