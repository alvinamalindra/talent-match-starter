# Export clean master data for Supabase (Postgres)

import pandas as pd
from pathlib import Path

df = pd.read_csv("outputs/master_joined.csv")

colmap = {
    "employee_id": "employee_id",
    "fullname": "fullname",
    "rating": "rating",
    "iq": "iq",
    "gtq": "gtq",
    "Growth Drive & Resilience": "growth_drive_and_resilience",
    "Curiosity & Experimentation": "curiosity_and_experimentation",
    "Insight & Decision Sharpness": "insight_and_decision_sharpness",
    "Quality Delivery Discipline": "quality_delivery_discipline",
    "Synergy & Team Orientation": "synergy_and_team_orientation",
}

# Keep only the mapped columns that exist
out = pd.DataFrame({dst: df[src] for src, dst in colmap.items() if src in df.columns})

Path("outputs").mkdir(exist_ok=True)
out.to_csv("outputs/employee_master_pg.csv", index=False)
print("Saved to outputs/employee_master_pg.csv")
print("Rows:", len(out))
