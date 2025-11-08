# tools/check_db.py
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root
url = os.getenv("SUPABASE_DB_URL")
print("ENV OK:", bool(url), "| URL starts with:", (url[:20] + "...") if url else None)

if not url:
    raise SystemExit("SUPABASE_DB_URL not found in .env")

engine = create_engine(url)
with engine.begin() as conn:
    ts = conn.execute(text("select now()")).fetchone()
    print("DB time:", ts[0])
    # sanity: check if view exists
    exists = conn.execute(text("""
        select to_regclass('public.final_score') is not null as view_exists
    """)).fetchone()[0]
    print("final_score view exists:", bool(exists))
print("✅ Connection test OK")