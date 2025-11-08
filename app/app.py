import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


load_dotenv()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
if not SUPABASE_DB_URL:
    st.error("SUPABASE_DB_URL is missing.")
    st.stop()

engine = create_engine(SUPABASE_DB_URL)

@st.cache_data(show_spinner=True)
def fetch_data():
    """Load the final view and coerce numerics."""
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM final_score_enriched;"), conn)
    # drop duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    numeric_cols = [
        "rating","final_match_rate",
        "tgv_cognitive","tgv_strategy","tgv_execution","tgv_teamwork",
        "profile_iq","profile_gtq","years_of_service_months"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = fetch_data()

st.title("Talent Match Dashboard")

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Total Employees", f"{len(df):,}")
with k2:
    st.metric("Avg Performance Rating", f"{df['rating'].mean():.2f}" if 'rating' in df else "—")
with k3:
    st.metric("Median Match Score", f"{df['final_match_rate'].median():.1f}%" if 'final_match_rate' in df else "—")


st.sidebar.header("Filters")
DISPLAY_FILTERS = [
    "company_name", "directorate_name", "division_name", "department_name",
    "position_name", "grade_name", "area_name", "education_name", "major_name",
]
filtered = df.copy()
for col in DISPLAY_FILTERS:
    if col in filtered.columns:
        opts = sorted(filtered[col].dropna().astype(str).unique())
        picks = st.sidebar.multiselect(col.replace("_", " ").title(), opts)
        if picks:
            filtered = filtered[filtered[col].astype(str).isin(picks)]

st.sidebar.markdown("---")
st.sidebar.caption("Tip: filters affect all sections below, including Top-5 and Benchmark.")

st.subheader("Top Segments by Headcount (with quality)")

def top_table(dfin: pd.DataFrame, group_col: str, title: str):
    if group_col not in dfin.columns:
        return
    agg = (
        dfin.groupby(group_col, dropna=True)
            .agg(
                Headcount=("employee_id", "count"),
                Avg_Rating=("rating", "mean"),
                Avg_Match=("final_match_rate", "mean")
            )
            .reset_index()
            .sort_values(["Headcount","Avg_Match"], ascending=[False, False])
            .head(5)
    )
    for c in ["Avg_Rating","Avg_Match"]:
        if c in agg.columns:
            agg[c] = agg[c].round(2 if c=="Avg_Rating" else 1)
    st.markdown(f"**{title}**")
    st.dataframe(
        agg.rename(columns={group_col: title[:-1] if title.endswith('s') else title}),
        use_container_width=True, hide_index=True
    )

c1, c2 = st.columns(2)
with c1:
    top_table(filtered, "division_name", "Divisions")
with c2:
    top_table(filtered, "department_name", "Departments")
top_table(filtered, "education_name", "Educations")

st.markdown("---")

st.subheader("Success Pattern Discovery")

num_cols = [c for c in [
    "rating","final_match_rate","tgv_cognitive","tgv_strategy","tgv_execution","tgv_teamwork",
    "profile_iq","profile_gtq","years_of_service_months"
] if c in filtered.columns]

corr_df = filtered[num_cols].corr().round(2) if num_cols else pd.DataFrame()
st.markdown("**Correlations with Performance Rating** (larger absolute value = stronger relationship):")
if not corr_df.empty:
    show = corr_df.loc[:, ["rating"]] if "rating" in corr_df.columns else corr_df
    st.dataframe(show.sort_values(by="rating", ascending=False), use_container_width=True)
else:
    st.info("No numeric columns available for correlation.")

# Top vs others gaps
delta_df = pd.DataFrame()
if "rating" in filtered.columns:
    top_mask = filtered["rating"] >= 5
    groups = []
    for col in ["tgv_cognitive","tgv_strategy","tgv_execution","tgv_teamwork","final_match_rate","profile_iq","profile_gtq"]:
        if col in filtered.columns:
            top_mean = filtered.loc[top_mask, col].mean()
            oth_mean = filtered.loc[~top_mask, col].mean()
            if pd.notna(top_mean) and pd.notna(oth_mean):
                groups.append({
                    "Metric": col, "Top5 Avg": round(top_mean,2),
                    "Others Avg": round(oth_mean,2),
                    "Delta(Top-Others)": round(top_mean - oth_mean, 2)
                })
    if groups:
        delta_df = pd.DataFrame(groups).sort_values("Delta(Top-Others)", ascending=False)
        st.markdown("**Top Performers vs Others — biggest gaps**")
        st.dataframe(delta_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Benchmark & Re-rank Candidates
st.subheader("Build a Benchmark & Re-rank Candidates")

pillar_candidates = ["tgv_cognitive","tgv_strategy","tgv_execution","tgv_teamwork"]
feat_cols = [c for c in pillar_candidates if c in df.columns]
if not feat_cols:
    guess = [c for c in df.columns if any(k in c for k in ["cognitive","strategy","execution","teamwork"])]
    feat_cols = guess[:4]

# Prepare labels and a string employee_id to match safely
filtered = filtered.copy()
filtered["employee_id_str"] = filtered["employee_id"].astype(str)
filtered["label_for_pick"] = filtered["fullname"].fillna("Unknown") + " (" + filtered["employee_id_str"] + ")"

bench_col1, bench_col2 = st.columns([2,1])
with bench_col1:
    picks = st.multiselect(
        "Select benchmark employees (exemplars)",
        options=filtered["label_for_pick"].tolist(),
        default=[]
    )
with bench_col2:
    show_k = st.slider("Top N to show", 5, 50, 20, 5)

def parse_id(label: str) -> str | None:
    """Accept alphanumeric IDs inside the last parentheses at the end."""
    m = re.search(r"\(([^)]+)\)\s*$", str(label))
    return m.group(1).strip() if m else None

@st.cache_data(show_spinner=False)
def compute_benchmark_and_scores_anyid(data: pd.DataFrame, labels: list[str], feat_cols: list[str]):
    import numpy as np
    if not labels or not feat_cols:
        return None, data.assign(benchmark_score=np.nan), 0, feat_cols

    ids = [parse_id(x) for x in labels if x]
    ids = [i for i in ids if i]
    bench_df = data[data["employee_id_str"].isin(ids)]

    if bench_df.empty:
        return None, data.assign(benchmark_score=np.nan), 0, feat_cols

    bench_vec = bench_df[feat_cols].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True)

    def cos_sim(row):
        a = row[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        b = bench_vec.to_numpy(dtype=float)
        if np.isnan(a).any() or np.isnan(b).any():
            return np.nan
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return np.nan
        return float(np.dot(a, b) / (na * nb))

    scored = data.copy()
    scored["benchmark_score"] = scored.apply(cos_sim, axis=1)
    return bench_vec, scored, len(bench_df), feat_cols

bench_vec, scored, matched_count, used_feats = compute_benchmark_and_scores_anyid(filtered, picks, feat_cols)

# Diagnostics
if picks:
    st.caption(f"Selected: {len(picks)} • Matched in data: {matched_count} • Features used: {', '.join(used_feats) if used_feats else '—'}")

c3, c4 = st.columns([1,1])
with c3:
    if picks:
        st.markdown("**Selected exemplars**")
        st.write(", ".join([p.split("(")[0].strip() for p in picks]))
    if bench_vec is not None:
        st.markdown("**Benchmark profile (avg of exemplars)**")
        st.dataframe(bench_vec.to_frame("Value").round(2), use_container_width=True)
    else:
        st.info("Pick valid exemplars to see the profile.")

with c4:
    if "benchmark_score" in scored.columns and not scored["benchmark_score"].isna().all():
        st.markdown("**Top candidates by similarity to benchmark**")
        show_cols = ["employee_id","fullname","company_name","division_name","department_name",
                     "position_name","grade_name","rating","final_match_rate","benchmark_score"]
        show_cols = [c for c in show_cols if c in scored.columns]
        top_rank = (scored.sort_values("benchmark_score", ascending=False)
                           .head(show_k)[show_cols])
        top_rank["benchmark_score"] = (100 * top_rank["benchmark_score"]).round(1)
        st.dataframe(top_rank, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Top list (CSV)",
            top_rank.to_csv(index=False).encode("utf-8"),
            file_name="benchmark_top_list.csv",
            mime="text/csv",
        )
    else:
        st.info("No similarity scores yet.")

st.markdown("---")

# Ranked Candidates (by original score)
st.subheader("Ranked Candidates (by original Match Score)")
rank_cols = [
    "employee_id","fullname","company_name","directorate_name","division_name","department_name",
    "position_name","grade_name","rating","final_match_rate",
    "tgv_cognitive","tgv_strategy","tgv_execution","tgv_teamwork",
]
rank_cols = [c for c in rank_cols if c in filtered.columns]
ranked = filtered.sort_values(
    by=["final_match_rate","rating"] if {"final_match_rate","rating"}.issubset(filtered.columns)
       else ["final_match_rate"] if "final_match_rate" in filtered.columns
       else ["rating"],
    ascending=False
)
st.dataframe(ranked[rank_cols], use_container_width=True)

# Match Score Overview
st.subheader("Match Score Overview")
with st.expander("What is the Match Score?"):
    st.write(
        """
        **Match Score** is a 0–100 index comparing each employee’s competency pattern
        (Cognitive, Strategy, Execution, Teamwork) against a high-performer benchmark
        (e.g., rating=5 or the exemplars you selected above). Higher = closer to benchmark.
        """
    )

if "final_match_rate" in filtered.columns:
    st.write("**Binned distribution of match scores**")
    bins = [0, 60, 75, 85, 100]
    labels = ["Low (<60)", "Medium (60–75)", "Strong (75–85)", "Excellent (85–100)"]
    tmp = filtered.dropna(subset=["final_match_rate"]).copy()
    tmp["band"] = pd.cut(tmp["final_match_rate"], bins=bins, labels=labels, include_lowest=True, right=True)
    band_counts = (
        tmp["band"].value_counts().reindex(labels).rename_axis("Band").reset_index(name="Headcount")
    )
    band_counts["% of Employees"] = (100 * band_counts["Headcount"] / max(1, len(tmp))).round(1).astype(str) + "%"
    st.bar_chart(band_counts.set_index("Band")["Headcount"])
    st.dataframe(band_counts, use_container_width=True, hide_index=True)

st.markdown("---")

import requests

st.markdown("---")
st.subheader("AI-Generated Job Profile (per brief)")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

with st.form("ai_profile_form", clear_on_submit=False):
    col_a, col_b, col_c = st.columns([1.2, 1.0, 1.2])
    with col_a:
        role_name = st.text_input("Role name", value="Data Analyst")
    with col_b:
        job_level = st.selectbox("Job level", ["Junior", "Middle", "Senior", "Lead"], index=1)
    with col_c:
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1)

    role_purpose = st.text_area(
        "Role purpose (1–2 sentences)",
        value="Deliver reliable analytics that inform product, operations, and business decisions."
    )

    model_choice = st.selectbox(
        "Model",
        [
            "anthropic/claude-3.5-sonnet",     # OpenRouter
            "openai/gpt-4o-mini",              # OpenRouter or OpenAI
            "google/gemini-1.5-flash"          # OpenRouter
        ],
        index=0
    )

    submitted = st.form_submit_button("Generate Job Profile", type="primary")

context_points = []
if bench_vec is not None:
    for k, v in bench_vec.round(2).items():
        if pd.notnull(v):
            context_points.append(f"{k.replace('_',' ').title()}: {float(v):.2f}")

seg_facts = []
for gcol in ["division_name", "department_name", "education_name"]:
    if gcol in filtered.columns:
        topg = (filtered.groupby(gcol, dropna=True)["employee_id"]
                .count().sort_values(ascending=False).head(3))
        if not topg.empty:
            items = ", ".join([f"{idx} ({val})" for idx, val in topg.items()])
            seg_facts.append(f"Top {gcol.replace('_',' ')} by headcount: {items}")

def _extract_id(lbl: str) -> str | None:
    import re
    m = re.search(r"\(([^)]+)\)\s*$", str(lbl))
    return m.group(1).strip() if m else None

exemplar_ids = []
if 'picks' in locals() and picks:
    exemplar_ids = [_extract_id(x) for x in picks if x]
    exemplar_ids = [x for x in exemplar_ids if x]

def build_profile_prompt(role_name, job_level, role_purpose, context_points, seg_facts, exemplar_ids, delta_df):
    context_bullets = "\n".join([f"- {p}" for p in context_points]) or "- (No benchmark vector available)"
    seg_bullets = "\n".join([f"- {p}" for p in seg_facts]) or "- (No segment facts available)"
    exemplars = ", ".join(exemplar_ids) if exemplar_ids else "(none selected)"

    gaps = ""
    if isinstance(delta_df, pd.DataFrame) and not delta_df.empty:
        top_gaps = delta_df.sort_values("Delta(Top-Others)", ascending=False).head(3)
        lines = [f"- {r['Metric']}: +{r['Delta(Top-Others)']:.2f}" for _, r in top_gaps.iterrows()]
        gaps = "\n".join(lines)

    return f"""You are helping build a Talent Match Intelligence system.

RUNTIME INPUTS:
- Role name: {role_name}
- Job level: {job_level}
- Role purpose: {role_purpose}
- Selected benchmark exemplar IDs: {exemplars}

DATA CONTEXT:
Benchmark (average TGV across selected exemplars):
{context_bullets}

Segment facts:
{seg_bullets}

Performance gaps (Top vs Others):
{gaps if gaps else "- (No gaps derived)"}

TASK — Produce a concise, business-ready JOB PROFILE with the following sections and style exactly:
1) Job Description (two short paragraphs).
2) Key Responsibilities (5–8 bullets; action/impact oriented).
3) Requirements (8–12 bullets blending technical & analytical skills; SQL/Python/BI if relevant; education/experience appropriate to {job_level}).
4) Key Competencies grouped by Talent Group Variables (TGV) → Talent Variables (TV).
   - For each TGV (Cognitive, Strategy, Execution, Teamwork), list 2–4 specific TVs as measurable capabilities.
   - Emphasize TVs that align with the benchmark tendencies above.
5) Success Indicators (4–6 measurable signals for the first 90–180 days).

STYLE:
- Be specific, unbiased, measurable when possible.
- Keep bullets compact (max 1 sentence).
- Tailor seniority language to {job_level}.
- Total length ~350–500 words.

OUTPUT FORMAT (Markdown):
# {role_name} — {job_level}
## Role Purpose
<1–2 sentences>

## Job Description
<two short paragraphs>

## Key Responsibilities
- ...

## Requirements
- ...

## Key Competencies (TGV → TV)
- **Cognitive**: TV1; TV2; TV3
- **Strategy**: TV1; TV2; TV3
- **Execution**: TV1; TV2; TV3
- **Teamwork**: TV1; TV2; TV3

## Success Indicators (90–180 days)
- ...
"""

def call_llm_openrouter_or_openai(prompt: str, model: str, temperature: float) -> str:
    """Use OpenRouter if OPENROUTER_API_KEY set; else OpenAI if OPENAI_API_KEY set; else return empty."""
    key_or = os.getenv("OPENROUTER_API_KEY")
    key_oa = os.getenv("OPENAI_API_KEY")

    try:
        if key_or:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key_or}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
                timeout=90
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

        if key_oa:
            # Fallback to OpenAI 
            use_model = model if model.startswith("gpt-") else "gpt-4o-mini"
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key_oa}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": use_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
                timeout=90
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

        return ""
    except Exception as e:
        return f"AI call failed: {e}"

if submitted:
    prompt = build_profile_prompt(role_name, job_level, role_purpose, context_points, seg_facts, exemplar_ids, delta_df)
    if OPENROUTER_API_KEY:
        with st.spinner("Generating job profile…"):
            md = call_llm_openrouter_or_openai(prompt, model_choice, temperature)
        if md:
            st.success("Profile generated:")
            st.markdown(md)
            st.download_button("Download Profile (Markdown)", md.encode("utf-8"), file_name=f"{role_name}_profile.md")
        else:
            st.error("No API key found or empty response.")
            st.caption("Please copy and run this prompt in any LLM for manual use.")
            st.code(prompt)
    else:
        st.warning("No OPENROUTER_API_KEY / OPENAI_API_KEY set. Copy this prompt LLM for manual use:")
        st.code(prompt)
