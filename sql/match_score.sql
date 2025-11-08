WITH benchmark AS (

  SELECT employee_id
  FROM performance_yearly p
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY employee_id
    ORDER BY year DESC
  ) = 1
  AND rating = 5
),

benchmark_stats AS (

  SELECT
    AVG(pp.iq) AS bench_iq,
    AVG(pp.gtq_total) AS bench_gtq_total,
    AVG(cp.strategic_thinking) AS bench_strategic_thinking,
    AVG(cp.collaboration) AS bench_collaboration

  FROM profiles_psych pp
  JOIN competencies_pivot cp USING (employee_id)
  JOIN benchmark b USING (employee_id)
),

latest_perf AS (
  -- each employee's most recent performance record
  SELECT *
  FROM (
    SELECT p.*,
           ROW_NUMBER() OVER (
             PARTITION BY employee_id
             ORDER BY year DESC
           ) AS rn
    FROM performance_yearly p
  ) s
  WHERE rn = 1
),

candidate AS (
  -- one row per candidate with all TVs
  SELECT
    lp.employee_id,
    e.fullname,
    lp.rating,
    pp.iq,
    pp.gtq_total,
    cp.strategic_thinking,
    cp.collaboration

  FROM latest_perf lp
  JOIN employees e USING (employee_id)
  LEFT JOIN profiles_psych pp USING (employee_id)
  LEFT JOIN competencies_pivot cp USING (employee_id)
),

tv_match AS (
  -- per-TV match percent
  SELECT
    c.*,
    LEAST(
      100.0,
      100.0 * c.iq / NULLIF(bs.bench_iq, 0)
    ) AS tv_iq_match,

    LEAST(
      100.0,
      100.0 * c.gtq_total / NULLIF(bs.bench_gtq_total, 0)
    ) AS tv_gtq_match,

    LEAST(
      100.0,
      100.0 * c.strategic_thinking / NULLIF(bs.bench_strategic_thinking, 0)
    ) AS tv_strategic_match,

    LEAST(
      100.0,
      100.0 * c.collaboration / NULLIF(bs.bench_collaboration, 0)
    ) AS tv_collab_match

  FROM candidate c
  CROSS JOIN benchmark_stats bs
),

tgv_match AS (
  -- group TVs → TGVs
  SELECT
    employee_id,
    fullname,
    rating,

    -- Cognitive ability group
    (tv_iq_match + tv_gtq_match) / 2.0 AS tgv_cognitive,

    -- Leadership / Teamwork group
    (tv_strategic_match + tv_collab_match) / 2.0 AS tgv_leadership_teamwork

  FROM tv_match
),

final_score AS (
  -- final score = avg of TGVs
  SELECT
    employee_id,
    fullname,
    rating,

    tgv_cognitive,
    tgv_leadership_teamwork,

    (
      tgv_cognitive
      + tgv_leadership_teamwork
    ) / 2.0 AS final_match_rate

  FROM tgv_match
)

SELECT *
FROM final_score
ORDER BY final_match_rate DESC;
