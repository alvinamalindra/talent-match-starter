-- Sorry all of it is not usable 100% i just put any query i have been use to check the result, dont use it pls

create table if not exists dim_companies (
  company_id       bigint primary key,
  company_name     text not null
);

create table if not exists dim_divisions (
  division_id      bigint primary key,
  division_name    text not null
);

create table if not exists dim_departments (
  department_id    bigint primary key,
  department_name  text not null
);

create table if not exists dim_directorates (
  directorate_id   bigint primary key,
  directorate_name text not null
);

create table if not exists dim_grades (
  grade_id         bigint primary key,
  grade_name       text not null
);

create table if not exists dim_positions (
  position_id      bigint primary key,
  position_name    text not null
);

create table if not exists dim_areas (
  area_id          bigint primary key,
  area_name        text not null
);

create table if not exists dim_education (
  education_id     bigint primary key,
  education_name   text not null
);

create table if not exists dim_majors (
  major_id         bigint primary key,
  major_name       text not null
);


create index if not exists idx_dim_companies_name     on dim_companies     (company_name);
create index if not exists idx_dim_divisions_name     on dim_divisions     (division_name);
create index if not exists idx_dim_departments_name   on dim_departments   (department_name);
create index if not exists idx_dim_directorates_name  on dim_directorates  (directorate_name);
create index if not exists idx_dim_grades_name        on dim_grades        (grade_name);
create index if not exists idx_dim_positions_name     on dim_positions     (position_name);
create index if not exists idx_dim_areas_name         on dim_areas         (area_name);
create index if not exists idx_dim_education_name     on dim_education     (education_name);
create index if not exists idx_dim_majors_name        on dim_majors        (major_name);


create or replace view final_score_enriched as
select
  f.employee_id,
  f.fullname,
  f.rating,
  f.tgv_cognitive, f.tgv_strategy, f.tgv_execution, f.tgv_teamwork,
  f.final_match_rate,

  p.company_id, p.division_id, p.department_id, p.directorate_id,
  p.grade_id, p.position_id, p.area_id, p.education_id, p.major_id,

  coalesce(dc.company_name,        'Unknown') as company_name,
  coalesce(dv.division_name,       'Unknown') as division_name,
  coalesce(dd.department_name,     'Unknown') as department_name,
  coalesce(dr.directorate_name,    'Unknown') as directorate_name,
  coalesce(dg.grade_name,          'Unknown') as grade_name,
  coalesce(dp.position_name,       'Unknown') as position_name,
  coalesce(da.area_name,           'Unknown') as area_name,
  coalesce(de.education_name,      'Unknown') as education_name,
  coalesce(dm.major_name,          'Unknown') as major_name,

  p.years_of_service_months,
  p.iq as profile_iq,
  p.gtq as profile_gtq,
  p.pauli, p.disc, p.mbti, p.tiki

from final_score f
left join employee_profile p using (employee_id)
left join dim_companies     dc on p.company_id     = dc.company_id
left join dim_divisions     dv on p.division_id    = dv.division_id
left join dim_departments   dd on p.department_id  = dd.department_id
left join dim_directorates  dr on p.directorate_id = dr.directorate_id
left join dim_grades        dg on p.grade_id       = dg.grade_id
left join dim_positions     dp on p.position_id    = dp.position_id
left join dim_areas         da on p.area_id        = da.area_id
left join dim_education     de on p.education_id   = de.education_id
left join dim_majors        dm on p.major_id       = dm.major_id
;