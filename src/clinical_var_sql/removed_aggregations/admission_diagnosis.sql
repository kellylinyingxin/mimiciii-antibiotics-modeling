DROP MATERIALIZED VIEW IF EXISTS admission_diagnosis CASCADE;
create materialized view admission_diagnosis AS

select hadm_id, diagnosis FROM mimiciii.admissions
order by hadm_id
