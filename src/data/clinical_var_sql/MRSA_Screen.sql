DROP MATERIALIZED VIEW IF EXISTS MRSA_screen;
CREATE MATERIALIZED VIEW MRSA_screen as

select subject_id, hadm_id, charttime, chartdate, spec_type_desc, org_name, interpretation 
from mimiciii.microbiologyevents
where spec_type_desc = 'MRSA SCREEN'
