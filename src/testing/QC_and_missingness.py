
#%%
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.externals.joblib import Memory
memory = Memory(cachedir='/tmp', verbose=0)
#@memory.cache above any def fxn.

%matplotlib inline
plt.style.use('ggplot')

from notebook.services.config import ConfigManager
cm = ConfigManager()
cm.update('livereveal', {
        'width': 1024,
        'height': 768,
        'scroll': True,
})

#%load_ext autotime
#%reload_ext autotime


#%%cohort import
os.chdir('/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling') #use to change working directory
wd= os.getcwd() #'/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling'

final_pt_df2 = pd.read_csv( wd + '/data/raw/csv/16082018_final_pt_df2.csv' , index_col=0)
patients= list(final_pt_df2['subject_id'].unique())
hadm_id= list(final_pt_df2['hadm_id'].unique())
icustay_id= list(final_pt_df2['icustay_id'].unique())
icustay_id= [int(x) for x in icustay_id]

#%%reading in all of my data that is not limited to 72 hour time window between t_0 and t+72

date= '27082018'
wd= os.getcwd() #'/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling'

vaso_dose_72=pd.read_csv(
    wd+'/data/raw/csv/72_hr_window/%s_vaso_dose_72.csv' %(date), index_col=0)
#ventsettings_72=pd.read_csv(
#    wd+'/data/raw/csv/72_hr_window/%s_ventsettings_72.csv' %(date), index_col=0)

ventcategory_df=pd.read_csv(
    wd+'/data/raw/csv/%s_ventcategory_df.csv' %(date), index_col=0)

echodata_72=pd.read_csv(
    wd+'/data/raw/csv/72_hr_window/%s_echodata_72.csv' %(date), index_col=0)
labs_all_nosummary_72=pd.read_csv(
    wd+'/data/raw/csv/72_hr_window/%s_labs_all_nosummary_72.csv' %(date), index_col=0)


weightfirstday_df=pd.read_csv(
    wd+'/data/raw/csv/72_hr_window/%s_weightfirstday_df.csv' %(date), index_col=0)
heightfirstday_df=pd.read_csv(
    wd+'/data/raw/csv/72_hr_window/%s_heightfirstday_df.csv' %(date), index_col=0)

vitals_all_nosummary_72=pd.read_csv(
    wd+'/data/raw/csv/72_hr_window/%s_vitals_all_nosummary_72.csv' %(date), index_col=0)
uti_all_72=pd.read_csv(
    wd+'/data/raw/csv/72_hr_window/%s_uti_all_72.csv' %(date), index_col=0)
bg_all_nosummary_72=pd.read_csv(
    wd+'/data/raw/csv/72_hr_window/%s_bg_all_nosummary_72.csv' %(date), index_col=0)

rrt_merged_allpt_df=pd.read_csv(
    wd+'/data/raw/csv/72_hr_window/%s_rrt_merged_allpt_df.csv' %(date), index_col=0)
gcs72_df=pd.read_csv(
    wd+'/data/raw/csv/72_hr_window/%s_gcs72_df.csv' %(date), index_col=0)

sofa_df_72=pd.read_csv(
    wd+'/data/raw/csv/%s_sofa_df_72.csv' %(date), index_col=0)

#%% Task1: %Missingness table
#print(sofa_df_72.head())

print(
    labs_all_nosummary_72['hadm_id'].nunique(), #15142/15207 unique hadm_id/icustay_id
    labs_all_nosummary_72['subject_id'].nunique() #12744/12801 unique patients in this
)

#%%