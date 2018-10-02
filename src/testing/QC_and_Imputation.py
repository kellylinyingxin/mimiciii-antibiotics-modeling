#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:55:47 2018

@author: geickelb1
"""
#%%importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.externals.joblib import Memory
memory = Memory(cachedir='/tmp', verbose=0)
#@memory.cache above any def fxn.

%matplotlib inline
plt.style.use('ggplot')

#%% importing cohort
os.chdir('/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling') #use to change working directory
wd= os.getcwd() #'/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling'

final_pt_df2 = pd.read_csv(wd + '/data/raw/csv/16082018_final_pt_df2.csv' , index_col=0)
patients= list(final_pt_df2['subject_id'].unique())
hadm_id= list(final_pt_df2['hadm_id'].unique())
icustay_id= list(final_pt_df2['icustay_id'].unique())
icustay_id= [int(x) for x in icustay_id]

#%% importing relevant dataframes
date= '27082018' 

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


#%% importing missingness table
#pd.DataFrame(missingness_df).to_csv(
#    '/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling/data/interim/%s_missingness_df.csv' %(date))

missingness_df=pd.read_csv(
    wd+'/data/interim/%s_missingness_df.csv' %(date), index_col=0)
impute_grps = sofa_df_72.pivot_table(values=['sofa'], index=['subject_id','day'], aggfunc=np.mean)

#%% 

# =============================================================================
    #from nelson's email after seeing missingness:
#This looks good. As expected, there are some patents (something like 3%) who don't have even some basic vital signs.
#As a pragmatic cutoff, I suggest that we take off patients who don't have at least an SpO2 reading (approx 3.5% of patients),
#I think that is a reasonable minimum of data to have. 
#I will be looking at this more carefully in the coming week or so.
# =============================================================================


# =============================================================================
#   goal: explore the 12-pandas-techniqus-python-data-manipulation 
# =============================================================================



#%%Pivot Table

impute_grps = sofa_df_72.pivot_table(values=['sofa'], index=['subject_id','day'], aggfunc=np.mean)