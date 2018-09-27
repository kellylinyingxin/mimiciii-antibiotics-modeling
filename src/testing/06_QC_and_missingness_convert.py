
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.externals.joblib import Memory
memory = Memory(cachedir='/tmp', verbose=0)
#@memory.cache above any def fxn.

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

from notebook.services.config import ConfigManager
cm = ConfigManager()
cm.update('livereveal', {
        'width': 1024,
        'height': 768,
        'scroll': True,
})

get_ipython().run_line_magic('load_ext', 'autotime')
get_ipython().run_line_magic('reload_ext', 'autotime')


# In[ ]:


#cohort import

wd= os.getcwd() #'/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling'

final_pt_df2 = pd.read_csv( wd + '/data/raw/csv/16082018_final_pt_df2.csv' , index_col=0)
patients= list(final_pt_df2['subject_id'].unique())
hadm_id= list(final_pt_df2['hadm_id'].unique())
icustay_id= list(final_pt_df2['icustay_id'].unique())
icustay_id= [int(x) for x in icustay_id]


# In[7]:


icustay_id


# In[4]:


#reading in all of my data that is not limited to 72 hour time window between t_0 and t+72
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


# In[7]:


final_pt_df2.head()


# In[8]:


vaso_dose_72.head(20)


# # task 1: % MISSINGNESS table
# need to look for the amount of patients who are missing any value in any of the categories. want to aggregate all variables and % missingness and source into 1 dataframe

# In[242]:


print(
    labs_all_nosummary_72['hadm_id'].nunique(), #15142/15207 unique hadm_id/icustay_id
    labs_all_nosummary_72['subject_id'].nunique() #12744/12801 unique patients in this
)


# In[243]:


def missingness_fxn(df,name, groupby,filteron):

    missing_df= pd.DataFrame(100* (1-(df.groupby(groupby)[filteron].nunique()/ len(hadm_id))))
    missing_df.reset_index(inplace=True)
    missing_df= missing_df.rename(index=str, columns={"label":'label', filteron:'%missingness'})
    missing_df['source']= name
    missing_df['data_label']=groupby
    return(missing_df)


# #### all dataframes to do %missingness on (add x when complete)
# - vaso_dose_72 : x
# - ventcategory_df : x
# - echodata_72 : ?
# - labs_all_nosummary_72 : x
# - weightfirstday_df : x
# - heightfirstday_df : x
# - vitals_all_nosummary_72 :x
# - uti_all_72 :x
# - bg_all_nosummary_72 :x
# - rrt_merged_allpt_df :x
# - gcs72_df : x
# - sofa_df_72: 

# In[585]:


missingness_df= pd.DataFrame()


# ### labs x

# In[586]:


print(
    labs_all_nosummary_72['hadm_id'].nunique(), #15142/15207 unique hadm_id/icustay_id
    labs_all_nosummary_72['subject_id'].nunique() #12744/12801 unique patients in this
)


# In[587]:


#na values
labs_all_nosummary_72.loc[labs_all_nosummary_72['valuenum'].isna(),:]


# In[588]:


missingness_df = pd.concat(
    [missingness_df,
        missingness_fxn(labs_all_nosummary_72,'labs_all_nosummary_72', 'label','icustay_id')]
)


# ### vaso_dose x

# In[667]:


print(
    vaso_dose_72['icustay_id'].nunique(), #4777/15207 unique icustay_id/icustay_id
)


# In[590]:


vaso_dose_72.head(2)


# In[591]:


vaso_dose_72.loc[vaso_dose_72['vaso_rate'].isna(),:]


# In[592]:


missingness_df = pd.concat(
    [missingness_df,
        missingness_fxn(vaso_dose_72,'vaso_dose_72', 'label','icustay_id')]
)


# ### ventcategory_df x

# In[593]:


ventcategory_df.head()


# In[594]:


#need to reshape the dataframe
ventcategory_df_melt = pd.melt(ventcategory_df, id_vars=['icustay_id','t_0'], var_name='label')

print(
    ventcategory_df_melt['icustay_id'].nunique(), #15207/15207 unique icustay_id/icustay_id
)


# In[595]:


ventcategory_df_melt.head(2)


# In[596]:


#na values
ventcategory_df_melt.loc[ventcategory_df_melt['value'].isnull()]


# In[597]:


missingness_df = pd.concat(
    [missingness_df,
        missingness_fxn(ventcategory_df_melt,'ventcategory_df', 'label','icustay_id')]
)


# ### echodata_72

# In[598]:


echodata_72.head(2)


# In[599]:


#num of icustay_id/hadm_id
print(
    echodata_72['hadm_id'].nunique(), #5421/15207 unique hadm_id/icustay_id
)


# In[600]:


#number of na values

pd.DataFrame(
    {
    'bsa':[echodata_72.loc[echodata_72['bsa'].isna(),'hadm_id'].count()], 
    'bp': [echodata_72.loc[echodata_72['bp'].isna(),'hadm_id'].count()], 
    'bpsys': [echodata_72.loc[echodata_72['bpsys'].isna(),'hadm_id'].count()], 
    'bpdias': [echodata_72.loc[echodata_72['bpdias'].isna(),'hadm_id'].count()],
    'hr': [echodata_72.loc[echodata_72['hr'].isna(),'hadm_id'].count()],
    'test': [echodata_72.loc[echodata_72['test'].isna(),'hadm_id'].count()]
             }
            )


# # NEED TO ASK ABOUT WHICH VALUES ARE MOST IMPORTANT IN THIS

# In[601]:


# missingness_df = pd.concat(
#     [missingness_df,
#         missingness_fxn(labs_all_nosummary_72,'labs_all_nosummary_72', 'label','icustay_id')]
# )


# ### weightfirstday_df x

# In[602]:


#delete once done debugging
weightfirstday_df=pd.read_csv(
    '/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling/data/raw/csv/72_hr_window/%s_weightfirstday_df.csv' %(date), index_col=0)


# In[603]:


#need to reshape the df

weightfirstday_df_melt= pd.melt(weightfirstday_df, id_vars='icustay_id', var_name='label')
#weightfirstday_df_melt


# In[604]:


print(
    weightfirstday_df_melt['icustay_id'].nunique(), #15207/15207 unique icustay_id/icustay_id
)


# In[605]:


#na values
weightfirstday_df_melt.loc[weightfirstday_df_melt['value'].isna(),'icustay_id'].count() #1522/15207 do not have any weight, 39522 total null rows.


# In[606]:


#missingness df
missingness_df = pd.concat(
    [missingness_df,
        missingness_fxn(weightfirstday_df_melt.loc[weightfirstday_df_melt['value'].notnull(),:],'weightfirstday_df', 'label','icustay_id')]
)


# In[607]:


missingness_df


# ### heightfirstday_df x

# In[608]:


heightfirstday_df.head(2)


# In[609]:


#need to reshape the df
heightfirstday_df_melt= pd.melt(heightfirstday_df, id_vars='icustay_id', var_name='label')


# In[610]:


print(
    heightfirstday_df_melt['icustay_id'].nunique(), #15207/15207 unique icustay_id/icustay_id
)


# In[611]:


#na values
print(
    heightfirstday_df_melt.loc[heightfirstday_df_melt['value'].isna(),'icustay_id'].count(), #22776 total null rows.
    heightfirstday_df.loc[heightfirstday_df['height'].isna(),'icustay_id'].count() #5212/15207 do not have any weight,
)


# In[612]:


#missingness df
missingness_df = pd.concat(
    [missingness_df,
        missingness_fxn(heightfirstday_df_melt.loc[heightfirstday_df_melt['value'].notnull(),:],'heightfirstday_df', 'label','icustay_id')]
)


# In[613]:


missingness_df


# ### vitals_all_nosummary_72 x

# In[614]:


vitals_all_nosummary_72.head(2)


# In[615]:


#na values
vitals_all_nosummary_df.loc[vitals_all_nosummary_df['vitalid'].notnull(),:]#.count() #6930 NULL values

vitals_all_nosummary_nonull= vitals_all_nosummary_72.loc[vitals_all_nosummary_72['vitalid'].notnull(),:]#.count() #6930 NULL values

#why do i have null vitalid's with actual values? need to dive into the sql.
###it looks like it may just extract a lot of values as null that are not relevant, can filter these out


# In[616]:


print(
    vitals_all_nosummary_nonull['icustay_id'].nunique(), #14714/15207 unique icustay_id/icustay_id
)


# In[617]:


vitals_all_nosummary_nonull


# In[618]:


missingness_df = pd.concat(
    [missingness_df,
        missingness_fxn(vitals_all_nosummary_nonull.rename(index=str, columns={'vitalid':'label'}),'vitals_all_nosummary_72', 'label','icustay_id')]
)


# ### uti_all_72 x

# In[619]:


uti_all_72=pd.read_csv(
    '/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling/data/raw/csv/72_hr_window/%s_uti_all_72.csv' %(date), index_col=0)


# In[620]:


uti_all_72.head(2)


# In[621]:


#uti_all_72_melt= pd.melt(uti_all_72, id_vars=['subject_id','hadm_id','itemid','charttime','t_0','delta'], var_name='label')


# In[622]:


uti_all_72_melt= pd.melt(
    uti_all_72[['hadm_id','charttime','value','valuenum','valueuom','label','fluid']],
    id_vars=['hadm_id','charttime'], var_name='label')


# In[623]:


uti_all_72_melt_notnull= uti_all_72_melt[uti_all_72_melt['value'].notnull()] #1320954


# In[624]:


uti_all_72_melt_notnull['label'].value_counts()


# In[625]:


print(
    uti_all_72_melt_notnull.nunique(), #6145/15207 unique hadm_id/hadm_id
)


# In[626]:


#na values
#uti_all_72_melt.loc[uti_all_72_melt['value'].isna(),:]#.count() #220159 na valuenum, 440318 when melted
#uti_all_72_melt[uti_all_72_melt['value'].isnull()]['hadm_id'].count() #440318


# In[627]:


#uti_all_72_melt[uti_all_72_melt['value'].notnull()]


# In[628]:


missingness_fxn(uti_all_72_melt_notnull,'uti_all_72_melt', 'label','hadm_id')


# In[629]:


missingness_df = pd.concat(
    [missingness_df,
        missingness_fxn(uti_all_72_melt_notnull,'uti_all_72_melt', 'label','hadm_id')]
)


# ### bg_all_nosummary_72 x

# In[630]:


bg_all_nosummary_72=pd.read_csv(
    '/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling/data/raw/csv/72_hr_window/%s_bg_all_nosummary_72.csv' %(date), index_col=0)


# In[631]:


#na values- removing all null values from value.
bg_all_nosummary_72= bg_all_nosummary_72.loc[bg_all_nosummary_72['value'].notnull(),:]

bg_all_nosummary_72.head(2)


# In[632]:


print(
    bg_all_nosummary_72['icustay_id'].nunique(), #11244/15207 unique icustay_id/icustay_id
)


# In[633]:


missingness_df = pd.concat(
    [missingness_df,
        missingness_fxn(bg_all_nosummary_72,'bg_all_nosummary_72', 'label','icustay_id')]
)


# In[634]:


#missingness_fxn(bg_all_nosummary_72,'bg_all_nosummary_72', 'label','icustay_id')


# ### rrt_merged_allpt_df x

# In[635]:


rrt_merged_allpt_df.head(10)


# In[636]:


#na values
rrt_merged_allpt_df_notnull = rrt_merged_allpt_df.loc[rrt_merged_allpt_df['rrt'].notnull(),:]


# In[637]:


print(
    rrt_merged_allpt_df_notnull['icustay_id'].nunique(), #1223/15207 unique icustay_id/icustay_id
)


# In[638]:


#missingness_fxn(rrt_merged_allpt_df_notnull.rename(index=str,columns={'rrt':"label"}),'rrt_merged_allpt_df_notnull', 'label','icustay_id')


# In[639]:


missingness_df = pd.concat(
    [missingness_df,
        missingness_fxn(rrt_merged_allpt_df_notnull.rename(index=str,columns={'rrt':"label"}),'rrt_merged_allpt_df_notnull', 'label','icustay_id')]
)


# ### gcs72_df x

# In[640]:


gcs72_df.head(2)


# In[641]:


#need to reshape the dataframe
gcs72_df_melt = pd.melt(gcs72_df, id_vars=['subject_id','hadm_id','icustay_id','t_0','ICU_admit','admit_plus_day','delta'], var_name='label')


# In[642]:


gcs72_df_melt_notnull= gcs72_df_melt[gcs72_df_melt['value'].notnull()] #62772 ->62650


# In[643]:


print(
    gcs72_df_melt_notnull['icustay_id'].nunique(), #10462/15207 unique icustay_id/icustay_id
)


# In[644]:


missingness_fxn(gcs72_df_melt_notnull,'gcs72_df_melt_notnull', 'label','icustay_id')


# In[645]:


missingness_df = pd.concat(
    [missingness_df,
        missingness_fxn(gcs72_df_melt_notnull,'gcs72_df_melt_notnull', 'label','icustay_id')]
)


# ### sofa_df_72

# In[650]:


sofa_df_72.head(2)


# In[654]:


#need to reshape the dataframe
sofa_df_melt = pd.melt(sofa_df_72, id_vars=['subject_id','hadm_id','icustay_id','t_0','day','ICU_admit','approx_charttime','floor_charttime'], var_name='label')


# In[659]:


len(sofa_df_melt_notnull)


# In[658]:


sofa_df_melt_notnull= sofa_df_melt[sofa_df_melt['value'].notnull()] #606651 ->302459


# In[660]:


print(
    sofa_df_melt_notnull['icustay_id'].nunique(), #13323/15207 unique icustay_id/icustay_id
)


# In[665]:


missingness_df = pd.concat(
    [missingness_df,
        missingness_fxn(sofa_df_melt_notnull,'sofa_df_melt_notnull', 'label','icustay_id')]
)


# In[666]:


missingness_df = missingness_df.sort_values('%missingness', ascending=False)

date= '27082018'

pd.DataFrame(missingness_df).to_csv(
    '/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling/data/interim/%s_missingness_df.csv' %(date))


# ### visualizations

# In[ ]:


"""
    get the # of filled vs missign values for each variable for our cohort.
    Args:
    -----
    df: dataframe
        name of dataframe initialized in this workbook
    valuenum_col: string
        name of column with the numerical or text variable in question
    label_col: string
        name of column with the lab value label in question
    Returns:
    -----
    filled_summary: df
        list of all variables with # filled values
    missing_summary: df
        list of all variables with # missing values
    percent_summary: df
        list of all variables with # filled values/ # total patients
    """

def missing_fxn(df, valuenum_col, label_col):
    if 'icustay_id' in list(df):
        df= df[df['icustay_id'].isin(list(cohort_df['icustay_id'].unique()))] #good
    else:
        df= df[df['subject_id'].isin(
            list(cohort_df['subject_id'].unique()))] 

    df_notnull = df[df[valuenum_col].notnull()]
    df_null = df[df[valuenum_col].isnull()]
    df_filtered = df.drop_duplicates(['subject_id',label_col])[label_col].value_counts()
    
    filled_summary= df_notnull.drop_duplicates(['subject_id',label_col])[label_col].value_counts()
    missing_summary = df_null.drop_duplicates(['subject_id',label_col])[label_col].value_counts()
    percent_summary = 100*(filled_summary / df_filtered)
    
    #missing_plot=(missing_summary.sort_index().plot.bar(title = f'# of missing values in {df}'))
    
    #filled_summary, missing_summary, percent_summary, 
    return(filled_summary, missing_summary, percent_summary)


# In[ ]:


var1, var2, var3,  = missing_fxn(labs_all48, 'valuenum', 'label')
fig, axs = plt.subplots(1,2,figsize=(10,4))
ax1= (var2.sort_index().plot.bar(ax=axs[0], title = '# of missing values in labs_all48'))
ax2= (var3.sort_index().plot.bar(color='grey',
                                ax=axs[1],
                                ylim=(min(var3)-10 ,100),
                                title = '% of values filled in labs_all48'))


# #### histogram of frequency over each hour.

# In[10]:


def hours_fxn(td):
    return ((td.days*24*3600) + td.seconds)//3600

def days_fxn(td):
    return td.days


# In[18]:


pd.to_timedelta(vaso_dose_72['delta'].head())


# In[21]:


for element in vaso_dose_72['label'].unique():
    print(element)


# In[22]:


size=11
fig=plt.figure(figsize=(size, size*(2/3)), dpi= 80, facecolor='w')
ax = plt.axes()

vaso_dose_72['hours']=pd.to_timedelta(vaso_dose_72['delta']).map(hours_fxn) #good

for element in vaso_dose_72['label'].unique():    
    (vaso_dose_72.loc[vaso_dose_72.loc[:,'label']==element, 'hours']
         .value_counts(sort=False)
         .sort_index()
         .plot(kind='bar', sort_columns=True))


# In[30]:


for element in vaso_dose_72['label'].unique():
    size=9
    fig=plt.figure(figsize=(size, size*(2/3)), dpi= 80, facecolor='w')
    ax = plt.axes()
    (vaso_dose_72.loc[vaso_dose_72.loc[:,'label']==element, 'hours']
         .value_counts(sort=False)
         .sort_index()
         .plot(kind='bar', sort_columns=True,color='blue'))
    print(element)
    ax.set_ylabel('frequency_for_all_pt', fontsize='large')
    ax.set_xlabel('hours_after_t_0', fontsize='large')
    ax.set_title('%s'%(element), fontsize='large')
    plt.show()


# In[42]:


#looking at overall missingness

for element in vaso_dose_72['label'].unique():
    print(element)
    print(vaso_dose_72.loc[vaso_dose_72.loc[:,'label']==element, 'icustay_id']
         .nunique())
    print(((vaso_dose_72.loc[vaso_dose_72.loc[:,'label']==element, 'icustay_id']
         .nunique())/len(icustay_id))*100,"% of icustay_id's with any value") 


# In[33]:


print(vaso_dose_72['icustay_id'].nunique())


# In[36]:


len(icustay_id)


