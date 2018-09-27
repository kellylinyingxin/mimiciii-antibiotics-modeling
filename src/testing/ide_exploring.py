#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 16:45:52 2018

@author: geickelb1
"""


import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import collections
import asyncio
import getpass
import re
from datetime import datetime as dt
import os,sys,re
import urllib3
from collections import Counter
import seaborn as sns
import random
    

final_pt_df2 = \
       pd.read_csv('/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling/data/raw/csv/16082018_final_pt_df2.csv', index_col=0)


ventsettings_df = \
       pd.read_csv('/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling/data/raw/csv/15082018_ventsettings_df.csv', index_col=0)

date = "27082018"

name_list =[
       'gcs72_df',
       'rtt_merged_allpt_df',
       'bg_all_nosummary_72',
       'uti_all_72',
       'vitals_all_nosummary_72',
       'labs_all_nosummary_72',
       'ventsettings_72',
       'vaso_dose_72']
