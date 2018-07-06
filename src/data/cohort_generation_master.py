
"""
cohort generation master script
"""

#libraries
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
import prettytable
from collections import Counter
import seaborn as sns
import random
from datetime import timedelta

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

#for now i will just use the csv from the prescription table, but in the future i'd like to update this to run that script

dtype = {'icustay_id': str,
         'NDC': str,
         'rxcui': str,
         'ingredient': str}
prescriptions = pd.read_csv('/Users/geickelb1/Documents/GitHub/mimiciii-antibiotics-modeling/data/wrangled/Newprescription.csv', index_col=0, dtype=dtype)

###ended here 07/05/18
###remind myself how i can reference another python file. 