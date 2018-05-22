#!/usr/bin/env python


"""
Module containing functions to subset the raw data:
keeps description, country, price, points and adds
column for price in GBP

"""

import sys
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.externals.joblib import Memory
memory = Memory(cachedir='/tmp', verbose=0)
#@memory.cache above any def fxn.


#not shown: imported antibiotic annotations into prescription table using code provided by
#Dr. Luo's postdoc, which pings API


#implicit before hand:
#prescriptions = pd.read_csv('/Users/geickelb1/Desktop/PhD_Misc/HSIP_442_Yuan_Lao/project/Newprescription.csv', index_col=0, dtype=dtype)




@memory.cache
def prescription_update_fxn(prescription_df):
    """
    take the prescription table w/ antibiotics generated from NDC->ATC conversion and update it to fix some antibiotics that were coded as false for some NDC codes but true for others. 
    Args:
    -----
    prescription_df: prescription table generated previously with anootated antiboitic column
        can use: pd.read_csv('/Users/geickelb1/Desktop/PhD_Misc/HSIP_442_Yuan_Lao/project/Newprescription.csv', index_col=0, dtype=dtype)

    Returns:
    -----
    ABrx2: df
        updated dataframe
    """
    #updating prescription table to include rows not initially captured by atc conversion
    prescriptions= prescription_df
    prescriptions_updated = list(prescriptions.loc[prescriptions.loc[:,"Antibiotics"]==True,'drug'].unique())
    
    #some prescriptions have multiple ndc codes, putting all "true" antibiotics with any true NDC code in a list
    true_ndc= prescriptions.loc[prescriptions.loc[:,"drug"].isin(prescriptions_updated),'ndc'].unique() 

    ABrx2= prescriptions.loc[prescriptions.loc[:,"ndc"].isin(true_ndc),:]
   
    #removing known problematic drugs
    drugs_that_dont_belong =['Furosemide','Dextrose 50%','Vancomycin Oral Liquid',
                         'Erythromycin 0.5% Ophth Oint','NEO*IV*Furosemide',
                         'Nystatin','Orthopedic Solution','Neomycin-Polymyxin-Bacitracin Ophth. Oint',
                         'Bacitracin Ophthalmic Oint','Bacitracin Ointment','Lasix',
                         'dextrose','MetronidAZOLE Topical 1 % Gel','Enalaprilat',
                         'NEO*PO*Furosemide (10mg/1ml)','Metronidazole Gel 0.75%-Vaginal','Spironolactone',
                         'Heparin',
                        'voriconazole','valgancyclovir','chloroquine','tamiflu','mefloquine','foscarnet',
                         'fluconazole','vorconazole','quinine','ribavirin','gancyclovir','chloroquine',
                         'atovaquone','ambisome', 'acyclovir', 'Acyclovir']

    drugs_that_dont_belong=[w.lower() for w in drugs_that_dont_belong]


    
    ABrx2= ABrx2.loc[~ABrx2.loc[:,"drug"].str.lower().isin(drugs_that_dont_belong),:] #tilde transforms isin to notin()

    # Constructing the fname
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    fname = f'data/interim/{today}-Prescription_updated.csv' #not sure why they add the f before. 

    # Saving the csv
    ABrx2.to_csv(fname, index = False)
    
    return(ABrx2)

if __name__ == '__main__':
    filename = sys.argv[1]
    print(filename)
    print(prescription_update_fxn(filename))   #what does this do?

