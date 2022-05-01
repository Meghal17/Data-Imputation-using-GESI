import pandas as pd
import numpy as np
# import tensorflow as tf
import sklearn
import os
import gc
gc.enable()
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")

from GRNN import GRNN
from Imputation import *
from Config import *


# Reading metadata aboutdatasets
meta_data = pd.read_excel(BASE + META)
categorical = meta_data[meta_data['Numerical'] == 0].reset_index(drop=True)
numerical = meta_data[meta_data['Categorical'] == 0].reset_index(drop=True)
combined = meta_data[(meta_data['Numerical'] != 0) & (meta_data['Categorical'] != 0)].reset_index(drop=True)

# Preprocess the metadata
numerical['Abbreviation'][numerical['Abbreviation'] == 'BUPA'] = 'Bupa'
categorical['Abbreviation'][categorical['Abbreviation'] == 'TTTEG'] = 'TTTTEG'
numerical_data = set(numerical['Abbreviation'])
categorical_data = set(categorical['Abbreviation'])
combined_data = set(combined['Abbreviation'])

log = ""
NRMS_DICT = dict()
AE_DICT = dict()
log_data = [log, NRMS_DICT, AE_DICT]

# Impute all datasets
print("\n Imputing Numerical Datasets\n")
for num_data in numerical_data:
	print("\nPerforming Imputation on {} dataset\n".format(num_data))
	dataset = numerical[numerical["Abbreviation"] == num_data]
	impute_numerical(dataset, log_data)
print("All Numerical Datasets Imputed. Preparing Zip file of imputed datasets.")

print("\nImputing Categorical Datasets\n")
for cat_data in categorical_data:
    print("\nPerforming Imputation on {} dataset\n".format(cat_data))
    dataset = categorical[categorical["Abbreviation"] == cat_data]
    impute_categorical(dataset, log_data)
print("All categorical Datasets Imputed. Preparing Zip file of imputed datasets.")

print("\nImputing Combined Datasets\n")
for com_data in combined_data:
    print("\nPerforming Imputation on {} dataset\n".format(com_data))
    dataset = combined[combined["Abbreviation"] == com_data]
    impute_combined(dataset, log_data)
print("All combined Datasets Imputed. Preparing Zip file of imputed datasets.")

# Update NRMS and AE Values 
df = pd.read_excel(BASE + '/Table-NRMS-AE.xlsx')
logs = log_data[0].split('/n')
for each in logs:
    x = each.split(' ')
    if len(x) > 0:
        if x[0] == 'Done':
            if x[6] == 'NRMS':
                df['NRMS'][df['Datasets'] == str(x[4])] = float(x[8])
                df['NRMS'][df['Datasets'] == str(x[4][0:6]).title() + str(x[4][6:])] = float(x[8])
            if x[6] == 'AE':
                df['AE'][df['Datasets'] == str(x[4])] = float(x[8])
                df['AE'][df['Datasets'] == str(x[4][0:6]).title() + str(x[4][6:])] = float(x[8])
df.to_excel("Table-NRMS-AE-final.xlsx", index=False) 