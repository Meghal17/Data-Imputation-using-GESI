import pandas as pd
import numpy as np
from tqdm import tqdm
import sklearn
import os
import gc
gc.enable()
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")

def calculate_NRMS(y_true, y_pred):
	upper_values = y_pred - y_true
	upper_normed = np.linalg.norm(upper_values, ord='fro')
	lower_normed = np.linalg.norm(y_true, ord='fro')
	return upper_normed / lower_normed

def calculate_AE_DICT(y_true, y_pred):
	y_true = y_true.flatten()
	y_pred = y_pred.flatten()
	return y_true[y_true != y_pred].shape[0] / float(y_true.shape[0])

# Helper Function
def read_subsets_and_original(BASE_PATH, ORIGINAL_BASE_PATH, dataset_name):
	sub_incomplete_dataset = os.listdir(BASE_PATH + dataset_name)
	# print ("Found Total Subsets : ", len(sub_incomplete_dataset))
	original = pd.read_excel(ORIGINAL_BASE_PATH + dataset_name + '.xlsx', header=None)
	original = original.infer_objects()
	subsets = {}
	for each in tqdm(sub_incomplete_dataset, total=len(sub_incomplete_dataset)):
		subsets[each.split('.')[0]] = pd.read_excel(BASE_PATH + dataset_name + '/' + each, header=None)
	return subsets, original

# Implementing SAGA feature selection
def SAGA_FEATURE_SELECTION(X_train, y_train):
	model_logistic = Ridge(solver='saga')
	sel_model_logistic = SelectFromModel(estimator=model_logistic)
	X_train_sfm_l1 = sel_model_logistic.fit_transform(X_train.values, y_train.values)
	Indicator_columns = sel_model_logistic.get_support()
	return Indicator_columns