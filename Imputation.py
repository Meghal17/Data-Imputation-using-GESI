import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from operator import itemgetter
from Config import *
from Helpers import *
from GRNN import GRNN
import sklearn
from sklearn.preprocessing import StandardScaler
import os

def impute_numerical(dataset, log_data):  
	import time
	begin_time = time.time()

	for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]): #For Each Numerical Dataset
		subsets, original = read_subsets_and_original(BASE_PATH, ORIGINAL_BASE_PATH, row['Abbreviation']) # Get All Subsets and Original Dataset
		subset_names = list(subsets.keys())
		#ITERATE OVER ALL SUBSETS OF A DATASET AND APPLY GRNN ON EACH ONE
		for each_subset_name in subset_names:

			#SELECTING A SUBSET
			selected_subset = subsets[each_subset_name]


			new_prediction = np.zeros(shape=original.shape) #SAMPLE ARRAY TO SAVE PREDICTIONS
			new_prediction = pd.DataFrame(data = new_prediction, columns=selected_subset.columns) 


			#COLUMNS ARRAY TO ITERATE
			all_cols = np.array(original.columns) 
			for each in tqdm(all_cols, total=len(all_cols)):



				#ONE COLUMN IN TEST AND OTHERS IN TRAINING
				train_cols = all_cols[all_cols != each] 
				test_col = each

				#CHECKING IF THERE ARE NULL VALUES IN OUR TEST COLUMNS
				nulls = selected_subset[each].isnull() 
				test_index = nulls[nulls == True].index
				train_index = nulls[nulls == False].index


				#IF THERE IS NO NULL VALUE THEN WO WONT APPLY GRNN
				if test_index.shape[0] == 0 or test_index.shape[0] / float(nulls.shape[0]) < 0.1:
					new_prediction[each] = original[each].copy()

				else:
					#TRAIN GRNN ON INDEX WHERE THERE IS NO NULL AND PREDICT ON NULL VALUES
					custom_GRNN = GRNN()
					SAGA_BASED_FEATURES = SAGA_FEATURE_SELECTION(original[train_cols].loc[train_index], original[test_col].loc[train_index]) #SAGA

					#Normalization
					normalizer = StandardScaler()

					train_X = original[train_cols[SAGA_BASED_FEATURES]].loc[train_index].values
					train_Y = original[test_col].loc[train_index].values

					test_X = original[train_cols[SAGA_BASED_FEATURES]].loc[test_index].values

					normalizer.fit(train_X, train_Y)

					normalizer_train_X = normalizer.transform(train_X)
					normalizer_test_X = normalizer.transform(test_X)


					custom_GRNN.fit(normalizer_train_X, train_Y, log_data)  # pass log argument here

					#PREDICT
					prediction_smothened = custom_GRNN.predict(normalizer_test_X)

					#FILL OUR SAVING ARRAY WITH PREDICTIONS
					new_prediction[each].loc[train_index] = selected_subset[each].loc[train_index]
					new_prediction[each].loc[test_index] = prediction_smothened
			if not os.path.isdir(os.path.join(IMPUTED_NUMERICAL, str(row['Abbreviation']))):
				os.mkdir(os.path.join(IMPUTED_NUMERICAL, str(row['Abbreviation'])))
			new_prediction.to_csv(IMPUTED_NUMERICAL + '/' + str(row['Abbreviation']) + "/imputed_" + each_subset_name + ".csv", index=False)
			NRMSE = calculate_NRMS(original.values, new_prediction.values)
			log_data[0] = log_data[0] + "Done Smoothing of : " + each_subset_name + " with NRMS : " + str(NRMSE) + '/n/n/n'
			log_data[1][each_subset_name] = NRMSE

	end_time = time.time()
	diff = end_time - begin_time
	print(diff) 


def impute_categorical(dataset, log_data):
	import time

	begin_time = time.time()
	for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):  # For Each Numerical Dataset
		subsets, original = read_subsets_and_original(BASE_PATH, ORIGINAL_BASE_PATH, row["Abbreviation"])  # Get All Subsets and Original Dataset
		# This block (next 4 lines) of code is diff from numerical impute
		new_columns = []
		for each in original.columns:
			x_each = str(each)
			new_columns.append(x_each.replace(" ", ""))

		xoriginal = original.copy()

		subset_names = list(subsets.keys())
		# ITERATE OVER ALL SUBSETS OF A DATASET AND APPLY GRNN ON EACH ONE
		for each_subset_name in subset_names:

			# SELECTING A SUBSET
			selected_subset = subsets[each_subset_name]
			ss = pd.concat([selected_subset, xoriginal])
			ss = pd.get_dummies(ss)  # APPLY ONE HOT ENCODING
			selected_subset = ss[0 : selected_subset.shape[0]]
			original = ss[selected_subset.shape[0] :]

			new_columns = []
			for each in original.columns:
				x_each = str(each)
				new_columns.append(x_each.replace(" ", ""))
			original.columns = new_columns

			new_columns = []
			for each in selected_subset.columns:
				x_each = str(each)
				new_columns.append(x_each.replace(" ", ""))
			selected_subset.columns = new_columns

			new_prediction = np.zeros(shape=original.shape)  # SAMPLE ARRAY TO SAVE PREDICTIONS
			new_prediction = pd.DataFrame(data=new_prediction, columns=selected_subset.columns)

			# COLUMNS ARRAY TO ITERATE
			all_cols = np.array(original.columns)
			for each in all_cols:
				# ONE COLUMN IN TEST AND OTHERS IN TRAINING
				train_cols = all_cols[all_cols != each]
				test_col = each

				# CHECKING IF THERE ARE NULL VALUES IN OUR TEST COLUMNS
				nulls = selected_subset[each].isnull()
				if len(nulls.shape) > 1:
					if nulls.shape[1] > 1:
						nulls = pd.DataFrame(nulls.values[:, 1], columns=[each])
				test_index = nulls[nulls == True].index
				train_index = nulls[nulls == False].index

				# IF THERE IS NO NULL VALUE THEN WO WONT APPLY GRNN
				if (test_index.shape[0] == 0 or test_index.shape[0] / float(nulls.shape[0]) < 0.1):
					new_prediction[each] = original[each].copy()

				elif test_index.shape[0] == test_index.shape[0]:
					new_prediction[each] = 0

				else:
					# TRAIN GRNN ON INDEX WHERE THERE IS NO NULL AND PREDICT ON NULL VALUES
					custom_GRNN = GRNN()
					custom_GRNN.fit(
						original[train_cols].loc[train_index].values,
						original[test_col].loc[train_index].values,
					)

					# PREDICT
					prediction_smothened = custom_GRNN.predict(
						original[train_cols].loc[test_index].values
					)

					# FILL OUR SAVING ARRAY WITH PREDICTIONS
					new_prediction[each].loc[train_index] = selected_subset[each].loc[
						train_index
					]
					if len(prediction_smothened.shape) > 1:
						if (
							prediction_smothened.shape[0] > 1
							and prediction_smothened.shape[1] > 1
						):
							prediction_smothened = prediction_smothened[0, :]
							cols = new_prediction.columns
							x = new_prediction.pop(each)
							new_prediction[each] = x.values[:, 0]
							new_prediction = new_prediction[cols]

					new_prediction[each].loc[test_index] = prediction_smothened

			if not os.path.isdir(os.path.join(IMPUTED_CAT,str(row['Abbreviation']))):
				os.mkdir(os.path.join(IMPUTED_CAT,str(row['Abbreviation'])))
			new_prediction.to_csv(IMPUTED_CAT + '/' + str(row['Abbreviation'])+ "/imputed_" + each_subset_name + ".csv", index=False)
			AE = calculate_AE_DICT(original.values, new_prediction.values)
			log_data[0] = (
				log_data[0]
				+ "Done Smoothing of : "
				+ each_subset_name
				+ " with AE : "
				+ str(AE)
				+ "/n/n/n"
			)
			log_data[2][each_subset_name] = AE

	end_time = time.time()
	diff = end_time - begin_time
	print(diff)


def impute_combined(dataset, log_data):

	begin_time = time.time()
	for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):  # For Each Numerical Dataset
		subsets, original = read_subsets_and_original(
			BASE_PATH, ORIGINAL_BASE_PATH, row["Abbreviation"]
		)  # Get All Subsets and Original Dataset
		dts = original.dtypes
		numerical_columns = dts[dts != "O"].index.values
		cat_columns = dts[dts == "O"].index.values
		original = original[numerical_columns.tolist() + cat_columns.tolist()]
		xoriginal = original.copy()

		subset_names = list(subsets.keys())
		# ITERATE OVER ALL SUBSETS OF A DATASET AND APPLY GRNN ON EACH ONE
		for each_subset_name in subset_names:

			# SELECTING A SUBSET
			selected_subset = subsets[each_subset_name]
			selected_subset = selected_subset[
				numerical_columns.tolist() + cat_columns.tolist()
			]

			ss = pd.concat([selected_subset, xoriginal])
			ss = pd.get_dummies(ss)  # APPLY ONE HOT ENCODING
			selected_subset = ss[0 : selected_subset.shape[0]]
			original = ss[selected_subset.shape[0] :]

			new_columns = []
			for each in original.columns:
				x_each = str(each)
				new_columns.append(x_each.replace(" ", ""))
			original.columns = new_columns

			new_columns = []
			for each in selected_subset.columns:
				x_each = str(each)
				new_columns.append(x_each.replace(" ", ""))
			selected_subset.columns = new_columns

			new_prediction = np.zeros(
				shape=original.shape
			)  # SAMPLE ARRAY TO SAVE PREDICTIONS
			new_prediction = pd.DataFrame(
				data=new_prediction, columns=selected_subset.columns
			)

			# COLUMNS ARRAY TO ITERATE
			all_cols = np.array(original.columns)
			for each in all_cols:

				# ONE COLUMN IN TEST AND OTHERS IN TRAINING
				train_cols = all_cols[all_cols != each]
				test_col = each

				# CHECKING IF THERE ARE NULL VALUES IN OUR TEST COLUMNS
				nulls = selected_subset[each].isnull()
				if len(nulls.shape) > 1:
					if nulls.shape[1] > 1:
						nulls = pd.DataFrame(nulls.values[:, 1], columns=[each])
				test_index = nulls[nulls == True].index
				train_index = nulls[nulls == False].index

				# IF THERE IS NO NULL VALUE THEN WO WONT APPLY GRNN
				if (
					test_index.shape[0] == 0
					or test_index.shape[0] / float(nulls.shape[0]) < 0.1
				):
					new_prediction[each] = original[each].copy()

				elif test_index.shape[0] == test_index.shape[0]:
					new_prediction[each] = 0

				else:
					# TRAIN GRNN ON INDEX WHERE THERE IS NO NULL AND PREDICT ON NULL VALUES
					custom_GRNN = GRNN()
					custom_GRNN.fit(
						original[train_cols].loc[train_index].values,
						original[test_col].loc[train_index].values,
						log_data
					)

					# PREDICT
					prediction_smothened = custom_GRNN.predict(
						original[train_cols].loc[test_index].values
					)

					# FILL OUR SAVING ARRAY WITH PREDICTIONS
					new_prediction[each].loc[train_index] = selected_subset[each].loc[train_index]
					if len(prediction_smothened.shape) > 1:
						if (prediction_smothened.shape[0] > 1 and prediction_smothened.shape[1] > 1):
							prediction_smothened = prediction_smothened[0, :]
							cols = new_prediction.columns
							x = new_prediction.pop(each)
							new_prediction[each] = x.values[:, 0]
							new_prediction = new_prediction[cols]

					new_prediction[each].loc[test_index] = prediction_smothened

			NRMSE = calculate_NRMS(original.values, new_prediction.values)
			log_data[0] = (log_data[0] + "Done Smoothing of : " + each_subset_name + " with NRMS : " + str(NRMSE) + "/n/n/n")
			log_data[1][each_subset_name] = NRMSE

			ccat_cols = []
			ocols = original.columns
			for each_cc in ocols:
				if each_cc not in numerical_columns.tolist():
					ccat_cols.append(each_cc)
			AE = calculate_AE_DICT(original[ccat_cols].values, new_prediction[ccat_cols].values)
			log_data[0] = (log_data[0] + "Done Smoothing of : " + each_subset_name + " with AE : " + str(AE) + "/n/n/n")
			log_data[2][each_subset_name] = AE

			if not os.path.isdir(os.path.join(IMPUTED_COMBINED,str(row['Abbreviation']))):
				os.mkdir(os.path.join(IMPUTED_COMBINED +'/'+ str(row['Abbreviation'])))
			new_prediction.to_csv(IMPUTED_COMBINED +'/'+ str(row['Abbreviation'])+ "/imputed_" + each_subset_name + ".csv", index=False)

	end_time = time.time()
	diff = end_time - begin_time
	print(diff)


