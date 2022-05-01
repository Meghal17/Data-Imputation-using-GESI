# Data-Imputation-using-GESI

This is a Python implementation of Generalized regression neural network Ensemble for Single Imputation ([GESI](https://www.sciencedirect.com/science/article/pii/S0925231210003188)) in Python.
Real Data is messy and more often than not a considerable fraction of feature values are missing for various reasons. Such missing values induce a bias in the performance of the modelling task at hand. One of the ways to deal with missing values in datasets is to estimate those missing values and then use the imputed data for the modelling task. GESI is one such algorithm that estimates the values of missing data using the existing values in the dataset. 

## Scope
Data imputation is performed on several datasets provided for an academic project. Imputation can be performed on datasets containing numerical and categorical feature values. The algorithm's performance is evaluated by calculating the NRMS values (for numerical data) and AE values (for categorical data).



## Files
1. GESI.py: Run this main file to perform imputations
2. GRNN.py: Implements the GRNN class to be used in GESI
3. Helpers.py: Contains helper functions to implements SAGA Feature selection, Calculate evaluation metrics, read, and process data.
4. Imputation.py: Contains 3 imputation functions:
   1. `impute_numerical()` : function to impute datasets where features are represented by numerical values only
   2. `impute_categorical()`: function to impute datasets where features are represented by categorical values only
   3. `impute_combined()`: function to impute datasets where some features are represented by categorical values and others by numerical values.
5. Config.py: contains info about various paths to read data and store results.


• References
1. Iffat A. Gheyas, Leslie S. Smith, A neural network-based framework for the reconstruction of incomplete data sets, Neurocomputing, Volume 73, Issues 16–18, 2010, Pages 3039-3065,
ISSN 0925-2312, https://doi.org/10.1016/j.neucom.2010.06.021. Available: (https://www.sciencedirect.com/science/article/pii/S0925231210003188)
