import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
import os
import gc
gc.enable()
from operator import itemgetter
from scipy import optimize
from sklearn.metrics import mean_squared_error as MSE
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array
from sklearn.gaussian_process import kernels

from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

#Build GRNN Model
class GRNN(BaseEstimator, RegressorMixin):
	#Initializing all the elements
	def __init__(self, kernel='RBF', sigma=0.5, n_splits=5, calibration='warm_start', method='L-BFGS-B', bnds=(0, None), n_restarts_optimizer=0, seed = 42):
		self.kernel = kernel
		self.sigma = sigma
		self.n_splits = n_splits
		self.calibration = calibration
		self.method = method
		self.iterations = 0
		self.bnds = bnds
		self.n_restarts_optimizer = n_restarts_optimizer
		self.seed = seed
		
	def fit(self, X, y, log_data):
		
		self.X_ = X
		self.y_ = y
		bounds = self.bnds
		
		np.seterr(divide='ignore', invalid='ignore')
		
		#Initializaing and establishing the cost function
		def cost(sigma_):
			kf = KFold(n_splits= self.n_splits)
			kf.get_n_splits(self.X_)
			cv_err = []
			for train_index, validate_index in kf.split(self.X_):
				X_tr, X_val = self.X_[train_index], self.X_[validate_index]
				y_tr, y_val = self.y_[train_index], self.y_[validate_index]
				Kernel_def_= getattr(kernels, self.kernel)(length_scale=sigma_)
				K_ = Kernel_def_(X_tr, X_val)
				# If the distances are very high/low, zero-densities must be prevented:
				K_ = np.nan_to_num(K_)
				psum_ = K_.sum(axis=0).T # Cumulate denominator of the Nadaraya-Watson estimator
				psum_ = np.nan_to_num(psum_)
				y_pred_ = (np.dot(y_tr.T, K_) / psum_)
				y_pred_ = np.nan_to_num(y_pred_)
				cv_err.append(MSE(y_val, y_pred_.T))
				break
			return cv_err[0] ## Mean error over the k splits                        
		
		#Establising the optimization function
		def optimization(x0_):
			rlog = ""
			if len(self.bnds) > 1:
			  self.bnds = (self.bnds[0], )


			try:
			  if len(x0_) > 1:
				x0_ = x0_[0]
			except:
			  rlog = "x0_ is Good Enough"

			# print ("x0_", x0_)
			# print ("Bounds : ", self.bnds)
			opt = optimize.minimize(cost, x0_, method=self.method, bounds=self.bnds)
			if opt['success'] is True:
				opt_sigma = opt['x']
				opt_cv_error = opt['fun']
			else:
				opt_sigma = np.full(len(self.X_[0]), np.nan)
				opt_cv_error = np.inf
				pass
			return [opt_sigma, opt_cv_error]
		
		#Regulating and calibrating sigma
		def calibrate_sigma(self):
			x0 = np.asarray(self.sigma) # Starting guess (either user-defined or measured with warm start)
			if self.n_restarts_optimizer > 0:
				# print ("################################")    
				optima = [optimization(x0)]            
				#First optimize starting from theta specified in kernel
				optima = [optimization(x0)] 
				# # Additional runs are performed from log-uniform chosen initial bandwidths
				r_s = np.random.RandomState(self.seed)
				for iteration in range(self.n_restarts_optimizer): 
					x0_iter = np.full(len(self.X_[0]), np.around(r_s.uniform(0,1), decimals=3))
					optima.append(optimization(x0_iter))             
			elif self.n_restarts_optimizer == 0: 
				# print ("Running SAD ONE")    
				optima = [optimization(x0)]            
			else:
				raise ValueError('n_restarts_optimizer must be a positive int!')
			
			# Select sigma from the run minimizing cost
			cost_values = list(map(itemgetter(1), optima))
			self.sigma = optima[np.argmin(cost_values)][0]
			self.cv_error = np.min(cost_values) 
			return self
		
		# global log  (commented because passing log as an argument to fit function)
		if self.calibration is 'warm_start':
			log_data[0] += 'Executing warm start...' + '/n'
			self.bnds = (bounds,)           
			x0 = np.asarray(self.sigma)
			optima = [optimization(x0)]            
			cost_values = list(map(itemgetter(1), optima))
			self.sigma = optima[np.argmin(cost_values)][0]
			log_data[0] += 'Warm start concluded. The optimum isotropic sigma is ' + str(self.sigma) + '/n'
			self.sigma = np.full(len(self.X_[0]), np.around(self.sigma, decimals=3))
			self.bnds = (bounds,)*len(self.X_[0])
			# print ('Executing gradient search...')
			calibrate_sigma(self)
			log_data[0] += 'Gradient search concluded. The optimum sigma is ' + str(self.sigma) + '/n'
		elif self.calibration is 'gradient_search':
			#print ('Executing gradient search...')
			self.sigma = np.full(len(self.X_[0]), self.sigma)
			self.bnds = (bounds,)*len(self.X_[0])
			calibrate_sigma(self)
			#print('Gradient search concluded. The optimum sigma is ' + str(self.sigma))
		else:
			pass
				   
		self.is_fitted_ = True
		# Return the regressor
		return self

	#Gathering all the above and predicting the values 
	def predict(self, X):
		# Input validation
		X = check_array(X)
		
		Kernel_def= getattr(kernels, self.kernel)(length_scale=self.sigma)
		K = Kernel_def(self.X_, X)
		# If the distances are very high/low, zero-densities must be prevented:
		K = np.nan_to_num(K)
		psum = K.sum(axis=0).T # Cumulate denominator of the Nadaraya-Watson estimator
		psum = np.nan_to_num(psum)
		return np.nan_to_num((np.dot(self.y_.T, K) / psum))
