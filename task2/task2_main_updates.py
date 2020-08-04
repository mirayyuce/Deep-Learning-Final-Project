import os
import sys
import glob
import json
import itertools
import math
import random as ra

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, LSTM
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import SGD
from keras.optimizers import Adam

from load_prepare_data import *
from rnn import *
from plotting import *


if __name__ == "__main__":


	decaying_lrs = [[1e-4, 1e-6], [1e-4, 1e-7], [1e-2, 1e-6], [1e-3, 1e-6]]
	alphas = [1e-7, 1e-6, 1e-5, 1e-4]
	pred_time = [5, 10, 20, 30]
	#pred_time = [5]
	train_time = [5, 10, 20]
	#train_time = [5]
	models = 2
	num_epochs = 1000


	# randomness
	seed = 7
	np.random.seed(seed)
	kfold = KFold(n_splits = 3, shuffle = True, random_state = seed)

	for l in train_time:
		if not os.path.exists("./Plots/Train/" + str(l)):
			os.makedirs("./Plots/Train/" + str(l))
		overall_mse = []
		data, targets, targets_original = prepare_data()
		data_scaled = preprocess_data(data)

		targets_selected = y_select(targets_original, l)

		# given time steps, y is prepared 
		targets_selected_for_input = y_select_targets(targets_selected, l)
			
		# prepare data for cv
		rnn_input = prepare_rnn_input(data_scaled, targets_selected_for_input, l).reshape(-1, l-1, 1 + data.shape[1])

		rnn_targets = prepare_rnn_targets(targets_selected, l).reshape(-1, l-1,1)

		best_weights = []
		lr_used = []
		alpha_used = []
		model1_mse = []
		model2_mse = []
		for m in range(models):
			if not os.path.exists("./Plots/Train/" + str(l) + '/Model ' + str(m)):
				os.makedirs("./Plots/Train/" + str(l) + "/Model " + str(m))
			print("Training " + str(l) + " epochs, Model " + str(m+1))
			# randomize hyperparams
			idx = np.random.randint(0, len(decaying_lrs))
			learningrate = decaying_lrs[idx]
			alpha = np.random.choice(alphas)
			lr_used.append(learningrate)
			alpha_used.append(alpha)
			# get data
			
			split = 1
			overall_mse_split1 = []
			overall_mse_split2 = []
			overall_mse_split3 = []

			predictions_split1 = []
			targets_split1 = []
			predictions_split2 = []
			targets_split2 = []
			predictions_split3 = []
			targets_split3 = []

			for train, valid in kfold.split(rnn_input, rnn_targets):
				if not os.path.exists("./Plots/Train/" + str(l) + "/Model " + str(m) + '/Split '+ str(split)):
					os.makedirs("./Plots/Train/" + str(l) + "/Model " + str(m) + '/Split '+ str(split))
				model = rnn(learning_rate=learningrate, num_epochs = num_epochs, alpha = alpha)
				split_loss = []
				split_score = []
				split_mse = []

				all_split_score = []
				all_split_mse = []
				all_split_loss=[]
				
				print("Split " + str(split))
				
				for i in range (num_epochs):
					for x, y in zip(rnn_input[train],rnn_targets[train]):
						split_loss.append(model.fit(x.reshape(-1, l-1, 1 + data.shape[1]), y.reshape(-1, l-1,1), epochs = 1, verbose=0).history['loss'])

				all_split_loss.append(np.mean(split_loss))

				for x, y in zip(rnn_input[valid],rnn_targets[valid]):
					split_score.append(model.evaluate(x.reshape(-1, l-1, 1 + data.shape[1]), y.reshape(-1, l-1,1), verbose=0) * 100)
					preds = model.predict(x.reshape(-1, l-1, 1 + data.shape[1]))
				
					split_mse.append(mean_squared_error(preds[0], y))

				"""all_split_score.append(np.mean(split_score))
					all_split_mse.append(np.mean(split_mse))"""
				print("MSE for this split = " + str(np.mean(split_mse)))


				"""overall_score = np.mean(all_split_score)
					overall_mse.append(np.mean(all_split_mse))
					overall_loss = np.mean(all_split_loss)
					best_weights.append(model.get_weights())
	
					best = np.argmin(overall_mse)
					#best_lr = lr_used[best]
					#best_alpha = alpha_used[best]"""

				new_model = rnn_stateful(learning_rate=learningrate, num_epochs = num_epochs, alpha = alpha)
				new_model.set_weights(model.get_weights())

				for x in rnn_input[train]:
					preds = new_model.predict(x.reshape(-1, l-1, 1 + data.shape[1]))
				
				# Prediction using best model
				overall_mse_test = []
				for s in pred_time:
					print("Start prediction for train " + str(l) + " and test " + str(s))
					if not os.path.exists("./Plots/Train/" + str(l) + "/Model " + str(m) + '/Split '+ str(split) + '/Test ' + str(s)):
						os.makedirs("./Plots/Train/" + str(l) + "/Model " + str(m) + '/Split '+ str(split) + '/Test ' + str(s))

					targets_selected_prediction = y_select(targets_original, s)

					# given time steps, y is prepared 
					targets_selected_for_input_prediction = y_select_targets(targets_selected_prediction, s)
						
					# prepare data for cv
					rnn_input_prediction = prepare_rnn_input(data_scaled, targets_selected_for_input, s).reshape(-1, s-1, 1 + data.shape[1])
					

					all_preds = []
					targets_of_split = []

					#for _, valid in kfold.split(rnn_input, rnn_targets):

					for sample_x, sample_y in zip(rnn_input_prediction[valid], targets_selected_for_input_prediction[valid]):

						new_model.reset_states()
						prediction = 0
						preds = []
						

						#for x in sample_x:
						#	prediction = new_model.predict(x.reshape(-1, 1, 1 + data.shape[1]), batch_size = 1)
						prediction = new_model.predict(sample_x.reshape(-1, s-1, 1 + data.shape[1]), batch_size = 1)


						for i in range(s,41):
							x = sample_x[0][0:-1]
			
							x_pred = np.c_[np.array([x]),prediction[0]]
					
							prediction = new_model.predict(x_pred.reshape(-1, 1, 1 + data.shape[1]), batch_size = 1)
						
							preds.append(prediction[0])
							
						all_preds.append(np.array(preds).reshape(41-s,))
						targets_of_split.append(sample_y)

					predictions = np.c_[np.array(targets_of_split).reshape(len(valid),s-1), all_preds]
					
					params = [learningrate, alpha]

					thefile = open(os.path.join("./Plots/Train/" + str(l) + "/Model " + str(m) + '/', 'Predictions - Test_' + str(s) + '_split_'+ str(split)+ '.txt'), 'w')
					
					for item in predictions:
					  thefile.write("%s\n" % item)

					mse_test = []

					"""for x, y in zip(predictions[:], targets_original[valid]):
						mse_test.append(mean_squared_error(x[s-1:-1], y[s-1:-1])"""
					splits_predictions = []
					for x, y in zip (predictions, targets[valid]):
						mse_test.append(mean_squared_error([x[-1]], y))
						splits_predictions.append(x[-1])

					if (split ==1):
						if (s == 5):
							mse_test_5 = np.asarray(mse_test)
							overall_mse_split1.append(mse_test_5)
							predictions_split1.append(splits_predictions)
							targets_split1.append(targets[valid])
						if (s ==10):
							mse_test_10 = np.asarray(mse_test)
							overall_mse_split1.append(mse_test_10)
							predictions_split1.append(splits_predictions)
							targets_split1.append(targets[valid])
						if (s ==20):
							mse_test_20 = np.asarray(mse_test)
							overall_mse_split1.append(mse_test_20)
							predictions_split1.append(splits_predictions)
							targets_split1.append(targets[valid])
						if (s ==30):
							mse_test_30 = np.asarray(mse_test)
							overall_mse_split1.append(mse_test_30)
							predictions_split1.append(splits_predictions)
							targets_split1.append(targets[valid])
					if (split== 2):
						if (s ==5):
							mse_test_5 = np.asarray(mse_test)
							overall_mse_split2.append(mse_test_5)
							predictions_split2.append(splits_predictions)
							targets_split2.append(targets[valid])
						if (s ==10):
							mse_test_10 = np.asarray(mse_test)
							overall_mse_split2.append(mse_test_10)
							predictions_split2.append(splits_predictions)
							targets_split2.append(targets[valid])
						if (s== 20):
							mse_test_20 = np.asarray(mse_test)
							overall_mse_split2.append(mse_test_20)
							predictions_split2.append(splits_predictions)
							targets_split2.append(targets[valid])
						if (s ==30):
							mse_test_30 = np.asarray(mse_test)
							overall_mse_split2.append(mse_test_30)
							predictions_split2.append(splits_predictions)
							targets_split2.append(targets[valid])
					if (split ==3):
						if (s ==5):
							mse_test_5 = np.asarray(mse_test)
							overall_mse_split3.append(mse_test_5)
							predictions_split3.append(splits_predictions)
							targets_split3.append(targets[valid])
						if (s== 10):
							mse_test_10 = np.asarray(mse_test)
							overall_mse_split3.append(mse_test_10)
							predictions_split3.append(splits_predictions)
							targets_split3.append(targets[valid])
						if (s ==20):
							mse_test_20 = np.asarray(mse_test)
							overall_mse_split3.append(mse_test_20)
							predictions_split3.append(splits_predictions)
							targets_split3.append(targets[valid])
						if (s== 30):
							mse_test_30 = np.asarray(mse_test)
							overall_mse_split3.append(mse_test_30)
							predictions_split3.append(splits_predictions)
							targets_split3.append(targets[valid])


					thefile = open(os.path.join("./Plots/Train/" + str(l) + "/Model " + str(m) + '/', 'MSE - Test_' + str(s) +'_split_'+ str(split)+ '.txt'), 'w')
					for item in mse_test:
					  thefile.write("%s\n" % item)
					
					#plot_all_learning_curves(predictions, targets_original, params, l, s, split_mse[split-1], split, m)
				#plot_box_plots(np.asarray(overall_mse_test), params, l, s,split)
				

				thefile = open(os.path.join("./Plots/Train/" + str(l) + "/Model " + str(m), 'targets_split'+ str(split) + '.txt'), 'w')
				for item in targets_original:
				  thefile.write("%s\n" % item)
				split+=1

			plot_network_vs_true_scatter(predictions_split1, predictions_split2, predictions_split3, targets_split1, targets_split2, targets_split3, np.asarray(overall_mse_split1), np.asarray(overall_mse_split2), np.asarray(overall_mse_split3), params, l, pred_time ,split, m)
			plot_box_plots(np.asarray(overall_mse_split1), np.asarray(overall_mse_split2), np.asarray(overall_mse_split3), params, l, pred_time ,split, m)
			if (m == 0):
				model1_mse.append(np.asarray(overall_mse_split1))
				model1_mse.append(np.asarray(overall_mse_split2))
				model1_mse.append(np.asarray(overall_mse_split3))
			if (m == 1):
				model2_mse.append(np.asarray(overall_mse_split1))
				model2_mse.append(np.asarray(overall_mse_split2))
				model2_mse.append(np.asarray(overall_mse_split3))
		plot_box_plots_all(model1_mse, model2_mse, l)
