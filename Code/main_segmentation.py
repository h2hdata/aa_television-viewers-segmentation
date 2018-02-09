# -*- coding: utf-8 -*-

#POC Code  :  TV VIEWERS SEGMENTATION  :  H2H DATA
#This code is for  POC  of  tv  viewers  segmentation which includes
#processes   starting   from   ingestion,   processing  and  results.
#It is meant for DEMO purposes only and not to be used in production.
#Copyright@ H2H DATA

#The entire prcess occurs in nine stages-
# 1. DATA INGESTION
# 2. DATA ANALYSIS 
# 3. DATA MUNGING
# 4. DATA EXPLORATION
# 5. DATA MODELING
# 6. HYPER-PARAMETERS OPTIMIZATION
# 7. PREDICTION
# 8. VISUAL ANALYSIS
# 9. RESULTS

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sc
import pylab as pl

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import randint as sp_randint
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

PATH_TO_DATA = '../Data/tvaudiencedataset.csv'
NAME_COLUMNS = ['channel ID','slot','week','genre ID','subGenre ID','user ID','program ID','event ID','duration']
COLS_REMAINING = ['channel ID','slot','week','genre ID','subGenre ID','user ID','duration']
PARAM_GRID_KMEANS = {'n_clusters': sp_randint(7,10), 'max_iter':sp_randint(200,500)}
PARAM_GRID_HAC = {'n_clusters': sp_randint(7,10)}
THRESHOLD_OUTLIER = 6
N_ITER_RS = 50
N_COMPONENTS = 2
RUN_ANALYSIS = True


'''
  DATA INGESTION
-------------------

 Data is ingested in form of pandas dataframe

 Data  to be used   :   tvaudiencedataset.csv

 Data Information : 
	  Size : 1.1 GB
	  Features Information :
		channel ID: channel id from 1 to 217.
		slot: hour inside the week relative to the start  of 
			  the view, from 1 to 24*7 = 168.
		week: week from 1 to 19. Weeks 14 and 19 should  not 
			  be used because they contain errors.
		genre ID: it is the id of the genre, form 1 to 8. 
			  Genre/subgenre mapping is given in a csv  file
		subGenre ID: it is the id of the subgenre, from 1 to
					 114.
		user ID: it is the id of the user.
		program ID: it is  the  id of  the program. The same 
					program   can   occur   multiple   times 
					(e.g. a tv show).
		event ID: it is the id of the particular instance of
				  a program.  It  is unique, but it can span 
				  multiple slots.
		duration: duration of the view. 
'''

def Ingest_data():
	"""
	   Read the data in form of a pandas dataframe.

	   Returns:
	   -------
	   data : (M,9) pd.Dataframe

	"""
	data = pd.read_csv( PATH_TO_DATA,
						names= NAME_COLUMNS,
						nrows=5000)
	return data

df = Ingest_data()

'''
  DATA ANALYSIS
-----------------

 Here,  data is  analysed and visualised to look for patterns 
 and   anamolies   using   graphs.  bar  graphs,   histograms,  
 scatter plots, etc. will be used

'''

def _scatter_plot(x, y):
	"""
	   Function to create a scatter plot of one column versus
	   another.

	   Parameters:
	   ----------
	   df : pd.Dataframe
			Input Data
	   x : string
		   Column name	   
	   y : string
		   Column name whose column to plot against 'x'

	   Returns:
	   --------
	   scatter plot between x and y.

	"""
	ax = df.plot(x=x, y=y, kind='scatter')
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	ax.set_title(x+" versus "+y)
	plt.draw()
	plt.pause(0.01)
	raw_input("Press enter to continue")

def _histogram_plot(x, y):
	"""
	   Function  to  create  a  histogram  plot of one column 
	   versus another.

	   Parameters:
	   ----------
	   df : pd.Dataframe
			Input Data
	   x : string
		   Column name	   
	   y : string
		   Column name whose column to plot against 'x'

	   Returns:
	   --------
	   histogram plot between x and y.

	"""
	ax = df.plot(x=x, y=y, kind='hist')
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	ax.set_title(x+" versus "+y)
	plt.draw()
	plt.pause(0.01)
	raw_input("Press enter to continue")

def _box_plot(x, y):
	"""
	   Function  to  create  a  box plot of one column versus 
	   another.

	   Parameters:
	   ----------
	   df : pd.Dataframe
			Input Data
	   x : string
		   Column name	   
	   y : string
		   Column name whose column to plot against 'x'

	   Returns:
	   --------
	   box plot between x and y.

	"""
	ax = df.plot(x=x, y=y, kind='box')
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	ax.set_title(x+" versus "+y)
	plt.draw()
	plt.pause(0.01)
	raw_input("Press enter to continue")

def _bar_chart(x):
	"""
	   Function  to  create  a bar chart of one column versus 
	   another.

	   Parameters:
	   ----------
	   df : pd.Dataframe
			Input Data
	   x : string
		   Column name	   

	   Returns:
	   --------
	   bar chart.

	"""
	if x is not None:
		ax = df.groupby(x).count().plot(kind='bar')
		ax.set_xlabel(x)
		ax.set_title(x)
		plt.draw()
		plt.pause(0.01)
		raw_input("Press enter to continue")
	else:
		ax = df.plot(kind='bar')
		plt.draw()
		plt.pause(0.01)
		raw_input("Press enter to continue")


def Analysis_data():
	"""
	   This function analyzes the data and looks for patterns
	   to help understand the data more clearly.

	   Returns:
	   --------
	   Charts, graphs for various Features.

	"""
	_scatter_plot('user ID','channel ID')
	_histogram_plot('user ID','genre ID')
	_scatter_plot('user ID','duration')
	_histogram_plot('user ID','slot')
	_histogram_plot('channel ID','slot')
	_histogram_plot('genre ID','duration')
	_box_plot('user ID','program ID')
	_bar_chart('user ID')
	_bar_chart('channel ID')
	_bar_chart('slot')
	_bar_chart('genre ID')
	_bar_chart('duration')
	_bar_chart('subGenre ID')
	_bar_chart('week')

# Analysis_data()

'''
 DATA MUNGING
---------------
 On the basis of information of data provided and on based on
 Data Analysis, data needs to be  rearranged, reorganised and 
 made  into a  proper structure to  make it  ready  for  Data
 modelling process.

'''

def Munging_data():
	"""
	   Rearrange and reorganise the  data  according  to  the 
	   problems needed to solve  so  the  modelling occurs on
	   proper datasets.

	   Returns:
	   --------
	   df : pd.Dataframe
			Dataset for further exploration

	"""
	df_ = df[~df['week'].isin([14,19])]
	return df_

df = Munging_data()


'''
 DATA EXPLORATION
-------------------
 The data might contain outliers, missing values and  various
 other  things  which  might  cause  problems in modelling or
 might lead  to  wrong  results.  These  anamolies need to be 
 corrected  and  data  must be  cleaned before  passing it to
 modelling.

'''

def _outlier_detection(points):
	"""
	   Function  to  find  outliers  in  data and change them
	   to Nan values for further treatment.

	   Parameters:
	   ----------
	   points : (M,1) numpy array 
				A column of data

	   Returns:
	   --------
	   arr : (M,1) numpy array
			 A column having True False values 

	"""
	if len(points.shape) == 1:
		points = points[:,None]
	median = np.median(points, axis=0)
	diff = np.sum((points - median)**2, axis=-1)
	diff = np.sqrt(diff)
	med_abs_deviation = np.median(diff)
	modified_z_score = 0.6745 * diff / med_abs_deviation
	return modified_z_score > THRESHOLD_OUTLIER

def _missing_value_treatment(data):
	"""
	   Function  to  find  missing  values and fill them with 
	   mode of the column.

	   Parameters:
	   ----------
	   data : (M,N) numpy array 
			   data

	   Returns:
	   --------
	   arr : (M,N) numpy array
			 Data with filled missing values

	"""

	mode_values = sc.mode(data, nan_policy='omit')[0]
	inds = np.where(np.isnan(data))
	data[inds] = np.take(mode_values, inds[1])
	return data

def Exploration_data():
	"""
	   Function to apply data exploration techniques.

	   Returns:
	   --------
	   df_explored : (M,N) pandas DataFrame
					 Data 

	"""
	df_ = df.drop('program ID',axis=1).drop('event ID',axis=1).astype(float)
	arr = np.array(df_)
	outliers = np.apply_along_axis(_outlier_detection, 0, arr)
	arr[outliers] = np.nan
	arr = _missing_value_treatment(arr)
	df_explored = pd.concat([pd.DataFrame(arr, columns=COLS_REMAINING), df[['program ID', 'event ID']]], axis=1)
	return df_explored

df = Exploration_data()


'''
 DATA MODELING
----------------
 The data finally preprocessed and cleaned  is  now  used for 
 clustering and obtaining  the results. We  use K-Means model
 and HAC model for  prediction  and trace as our   evaluation
 metric 

'''

def _kmeans_model():
	"""
	   Function to initialize a KMEANS model.

	   Returns:
	   --------
	   clf : K-means model

	"""

	clf = KMeans()
	return clf

def _HAC_model():
	"""
	   Function to initialize a HAC model.

	   Returns:
	   --------
	   clf : HAC model

	"""

	clf = AgglomerativeClustering()
	return clf


'''
 HYPER-PARAMETERS OPTIMIZATION
--------------------
 The  hyperparameters  of a model need to be optimized to get 
 best possible score of prediction. For  this purpose, we use
 Random Grid Search.

'''

def _parameter_selection():
	"""
	   Function to select a random number of  hyperparameters
	   combinations   from  entire  grid  of  hyperparameters

	   Returns:
	   --------
	   rounded_list_kmeans : List of dictionaries
			 				 hyperparameters combinations for
			 				 Kmeans model
	   rounded_list_HAC : List of dictionaries
			 				 hyperparameters combinations for
			 				 HAC model

	"""

	param_list_kmeans = list(ParameterSampler(PARAM_GRID_KMEANS, n_iter=N_ITER_RS))
	rounded_list_kmeans = [dict((k, round(v, 6)) for (k, v) in d.items())
					for d in param_list_kmeans]
	rounded_list_kmeans = [dict(t) for t in set([tuple(d.items()) for d in rounded_list_kmeans])]

	param_list_HAC = list(ParameterSampler(PARAM_GRID_HAC, n_iter=N_ITER_RS))
	rounded_list_HAC = [dict((k, round(v, 6)) for (k, v) in d.items())
					for d in param_list_HAC]
	rounded_list_HAC = [dict(t) for t in set([tuple(d.items()) for d in rounded_list_HAC])]

	return rounded_list_kmeans,rounded_list_HAC

def _random_search(X, clf, param_distributions):
	"""
	   Function to fit the model by selecting params from the
	   selected parameters combinations and saving the metric 
	   value.

	   Parameters:
	   ----------
	   X : (M,N) numpy array 
			   data
	   clf : model object 
			 initialized model 
	   param_distributions : List of dictionaries
	   						 paramters combinations	 

	   Returns:
	   --------
	   best : tuple
			  best metric value and corresponding parameters

	"""

	out = (0,0)
	for params in param_distributions:
		params = {k:int(v) for k,v in params.iteritems()}
		clf.set_params(**params)
		cluster_labels = clf.fit_predict(X)
		metric_value = silhouette_score(X,cluster_labels)
		if metric_value > out[0]:
			out = (metric_value,params)

	best = out
	return best

def Optimization_model():
	"""
	   Function to optimize  the model and select best params
	   combination from grid. It runs  both  KMEANS  and  HAC

	   Returns:
	   --------
	   best_model, name : tuple, string
			  			  best  metric  value, parameters and
			  			  model name

	"""
	param_kmeans, param_HAC = _parameter_selection()
	arr = np.array(df.astype(int))
	
	clf_kmeans = _kmeans_model()
	best_kmeans = _random_search(arr, clf_kmeans, param_kmeans)

	clf_HAC = _HAC_model()
	best_HAC = _random_search(arr, clf_HAC, param_HAC)

	if best_HAC[0] > best_kmeans[0]:
		return best_HAC, 'HAC'

	return best_kmeans, 'KMEANS'

(score, best_params), model = Optimization_model()


'''
 PREDICTION
-------------
 Here, we finally predict the  cluster labels  from our  best
 parameters model obtained from optimization.

'''

def Prediction():
	"""
	   Function to fit the model by best parameters  obtained
	   and predict the cluster labels.

	   Returns:
	   --------
	   df_ : pandas Dataframe
			 data with predicted column

	"""	
	if model == 'HAC':
		clf = _HAC_model()
		clf.set_params(**best_params)
		labels = clf.fit_predict(np.array(df.astype(int)))
		df_ = pd.concat([df,pd.DataFrame(labels,columns=['Cluster'])], axis=1)
		return df_

	clf = _kmeans_model()
	clf.set_params(**best_params)
	labels = clf.fit_predict(np.array(df.astype(int)))
	df_ = pd.concat([df,pd.DataFrame(labels,columns=['Cluster'])], axis=1)
	return df_

df = Prediction()

'''
 VISUAL ANALYSIS
------------------
 After predicting the clusters in data,  we now visualize the 
 clusters  using  a  scatter  plot. To plot, we will need  to
 apply  the  dimensionality  reduction  on  data  to  get two
 components from data.

'''
def Visual_analysis():
	"""
	   Function to analyze the clusters  formed  visually  by
	   plotting scatter plot. PCA will  be  performed  on the 
	   data   to  get  two  components  from   all   Features

	   Returns:
	   --------
	   scatter plot

	"""

	arr = df.drop('Cluster',axis=1).drop('event ID',axis=1)
	pca = PCA(n_components=N_COMPONENTS).fit(arr)
	pca_2d = pca.transform(arr)
	pl.figure(model+' with '+str(best_params['n_clusters'])+' clusters with a score '+str(score))
	pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=df['Cluster'])
	pl.draw()
	pl.pause(0.01)
	raw_input("PRESS ENTER TO END")

Visual_analysis()

'''
 RESULTS
---------
 Get the predicted labels for records.

'''

print df
df[['event ID','Cluster']].to_csv('/root/Documents/POCs/TV Viewer Segmentation/Output/Output_clusters.csv',index=False)