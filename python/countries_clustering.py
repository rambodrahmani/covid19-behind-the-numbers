#!/usr/bin/env python3

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

##
# CountriesClustering class.
#
# Contains all the functions related to computing and plotting countries
# K-Means clustering based on COVID-19 confirmed deaths.
##
class CountriesClustering:
	def euclideanDistance(self, historicalDF):
		if not historicalDF is None:
			# extract features of interest
			historicalDF = historicalDF[['date', 'location', 'new_deaths_per_million']]

			# convert dataframe to multiple time series structure
			historicalDF = historicalDF.pivot(index='location', columns='date', values='new_deaths_per_million')

			# replace NaNs generated in the conversion to time series with zeros
			historicalDF = historicalDF.fillna(0)

			# K-means time series clustering parameters
			nClusters = 3
			clustersColors = ['red', 'blue', 'green']
			clustersNames = ['Red Cluster', 'Blue Cluster', 'Green Cluster']

			# random seed used to initialize the centers
			seed = 1
			np.random.seed(seed)

			# extract training set and convert to time series dataset
			historicalDFTrain = to_time_series_dataset(historicalDF.copy())

			# time series K-means clustering Model using Euclidean distance
			print('K-Means Model using Euclidean distance')
			euclideanKM = TimeSeriesKMeans(n_clusters = nClusters, metric = "euclidean", verbose = True, random_state = seed, max_iter = 50)
			y_pred = euclideanKM.fit_predict(historicalDFTrain)
			clusters = pd.Series(data = y_pred, index = historicalDF.index)
			print('Mean Silhouette Coefficient of all samples: ' + str(silhouette_score(historicalDFTrain, clusters, metric="euclidean")))

			# plot generated clusters
			f, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, sharey = True, figsize = (5, 8))
			for yi, cl, xs in zip(range(nClusters), [2,1,0], [ax1, ax2, ax3]):
				data = historicalDF.rolling(7, axis = 1, min_periods = 1).mean().fillna(0).loc[clusters[clusters == cl].index]
				data.T.plot(legend = False, alpha = .2, color = 'black', ax = xs)
				data.mean(axis = 0).plot(linewidth = 3., color = clustersColors[cl], ax = xs)
				n = len(data)
				print('{}, N = {}'.format(clustersNames[cl], n))
				print(clusters[clusters == cl].index.to_list())
			ax1.spines['top'].set_visible(False)
			ax1.spines['right'].set_visible(False)
			ax2.spines['right'].set_visible(False)
			ax3.spines['right'].set_visible(False)
			f.subplots_adjust(hspace = 0)
			plt.ylim(0, 40)
			plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible = False)
			plt.show()

			########################################################################
			# `iplot` can only run inside an IPython Notebook.
			########################################################################
			data = dict(type = 'choropleth',
							locations = clusters.index.to_list(), 
							colorscale = [(0,"red"), (0.5,"blue"), (1,"green")],
							locationmode = 'country names',
							z = clusters.values,
							text = clusters.index.to_list(),
							colorbar = {'title':'Clusters'})
			layout = dict(title = 'Countries COVID-19 Clusters - K-means clustering Model using Euclidean distance',
							geo = dict(showframe = False, projection = {'type': 'equirectangular'}))
			fig = choromap3 = go.Figure(data = [data], layout = layout)
			fig.update_layout(autosize = False, width = 2200, height = 2200,)
			iplot(choromap3)
			########################################################################
	
	def dynamicTimeWarping(self, historicalDF):
		if not historicalDF is None:
			# extract features of interest
			historicalDF = historicalDF[['date', 'location', 'new_deaths_per_million']]

			# convert dataframe to multiple time series structure
			historicalDF = historicalDF.pivot(index='location', columns='date', values='new_deaths_per_million')

			# replace NaNs generated in the conversion to time series with zeros
			historicalDF = historicalDF.fillna(0)

			# K-means time series clustering parameters
			nClusters = 3
			clustersColors = ['red', 'blue', 'green']
			clustersNames = ['Red Cluster', 'Blue Cluster', 'Green Cluster']

			# random seed used to initialize the centers
			seed = 1
			np.random.seed(seed)

			# extract training set and convert to time series dataset
			historicalDFTrain = to_time_series_dataset(historicalDF.copy())

			# time series K-means clustering Model using Dynamic Time Warping
			print('K-Means Model using Dynamic Time Warping')
			dtwKM = TimeSeriesKMeans(n_clusters = nClusters, metric = "dtw", verbose = True, random_state = seed, max_iter = 50, max_iter_barycenter = 50)
			y_pred = dtwKM.fit_predict(historicalDFTrain)
			clusters = pd.Series(data = y_pred, index = historicalDF.index)
			print('Mean Silhouette Coefficient of all samples: ' + str(silhouette_score(historicalDFTrain, clusters, metric="dtw")))

			# plot generated clusters
			f, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, sharey = True, figsize = (5, 8))
			for yi, cl, xs in zip(range(nClusters), [2,1,0], [ax1, ax2, ax3]):
				data = historicalDF.rolling(7, axis = 1, min_periods = 1).mean().fillna(0).loc[clusters[clusters == cl].index]
				data.T.plot(legend = False, alpha = .2, color = 'black', ax = xs)
				data.mean(axis = 0).plot(linewidth = 3., color = clustersColors[cl], ax = xs)
				n = len(data)
				print('{}, N = {}'.format(clustersNames[cl], n))
				print(clusters[clusters == cl].index.to_list())
			ax1.spines['top'].set_visible(False)
			ax1.spines['right'].set_visible(False)
			ax2.spines['right'].set_visible(False)
			ax3.spines['right'].set_visible(False)
			f.subplots_adjust(hspace = 0)
			plt.ylim(0, 40)
			plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible = False)
			plt.show()

			########################################################################
			# `iplot` can only run inside an IPython Notebook.
			########################################################################
			data = dict(type = 'choropleth',
							locations = clusters.index.to_list(), 
							colorscale = [(0,"red"), (0.5,"blue"), (1,"green")],
							locationmode = 'country names',
							z = clusters.values,
							text = clusters.index.to_list(),
							colorbar = {'title':'Clusters'})
			layout = dict(title = 'Countries COVID-19 Clusters - K-means clustering Model using Dynamic Time Warping',
							geo = dict(showframe = False, projection = {'type': 'equirectangular'}))
			fig = choromap3 = go.Figure(data = [data], layout = layout)
			fig.update_layout(autosize = False, width = 2200, height = 2200,)
			iplot(choromap3)
			########################################################################
