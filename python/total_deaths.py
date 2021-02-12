#!/usr/bin/env python

################################################################################
# Randomly plots countries COVID-19 daily deaths (per one million population)
# time series chart.
################################################################################

import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

def perMilionPlot():
    # load historical data
    historicalDF = util.loadHistoricalData()

    # extract features of interest
    historicalDF = historicalDF[['date', 'location', 'new_deaths_per_million']]

    # convert dataframe to multiple time series structure
    historicalDF = historicalDF.pivot(index='date', columns='location', values='new_deaths_per_million')

    # replace NaNs generated in the conversion to time series with zeros
    historicalDF = historicalDF.fillna(0)

    # convert dataframe Index to DatetimeIndex
    historicalDF.index = pd.to_datetime(historicalDF.index)

    # resample data weekly to obtain a better plot
    weeklyDF = historicalDF.resample('W').sum()

    # select random countries
    weeklyDF = weeklyDF[['Belgium', 'Armenia', 'Austria', 'Bulgaria', 'France', 'United States', 'Spain', 'Germany', 'Italy', 'Brazil']];

    # plot time series starting from 2020-02-15
    timeSeriesPlot = weeklyDF.loc['2020-02-15':].plot()
    timeSeriesPlot.set(title = "COVID-19 Weekly Deaths Per One Million Population")
    timeSeriesPlot.legend(title = "");
    plt.xlabel('', fontsize = 16)
    plt.ylabel('Weekly Deaths', fontsize = 16)
    plt.show()

def allCountriesPerMilionPlot():
    # load historical data
    historicalDF = util.loadHistoricalData()

    # extract features of interest
    historicalDF = historicalDF[['date', 'location', 'new_deaths_per_million']]

    # convert dataframe to multiple time series structure
    historicalDF = historicalDF.pivot(index='date', columns='location', values='new_deaths_per_million')

    # replace NaNs generated in the conversion to time series with zeros
    historicalDF = historicalDF.fillna(0)

    # convert dataframe Index to DatetimeIndex
    historicalDF.index = pd.to_datetime(historicalDF.index)

    # resample data weekly to obtain a better plot
    weeklyDF = historicalDF.resample('W').sum()

    # plot time series starting from 2020-02-15
    timeSeriesPlot = weeklyDF.loc['2020-02-15':].plot(legend = False, color = 'Black', linewidth = 0.2)
    timeSeriesPlot.set(title = "COVID-19 Weekly Deaths Per One Million Population Worldwide")
    plt.xlabel('', fontsize = 16)
    plt.ylabel('Weekly Deaths', fontsize = 16)
    plt.show()

def clusteringPlot():
    # load historical data
    historicalDF = util.loadHistoricalData()

    # extract features of interest
    historicalDF = historicalDF[['date', 'location', 'new_deaths_per_million']]

    # convert dataframe to multiple time series structure
    historicalDF = historicalDF.pivot(index='date', columns='location', values='new_deaths_per_million')

    # replace NaNs generated in the conversion to time series with zeros
    historicalDF = historicalDF.fillna(0)

    # convert dataframe Index to DatetimeIndex
    historicalDF.index = pd.to_datetime(historicalDF.index)

    # resample data weekly to obtain a better plot
    weeklyDF = historicalDF.resample('W').sum()
    
    # K-means
    cluster = 3
    X_train = historicalDF.rolling(7, axis=1, min_periods=1).mean().fillna(0)
    colors = ['blue', 'red', 'green']
    names = ['blue cluster','red cluster','green cluster']
    seed = 1
    np.random.seed(seed)
    X_train = to_time_series_dataset(X_train.copy())

    print('COVID-19 deaths vs time curves')
    km = TimeSeriesKMeans(n_clusters=cluster, verbose=True, random_state=seed,
                            max_iter=10)

    y_pred = km.fit_predict(X_train)
    clusters = pd.Series(data=y_pred, index=historicalDF.index)

    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True,figsize=(5, 8))

    for yi,cl,xs in zip(range(cluster),[2,1,0],[ax1,ax2,ax3]):
        data = historicalDF.rolling(7, axis=1, min_periods=1).mean().fillna(0).loc[clusters[clusters == cl].index]
        data.T.plot(legend=False, alpha=.2,color='black', ax=xs)
        data.mean(axis=0).plot(linewidth=3., color=colors[cl], ax=xs)
        n = len(data)
        print('{}, N = {}'.format(names[cl], n))

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    f.subplots_adjust(hspace=0)
    plt.ylim(-0.02, 3.5)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    # plot time series starting from 2020-02-15
    timeSeriesPlot = weeklyDF.loc['2020-02-15':].plot(legend = False, color = 'Black', linewidth = 0.2)
    timeSeriesPlot.set(title = "COVID-19 Weekly Deaths Per One Million Population Worldwide")
    plt.xlabel('', fontsize = 16)
    plt.ylabel('Weekly Deaths', fontsize = 16)
    plt.show()
