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
