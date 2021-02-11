#!/usr/bin/env python

################################################################################
# Plots TOP 15 countries COVID-19 total confirmed cases histogram.
################################################################################

import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

def plot():
    # load raw data
    historicalDF = util.loadHistoricalData()

    # group raw data by country aggregating on total cases
    historicalDF = historicalDF.groupby("location", as_index = False)["total_cases"].last()
    historicalDF.columns = ['location', 'total_cases']

    # sort by total cases count
    historicalDF = historicalDF.sort_values(by=['total_cases'], ascending = False)

    # extract top 15 countries by total cases count
    historicalDF = historicalDF[0:15]

    # plot top 15 countries total cases histogram
    hist = historicalDF.plot.bar(x = 'location', y = 'total_cases', rot = 0)
    hist.set(title = "Confirmed COVID-19 Cases (Top 15 Worldwide)")
    hist.legend(["Total Cases"]);
    plt.xlabel('Country', fontsize = 16)
    plt.ylabel('Total Confirmed Cases', fontsize = 16)
    plt.ticklabel_format(style = 'plain', axis = 'y')

    # show the plot
    plt.show()

def perMilionPlot():
    # load raw data
    historicalDF = util.loadHistoricalData()

    # group raw data by country aggregating on total cases
    historicalDF = historicalDF.groupby("location", as_index = False)["total_cases_per_million"].last()
    historicalDF.columns = ['location', 'total_cases_per_million']

    # sort by total cases count
    historicalDF = historicalDF.sort_values(by=['total_cases_per_million'], ascending = False)

    # extract top 15 countries by total cases count
    historicalDF = historicalDF[0:15]

    # plot top 15 countries total cases histogram
    hist = historicalDF.plot.bar(x = 'location', y = 'total_cases_per_million', rot = 0)
    hist.set(title = "Confirmed COVID-19 Cases Per One Million Population (Top 15 Worldwide)")
    hist.legend(["Total Cases"]);
    plt.xlabel('Country', fontsize = 16)
    plt.ylabel('Total Confirmed Cases', fontsize = 16)
    plt.ticklabel_format(style = 'plain', axis = 'y')

    # show the plot
    plt.show()
