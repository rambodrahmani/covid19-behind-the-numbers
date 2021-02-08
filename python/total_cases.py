#!/usr/bin/env python

################################################################################
# Show TOP 15 countries COVID-19 total confirmed cases histogram.
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

def plot():
    # load raw data
    historical_df = pd.read_csv("historical-covid-data.csv")

    # group raw data by country aggregating on total cases
    by_location = historical_df.groupby("location", as_index = False)["total_cases"].last()
    by_location.columns = ['location', 'total_cases']

    # remove world and continents data
    by_location = by_location[~by_location.location.str.contains("World")]
    by_location = by_location[~by_location.location.str.contains("Asia")]
    by_location = by_location[~by_location.location.str.contains("Africa")]
    by_location = by_location[~by_location.location.str.contains("North America")]
    by_location = by_location[~by_location.location.str.contains("South America")]
    by_location = by_location[~by_location.location.str.contains("America")]
    by_location = by_location[~by_location.location.str.contains("Antarctica")]
    by_location = by_location[~by_location.location.str.contains("Australia")]
    by_location = by_location[~by_location.location.str.contains("Europe")]
    by_location = by_location[~by_location.location.str.contains("Europe Union")]

    # sort by total cases count
    by_location = by_location.sort_values(by=['total_cases'], ascending = False)

    # extract top 15 countries
    by_location = by_location[0:15]

    # plot top 15 countries histogram
    by_location_hist = by_location.plot.bar(x='location', y='total_cases', rot=0)
    by_location_hist.set(xlabel="Locations")
    by_location_hist.set(ylabel="Total Cases")
    by_location_hist.set(title="Confirmed COVID-19 Cases (Top 15 Worldwide)")
    by_location_hist.legend(["Total Cases"]);
    plt.xlabel('Country', fontsize=16)
    plt.ylabel('Total Confirmed Cases', fontsize=16)
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()
    
def per_milion_plot():
    # load raw data
    historical_df = pd.read_csv("historical-covid-data.csv")

    # group raw data by country aggregating on total cases
    by_location = historical_df.groupby("location", as_index = False)["total_cases_per_million"].last()
    by_location.columns = ['location', 'total_cases']

    # remove world and continents data
    by_location = by_location[~by_location.location.str.contains("World")]
    by_location = by_location[~by_location.location.str.contains("Asia")]
    by_location = by_location[~by_location.location.str.contains("Africa")]
    by_location = by_location[~by_location.location.str.contains("North America")]
    by_location = by_location[~by_location.location.str.contains("South America")]
    by_location = by_location[~by_location.location.str.contains("America")]
    by_location = by_location[~by_location.location.str.contains("Antarctica")]
    by_location = by_location[~by_location.location.str.contains("Australia")]
    by_location = by_location[~by_location.location.str.contains("Europe")]
    by_location = by_location[~by_location.location.str.contains("Europe Union")]

    # sort by total cases count
    by_location = by_location.sort_values(by=['total_cases'], ascending = False)

    # extract top 15 countries
    by_location = by_location[0:15]

    # plot top 15 countries histogram
    by_location_hist = by_location.plot.bar(x='location', y='total_cases', rot=0)
    by_location_hist.set(xlabel="Locations")
    by_location_hist.set(ylabel="Total Cases")
    by_location_hist.set(title="Confirmed COVID-19 Cases Per One Million Population (Top 15 Worldwide)")
    by_location_hist.legend(["Total Cases per one million population"]);
    plt.xlabel('Country', fontsize=16)
    plt.ylabel('Total Confirmed Cases', fontsize=16)
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()
