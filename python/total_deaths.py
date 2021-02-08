#!/usr/bin/env python

################################################################################
# Show TOP 15 countries COVID-19 daily deaths (per one million population) time
# series chart.
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

def plot():
    # load raw data
    historical_df = pd.read_csv("historical-covid-data.csv")

    # group raw data by country aggregating on total deaths
    by_location = historical_df.groupby("location", as_index = False)["total_deaths_per_million"].last()
    by_location.columns = ['location', 'total_deaths_per_million']

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

    # sort by total deaths count
    by_location = by_location.sort_values(by=['total_deaths_per_million'], ascending = False)

    # extract top 50 countries
    by_location = by_location[0:50]
    top_countries = by_location['location'].values

    # extract features of interest
    historical_df = historical_df[['date', 'location', 'new_deaths_per_million']]

    # find countries with too many NA values
    na_count = historical_df.groupby("location")["new_deaths_per_million"].apply(lambda x: (x<= 0).sum()).reset_index(name='count')
    countries_missing_data = na_count[na_count['count'] > 170].location.values

    # preprocessing: set values <= 0 to NA
    preprocessed_historical_df = historical_df
    preprocessed_historical_df[preprocessed_historical_df <= 0] = np.nan
    
    # replace missing values with the mean
    imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    preprocessed_historical_df[['new_deaths_per_million']] = pd.DataFrame(imp_mean.fit_transform(preprocessed_historical_df[['new_deaths_per_million']]))
    preprocessed_historical_df.columns = historical_df.columns
    
    # select top 15 countries country name and daily death count
    preprocessed_historical_df = preprocessed_historical_df[preprocessed_historical_df.location.isin(top_countries)]
    preprocessed_historical_df = preprocessed_historical_df[~preprocessed_historical_df.location.isin(countries_missing_data)]
    preprocessed_historical_df = preprocessed_historical_df[['date', 'location', 'new_deaths_per_million']]

    # convert dataframe to multiple time series structure
    preprocessed_historical_df = preprocessed_historical_df.pivot(index='date', columns='location', values='new_deaths_per_million')

    # replace NaNs generated in the conversion with zeros
    preprocessed_historical_df = preprocessed_historical_df.fillna(0)
    
    # plot time series
    preprocessed_historical_df.loc['2020-03-01':].plot()
    preprocessed_historical_df[['Italy', 'Germany', 'Spain', 'Brazil', 'United States']].loc['2020-03-01':].plot()
    plt.show()
