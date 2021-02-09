#!/usr/bin/env python

################################################################################
# Randomly plots countries COVID-19 daily deaths (per one million population)
# time series chart.
################################################################################

from tslearn.clustering import TimeSeriesKMeans

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

    # extract top 50 countries: these will be filtered again later
    by_location = by_location[0:50]
    top_countries = by_location['location'].values

    # extract features of interest
    historical_df = historical_df[['date', 'location', 'new_deaths_per_million']]

    # find countries with too many NA values
    na_count = historical_df.groupby("location")["new_deaths_per_million"].apply(lambda x: (x<= 0).sum()).reset_index(name='count')
    countries_missing_data = na_count[na_count['count'] > 150].location.values

    # preprocessing: set values <= 0 to NA
    preprocessed_historical_df = historical_df
    preprocessed_historical_df[preprocessed_historical_df <= 0] = np.nan

    # replace missing values with the mean
    imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    preprocessed_historical_df[['new_deaths_per_million']] = pd.DataFrame(imp_mean.fit_transform(preprocessed_historical_df[['new_deaths_per_million']]))
    preprocessed_historical_df.columns = historical_df.columns

    # select top countries country name and daily death count
    preprocessed_historical_df = preprocessed_historical_df[preprocessed_historical_df.location.isin(top_countries)]
    preprocessed_historical_df = preprocessed_historical_df[~preprocessed_historical_df.location.isin(countries_missing_data)]
    preprocessed_historical_df = preprocessed_historical_df[['date', 'location', 'new_deaths_per_million']]

    # convert dataframe to multiple time series structure
    preprocessed_historical_df = preprocessed_historical_df.pivot(index='date', columns='location', values='new_deaths_per_million')

    # replace NaNs generated in the conversion to timeseries with zeros
    preprocessed_historical_df = preprocessed_historical_df.fillna(0)

    # convert dataframe index to time index
    preprocessed_historical_df.index = pd.to_datetime(preprocessed_historical_df.index)

    # resample data to obtain a better plot
    data_resampled = preprocessed_historical_df.resample('W').sum()

    # select top 5 countries
    data_resampled = data_resampled[['Belgium', 'Armenia', 'Austria', 'Bulgaria', 'France', 'United States', 'Spain', 'Germany', 'Italy', 'Brazil']];

    # plot time series
    time_series_plot = data_resampled.loc['2020-02-15':].plot()
    time_series_plot.set(title = "COVID-19 Weekly Deaths Per One Million Population")
    time_series_plot.legend(title = "");
    plt.xlabel('', fontsize = 16)
    plt.ylabel('Weekly Deaths', fontsize = 16)
    plt.show()

def plot_all_countries():
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

    # extract countries names without continents
    only_countries = by_location['location'].values

    # extract features of interest
    historical_df = historical_df[['date', 'location', 'new_deaths_per_million']]

    # find countries with too many NA values
    na_count = historical_df.groupby("location")["new_deaths_per_million"].apply(lambda x: (x<= 0).sum()).reset_index(name='count')
    countries_missing_data = na_count[na_count['count'] > 150].location.values

    # preprocessing: set values <= 0 to NA
    preprocessed_historical_df = historical_df
    preprocessed_historical_df[preprocessed_historical_df <= 0] = np.nan

    # replace missing values with the mean
    imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    preprocessed_historical_df[['new_deaths_per_million']] = pd.DataFrame(imp_mean.fit_transform(preprocessed_historical_df[['new_deaths_per_million']]))
    preprocessed_historical_df.columns = historical_df.columns

    # select top 15 countries country name and daily death count
    preprocessed_historical_df = preprocessed_historical_df[preprocessed_historical_df.location.isin(only_countries)]
    preprocessed_historical_df = preprocessed_historical_df[~preprocessed_historical_df.location.isin(countries_missing_data)]
    preprocessed_historical_df = preprocessed_historical_df[['date', 'location', 'new_deaths_per_million']]

    # convert dataframe to multiple time series structure
    preprocessed_historical_df = preprocessed_historical_df.pivot(index = 'date', columns = 'location', values = 'new_deaths_per_million')

    # replace NaNs generated in the conversion to timeseries with zeros
    preprocessed_historical_df = preprocessed_historical_df.fillna(0)

    # convert dataframe index to time index
    preprocessed_historical_df.index = pd.to_datetime(preprocessed_historical_df.index)

    # resample data to obtain a better plot
    data_resampled = preprocessed_historical_df.resample('W').sum()

    model = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10, random_state=seed)
    model.fit(data_resampled)

    # plot time series
    time_series_plot = data_resampled.loc['2020-02-15':].plot(legend = False, color = 'Black', linewidth = 0.2)
    time_series_plot.set(title = "COVID-19 Weekly Deaths Per One Million Population Worldwide")
    plt.xlabel('', fontsize = 16)
    plt.ylabel('Weekly Deaths', fontsize = 16)
    plt.show()
