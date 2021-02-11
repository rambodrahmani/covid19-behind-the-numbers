#!/usr/bin/env python

################################################################################
# Utilities functions.
################################################################################

import os.path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

HISTORICAL_DATASET_PATH = "historical-covid-data.csv"
PREPROCESSED_HISTORICAL_DATASET_PATH = "resampled-historical-covid-data.csv"

##
# Loads and preprocesses the historical dataframe.
##
def loadHistoricalData():
    # check if the csv file exists
    if os.path.isfile(HISTORICAL_DATASET_PATH):
        if os.path.isfile(PREPROCESSED_HISTORICAL_DATASET_PATH):
            historicalDF = pd.read_csv(PREPROCESSED_HISTORICAL_DATASET_PATH)
            return historicalDF
        else:
            # read csv file
            historicalDF = pd.read_csv(HISTORICAL_DATASET_PATH)
            
            # remove world and continents data
            historicalDF = historicalDF[~historicalDF.location.str.contains("Asia")]
            historicalDF = historicalDF[~historicalDF.location.str.contains("World")]
            historicalDF = historicalDF[~historicalDF.location.str.contains("Africa")]
            historicalDF = historicalDF[~historicalDF.location.str.contains("Europe")]
            historicalDF = historicalDF[~historicalDF.location.str.contains("America")]
            historicalDF = historicalDF[~historicalDF.location.str.contains("Australia")]
            historicalDF = historicalDF[~historicalDF.location.str.contains("Antarctica")]
            historicalDF = historicalDF[~historicalDF.location.str.contains("Europe Union")]
            historicalDF = historicalDF[~historicalDF.location.str.contains("North America")]
            historicalDF = historicalDF[~historicalDF.location.str.contains("South America")]

            # only select columns of interest for the data mining process
            historicalDF = historicalDF[['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_cases_per_million', 'new_cases_per_million', 'total_deaths_per_million', 'new_deaths_per_million']]

            # preprocessing: replace negative values with NaN
            historicalDF[historicalDF <= 0] = np.nan

            # extract location column values
            byLocation = historicalDF.groupby("location", as_index = False)["total_cases"].last()
            byLocation.columns = ['location', 'total_cases']
            locations = byLocation['location'].values

            # empty dataframe with the same columns
            imputedDF = pd.DataFrame(columns = historicalDF.columns)

            # preprocessing: replace missing values with the mean by location
            for location in locations:
                meanImputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 0)
                tempDF = historicalDF[historicalDF['location'] == location]
                totalCases = pd.DataFrame(meanImputer.fit_transform(tempDF[['total_cases']]))
                newCases = pd.DataFrame(meanImputer.fit_transform(tempDF[['new_cases']]))
                totalDeaths = pd.DataFrame(meanImputer.fit_transform(tempDF[['total_deaths']]))
                newDeaths = pd.DataFrame(meanImputer.fit_transform(tempDF[['new_deaths']]))
                totalCasesPerMillion = pd.DataFrame(meanImputer.fit_transform(tempDF[['total_cases_per_million']]))
                newCasesPerMillion = pd.DataFrame(meanImputer.fit_transform(tempDF[['new_cases_per_million']]))
                totalDeathsPerMillion = pd.DataFrame(meanImputer.fit_transform(tempDF[['total_deaths_per_million']]))
                newDeathsPerMillion = pd.DataFrame(meanImputer.fit_transform(tempDF[['new_deaths_per_million']]))

                # check all values are available for this location
                if not totalCases.empty:
                    composedDF = pd.DataFrame(list(zip(tempDF['iso_code'].values, tempDF['continent'].values, tempDF['location'].values, tempDF['date'].values, totalCases[0].values, newCases[0].values, totalDeaths[0].values, newDeaths[0].values, totalCasesPerMillion[0].values, newCasesPerMillion[0].values, totalDeathsPerMillion[0].values, newDeathsPerMillion[0].values)), columns = historicalDF.columns)
                    imputedDF = imputedDF.append(composedDF, ignore_index = True, sort = True)
            
            imputedDF.to_csv(PREPROCESSED_HISTORICAL_DATASET_PATH, index=False)
            return imputedDF
    else:
        print("Historical data .CSV file not found.")
