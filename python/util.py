#!/usr/bin/env python

################################################################################
# Utilities functions.
################################################################################

import os.path
import pandas as pd

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

HISTORICAL_DATASET_PATH = "historical-covid-data.csv"

##
# Loads and preprocesses the historical dataframe.
##
def loadHistoricalData():
    # check if the csv file exists
    if os.path.isfile(HISTORICAL_DATASET_PATH):
        # read csv file
        historical_df = pd.read_csv(HISTORICAL_DATASET_PATH)

        # remove world and continents data
        historical_df = historical_df[~historical_df.location.str.contains("Asia")]
        historical_df = historical_df[~historical_df.location.str.contains("World")]
        historical_df = historical_df[~historical_df.location.str.contains("Africa")]
        historical_df = historical_df[~historical_df.location.str.contains("Europe")]
        historical_df = historical_df[~historical_df.location.str.contains("America")]
        historical_df = historical_df[~historical_df.location.str.contains("Australia")]
        historical_df = historical_df[~historical_df.location.str.contains("Antarctica")]
        historical_df = historical_df[~historical_df.location.str.contains("Europe Union")]
        historical_df = historical_df[~historical_df.location.str.contains("North America")]
        historical_df = historical_df[~historical_df.location.str.contains("South America")]

        return historical_df
    else:
        print("Historical data .CSV file not found.")
