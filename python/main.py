#!/usr/bin/env python

##
# Entry point.
##

import pandas as pd
import matplotlib.pyplot as plt

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

# load the first dataset
historical_df = pd.read_csv("historical-covid-data.csv")

print(historical_df.head())

by_location = historical_df.groupby("location")["total_cases"].last()

print(by_location.head())

by_location.plot(kind='bar')
