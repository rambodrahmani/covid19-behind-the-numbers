#!/usr/bin/env python

##
# Main program.
##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

# load raw data
historical_df = pd.read_csv("historical-covid-data.csv")

# group raw data by country aggregating on total cases
by_location = historical_df.groupby("location", as_index = False)["total_cases"].last()
by_location.columns = ['location', 'total_cases']

# remove world and continents data
by_location = by_location[~by_location.location.str.contains("World")]
by_location = by_location[~by_location.location.str.contains("Europe")]
by_location = by_location[~by_location.location.str.contains("Europe Union")]
by_location = by_location[~by_location.location.str.contains("America")]
by_location = by_location[~by_location.location.str.contains("North America")]
by_location = by_location[~by_location.location.str.contains("Asia")]
by_location = by_location[~by_location.location.str.contains("Africa")]

# sort by total cases count
by_location = by_location.sort_values(by=['total_cases'], ascending = False)

# extract top 15 countries
by_location = by_location[0:15]

# plot top 15 countries histogram
by_location_hist = by_location.plot.bar(x='location', y='total_cases', rot=0)
by_location_hist.set(xlabel="Locations", ylabel="Active Cases", title="Confirmed COVID-19 cases worldwide (Top 15)")
plt.ticklabel_format(style='plain', axis='y')
plt.show()
