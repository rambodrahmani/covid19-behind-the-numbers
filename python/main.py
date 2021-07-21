#!/usr/bin/env python3

import cmd
import sys
from dataset import Dataset
from confirmed_cases import ConfirmedCases
from confirmed_deaths import ConfirmedDeaths
from predictive_models import PredictiveModels
from countries_clustering import CountriesClustering

__author__ = 'Rambod Rahmani'
__copyright__ = 'Copyright (C) 2021 Rambod Rahmani'
__license__ = 'GPLv3'

##
# App command loop.
##
class App(cmd.Cmd):
    intro = 'Welcome to the COVID-19 Toolbox.\n\nType help or ? to list commands.\n'
    prompt = '> '

    dataset = Dataset()
    confirmedCases = ConfirmedCases()
    confirmedDeaths = ConfirmedDeaths()
    countriesClustering = CountriesClustering()
    predictiveModels = PredictiveModels()

    def do_update_historical_ds(self, arg):
        'Update COVID-19 historical dataset to the latest available version.'
        self.dataset.updateHistoricalDataset()

    def do_print_historical_ds_info(self, arg):
        'Print COVID-19 historical dataset info.'
        self.dataset.printHistoricalDatasetInfo()

    def do_print_preconditions_ds_info(self, arg):
        'Print COVID-19 preconditions dataset info.'
        self.dataset.printPreconditionsDatasetInfo()

    def do_plot_total_cases(self, arg):
        'Plot COVID-19 total confirmed cases histogram of the TOP 15 countries.'
        self.confirmedCases.plot(self.dataset.loadHistoricalDataset())
        
    def do_plot_total_cases_per_million(self, arg):
        'Plot COVID-19 total confirmed cases per one million population histogram of the TOP 15 countries.'
        self.confirmedCases.perMilionPlot(self.dataset.loadHistoricalDataset())

    def do_plot_weekly_deaths_per_million(self, arg):
        'Plot COVID-19 weekly deaths per one million population time series of random countries.'
        self.confirmedDeaths.perMilionPlot(self.dataset.loadHistoricalDataset())
        
    def do_plot_weekly_deaths_all_countries(self, arg):
        'Plot COVID-19 weekly deaths per one million population time series of all countries.'
        self.confirmedDeaths.allCountriesPerMilionPlot(self.dataset.loadHistoricalDataset())

    def do_timeseries_clustering_euclidean(self, arg):
        'Plot COVID-19 daily deaths per one million population time series euclidean-based K-Means clusters.'
        self.countriesClustering.euclideanDistance(self.dataset.loadHistoricalDataset())

    def do_timeseries_clustering_dtw(self, arg):
        'Plot COVID-19 daily deaths per one million population time series DTW-based K-Means clusters.'
        self.countriesClustering.dynamicTimeWarping(self.dataset.loadHistoricalDataset())

    def do_personalized_predictive_models(self, arg):
        'Build personalized predictive models for symptomatic COVID-19 patients using medical preconditions.'
        self.predictiveModels.buildPredictiveModels(self.dataset.loadPreconditionsDataset())

    def do_exit(self, arg):
        'Exit COVID-19 Toolbox.'
        sys.exit()

##
# Entry point.
##
if __name__ == '__main__':
    App().cmdloop()
