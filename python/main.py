#!/usr/bin/env python

################################################################################
# Main program.
################################################################################

import cmd
import sys
import util
import cases
import deaths
import preconditions

__author__ = 'Rambod Rahmani'
__copyright__ = 'Copyright (C) 2021 Rambod Rahmani'
__license__ = 'GPLv3'

##
# App main infinite loop.
##
class App(cmd.Cmd):
    intro = 'Welcome to the COVID-19 Toolbox.\n\nType help or ? to list commands.\n'
    prompt = '> '

    def do_update_historical_data(self, arg):
        'Update COVID-19 historical data to the latest available version.'
        util.updateHistoricalData()

    def do_print_historical_df_info(self, arg):
        'Prints COVID-19 historical dataset info.'
        util.printHistoricalDFInfo()

    def do_total_cases(self, arg):
        'Plots COVID-19 total confirmed cases histogram of the TOP 15 countries.'
        cases.plot()
        
    def do_total_cases_per_million(self, arg):
        'Plots COVID-19 total confirmed cases per one million population histogram of the TOP 15 countries.'
        cases.perMilionPlot()

    def do_weekly_deaths_per_million(self, arg):
        'Plots COVID-19 weekly deaths per one million population time series of random countries.'
        deaths.perMilionPlot()
        
    def do_weekly_deaths_all_countries(self, arg):
        'Plots COVID-19 weekly deaths per one million population time series of all countries.'
        deaths.allCountriesPerMilionPlot()
        
    def do_daily_deaths_clusters(self, arg):
        'Plots COVID-19 daily deaths per one million population time series clusters.'
        deaths.clustersPlot()

    def do_print_preconditions_df_info(self, arg):
        'Prints COVID-19 preconditions dataset info.'
        util.printPreconditionsDFInfo()

    def do_personalized_predictive_models(self, arg):
        'Builds personalized predictive models for symptomatic COVID-19 patients using medical preconditions.'
        preconditions.buildPredictiveModels()

    def do_exit(self, arg):
        'Exit COVID-19 Toolbox.'
        sys.exit()

##
# Entry point.
##
if __name__ == '__main__':
    App().cmdloop()
