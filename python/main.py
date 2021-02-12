#!/usr/bin/env python

################################################################################
# Main program.
################################################################################

import cmd
import sys
import util
import total_cases
import total_deaths

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

    def do_total_cases(self, arg):
        'Plot COVID-19 total confirmed cases histogram of the TOP 15 countries.'
        total_cases.plot()
        
    def do_total_cases_per_million(self, arg):
        'Plot COVID-19 total confirmed cases per one million population histogram of the TOP 15 countries.'
        total_cases.perMilionPlot()

    def do_total_deaths_per_million(self, arg):
        'Plot COVID-19 daily deaths per one million population time series of random countries.'
        total_deaths.perMilionPlot()
        
    def do_total_deaths_all_countries(self, arg):
        'Plot COVID-19 daily deaths per one million population time series of all countries.'
        total_deaths.allCountriesPerMilionPlot()
        
    def do_total_deaths_clustering(self, arg):
        'Plot COVID-19 daily deaths per one million population time series clusters.'
        total_deaths.clusteringPlot()

    def do_exit(self, arg):
        'Exit COVID-19 Toolbox.'
        sys.exit()

##
# Entry point.
##
if __name__ == '__main__':
    App().cmdloop()
