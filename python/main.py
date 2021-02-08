#!/usr/bin/env python

################################################################################
# Main program.
################################################################################

import cmd
import sys
import util
import total_cases

__author__ = 'Rambod Rahmani'
__copyright__ = 'Copyright (C) 2021 Rambod Rahmani'
__license__ = 'GPLv3'

##
# App main infinite loop.
##
class App(cmd.Cmd):
    intro = 'Welcome to the COVID-19 Toolbox.\n\nType help or ? to list commands.\n'
    prompt = '> '
    
    def do_total_cases(self, arg):
        'Show TOP 15 countries COVID-19 total confirmed cases histogram.'
        total_cases.plot()
        
    def do_total_cases_per_million(self, arg):
        'Show TOP 15 countries COVID-19 total confirmed cases per one million population histogram.'
        total_cases.per_milion_plot()
    
    def do_exit(self, arg):
        'Exit COVID-19 Toolbox.'
        sys.exit()

##
# Entry point.
##
if __name__ == '__main__':
    App().cmdloop()