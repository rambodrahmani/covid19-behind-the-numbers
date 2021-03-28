#!/usr/bin/env python

################################################################################
# 
################################################################################

import util
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, association_rules

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

def buildPredictiveModels():
    # load preconditions data
    preconditionsDF = util.loadPreconditionsData()

    # select only covid-19 positive patients
    preconditionsDF = preconditionsDF[preconditionsDF['covid19'] == True]
    
    # check historical data was correctly loaded
    if not preconditionsDF is None:
        # drop non categorical columns
        preconditionsDF = preconditionsDF.drop(['recovery_date', 'sex', 'age'], axis=1)

        # compute apriori
        aprioriFrequentItemsets = apriori(preconditionsDF, min_support=0.02, use_colnames=True)
        aprioriRules = association_rules(aprioriFrequentItemsets, metric="confidence", min_threshold=0.7)
        print(aprioriRules.to_string())

        # compute fpgrowth
        fpgrowthFrequentItemsets = fpgrowth(preconditionsDF, min_support=0.02, use_colnames=True)
        fpgrowthRules = association_rules(fpgrowthFrequentItemsets, metric="confidence", min_threshold=0.7)
        print(fpgrowthRules.to_string())

        # compute fpmax
        fpmaxFrequentItemsets = fpmax(preconditionsDF, min_support=0.02, use_colnames=True)
        fpmaxRules = association_rules(fpmaxFrequentItemsets, metric="confidence", min_threshold=0.7)
        print(fpmaxRules.to_string())
