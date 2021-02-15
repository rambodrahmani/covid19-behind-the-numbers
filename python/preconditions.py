#!/usr/bin/env python

################################################################################
# 
################################################################################

import util
from apyori import apriori

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

def buildPredictiveModels():
    # load preconditions data
    preconditionsDF = util.loadPreconditionsData()
    
    # check historical data was correctly loaded
    if not preconditionsDF is None:
        records = []
        for i in range(0, 100000):
            records.append([str(preconditionsDF.values[i,j]) for j in range(0, 19)])
        associationRules = apriori(records, min_support = 0.02, min_confidence = 0.05, min_lift = 3, min_length = 2)
        associationRules = list(associationRules)
        print(associationRules)
