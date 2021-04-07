#!/usr/bin/env python3

from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, association_rules

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

class Preconditions:
	def buildPredictiveModels(self, preconditionsDF):
		if not preconditionsDF is None:
			# select only covid-19 positive patients
			preconditionsDF = preconditionsDF[preconditionsDF['covid19'] == True]

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
			
			# compute all confidence, max confidence, Kulc and cosine measures as
			# well as the imbalance ratio for the association rules
