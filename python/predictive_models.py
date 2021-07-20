#!/usr/bin/env python3

import time
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, association_rules

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

class PredictiveModels:
	def buildPredictiveModels(self, preconditionsDF):
		if not preconditionsDF is None:
			# select only covid-19 positive patients
			preconditionsDF = preconditionsDF[preconditionsDF['covid19'] == True]

			# drop non categorical columns
			preconditionsDF = preconditionsDF.drop(['recovery_date', 'sex', 'age'], axis=1)
			
			# compute apriori
			aprioriStartTime = time.time()
			aprioriFrequentItemsets = apriori(preconditionsDF, min_support=0.01, use_colnames=True)
			aprioriAssociationRules = association_rules(aprioriFrequentItemsets, metric="confidence", min_threshold=0.7)
			aprioriComputationTime = (time.time() - aprioriStartTime)
			aprioriAssociationRulesCount = len(aprioriAssociationRules.index)
			print(aprioriAssociationRules)
			print(f"--- {aprioriAssociationRulesCount} Association Rules computed in {aprioriComputationTime} seconds ---\n")
			print(aprioriAssociationRules.to_string() + "\n\n")
			
			# compute fpgrowth
			fpgrowthStartTime = time.time()
			fpgrowthFrequentItemsets = fpgrowth(preconditionsDF, min_support=0.01, use_colnames=True)
			fpgrowthAssociationRules = association_rules(fpgrowthFrequentItemsets, metric="confidence", min_threshold=0.7)
			fpgrowthComputationTime = (time.time() - fpgrowthStartTime)
			fpgrowthAssociationRulesCount = len(fpgrowthAssociationRules.index)
			print(fpgrowthAssociationRules)
			print(f"--- {fpgrowthAssociationRulesCount} Association Rules computed in {fpgrowthComputationTime} seconds ---\n")
			print(fpgrowthAssociationRules.to_string() + "\n\n")

			# compute all confidence, max confidence, Kulc and cosine measures as
			# well as the imbalance ratio for the association rules
