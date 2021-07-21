#!/usr/bin/env python3

import time
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, association_rules

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

##
# PredictiveModels class.
#
# Contains all the functions related to computing and printing associations
# rules based on the preconditions dataset.
##
class PredictiveModels:
	def buildPredictiveModels(self, preconditionsDF):
		if not preconditionsDF is None:
			# select only covid-19 positive patients
			preconditionsDF = preconditionsDF[preconditionsDF['covid19'] == True]

			# drop non categorical columns
			preconditionsDF = preconditionsDF.drop(['recovery_date', 'sex', 'age'], axis = 1)
			
			# compute apriori
			aprioriStartTime = time.time()
			aprioriFrequentItemsets = apriori(preconditionsDF, min_support = 0.01, use_colnames = True)
			aprioriAssociationRules = association_rules(aprioriFrequentItemsets, metric = "confidence", min_threshold = 0.7)
			aprioriComputationTime = (time.time() - aprioriStartTime)
			aprioriAssociationRulesCount = len(aprioriAssociationRules.index)

			# compute Kulczynski measure and IR for apriori results
			apriori_support = aprioriAssociationRules['support']
			apriori_antecedent_support = aprioriAssociationRules['antecedent support']
			apriori_consequent_support = aprioriAssociationRules['consequent support']
			aprioriAssociationRules['kulczynski'] = self.computeKulczynski(apriori_support, apriori_antecedent_support, apriori_consequent_support)
			aprioriAssociationRules['imbalance ratio'] = self.computeIR(apriori_support, apriori_antecedent_support, apriori_consequent_support)

			# print apriori association rules
			print(aprioriAssociationRules)
			print(f"--- {aprioriAssociationRulesCount} Association Rules computed in {aprioriComputationTime} seconds ---\n")
			print(aprioriAssociationRules.to_string() + "\n\n")
			
			# compute fpgrowth
			fpgrowthStartTime = time.time()
			fpgrowthFrequentItemsets = fpgrowth(preconditionsDF, min_support = 0.01, use_colnames = True)
			fpgrowthAssociationRules = association_rules(fpgrowthFrequentItemsets, metric = "confidence", min_threshold = 0.7)
			fpgrowthComputationTime = (time.time() - fpgrowthStartTime)
			fpgrowthAssociationRulesCount = len(fpgrowthAssociationRules.index)

			# compute Kulczynski measure and IR for fpgrowth results
			fpgrowth_support = fpgrowthAssociationRules['support']
			fpgrowth_antecedent_support = fpgrowthAssociationRules['antecedent support']
			fpgrowth_consequent_support = fpgrowthAssociationRules['consequent support']
			fpgrowthAssociationRules['kulczynski'] = self.computeKulczynski(fpgrowth_support, fpgrowth_antecedent_support, fpgrowth_consequent_support)
			fpgrowthAssociationRules['imbalance ratio'] = self.computeIR(fpgrowth_support, fpgrowth_antecedent_support, fpgrowth_consequent_support)

			# print fpgrowth association rules
			print(fpgrowthAssociationRules)
			print(f"--- {fpgrowthAssociationRulesCount} Association Rules computed in {fpgrowthComputationTime} seconds ---\n")
			print(fpgrowthAssociationRules.to_string() + "\n\n")

	def computeKulczynski(self, support, antecedent_support, consequent_support):
		return (1/2)*((support/consequent_support) + (support/antecedent_support))

	def computeIR(self, support, antecedent_support, consequent_support):
		return (abs(antecedent_support - consequent_support))/(antecedent_support + consequent_support - support)