#!/usr/bin/env python3

import csv
import os.path
import requests
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

# Datasets paths
HISTORICAL_DATASET_PATH = "../dataset/owid-covid-data.csv"
PREPROCESSED_HISTORICAL_DATASET_PATH = "../dataset/preprocessed-owid-covid-data.csv"
PRECONDITIONS_DATASET_PATH = "../dataset/datos_abiertos_covid19.zip"
PREPROCESSED_PRECONDITIONS_DATASET_PATH = "../dataset/preprocessed-datos_abiertos_covid19.zip"

##
# Dataset class.
#
# Contains all the functions related to loading, preprocessing, saving and
# printing the content of the different datasets.
##
class Dataset:
	def loadHistoricalDataset(self):
		if os.path.isfile(HISTORICAL_DATASET_PATH):
			if os.path.isfile(PREPROCESSED_HISTORICAL_DATASET_PATH):
				historicalDF = pd.read_csv(PREPROCESSED_HISTORICAL_DATASET_PATH)
				return historicalDF
			else:
				historicalDF = pd.read_csv(HISTORICAL_DATASET_PATH)

				# remove world and aggregated continents data
				historicalDF = historicalDF[~historicalDF.location.str.contains("Asia")]
				historicalDF = historicalDF[~historicalDF.location.str.contains("World")]
				historicalDF = historicalDF[~historicalDF.location.str.contains("Africa")]
				historicalDF = historicalDF[~historicalDF.location.str.contains("Europe")]
				historicalDF = historicalDF[~historicalDF.location.str.contains("America")]
				historicalDF = historicalDF[~historicalDF.location.str.contains("Europe Union")]
				historicalDF = historicalDF[~historicalDF.location.str.contains("North America")]
				historicalDF = historicalDF[~historicalDF.location.str.contains("South America")]

				# only select features of interest for the data mining process
				historicalDF = historicalDF[['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_cases_per_million', 'new_cases_per_million', 'total_deaths_per_million', 'new_deaths_per_million']]

				# extract location column values
				byLocation = historicalDF.groupby("location", as_index = False)["total_cases"].last()
				byLocation.columns = ['location', 'total_cases']
				locations = byLocation['location'].values

				# empty dataframe with the same columns
				imputedDF = pd.DataFrame(columns = historicalDF.columns)

				# preprocessing: replace negative values with NaN
				# replace missing values (NaN) with sliding window mean
				for location in locations:
					constantImputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 0)
					tempDF = historicalDF[historicalDF['location'] == location]

					totalCases = tempDF[['total_cases']].copy()
					totalCases[totalCases <= 0] = np.nan
					totalCases = pd.DataFrame(constantImputer.fit_transform(totalCases))
					totalCases = totalCases.rolling(10, center = True, min_periods = 1).mean()

					newCases = tempDF[['new_cases']].copy()
					newCases[newCases <= 0] = np.nan
					newCases = pd.DataFrame(constantImputer.fit_transform(newCases))
					newCases = newCases.rolling(10, center = True, min_periods = 1).mean()

					totalDeaths = tempDF[['total_deaths']].copy()
					totalDeaths[totalDeaths <= 0] = np.nan
					totalDeaths = pd.DataFrame(constantImputer.fit_transform(totalDeaths))
					totalDeaths = totalDeaths.rolling(10, center = True, min_periods = 1).mean()

					newDeaths = tempDF[['new_deaths']].copy()
					newDeaths[newDeaths <= 0] = np.nan
					newDeaths = pd.DataFrame(constantImputer.fit_transform(newDeaths))
					newDeaths = newDeaths.rolling(10, center = True, min_periods = 1).mean()

					totalCasesPerMillion = tempDF[['total_cases_per_million']].copy()
					totalCasesPerMillion[totalCasesPerMillion <= 0] = np.nan
					totalCasesPerMillion = pd.DataFrame(constantImputer.fit_transform(totalCasesPerMillion))
					totalCasesPerMillion = totalCasesPerMillion.rolling(10, center = True, min_periods = 1).mean()

					newCasesPerMillion = tempDF[['new_cases_per_million']].copy()
					newCasesPerMillion[newCasesPerMillion <= 0] = np.nan
					newCasesPerMillion = pd.DataFrame(constantImputer.fit_transform(newCasesPerMillion))
					newCasesPerMillion = newCasesPerMillion.rolling(10, center = True, min_periods = 1).mean()

					totalDeathsPerMillion = tempDF[['total_deaths_per_million']].copy()
					totalDeathsPerMillion[totalDeathsPerMillion <= 0] = np.nan
					totalDeathsPerMillion = pd.DataFrame(constantImputer.fit_transform(totalDeathsPerMillion))
					totalDeathsPerMillion = totalDeathsPerMillion.rolling(10, center = True, min_periods = 1).mean()

					newDeathsPerMillion = tempDF[['new_deaths_per_million']].copy()
					newDeathsPerMillion[newDeathsPerMillion <= 0] = np.nan
					newDeathsPerMillion = pd.DataFrame(constantImputer.fit_transform(newDeathsPerMillion))
					newDeathsPerMillion = newDeathsPerMillion.rolling(10, center = True, min_periods = 1).mean()

					# check all values are available for current country
					if not totalCases.empty:
						# append current country data
						composedDF = pd.DataFrame(list(zip(tempDF['iso_code'].values, tempDF['continent'].values, tempDF['location'].values, tempDF['date'].values, totalCases[0].values, newCases[0].values, totalDeaths[0].values, newDeaths[0].values, totalCasesPerMillion[0].values, newCasesPerMillion[0].values, totalDeathsPerMillion[0].values, newDeathsPerMillion[0].values)), columns = historicalDF.columns)
						imputedDF = imputedDF.append(composedDF, ignore_index = True, sort = True)

				# store preprocessed dataframe for future use
				imputedDF.to_csv(PREPROCESSED_HISTORICAL_DATASET_PATH, index = False)

				return imputedDF
		else:
			print("Historical dataset .csv file not found. Please run update_historical_data first.")

	def updateHistoricalDataset(self):
		url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
		response = requests.get(url)

		if os.path.isfile(HISTORICAL_DATASET_PATH):
			os.remove(HISTORICAL_DATASET_PATH)

		if os.path.isfile(PREPROCESSED_HISTORICAL_DATASET_PATH):
			os.remove(PREPROCESSED_HISTORICAL_DATASET_PATH)

		with open(HISTORICAL_DATASET_PATH, 'w') as f:
			writer = csv.writer(f)
			for line in response.iter_lines():
				writer.writerow(line.decode('utf-8').split(','))

	def printHistoricalDatasetInfo(self):
		historicalDF = self.loadHistoricalDataset()	
		if not historicalDF is None:
			print(historicalDF.info(verbose = True))

	def loadPreconditionsDataset(self):
		if os.path.isfile(PRECONDITIONS_DATASET_PATH):
			if os.path.isfile(PREPROCESSED_PRECONDITIONS_DATASET_PATH):
				preconditionsDF = pd.read_csv(PREPROCESSED_PRECONDITIONS_DATASET_PATH)
				return preconditionsDF
			else:
				preconditionsDF = pd.read_csv(PRECONDITIONS_DATASET_PATH)

				# only select features of interest for the data mining process
				preconditionsDF = preconditionsDF[['FECHA_INGRESO', 'SEXO', 'EDAD', 'CLASIFICACION_FINAL', 'FECHA_DEF', 'NEUMONIA', 'EMBARAZO', 'DIABETES', 'EPOC', 'ASMA', 'INMUSUPR', 'HIPERTENSION', 'OTRA_COM', 'CARDIOVASCULAR', 'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO', 'TIPO_PACIENTE', 'INTUBADO', 'UCI']]

				# translate features names to english
				preconditionsDF.columns = ['recovery_date', 'sex', 'age', 'covid19', 'deceased', 'pneumonia', 'pregnancy', 'diabetes', 'copd', 'asthma', 'immunosuppression', 'hypertension', 'other_diseases', 'cardiovascular', 'obesity', 'chronic_kidney_failure', 'smoking', 'hospitalization', 'intubation', 'icu']

				# map numerical values to categorical values
				preconditionsDF['sex'] = preconditionsDF['sex'].map({1: 'female', 2: 'male'})
				preconditionsDF.loc[preconditionsDF['sex'] == 'female', 'female'] = True
				preconditionsDF.loc[preconditionsDF['sex'] == 'male', 'female'] = False
				preconditionsDF.loc[preconditionsDF['sex'] == 'female', 'male'] = False
				preconditionsDF.loc[preconditionsDF['sex'] == 'male', 'male'] = True
				preconditionsDF['covid19'] = preconditionsDF['covid19'].map({1: True, 2: True, 3: True, 4: False, 5: False, 6: False, 7: False})
				preconditionsDF['deceased'] = preconditionsDF['deceased'].map(lambda x: x != '9999-99-99')
				preconditionsDF['pneumonia'] = preconditionsDF['pneumonia'].map({1: True, 2: False, 97: False, 98: False, 99: False})
				preconditionsDF['pregnancy'] = preconditionsDF['pregnancy'].map({1: True, 2: False, 97: False, 98: False, 99: False})
				preconditionsDF['diabetes'] = preconditionsDF['diabetes'].map({1: True, 2: False, 97: False, 98: False, 99: False})
				preconditionsDF['copd'] = preconditionsDF['copd'].map({1: True, 2: False, 97: False, 98: False, 99: False})
				preconditionsDF['asthma'] = preconditionsDF['asthma'].map({1: True, 2: False, 97: False, 98: False, 99: False})
				preconditionsDF['immunosuppression'] = preconditionsDF['immunosuppression'].map({1: True, 2: False, 97: False, 98: False, 99: False})
				preconditionsDF['hypertension'] = preconditionsDF['hypertension'].map({1: True, 2: False, 97: False, 98: False, 99: False})
				preconditionsDF['other_diseases'] = preconditionsDF['other_diseases'].map({1: True, 2: False, 97: False, 98: False, 99: False})
				preconditionsDF['cardiovascular'] = preconditionsDF['cardiovascular'].map({1: True, 2: False, 97: False, 98: False, 99: False})
				preconditionsDF['obesity'] = preconditionsDF['obesity'].map({1: True, 2: False, 97: False, 98: False, 99: False})
				preconditionsDF['chronic_kidney_failure'] = preconditionsDF['chronic_kidney_failure'].map({1: True, 2: False, 97: False, 98: False, 99: False})
				preconditionsDF['smoking'] = preconditionsDF['smoking'].map({1: True, 2: False, 97: False, 98: False, 99: False})
				preconditionsDF['hospitalization'] = preconditionsDF['hospitalization'].map({1: False, 2: True, 99: False})
				preconditionsDF['intubation'] = preconditionsDF['intubation'].map({1: True, 2: False, 97: False, 98: False, 99: False})
				preconditionsDF['icu'] = preconditionsDF['icu'].map({1: True, 2: False, 97: False, 98: False, 99: False})

				preconditionsDF = preconditionsDF.fillna(False)

				# store preprocessed dataframe for future use
				preconditionsDF.to_csv(PREPROCESSED_PRECONDITIONS_DATASET_PATH, index = False, compression = "zip")

				return preconditionsDF;
		else:
			print("Preconditions dataset .csv file not found.")

	def printPreconditionsDatasetInfo(self):
		preconditionsDF = self.loadPreconditionsDataset()
		if not preconditionsDF is None:
			print(preconditionsDF.info(verbose = True))
