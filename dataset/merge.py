#!/usr/bin/env python

################################################################################
# Utilities functions.
################################################################################

import csv
import os.path
import requests
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

__author__ = "Rambod Rahmani"
__copyright__ = "Copyright (C) 2021 Rambod Rahmani"
__license__ = "GPLv3"

# load preconditions dataset
preconditionsDF = pd.read_csv('210206COVID19MEXICO.csv')

# select features of interest
preconditionsDF = preconditionsDF[['FECHA_INGRESO', 'SEXO', 'EDAD', 'CLASIFICACION_FINAL', 'FECHA_DEF', 'NEUMONIA', 'EMBARAZO', 'DIABETES', 'EPOC', 'ASMA', 'INMUSUPR', 'HIPERTENSION', 'OTRA_COM', 'CARDIOVASCULAR', 'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO', 'TIPO_PACIENTE', 'INTUBADO', 'UCI']]

# translate features names
preconditionsDF.columns = ['recovery_date', 'sex', 'age', 'classification', 'deceased', 'pneumonia', 'pregnancy', 'diabetes', 'copd', 'asthma', 'immunosuppression', 'hypertension', 'other_diseases', 'cardiovascular', 'obesity', 'chronic_kidney_failure', 'smoking', 'hospitalization', 'intubation', 'icu']

# map numerical values to categorical values
preconditionsDF['sex'] = preconditionsDF['sex'].map({1: 'female', 2: 'male'})
preconditionsDF['classification'] = preconditionsDF['classification'].map({1: 'covid19', 2: 'covid19', 3: 'covid19', 4: 'no_covid19', 5: 'no_covid19', 6: 'no_covid19', 7: 'no_covid19'})
preconditionsDF['deceased'] = preconditionsDF['deceased'].map(lambda x: x < '9999-01-01')
preconditionsDF['pneumonia'] = preconditionsDF['pneumonia'].map({1: 'yes', 2: 'no', 97: 'no', 98: 'no', 99: 'no'})
preconditionsDF['pregnancy'] = preconditionsDF['pregnancy'].map({1: 'yes', 2: 'no', 97: 'no', 98: 'no', 99: 'no'})
preconditionsDF['diabetes'] = preconditionsDF['diabetes'].map({1: 'yes', 2: 'no', 97: 'no', 98: 'no', 99: 'no'})
preconditionsDF['copd'] = preconditionsDF['copd'].map({1: 'yes', 2: 'no', 97: 'no', 98: 'no', 99: 'no'})
preconditionsDF['asthma'] = preconditionsDF['asthma'].map({1: 'yes', 2: 'no', 97: 'no', 98: 'no', 99: 'no'})
preconditionsDF['immunosuppression'] = preconditionsDF['immunosuppression'].map({1: 'yes', 2: 'no', 97: 'no', 98: 'no', 99: 'no'})
preconditionsDF['hypertension'] = preconditionsDF['hypertension'].map({1: 'yes', 2: 'no', 97: 'no', 98: 'no', 99: 'no'})
preconditionsDF['other_diseases'] = preconditionsDF['other_diseases'].map({1: 'yes', 2: 'no', 97: 'no', 98: 'no', 99: 'no'})
preconditionsDF['cardiovascular'] = preconditionsDF['cardiovascular'].map({1: 'yes', 2: 'no', 97: 'no', 98: 'no', 99: 'no'})
preconditionsDF['obesity'] = preconditionsDF['obesity'].map({1: 'yes', 2: 'no', 97: 'no', 98: 'no', 99: 'no'})
preconditionsDF['chronic_kidney_failure'] = preconditionsDF['chronic_kidney_failure'].map({1: 'yes', 2: 'no', 97: 'no', 98: 'no', 99: 'no'})
preconditionsDF['smoking'] = preconditionsDF['smoking'].map({1: 'yes', 2: 'no', 97: 'no', 98: 'no', 99: 'no'})
preconditionsDF['hospitalization'] = preconditionsDF['hospitalization'].map({1: 'no', 2: 'yes', 99: 'no'})
preconditionsDF['intubation'] = preconditionsDF['intubation'].map({1: 'yes', 2: 'no', 97: 'no', 98: 'no', 99: 'no'})
preconditionsDF['icu'] = preconditionsDF['icu'].map({1: 'yes', 2: 'no', 97: 'no', 98: 'no', 99: 'no'})
