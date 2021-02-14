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
preconditionsDF.columns = ['recovery_date', 'sex', 'age', 'classification', 'death_date', 'pneumonia', 'pregnancy', 'diabetes', 'copd', 'asthma', 'immunosuppression', 'hypertension', 'other_diseases', 'cardiovascular', 'obesity', 'chronic_kidney_failure', 'smoking', 'hospitalization', 'intubation', 'icu']

# map numerical values to categorical values
preconditionsDF['sex'] = preconditionsDF['sex'].map({1: 'female', 2: 'male'})
preconditionsDF['classification'] = preconditionsDF['classification'].map({1: 'covid19', 2: 'covid19', 3: 'covid19', 4: 'no_covid19', 5: 'no_covid19', 6: 'no_covid19', 7: 'no_covid19'})

print(preconditionsDF)
