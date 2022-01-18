###################################################################
############## Import des librairies nécessaire ###################
###################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from IPython.display import display

from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split

###################################################################
############ Charge le dataset dans la variable .. ################
###################################################################

df = sklearn.datasets.fetch_openml(name="fps-in-video-games")

###################################################################
# Effectue un traitement sur un DataFrame crée à partir du dataset#
###################################################################

bc_df = pd.DataFrame(df.data, columns=df.feature_names)

bc_df_reduced = bc_df.dropna(axis='columns')

bc_df_reduced_rows = bc_df.drop(columns=['GpuNumberOfExecutionUnits']).dropna()

#print("X_train shape: {}".format(bc_df['GpuMemoryBus'][59]))

#print(bc_df.columns)
#print(bc_df_reduced.columns)

dups_color = bc_df_reduced_rows.pivot_table(columns=['GameName'], aggfunc='size')
print (dups_color)

###################################################################
########### Compte le nombre de NaN dans le dataset ###############
###################################################################

bc_df = pd.DataFrame(df.data, columns=df.feature_names)

tab = []
j = 0

for feature in bc_df.columns:
    tab.append(bc_df[feature].isna().sum())

print(bc_df.isna().sum().sum())
print(tab)