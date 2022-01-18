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
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

###################################################################
########## Charge le dataset dans la variable dataset #############
###################################################################

### Import du dataset depuis OpenML (lien du dataset : https://www.openml.org/d/42737)

dataset = sklearn.datasets.fetch_openml(name="fps-in-video-games")

###################################################################
# Effectue un traitement sur un DataFrame crée à partir du dataset#
###################################################################

dataset_df = pd.DataFrame(data=np.c_[dataset.data,dataset.target], columns=dataset.feature_names+['target'])

dataset_df_reduced = dataset_df.dropna(axis='columns')

dataset_df_reduced_rows = dataset_df.drop(columns=['GpuNumberOfExecutionUnits','CpuCacheL3']).dropna()

dataset_df_reduced_rows_pivot = dataset_df_reduced_rows.pivot_table(columns=['GameName'], aggfunc='size')

print(dataset_df_reduced_rows.transpose())

###################################################################
########### Compte le nombre de NaN dans le dataset ###############
###################################################################

bc_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

tab = []
j = 0

for feature in bc_df.columns:
    tab.append(bc_df[feature].isna().sum())

print(bc_df.isna().sum().sum())
print(tab)

###################################################################
#### Fait correspondre les éléments texte du dataset en entier ####
###################################################################

dataset_copy = dataset_df_reduced_rows
dict_init = {}
for name, values in dataset_df_reduced_rows.iteritems():
    compt = 0
    tab = {}
    for i in range(values.size):
        if isinstance(dataset_copy[name][i], str):
            if dataset_copy[name][i] not in tab:
                tab[dataset_copy[name][i]] = compt
                dataset_copy[name][i] = compt
                compt += 1
            else:
                dataset_copy[name][i] = tab[dataset_copy[name][i]]
        else:
            break
    print(len(tab))
    dict_init[name] = tab

###################################################################
########################### Entrainement ##########################
###################################################################

# print(dataset_copy)
# print(dict_init)

############################ simple

X_train, X_test, y_train, y_test = train_test_split(
    dataset_df_reduced_rows.drop(columns=['target']), dataset_df_reduced_rows['target'], random_state=0)

# lr = LinearRegression().fit(X_train,y_train)
lr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)

# print("Training set score: {:.2f}".format(lr.score(X_train,y_train)))
# print("Test set score: {:.2f}".format(lr.score(X_test,y_test)))

############################# plus complexe

# scores = cross_val_score(lr, X_test, y_test, cv=20)
# print(scores)
print(lr.score(X_test, y_test))