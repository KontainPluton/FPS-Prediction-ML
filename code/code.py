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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from math import trunc
from sklearn import tree
from sklearn.linear_model import Ridge

###################################################################
########## Charge le dataset dans la variable dataset #############
###################################################################

# Import du dataset depuis OpenML (lien du dataset : https://www.openml.org/d/42737)

dataset = sklearn.datasets.fetch_openml(name="fps-in-video-games")

###################################################################
# Effectue un traitement sur un DataFrame crée à partir du dataset#
###################################################################

dataset_df = pd.DataFrame(
    data=np.c_[dataset.data, dataset.target], columns=dataset.feature_names+['target'])

dataset_df_reduced = dataset_df.dropna(axis='columns')

dataset_df_reduced_rows = dataset_df.drop(
    columns=['GpuNumberOfExecutionUnits', 'GpuNumberOfComputeUnits', 'CpuNumberOfTransistors', 'CpuDieSize', 'CpuCacheL3', 'Dataset']).dropna().reset_index().drop(columns=['index'])

dataset_df_reduced_rows = dataset_df_reduced_rows.drop_duplicates(
    subset=dataset_df_reduced_rows.columns.difference(['target']))

dataset_df_reduced_rows_pivot = dataset_df_reduced_rows.pivot_table(
    columns=['GameName'], aggfunc='size')

print("Nombre de jeux : ", dataset_df_reduced_rows_pivot)

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

dataset_reduced_without_string = dataset_df_reduced_rows.copy()
dict_game = {}
for name, values in dataset_df_reduced_rows.iteritems():
    compt = 0
    tab = {}
    for i in range(values.size):
        if isinstance(dataset_reduced_without_string[name][i], str):
            if dataset_reduced_without_string[name][i] not in tab:
                tab[dataset_reduced_without_string[name][i]] = compt
                dataset_reduced_without_string[name][i] = compt
                if name == 'GameName':
                    dict_game[name] = compt
                compt += 1
            else:
                dataset_reduced_without_string[name][i] = tab[dataset_reduced_without_string[name][i]]
        else:
            break
    print(len(tab))

###################################################################
########################### Entrainement ##########################
###################################################################

dataset_copy = dataset_reduced_without_string.copy()
dataset_copy = dataset_copy.sample(random_state=0, n=dataset_copy.shape[0])
X = dataset_copy.drop(columns=['target'])
y = dataset_copy['target']

#print(X.info())

# Simple

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

lr = LinearRegression().fit(X_train,y_train)

print("Training set score: {:.2f}".format(lr.score(X_train,y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test,y_test)))

# plus complexe

pipeline = Pipeline([('transformer', scalar), ('estimator', LinearRegression())])

scores = cross_val_score(pipeline, X, y, cv=20)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# print(lr.score(X_test, y_test))

pipeline = Pipeline([('transformer', scalar), ('estimator', Ridge(alpha=1.0))])

scores = cross_val_score(pipeline, X, y, cv=20)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#clf = svm.SVC(kernel='linear', C=1, random_state=42)
#scores = cross_val_score(clf, X, y, cv=5)
#print(scores)

###################################################################
###### Traitement pour passer le problème en classification #######
###################################################################

# pas de l'intervale des fps. Ex : si pas = 10, alors de 0 à 10 fps, y deviendra 0, de 11 à 20, y deviendra 1
pas = 10

dataset_copy = dataset_reduced_without_string.copy()

print(dataset_copy['target'])

for j in range(len(dataset_copy['target'])):
    dataset_copy['target'][j] = trunc(dataset_copy['target'][j] / pas)

dataset_copy = dataset_copy.astype({'target': 'int32'})
print(dataset_copy['target'])

##################################################################
####### Application arbre décision / régression logistique #######
##################################################################

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(dataset_copy.drop(columns=['target']), dataset_copy['target'], test_size=0.33)

scalar = StandardScaler()
X_train_c = scalar.fit_transform(X_train_c)
X_test_c = scalar.fit_transform(X_test_c)

# clf = LogisticRegression(solver='lbfgs', max_iter=2000)
clf = tree.DecisionTreeClassifier(random_state=0, max_depth=11)

clf.fit(X_train_c, y_train_c)

train_score_c = clf.score(X_train_c, y_train_c)
test_score_c = clf.score(X_test_c, y_test_c)

print("Training set score: {:.2f} ".format(train_score_c))
print("Test set score: {:.2f} ".format(test_score_c))

##################################################################
############## Essai d'utiliser seulement un jeu #################
##################################################################

dataset_copy = dataset_reduced_without_string.copy()
dataset_reduced_without_string = dataset_reduced_without_string[dataset_reduced_without_string['GameName'] == dict_game['counterStrikeGlobalOffensive']]
dataset_reduced_without_string = dataset_reduced_without_string.reset_index().drop(columns=['index'])

print(dataset_copy)

dataset_copy = dataset_copy.sample(random_state=0, n=dataset_copy.shape[0])
X = dataset_copy.drop(columns=['target'])
y = dataset_copy['target']
pipeline = Pipeline([('transformer', scalar), ('estimator', LinearRegression())])

scores = cross_val_score(pipeline, X, y, cv=20)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))