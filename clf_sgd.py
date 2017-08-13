# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, RobustScaler
from sklearn.feature_selection import VarianceThreshold


target = 'variable'
data_dir = '/home/ilya/github/ogle'
df_vars = pd.read_pickle(os.path.join(data_dir, "features_vars.pkl"))
df_vars[target] = 1
df_const = pd.read_pickle(os.path.join(data_dir, "features_const.pkl"))
df_const[target] = 0
df = pd.concat((df_vars, df_const), ignore_index=True)
features_names = list(df.columns)
features_names.remove(target)
X = df[features_names].values
y = df[target].values


kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)


def objective(space):
    pprint(space)
    clf = SGDClassifier(loss='log', penalty='l1', alpha=space['alpha'],
                        class_weight={0: 1, 1: space['cw']},
                        max_iter=1000, tol=10**(-3), eta0=space['eta0'],
                        learning_rate='constant')

    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    estimators.append(('variance_thresholder', VarianceThreshold()))
    estimators.append(('scaler', RobustScaler()))
    estimators.append(('clf', clf))
    pipeline = Pipeline(estimators)

    y_preds = cross_val_predict(pipeline, X, y, cv=kfold, n_jobs=4)
    CMs = list()
    for train_idx, test_idx in kfold.split(X, y):
        CMs.append(confusion_matrix(y[test_idx], y_preds[test_idx]))
    CM = np.sum(CMs, axis=0)

    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print("TP = {}".format(TP))
    print("FP = {}".format(FP))
    print("FN = {}".format(FN))

    f1 = 2. * TP / (2. * TP + FP + FN)
    print("F1 : ", f1)

    return{'loss': 1-f1, 'status': STATUS_OK}

space = {'alpha': hp.loguniform('C', -6.0, 1.0),
         'cw': hp.uniform('cw', 0.5, 30),
         'eta0': hp.loguniform('eta0', -2.0, 0.0)}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

pprint(hyperopt.space_eval(space, best))
best_pars = hyperopt.space_eval(space, best)