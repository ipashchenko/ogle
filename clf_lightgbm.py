# -*- coding: utf-8 -*-
import os
# import sys
# sys.setrecursionlimit(100000)
# import dill
import numpy as np
import pandas as pd
import hyperopt
from pprint import pprint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from lightgbm import LGBMClassifier
from utils import remove_correlated_features
from tsfresh import select_features


# FIXME: Insert info on class after removing correlated features!
target = 'variable'
data_dir = '/home/ilya/Dropbox/papers/ogle2/data/new_features/'
df_vars = pd.read_pickle(os.path.join(data_dir, "features_vars_sc20.pkl"))
df_vars[target] = 1
df_const = pd.read_pickle(os.path.join(data_dir, "features_const_sc20.pkl"))
df_const[target] = 0
df = pd.concat((df_vars, df_const), ignore_index=True)
df = df.loc[:, ~df.columns.duplicated()]
df = remove_correlated_features(df, r=0.95)
nan_columns = df.columns[df.isnull().any()].tolist()
df.drop(nan_columns, axis=1, inplace=True)
y = df[target].values
df = select_features(df, y)

features_names = list(df.columns)
features_names.remove(target)
X = df[features_names].values


kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)


def objective(space):
    pprint(space)
    clf = LGBMClassifier(n_estimators=space["n_estimators"],
                         subsample=space["subsample"],
                         colsample_bytree=space["colsample_bytree"],
                         reg_alpha=space["reg_alpha"],
                         reg_lambda=space["reg_lambda"],
                         min_child_samples=space["min_child_samples"],
                         learning_rate=space["learning_rate"],
                         scale_pos_weight={0: 1, 1: space["scale_pos_weight"]},
                         sub_feature=space["sub_feature"],
                         early_stopping_round=20)

    pipeline = Pipeline([('imputer', Imputer(missing_values='NaN',
                          strategy='median', axis=0, verbose=2)),
                         ('clf', clf)])

    CMs = list()
    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        fit_params = {"clf__eval_set": (X_test, y_test)}
        pipeline.fit(X_train, y_train, **fit_params)
        y_preds = pipeline.predict(X_test)
        CMs.append(confusion_matrix(y[test_idx], y_preds))
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


space = {'n_estimators': hp.choice('n_estimators', np.arange(10, 101, 10, dtype=int)),
         'subsample': hp.uniform('subsample', 0.0, 1.0),
         'colsample_bytree': hp.uniform('colsample_bytree', 0.0, 1.0),
         'reg_alpha': hp.uniform('reg_alpha', 0.0, 20.0),
         'reg_lambda': hp.uniform('reg_lambda', 0.0, 20.0),
         'rsm': hp.uniform('rsm', 0.0, 1.0),
         'min_child_samples': hp.choice('min_child_samples', np.arange(1, 50, 1, dtype=int)),
         'learning_rate': hp.loguniform('learning_rate', -5.0, -1.0),
         'scale_pos_weight': hp.uniform('scale_pos_weight', 1., 10.),
         'sub_feature': hp.uniform('sub_feature', 0, 1)}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=300,
            trials=trials)

pprint(hyperopt.space_eval(space, best))
best_pars = hyperopt.space_eval(space, best)