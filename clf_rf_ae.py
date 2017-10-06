# -*- coding: utf-8 -*-
import os
import sys
sys.setrecursionlimit(100000)
import dill
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix
import hyperopt as hp
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from utils import remove_correlated_features
from tsfresh import select_features
from ae_transform import AEExtract


# FIXME: Insert info on class after removing correlated features!
target = 'variable'
data_dir = '/home/ilya/Dropbox/papers/ogle2/data/new_features/'
# data_dir = '/home/ilya/github/ogle'
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
fit_params = {'validation_split': 0.25,
              'epochs': 1000,
              'batch_size': 1024}

combined_features = FeatureUnion([
    ('imputer', Imputer(missing_values='NaN',
                        strategy='median',
                        axis=0, verbose=2)),
    ('aef', Pipeline([('scaler', MinMaxScaler()),
                      ('ae', AEExtract(25, (90, 60),
                                       loss='mean_squared_error',
                                       fit_params=fit_params))]))])

# # Create new feature
# X_train_test = list()
# # First, transform our sample using folds
# for train_idx, test_idx in kfold.split(X, y):
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
#     X_train_ = combined_features.fit_transform(X_train, y_train)
#     X_test_ = combined_features.transform(X_test)
#     X_train_test.append((X_train_, X_test_))


def objective(space):
    pprint(space)
    clf = RandomForestClassifier(n_estimators=space['n_estimators'],
                                 max_depth=space['max_depth'],
                                 max_features=space['max_features'],
                                 min_samples_split=space['mss'],
                                 min_samples_leaf=space['msl'],
                                 class_weight={0: 1, 1: space['cw']},
                                 verbose=1, random_state=1, n_jobs=4)

    pipeline = Pipeline([('features', combined_features),
                         ('clf', clf)
                         ], memory='/home/ilya/github/ogle/cache')

    y_preds = cross_val_predict(pipeline, X, y, cv=kfold, n_jobs=1)
    CMs = list()
    # for i, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    for train_idx, test_idx in kfold.split(X, y):
        # CMs.append(confusion_matrix(y[test_idx], y_preds[test_idx]))
        # X_train, X_test = X_train_test[i]
        # y_train, y_test = y[train_idx], y[test_idx]
        # clf.fit(X_train, y_train)
        # y_preds = clf.predict(X_test)
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


space = {'n_estimators': hp.choice('n_estimators', np.arange(200, 501, 25,
                                                             dtype=int)),
         'max_depth': hp.choice('max_depth', np.arange(15, 20, dtype=int)),
         'max_features': hp.choice('max_features',
                                   np.arange(15, 30, dtype=int)),
         'mss': hp.choice('mss', np.arange(2, 40, 1, dtype=int)),
         'cw': hp.uniform('cw', 1, 5),
         'msl': hp.choice('msl', np.arange(1, 11, dtype=int))}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

pprint(hp.space_eval(space, best))
best_pars = hp.space_eval(space, best)