# -*- coding: utf-8 -*-
# This differs from clf_xgb_cv_f1.py that it can contain any transformations of
# features while last one can't (because it uses xgb.cv that

import os
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.metrics import f1_score
import xgboost as xgb
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix


target = 'variable'
data_dir = '/home/ilya/github/ogle'
df_vars = pd.read_pickle(os.path.join(data_dir, "features_vars.pkl"))
df_vars[target] = 1
df_const = pd.read_pickle(os.path.join(data_dir, "features_const.pkl"))
df_const[target] = 0
df = pd.concat((df_vars, df_const), ignore_index=True)
df = df.loc[:, ~df.columns.duplicated()]
features_names = list(df.columns)
features_names.remove(target)
X = df[features_names].values
y = df[target].values

# This should leak info from train to test samples because it works on whole
# feature (remove it or no)
# X = VarianceThreshold().fit_transform(X)

kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)


def xg_f1(y, t):
    t = t.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y]
    return 'f1', 1-f1_score(t, y_bin)


def objective(space):
    pprint(space)
    clf = xgb.XGBClassifier(n_estimators=10000, learning_rate=space['lr'],
                            max_depth=space['max_depth'],
                            min_child_weight=space['min_child_weight'],
                            subsample=space['subsample'],
                            colsample_bytree=space['colsample_bytree'],
                            colsample_bylevel=space['colsample_bylevel'],
                            gamma=space['gamma'],
                            scale_pos_weight=space['scale_pos_weight'],
                            max_delta_step=space['mds'],
                            seed=1)
    # Try using pipeline
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    estimators.append(('var_threshold', VarianceThreshold()))
    estimators.append(('clf', clf))
    pipeline = Pipeline(estimators)

    best_n = ""
    CMs = list()
    for train_indx, test_indx in kfold.split(X, y):
        X_train, y_train = X[train_indx], y[train_indx]
        X_valid, y_valid = X[test_indx], y[test_indx]

        for name, transform in pipeline.steps[:-1]:
            transform.fit(X_train)
            X_valid_ = transform.transform(X_valid)
            X_train_ = transform.transform(X_train)

        eval_set = [(X_train_, y_train),
                    (X_valid_, y_valid)]

        pipeline.fit(X_train, y_train,
                     clf__eval_set=eval_set, clf__eval_metric=xg_f1,
                     clf__early_stopping_rounds=50)

        pred = pipeline.predict(X_valid)
        CMs.append(confusion_matrix(y[test_indx], pred))
        best_n = best_n + " " + str(clf.best_ntree_limit)

    CM = np.sum(CMs, axis=0)

    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print("TP = {}".format(TP))
    print("FP = {}".format(FP))
    print("FN = {}".format(FN))

    f1 = 2. * TP / (2. * TP + FP + FN)
    print("=== F1 : {} ===".format(f1))

    return {'loss': 1-f1, 'status': STATUS_OK,
            'attachments': {'best_n': best_n}}


space = {'max_depth': hp.choice("x_max_depth", np.arange(4, 12, 1, dtype=int)),
         'min_child_weight': hp.quniform('x_min_child', 1, 20, 1),
         'subsample': hp.quniform('x_subsample', 0.5, 1, 0.025),
         'colsample_bytree': hp.quniform('x_csbtree', 0.5, 1, 0.025),
         'colsample_bylevel': hp.quniform('x_csblevel', 0.5, 1, 0.025),
         'gamma': hp.uniform('x_gamma', 0.0, 20),
         'scale_pos_weight': hp.quniform('x_spweight', 1, 30, 2),
         'mds': hp.choice('mds', np.arange(0, 11, dtype=int)),
         'lr': hp.loguniform('lr', -4.7, -1.25)
         }


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

pprint(hyperopt.space_eval(space, best))

best_pars = hyperopt.space_eval(space, best)
best_n = trials.attachments['ATTACH::{}::best_n'.format(trials.best_trial['tid'])]
best_n = int(best_n)

clf = xgb.XGBClassifier(n_estimators=int(1.25 * best_n),
                        learning_rate=best_pars['lr'],
                        max_depth=best_pars['max_depth'],
                        min_child_weight=best_pars['min_child_weight'],
                        subsample=best_pars['subsample'],
                        colsample_bytree=best_pars['colsample_bytree'],
                        colsample_bylevel=best_pars['colsample_bylevel'],
                        gamma=best_pars['gamma'],
                        scale_pos_weight=best_pars['scale_pos_weight'],
                        seed=1)

estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('clf', clf))
pipeline = Pipeline(estimators)

# Fit classifier with best hyperparameters on the whole data set
pipeline.fit(X, y)