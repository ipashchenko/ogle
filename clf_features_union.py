# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn import decomposition
from utils import remove_correlated_features


target = 'variable'
data_dir = '/home/ilya/github/ogle'
df_vars = pd.read_pickle(os.path.join(data_dir, "features_vars.pkl"))
df_vars[target] = 1
df_const = pd.read_pickle(os.path.join(data_dir, "features_const.pkl"))
df_const[target] = 0
df = pd.concat((df_vars, df_const), ignore_index=True)
df = df.loc[:, ~df.columns.duplicated()]
df = remove_correlated_features(df, r=0.95)
features_names = list(df.columns)
features_names.remove(target)
l1_features =\
    np.array([u'CAR_sigma', u'Freq3_harmonics_rel_phase_3',
              u'value__mean_abs_change_quantiles__qh_0.6__ql_0.4',
              u'Freq3_harmonics_rel_phase_2',
              u'value__mean_abs_change_quantiles__qh_0.4__ql_0.4',
              u'value__mean_abs_change_quantiles__qh_0.8__ql_0.6',
              u'StructureFunction_index_31',
              u'value__mean_abs_change_quantiles__qh_0.6__ql_0.8',
              u'SmallKurtosis', u'Psi_eta', u'value__symmetry_looking__r_0.25',
              u'value__longest_strike_above_mean',
              u'value__mean_abs_change_quantiles__qh_0.2__ql_0.8',
              u'value__symmetry_looking__r_0.65',
              u'value__symmetry_looking__r_0.45',
              u'value__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_4__w_2',
              u'value__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_3__w_10',
              u'value__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_3__w_20',
              u'PeriodLS'],
              dtype='<U69')
X = df[l1_features].values
y = df[target].values


kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)


def objective(space):
    pprint(space)

    combined_features = FeatureUnion([
        ('raw', Pipeline([('variance_thresholder', VarianceThreshold()),
                          ('scaler', RobustScaler())])),
        ('poly', Pipeline([('variance_thresholder', VarianceThreshold()),
                           ('scaler', RobustScaler()),
                           ('polyf', PolynomialFeatures())])),
        ('pca', Pipeline([('variance_thresholder', VarianceThreshold()),
                          ('scaler', RobustScaler()),
                          ('pca', decomposition.PCA(n_components=5,
                                                    random_state=1))]))
    ])

    clf = LogisticRegression(C=space['C'], class_weight={0: 1, 1: space['cw']},
                             random_state=1, max_iter=300, n_jobs=1,
                             tol=10. ** (-3), penalty='l1')

    pipeline = Pipeline([('features', combined_features),
                         ('estimator', Pipeline([('scaler', RobustScaler()),
                                                 ('clf', clf)]))
                        ])

    # This for w/o CV-predict
    # CMs = list()
    #
    # for i, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    #     print("Doing fold {} of 4".format(i+1))
    #     X_train, y_train = X[train_idx], y[train_idx]
    #     X_test, y_test = X[test_idx], y[test_idx]
    #     pipeline.fit(X_train, y_train)
    #     y_pred = pipeline.predict(X_test)
    #     CMs.append(confusion_matrix(y_test, y_pred))
    #
    # CM = np.sum(CMs, axis=0)

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

space = {'C': hp.loguniform('C', -9.0, 3.0),
         'cw': hp.uniform('cw', 1, 30)}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

pprint(hyperopt.space_eval(space, best))
best_pars = hyperopt.space_eval(space, best)

# clf = LogisticRegression(C=0.011, class_weight={0: 1, 1: 4.36},
#                          random_state=1, max_iter=300, n_jobs=1,
#                          tol=10. ** (-2), penalty='l1')
#
# estimators = list()
# estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
#                                       axis=0, verbose=2)))
# estimators.append(('variance_thresholder', VarianceThreshold()))
# estimators.append(('scaler', RobustScaler()))
# estimators.append(('clf', clf))
# pipeline = Pipeline(estimators)
# pipeline.fit(X, y)