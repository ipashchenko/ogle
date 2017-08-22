from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import os
import pandas as pd
from tempfile import mkdtemp
from shutil import rmtree
from sklearn import manifold
from sklearn.base import TransformerMixin
from pprint import pprint
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, RobustScaler, minmax_scale
from sklearn.feature_selection import VarianceThreshold
from utils import remove_correlated_features, AETransform, TSNETransform


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
X = df[features_names].values
X = VarianceThreshold().fit_transform(X)
X = minmax_scale(X)
y = df[target].values

cachedir = mkdtemp()

kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)


def objective(space):
    pprint(space)
    clf = RandomForestClassifier(class_weight={0: 1, 1: space['cw']},
                                 n_estimators=200, min_samples_leaf=5,
                                 min_impurity_split=10, max_features=15,
                                 max_depth=15)

    estimators = list()
    # estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
    #                                       axis=0, verbose=2)))
    # estimators.append(('variance_thresholder', VarianceThreshold()))
    # estimators.append(('scaler', RobustScaler()))
    estimators.append(('aue', AETransform()))
    # estimators.append(('tsne ', TSNETransform()))
    estimators.append(('clf', clf))
    pipeline = Pipeline(estimators)

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

space = {'cw': hp.uniform('cw', 1, 30)}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials)

rmtree(cachedir)

pprint(hyperopt.space_eval(space, best))
best_pars = hyperopt.space_eval(space, best)