# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import manifold
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from sklearn.feature_selection import VarianceThreshold
from utils import remove_correlated_features, AETransform


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

X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)


estimators = list()
# estimators.append(('variance_thresholder', VarianceThreshold()))
# estimators.append(('scaler', MinMaxScaler()))
estimators.append(('ae', AETransform(dim=15)))
# tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=30,
#                      early_exaggeration=4)
# estimators.append(('tsne', tsne))
pipeline = Pipeline(estimators)
# pipeline.set_params(ae__fit_params={'validation_data': (X_test, X_test)})

pipeline.fit(X_train, ae__validation_data=(X_test, X_test))