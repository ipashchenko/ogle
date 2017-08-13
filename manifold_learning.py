# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D


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

n_neighbors = 10


fig = plt.figure(figsize=(15, 8))
plt.suptitle("Manifold Learning with %i cases, %i variables, %i neighbors"
             % (len(y), np.count_nonzero(y), n_neighbors),
             fontsize=14)

estimators = list()
estimators.append(('variance_thresholder', VarianceThreshold()))
estimators.append(('scaler', StandardScaler()))
tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=30,
                     early_exaggeration=4)
estimators.append(('tsne', tsne))
pipeline = Pipeline(estimators)

X_ = pipeline.fit_transform(X)
X_0 = X_[y == 0]
X_1 = X_[y == 1]

ax = fig.add_subplot(241)
ax.scatter(X_0[:, 0], X_0[:, 1], color='g', alpha=0.5)
ax.scatter(X_1[:, 0], X_1[:, 1], color='r', alpha=0.5)
plt.title("t-SNE")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


estimators = list()
estimators.append(('variance_thresholder', VarianceThreshold()))
estimators.append(('scaler', StandardScaler()))
iso = manifold.Isomap(n_neighbors=n_neighbors, n_components=2)
estimators.append(('iso', iso))
pipeline = Pipeline(estimators)

X_ = pipeline.fit_transform(X)
X_0 = X_[y == 0]
X_1 = X_[y == 1]

ax = fig.add_subplot(242)
ax.scatter(X_0[:, 0], X_0[:, 1], color='g', alpha=0.5)
ax.scatter(X_1[:, 0], X_1[:, 1], color='r', alpha=0.5)
plt.title("IsoMap")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


estimators = list()
estimators.append(('variance_thresholder', VarianceThreshold()))
estimators.append(('scaler', StandardScaler()))
mds = manifold.MDS(2, max_iter=100, n_init=1)
estimators.append(('mds', mds))
pipeline = Pipeline(estimators)

X_ = pipeline.fit_transform(X)
X_0 = X_[y == 0]
X_1 = X_[y == 1]

ax = fig.add_subplot(243)
ax.scatter(X_0[:, 0], X_0[:, 1], color='g', alpha=0.5)
ax.scatter(X_1[:, 0], X_1[:, 1], color='r', alpha=0.5)
plt.title("MDS")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')



estimators = list()
estimators.append(('variance_thresholder', VarianceThreshold()))
estimators.append(('scaler', StandardScaler()))
se = manifold.SpectralEmbedding(n_components=2,
                                n_neighbors=n_neighbors)
estimators.append(('se', se))
pipeline = Pipeline(estimators)

X_ = pipeline.fit_transform(X)
X_0 = X_[y == 0]
X_1 = X_[y == 1]

ax = fig.add_subplot(244)
ax.scatter(X_0[:, 0], X_0[:, 1], color='g', alpha=0.5)
ax.scatter(X_1[:, 0], X_1[:, 1], color='r', alpha=0.5)
plt.title("SE")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


# All suck
methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']
for i, method in enumerate(methods):
    print("Using {} with method {}".format(labels[i], method))

    estimators = list()
    estimators.append(('variance_thresholder', VarianceThreshold()))
    estimators.append(('scaler', StandardScaler()))
    esolver = 'auto'
    if method == 'hessian' or method == 'ltsa':
        esolver = 'dense'
    lle = manifold.LocallyLinearEmbedding(n_neighbors, 2, eigen_solver=esolver,
                                          method=method)
    estimators.append(('lle', lle))
    pipeline = Pipeline(estimators)

    X_ = pipeline.fit_transform(X)
    X_0 = X_[y == 0]
    X_1 = X_[y == 1]

    ax = fig.add_subplot(245+i)
    ax.scatter(X_0[:, 0], X_0[:, 1], color='g', alpha=0.5)
    ax.scatter(X_1[:, 0], X_1[:, 1], color='r', alpha=0.5)
    plt.title(labels[i])
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

plt.show()


fig = plt.figure(figsize=(15, 8))
estimators = list()
estimators.append(('variance_thresholder', VarianceThreshold()))
estimators.append(('scaler', StandardScaler()))
tsne = manifold.TSNE(n_components=3, random_state=0, perplexity=30,
                     early_exaggeration=4, init='pca', method='exact')
estimators.append(('tsne', tsne))
pipeline = Pipeline(estimators)

X_ = pipeline.fit_transform(X)
X_0 = X_[y == 0]
X_1 = X_[y == 1]

ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_0[:, 0], X_0[:, 1], X_0[:, 2], color='g', alpha=0.15)
ax.scatter(X_1[:, 0], X_1[:, 1], X_1[:, 2], color='r', alpha=0.5)
plt.title("t-SNE with 3 components")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')