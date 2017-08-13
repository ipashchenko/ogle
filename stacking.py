import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class StackingEstimator(object):
    """
    Stacking method. Use prediction of several base classifiers as features with
    the original (raw) features.

    Example usage:

    >>> meta_pipeline = Pipeline(estimators)
    >>> base_estimators = [pipeline_rf, pipeline_lr, pipeline_xgb, pipeline_knn,
                           pipeline_nn, pipeline_svm]
    >>> stacking_ensemble = StackingEstimator(base_estimators, meta_pipeline)
    >>> stacking_ensemble.fit(X_train, y_train, seed=1)
    >>> y_pred = stacking_ensemble.predict(X_test)

    """
    def __init__(self, base_estimators, meta_estimator):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator

    def fit(self, X, y, seed=1):
        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.50,
                                    random_state=seed)
        for train_index1, train_index2 in cv.split(X, y):
            X_train1, X_train2 = X[train_index1], X[train_index2]
            y_train1, y_train2 = y[train_index1], y[train_index2]

        # Fit base learners using (X_train1, y_train1)
        for base_estimator in self.base_estimators:
            base_estimator.fit(X_train1, y_train1)

        # Predict responses for X_train2 using learned estimators
        y_predicted = list()
        for base_estimator in self.base_estimators:
            y_pred = base_estimator.predict_proba(X_train2)
            y_predicted.append(y_pred)

        # Stack predictions and X_train2 (#samples, #features)
        X_stacked = X_train2.copy()
        # X_stacked = list()
        for y_pred in y_predicted:
            # X_stacked.append(y_pred[:, 1])
            X_stacked = np.hstack((X_stacked, y_pred[:, 1][..., np.newaxis]))
        # X_stacked = np.dstack(X_stacked)[0, ...]

        # Fit meta estimator on stacked data
        self.meta_estimator.fit(X_stacked, y_train2)

    def predict(self, X):
        # First find predictions of base learners
        y_predicted = list()
        for base_estimator in self.base_estimators:
            y_pred = base_estimator.predict_proba(X)
            y_predicted.append(y_pred)

        # Stack predictions with original data
        X_stacked = X.copy()
        # X_stacked = list()
        for y_pred in y_predicted:
            # X_stacked.append(y_pred[:, 1])
            X_stacked = np.hstack((X_stacked, y_pred[:, 1][..., np.newaxis]))
        # X_stacked = np.dstack(X_stacked)[0, ...]

        # Predict using meta learner
        return self.meta_estimator.predict(X_stacked)
