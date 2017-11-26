import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import os
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe
from keras import callbacks
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.regularizers import l1
from keras.optimizers import SGD
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import remove_correlated_features
from tsfresh import select_features
import pandas as pd



class Metrics(callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        self.f1s = f1_score(targ, predict)
        return


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
    model = Sequential()
    model.add(Dense(3, input_dim=3, kernel_initializer="normal",
                    activation="relu"))
    model.add(Dense(space["n1"], kernel_initializer="normal",
                    kernel_constraint=maxnorm(space["maxnorm1"]),
                    kernel_regularizer=l1(space["kernel_reg1_l1"])))
    if space["n_hlayers"] > 1:
        if space["n_hlayers"] == 2:
            model.add(Dense(space["n2"], kernel_initializer="normal"))
        elif space["n_hlayers"] == 3:
            model.add(Dense(space["n2"], kernel_initializer="normal"))
            model.add(Dense(space["n3"], kernel_initializer="normal"))
        elif space["n_hlayers"] == 4:
            model.add(Dense("n2", kernel_initializer="normal"))
            model.add(Dense("n3", kernel_initializer="normal"))
            model.add(Dense("n4", kernel_initializer="normal"))
        else:
            raise Exception("Should be 1, 2, 3 or 4 HL")
    model.add(Dense(1, init='normal', activation='sigmoid'))

    # Compile model
    learning_rate = 0.2
    decay_rate = 0.001
    momentum = 0.9
    sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=momentum,
              nesterov=False)
    f1 = Metrics()
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=[f1])

    earlyStopping = callbacks.EarlyStopping(monitor=f1, patience=10,
                                            verbose=1, mode='auto')

    CMs = list()
    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train,
                                                            test_size=0.5,
                                                            stratify=y_train,
                                                            random_state=1)

        estimators = list()
        estimators.append(('scaler', StandardScaler()))
        estimators.append(('clf', model))
        pipeline = Pipeline(estimators)
        fit_params = {"clf__validation_data": (X_val, y_val),
                      "clf__batch_size": space["batch_size"],
                      "clf__nb_epochs": 1000,
                      "clf__varbose": 2,
                      "clf__calbacks": [earlyStopping],
                      "clf__class_weight": {0: 1, 1: space['cw']}}
        pipeline.fit(X_train_, y_train_, **fit_params)
        # model.fit(X_train, y_train,
        #           batch_size=space['batch_size'],
        #           nb_epoch=1000,
        #           verbose=2,
        #           validation_data=(X_test, y_test),
        #           callbacks=[earlyStopping],
        #           class_weight={0: 1, 1: space['cw']})
        y_pred = pipeline.predict(X_test, batch_size=space['batch_size'])

        # y_pred = [1. if y_ > 0.5 else 0. for y_ in y_pred]

        CMs.append(confusion_matrix(y_test, y_pred))

    CM = np.sum(CMs, axis=0)

    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print("TP = {}".format(TP))
    print("FP = {}".format(FP))
    print("FN = {}".format(FN))

    f1 = 2. * TP / (2. * TP + FP + FN)
    print("F1: ", f1)

    print("== F1 : {} ==".format(f1))
    return {'loss': 1-f1, 'status': STATUS_OK}

space = {
        'Dropout': hp.quniform('Dropout', 0., 0.5, 0.05),
        'Dense': hp.choice('Dense', (9, 13, 18, 22, 27)),
        'Dropout_1': hp.quniform('Dropout_1', 0., 0.5, 0.05),
        # 'conditional': hp.choice('conditional', [{'n_layers': 'two'},
        #                                          {'n_layers': 'three',
        #                                           'Dense_2': hp.choice('Dense_2', (9, 18, 36)),
        #                                           'Dropout_2': hp.uniform('Dropout_2', 0., 1.),
        #                                           'w3': hp.choice('w3', (1, 2, 3, 5, 7))}]),
        'use_3_layers': hp.choice('use_3_layers', [False,
                                                   {'Dense_2': hp.choice('Dense_2', (9, 13, 18, 22, 27)),
                                                    'Dropout_2': hp.quniform('Dropout_2', 0., 0.5, 0.05),
                                                    'w2': hp.choice('w2', (2, 3, 4, 5))},
                                                   {'Dense_2': hp.choice('Dense_2', (9, 13, 18, 22, 27)),
                                                    'Dropout_2': hp.quniform('Dropout_2', 0., 0.5, 0.05),
                                                    'w2': hp.choice('w2', (2, 3, 4, 5)),
                                                    'Dense_3': hp.choice('Dense_2', (9, 13, 18, 22, 27)),
                                                    'Dropout_3': hp.quniform('Dropout_2', 0., 0.5, 0.05),
                                                    'w3': hp.choice('w3', (2, 3, 4, 5))}]),
        'w1': hp.choice('w1', (2, 3, 4, 5)),
        'w2': hp.choice('w2', (2, 3, 4, 5)),
        # 'momentum': hp.quniform('momentum', 0.5, 0.95, 0.05),
        # 'cw': hp.qloguniform('cw', 0, 6, 1),
        'cw': hp.quniform('cw', 1, 20, 1),
        'batch_size': hp.choice('batch_size', (256, 512, 1024))
    }

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=300,
            trials=trials)

print(hyperopt.space_eval(space, best))
best_pars = hyperopt.space_eval(space, best)