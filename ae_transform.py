import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.utils.validation import check_is_fitted
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from pomegranate import NaiveBayes, GaussianKernelDensity


# FIXME: Add fit arguments to constructor of ``AEExtractor`` or get how to pass
# parameters to nested pipelines.
# TODO: As i don't use "binary_crossentropy" loss - experiment with other
# activations.
class AEBase(BaseEstimator):
    def __init__(self, code_dim=30, dims=None, loss=None, early_stopping=5,
                 fit_params=None):
        """
        Autoencoder transformer. Note that because sigmoid activation used at
        the final layer of the decoder the data must be in (0, 1) range for
        "binary_crossentropy" loss to get positive values.

        :param code_dim:
            Dimension of code.
        :param dims:
            Iterable of dimensions for each of layers (except coding). E.g.
            ``[90, 60, 30]`` - means that there will be 3 layers with 90, 60 and
            30 number of units and coding layes, specified by ``code_dim``.
        :param loss: (optional)
            Keras loss function used for fitting. If ``None`` then
            ``mean_squared_error``. (default: ``None``)

        :note:
            To show pics type in command line: ``tensorboard --logdir=./logs``
        """
        super(AEBase, self).__init__()
        self.code_dim = code_dim
        self.dims = dims
        if loss is None:
            loss = 'mean_squared_error'
        self.loss = loss
        self.early_stopping = early_stopping
        self.fit_params = fit_params

    def fit(self, X, y=None, **fit_params):
        print("Shape X (before 0) : {}".format(X.shape))
        # Select only ``0``s to fit reconstruction
        X_0 = X[np.where(y == 0)]
        print("Shape X_0 : {}".format(X_0.shape))
        # X_0 = X
        ncol = X_0.shape[1]
        input_dim = Input(shape=(ncol,))
        encoding_dim = Input(shape=(self.code_dim,))

        try:
            x = Dense(self.dims[0], activation='relu')(input_dim)
            for dim in self.dims[1:]:
                x = Dense(dim, activation='relu')(x)
        # When ``self.dims=[]``
        except IndexError:
            x = input_dim
        encoded = Dense(self.code_dim, activation='linear')(x)

        try:
            x = Dense(self.dims[-1], activation='relu')(encoded)
            for dim in self.dims[-2::-1]:
                x = Dense(dim, activation='relu')(x)
        # When ``self.dims=[]``
        except IndexError:
            x = encoded
        decoded = Dense(ncol, activation='sigmoid')(x)

        encoder = Model(inputs=input_dim, outputs=encoded, name='encoder')
        self.encoder_ = encoder

        self.autoencoder_ = Model(inputs=input_dim, outputs=decoded)
        # Possible way:
        # self.autoencoder = Model(inputs=input_dim,
        #                          outputs=decoder(encoder(input_dim)))
        n_layers = len(self.dims)
        deco = self.autoencoder_.layers[-(n_layers+1)](encoding_dim)
        for i in range(1, n_layers+1)[::-1]:
            deco = self.autoencoder_.layers[-i](deco)
        decoder = Model(inputs=encoding_dim, outputs=deco, name='decoder')
        self.decoder_ = decoder

        self.autoencoder_.compile(optimizer='adam', loss=self.loss)

        # checkpointer = ModelCheckpoint(filepath="ae_model.hdf5",
        #                                verbose=0,
        #                                save_best_only=True)
        # tensorboard = TensorBoard(log_dir='./logs',
        #                           histogram_freq=0,
        #                           write_graph=True,
        #                           write_images=True)
        earlystopping = EarlyStopping(monitor='val_loss',
                                      patience=self.early_stopping,
                                      verbose=2)
        if not fit_params:
            print("Using __init__ fit_params")
            fit_params = self.fit_params
        history = self.autoencoder_.fit(X_0, X_0, shuffle=True, verbose=2,
                                        callbacks=[earlystopping],
                                        **fit_params).history
        self.history_ = history

        return self

    def summary(self):
        check_is_fitted(self, ['autoencoder_'])
        return self.autoencoder_.summary()


    # def plot_history(self, save_file=None):
    #     check_is_fitted(self, ['autoencoder_'])
    #     fig, axes = plt.subplots(1, 1)
    #     axes.plot(self.history_['loss'])
    #     axes.plot(self.history_['val_loss'])
    #     axes.set_title('model loss')
    #     axes.set_ylabel('loss')
    #     axes.set_xlabel('epoch')
    #     axes.legend(['train', 'test'], loc='upper right')
    #     fig.tight_layout()
    #     if save_file:
    #         fig.savefig(save_file, dpi=300)


class AETransform(AEBase, TransformerMixin):

    def transform(self, X, **transform_params):
        check_is_fitted(self, ['autoencoder_'])
        return self.autoencoder_.predict(X)


class AEExtract(AETransform):
    def __init__(self, code_dim=30, dims=None, loss=None, fit_params=None):
        """
        Autoencoder features extractor.

        :param code_dim:
            Dimension of code.
        :param dims:
            Iterable of dimensions for each of layers (except coding). E.g.
            ``[90, 60, 30]`` - means that there will be 3 layers with 90, 60 and
            30 number of units and coding layes, specified by ``code_dim``.

        :note:
            To show pics type in command line: ``tensorboard --logdir=./logs``
        """
        super(AEExtract, self).__init__(code_dim=code_dim, dims=dims, loss=loss,
                                        fit_params=fit_params)

    def transform(self, X, **transform_params):
        check_is_fitted(self, ['autoencoder_'])
        X_ = self.autoencoder_.predict(X)
        return np.mean(np.power(X - X_, 2), axis=1).reshape(-1, 1)


class AEClassifier(AEBase, ClassifierMixin):

    # FIXME: CHoose bandwidth for GM using CV
    def fit(self, X, y=None, **fit_params):
        # Fit Autoencoder
        super(AEClassifier, self).fit(X, y, **fit_params)
        # Fit GM NB on the results
        X_0 = X[np.where(y == 0)]
        X_1 = X[np.where(y == 1)]
        X_0_ = self.autoencoder_.predict(X_0)
        X_1_ = self.autoencoder_.predict(X_1)
        reconstr_err_0 = np.mean(np.power(X_0 - X_0_, 2), axis=1)
        reconstr_err_1 = np.mean(np.power(X_1 - X_1_, 2), axis=1)
        # Set weights of classes for fitting GM NB
        weights = np.ones(len(reconstr_err_0)+len(reconstr_err_1), dtype=float)
        weights[len(reconstr_err_0):] = (len(y)-np.count_nonzero(y))/float(np.count_nonzero(y))
        self.nb_clf_ = NaiveBayes([GaussianKernelDensity(bandwidth=0.05),
                                   GaussianKernelDensity(bandwidth=0.05)])
        data = np.array(list(reconstr_err_0) + list(reconstr_err_1))
        data = np.log(data)
        data = data.reshape(-1, 1)
        y_ = np.zeros(len(data))
        y_[-len(reconstr_err_1):] = np.ones(len(reconstr_err_1))
        self.nb_clf_.fit(data, y_, weights=weights)

    def predict_proba(self, X):
        check_is_fitted(self, ['autoencoder_', 'nb_clf_'])
        X_ = self.autoencoder_.predict(X)
        reconstr_err = np.mean(np.power(X - X_, 2), axis=1)
        reconstr_err = np.log(reconstr_err)
        reconstr_err = reconstr_err.reshape(-1, 1)
        return self.nb_clf_.predict_proba(reconstr_err)[:, 1]

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.array(probs > 0.5, dtype=int)


if __name__ == '__main__':
    import os
    import pandas as pd
    from utils import remove_correlated_features
    from sklearn.model_selection import train_test_split
    from tsfresh import select_features
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import MinMaxScaler


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
    # df = select_features(df, y)

    features_names = list(df.columns)
    features_names.remove(target)
    X = df[features_names].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        stratify=y)
    mms = MinMaxScaler()
    X_train = mms.fit_transform(X_train)
    X_test = mms.fit_transform(X_test)
    fit_params = {'validation_split': 0.2,
                  'epochs': 1000,
                  'batch_size': 1024}
    aeclf = AEClassifier(40, (350, 175, 85), loss='mean_squared_error',
                         fit_params=fit_params)
    aeclf.fit(X_train, y_train)
    y_pred = aeclf.predict(X_test)
    print(classification_report(y_test, y_pred))


