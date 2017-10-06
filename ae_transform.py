import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.utils.validation import check_is_fitted
from sklearn.base import TransformerMixin, BaseEstimator
from keras.callbacks import EarlyStopping


# FIXME: Add fit arguments to constructor of ``AEExtractor`` or get how to pass
# parameters to nested pipelines.
# TODO: As i don't use "binary_crossentropy" loss - experiment with other
# activations.
class AETransform(TransformerMixin, BaseEstimator):
    def __init__(self, code_dim=30, dims=None, loss=None, fit_params=None):
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

        :note:
            To show pics type in command line: ``tensorboard --logdir=./logs``
        """
        super(AETransform, self).__init__()
        self.code_dim = code_dim
        self.dims = dims
        if loss is None:
            loss = 'mean_absolute_error'
        self.loss = loss
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
        earlystopping = EarlyStopping(monitor='val_loss', patience=5,
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

    def transform(self, X, **transform_params):
        check_is_fitted(self, ['autoencoder_'])
        return self.autoencoder_.predict(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y=y, **fit_params).transform(X)

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