import numpy as np
from collections import OrderedDict
from toposort import toposort
from keras.layers import Input, Dense
from keras.models import Model
from sklearn import manifold
from sklearn.base import TransformerMixin, BaseEstimator


def find_correlated_features(df, r=0.99):
    corr = df.corr()
    corr.loc[:, :] = np.tril(corr, k=-1)
    corr = corr.stack()
    pairs = corr[corr > r].to_dict().keys()
    names = df.columns
    names_correlated = OrderedDict()
    names_uncorrelated = list()
    for name in names:
        names_correlated[name] = set()
        for pair in pairs:
            if pair[0] == name:
                names_correlated[name].add(pair[1])
        if not names_correlated[name]:
            names_uncorrelated.append(name)

    for name in names_uncorrelated:
        names_correlated.pop(name)

    return names_correlated


# def remove_correlated_features(df, r=0.99, inplace=False):
#     names_correlated = find_correlated_features(df, r=r)
#     sorted_names = list(toposort(names_correlated))
#     for names in sorted_names:
#         if inplace:
#             df.drop(names, axis=1, inplace=True)
#         else:
#             df = df.drop(names, axis=1)
#     return df


def remove_correlated_features(df, r=0.99):
    names_correlated = find_correlated_features(df, r=r)
    sorted_names = list(toposort(names_correlated))

    removed_features = list()

    for names in sorted_names:
        for name in names:
            # If feature not on the left - remove it
            if name not in names_correlated.keys():
                for key, value in names_correlated.items():
                    try:
                        names_correlated[key].remove(name)
                        removed_features.append(name)
                    except KeyError:
                        pass

    for name in names_correlated.keys():
        if not names_correlated[name]:
            for key, value in names_correlated.items():
                try:
                    names_correlated[key].remove(name)
                    removed_features.append(name)
                    names_correlated.pop(name)
                except KeyError:
                    pass

    names = list(set(removed_features))
    print("Removed features : ")
    print(names)

    return df.drop(names, axis=1)


# TODO: Add # of layers and their dims, activation, etc. as parameters in
# constructor
class AETransform(TransformerMixin, BaseEstimator):
    """
    Autoencoder transformer. Note that because sigmoid activation used at the
    final layer of the decoder the data must be in (0, 1) range for
    "binary_crossentropy" loss to get positive values.
    """
    def __init__(self, dim=15):
        super(AETransform, self).__init__()
        self.encoder = None
        self.autoencoder = None
        self.dim = dim

    def fit(self, X, y=None, **fit_params):
        ncol = X.shape[1]
        input_dim = Input(shape=(ncol,))
        encoding_dim = self.dim

        # Encoder layers
        x = Dense(encoding_dim*6, activation='relu')(input_dim)
        x = Dense(encoding_dim*4, activation='relu')(x)
        x = Dense(encoding_dim*2, activation='relu')(x)
        encoded = Dense(encoding_dim, activation='linear')(x)
        # encoded = Dense(encoding_dim, activation='relu')(input_dim)

        # Decoder layers
        # x = Dense(encoding_dim*2, activation='relu')(encoded)
        x = Dense(encoding_dim*2, activation='relu')(encoded)
        x = Dense(encoding_dim*4, activation='relu')(x)
        x = Dense(encoding_dim*6, activation='relu')(x)
        decoded = Dense(ncol, activation='sigmoid')(x)
        # decoded = Dense(ncol, activation='sigmoid')(encoded)

        # Combine encoder & decoder into autoencoder model
        self.autoencoder = Model(input=input_dim, output=decoded)

        # Configure & train autoencoder
        self.autoencoder.compile(optimizer='adam',
                                 loss='binary_crossentropy')
        self.autoencoder.fit(X, X, nb_epoch=1000, batch_size=2048, shuffle=True,
                             verbose=2, **fit_params)
        # Encoder to extract reduced dimensions from the above autoencoder
        self.encoder = Model(input=input_dim, output=encoded)
        return self

    def summary(self):
        if self.autoencoder is not None:
            return self.autoencoder.summary()

    def transform(self, X, **transform_params):
        return self.encoder.predict(X)

    # def fit_transform(self, X, y=None, **fit_params):
    #     return self.fit(X, **fit_params).transform(X)


class TSNETransform(TransformerMixin):
    def __init__(self, n_components=2, random_state=0):
        super(TSNETransform, self).__init__()
        self.tsne = manifold.TSNE(n_components=n_components,
                                  random_state=random_state, perplexity=30,
                                  early_exaggeration=4)

    def fit(self, X, y=None, **fit_params):
        self.tsne.fit(X)

    def transform(self, X, **transform_params):
        return self.tsne.fit_transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.tsne.fit_transform(X)