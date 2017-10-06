import numpy as np
from collections import OrderedDict
from toposort import toposort


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
