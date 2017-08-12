from __future__ import print_function
import os
import pandas as pd
import numpy as np
from lc import LC
import multiprocessing
import multiprocessing.pool


class NoDaemonProcess(multiprocessing.Process):
    """
    From https://stackoverflow.com/a/8963618
    """
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def calculate_features_for_file(fname):
    lc = LC(fname)
    features_fats = lc.generate_features_fats()
    features_tsfresh = lc.generate_features_tsfresh()
    lc.add_features(features_fats)
    lc.add_features(features_tsfresh)
    return lc.features


def calculate_features_for_files(fnames, threads=4):
    pool = MyPool(processes=threads)
    result = pool.map(calculate_features_for_file, fnames)
    pool.close()
    pool.join()
    return pd.concat(result, axis=0)


data_dir = "/home/ilya/Dropbox/papers/ogle2/data"
vars_file = "/home/ilya/Downloads/LMC_SC20__corrected_list_of_variables/raw_index_values/vast_lightcurve_statistics_variables_only.log"
const_file = "/home/ilya/Downloads/LMC_SC20__corrected_list_of_variables/raw_index_values/vast_lightcurve_statistics_constant_only.log"
vars_fname_file = os.path.join(data_dir, "LMC_SC20_vars_lcfnames.txt")
const_fname_file = os.path.join(data_dir, "LMC_SC20_const_lcfnames.txt")

for source_file, destination_file in zip((vars_file, const_file),
                                         (vars_fname_file, const_fname_file)):
    data = np.loadtxt(source_file, dtype=str)
    names = list()
    for name in data[:, 4]:
        names.append(name[4:])
    with open(destination_file, 'w') as fo:
        for name in names:
            print(name, file=fo)

# # For variables
# with open(vars_fname_file, 'r') as fo:
#     vars_fnames = fo.readlines()
# vars_fnames = [var_fname.strip('\n') for var_fname in vars_fnames]
# vars_fnames = [os.path.join(data_dir, var_fname) for var_fname in vars_fnames]
# import time
# t0 = time.time()
# features_df = calculate_features_for_files(vars_fnames, threads=5)
# print(time.time() - t0)
#
# features_df.to_pickle(os.path.join(data_dir, 'features_vars.pkl'))

# For constant
import time
with open(const_fname_file, 'r') as fo:
    consts_fnames = fo.readlines()
consts_fnames = [const_fname.strip('\n') for const_fname in consts_fnames]
consts_fnames = [os.path.join(data_dir, const_fname) for const_fname in consts_fnames]
t0 = time.time()
features_df = calculate_features_for_files(consts_fnames, threads=5)
print(time.time() - t0)

features_df.to_pickle(os.path.join(data_dir, 'features_const.pkl'))
