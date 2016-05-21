__author__ = 'Adam Harasimowicz'

import numpy as np
import pandas as pd
import pdb
import xgboost as xgb

import helpers.consts as hc
from helpers.metrics import map5eval


def build_features(file_path):
    x, y = pd.DataFrame(), pd.Series(dtype=int)
    reader = pd.read_csv(file_path, chunksize=10**6)
    selected_cols = ['user_location_city', 'orig_destination_distance', 'hotel_country',
                     'hotel_market']

    for i, chunk in enumerate(reader):
        print 'Processing chunk', i + 1
        x = x.append(chunk.loc[:, selected_cols], ignore_index=True)
        if 'hotel_cluster' in chunk.columns:
            y = y.append(chunk['hotel_cluster'])
        break
    print x.shape, y.shape

    return x, y


if __name__ == '__main__':
    train, labels = build_features(hc.TRAIN_FILE)

    param = {'objective': 'multi:softprob', 'num_class': 100, 'nthreaad': 2, 'eta': 0.5,
             'subsample': 0.5, 'max_depth': 8}
    dtrain = xgb.DMatrix(train.as_matrix(), labels.values, missing=np.nan)
    cv = xgb.cv(param, dtrain, num_boost_round=200, nfold=2, seed=17, show_progress=False,
                feval=map5eval)
    print cv.values
