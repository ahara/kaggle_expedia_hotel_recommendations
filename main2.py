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
    selected_cols = ['user_location_city', 'user_location_country', 'user_location_region',
                     'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt',
                     'srch_destination_id', 'srch_destination_type_id',
                     'hotel_continent', 'hotel_country', 'hotel_market',
                     'channel', 'is_mobile', 'is_package', 'site_name']

    param = {'objective': 'multi:softprob', 'num_class': 100, 'nthreaad': 7, 'eta': 0.2,
             'subsample': 0.5, 'max_depth': 10, 'silent': 1}
    dtrain, dtest, dbooked = None, None, None

    for i, chunk in enumerate(reader):
        x, y = pd.DataFrame(), pd.Series(dtype=int)
        print 'Processing chunk', i + 1
        x = chunk.loc[:, selected_cols]
        x['date_time_month'] = map(lambda d: int(d[5:7]), chunk['date_time'])
        x['srch_ci_month'] = map(lambda d: int(d[5:7]) if isinstance(d, str) else np.nan, chunk['srch_ci'])
        x['srch_co_month'] = map(lambda d: int(d[5:7]) if isinstance(d, str) else np.nan, chunk['srch_co'])
        if 'hotel_cluster' in chunk.columns:
            y = chunk['hotel_cluster']
        #break  # Read only 1 chunk
        booked = chunk['is_booking'].values == 1
        mb = np.mean(booked)
        dtest = xgb.DMatrix(x.as_matrix(), y.values, missing=np.nan, weight=booked*(1-mb)+(1-booked)*mb)
        dbooked = xgb.DMatrix(x.as_matrix()[booked, :], y.values[booked], missing=np.nan)
        if i > 0:
            evals = [(dtest, 'eval'), (dbooked, 'booked'), (dtrain, 'train')]
            bst = xgb.train(param, dtrain, num_boost_round=10, feval=map5eval, maximize=True,
                            evals=evals, xgb_model='xgb2.model' if i > 1 else None)
            bst.save_model('xgb2.model')
        dtrain = dtest

    return x, y


if __name__ == '__main__':
    train, labels = build_features(hc.TRAIN_FILE)
    exit(0)

    param = {'objective': 'multi:softprob', 'num_class': 100, 'nthreaad': 2, 'eta': 0.3,
             'subsample': 0.5, 'max_depth': 8, 'silent': 1}
    s = 10**6
    dtrain = xgb.DMatrix(train.as_matrix()[:s, :], labels.values[:s], missing=np.nan)
    dtest = xgb.DMatrix(train.as_matrix()[s:, :], labels.values[s:], missing=np.nan)
    bst = xgb.train(param, dtrain, num_boost_round=10, feval=map5eval, maximize=True,
                    evals=[(dtest, 'eval'), (dtrain, 'train')])
    #cv = xgb.cv(param, dtrain, num_boost_round=200, nfold=2, seed=17, show_progress=False,
    #            feval=map5eval)
    #print cv.values
