__author__ = 'Adam Harasimowicz'

import csv
import numpy as np
import pandas as pd
import pdb
import xgboost as xgb

import helpers.consts as hc
from helpers.metrics import map5eval


def build_model(file_path, feature_names, max_chunks=None):
    reader = pd.read_csv(file_path, chunksize=10**5)

    param = {'objective': 'multi:softprob', 'num_class': 100, 'nthread': 2, 'eta': 0.2,
             'subsample': 0.5, 'max_depth': 10, 'silent': 1}
    dtrain, dtest, dbooked = None, None, None

    for i, chunk in enumerate(reader):
        if max_chunks is not None and i > max_chunks:
            break
        x, y = pd.DataFrame(), pd.Series(dtype=int)
        print 'Processing chunk', i + 1
        x = chunk.loc[:, feature_names]
        if 'hotel_cluster' in chunk.columns:
            y = chunk['hotel_cluster']
        booked = chunk['is_booking'].values == 1
        mb = np.mean(booked)
        weights = booked * (1 - mb) + (1 - booked) * mb
        dtest = xgb.DMatrix(x.as_matrix(), y.values, missing=np.nan, weight=weights)
        dbooked = xgb.DMatrix(x.as_matrix()[booked, :], y.values[booked], missing=np.nan)
        if i > 0:
            evals = [(dtest, 'eval'), (dbooked, 'booked'), (dtrain, 'train')]
            bst = xgb.train(param, dtrain, num_boost_round=10, feval=map5eval, maximize=True,
                            evals=evals, xgb_model='xgb_%d.model' % (i - 1) if i > 1 else None)
            bst.save_model('xgb_%d.model' % i)
        dtrain = dtest


def make_predictions(file_path, feature_names, model_path, output_file):
    reader = pd.read_csv(file_path, chunksize=10**5)
    clf = xgb.Booster(model_file=model_path)

    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for i, chunk in enumerate(reader):
            print 'Processing chunk', i + 1
            x = chunk.loc[:, feature_names]
            data = xgb.DMatrix(x.as_matrix(), missing=np.nan)
            predictions = clf.predict(data)
            top_clusters = np.fliplr(predictions.argsort(axis=1)[:, -hc.TOP_N:])
            writer.writerows(top_clusters)


if __name__ == '__main__':
    selected_cols = ['user_location_city', 'user_location_country', 'user_location_region',
                     'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt',
                     'srch_destination_id', 'srch_destination_type_id',
                     'hotel_continent', 'hotel_country', 'hotel_market',
                     'channel', 'is_mobile', 'is_package', 'site_name']
    #build_model(hc.TRAIN_FILE, selected_cols)
    #make_predictions(hc.TEST_FILE, selected_cols, 'xgb24M.model', 'xgb_test_preds.csv')
    make_predictions(hc.TRAIN_FILE, selected_cols, 'xgb24M.model', 'xgb_train_preds.csv')
