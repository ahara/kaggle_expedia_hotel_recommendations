import csv
import cPickle
import numpy as np
import pandas as pd
import xgboost as xgb

import helpers.consts as hc
from helpers.metrics import map5eval, map5


def build_model(input_file, feature_names, max_chunks=None, chunk_size=10**5):
    """
    Builds XGBoost model iteratively adding new trees for every new data
    chunk. During training model is validated based on data in next chunk.
    After each training iteration, model is saved on disk.


    Parameters
    ----------
    input_file : string
        Path to CSV file with training set

    feature_names : list
        Selected column names which will be used for model training

    max_chunks: int or None, optional,  default: None
        Number of chunk which will be read from input file. If None then
        all will content is used.

    chunk_size : int, optional, default: 10^5
        Number of records in one data chunk. Higher values mean higher memory
        consumption and better model regularization. While low values
        can lead to overfitting. Thus, in such case XGBoost depth should be
        decreased or additional regularization parameters set. (Details:
        https://github.com/dmlc/xgboost/blob/master/doc/parameter.md)

    """
    dtrain = None  # Stores feature matrix used for model training
    model_params = {'objective': 'multi:softprob', 'num_class': 100, 'nthread': 8,
                    'eta': 0.05, 'subsample': 0.5, 'max_depth': 10, 'silent': 1}

    reader = pd.read_csv(input_file, chunksize=chunk_size)

    i = 0
    best_ntrees = []

    for chunk in reader:
        # Early stopping
        if max_chunks is not None and i > max_chunks:
            break

        print 'Processing chunk', i + 1
        x = chunk.loc[:, feature_names]
        y = chunk['hotel_cluster']

        # Balance weights between categories of booking/non-booking events
        booked = chunk['is_booking'].values == 1
        mb = np.mean(booked)
        weights = booked * (1 - mb) + (1 - booked) * mb

        # Validation sets with all events and only booking events
        dval = xgb.DMatrix(x.as_matrix(), y.values, missing=np.nan, weight=weights)
        dbooked = xgb.DMatrix(x.as_matrix()[booked, :], y.values[booked], missing=np.nan)

        # Skip training in the first iteration to have loaded two chunks - one
        # for training and one for validation
        if i > 0:
            evals = [(dtrain, 'train'), (dval, 'val')]
            bst = xgb.train(model_params, dtrain, num_boost_round=100, feval=map5eval,
                            maximize=True, evals=evals, early_stopping_rounds=5)
            bst.save_model('xgb_%d.model' % (i - 1))
            best_ntrees.append(bst.best_iteration)
            predictions = None
            for j in range(i):
                clf = xgb.Booster(model_file='xgb_%d.model' % j)
                if predictions is None:
                    predictions = clf.predict(dval, ntree_limit=best_ntrees[j])
                else:
                    predictions + clf.predict(dval, ntree_limit=best_ntrees[j])
            print map5(predictions, y.values)
        dtrain = dval
        i += 1

    evals = [(dtrain, 'train')]
    bst = xgb.train(model_params, dtrain, num_boost_round=100, feval=map5eval,
                    maximize=True, evals=evals, early_stopping_rounds=5)
    bst.save_model('xgb_%d.model' % (i - 1))
    best_ntrees.append(bst.best_iteration)
    cPickle.dump(best_ntrees, open('best_ntrees', 'w'))


def make_predictions(input_file, feature_names, model_path, output_file):
    """
    Loads already trained model and generate predictions of top n hotel
    clusters. Predictions are saved in CSV file without header or any
    identifiers. Thus, records should be identified based on their
    position.

    Parameters
    ----------
    input_file : string
        Path to CSV with input data

    feature_names : list
        Column names which were used for model training

    model_path : string
        Path to saved XGBoost model

    output_file : string
        Path to CSV file where predictions will be saved

    """
    reader = pd.read_csv(input_file, chunksize=10**5)
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
    # Build model and generate predictions for test set
    selected_cols = ['user_location_city', 'user_location_country', 'user_location_region',
                     'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt',
                     'srch_destination_id', 'srch_destination_type_id',
                     'hotel_continent', 'hotel_country', 'hotel_market',
                     'channel', 'is_mobile', 'is_package', 'site_name']
    max_chunks = None  # Use only part of data due to time needed for training
    build_model(hc.TRAIN_FILE, selected_cols, max_chunks=max_chunks, chunk_size=2*10**6)
    make_predictions(hc.TEST_FILE, selected_cols, 'xgb_%d.model' % max_chunks,
                     'xgb_test_preds.csv')