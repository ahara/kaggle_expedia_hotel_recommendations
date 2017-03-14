import csv
import numpy as np

from copy import deepcopy
from dateutil import parser
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

import helpers.consts as hc
from helpers.metrics import map5


def str_to_int(s, default=0):
    try:
        return int(s)
    except ValueError:
        return default


def date_diff_days(d_earlier, d_later, default=0):
    try:
        diff = parser.parse(d_later) - parser.parse(d_earlier)
        if diff.days < 0:  # Do not return negative values as encoder or naive bayes have problem with processing them
            return default
        return diff.days
    except ValueError:
        return default  # Empty date (eventually wrong format)


def read_data(column_names):
    with open(hc.TRAIN_FILE, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        x = []

        for i, row in enumerate(reader):
            if i % 10**6 == 0:
                print 'Processed lines:', i
            if '2014' in row['date_time']:
                continue

            record = []
            for c in column_names:
                record.append(row[c])
            record.append(str_to_int(row['date_time'][5:7]))  # Add event month
            record.append(str_to_int(row['srch_ci'][5:7]))  # Add check-in month
            record.append(str_to_int(row['user_id']) % 10000)  # Add user buckets
            record.append(date_diff_days(row['srch_ci'], row['srch_co']))  # Calculate length of stay
            x.append(record)

        enc = OneHotEncoder(dtype=int, handle_unknown='ignore')
        enc.fit(np.array(x))

        return enc


def read_data_generator(column_names, enc, chunk_size=10**6, train_set=True, val_set=True):
    with open(hc.TRAIN_FILE, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        x, y = [], []

        for i, row in enumerate(reader):
            if '2013' in row['date_time'] and not train_set:
                continue
            if '2014' in row['date_time'] and not val_set:
                continue

            y.append(int(row['hotel_cluster']))
            record = []
            for c in column_names:
                record.append(row[c])
            record.append(str_to_int(row['date_time'][5:7]))  # Add event month
            record.append(str_to_int(row['srch_ci'][5:7]))  # Add check-in month
            record.append(str_to_int(row['user_id']) % 10000)  # Add user buckets
            record.append(date_diff_days(row['srch_ci'], row['srch_co']))  # Calculate length of stay
            x.append(record)
            if len(x) % chunk_size == 0 and len(x) > 0:
                xt = enc.transform(np.array(x))
                yt = np.array(y)
                x, y = [], []
                yield xt, yt
        if len(x) > 0:
            xt = enc.transform(np.array(x))
            yt = np.array(y)
            yield xt, yt


def make_predictions(clf, classes, column_names, enc):  # Make predictions which can be integrated with leakage
    cntr = 0
    for t, l in read_data_generator(column_names, enc, chunk_size=10**6, val_set=False):
        cntr += 1
        print 'Learn model', cntr
        clf.partial_fit(t, l, classes=classes)
    with open('mnb_train.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for t, _ in read_data_generator(selected_cols, enc, chunk_size=10**6, train_set=False):
            # Make predictions for all data
            for p in clf.predict_proba(t):
                writer.writerow(map(lambda x: round(x, 5), p))


if __name__ == '__main__':
    selected_cols = ['user_location_city', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'is_booking',
                     'srch_destination_id', 'hotel_market', 'channel', 'is_mobile', 'is_package']
    encoder = read_data(selected_cols)

    clf = KNeighborsClassifier(n_neighbors=5)
    classes = np.array(range(100))

    #make_predictions(clf, classes, selected_cols, encoder)
    #exit(0)

    # Train model for without validation set
    i = 0
    models = []
    for train, labels in read_data_generator(selected_cols, encoder, val_set=False, chunk_size=10**5):
        print train.shape, labels[:20]
        i += 1
        print 'Learning model', i
        clf.fit(train, labels)
        models.append(deepcopy(clf))

    feature_n = encoder.active_features_.shape[0]
    print 'Number of features:', feature_n

    cores, counter = 0.0, 0
    for val, labels in read_data_generator(selected_cols, encoder, train_set=False, chunk_size=10**5):
        preds = None
        for c in models:
            p = c.predict_proba(val)
            preds = p if preds is None else preds + p
        score = map5(preds, labels)
        print score
        scores = score * labels.shape[0]
        counter += labels.shape[0]

    print 'Final score', scores / counter
