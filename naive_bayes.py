import copy
import csv
import numpy as np

from dateutil import parser
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import MultinomialNB
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
        xb, yb, xnb, ynb = [], [], [], []

        for i, row in enumerate(reader):
            if i % 10**6 == 0:
                print 'Processed lines:', i
            if not int(row['is_booking']) and i % 4 != 0:
                continue

            x, y = (xb, yb) if int(row['is_booking']) else (xnb, ynb)

            y.append(int(row['hotel_cluster']))
            record = []
            for c in column_names:
                record.append(row[c])
            record.append(str_to_int(row['date_time'][5:7]))  # Add event month
            record.append(str_to_int(row['srch_ci'][5:7]))  # Add check-in month
            record.append(str_to_int(row['user_id']) % 10000)  # Add user buckets
            record.append(date_diff_days(row['srch_ci'], row['srch_co']))  # Calculate length of stay
            x.append(record)

        enc = OneHotEncoder(dtype=int, handle_unknown='ignore')
        xb, yb, xnb, ynb = map(np.array, [xb, yb, xnb, ynb])
        enc.fit(np.vstack((xb, xnb)))
        xb = enc.transform(xb)
        xnb = enc.transform(xnb)

        return xb, yb, xnb, ynb, enc


def read_data_generator(column_names, enc, chunk_size=10**6, only_non_booking=True):
    with open(hc.TRAIN_FILE, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        x, y = [], []

        for i, row in enumerate(reader):
            if int(row['is_booking']) and only_non_booking:
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
    for t, l in read_data_generator(column_names, enc, chunk_size=10**6, only_non_booking=False):
        cntr += 1
        print 'Learn model', cntr
        clf.partial_fit(t, l, classes=classes)
        if cntr > 19:  # Skip approx. half of data to use for evaluation
            continue
    with open('mnb_train.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for t, _ in read_data_generator(selected_cols, enc, chunk_size=10**6, only_non_booking=False):
            # Make predictions for all data
            predictions = clf.predict_proba(t)
            top_clusters = np.fliplr(predictions.argsort(axis=1)[:, -hc.TOP_N:])
            writer.writerows(top_clusters)


if __name__ == '__main__':
    selected_cols = ['user_location_city', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'is_booking',
                     'srch_destination_id', 'hotel_market', 'channel', 'is_mobile', 'is_package']
    train, labels, _, _, encoder = read_data(selected_cols)
    print train.shape, labels.shape

    clf_base = MultinomialNB(alpha=0.33)
    classes = np.array(range(100))

    make_predictions(clf_base, classes, selected_cols, encoder)
    exit(0)

    # Train model on non-booking data which are not used for validation
    i = 0
    for train_nb, labels_nb in read_data_generator(selected_cols, encoder):
        print labels_nb[:20]
        i += 1
        print 'Learning model', i
        clf_base.partial_fit(train_nb, labels_nb, classes=classes)

    kf = KFold(train.shape[0], n_folds=5)
    predictions = np.zeros((train.shape[0], 100))

    for tid, vid in kf:
        clf = copy.deepcopy(clf_base)  # Use copy to avoid information leakage between folds
        clf.partial_fit(train[tid, :], labels[tid], classes=classes)
        predictions[vid, :] = clf.predict_proba(train[vid, :])

    print map5(predictions, labels)
