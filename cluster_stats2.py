import csv
import numpy as np
from collections import Counter, defaultdict

import helpers.consts as hc


def train(all_data=True):
    """
    Calculate hotel cluster popularity for different groups of
    features.

    Parameters
    ----------
    all_data : boolean
        If True then all data are used for training. If False then
        only approx. half of the data is used.

    Return
    ------
    Dictionaries with hotel cluster statistics
    """
    # Variables for storing hotel cluster popularity
    p_ulc_odd = defaultdict(Counter)
    p_sdi_odd_ulc = defaultdict(Counter)
    p_sdi_hm = defaultdict(Counter)
    p_sdi = defaultdict(Counter)
    p_hc = defaultdict(Counter)
    p_hm = defaultdict(Counter)
    p_cl = Counter()
    p_u_sdi_hm = defaultdict(Counter)
    p_u_sdi = defaultdict(Counter)
    p_sdi_hm_ch = defaultdict(Counter)

    with open(hc.TRAIN_FILE, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        reader.next()  # Skip header

        for i, row in enumerate(reader):
            if i % 10**7 == 0:
                print 'Processed %d lines' % i
            # Skip data from 2014 to use them later for evaluation if proper mode is set
            if not all_data and '2014' in row[0]:
                continue

            # Extract fields from row
            user_location_city = row[5]
            orig_destination_distance = row[6]
            user_id = row[7]
            srch_children_cnt = int(row[14]) > 0
            srch_destination_id = row[16]
            is_booking = int(row[18])
            hotel_country = row[21]
            hotel_market = row[22]
            hotel_cluster = row[23]

            # Assign weights for booking/non-booking events
            weight = 1 + is_booking + 2 * (int(row[0][:4]) - 2013)

            # Calculate clusters popularity
            if len(user_location_city) and len(orig_destination_distance):
                p_ulc_odd[(user_location_city, orig_destination_distance)][hotel_cluster] += weight
                if len(srch_destination_id):
                    p_sdi_odd_ulc[(srch_destination_id, orig_destination_distance, user_location_city)][hotel_cluster] += weight

            if len(srch_destination_id) and len(hotel_market):
                p_sdi_hm[(srch_destination_id, hotel_market)][hotel_cluster] += weight
                p_sdi_hm_ch[(srch_destination_id, hotel_market, srch_children_cnt)][hotel_cluster] += weight

            if len(user_id) and len(srch_destination_id) and len(hotel_market):
                p_u_sdi_hm[(user_id, srch_destination_id, hotel_market)][hotel_cluster] += weight

            #if len(user_id) and len(srch_destination_id):
            #    p_u_sdi[(user_id, srch_destination_id)][hotel_cluster] += weight

            if len(srch_destination_id):
                p_sdi[srch_destination_id][hotel_cluster] += weight

            if len(hotel_market):
                p_hm[hotel_market][hotel_cluster] += weight

            if len(hotel_country):
                p_hc[hotel_country][hotel_cluster] += weight

            p_cl[hotel_cluster] += weight

    with open(hc.TEST_FILE, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')

        for row in reader:
            user_location_city = row['user_location_city']
            orig_destination_distance = row['orig_destination_distance']
            srch_destination_id = row['srch_destination_id']
            k_ulc_odd = (user_location_city, orig_destination_distance)
            if k_ulc_odd in p_ulc_odd:
                user_id = row['user_id']
                srch_children_cnt = int(row['srch_children_cnt']) > 0
                hotel_country = row['hotel_country']
                hotel_market = row['hotel_market']
                hotel_cluster = p_ulc_odd[k_ulc_odd].most_common(1)[0][0]

                k_sdi_odd_ulc = (srch_destination_id, orig_destination_distance, user_location_city)
                weight = 1
                if k_sdi_odd_ulc in p_sdi_odd_ulc:
                    weight = 2
                    hotel_cluster = p_sdi_odd_ulc[k_sdi_odd_ulc].most_common(1)[0][0]

                if len(srch_destination_id) and len(hotel_market):
                    p_sdi_hm[(srch_destination_id, hotel_market)][hotel_cluster] += weight
                    p_sdi_hm_ch[(srch_destination_id, hotel_market, srch_children_cnt)][hotel_cluster] += weight

                if len(user_id) and len(srch_destination_id) and len(hotel_market):
                    p_u_sdi_hm[(user_id, srch_destination_id, hotel_market)][hotel_cluster] += weight

                #if len(user_id) and len(srch_destination_id):
                #    p_u_sdi[(user_id, srch_destination_id)][hotel_cluster] += weight

                if len(srch_destination_id):
                    p_sdi[srch_destination_id][hotel_cluster] += weight

                if len(hotel_market):
                    p_hm[hotel_market][hotel_cluster] += weight

                if len(hotel_country):
                    p_hc[hotel_country][hotel_cluster] += weight

                p_cl[hotel_cluster] += weight

    return p_ulc_odd, p_sdi_hm, p_sdi, p_hc, p_hm, p_cl, p_u_sdi_hm, p_u_sdi, p_sdi_hm_ch, p_sdi_odd_ulc


def test(p_ulc_odd, p_sdi_hm, p_sdi, p_hc, p_hm, p_cl, p_u_sdi_hm, p_u_sdi, p_sdi_hm_ch, p_sdi_odd_ulc):
    """
    Use statistics gathered during training to produce submission
    file with top n predictions of hotel clusters for the test set.

    Parameters
    ----------
    p_ulc_odd : dict
        Hotel cluster statistics for orig_destination_distance and
        user_location_city

    p_sdi_hc_hm : dict
        Hotel cluster statistics for srch_destination_id, hotel_country
        and hotel_market

    p_sdi : dict
        Hotel cluster statistics for srch_destination_id

    p_hc : dict
        Hotel cluster statistics for hotel_country

    p_hc_hm : dict
        Hotel cluster statistics for hotel_country and hotel_market

    p_cl : dict
     The most popular hotel clusters among training set
    """

    with open(hc.TEST_FILE, 'r') as f, open('submission.csv', 'w') as out:
        reader = csv.DictReader(f, delimiter=',')
        writer = csv.writer(out, delimiter=',')
        writer.writerow(['id', 'hotel_cluster'])

        for i, row in enumerate(reader):
            if i % 10**6 == 0:
                print 'Wrote %d lines' % i

            predictions = predict(p_ulc_odd, p_sdi_hm, p_sdi, p_hc, p_hm, p_cl, p_u_sdi_hm,
                                  p_u_sdi, p_sdi_hm_ch, p_sdi_odd_ulc, row)
            writer.writerow([row['id'], ' '.join(map(str, predictions))])


def validate(p_ulc_odd, p_sdi_hm, p_sdi, p_hc, p_hm, p_cl, p_u_sdi_hm, p_u_sdi, p_sdi_hm_ch, p_sdi_odd_ulc):
    """
    Use approx. half of the training set to validate predictions of
    hotel clusters.

    Parameters
    ----------
    p_ulc_odd : dict
        Hotel cluster statistics for orig_destination_distance and
        user_location_city

    p_sdi_hc_hm : dict
        Hotel cluster statistics for srch_destination_id, hotel_country
        and hotel_market

    p_sdi : dict
        Hotel cluster statistics for srch_destination_id

    p_hc : dict
        Hotel cluster statistics for hotel_country

    p_hc_hm : dict
        Hotel cluster statistics for hotel_country and hotel_market

    p_cl : dict
     The most popular hotel clusters among training set
    """
    metric = 0.0  # Stores precision
    cntr = 0  # Counts number of samples used for evaluation

    with open(hc.TRAIN_FILE, 'r') as f, open('mnb_train.csv', 'r') as f2:
        reader = csv.DictReader(f, delimiter=',')
        mnb = csv.reader(f2, delimiter=',')

        for i, row in enumerate(reader):
            if i % 10**7 == 0:
                print 'Processed %d lines' % i
            if '2013' in row['date_time']:
                continue
            cntr += 1

            mnb_preds = np.array(mnb.next(), dtype=float)

            predictions = predict(p_ulc_odd, p_sdi_hm, p_sdi, p_hc, p_hm, p_cl, p_u_sdi_hm,
                                  p_u_sdi, p_sdi_hm_ch, p_sdi_odd_ulc, row, mnb_preds)

            # Calculate MAP@5 metric iteratively to reduce memory usage
            for j, p in enumerate(predictions):
                metric += (int(row['hotel_cluster']) == p) / (j + 1)

    print 'MAP@5', metric / cntr


def predict(p_ulc_odd, p_sdi_hm, p_sdi, p_hc, p_hm, top_clusters, p_u_sdi_hm, p_u_sdi, p_sdi_hm_ch, p_sdi_odd_ulc, row, preds_ext=None):
    """
    Generate prediction of top n hotel clusters based on data statistics
    learnt during training phase

    Parameters
    ----------
    p_ulc_odd : dict
        Hotel cluster statistics for orig_destination_distance and
        user_location_city

    p_sdi_hc_hm : dict
        Hotel cluster statistics for srch_destination_id, hotel_country
        and hotel_market

    p_sdi : dict
        Hotel cluster statistics for srch_destination_id

    p_hc : dict
        Hotel cluster statistics for hotel_country

    p_hc_hm : dict
        Hotel cluster statistics for hotel_country and hotel_market

    top_clusters : dict
        The most popular hotel clusters among whole training set

    row : dict
        Sample for which predictions are generated

    Return
    ------
    List of top n hotel clusterss
    """

    predictions = np.zeros((100,), dtype=float)

    # Exploit data leakage
    k_sdi_odd_ulc = (row['srch_destination_id'], row['orig_destination_distance'], row['user_location_city'])
    if k_sdi_odd_ulc in p_sdi_odd_ulc:
        predictions += 20 * get_hotel_cluster_proba(p_sdi_odd_ulc, k_sdi_odd_ulc)

    k_ulc_odd = (row['user_location_city'], row['orig_destination_distance'])
    if k_ulc_odd in p_ulc_odd:
        predictions += 15 * get_hotel_cluster_proba(p_ulc_odd, k_ulc_odd)

    # Generate predictions based on statistics from training set
    k_sdi_hm_ch = (row['srch_destination_id'], row['hotel_market'], int(row['srch_children_cnt']) > 0)
    if k_sdi_hm_ch in p_sdi_hm_ch:  # Testing in exp
        predictions += 10 * get_hotel_cluster_proba(p_sdi_hm_ch, k_sdi_hm_ch)

    k_u_sdi_hm = (row['user_id'], row['srch_destination_id'], row['hotel_market'])
    if k_u_sdi_hm in p_u_sdi_hm:
        predictions += 10 * get_hotel_cluster_proba(p_u_sdi_hm, k_u_sdi_hm)

    k_sdi_hm = (row['srch_destination_id'], row['hotel_market'])
    if k_sdi_hm in p_sdi_hm:
        predictions += 5 * get_hotel_cluster_proba(p_sdi_hm, k_sdi_hm)
    elif row['srch_destination_id'] in p_sdi:
        predictions += 5 * get_hotel_cluster_proba(p_sdi, row['srch_destination_id'])

    if row['hotel_market'] in p_hm:
        predictions += 2 * get_hotel_cluster_proba(p_hm, row['hotel_market'])

    if row['hotel_country'] in p_hc:
        predictions += 1 * get_hotel_cluster_proba(p_hc, row['hotel_country'])

    predictions += get_hotel_cluster_proba_simple(top_clusters)
    predictions += preds_ext

    return np.fliplr([predictions.argsort()[-hc.TOP_N:]])[0]


def get_hotel_cluster_proba(p, key):
    return get_hotel_cluster_proba_simple(p[key])


def get_hotel_cluster_proba_simple(p):
    partial = np.zeros((100,), dtype=float)
    s = 0.0
    for k, v in p.items():
        partial[int(k)] = v
        s += v
    return partial / s


if __name__ == '__main__':
    #test(*train())
    validate(*train(all_data=False))
