import csv
import datetime
from collections import Counter, defaultdict
from more_itertools import unique_everseen

import helpers.consts as hc


def train(all_data=True):
    p_odd_ulc = defaultdict(Counter)
    p_sdi_hc_hm = defaultdict(Counter)
    p_sdi = defaultdict(Counter)
    p_hc = defaultdict(Counter)
    p_hc_hm = defaultdict(Counter)
    p_cl = Counter()

    # Calc counts
    with open(hc.TRAIN_FILE, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        reader.next()  # Skip header

        for i, row in enumerate(reader):
            if i % 10**7 == 0:
                print 'Processed %d lines' % i
            if not all_data and i > 19 * 10**6:  # Skip approx. half of data to use for evaluation
                continue

            book_year = int(row[0][:4])
            user_location_city = row[5]
            orig_destination_distance = row[6]
            srch_destination_id = row[16]
            is_booking = int(row[18])
            hotel_country = row[21]
            hotel_market = row[22]
            hotel_cluster = row[23]

            append_1 = 3 + 17 * is_booking
            append_2 = 1 + 5 * is_booking

            if user_location_city != '' and orig_destination_distance != '':
                p_odd_ulc[(user_location_city, orig_destination_distance)][hotel_cluster] += 1

            if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014:
                p_sdi_hc_hm[(srch_destination_id, hotel_country, hotel_market)][hotel_cluster] += append_1

            if srch_destination_id != '':
                p_sdi[srch_destination_id][hotel_cluster] += append_1

            if hotel_country != '' and hotel_market != '':
                p_hc_hm[(hotel_country, hotel_market)][hotel_cluster] += append_1

            if hotel_country != '':
                p_hc[hotel_country][hotel_cluster] += append_2

            p_cl[hotel_cluster] += 1

    return p_odd_ulc, p_sdi_hc_hm, p_sdi, p_hc, p_hc_hm, p_cl


def test(p_odd_ulc, p_sdi_hc_hm, p_sdi, p_hc, p_hc_hm, p_cl):
    now = datetime.datetime.now()
    output_file = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    top_clusters = p_cl.most_common(hc.TOP_N)

    with open(hc.TEST_FILE, 'r') as f, open(output_file, 'w') as out:
        reader = csv.DictReader(f, delimiter=',')
        writer = csv.writer(out, delimiter=',')
        writer.writerow(['id', 'hotel_cluster'])

        for i, row in enumerate(reader):
            if i % 10**6 == 0:
                print 'Wrote %d lines' % i

            predictions = predict(p_odd_ulc, p_sdi_hc_hm, p_sdi, p_hc, p_hc_hm, top_clusters, row)

            writer.writerow([row['id'], ' '.join(predictions)])


def validate(p_odd_ulc, p_sdi_hc_hm, p_sdi, p_hc, p_hc_hm, p_cl):
    metric = 0.
    cntr = 0
    top_clusters = p_cl.most_common(hc.TOP_N)

    with open(hc.TRAIN_FILE, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')

        for i, row in enumerate(reader):
            external_predictions = None  #get_external_predictions()
            if i % 10**7 == 0:
                print 'Processed %d lines' % i
            if i <= 19 * 10**6:
                continue
            cntr += 1

            predictions = predict(p_odd_ulc, p_sdi_hc_hm, p_sdi, p_hc, p_hc_hm, top_clusters, row, external_predictions)

            for j, p in enumerate(predictions):
                metric += (row['hotel_cluster'] == p) / (j + 1)

    print 'MAP@5', metric / cntr


def predict(p_odd_ulc, p_sdi_hc_hm, p_sdi, p_hc, p_hc_hm, top_clusters, row, ext_preds):
    predictions = []

    k_odd_ulc = (row['user_location_city'], row['orig_destination_distance'])
    if k_odd_ulc in p_odd_ulc:
        predictions = find_with_add_clusters(p_odd_ulc, k_odd_ulc, predictions)

    k_sdi_hc_hm = (row['srch_destination_id'], row['hotel_country'], row['hotel_market'])
    if k_sdi_hc_hm in p_sdi_hc_hm:
        predictions = find_with_add_clusters(p_sdi_hc_hm, k_sdi_hc_hm, predictions)
    elif row['srch_destination_id'] in p_sdi:
        predictions = find_with_add_clusters(p_sdi, row['srch_destination_id'], predictions)

    k_hc_hm = (row['hotel_country'], row['hotel_market'])
    if k_hc_hm in p_hc_hm:
        predictions = find_with_add_clusters(p_hc_hm, k_hc_hm, predictions)

    if row['hotel_country'] in p_hc:
        predictions = find_with_add_clusters(p_hc, row['hotel_country'], predictions)

    return add_clusters(top_clusters, predictions)


def find_with_add_clusters(dict_data, key, current_clusters):
    """
    Add new prediction to hotel clusters based on statistics
    gathered for particular key.

    Parameters
    ----------
    dict_data : dict
        Dictionary with training data statistics

    key : tuple or string
        Key for which new predictions should be added

    current_clusters : list
        Current top predictions for hotel clusters

    Return
    ------
    List with top hotel clusters
    """
    d = dict_data[key]
    top_items = d.most_common(hc.TOP_N)

    return add_clusters(top_items, current_clusters)


def add_clusters(potential_clusters, current_clusters):
    """
    Add new predictions to current results with avoiding
    introducing duplicated clusters.

    Parameters
    ----------
    potential_clusters : list
        New predictions for hotel clusters

    current_clusters : list
        Current top predictions for hotel clusters

    Return
    ------
    List with top hotel clusters
    """
    cluster_names = list(zip(*potential_clusters)[0])

    return list(unique_everseen(current_clusters + cluster_names))[:hc.TOP_N]


def get_external_predictions():
    with open('mnb_train.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        yield reader.next()

if __name__ == '__main__':
    validate(*train(False))
