import numpy as np


def map5eval(preds, dtrain):
    """
    Wrapper for MAP@5 evaluation metric compatible with XGBoost
    """
    labels = dtrain.get_label()
    return 'MAP@5', map5(preds, labels)


def map5(preds, labels):
    """
    Calculate Mean Average Precision score for top 5 elements (MAP@5)

    Note: It is simplified version of the metric and it assumes that
    the true label contains only one result
    """
    n = 5
    predicted = np.fliplr(preds.argsort(axis=1)[:, -n:])
    metric = 0.
    for i in range(n):
        metric += np.sum(labels == predicted[:, i]) / (i + 1)
    metric /= labels.shape[0]
    return metric