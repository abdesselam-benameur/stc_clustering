import numpy as np


def reduce_data(x, y, ratios, n_clusters):
    """
    Reduce the dataset by taking different ration in each class randomly and shuffle the data

    Args:
    x: np.array, shape=(n_samples, n_features)
    y: np.array, shape=(n_samples,)
    ratios: list of float, length=n_clusters
    n_clusters: int, number of clusters

    Returns:
    x: np.array, shape=(n_samples, n_features)
    y: np.array, shape=(n_samples,)

    """
    x_new = []
    y_new = []
    for i in range(1, n_clusters + 1):
        idx = np.where(y == i)[0]
        # set a numpy seed

        idx = np.random.choice(idx, int(len(idx) * ratios[i - 1]), replace=False)
        x_new.append(x[idx])
        y_new.append(y[idx])
    x = np.concatenate(x_new, axis=0)
    y = np.concatenate(y_new, axis=0)
    return x, y
