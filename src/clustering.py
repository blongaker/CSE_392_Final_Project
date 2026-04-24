import numpy as np
from sklearn.cluster import KMeans

def cluster_ics_and_params(y0, params, n_batches):

    # Form combined matrix
    combined = np.column_stack((y0, params)) # These should have the same number of rows

    # Cluster, where each row is a point
    kmeans = KMeans(n_clusters=n_batches)
    labels = kmeans.fit_predict(combined)

    # TODO: Sort rows by cluster and return y0 and params in order
    return labels
