import numpy as np
from scipy.spatial.distance import pdist, squareform


def indice_dunn(X, labels):
    """Indice de Dunn.

    Plus petite distance entre deux points de clusters différents,
    divisée par le plus grand diamètre d'un cluster.
    Plus il est grand, plus les clusters sont compacts et bien séparés.
    """
    labels = np.asarray(labels)
    D = squareform(pdist(X))
    clusters = np.unique(labels)

    diametre_max = max(D[np.ix_(labels == c, labels == c)].max()
                       for c in clusters)

    distance_min = min(D[np.ix_(labels == a, labels == b)].min()
                       for i, a in enumerate(clusters)
                       for b in clusters[i + 1:])

    return distance_min / diametre_max


def inertie(X, labels):
    """Inertie intra-cluster.

    Somme des carrés des distances de chaque point au centre de son cluster.
    Plus elle est faible, plus les clusters sont compacts.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    return sum(((X[labels == c] - X[labels == c].mean(axis=0)) ** 2).sum()
               for c in np.unique(labels))