import numpy as np
from sklearn.metrics.cluster import *
from sklearn.metrics import pairwise_distances

def evaluate_clustering(name, n_clusters, y, y_pred, time):
    print('clustering by {} for n_clusters {}'.format(name, n_clusters))
    print('n_clusters orignial {}'.format(len(np.unique(y))))
    print('n_clusters detected {}'.format(len(np.unique(y_pred))))
    print('n unclustered points {} out of {}'.format(len(y_pred[y_pred==-1]), len(y)))
        
    print('Clustering using {}, time elapesd {}'.format(name, time))
    print('Clustering Accuracy {}'.format(cluster_acc(y[y_pred!=-1], y_pred[y_pred!=-1])))
    print('Clustering purity {}'.format(purity_score(y[y_pred!=-1], y_pred[y_pred!=-1])))
    print('Clustering homogeneity {}'.format(homogeneity_score(y[y_pred!=-1], y_pred[y_pred!=-1])))
    print('Clustering adjusted_rand_score {}'.format(adjusted_rand_score(y[y_pred!=-1], y_pred[y_pred!=-1])))
    print('Clustering adjusted_mutual_info_score {}'.format(adjusted_mutual_info_score(y[y_pred!=-1], y_pred[y_pred!=-1])))
    print('Clustering completeness_score {}'.format(completeness_score(y[y_pred!=-1], y_pred[y_pred!=-1])))
    print('Clustering v_measure_score {}'.format(v_measure_score(y[y_pred!=-1], y_pred[y_pred!=-1]))) 


from sklearn.utils.linear_assignment_ import linear_assignment

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    
from sklearn.metrics import accuracy_score
def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


    