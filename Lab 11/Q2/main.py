import numpy as np
import torch
from utils import visualize, store_labels
import pandas as pd

def get_clusters(X, n_clusters=10):
    '''
    Inputs:
        X: coordinates of the points
        n_clusters: (optional) number of clusters required
    Output:
        labels: The cluster index assigned to each point in X, same length as len(X)
    '''
    #### TODO: ####
    iters = 1000
    id = np.arange(X.shape[0])
    clusters = X[np.random.choice(id,size=n_clusters)]
    labels = np.zeros((X.shape[0],1))

    for a in range(iters) :
        newCenters = np.zeros((n_clusters,X.shape[1]))
        count = np.zeros((n_clusters,1))
        for i in range(len(X)) :
            dist = np.linalg.norm(clusters-X[i],axis=1)
            index = np.argmin(dist)
            labels[i] = index
            count[index]+=1
            newCenters[index]+=X[i]
            
        for i in range(n_clusters) :
            if count[i] != 0 :
                newCenters[i] = newCenters[i]/count[i]
                clusters[i] = newCenters[i]

    ###############    

    return labels


if __name__ == "__main__":
    data = pd.read_csv("mnist_samples.csv").values
    labels = get_clusters(data)
    store_labels(labels)
    visualize(data, labels, alpha=0.2)
