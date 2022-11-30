import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.manifold import TSNE

def visualize(X, y, alpha=0.1):
    '''
    Inputs:
        X: The coordinates of the points in latent space
        y: Cluster index assigned to the coordinate
    '''
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(X)
    tsne_results = np.concatenate([np.array(tsne_results),np.array(y).reshape(-1,1)],axis=1)

    for i in range(10):
        temp = tsne_results[tsne_results[:,2] == i]
        plt.scatter(temp[:,0],temp[:,1],alpha=alpha)
    
    plt.show()

    return None

def store_labels(labels):
    with open("q2_labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    return None