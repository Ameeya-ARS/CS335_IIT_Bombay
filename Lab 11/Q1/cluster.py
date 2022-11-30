import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from utils import *

if __name__=='__main__':
    data = load_points_from_json('points_2D.json')
    data[:,1] = data[:,0]**2 + data[:,1]**2

    kmeans_model = KMeans(n_clusters=5,random_state=0)
    kmeans_model.fit(data)

    store_labels_to_json(k=5,labels=kmeans_model.labels_,filepath='labels.json')
