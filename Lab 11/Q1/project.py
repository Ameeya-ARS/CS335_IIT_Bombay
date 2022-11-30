from sklearn.decomposition import PCA
from utils import *

if __name__=='__main__':
    data = load_points_from_json('points_4D.json')
    pca_model=PCA(n_components=2)
    pca_model.fit(data)
    final_data=pca_model.transform(data)
    store_points_to_json(final_data,'points_2D.json')