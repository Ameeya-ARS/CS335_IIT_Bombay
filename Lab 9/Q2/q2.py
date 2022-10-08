import matplotlib.pyplot as plt
import numpy as np

def kernel_pca(X: np.ndarray, kernel: str) -> np.ndarray:
    '''
    Returns projections of the the points along the top two PCA vectors in the high dimensional space.

        Parameters:
                X      : Dataset array of size (n,2)
                kernel : Kernel type. Can take values from ('poly', 'rbf', 'radial')

        Returns:
                X_pca : Projections of the the points along the top two PCA vectors in the high dimensional space of size (n,2)
    '''
    K = np.zeros((X.shape[0],X.shape[0]))
    if kernel=='rbf':
        gamma = 15
        X1 = np.sum(X**2, axis=-1)
        X2 = np.sum(X**2, axis=-1)
        K =np.exp(-gamma*(X1[:,None] + X2[None,:] - 2*np.dot(X,X.T)))
    if kernel=='poly':
        d = 5
        K = (np.dot(X,X.T))+1
        K = K**d
    if kernel=='radial':
        R = np.linalg.norm(X, axis=1)
        R = np.reshape(R, (R.shape[0], 1))
        R = np.matmul(R, R.T)
        theta= np.arctan2(X[:,1], X[:,0])
        theta = np.reshape(theta, (theta.shape[0], 1))
        theta = np.matmul(theta, theta.T)
        K = R + theta
        
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    X_pca = np.column_stack([eigvecs[:, i] for i in range(2)])
    return X_pca

if __name__ == "__main__":
    from sklearn.datasets import make_moons, make_circles
    from sklearn.linear_model import LogisticRegression
  
    X_c, y_c = make_circles(n_samples = 500, noise = 0.02, random_state = 517)
    X_m, y_m = make_moons(n_samples = 500, noise = 0.02, random_state = 517)

    X_c_pca = kernel_pca(X_c, 'radial')
    X_m_pca = kernel_pca(X_m, 'rbf')
    
    plt.figure()
    plt.title("Data")
    plt.subplot(1,2,1)
    plt.scatter(X_c[:, 0], X_c[:, 1], c = y_c)
    plt.subplot(1,2,2)
    plt.scatter(X_m[:, 0], X_m[:, 1], c = y_m)
    plt.show()

    plt.figure()
    plt.title("Kernel PCA")
    plt.subplot(1,2,1)
    plt.scatter(X_c_pca[:, 0], X_c_pca[:, 1], c = y_c)
    plt.subplot(1,2,2)
    plt.scatter(X_m_pca[:, 0], X_m_pca[:, 1], c = y_m)
    plt.show()
