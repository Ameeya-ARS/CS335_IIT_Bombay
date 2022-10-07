import numpy as np
import time

def brute_dct(X: np.ndarray) -> np.ndarray:
    
    m,n = X.shape
    result = np.empty((m,n))
    for i in range(m):
        for j in range(n):
            temp = 0
            for a in range(m):
                for b in range(n):
                    temp = temp + 4*X[a,b]*np.cos((np.pi/m)*(a+0.5)*i)*np.cos((np.pi/n)*(b+0.5)*j)
            result[i, j] = temp

    return result

def vectorized_dct(X: np.ndarray) -> np.ndarray:
    '''
    @params
        X : np.float64 array of size(m,n)
    return np.float64 array of size(m,n)
    '''
    # TODO
    m,n = X.shape
    
    A = np.array(list(range(m))).reshape((m,1)) + (1/2)
    P = np.array(list(range(m))).reshape((1,m))
    B = np.array(list(range(n))).reshape((n,1)) + (1/2)
    Q = np.array(list(range(n))).reshape((1,n))

    temp1 = np.multiply(A,P)
    temp1 = (np.pi/m) * temp1
    cos_val_1 = np.cos(temp1)
    temp2 = np.multiply(B,Q)
    temp2 = (np.pi/n) * temp2
    cos_val_2 = np.cos(temp2)

    cos_val = np.matmul(cos_val_1.T,X)

    result = np.zeros(shape=(m,n))
    result = 4 * np.matmul(cos_val,cos_val_2)
    
    return result
    # END TODO

def get_document_ranking(D: np.ndarray, Q: np.ndarray) -> np.ndarray:
    '''
    @params
        D : n x w x k numpy float64 array 
            where each n x w slice represents a document
            with n vectors of length w 
        Q : m x w numpy float64 array
            which represents a query with m vectors of length w

    return np.ndarray of shape (k,) of docIDs sorted in descending order by relevance score
    '''
    # TODO
    k = D.shape[2]

    Q_transpose = Q.T
    temp = np.zeros((k,))

    for i in range(k) :
        M = np.matmul(D[:,:,i],Q_transpose)
        max_values = M.max(axis=0)
        max_values_sum = max_values.sum()
        temp[i] = max_values_sum
    
    index = np.argsort(temp)[::-1]

    return index
    # END TODO

def compute_state_probability(M: np.ndarray,S_T: int, T: int, S_a: int) -> np.float64:
    '''
    @params
        M   : nxn numpy float64 array
        S_T : int
        T   : int
        S_a : int

    return np.float64
    '''
    # 
    n = M.shape[0]
    result = np.copy(M)

    for a in range(T) :
        for i in range(n) :
            for j in range(n) :
                temp = 0
                for k in range(n) :
                    temp = temp + M[i,k] * result[k,j]
                M[i,j] = temp

    return M[S_a,S_T]
    # END TODO


def compute_state_probability_vec(M: np.ndarray,S_T: int, T: int, S_a: int) -> np.float64:
    '''
    @params
        M   : nxn numpy float64 array
        S_T : int
        T   : int
        S_a : int

    return np.float64
    '''
    # TODO
    temp = np.linalg.matrix_power(M,T)
    return temp[S_a,S_T]
    # END TODO

if __name__=="__main__":

    np.random.seed(0)

    # Question 1
    X = np.random.randn(20,30)
    tic = time.time()
    brute_dct_X = brute_dct(X)
    print("Time for non vectorised DCT = ", time.time()-tic)

    tic = time.time()
    vectorized_dct_X = vectorized_dct(X)
    assert X.shape == vectorized_dct_X.shape, "Return matrix of the same shape as X"
    print("Time for vectorised DCT = ", time.time()-tic)
    print("Equality of both DCTs : ", np.all(np.abs(brute_dct_X - vectorized_dct_X) < 5e-4))

    np.savetxt('q1_output.txt', vectorized_dct_X, fmt="%s")
    
    
    # Question 2
    D = np.random.rand(30,6,100)
    Q = np.random.rand(5,6)

    doc_ranking = get_document_ranking(D,Q)
    np.savetxt('q2_output.txt', doc_ranking, fmt="%s")
    
   
    # Question 3
    N = 6
    S_T = 4
    S_a = 2
    T = 5000
    M = np.zeros(shape=(N,N),dtype=np.float64)
    M[1][0] = 0.09
    M[0][1] = 0.23
    M[5][1] = 0.62
    M[1][2] = 0.06
    M[0][3] = 0.77
    M[2][3] = 0.63
    M[3][4] = 0.65
    M[5][4] = 0.38
    M[1][5] = 0.85
    M[2][5] = 0.37
    M[3][5] = 0.35
    M[4][5] = 1.0

    tic = time.time()
    prob_non_vec = compute_state_probability(M,S_T,T,S_a)
    print("[Q3] Time for non vectorised = ", time.time()-tic)

    tic = time.time()
    prob_vec = compute_state_probability_vec(M,S_T,T,S_a)
    print("[Q2] Time for vectorised = ", time.time()-tic)
    print("Equality of both probs : ", abs(prob_non_vec - prob_vec) < 5e-4)
    with open('q3_output.txt','w') as f:
        f.write(str(prob_vec))
