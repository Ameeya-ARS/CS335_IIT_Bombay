import numpy as np
from utils import *
############################ Q1 ################################
# Initialize global variable. DO NOT MAKE CHANGES HERE. 
# Change values of kernel parameters in function q1(), NOT HERE.
kernel_params = {'sigma_gauss':1,
                'gamma_rbf':1,
                'sigma_laplace':1}

########## SVM with Gaussian kernel ##########

def gauss_kernel(a,b):
    '''
    Define a Gaussian kernel for a SVM Regressor
    @params
        a: numpy.ndarray shape = (n,d)
        b: numpy.ndarray shape = (n,d)
    return: numpy.ndarray shape = ?
    '''
    sigma = kernel_params['sigma_gauss']
    ######START: TO CODE########
    out = np.zeros((a.shape[0],b.shape[0]))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            out[i][j] = np.exp((-1*(np.linalg.norm(a[i]-b[j]))**2)/(2*sigma*sigma))
    return out
    ######END: TO CODE########

########## SVM with RBF kernel ##########

def rbf_kernel(a,b):
    '''
    Define a Gaussian RBF kernel for a SVM Regressor
    @params
        a: numpy.ndarray shape = (n,d)
        b: numpy.ndarray shape = (n,d)
    return: numpy.ndarray shape = ?
    '''
    gamma = kernel_params['gamma_rbf']
    ######START: TO CODE########
    out = np.zeros((a.shape[0],b.shape[0]))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            out[i][j] = np.exp(-1*gamma*(np.linalg.norm(a[i]-b[j]))**2)
    return out
    ######END: TO CODE########

 ########## SVM with Laplacian RBF kernel ##########

def laplace_kernel(a,b):
    '''
    Define a Laplacian RBF kernel for a SVM Regressor
    @params
        a: numpy.ndarray shape = (n,d)
        b: numpy.ndarray shape = (n,d)
    return: numpy.ndarray shape = ?
    '''
    sigma = kernel_params['sigma_laplace']
    ######START: TO CODE########
    out = np.zeros((a.shape[0],b.shape[0]))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            out[i][j] = np.exp((-1*(np.linalg.norm(a[i]-b[j])))/(sigma))
    return out
    ######END: TO CODE########
        
def q1():
    global kernel_params
    print("Dataset 1\n")
    X_train, X_test, y_train, y_test = get_dataset1()

    # The tunable hyperparmeters for the 3 kernels - all initialized to 1
    # Change values to the optimal values for dataset 1
    ######START: TO DO########
    kernel_params = {'sigma_gauss':1,
                    'gamma_rbf':1,
                    'sigma_laplace':1}
    ######END: TO DO########
    
    reg1 = SVM_Regression(kernel=gauss_kernel)
    reg1.train(X_train,y_train)
    print("Gaussian Kernel Score: ",reg1.get_score(X_test,y_test)) #Higher score = better fit
    reg1.plot(X_train,y_train)
    
    reg2 = SVM_Regression(kernel=rbf_kernel)
    reg2.train(X_train,y_train)
    print("RBF Kernel Score: ",reg2.get_score(X_test,y_test)) #Higher score = better fit
    reg2.plot(X_train,y_train)
    
    reg3 = SVM_Regression(kernel=laplace_kernel)
    reg3.train(X_train,y_train)
    print("Laplacian RBF Kernel Score: ",reg3.get_score(X_test,y_test)) #Higher score = better fit
    reg3.plot(X_train,y_train)
    
    print("Dataset 2\n")
    X_train, X_test, y_train, y_test = get_dataset2()

    # The tunable hyperparmeters for the 3 kernels - all initialized to 1
    # Change values to the optimal values for dataset 2
    ######START: TO DO########
    kernel_params = {'sigma_gauss':1,
                    'gamma_rbf':1,
                    'sigma_laplace':1}
    ######END: TO DO########
    
    reg1 = SVM_Regression(kernel=gauss_kernel)
    reg1.train(X_train,y_train)
    print("Gaussian Kernel Score: ",reg1.get_score(X_test,y_test)) #Higher score = better fit

    reg2 = SVM_Regression(kernel=rbf_kernel)
    reg2.train(X_train,y_train)
    print("RBF Kernel Score: ",reg2.get_score(X_test,y_test)) #Higher score = better fit

    reg3 = SVM_Regression(kernel=laplace_kernel)
    reg3.train(X_train,y_train)
    print("Laplacian RBF Kernel Score: ",reg3.get_score(X_test,y_test)) #Higher score = better fit


if __name__ == "__main__":
    print("#"*15, "Q1", "#"*15)
    try:
        q1()
    except:
        print("Error while solving Q1")