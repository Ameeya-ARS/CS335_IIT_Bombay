from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from utils import *
############################ Q1 ################################

def q1():

    X_train, X_test, y_train, y_test = get_data()


    ########### Linear SVM ###########
    clf = SVM_Classifier()
    clf.train(X_train, y_train)
    print("Linear SVM",clf.get_score(X_test, y_test))
    clf.plot(X_test,y_test,1)
    


    ########## Linear SVM with features ###########

    # To Do: Define polynomial features by expanding (x1 + x2)^n
    # Where x1 and x2 are the original features and n is a hyperparameter specifying the highest degree

    def poly_features(x):
    ######START: TO CODE########
        xprime = np.array([
            x[:,0],
            x[:,1],
            # Add features of higher degree
        ]).T
        a = x[:,0]
        b = x[:,1]
        n=3
        temp = np.zeros((n+1,x.shape[0]))
        for i in range(n+1):
            temp[i] = np.multiply(np.power(a,i),np.power(b,n-i))
        xprime = np.concatenate((xprime,temp.T),axis=1)
        return xprime
    ######END: TO CODE########

    Xprime_train, Xprime_test = poly_features(X_train), poly_features(X_test)

    clf2 = SVM_Classifier()
    clf2.train(Xprime_train, y_train)
    print("SVM using polynomial features",clf2.get_score(Xprime_test, y_test))

    ########## SVM with polynomial kernel ##########

    def my_kernel(X, Y):
        '''
        Define a polynomial kernel to return features of a higher degree
        You can add more parameters to suit your approach
        '''
    ######START: TO CODE########
        K = np.zeros((X.shape[0],Y.shape[0]))
        K = (np.dot(X,Y.T))+1
        X = K**3
        return X
    ######END: TO CODE########

    clf3 = SVM_Classifier(kernel=my_kernel)
    clf3.train(X_train,y_train)
    print("SVM using polynomial kernel",clf3.get_score(X_test,y_test))
    clf3.plot(X_test,y_test,2)
