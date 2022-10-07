import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
plt.ioff()
#DO NOT MODIFY THIS FILE#

def get_dataset1():
    X,y = pickle.load(open("data/dataset1.pkl", "rb"))
    return train_test_split(X, y, test_size=0.4, random_state=42)

def get_dataset2():
    data = pd.read_csv('data/dataset2.csv', index_col=0)
    X = data.iloc[:,:-1].to_numpy()
    y = data.iloc[:,-1].to_numpy()
    return train_test_split(X, y, test_size=0.4, random_state=42)

class SVM_Regression:
    def __init__(self, kernel=None):
        self.reg = svm.SVR(kernel=kernel)
            
    def train(self, X_train, y_train):
        self.reg.fit(X_train, y_train)
    
    def get_score(self, X_test, y_test):
        yfit = self.reg.predict(X_test)
        return -mean_squared_error(yfit,y_test)
        
    def plot(self, X_, y_):
        try:
            Xplot = np.sort(X_,axis=0)
            yfit = self.reg.predict(Xplot)
            plt.scatter(X_, y_, s=5, color="blue", label="Training Data")
            plt.plot(Xplot, yfit, lw=2, color="red", label="Fitted Model")
            plt.legend()
            plt.show()
        except ValueError:
            print("Error in plot generation. Can't plot a model trained on more than two features.")