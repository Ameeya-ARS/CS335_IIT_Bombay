from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import pickle
plt.ioff()
#DO NOT MODIFY THIS FILE#
def get_data():
    X,y = pickle.load(open("input/data_q1.pkl", "rb"))
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.4, random_state=42)

class SVM_Classifier:
    def __init__(self, kernel=None):
        if kernel == None:
            self.clf = svm.SVC(kernel='linear')
        else:
            self.clf = svm.SVC(kernel=kernel)
            
    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
    
    def get_score(self, X_test, y_test):
        return self.clf.score(X_test, y_test)
        
    def plot(self, X_, y_, n=1):
        try:
            cm = plt.cm.RdBu
            cm_bright = ListedColormap(["#FF0000", "#0000FF"])
            x_min, x_max = X_[:, 0].min() - 0.5, X_[:, 0].max() + 0.5
            y_min, y_max = X_[:, 1].min() - 0.5, X_[:, 1].max() + 0.5
            ax = plt.subplot()
            DecisionBoundaryDisplay.from_estimator(
                self.clf, X_, cmap=cm, alpha=0.8, ax=ax, eps=0.5
            )
            ax.scatter(
                X_[:, 0], X_[:, 1], c=y_, cmap=cm_bright, edgecolors="k"
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            plot_type = 'linear'
            if n!=1:
                plot_type = 'poly'
            plt.savefig("output/q1_"+plot_type+".png")
            plt.clf()
        except ValueError:
            print("Can't plot a classifier trained on more than two features. Either reduce the features or use a kernel.")