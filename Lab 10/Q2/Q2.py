from doctest import testfile
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    infile = './trainingAndTest/training.json'
    testfile = './trainingAndTest/sample-test.in.json'
    outfile = 'temp.txt'
    with open(infile, "r") as f:
        num_samples = int(f.readline().strip())
        X_train = np.full((num_samples,9),fill_value=np.nan)
        Y_train = []
        i = 0
        index_dict = {"Physics":0, "Chemistry":1, "English":2, "Biology":3, "PhysicalEducation":4, "Accountancy":5, "BusinessStudies":6, "ComputerScience":7, "Economics":8}
        while True:
            line = f.readline()
            if not line:
                break
            line = eval(line)
            for key in line:
                if key in index_dict:
                    X_train[i][index_dict[key]] = line[key]
            Y_train.append(line["Mathematics"])
            i+=1
    Y_train = np.array(Y_train)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    model = LinearRegression()
    model.fit(X_train,Y_train)
    Y_pred_train = np.minimum(np.maximum(np.round(model.predict(X_train)),0),8)
    score_train = 100*np.sum(np.abs(Y_pred_train-Y_train)<=1)/len(Y_train)
    # print(score_train)

    with open(testfile, "r") as f:
        num_samples = int(f.readline().strip())
        X_test = np.full((num_samples,9),fill_value=np.nan)
        i = 0
        index_dict = {"Physics":0, "Chemistry":1, "English":2, "Biology":3, "PhysicalEducation":4, "Accountancy":5, "BusinessStudies":6, "ComputerScience":7, "Economics":8}
        while True:
            line = f.readline()
            if not line:
                break
            line = eval(line)
            for key in line:
                if key in index_dict:
                    X_test[i][index_dict[key]] = line[key]
            i+=1
    X_test = imp.transform(X_test)
    Y_pred = np.minimum(np.maximum(np.round(model.predict(X_test)),0),8)
    # with open(outfile, "w") as f:
    #     for val in Y_pred:
    #         f.write(f"{int(val)}\n")

    infiles = ['./Testcases/input/input00.txt', './Testcases/input/input01.txt']
    outfiles = ['./output00.txt', './output01.txt']
    for a in range(2):
        with open(infiles[a], "r") as f:
            num_samples = int(f.readline().strip())
            X_test = np.full((num_samples,9),fill_value=np.nan)
            i = 0
            index_dict = {"Physics":0, "Chemistry":1, "English":2, "Biology":3, "PhysicalEducation":4, "Accountancy":5, "BusinessStudies":6, "ComputerScience":7, "Economics":8}
            while True:
                line = f.readline()
                if not line:
                    break
                line = eval(line)
                for key in line:
                    if key in index_dict:
                        X_test[i][index_dict[key]] = line[key]
                i+=1
        X_test = imp.transform(X_test)
        Y_pred = np.minimum(np.maximum(np.round(model.predict(X_test)),0),8)
        with open(outfiles[a], "w") as f:
            for val in Y_pred:
                f.write(f"{int(val)}\n")
