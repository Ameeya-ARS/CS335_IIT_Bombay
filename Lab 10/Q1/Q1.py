import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import warnings
warnings.filterwarnings("ignore")
plt.ioff()

if __name__=="__main__":

    input_files = ['input00.txt','input01.txt','input02.txt']
    output_files = ['output00.txt','output01.txt','output02.txt']
    for a in range(len(input_files)):
        f = open(input_files[a],'r')
        data = f.read().split('\n')
        N = int(data[0])
        date_train, month_train, year_train, time_train, price_train = [], [], [], [], []
        date_test, month_test, year_test, time_test = [], [], [], []
        val_train, val_test = [], []
        for i in range(1,N+1):
            line = data[i]
            parts = line.split()
            if(parts[2][0]=='M'):
                parts_of_date = parts[0].split('/')
                date_test.append(parts_of_date[1])
                month_test.append(parts_of_date[0])
                year_test.append(parts_of_date[2])
                time_test.append(parts[1].split(':'))
                dt = datetime(int(year_test[-1]),int(month_test[-1]),int(date_test[-1]),int(time_test[-1][0]),int(time_test[-1][1]),int(time_test[-1][2]))
                val_test.append(int(dt.timestamp()))
            else:
                parts_of_date = parts[0].split('/')
                date_train.append(parts_of_date[1])
                month_train.append(parts_of_date[0])
                year_train.append(parts_of_date[2])
                time_train.append(parts[1].split(':'))
                price_train.append(float(parts[2]))
                dt = datetime(int(year_train[-1]),int(month_train[-1]),int(date_train[-1]),int(time_train[-1][0]),int(time_train[-1][1]),int(time_train[-1][2]))
                val_train.append(int(dt.timestamp()))

        val_arr = np.array(val_train)
        temp_mean = np.mean(val_arr)
        temp_std = np.std(val_arr)
        val_arr = (val_arr - temp_mean)/temp_std
        df = pd.DataFrame()
        df['val'] = val_arr
        df['prices'] = price_train
        X = df.drop(['prices'],axis=1)
        y = df['prices']

        kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X,y)

        val_arr = np.array(val_test)
        val_arr = (val_arr - temp_mean)/temp_std
        df = pd.DataFrame()
        df['val'] = val_arr
        y_pred = model.predict(df).tolist()
        file1 = open(output_files[a],"w")
        for i in range(len(y_pred)) :
            file1.write(str(y_pred[i]))
            file1.write('\n')
        file1.close()
        f.close()
