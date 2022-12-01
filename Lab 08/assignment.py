
import time
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
from sklearn import svm
from Q1 import *
from Q2 import *
from Q3 import *
from Q4 import *
# READ THE COMMENTS CAREFULLY #

if __name__ == "__main__":
    print("#"*15, "Q1", "#"*15)
    try:
        q1()
    except:
        print("Error while solving Q1")
    print("#"*15, "Q2", "#"*15)
    print("Kernel 1")
    try:
        model = svm.SVC(kernel=Q2_kernel_1, C=10, gamma=1/2, shrinking=False)
        kernel_1_sample_1_x = np.loadtxt("input/q2_kernel_1_sample_1_x.txt")
        kernel_1_sample_1_y = np.loadtxt("input/q2_kernel_1_sample_1_y.txt")
        kernel_1_sample_2_x = np.loadtxt("input/q2_kernel_1_sample_2_x.txt")
        kernel_1_sample_2_y = np.loadtxt("input/q2_kernel_1_sample_2_y.txt")
        model.fit(kernel_1_sample_2_x,kernel_1_sample_2_y)
        kernel_1_sample_2_ypred = model.predict(kernel_1_sample_2_x)
        kernel_1_sampel_1_ypred=  model.predict(kernel_1_sample_1_x)
        print("Training accuracy", sum(kernel_1_sample_2_ypred==kernel_1_sample_2_y)/len(kernel_1_sample_2_ypred))
        print("Test accuracy", sum(kernel_1_sampel_1_ypred==kernel_1_sample_1_y)/len(kernel_1_sampel_1_ypred))
    except:
        print("Error while solving Q2 kernel 1")

    print("Kernel 2")
    try:
        model = svm.SVC(kernel=Q2_kernel_2, C=10, gamma=1/2, shrinking=False)
        kernel_2_sample_1_x = np.loadtxt("input/q2_kernel_2_sample_1_x.txt")
        kernel_2_sample_1_y = np.loadtxt("input/q2_kernel_2_sample_1_y.txt")
        kernel_2_sample_2_x = np.loadtxt("input/q2_kernel_2_sample_2_x.txt")
        kernel_2_sample_2_y = np.loadtxt("input/q2_kernel_2_sample_2_y.txt")
        model.fit(kernel_2_sample_2_x,kernel_2_sample_2_y)
        kernel_2_sample_2_ypred = model.predict(kernel_2_sample_2_x)
        kernel_2_sampel_1_ypred=  model.predict(kernel_2_sample_1_x)
        print("Training accuracy", sum(kernel_2_sample_2_ypred==kernel_2_sample_2_y)/len(kernel_2_sample_2_ypred))
        print("Test accuracy", sum(kernel_2_sampel_1_ypred==kernel_2_sample_1_y)/len(kernel_2_sampel_1_ypred))
    except:
        print("Error while solving Q2 kernel 2")

    print("Kernel 3")
    try:
        model = svm.SVC(kernel=Q2_kernel_3, C=10, gamma=1/2, shrinking=False)
        kernel_3_sample_1_x = np.loadtxt("input/q2_kernel_3_sample_1_x.txt")
        kernel_3_sample_1_y = np.loadtxt("input/q2_kernel_3_sample_1_y.txt")
        kernel_3_sample_2_x = np.loadtxt("input/q2_kernel_3_sample_2_x.txt")
        kernel_3_sample_2_y = np.loadtxt("input/q2_kernel_3_sample_2_y.txt")
        model.fit(kernel_3_sample_2_x,kernel_3_sample_2_y)
        kernel_3_sample_2_ypred = model.predict(kernel_3_sample_2_x)
        kernel_3_sampel_1_ypred=  model.predict(kernel_3_sample_1_x)
        print("Training accuracy", sum(kernel_3_sample_2_ypred==kernel_3_sample_2_y)/len(kernel_3_sample_2_ypred))
        print("Test accuracy", sum(kernel_3_sampel_1_ypred==kernel_3_sample_1_y)/len(kernel_3_sampel_1_ypred))
    except:
        print("Error while solving Q2 kernel 2")

    print("#"*15, "Q3", "#"*15)
    try:
        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.2, num=3)

        conv_param = {'stride': 2, 'pad': 1}
        out, _ = q3(x, w, b, conv_param)
        correct_out = np.array([[[[-0.08759809, -0.10987781],
                                [-0.18387192, -0.2109216 ]],
                                [[ 0.21027089,  0.21661097],
                                [ 0.22847626,  0.23004637]],
                                [[ 0.50813986,  0.54309974],
                                [ 0.64082444,  0.67101435]]],
                                [[[-0.98053589, -1.03143541],
                                [-1.19128892, -1.24695841]],
                                [[ 0.69108355,  0.66880383],
                                [ 0.59480972,  0.56776003]],
                                [[ 2.36270298,  2.36904306],
                                [ 2.38090835,  2.38247847]]]])

        # Compare your output to ours; difference should be around e-8
        print('Testing your convolution implementation')
        print('error: ', relative_error(out, correct_out))
    except:
        print("Error while solving Q3")

    print("#"*15, "Q4", "#"*15)
    try:
        image_path = "input/Valve.png"
        sobel_fil_img = q4(image_path)
        print("Q4 has finished executing")
    except:
        print("Error while solving Q4")
    print("#"*15, " DONE ", "#"*15)
