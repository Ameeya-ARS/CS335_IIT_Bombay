
import time
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
from sklearn import svm


############################ Q3 ################################

def q3(x, w, b, conv_param):
    """
    A naive implementation of convolution.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.
    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
   
    padding, stride = conv_param['pad'], conv_param['stride']
    (N, C, H, W) = x.shape
    (F, C, HH, WW) = w.shape
    H1 = 1 + (H + 2* padding - HH ) // stride
    W1 = 1 + (W + 2 * padding - WW ) // stride
    if padding != 0 : 
      x_temp = np.pad(x,[(0,0),(0,0),(padding,padding),(padding,padding)])
    else :
      x_temp = np.copy(x)
    
    out = np.zeros((N,F,H1,W1))
    
    for n in range(N) :
      for f in range(F) :
        for i in range(H1) :
          for j in range(W1) :
            out[n,f,i,j] = np.sum(x_temp[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] * w[f]) + b[f]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache

def gram(x):
  ######START: TO CODE########
  #Returns the gram matrix
  x = np.dot(x.T,x)
  return x
  ######END: TO CODE########


def relative_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))