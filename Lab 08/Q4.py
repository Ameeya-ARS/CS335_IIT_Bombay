
import time
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
from sklearn import svm


############################ Q4 ################################

def q4(img):
    '''
    @params
    img: string with image path
    return: Sobel Filtered image for X and Y direction type: np.ndarray
    '''
    image_file = img
    input_image = imread(image_file)  # this is the array representation of the input image
    [nx, ny, nz] = np.shape(input_image)  # nx: height, ny: width, nz: colors (RGB) 

    
    r_img, g_img, b_img = input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2]
    gamma = 1.400  
    r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
    grayscale_image = r_const * r_img ** gamma + g_const * g_img ** gamma + b_const * b_img ** gamma

    ######START: TO CODE########
    #Define the Sobel Filter for X and Y direction
    Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    ######END: TO CODE########

    [rows, columns] = np.shape(grayscale_image)  # we need to know the shape of the input grayscale image
    sobel_filtered_image = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)

    ######START: TO CODE########
    # Now we "sweep" the image in both x and y directions and compute the output and store in sobel_filtered_image
    def convolution(image, kernel):
        # Flip the kernel
        kernel = np.flipud(np.fliplr(kernel))
        output = np.zeros_like(image)

        # Add zero padding to the input image
        image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
        image_padded[1:-1, 1:-1] = image

        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                output[y, x] = (kernel * image_padded[y: y+3, x: x+3]).sum()

        return output
    new_image_x = convolution(grayscale_image, Gx)
    new_image_y = convolution(grayscale_image, Gy)
    sobel_filtered_image = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    sobel_filtered_image *= 255.0 / sobel_filtered_image.max()
    ######END: TO CODE########


    plt.imsave('output/imageWithEdges.png', sobel_filtered_image, cmap=plt.get_cmap('gray'))
    assert type(sobel_filtered_image) == np.ndarray
    return sobel_filtered_image