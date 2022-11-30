import numpy as np
from matplotlib import pyplot as plt
import math
import pickle as pkl

np.seterr(divide='ignore', invalid='ignore')
def preprocessing(X):
    """
    Implement Normalization for input image features

    Args:
    X : numpy array of shape (n_samples, n_features)

    Returns:
    X_out: numpy array of shape (n_samples, n_features) after normalization
    """
    X_out = None

    # TODO
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X_out = (X - mu)/sigma
    X_out[np.isnan(X_out)] = 0
    # END TODO

    assert X_out.shape == X.shape

    return X_out


def split_data(X, Y, train_ratio=0.8):
    '''
    Split data into train and validation sets
    The first floor(train_ratio*n_sample) samples form the train set
    and the remaining the validation set

    Args:
    X - numpy array of shape (n_samples, n_features)
    Y - numpy array of shape (n_samples, 1)
    train_ratio - fraction of samples to be used as training data

    Returns:
    X_train, Y_train, X_val, Y_val
    '''
    # Try Normalization and scaling and store it in X_transformed
    X_transformed = X

    # TODO
    X_transformed = preprocessing(X)
    # END TODO

    assert X_transformed.shape == X.shape

    num_samples = len(X)
    indices = np.arange(num_samples)
    num_train_samples = math.floor(num_samples * train_ratio)
    train_indices = np.random.choice(indices, num_train_samples, replace=False)
    val_indices = list(set(indices) - set(train_indices))
    X_train, Y_train, X_val, Y_val = X_transformed[train_indices], Y[
        train_indices], X_transformed[val_indices], Y[val_indices]

    return X_train, Y_train, X_val, Y_val


class FlattenLayer:
    '''
    This class converts a multi-dimensional into 1-d vector
    '''

    def __init__(self, input_shape):
        '''
         Args:
          input_shape : Original shape, tuple of ints
        '''
        self.input_shape = input_shape

    def forward(self, input):
        '''
        Converts a multi-dimensional into 1-d vector
        Args:
          input : training data, numpy array of shape (n_samples , self.input_shape)

        Returns:
          input: training data, numpy array of shape (n_samples , -1)
        '''
        # TODO
        self.n = input.shape[0]
        self.dim = input.ndim
        output = np.reshape(input,(input.shape[0],-1))
        # Modify the return statement to return flattened input
        return output
        # END TODO
        
    
    def backward(self, output_error, learning_rate):
        '''
        Converts back the passed array to original dimention 
        Args:
        output_error :  numpy array 
        learning_rate: float

        Returns:
        output_error: A reshaped numpy array to allow backward pass
        '''
        # TODO
        if self.dim == 3:
            output_error = np.reshape(output_error, (self.n, self.input_shape[0], self.input_shape[1]))
        if self.dim != 3:
            output_error = np.reshape(output_error, (self.n, self.input_shape))
        # Modify the return statement to return reshaped array
        return output_error
        # END TODO
        
        
class FCLayer:
    '''
    Implements a fully connected layer  
    '''
    def __init__(self, input_size, output_size):
        '''
        Args:
         input_size : Input shape, int
         output_size: Output shape, int 
        '''
        self.input_size = input_size
        self.output_size = output_size
        # TODO
        self.weights = np.random.normal(size=(self.input_size,self.output_size))#initilaise weights for this layer
        self.bias = np.random.normal(size=(self.output_size))#initilaise bias for this layer
        # END TODO

    def forward(self, input):
        '''
        Performs a forward pass of a fully connected network
        Args:
          input : training data, numpy array of shape (n_samples , self.input_size)

        Returns:
           numpy array of shape (n_samples , self.output_size)
        '''
        # TODO
        self.input = input
        res = self.weights.T[None,:,:]@input[:,:,None] + self.bias[None,:,None]
        res = np.squeeze(res)
        # Modify the return statement to return numpy array of shape (n_samples , self.output_size)
        return res
        # END TODO
        

    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a fully connected network along with updating the parameter 
        Args:
          output_error :  numpy array 
          learning_rate: float

        Returns:
          Numpy array resulting from the backward pass
        '''
        # TODO
        b_ = np.sum(output_error, axis=0)
        w_ = np.sum(output_error[:,None,:]*self.input[:,:,None], axis=0)
        w = self.weights.copy()
        self.weights = self.weights -learning_rate*w_
        self.bias = self.bias -learning_rate*b_
        res = np.squeeze(w[None,:,:]@output_error[:,:,None])

        # Modify the return statement to return numpy array resulting from backward pass
        return res
        # END TODO
        
        
class ActivationLayer:
    '''
    Implements a Activation layer which applies activation function on the inputs. 
    '''
    def __init__(self, activation, activation_prime):
        '''
          Args:
          activation : Name of the activation function (sigmoid,tanh or relu)
          activation_prime: Name of the corresponding function to be used during backpropagation (sigmoid_prime,tanh_prime or relu_prime)
        '''
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):
        '''
        Applies the activation function 
        Args:
          input : numpy array on which activation function is to be applied

        Returns:
           numpy array output from the activation function
        '''
        # TODO
        self.input = input
        res = self.activation(input)
        # Modify the return statement to return numpy array of shape (n_samples , self.output_size)
        return res
        # END TODO
        

    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a fully connected network along with updating the parameter 
        Args:
          output_error :  numpy array 
          learning_rate: float

        Returns:
          Numpy array resulting from the backward pass
        '''
        # TODO
        res = self.activation_prime(self.input)*output_error
        # Modify the return statement to return numpy array resulting from backward pass
        return res
        # END TODO
        
        

class SoftmaxLayer:
    '''
      Implements a Softmax layer which applies softmax function on the inputs. 
    '''
    def __init__(self, input_size):
        self.input_size = input_size
    
    def forward(self, input):
        '''
        Applies the softmax function 
        Args:
          input : numpy array on which softmax function is to be applied

        Returns:
           numpy array output from the softmax function
        '''
        # TODO
        self.input = input
        temp = np.max(input, axis=1)
        exps = np.exp(input-temp[:,None])
        res =  exps/np.sum(exps, axis=1)[:,None]
        # Modify the return statement to return numpy array of shape (n_samples , self.output_size)
        return res
        # END TODO
        
    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a Softmax layer
        Args:
          output_error :  numpy array 
          learning_rate: float

        Returns:
          Numpy array resulting from the backward pass
        '''
        # TODO
        temp = np.max(self.input,axis=1)
        exps = np.exp(self.input-temp[:,None])
        s = np.sum(exps, axis=1)
        exps = exps/s[:,None]
        a = -exps[:,:,None]*exps[:,None,:]
        t = np.eye(exps.shape[1])
        a += t[None,:,:]*exps[:,:,None]
        res = np.squeeze(a@output_error[:,:,None])
        # Modify the return statement to return numpy array resulting from backward pass
        return res
        # END TODO
        
        
def sigmoid(x):
    '''
    Sigmoid function 
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying simoid function
    '''
    # TODO
    res = np.zeros(x.shape)
    res[x>=500] = 1
    res[x<=-500] = 0
    res[np.logical_and(x<500, x>-500)] = 1/(1+np.exp(-x[np.logical_and(x<500, x>-500)])) # to avoid overflow and underflow error
    # Modify the return statement to return numpy array resulting from backward pass
    return res
    # END TODO

def sigmoid_prime(x):
    '''
     Implements derivative of Sigmoid function, for the backward pass
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying derivative of Sigmoid function
    '''
    # TODO
    res = sigmoid(x)*(1-sigmoid(x))
    # Modify the return statement to return numpy array resulting from backward pass
    return res
    # END TODO

def tanh(x):
    '''
    Tanh function 
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying tanh function
    '''
    # TODO
    res = np.tanh(x)
    # Modify the return statement to return numpy array resulting from backward pass
    return res
    # END TODO

def tanh_prime(x):
    '''
     Implements derivative of Tanh function, for the backward pass
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying derivative of Tanh function
    '''
    # TODO
    res = 1-tanh(x)**2
    # Modify the return statement to return numpy array resulting from backward pass
    return res
    # END TODO

def relu(x):
    '''
    ReLU function 
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying ReLU function
    '''
    # TODO
    res = np.maximum(0,x)
    # Modify the return statement to return numpy array resulting from backward pass
    return res
    # END TODO

def relu_prime(x):
    '''
     Implements derivative of ReLU function, for the backward pass
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying derivative of ReLU function
    '''
    # TODO
    res = x
    res[res>0]=1
    res[res<=0]=0
    # Modify the return statement to return numpy array resulting from backward pass
    return res
    # END TODO
    
def cross_entropy(y_true, y_pred):
    '''
    Cross entropy loss 
    Args:
        y_true :  Ground truth labels, numpy array 
        y_true :  Predicted labels, numpy array 
    Returns:
       loss : float
    '''
    # TODO
    loss = -np.sum(np.log(np.maximum(y_pred[np.arange(y_pred.shape[0]),y_true], 0.001)))
    # Modify the return statement to return numpy array resulting from backward pass
    return loss
    # END TODO

def cross_entropy_prime(y_true, y_pred):
    '''
    Implements derivative of cross entropy function, for the backward pass
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying derivative of cross entropy function
    '''
    # TODO
    #Modify the return statement to return numpy array resulting from backward pass
    out = np.zeros(y_pred.shape)
    out[np.arange(y_pred.shape[0]),y_true] = -1/np.maximum(y_pred[np.arange(y_pred.shape[0]),y_true],0.001)
    return out
    # END TODO
    
    
def fit(X_train, Y_train, dataset_name):

    '''
    Create and trains a feedforward network

    Do not forget to save the final model/weights of the feed forward network to a file. Use these weights in the `predict` function 
    Args:
        X_train -- np array of share (num_test, 2048) for flowers and (num_test, 28, 28) for mnist.
        Y_train -- np array of share (num_test, 2048) for flowers and (num_test, 28, 28) for mnist.
        dataset_name -- name of the dataset (flowers or mnist)
    
    '''
     
    # Note that this just a template to help you create your own feed forward network 
    # TODO

    # define your network
    # This example network below would work only for mnist
    # you will need separtae networks for the two datasets
    X_train = preprocessing(X_train)
    mu = np.mean(X_train,axis=0)
    sigma = np.std(X_train,axis=0)
    dict1 = {'mu':mu, 'sigma':sigma}

    if dataset_name == 'mnist':
        network = [
            FlattenLayer(input_shape=(28, 28)),
            FCLayer(28 * 28, 32),
            ActivationLayer(sigmoid, sigmoid_prime),
            FCLayer(32, 10),
            SoftmaxLayer(10)
        ] # This creates feed forward 

        # Choose appropriate learning rate and no. of epoch
        epochs = 40
        learning_rate = 0.01
        batch_size = 16
        # Change training loop as you see fit

    elif dataset_name == 'flowers':
        network = [
            FlattenLayer(input_shape=(2048)),
            FCLayer(2048, 32),
            ActivationLayer(sigmoid, sigmoid_prime),
            FCLayer(32, 5),
            SoftmaxLayer(5)
        ] # This creates feed forward 

        # # Choose appropriate learning rate and no. of epoch
        epochs = 50
        learning_rate = 0.01
        batch_size = 16
        # Change training loop as you see fit


    for epoch in range(epochs):
        error = 0
        idx = 0
        while idx < len(X_train):
            # forward
            x = X_train[idx:idx+batch_size]
            y_true = Y_train[idx:idx+batch_size]
            output = x
            for layer in network:
                output = layer.forward(output)
            
            # error (display purpose only)
            error += cross_entropy(y_true, output)

            # backward
            output_error = cross_entropy_prime(y_true, output)
            for layer in reversed(network):
                output_error = layer.backward(output_error, learning_rate)
            idx += batch_size
        error /= len(X_train)
        print('%d/%d, error=%f' % (epoch + 1, epochs, error))

    #Save you model/weights as ./models/{dataset_name}_model.pkl
    dict1['network']=network
    with open(f"./models/{dataset_name}_model.pkl", "wb") as f:
        pkl.dump(dict1, f)
    ## END TODO
    
def predict(X_test, dataset_name):
    """

    X_test -- np array of share (num_test, 2048) for flowers and (num_test, 28, 28) for mnist.

    This is the function that we will call from the auto grader. 

    This function should only perform inference, please donot train your models here.

    Steps to be done here:
    1. Load your trained model/weights from ./models/{dataset_name}_model.pkl
    2. Ensure that you read model/weights using only the libraries we have given above.
    3. In case you have saved weights only in your file, itialize your model with your trained weights.
    4. Compute the predicted labels and return it

    Return:
    Y_test - nparray of shape (num_test,)
    """
    Y_test = np.zeros(X_test.shape[0],)
    # TODO
    with open(f"./models/{dataset_name}_model.pkl","rb") as f:
        dict1 = pkl.load(f)
    network = dict1['network']
    mu_ = dict1['mu']
    sigma_ = dict1['sigma']
    out = (X_test-mu_)/sigma_
    for layer in network:
        out = layer.forward(out)
    Y_test = np.argmax(out,axis=1)
    # END TODO
    assert Y_test.shape == (X_test.shape[0],) and type(Y_test) == type(X_test), "Check what you return"
    return Y_test
    
"""
Loading data and training models
"""
if __name__ == "__main__":    
    
    dataset = "flowers"
    with open(f"./data/{dataset}_train.pkl", "rb") as file:
        train_flowers = pkl.load(file)
        print(f"train_x -- {train_flowers[0].shape}; train_y -- {train_flowers[1].shape}")
    
    fit(train_flowers[0],train_flowers[1],'flowers')

