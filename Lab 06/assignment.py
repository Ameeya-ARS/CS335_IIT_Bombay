#!/usr/bin/env python
# coding: utf-8

# In[14]:
import pickle
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import random
import pandas as pd


def set_seed(seed: int = 42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    torch.manual_seed(seed)


class LogisticRegression:
    def __init__(self, xdim, ydim, args) -> None:
        """This is the model class for Logistic Regression
        
        Args:
            xdim - Dimension of input
            ydim - Number of output nodes. 1 for 2-class classification.
            args - dictionary containing hyper-parameters used in the model 
        """
        self.xdim = xdim
        self.ydim = ydim
        self.args = args

        self.weights, self.bias = self.init_weights(xdim, ydim)
        assert self.weights.shape == (xdim,)

    def init_weights(self, xdim, ydim):
        """Initializes the weights of the logistic regression model. 
        We have weights of shape (xdim,) and bias a scalar

        Returns:
            w, b
        """

        ## TODO

        w = torch.zeros((xdim,))
        b = torch.zeros((1,))

        ## END TODO

        return w, b

    def forward(self, batch_x):
        """ Runs forward on a batch of features x and returns the predictions in yhat variable
        The code must be vectorized.

        Returns:
            The predictions in a tensor of shape  (batch_x.shape[0], )
        """
        assert (len(batch_x.size()) == 2)
        yhat = None

        ## TODO

        yhat = torch.sigmoid(torch.matmul(batch_x, self.weights) + self.bias)

        ## End TODO

        assert yhat.shape == (batch_x.shape[0],)
        return yhat

    def backward(self, batch_x, batch_y, batch_yhat):
        """Computes the gradients and updates the weights with the learning rate set to self.args.lr
            
            Returns the updated weights
        """
        assert (len(batch_x.size()) == 2)
        assert (len(batch_y.size()) == 1)
        assert (len(batch_yhat.size()) == 1)

        weights_new, bias_new = None, None

        ## TODO

        w = torch.matmul(torch.t(batch_x),(batch_yhat-batch_y))
        b = torch.sum(batch_yhat-batch_y)

        weights_new = self.weights - (self.args.lr*w)
        bias_new = self.bias - (self.args.lr*b)

        ## End TODO

        self.weights = weights_new
        self.bias = bias_new
        return weights_new, bias_new

    def loss(self, y, y_hat):
        """Computes the loss and returns a scalar
        """
        assert (len(y.size()) == 1)
        assert (len(y_hat.size()) == 1)
        loss = 0

        ## TODO
        temp = y*torch.log(y_hat) +  (1-y)*torch.log(1-y_hat)
        loss = -torch.mean(temp)
        ## End TODO

        assert len(loss.shape) == 0, "loss should be a torch scalar"
        return loss

    def score(self, yhat, y):
        """Computes the accuracy of the model
        """
        assert (len(y.size()) == 1)
        assert (len(yhat.size()) == 1)
        yhat = yhat > 0.5
        return torch.sum(yhat == y) / float(y.shape[0])


class LinearRegression:
    def __init__(self, xdim, args) -> None:
        """This is the model class for Linear Regression
        
        Args:
            xdim - Dimension of input
            args - dictionary containing hyper-parameters used in the model 
        """
        self.xdim = xdim
        self.args = args

        self.weights, self.bias = self.init_weights(xdim)
        assert self.weights.shape == (xdim,)

    def init_weights(self, xdim):
        """Initializes the weights of the logistic regression model. 
        We have weights of shape (xdim,) and bias a scalar

        Returns:
            w, b
        """
        w, b = None, None

        ## TODO
        w = torch.zeros((xdim,))
        b = torch.zeros((1,))

        ## END TODO

        return w, b

    def forward(self, batch_x):
        """ Runs forward on a batch of features x and returns the predictions in yhat variable
        The code must be vectorized.

        Returns:
            the predictions
        """
        assert (len(batch_x.size()) == 2)

        yhat = None

        ## TODO
        yhat = torch.matmul(batch_x, self.weights) + self.bias
        ## End TODO

        assert yhat.shape == (batch_x.shape[0],)
        return yhat

    def backward(self, batch_x, batch_y, batch_yhat):
        """Computes the gradients and updates the weights with the self.args.lr
            
            Returns the updated weights
        """
        assert (len(batch_x.size()) == 2)
        assert (len(batch_y.size()) == 1)
        assert (len(batch_yhat.size()) == 1)

        weights_new, bias_new = None, None

        ## TODO
        w = (2/batch_x.shape[0])*torch.matmul(torch.t(batch_x),(batch_yhat-batch_y))
        b = (2/batch_x.shape[0])*torch.sum(batch_yhat-batch_y)

        weights_new = self.weights - (self.args.lr*w)
        bias_new = self.bias - (self.args.lr*b)
        ## End TODO

        self.weights = weights_new
        self.bias = bias_new
        return weights_new, bias_new

    def loss(self, y, y_hat):
        """Computes the loss and returns a scalar
        """
        assert (len(y.size()) == 1)
        assert (len(y_hat.size()) == 1)
        loss = 0

        ## TODO
        loss = torch.mean(torch.square(y-y_hat))
        ## End TODO

        assert len(loss.shape) == 0, "loss should be a torch scalar"
        return loss

    def score(self, yhat, y):
        """Computes the negative opf mean squared error as score.

        Args:
            yhat (_type_): predictions
            y (_type_): _targets

        Returns:
            negative of mse
        """
        assert (len(y.size()) == 1)
        assert (len(yhat.size()) == 1)
        return - torch.mean(torch.square(y - yhat))


class EarlyStoppingModule(object):
    """
      Module to keep track of validation score across epochs
      Stop training if score not imroving exceeds patience
    """

    def __init__(self, args):
        """
          input : args
          patience: number of epochs to wait for improvement in validation score
          delta: minimum difference between two validation scores that can be considered as an improvement
          best_score: keeps track of best validation score observed till now (while training)
          num_bad_epochs: keeps track of number of training epochs in which no improvement has been observed
          should_stop_now: boolean flag deciding whether training should early stop at this epoch
        """
        self.args = args
        self.patience = args.patience
        self.delta = args.delta
        self.best_score = None
        self.num_bad_epochs = 0
        self.should_stop_now = False

    def save_best_model(self, model, epoch):
        fname = f"./{self.args.model_type}_bestValModel.pkl"
        pickle.dump(model, open(fname, "wb"))
        print(f"INFO:: Saving best validation model at epoch {epoch}")

    def load_best_model(self):
        fname = f"./{self.args.model_type}_bestValModel.pkl"
        try:
            model = pickle.load(open(fname, "rb"))
            print(f"INFO:: Loading best validation model from {fname}")
        except Exception as e:
            print(f"INFO:: Cannot load best model due to exception: {e}")

        return model

    def check(self, curr_score, model, epoch):
        """Checks whether the current model has the best validation accuracy and decides to stop or proceed.
        If the current score on validation dataset is the best so far, it saves the model weights and bias.

        Args:
            curr_score (_type_): Score of the current model
            model (_type_): Trained Logistic/Linear model
            epoch (_type_): current epoch

        Returns:
            self.stop_now: Whether or not to stop

        Task1: Check for stoppage as per the early stopping criteria
        Task2: Save best model as required
        """

        ## TODO

        if self.best_score is None or curr_score > self.best_score + self.delta:
            self.best_score = curr_score
            self.save_best_model(model, epoch)
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if(self.num_bad_epochs>self.patience):
                self.should_stop_now = True

        ## END TODO
        return self.should_stop_now


def minibatch(trn_X, trn_y, batch_size):
    """
    Function that yields the next minibatch for your model training
    IMPORTANT: DO NOT RETURN
    """
    # TODO

    for i in range(trn_y.shape[0]//batch_size):
        x = trn_X[i*batch_size:(i + 1)*batch_size]
        y = trn_y[i*batch_size:(i + 1)*batch_size]

        yield x, y

    # TODO
    pass


def split_data(X: torch.Tensor, y: torch.Tensor, split_per=0.6):
    """Splits the dataset into train and test.
    """
    shuffled_idxs = torch.randperm(X.shape[0])
    trn_idxs = shuffled_idxs[0:int(split_per * X.shape[0])]
    tst_idxs = shuffled_idxs[int(split_per * X.shape[0]):]
    return X[trn_idxs], y[trn_idxs], X[tst_idxs], y[tst_idxs]


def train(args, X_tr, y_tr, X_val, y_val, model):
    es = EarlyStoppingModule(args)
    losses = []
    val_acc = []
    epoch_num = 0
    while (epoch_num <= args.max_epochs):
        for idx, (batch_x, batch_y) in enumerate(minibatch(X_tr, y_tr, args.batch_size)):
            if idx == 0:
                assert batch_x.shape[0] == args.batch_size
            batch_yhat = model.forward(batch_x)
            losses.append(model.loss(batch_y, batch_yhat).item())
            updated_wts = model.backward(batch_x, batch_y, batch_yhat)

        val_score = model.score(model.forward(torch.Tensor(X_val)), torch.Tensor(y_val))
        print(f"INFO:: Validation score at epoch {epoch_num}: {val_score}")
        val_acc.append(val_score)
        if es.check(val_score, model, epoch_num):
            break
        epoch_num += 1
    return losses, val_acc


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


if __name__ == '__main__':
    set_seed(0)

    """
    Logistic Regression
    """
    print("LOGISTIC REGRESSION")
    args = {'batch_size': 128,
            'max_epochs': 500,
            'lr': 1e-2,
            'patience': 15,
            'delta': 1e-4,
            'temp': 1.0,
            'model_type': 'logistic-moons'}

    args = objectview(args)

    """This is on synthetic 2 moons dataset"""
    print("MODEL:: Logistic Regression on 2-moons")
    datasets = make_moons(n_samples=1000, noise=0.05, random_state=0)
    X, y = datasets
    X, y = torch.Tensor(X), torch.Tensor(y)

    X_train, y_train, X_val, y_val = split_data(X, y, )

    LogR = LogisticRegression(X_train.shape[1], 1, args)
    losses, val_acc = train(args, X_train, y_train, X_val, y_val, model=LogR)

    es = EarlyStoppingModule(args)
    best_model = es.load_best_model()

    print(LogR.score(best_model.forward(torch.Tensor(X_val)), torch.Tensor(y_val)))

    """This is on Real-world titanic dataset. We have done all the data pre-processing."""
    print("MODEL:: Logistic Regression on Titanic")
    
    args = {'batch_size': 64,
            'max_epochs': 500,
            'lr': 1e-2,
            'patience': 15,
            'delta': 1e-4,
            'temp': 1.0,
            'model_type': 'logistic-titanic'}
    
    args = objectview(args)
    
    with open("titanic_trn.pkl", "rb") as file:
        X, y = pickle.load(file)
    X, y = torch.Tensor(X), torch.Tensor(y)
    
    X_train, y_train, X_val, y_val = split_data(X, y, )
    
    LogR = LogisticRegression(X_train.shape[1], 1, args)
    losses, val_acc = train(args, X_train, y_train, X_val, y_val, model=LogR)
    
    es = EarlyStoppingModule(args)
    best_model = es.load_best_model()
    
    print(LogR.score(best_model.forward(torch.Tensor(X_val)), torch.Tensor(y_val)))
    
    """
    Linear Regression
    """
    print("MODEL:: LINEAR REGRESSION")
    args = {'batch_size': 32,
            'max_epochs': 500,
            'lr': 1e-1,
            'patience': 15,
            'delta': 1e-4,
            'model_type': 'linear'}
    
    args = objectview(args)
    
    data = pd.read_csv('dataset.csv', index_col=0)
    X = (data.iloc[:, :-1].to_numpy())
    y = (data.iloc[:, -1].to_numpy())
    X, y = torch.Tensor(X), torch.Tensor(y)
    
    X_train, y_train, X_val, y_val = split_data(X, y, )
    
    LinR = LinearRegression(X_train.shape[1], args)
    losses, val_acc = train(args, X_train, y_train, X_val, y_val, model=LinR)
    
    es = EarlyStoppingModule(args)
    best_model = es.load_best_model()
    
    print(LinR.score(best_model.forward(torch.Tensor(X_val)), torch.Tensor(y_val)))
