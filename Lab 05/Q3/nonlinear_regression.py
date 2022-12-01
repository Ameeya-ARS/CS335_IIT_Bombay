from matplotlib import pyplot as plt
import torch
from torch import nn
import numpy as np

data_x,data_y = np.load('data.npy')
data_Y = torch.from_numpy(data_y)
# Concatenate second order polynomial feature for x and store the result in data_X
data_X = torch.reshape(torch.from_numpy(data_x),(data_x.shape[0],1))
temp = torch.reshape(torch.from_numpy(np.square(data_x)),(data_x.shape[0],1))
data_X = torch.cat((data_X,temp),1)

def plotlosses(model,epoch, loss,name):
    plt.plot(data_x,data_y,'r.',label='Data')
    plt.plot(data_x,model(data_X).detach().numpy(),'b.',label='Prediction')
    plt.legend(loc='lower right')
    plt.title(f'EPOCH : {epoch} | LOSS : {loss}')
    plt.savefig(f'{name}_{epoch}.png')
    plt.close()

def train(X, Y, model, loss_fn, optim, max_iter, name=''):
    loss_list = []
    for epoch in range(max_iter):
        Y_pred = model(X)
        loss = loss_fn(Y_pred, Y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        if epoch % 1000 == 0:
            plotlosses(model,epoch, loss.data.item(),name)
            loss_list += [loss.data.item()]
    return loss_list

# Set a manual seed of 100 for the torch module
torch.manual_seed(100)
class NLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(2,1).double())
        self.b1 = nn.Parameter(torch.randn(1).double())
        self.w2 = nn.Parameter(torch.randn(1,1).double())
        self.b2 = nn.Parameter(torch.randn(1).double())
    
    def forward(self, X):
        # Implement the evaluation function
        temp1 = torch.matmul(X,self.w1) + self.b1
        temp2 = (self.w2*torch.tanh(temp1)) + self.b2
        res = torch.flatten(temp2)
        return res


nl_model = NLModel()

loss_function = nn.MSELoss()

nl_optim = torch.optim.SGD(nl_model.parameters(), lr=0.02)

loss_list = train(data_X,data_Y,nl_model,loss_function,nl_optim,10001,'non_linear')

# Save the the weights in lexicographic order AS PER THE PDF in 'weights.npy'
weight_list=torch.tensor([nl_model.w1[0][0],nl_model.w1[1][0],nl_model.b1,nl_model.w2[0][0],nl_model.b2])
weight_list = weight_list.numpy()

np.save('weights',weight_list)
np.save('losses', loss_list)
