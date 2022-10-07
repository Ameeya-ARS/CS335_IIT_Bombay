#!/usr/bin/env python
# coding: utf-8

# In[14]:
import torch
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import pickle as pkl
import time

def set_seed(seed:int=42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    torch.manual_seed(seed)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))    
        x = self.predict(x)      
        return x

def train_simple_regression(x, y):
    net = Net(n_feature=1, n_hidden=10, n_output=1)   
    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)

    plt.ion() 

    for t in range(200):
        prediction = net(x)
        optimizer.zero_grad()

        loss = torch.sum((prediction - y.t())**2)/y.shape[1]

        loss.backward()         
        optimizer.step() 
        
        # IMPORTANT: During submission, pls make comemnt this code by making if True --> if false
        # If code produces plots, the asuto-grading script would fail 
        if False:
            if t % 5 == 0:
                plt.cla()
                plt.scatter(x.data.numpy(), y.data.numpy())
                plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
                plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
                plt.pause(0.1)

        plt.ioff()
        plt.show()

    # Get the predictions from the model
    preds = net(x)
    return preds


# Set embedding starts here
class SetEmbed(torch.nn.Module):
    def __init__(self, num_feature):
        super(SetEmbed, self).__init__()
       
        self.init_layer = [torch.nn.Linear(num_feature, 16),\
                    torch.nn.ReLU(),\
                    torch.nn.Linear(16, 12)]
        self.init_layer = torch.nn.Sequential(*self.init_layer)

        self.net = [torch.nn.Linear(12, 4)]
        self.net = torch.nn.Sequential(*self.net)
         
         
    def forward(self, x, set_sizes=None):
        """
          input: x of shape  BATCH_SIZE * MAX_SET_SIZE * EMEDDING_DIM
                Batch Size: indicates the number of sets being provided as input
                MaxSetSize: is the maximum cardinality of the sets. Sets with fewer elements are padded with 0
          input: set_sizes is a list of integers denoting the actual number of items in each of the input sets
        """
        assert len(x.shape)==3
        assert(x.shape[0]==len(set_sizes))
        x  = self.init_layer(x)
        ############ TODO: Put your code here ##########
        """
          Hint: Create appropriate mask and apply on x
        """
        for i in range(x.shape[0]):
            x[i,set_sizes[i]:,:]=0
        ################################################
        x = torch.sum(x,dim=1)
        x  = self.net(x)      
        return x


# Ranking loss start here
def naive_pairwise_ranking_loss(predPos, predNeg, qidPos, qidNeg):
    """
    predPos: Tensor of shape (num_pos_pairs) 
             Each entry is a REAL valued score for the some positive query-corpus pair 
    qidPos:  Tensor of shape (num_pos_pairs) 
             Each entry is a INTEGER denoting the query id for the corresponding positive query-corpus pair, whose scores are stored in predPos. To clarify, if predPos[0] is M(qi, ci), then qidPos[0] would be i. 
    predNeg: Tensor of shape (num_neg_pairs)
             Each entry is a REAL valued score for the some negative query-corpus pair 
    qidNeg:  Tensor of shape (num_neg_pairs) 
             Each entry is a INTEGER denoting the query id for the corresponding negative query-corpus pair, whose scores are stored in predNeg.  To clarify, if predNeg[0] is M(qi, ci), then qidNeg[0] would be i.                  
    """
    loss = 0 
    num_comparisons = 0
    num_pos_pairs = len(qidPos)
    num_neg_pairs = len(qidNeg)
    for pos_loc in range(num_pos_pairs):
        for neg_loc in range(num_neg_pairs):
            if qidPos[pos_loc] == qidNeg[neg_loc]:
                loss += torch.nn.ReLU()(predNeg[neg_loc]-predPos[pos_loc])
                num_comparisons += 1
    return loss / num_comparisons

def incorrectly_tensorized_pairwise_ranking_loss(predPos, predNeg, qidPos, qidNeg):
    ############ TODO: Fix below line of code ##########
    return (torch.nn.ReLU()(predNeg[:,None] - predPos[None,:]) * (qidNeg[:,None] == qidPos[None,:])).sum() / (qidNeg[:,None] == qidPos[None,:]).sum()


def naive_subset_selection(X, S):
    """
    X : Tensor of shape (N x D)
    S : List, where each element is a subset (a subset is represented by tensor of indices)
    """
    output = []
    for s in S:
        A = X[s,:]
        B = X[s,:].T
        output.append(torch.mm(A, B))
    return output


def fast_subset_selection(X, S):
    """
    X : Tensor of shape (N x D)
    S : List, where each element is a subset (a subset is represented by tensor of indices)
    """
    output = None
    ############ TODO: Put your code here ##########
    temp = torch.matmul(X,X.T)
    output = []
    for s in S:
        res1 = temp[s,:]
        res2 = res1[:,s]
        output.append(res2)
    ################################################
    return output


if __name__ == "__main__":

    # Simple Regression
    with open("simple_reg.pkl", "rb") as file:
        x, y = pkl.load(file)

    predictions = train_simple_regression(x, y)

    # Set embeddings

    # you can change input_dim and set_sizes to debug the code
    input_dim = 12
    set_sizes = [5,7,13]

    datain_list = [torch.rand(x, input_dim) for x in set_sizes]
    datain_padded =  torch.nn.utils.rnn.pad_sequence(datain_list,batch_first=True)

    set_seed()
    ds = SetEmbed(input_dim)

    # This assert should pass after the bug is fixed
    assert torch.allclose(torch.concat([ds(datain_list[idx].unsqueeze(0),[set_sizes[idx]]) for idx in range(len(set_sizes))]), ds(datain_padded,set_sizes) )

    # Ranking Loss
    predPos = torch.rand(100)
    predNeg = torch.rand(100)
    qidPos = list(range(10))*10
    qidNeg = list(range(10))*10
    random.shuffle(qidPos)
    random.shuffle(qidNeg)
    qidPos = torch.tensor(qidPos)
    qidNeg = torch.tensor(qidNeg)

    # This assert should pass after the bug is fixed
    assert torch.isclose(incorrectly_tensorized_pairwise_ranking_loss(predPos, predNeg, qidPos, qidNeg), naive_pairwise_ranking_loss(predPos, predNeg, qidPos, qidNeg))

    # Subset selection from matrices
    N = 100
    d = 1000
    X = torch.rand(N, d).to(dtype=torch.float64)
    S = []
    
    for i in range(1000):
        n = random.randint(0, N)
        S.append(torch.randperm(N)[:n])  # Store the subset of indices in the list S
    
    t1 = time.time()
    output_naive = naive_subset_selection(X, S)
    t2 = time.time()
    output_fast = fast_subset_selection(X, S)
    t3 = time.time()

    assert(abs(output_naive[0] - output_fast[0]).sum() < 0.001)  # Check the correctness of the fast algorithm
    
    # Check the efficiency of the fast algorithm
    print("Time taken for naive selection = ", t2 - t1) 
    print("Time taken for fast selection = ", t3 - t2)
