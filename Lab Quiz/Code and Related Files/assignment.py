import numpy as np
import torch
import pickle as pkl
import torch.nn as nn
import torch.utils.data as data_utils


def train_val_split(X:torch.Tensor, y1:torch.Tensor, y2:torch.Tensor, train_pc):
    """This function splits the training dataset into train and validation datasets

    Args:
        X (_type_): The input torch 2D tensor
        y1 (_type_): classification target vector tensor
        y2 (_type_): regression target vector tensor
        train_pc (_type_): float \in (0, 1)
    
    Returns:
        X_trn, y1_trn, y2_trn, X_val, y1_val, y2_val
    """

    X_trn, y1_trn, y2_trn, X_val, y1_val, y2_val = None, None, None, None, None, None

    ## Start TODO
    np.random.seed(0)
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    num_train_samples = np.floor(num_samples * train_pc)
    train_indices = np.random.choice(indices, int(num_train_samples), replace=False)
    val_indices = list(set(indices) - set(train_indices))
    X_trn, y1_trn, y2_trn, X_val, y1_val, y2_val = X[train_indices], y1[train_indices],y2[train_indices], X[val_indices], y1[val_indices], y2[val_indices]

    ## End TODO

    assert X_trn.shape[0] + X_val.shape[0] == X.shape[0]
    return  X_trn, y1_trn, y2_trn, X_val, y1_val, y2_val

def accuracy(preds:torch.Tensor, targets:torch.Tensor):
    """Rerurns the accuracy of the model predictions

    Args:
        preds (_type_): vector of classification predictions tensor
        targets (_type_): vector of ground truth classification targets tensor
    Returns:
        acc: float between (0, 1)
    """
    acc = None

    ## Start TODO
    pred_n = preds.cpu().detach().numpy()
    targets_n = targets.cpu().detach().numpy()
    N = targets_n.shape[0]
    acc = (targets_n == pred_n).sum() / N

    ## End TODO

    return acc

def precision(preds:torch.Tensor, targets:torch.Tensor):
    """Rerurns the precision of the model predictions

    Args:
        preds (_type_): vector of classification predictions tensor
        targets (_type_): vector of ground truth classification targets tensor
    Returns:
        precision: float between (0, 1)
    """
    precision = None

    ## Start TODO
    pred_n = preds.cpu().detach().numpy()
    targets_n = targets.cpu().detach().numpy()
    TP = ((pred_n == 1) & (targets_n == 1)).sum()
    FP = ((pred_n == 1) & (targets_n == 0)).sum()
    precision = TP / (TP+FP)

    ## End TODO

    return precision

def recall(preds:torch.Tensor, targets:torch.Tensor):
    """Rerurns the recall of the model predictions

    Args:
        preds (_type_): vector of classification predictions tensor
        targets (_type_): vector of ground truth classification targets tensor
    Returns:
        recall: float between (0, 1)
    """
    recall = None

    ## Start TODO
    pred_n = preds.cpu().detach().numpy()
    targets_n = targets.cpu().detach().numpy()
    TP = ((pred_n == 1) & (targets_n == 1)).sum()
    FN = ((pred_n == 0) & (targets_n == 1)).sum()
    recall = TP / (TP+FN)

    ## End TODO

    return recall

def f1_score(preds:torch.Tensor, targets:torch.Tensor):
    """Rerurns the F1-Score of the model predictions

    Args:
        preds (_type_): vector of classification predictions tensor
        targets (_type_): vector of ground truth classification targets tensor
    Returns:
        f1_score: float between (0, 1)
    """
    f1 = None

    ## Start TODO
    p = precision(preds,targets)
    r = recall(preds,targets)
    f1 = (2.0*p*r)/(p+r)

    ## End TODO

    return f1

def mean_squared_error(preds:torch.Tensor, targets:torch.Tensor):
    """Rerurns the mse of the model predictions

    Args:
        preds (_type_): vector of regression predictions tensor
        targets (_type_): vector of ground truth regression targets tensor
    Returns:
        mse: float
    """
    mse = None

    ## Start TODO
    pred_n = preds.cpu().detach().numpy()
    targets_n = targets.cpu().detach().numpy()
    mse = np.mean(np.square(targets_n-pred_n))

    ## End TODO

    return mse

def mean_absolute_error(preds:torch.Tensor, targets:torch.Tensor):
    """Rerurns the mae of the model predictions

    Args:
        preds (_type_): vector of regression predictions tensor
        targets (_type_): vector of ground truth regression targets tensor
    Returns:
        mae: float between
    """
    mae = None

    ## Start TODO
    pred_n = preds.cpu().detach().numpy()
    targets_n = targets.cpu().detach().numpy()
    mae = np.mean(np.abs(targets_n-pred_n))

    ## End TODO

    return mae


def predict_labels(model:nn.Module, X_tst:torch.Tensor):
    """This function makes the predictions for the multi-task model. 

    Args:
        model (nn.Module): trained torch model
        X_tst (torch.Tensor): test Tensor
    Returns:
        y1_preds: a tensor vector containing classificatiopn predictions
        y2_preds: a tensor vector containing regression predictions
    """
    y1_preds, y2_preds = None, None

    ## start TODO
    y1, y2 = model(X_tst)
    y1_p = torch.reshape(y1,(-1,))
    y2_preds = torch.reshape(y2,(-1,))
    y1_preds = torch.where(y1_p <= 0.5, 0, 1)


    ## End TODO

    assert len(y1_preds.shape) == 1 and len(y2_preds.shape) == 1
    assert y1_preds.shape[0] == X_tst.shape[0] and y2_preds.shape[0] == X_tst.shape[0]
    assert len(torch.where(y1_preds == 0)[0]) + len(torch.where(y1_preds == 1)[0]) == X_tst.shape[0], "y1_preds should only contain classification targets"
    return y1_preds, y2_preds

def evaluate(loader, model):
    
    model.eval() # This enables the evaluation mode for the model

    eval_score = 0
    for data in loader:
        with torch.no_grad():
            pred1, pred2 = model(data[0])
            label1 = data[1]
            label2 = data[2] 
            label1 = torch.reshape(label1,pred1.shape)
            label2 = torch.reshape(label2,pred2.shape)
            loss1 = torch.nn.MSELoss()
            loss2 = torch.nn.MSELoss()
            lambd = 1    
            loss = loss1(pred1,label1) + lambd*(loss2(pred2,label2))  
            eval_score = -1.0*loss

    return eval_score

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared_layer1 = nn.Linear(28, 20)
        self.shared_layer2 = nn.Linear(20, 10)

        self.sf = nn.Sigmoid()
        self.linear_output = nn.Linear(10, 1)

    def forward(self, x):
        x = self.shared_layer1(x)
        x = self.shared_layer2(x)
        sf_out = self.linear_output(x)
        sf_out = self.sf(sf_out)
        linear_out = self.linear_output(x)

        return sf_out, linear_out


if __name__ == "__main__":

    # Load the dataset
    with open("dataset_train.pkl", "rb") as file:
        dataset_trn = pkl.load(file)
        X_trn, y1_trn, y2_trn = dataset_trn
        X_trn, y1_trn, y2_trn = torch.Tensor(X_trn), torch.Tensor(y1_trn), torch.Tensor(y2_trn)
    with open("dataset_test.pkl", "rb") as file:
        X_tst = pkl.load(file)
        X_tst = torch.Tensor(X_tst)
    
    X_trn, y1_trn, y2_trn, X_val, y1_val, y2_val = train_val_split(X=X_trn, y1=y1_trn, y2=y2_trn, train_pc=0.7)

    model = None
    ## start TODO
    # Your model definition, model training, validation etc goes here
    model = Network()
    loss1 = torch.nn.MSELoss()
    loss2 = torch.nn.MSELoss()
    lambd = 2  #select lambda sa per your choice

    tr_dataset = data_utils.TensorDataset(X_trn, y1_trn, y2_trn)
    loader = data_utils.DataLoader(tr_dataset)
    eval_dataset = data_utils.TensorDataset(X_val, y1_val, y2_val)
    eval_loader = data_utils.DataLoader(eval_dataset)
    # build model
    opt = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()),lr=1e-3)
    losses = []
    val_accs = []
    for epoch in range(1000):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred1, pred2 = model(batch[0])
            label1 = batch[1]
            label2 = batch[2]
            label1 = torch.reshape(label1,pred1.shape)
            label2 = torch.reshape(label2,pred2.shape)
            loss = loss1(pred1,label1) + lambd*(loss2(pred2,label2))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        losses.append(total_loss)
        val_acc = evaluate(eval_loader, model)
        val_accs.append(val_acc)
        #print("Epoch ", epoch, "Loss: ", total_loss, "Val Acc.: ", val_acc)

    ## END TODO

    y1_preds, y2_preds = predict_labels(model, X_tst=X_tst)
    
    # You can test the metrics code -- accuracy, precision etc. using training data for correctness of your implementation

    # dump the outputs
    with open("output.pkl", "wb") as file:
        pkl.dump((y1_preds, y2_preds), file)
    with open("model.pkl", "wb") as file:
        pkl.dump(model, file)
