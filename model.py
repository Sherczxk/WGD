import pickle
import math
import torch
import torch.nn as nn 
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric as tg
import torch.autograd as autograd
from torch.nn.modules.module import Module
from torch.autograd import Variable
import numpy as np
import scipy.sparse as sp
import random
from scipy.stats import wasserstein_distance
import ot
import time
from collections import defaultdict
from utils import *
from copy import deepcopy
from dataloader import *
import barycenter

    
def Wcostmatrix(X,Y):
    '''
    input: X pre_base n0 k-D Gaussians components tensor [2,n0,k]
           Y new_base n1 k-D Gaussians components tensor [2,n1,k]
    return: cost matrix tensor [n0,n1]
    '''
    cost_m = torch.empty(X.shape[0],Y.shape[0]).cuda()
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            cost_m[i][j] = torch.norm(X[i]-Y[j],p=1)
    return cost_m 

def GMM_bary_weight(discrete_gmm,weight, costMatrix):
    """
    Input :
     - discrete_gmm: tensor [L,n0] -> weights of n0 different components
     - weight : tensor [L] -> weights of L different GMMs
     - costMatrix : tensor [n0,n1]  -> cost matrix

    Output :
     - Weights :tensor [n1] corresponding weights
    """

    weight = weight.unsqueeze(0)
    Weights = barycenter.compute(discrete_gmm.T,costMatrix,weight)
    return Weights
    

def Barycenter_update(input_gmms,adj_list,degree,bary_weight, costMatrix):
    n = input_gmms.shape[0]
    output_gmm = list()
    for j in range(n): 
        neighs_list = list(adj_list[j])
        neighs_list.append(j)
        if bary_weight == 'laplace':
            weight = torch.FloatTensor([(degree[j]*degree[k])**(-0.5) for k in neighs_list]).cuda()
            bary_weights = weight/torch.sum(weight)
        elif bary_weight == 'uniform':
            bary_weights = 1./degree[j]*torch.ones(len(neighs_list)).cuda()
        elif bary_weight == 'self_loop':
            weights = [1./(2*(degree[j]-1))]*(degree[j]-1)
            weights.append(1-sum(weights))
            bary_weights = torch.FloatTensor(weights).cuda()
            #bary_weights = weights/torch.sum(weights)
        input_gmm = input_gmms[neighs_list]
        
        bary_weights = bary_weights.unsqueeze(0)
        bary = barycenter.compute(input_gmm.T,costMatrix,bary_weights)

        output_gmm.append(bary)
    output_gmm = torch.stack(output_gmm)
    return output_gmm

class WGCN_layer(torch.nn.Module):
    def __init__(self,adj_list,degree):
        super(WGCN_layer, self).__init__()
        
        self.adj_list = adj_list
        self.degree = degree
    
    def forward(self,trans_X,costMatrix):
        bary_weight = 'uniform'
        new_X = Barycenter_update(trans_X,self.adj_list,self.degree,bary_weight, costMatrix) 
        return new_X
        
class WGCN_layer(torch.nn.Module):
    def __init__(self,adj_list,degree):
        super(WGCN_layer, self).__init__()
        
        self.adj_list = adj_list
        self.degree = degree
    
    def forward(self,trans_X,costMatrix):
        bary_weight = 'uniform'
        new_X = Barycenter_update(trans_X,self.adj_list,self.degree,bary_weight, costMatrix) 
        return new_X
        
class WGCN(torch.nn.Module):
    def __init__(self,data,rate,missing_type,hidden_dim,class_num,n_component,drop_out=0.5, bias=True,**kwargs):
        super(WGCN, self).__init__()
        
        imputer_type = 'zero'
        cost_type='normalize'
        self.nn_type = 'gcn'
        lambda_ = 0
        
        features = deepcopy(data.features)
        features = missing_feature(features,rate,missing_type)
        
        self.X = self.impute_feature(features,imputer_type)
        
        self.n_component = n_component
        self.n_node, self.feature_dim = data.features.shape
        self.adj_list,self.degree =  adjlist(data.edge_index)
        self.drop_out = drop_out
        self.edge_index = data.edge_index.cuda()       
        
        self.wgc1 = WGCN_layer(self.adj_list,self.degree)
        self.wgc2 = WGCN_layer(self.adj_list,self.degree)
        
        if self.nn_type == 'mlp':
            self.layer_1 = nn.Linear(self.feature_dim, hidden_dim,bias=True)
            self.layer_2 = nn.Linear(hidden_dim, class_num,bias=True)
        elif self.nn_type == 'gcn':
            self.layer_1 = tg.nn.GCNConv(self.feature_dim,hidden_dim)
            self.layer_2 = tg.nn.GCNConv(hidden_dim,class_num)
        
        self.U,self.S, self.base = reduction_transform( self.X,self.n_component)
        #print('U',self.U)
        self.costMatrix = self.compute_cost(cost_type)
        #print(self.costMatrix)
        
        self.x = self.get_input(lambda_)
        
    def impute_feature(self,features,imputer_type):
        if imputer_type == 'mean':
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif imputer_type == 'zero':
            imp = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=0)
        init_x = imp.fit_transform(features.numpy())
        x = torch.from_numpy(init_x).float().cuda()
        return x
    
    def compute_cost(self,cost_type):
        #costMatrix = normalize_features(Wcostmatrix(self.base.T,self.base.T))   
        costMatrix = torch.zeros(self.n_component,self.n_component).float().cuda()
        for i in range(self.n_component):
            for j in range(self.n_component):
                if i != j:
                    costMatrix[i][j] = (self.S[i]*self.S[j])**-0.5
        if cost_type == 'max':
            costMatrix = costMatrix /costMatrix.max()
        elif cost_type == 'normalize':
            costMatrix = normalize_features(costMatrix)
        return costMatrix
        
    def get_input(self,lambda_):
        input_X = normalize_features(torch.exp(self.U))
        #print('input',input_X) 
        x1 =self.wgc1(input_X,self.costMatrix)
        #print('x1',x1)
        x2 = self.wgc2(x1 ,self.costMatrix)
        #print('x2',x2)
        
        #x2 = Gram_Schmidt_ortho((x2.T-means).T)
        x2 = Gram_Schmidt_ortho(torch.log(x2))
        #print('G2',x2)
        #original_U = self.U
        original_U = transform(self.U)
        #print('original_U',original_U)
        x = inverse_transform((1-lambda_)*x2+lambda_*original_U,self.S,self.base)
        #self.x = self.x-torch.mean(self.x,0)
        return x
        
    def forward(self,nodes):
        #x = normalize_features(x)
        if self.nn_type == 'mlp':
            x = F.relu(self.layer_1(self.x))     
            x = F.dropout(x, self.drop_out, training=self.training)
            x = self.layer_2(x)
        elif self.nn_type == 'gcn':
            x = F.relu(self.layer_1(self.x,self.edge_index))     
            x = F.dropout(x, self.drop_out, training=self.training)
            x = self.layer_2(x,self.edge_index)            
        x = F.log_softmax(x, dim=1)
        return x

    
def model (opt,data,hidden_dim,class_num,missing_type,n_component,rate,lr,weight_decay,epochs =2000):
    #train_index,val_index,test_index = split_data(data)
    
    data_size = data.features.shape[0]
    node_index = list(range(data_size))
    model = WGCN(data,rate,missing_type,hidden_dim,class_num,n_component).cuda()
    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(),lr= lr,weight_decay= weight_decay)
    elif opt == 'SGD':
        optimizer = optim.SGD(model.parameters(),lr= lr,weight_decay= weight_decay)
    nodes = range(len(data.labels))
    data.labels = data.labels.cuda()
    
    def train():
        model.train()
        optimizer.zero_grad()
        train_out = model(data.train_mask)[data.train_mask]
        loss_train = F.nll_loss(train_out,data.labels[data.train_mask])
        #print('train_loss',loss_train.item())
        acc_train = accuracy(train_out, data.labels[data.train_mask])
        #print('acc_train',acc_train.item())
        loss_train.backward()
        optimizer.step()
        

    def test():
        model.eval()
        val_out = model(data.val_mask)[data.val_mask]
        test_out = model(data.test_mask)[data.test_mask]
        acc_test = accuracy(test_out,data.labels[data.test_mask])
        acc_val = accuracy(val_out, data.labels[data.val_mask])
        #print('acc_val:%s,acc_test:%s'%(acc_val.item(),acc_test.item()))
        return acc_val.item(), acc_test.item()

    #print("start training----")
    acc = []
    best_dict = {
        "val_acc":0,
        "test_acc":0,
        "epoch":-1
    }
    patience = 100
    patience_counter = 0
    for i in range(epochs):
        if patience_counter >= patience: 
            break
        train()
        val_acc, test_acc = test()
        #print('-------------------------%d epoch finishing -------------------------------'%(i))
        if val_acc < best_dict["val_acc"]:
            patience_counter = patience_counter + 1
        else:
            best_dict["val_acc"] = val_acc
            best_dict["test_acc"] = test_acc
            best_dict["epoch"] = i
            patience_counter = 0
    #print("finish----------")
    print(best_dict)
    return best_dict

