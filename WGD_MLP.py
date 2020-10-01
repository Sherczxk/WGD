import pickle
import math
import torch
import torch.nn as nn 
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric as tg
import torch.autograd as autograd
from torch.nn.modules.module import Module
from torch.autograd import Variable
import numpy as np
import random
import time
from collections import defaultdict
from utils import *
from copy import deepcopy
from dataloader import *
import barycenter


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
        input_gmm = input_gmms[neighs_list]
        
        bary_weights = bary_weights.unsqueeze(0)
        bary = barycenter.compute(input_gmm.T,costMatrix,bary_weights)

        output_gmm.append(bary)
    output_gmm = torch.stack(output_gmm)
    return output_gmm


class WGCN_layer(torch.nn.Module):
    def __init__(self,adj_list,degree,h):
        super(WGCN_layer, self).__init__()
        
        self.adj_list = adj_list
        self.degree = degree
        self.h = h
        
    def forward(self,trans_X,costMatrix):
        bary_weight = 'uniform'
        for i in range(self.h):
            trans_X = Barycenter_update(trans_X,self.adj_list,self.degree,bary_weight, costMatrix) 
        return trans_X
        
class WGCN(torch.nn.Module):
    def __init__(self,data,rate,missing_type,hidden_dim,class_num,n_component,h_hop,mlp_layer,lambda_, bias=True,**kwargs):
        super(WGCN, self).__init__()
        
        imputer_type = 'mean'
        cost_type= None#'max','normalize'
        self.nn_type = 'mlp'
        self.drop_out = 0.5
        self.mlp_layer = mlp_layer
        
        features = deepcopy(data.features)
        features,_ = missing_feature(features,rate,missing_type)
        
        self.X = self.impute_feature(features,imputer_type)
        
        self.n_component = n_component
        self.n_node, self.feature_dim = data.features.shape
        self.adj_list,self.degree =  adjlist(data.edge_index)
        self.edge_index = data.edge_index.cuda()       
        
        self.wgc = WGCN_layer(self.adj_list,self.degree,h_hop)
        
        if self.nn_type == 'mlp':
            self.mlp_in = nn.Linear(self.feature_dim, 4*hidden_dim,True)
            self.mlp_hidden = nn.ModuleList([nn.Linear(4*hidden_dim, 4*hidden_dim,True) for i in range(self.mlp_layer-2)])
            self.mlp_out = nn.Linear(4*hidden_dim, class_num,True)
        elif self.nn_type == 'gcn':
            self.layer_1 = tg.nn.GCNConv(self.feature_dim,hidden_dim)
            self.layer_2 = tg.nn.GCNConv(hidden_dim,class_num)
        
        self.U,self.S, self.base = reduction_transform( self.X,self.n_component)
        self.costMatrix = self.compute_cost(cost_type)
        
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
        costMatrix = torch.zeros(self.n_component,self.n_component).float().cuda()
        for i in range(self.n_component):
            for j in range(self.n_component):
                if i != j:
                    costMatrix[i][j] = abs(self.S[i]-self.S[j])
        if cost_type == 'max':
            costMatrix = costMatrix /costMatrix.max()
        elif cost_type == 'normalize':
            costMatrix = normalize_features(costMatrix)   
        return costMatrix
    def get_input(self,lambda_):
        x = self.U
        u= torch.zeros(x.shape).cuda()
        for i in range(lambda_):
            x = normalize_features(torch.exp(x))
            x =self.wgc(x,self.costMatrix)
            x = Gram_Schmidt_ortho(torch.log(x))
            u += x
        
        u = u/lambda_
        x = inverse_transform(u,self.S,self.base)
        return x
        
    def forward(self,nodes):
        if self.nn_type == 'mlp':
            x = F.relu(self.mlp_in(self.x)) 
            x = F.dropout(x, self.drop_out, training=self.training)
            for i in range(self.mlp_layer-2):
                x = F.relu(self.mlp_hidden[i](x)) 
                x = F.dropout(x, self.drop_out, training=self.training)
            x = self.mlp_out(x)
        elif self.nn_type == 'gcn':
            x = F.relu(self.layer_1(self.x,self.edge_index))     
            x = F.dropout(x, self.drop_out, training=self.training)
            x = self.layer_2(x,self.edge_index)            
        x = F.log_softmax(x, dim=1)
        return x

def model (opt,data,hidden_dim,class_num,missing_type,n_component,h_hop,mlp_layer,lambda_,rate,lr,weight_decay,epochs =2000):
    
    data_size = data.features.shape[0]
    node_index = list(range(data_size))
    model = WGCN(data,rate,missing_type,hidden_dim,class_num,n_component,h_hop,mlp_layer,lambda_).cuda()
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
        print('acc_train',acc_train.item())
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
    return best_dict["test_acc"]

#'''
datasets = ['cora', 'citeseer','pubmed']

root = './data'
dataset ="pubmed"
data = load_data(dataset)

n_component= 64
hidden_dim = 32
lr = 0.01
#weight_decays = [0.05,0.045,0.04,0.035,0.03,0.02,0.01,0.005,0.001]
weight_decays = [7e-3,7e-3,4e-3,3e-3,3e-3,1e-3,7e-4,5e-4,0]
#weight_decays = [2.00E-02,2.00E-02,1.00E-02,8.00E-03,8.00E-03,5.00E-03,3.00E-03,1.00E-03,8.00E-04]
#weight_decays = [0.01,0.01,8.00E-03,5.00E-03,4.00E-03,2.00E-03,2.00E-03,1.00E-03,8.00E-04]
class_num = int(torch.max(data.labels)) + 1
opt = 'Adam'
missing_type = 'part'
rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
lambda_ = 7 
h_hop = 2
mlp_layer = 2

epochs =2000
seeds = [757953235, 892232741, 1526661, 133569720, 1033848620,\
         314466866, 57396115, 17626403, 515193964, 189336409]

results = list()
print('model:WGCN,data:%s, missing_type:%s'%(dataset,missing_type))

for i,rate in enumerate(rates):
    weight_decay = weight_decays[i]
    print('data:%s, missing_type:%s, rate:%s, weight_decay:%f'%(dataset,missing_type,rate,weight_decay))
    acc = list()
    for seed in seeds:
        set_seed(seed)
        t = model (opt,data,hidden_dim,class_num,missing_type,n_component,h_hop,mlp_layer,lambda_,rate,lr,weight_decay,epochs)
        acc.append(t)
        print(acc)
    print('mean:%f std:%f' %(np.mean(acc), np.std(acc)))
    results.append(np.mean(acc))
print('data:%s, missing_type:%s'%(dataset,missing_type))
print(results)

