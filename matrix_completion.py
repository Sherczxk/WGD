import os,sys,inspect
import os
import joblib
import numpy as np
import h5py
import scipy.sparse.linalg as la
import scipy.sparse as sp
import scipy
import time
from utils import *

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load_matlab_file(path_file, name_field):
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir   = np.asarray(ds['ir'])
            jc   = np.asarray(ds['jc'])
            out  = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out

def load_matrix_data(dataset_str):
    if dataset_str == 'synthetic_netflix':
        path_dataset = './dataset/synthetic_netflix/synthetic_netflix.mat'
        Wrow = load_matlab_file(path_dataset, 'Wrow').todense() #sparse
        Wcol = load_matlab_file(path_dataset, 'Wcol').todense() #sparse
        graph_type = 'multi'
        
    elif dataset_str == 'douban':
        path_dataset = './dataset/douban/training_test_dataset.mat'
        Wrow = load_matlab_file(path_dataset, 'W_users') #dense
        Wcol = None
        graph_type = 'row'
        
    elif dataset_str == 'flixster':
        path_dataset = './dataset/flixster/training_test_dataset_10_NNs.mat'
        Wrow = load_matlab_file(path_dataset, 'W_users') # dense
        Wcol = load_matlab_file(path_dataset, 'W_movies') # dense
        graph_type = 'multi'
        
    elif dataset_str == 'movielens':
        path_dataset = './dataset/movielens/split_1.mat'
        Wrow = load_matlab_file(path_dataset, 'W_users').todense() #sparse
        Wcol = load_matlab_file(path_dataset, 'W_movies').todense() #sparse
        graph_type = 'multi'
        
    elif dataset_str == 'yahoo_music':
        path_dataset = './dataset/yahoo_music/training_test_dataset_10_NNs.mat' 
        Wcol = load_matlab_file(path_dataset, 'W_tracks') #dense
        Wrow = None
        graph_type = 'col'
    
    M = load_matlab_file(path_dataset, 'M')
    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest') 
    
    #np.random.seed(0)
    pos_tr_samples = np.where(Otraining)
    num_tr_samples = len(pos_tr_samples[0])
    list_idx = list(range(num_tr_samples))
    np.random.shuffle(list_idx)
    idx_data = list_idx[:num_tr_samples//2]
    idx_train = list_idx[num_tr_samples//2:]

    pos_data_samples = (pos_tr_samples[0][idx_data], pos_tr_samples[1][idx_data])
    pos_tr_samples = (pos_tr_samples[0][idx_train], pos_tr_samples[1][idx_train])

    Odata = np.zeros(M.shape)
    Otraining = np.zeros(M.shape)

    for k in range(len(pos_data_samples[0])):
        Odata[pos_data_samples[0][k], pos_data_samples[1][k]] = 1

    for k in range(len(pos_tr_samples[0])):
        Otraining[pos_tr_samples[0][k], pos_tr_samples[1][k]] = 1

    #print ('Num data samples: %d'% (np.sum(Odata)))
    #print ('Num train samples: %d'% (np.sum(Otraining)))
    #print ('Num train+data samples: %d'% (np.sum(Odata+Otraining)))
    #print('M',M.shape)
    M = torch.from_numpy(M).cuda()
    Odata = torch.from_numpy(Odata).float().cuda()
    Otraining = torch.from_numpy(Otraining).float().cuda()
    Otest = torch.from_numpy(Otest).float().cuda()
    if graph_type != 'col':
        Wrow = torch.from_numpy(Wrow).cuda()
    if graph_type != 'row':
        Wcol = torch.from_numpy(Wcol).cuda()
    return M,Odata,Otraining,Otest,Wrow,Wcol,graph_type

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

def adj_matrix_to_list(adj):
    adj_list = {}
    degree = {}
    node_idx = torch.Tensor(list(range(adj.shape[0])))
    for i in range(adj.shape[0]):
        adj[i,i] = 1
        neigh = adj[i].bool()
        adj_list[i] = node_idx[neigh].numpy()
        degree[i] = adj_list[i].shape[0]
    return adj_list,degree

def Barycenter_update(input_gmms,adj_list,degree,bary_weight, costMatrix):
    n = input_gmms.shape[0]
    output_gmm = list()
    for j in range(n): 
        neighs_list = adj_list[j]
        if bary_weight == 'laplace':
            weight = torch.FloatTensor([(degree[j]*degree[k])**(-0.5) for k in neighs_list]).cuda()
            bary_weights = weight/torch.sum(weight)
        elif bary_weight == 'uniform':
            bary_weights = 1./degree[j]*torch.ones(degree[j]).cuda()
        input_gmm = input_gmms[neighs_list]
        
        bary_weights = bary_weights.unsqueeze(0)
        bary = barycenter.compute(input_gmm.T,costMatrix,bary_weights)

        output_gmm.append(bary)
    output_gmm = torch.stack(output_gmm)
    return output_gmm


class Bary_layer(torch.nn.Module):
    def __init__(self,adj_list,degree):
        super(Bary_layer, self).__init__()
        
        self.adj_list = adj_list
        self.degree = degree
    
    def forward(self,U,costMatrix,ortho=True):
        bary_weight = 'uniform'
        U_bar = torch.exp(U)
        if ortho:
            U_bar = normalize_features(U_bar)
        U_bar = Barycenter_update(U_bar,self.adj_list,self.degree,bary_weight, costMatrix) 
        #U_bar = Barycenter_update(U_bar,self.adj_list,self.degree,bary_weight, costMatrix) 
        U_bar = torch.log(U_bar)
        if ortho:
            U_bar = Gram_Schmidt_ortho(U_bar)
        return U_bar
        
class WGCN(torch.nn.Module):
    def __init__(self,features,r_adj,c_adj,graph_type,n_component,hidden_dim,drop_out, num_bary,layer_num,bias=True,**kwargs):
        super(WGCN, self).__init__()
        
        cost_type= None#'max','normalize'
        
        self.n_component = n_component
        self.hidden_dim = hidden_dim
        self.drop_out = drop_out
        self.layer_num=layer_num
        
        n_bary  = num_bary+1
        if graph_type != 'col':
            self.U_adj_list,self.U_degree =  adj_matrix_to_list(r_adj)
            self.u_bary = Bary_layer(self.U_adj_list,self.U_degree)
            
            self.u_mlp = nn.Linear(n_bary*self.n_component, 4*self.hidden_dim,True)
            self.u_hidden = nn.ModuleList([nn.Linear(4*self.hidden_dim, 4*self.hidden_dim,True) for i in range(self.layer_num-2)])
            self.u_mlp_out = nn.Linear(4*self.hidden_dim, self.n_component,True)
        
        if graph_type != 'row':
            self.V_adj_list,self.V_degree =  adj_matrix_to_list(c_adj)     
            self.v_bary = Bary_layer(self.V_adj_list,self.V_degree)   
            self.v_mlp = nn.Linear(n_bary*self.n_component, 4*self.hidden_dim,True)
            self.v_hidden = nn.ModuleList([nn.Linear(4*self.hidden_dim, 4*self.hidden_dim,True) for i in range(self.layer_num-2)])
            self.v_mlp_out = nn.Linear(4*self.hidden_dim, self.n_component,True)

        self.mlp = nn.Linear(features.shape[1],features.shape[1],True)
        self.U,self.S, self.V = reduction_transform( features,self.n_component)
        #print('U',self.U)
        self.costMatrix = self.compute_cost(cost_type)
        #print(self.costMatrix)
        
        self.U_tilde,self.V_tilde = self.get_input(self.U,self.V,graph_type,num_bary)
        self.graph_type = graph_type
        
    
    def compute_cost(self,cost_type):
        costMatrix = torch.zeros(self.n_component,self.n_component).float().cuda()
        for i in range(self.n_component):
            for j in range(self.n_component):
                if i != j:
                    costMatrix[i][j] = abs(self.S[i]**2-self.S[j]**2)
        if cost_type == 'max':
            costMatrix = costMatrix /costMatrix.max()
        elif cost_type == 'normalize':
            costMatrix = normalize_features(costMatrix)   
        return costMatrix
    def get_input(self,U,V,graph_type,num_bary):
        if graph_type != 'col':
            U_list = list()
            for i in range(num_bary):
                U_list.append(U)
                bary_U = self.u_bary(U,self.costMatrix,True)
                U = bary_U
            U_list.append(U)
            U_tilde = torch.cat(U_list,1)
        else:
            U_tilde = U
        if graph_type != 'row':  
            V_list = list()
            for i in range(num_bary):
                V_list.append(V)
                bary_V = self.v_bary(V,self.costMatrix,True)
                V = bary_V
            V_list.append(V)
            V_tilde = torch.cat(V_list,1)
        else:
            V_tilde = V
        return U_tilde,V_tilde
        
    def forward(self):
        lambda_ = 0.5
        if self.graph_type != 'col':
            U = F.relu(self.u_mlp(self.U_tilde))
            if self.drop_out:
                U = F.dropout(U,self.drop_out,  training=self.training)
            for i in range(self.layer_num-2):
                U = F.relu(self.u_hidden[i](U))
                if self.drop_out:
                    U = F.dropout(U,self.drop_out,  training=self.training)
            U = self.u_mlp_out(U)

            U_out,us = L2_normalization(U)
        else:
            U_out = self.U
            us = torch.ones(self.U.shape[1]).cuda()
        if self.graph_type != 'row':
            V = F.relu(self.v_mlp(self.V_tilde))
            if self.drop_out:
                V = F.dropout(V,self.drop_out,  training=self.training)
            for i in range(self.layer_num-2):
                V = F.relu(self.v_hidden[i](V))
                if self.drop_out:
                    V = F.dropout(V,self.drop_out,  training=self.training)
            V  = self.v_mlp_out(V)

            V_out,vs = L2_normalization(V)
        else:
            V_out = self.V
            vs = torch.ones(self.V.shape[1]).cuda()
        reconstruct_x = inverse_transform(U_out,self.S,V_out)
        return reconstruct_x

    
def model (dataset,n_component,hidden_dim,num_bary,drop_out=0,layer_num=4,ce=0,lr=0.001,weight_decay=5e-3,total_iter =10000):
    
    
    features,Odata,Otraining,Otest,r_adj,c_adj,graph_type=load_matrix_data(dataset)
    #print('features',features)
    model = WGCN(Odata*features,r_adj,c_adj,graph_type,n_component,hidden_dim,drop_out,num_bary,layer_num).cuda()
    optimizer = optim.Adam(model.parameters(),lr= lr,weight_decay= weight_decay)
    #print(torch.max(features).item(),torch.mean(features).item())
    
    def train():
        model.train()
        optimizer.zero_grad()
        reconstruct_x = model()
        if dataset == 'flixster':
            norm_X = ce+5*(reconstruct_x-torch.min(reconstruct_x))/(torch.max(reconstruct_x-torch.min(reconstruct_x)))
        elif dataset == 'yahoo_music':
            norm_X = ce+100*(reconstruct_x-torch.min(reconstruct_x))/(torch.max(reconstruct_x-torch.min(reconstruct_x)))
        elif dataset == 'movielens':
            norm_X = ce+5*(reconstruct_x-torch.min(reconstruct_x))/(torch.max(reconstruct_x-torch.min(reconstruct_x)))
        elif dataset == 'douban':
            norm_X = ce+4*(reconstruct_x-torch.min(reconstruct_x))/(torch.max(reconstruct_x-torch.min(reconstruct_x)))
        frob_tensor = (Otraining + Odata)*(norm_X - features)
        loss_train = (frobenius_norm(frob_tensor))**2/torch.sum(Otraining+Odata)
        loss_train.backward()
        optimizer.step()
        return loss_train.item()
        

    def test():
        model.eval()
        reconstruct_x = model()
        if dataset == 'flixster':
            norm_X = ce+5*(reconstruct_x-torch.min(reconstruct_x))/(torch.max(reconstruct_x-torch.min(reconstruct_x)))
        elif dataset == 'yahoo_music':
            norm_X = ce+100*(reconstruct_x-torch.min(reconstruct_x))/(torch.max(reconstruct_x-torch.min(reconstruct_x)))
        elif dataset == 'movielens':
            norm_X = ce+5*(reconstruct_x-torch.min(reconstruct_x))/(torch.max(reconstruct_x-torch.min(reconstruct_x)))
        elif dataset == 'douban':
            norm_X = ce+4*(reconstruct_x-torch.min(reconstruct_x))/(torch.max(reconstruct_x-torch.min(reconstruct_x)))
        predictions = Otest*(norm_X - features)
        predictions_error = frobenius_norm(predictions)
        return predictions_error.item()

    iter_test = 10
    loss_train_list = []
    test_error_list = []
    num_iter = 0
    for i in range(total_iter):
        loss_train = train()
        test_error = test()
        loss_train_list.append(loss_train)
        if (np.mod(num_iter, iter_test)==0):
            test_error_list.append(test_error)
        num_iter += 1
    best_iter = (np.where(np.asarray(loss_train_list)==np.min(loss_train_list))[0][0]//iter_test)*iter_test
    best_pred_error = test_error_list[best_iter//iter_test]
    RMSE = ((best_pred_error)**2/np.sum(Otest.cpu().numpy()))**0.5
    print(' RMSE:',RMSE)
    return RMSE


multi_graph = ['flixster']
n_components = [25]
num_barys = [5]
hidden_dims= [50]
rmse = {}
for dataset in multi_graph:
    print(dataset)
    results = []
    for num_bary in num_barys:
        for n_component in n_components:
            for hidden_dim in hidden_dims:
                result = model (dataset,n_component,hidden_dim,num_bary) 
                results.append(result) 
                print('num_bary:%d,n_component:%d,hidden_dim:%d,result:%f'%(num_bary,n_component,hidden_dim,result))
            rmse[n_component] = results
print(rmse)