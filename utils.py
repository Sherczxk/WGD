import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
import ot
import pickle
from more_itertools import locate
import barycenter
from sklearn.impute import SimpleImputer

def adjlist(edge_index):
    edge_index = edge_index.cpu().numpy()
    adjlist = defaultdict(set)
    degree = defaultdict(set)
    for i in range(edge_index.shape[1]):
        to_nodes_0 = list(locate(edge_index[0], lambda a: a == i))
        for j in to_nodes_0:
            adjlist[i].add(edge_index[1][j])
        to_nodes_1 = list(locate(edge_index[1], lambda a: a == i))
        for j in to_nodes_1:
            adjlist[i].add(edge_index[0][j])
        degree[i] = len(adjlist[i])+1
    return adjlist,degree

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def pickle_load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def normalize_features(mx):
    """Row-normalize matrix"""
    row_sum = torch.sum(mx,1)
    row_sum = row_sum**-1
    row_sum[torch.isinf(row_sum)] = 0.
    r_sum = torch.diag(row_sum)
    x = torch.matmul(r_sum,mx)
    return x


def reduction_transform(features,n_components):
    '''
    input: features tensor [n,m]
    return: transform_features <features,V> tensor [n,n_components]; V tensor [m,n_components]
    '''
    U_,S_,V_ = torch.pca_lowrank(features,q=n_components)
    #means = torch.mean(features,0)
    #U_,S_,V_ = torch.svd_lowrank(features-means,q=n_components)
    '''
    U,S,V = torch.svd(features,some=False)
    U_ = U.T[:n_components].T
    S_ = S[:n_components]
    V_ = V.T[:n_components].T
    '''
    return U_,S_,V_

def inverse_transform(U,S,V):
    temp = torch.matmul(U,torch.diag(S))
    return torch.matmul(temp,V.T)

def missing_feature(features,rate,missing_type='part'):
    if missing_type == 'part':
        mask = torch.rand(features.shape) <= rate
    elif missing_type == 'whole':
        node_mask = torch.rand(size=(features.shape[0], 1))
        mask = (node_mask <= rate).repeat(1, features.shape[1])
        #mask_ = torch.rand(features.shape[0]) <= rate
        #mask = mask_.unsqueeze(1).repeat(1, features.shape[1])
    else:
        raise ValueError("Missing Type %s is not defined"%(missing_type))
    features[mask] = torch.tensor(float('nan'))
    return features


def Gram_Schmidt_ortho(base):
    d,m = base.shape
    for k in range(min(d,m)):
        for j in range(k):
            tmp = torch.matmul(base[:,k], base[:,j])
            base[:,k] = base[:,k] - tmp*base[:,j]
        base[:,k] = base[:,k]/torch.sqrt(torch.sum(base[:,k]**2))
    return base

def transform(U):
    U = torch.exp(U)
    U = normalize_features(U)
    U = torch.log(U)
    U = Gram_Schmidt_ortho(U)
    return U