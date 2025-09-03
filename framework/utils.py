import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
# import ot
import pickle
import barycenter
from torch_scatter import scatter_add
from sklearn.impute import SimpleImputer

def adjlist(edge_index):
    edge_index = edge_index.cpu().numpy()
    adjlist = defaultdict(set)
    degree = defaultdict(set)
    k=0
    for n in range(edge_index.shape[1]):
        for idx in range(k,edge_index.shape[1]):
            if edge_index[0][idx] == n:
                adjlist[n].add(edge_index[1][idx])
                adjlist[edge_index[1][idx]].add(n)
            else:
                k = idx
                break
        degree[n] = len(adjlist[n])+1
    return adjlist,degree


def get_degree_adj_list(edge_index, n_nodes):
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    adj_list = [[] for _ in range(n_nodes)]
    for i in range(len(row)):
        adj_list[row[i]] += [col[i].item()]
    # adj_list = [torch.LongTensor(adj).to(edge_index.device) for adj in adj_list]
    return deg.long(), adj_list

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def pickle_load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def normalize_features(mx):
    """Row-normalize matrix"""
    row_sum = torch.sum(mx,1)
    row_sum = row_sum**-1
    row_sum[torch.isinf(row_sum)] = 0.
    # r_sum = torch.diag(row_sum)
    # x = torch.matmul(r_sum,mx)
    x = (mx.T*row_sum).T
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
        node_mask = torch.rand(size=(features.shape[0], 1)).cuda()
        mask = (node_mask <= rate).repeat(1, features.shape[1])
        #mask_ = torch.rand(features.shape[0]) <= rate
        #mask = mask_.unsqueeze(1).repeat(1, features.shape[1])
    elif missing_type == 'mixed':
        whole_rate, partial_rate = rate
        rand = torch.rand(features.shape[0]) 
        partial_index = list()
        for i,k in enumerate(rand):
            if k >whole_rate:
                partial_index.append(i)
        mask = torch.ones(features.shape).bool().cuda()
        mask_ = torch.rand(len(partial_index),features.shape[1]) < partial_rate 
        for i,idx in enumerate(partial_index):
            mask[idx] = mask_[i]
    else:
        raise ValueError("Missing Type %s is not defined"%(missing_type))
    features[mask] = torch.tensor(float('nan')).cuda()
    M = torch.ones(features.shape).cuda()
    M[mask] = 0
    if torch.isnan(features).all():
        raise ValueError("All the feature is missing,please try another random seed.")
    else:
        return features,M.bool()


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
    #U,s = L2_normalization(U)
    return U#,s

def rand_projections(heads,feature_dim,hidden_dim,ortho=True):
    """This function generates `num_samples` random samples from the latent space's unit sphere.
        Args:
            d - embedding_dim (int): embedding dimensionality
            m - num_samples (int): number of random projection samples
        Return:
            torch.Tensor: tensor of size (d,m)
    """
    '''
    projections = [w / np.sqrt((w**2).sum(0))  # L2 normalization
                   for w in np.random.normal(size=(d, m))]
    projections = np.asarray(projections)   #(d,m)
    projs = torch.from_numpy(projections).float().cuda()
    '''
    #projs = []
    inverse_projs = []
    proj = torch.randn(heads,feature_dim,hidden_dim).cuda()
    for i in range(heads):
        col_sum = torch.sum(proj[i]**2,0)**-0.5
        col_sum[torch.isinf(col_sum)] = 0
        col_sum = torch.diag(col_sum)
        proj[i] = torch.matmul(proj[i],col_sum)
        if ortho:
            proj[i] = Gram_Schmidt_ortho(proj[i])
        #projs.append(proj) 
        inverse_projs.append(proj[i].T)
        #inverse_projs.append(torch.pinverse(proj[i]))
    return proj, torch.stack(inverse_projs).cuda()

def L2_normalization(X):
    s = torch.sum(X**2,0)
    col_sum = s**-0.5
    col_sum[torch.isinf(col_sum)] = 0
    col_sum = torch.diag(col_sum)
    X = torch.matmul(X,col_sum)
    return X,s

def frobenius_norm(tensor):
    tensor_sum = torch.sum(tensor**2)**0.5
    return tensor_sum

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD