import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import torch
import os
from torch_geometric.utils import to_undirected, add_remaining_self_loops
import torch_geometric.transforms as transforms
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB, WikipediaNetwork, Flickr
import torch_geometric.transforms as T
from torch_sparse import coalesce
from utils import get_degree_adj_list



label_rate = {'Cora':0.052,'Citeseer':0.036,'Pubmed':0.003,'Computers':0.015,'Photo':0.021}
DATA_PATH = os.path.expanduser("data/")
development_seed = 1684992425

class Data(object):
    def __init__(self, edge_index, features, labels, train_mask, val_mask, test_mask):
        #self.adj = adj
        self.edge_index = edge_index
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_features = features.size(1)
        self.num_classes = int(torch.max(labels)) + 1

    def to(self, device):
        #self.adj = self.adj.to(device)
        self.edge_index = self.edge_index.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)



def load_data(name, seed, root=None, self_loop=True, undirected=True):
    if root is None:
        root = DATA_PATH
    evaluator = None
    split_idx = None
    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root, name)
    elif name in ["OGBN-Arxiv", "OGBN-Products"]:
        dataset = PygNodePropPredDataset(name=name.lower(), transform=transforms.ToSparseTensor(), root=DATA_PATH)
        evaluator = Evaluator(name=name.lower())
        split_idx = dataset.get_idx_split()
    elif name in ["Photo", "Computers"]:
        dataset = Amazon(root, name)
    elif name in ["Cornell", "Texas", "Wisconsin"]:
        dataset = WebKB(root, name)
    elif name in ["chameleon", "crocodile", "squirrel"]:
        dataset = WikipediaNetwork(root, name)
    elif name in ["flickr"]:
        dataset = Flickr(root+'flickr',transform=transforms.ToSparseTensor())
    else:
        raise Exception("Unknown dataset.")
   
    data = set_train_val_test_split(seed, dataset.data, name, split_idx)

    # Make graph undirected so that we have edges for both directions and add self loops
    if undirected:
        data.edge_index = to_undirected(data.edge_index)
    if self_loop:
        data.edge_index, _ = add_remaining_self_loops(data.edge_index, num_nodes=data.x.shape[0])

    degree,adj_list =  get_degree_adj_list(data.edge_index,data.x.shape[0])
    data.degree = degree
    data.adj_list = adj_list
    return data, evaluator


def set_train_val_test_split(seed: int, data: Data, dataset_name: str, split_idx: int = None) -> Data:

    if dataset_name in [
        "Cora",
        "CiteSeer",
        "PubMed",
        "Photo",
        "Computers",
        "CoauthorCS",
        "CoauthorPhysics",
    ]:
        # Use split from "Diffusion Improves Graph Learning" paper, which selects 20 nodes for each class to be in the training set
        num_val = 500 
        data = set_per_class_train_val_test_split(
            seed=seed, data=data, num_val=num_val, num_train_per_class=20, 
        )
    elif dataset_name in ["OGBN-Arxiv", "OGBN-Products"]:
        # OGBN datasets have pre-assigned split
        data.train_mask = split_idx["train"]
        data.val_mask = split_idx["valid"]
        data.test_mask = split_idx["test"]
    elif dataset_name in ["Cornell", "Texas", "Wisconsin", "chameleon", "crocodile", "squirrel"]:
        data = set_ratio_train_val_test_split(data,train_ratio=0.6, val_ratio=0.2, Flag=0)
    elif dataset_name in ["flickr"]:
        data = set_ratio_per_class_train_val_test_split(data, train_ratio=0.5, val_ratio=0.25)
    else:
        raise ValueError(f"We don't know how to split the data for {dataset_name}")

    return data



def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def split_data(labels, n_train_per_class, n_val, seed):
    np.random.seed(seed)
    n_class = int(torch.max(labels)) + 1
    train_idx = np.array([], dtype=np.int64)
    remains = np.array([], dtype=np.int64)
    for c in range(n_class):
        candidate = torch.nonzero(labels == c).T.numpy()[0]
        np.random.shuffle(candidate)
        train_idx = np.concatenate([train_idx, candidate[:n_train_per_class]])
        remains = np.concatenate([remains, candidate[n_train_per_class:]])
    np.random.shuffle(remains)
    val_idx = remains[:n_val]
    test_idx = remains[n_val:]
    train_mask = index_to_mask(train_idx, labels.size(0))
    val_mask = index_to_mask(val_idx, labels.size(0))
    test_mask = index_to_mask(test_idx, labels.size(0))
    return train_mask, val_mask, test_mask

def set_per_class_train_val_test_split(
    seed: int, data: Data, num_val: int = 500, num_train_per_class: int = 20,
) -> Data:
    rnd_state = np.random.RandomState(development_seed)
    num_nodes = data.y.shape[0]
    val_idx = rnd_state.choice(num_nodes, num_val, replace=False)
    development_idx = [i for i in np.arange(num_nodes) if i not in val_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(int(data.y.max() + 1)):
        class_idx = np.array(development_idx)[np.where(data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, num_train_per_class, replace=False))

    test_idx = [i for i in development_idx if i not in train_idx]

    data.train_mask = index_to_mask(train_idx, num_nodes)
    data.val_mask = index_to_mask(val_idx, num_nodes)
    data.test_mask = index_to_mask(test_idx, num_nodes)
    return data



def set_ratio_train_val_test_split(data,train_ratio=0.6, val_ratio=0.2, Flag=0):
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing
    nclass  =int(data.y.max() + 1)
    num_nodes = data.y.shape[0]
    percls_trn = int(round(train_ratio*num_nodes/nclass))
    val_lb = int(round(val_ratio*num_nodes))
    indices = []
    for i in range(nclass):
        index = (data.y == i).nonzero(as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=num_nodes)
        data.val_mask = index_to_mask(val_index, size=num_nodes)
        data.test_mask = index_to_mask(rest_index, size=num_nodes)
    return data
#'''



def set_ratio_per_class_train_val_test_split(data,train_ratio=0.6, val_ratio=0.2, Flag=0):
    # * sum of round(train_rate*len(per_classes)) labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing
    nclass  =int(data.y.max() + 1)
    num_nodes = data.y.shape[0]
    # percls_trn = int(round(train_ratio*num_nodes/nclass))
    # val_lb = int(round(val_ratio*num_nodes))
    # indices = []
    train_index, val_index, test_index = [], [], []
    for i in range(nclass):
        index = (data.y == i).nonzero(as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0))]
        # indices.append(index)
        
        # rnd_state = np.random.RandomState(seed)
        train_num = int(round(len(index)*train_ratio))
        val_num = int(round(len(index)*val_ratio))
        train_index.extend(index[:train_num])
        val_index.extend(index[train_num:train_num+val_num])
        test_index.extend(index[train_num+val_num:])

    data.train_mask = index_to_mask(train_index, size=num_nodes)
    data.val_mask = index_to_mask(val_index, size=num_nodes)
    data.test_mask = index_to_mask(test_index, size=num_nodes)
    return data
#'''

def get_degree(edge_list):
    row, col = edge_list
    deg = torch.bincount(row)
    return deg


def normalize_adj(edge_list):
    deg = get_degree(edge_list).cuda()
    row, col = edge_list
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    weight = torch.ones(edge_list.size(1)).cuda()
    v = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj