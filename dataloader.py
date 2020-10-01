import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import torch
import os
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
import torch_geometric.transforms as T



label_rate = {'Cora':0.052,'Citeseer':0.036,'Pubmed':0.003,'Computers':0.015,'Photo':0.021}
path = os.path.expanduser("./data/")

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


def load_data(dataset_str, ntrain=None, seed=None):
    if dataset_str in ['cora', 'citeseer','pubmed']:
        data = load_planetoid_data(dataset_str)
    elif dataset_str == 'Pubmed':
        data = get_pubmed(path,dataset_str)
    elif dataset_str in ['amaphoto', 'amacomp']:
        data = load_amazon_data(dataset_str, ntrain, seed)
    elif dataset_str in ['arxiv', 'proteins']:
        data = load_ogb_data(dataset_str, ntrain, seed)
    else:
        raise ValueError("Dataset {0} does not exist".format(dataset_str))
    return data

def load_ogb_data(dataset_str,ntrain,seed):
    if dataset_str == 'arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        data = dataset[0]
        edge_index = to_undirected(data.edge_index, data.num_nodes)
        features = data.x
        labels = data.y
    '''
    elif dataset_str == 'proteins':
        dataset = PygNodePropPredDataset(name='ogbn-proteins',transform=T.ToSparseTensor())
        data = dataset[0]
        # Move edge features to node features.
        features = data.adj_t.mean(dim=1)
        data.adj_t.set_value_(None)

        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train'].to(device)
    '''
    split_idx = dataset.get_idx_split()
    train_mask = split_idx['train']
    val_mask = split_idx['valid']
    test_mask = split_idx['test']  
    
    data = Data(edge_index, features, labels, train_mask, val_mask, test_mask)
    return data
        
def load_amazon_data(dataset_str, ntrain, seed):
    with np.load('data/amazon/' + dataset_str + '.npz', allow_pickle=True) as loader:
        loader = dict(loader)
    adj_mat = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                            shape=loader['adj_shape']).tocoo()
    edge_list1 = torch.cat((torch.tensor(adj_mat.row).type(torch.int64).view(1, -1),
                            torch.tensor(adj_mat.col).type(torch.int64).view(1, -1)), dim=0)
    edge_list2 = torch.cat((torch.tensor(adj_mat.col).type(torch.int64).view(1, -1),
                            torch.tensor(adj_mat.row).type(torch.int64).view(1, -1)), dim=0)
    edge_index = torch.cat([edge_list1, edge_list2], dim=1)

    #edge_list = add_self_loops(edge_list, loader['adj_shape'][0])
    #adj = normalize_adj(edge_index)
    if 'attr_data' in loader:
        feature_mat = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                    shape=loader['attr_shape']).todense()
    elif 'attr_matrix' in loader:
        feature_mat = loader['attr_matrix']
    else:
        feature_mat = None
    features = torch.tensor(feature_mat)

    if 'labels_data' in loader:
        labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                               shape=loader['labels_shape']).todense()
    elif 'labels' in loader:
        labels = loader['labels']
    else:
        labels = None
    labels = torch.tensor(labels).long()

    # load fixed train nodes
    if ntrain is None:
        with np.load('data/amazon/' + dataset_str + '_mask.npz', allow_pickle=True) as masks:
            train_idx, val_idx, test_idx = masks['train_idx'], masks['val_idx'], masks['test_idx']
            train_mask = index_to_mask(train_idx, labels.size(0))
        val_mask = index_to_mask(val_idx, labels.size(0))
        test_mask = index_to_mask(test_idx, labels.size(0))
    else:
        train_mask, val_mask, test_mask = split_data(labels, ntrain, 500, seed)

    data = Data(edge_index, features, labels, train_mask, val_mask, test_mask)

    return data


def load_planetoid_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open("data/planetoid/ind.{}.{}".format(dataset_str, name), 'rb') as f:
            if sys.version_info > (3, 0):
                out = pkl.load(f, encoding='latin1')
            else:
                out = objects.append(pkl.load(f))

            if name == 'graph':
                objects.append(out)
            else:
                out = out.todense() if hasattr(out, 'todense') else out
                objects.append(torch.tensor(out))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx = parse_index_file("data/planetoid/ind.{}.test.index".format(dataset_str))
    train_idx = torch.arange(y.size(0), dtype=torch.long)
    val_idx = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    sorted_test_idx = np.sort(test_idx)

    if dataset_str == 'citeseer':
        len_test_idx = max(test_idx) - min(test_idx) + 1
        tx_ext = torch.zeros(len_test_idx, tx.size(1))
        tx_ext[sorted_test_idx - min(test_idx), :] = tx
        ty_ext = torch.zeros(len_test_idx, ty.size(1), dtype=torch.int)
        ty_ext[sorted_test_idx - min(test_idx), :] = ty

        tx, ty = tx_ext, ty_ext

    features = torch.cat([allx, tx], dim=0)
    features[test_idx] = features[sorted_test_idx]

    labels = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    labels[test_idx] = labels[sorted_test_idx]

    edge_index = adj_list_from_dict(graph)
    #edge_list = add_self_loops(edge_list, features.size(0))
    #adj = normalize_adj(edge_index)

    train_mask = index_to_mask(train_idx, labels.shape[0])
    val_mask = index_to_mask(val_idx, labels.shape[0])
    test_mask = index_to_mask(test_idx, labels.shape[0])

    data = Data(edge_index, features, labels, train_mask, val_mask, test_mask)

    return data

def get_pubmed(path,dataset_str):
    label_r = label_rate[dataset_str]
    dataset = Planetoid(path+dataset_str, dataset_str)
    features = dataset.data.x.numpy()
    labels = dataset.data.y
    edge_index = dataset.data.edge_index
    
    train_mask = dataset.data.train_mask
    test_mask = dataset.data.test_mask
    val_mask = dataset.data.val_mask
    nidx_test = []
    nidx_val = []
    nidx_train = []
    for idx, mask in enumerate(train_mask):
        if mask:
            nidx_train.append(idx)
    for idx, mask in enumerate(test_mask):
        if mask:
            nidx_test.append(idx)
    for idx, mask in enumerate(val_mask):
        if mask:
            nidx_val.append(idx)
    
    features = torch.FloatTensor(features).float()
    labels = torch.LongTensor(labels)
    
    data = Data(edge_index, features, labels, train_mask, val_mask, test_mask)
    return data

def adj_list_from_dict(graph):
    G = nx.from_dict_of_lists(graph)
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    indices = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    return indices


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


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


def add_self_loops(edge_list, size):
    i = torch.arange(size, dtype=torch.int64).view(1, -1)
    self_loops = torch.cat((i, i), dim=0)
    edge_list = torch.cat((edge_list, self_loops), dim=1)
    return edge_list


def get_degree(edge_list):
    row, col = edge_list
    deg = torch.bincount(row)
    return deg


def normalize_adj(edge_list):
    deg = get_degree(edge_list)
    row, col = edge_list
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj