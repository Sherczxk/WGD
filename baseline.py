import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import SGConv
from torch_geometric.utils import add_self_loops, degree
from torch.nn import init
import pdb


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(hidden_dim, output_dim)


    def forward(self, feature,data,use_feature):
        if use_feature:
            x = feature.cuda()
        else:
            x = torch.eye(data.x.shape[0],data.x.shape[1]).cuda()
        x = self.linear_first(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, feature,data, use_feature):
        if use_feature:
            x = feature.cuda()
        else:
            x = torch.eye(data.x.shape[0],data.x.shape[1]).cuda()
        edge_index = data.edge_index
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num=2, dropout=True, **kwargs):
        super(SAGE, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.SAGEConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.SAGEConv(hidden_dim, output_dim)

    def forward(self,feature, data, use_feature):
        if use_feature:
            x = feature.cuda()
        else:
            x = torch.eye(data.x.shape[0],data.x.shape[1]).cuda()
        edge_index = data.edge_index
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x



class tgGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,heads, dropout=0.5, **kwargs):
        super(tgGAT, self).__init__()
        self.heads = heads
        self.dropout = dropout
        self.attentions = [tg.nn.GATConv(input_dim, hidden_dim, dropout=dropout, concat=True) for _ in range(8)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.conv_first = tg.nn.GATConv(input_dim, hidden_dim,8,dropout=0.5,concat=True, negative_slope=0.2, bias=True)
        self.conv_out = tg.nn.GATConv(8*hidden_dim, output_dim,heads,concat=False,dropout=0.5,negative_slope=0.2, bias=True)

    def forward(self,feature, data, use_feature):
        if use_feature:
            x = feature.cuda()
        else:
            x = torch.eye(data.x.shape[0],data.x.shape[1]).cuda()
        edge_index = data.edge_index
        x = F.dropout(x, training=self.training)
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        #x = self.conv_first(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        
        x = F.elu(x)
        x = F.log_softmax(x, dim=1)
        return x



class GIN(torch.nn.Module):
    def __init__(self, input_dim,hidden_dim, output_dim,layer_num=2, dropout=True, **kwargs):
        super(GIN, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        self.conv_first_nn = nn.Linear(input_dim, hidden_dim)
        self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        self.conv_hidden_nn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_hidden = nn.ModuleList([tg.nn.GINConv(self.conv_hidden_nn[i]) for i in range(layer_num - 2)])

        self.conv_out_nn = nn.Linear(hidden_dim, output_dim)
        self.conv_out = tg.nn.GINConv(self.conv_out_nn)

    def forward(self,feature, data,use_feature):
        if use_feature:
            x = feature.cuda()
        else:
            x = torch.eye(data.x.shape[0],data.x.shape[1]).cuda()
        edge_index = data.edge_index
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class SGC(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super(SGC, self).__init__()
        self.conv1 = SGConv(
            num_features, num_classes, K=2, cached=False)
        
    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self,feature, data, use_feature):
        if use_feature:
            x = feature.cuda()
        else:
            x = torch.eye(data.x.shape[0],data.x.shape[1]).cuda()
        edge_index = data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)
