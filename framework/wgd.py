import torch
import ot
from utils import normalize_features, Gram_Schmidt_ortho, inverse_transform, reduction_transform, get_degree_adj_list
from barycenter import sliced_wasserstein_barycenter_batch
import time


class WGCN(torch.nn.Module):
    def __init__(self,n_component,h_hop,layer_L, bary_comp_para, **kwargs):
        super(WGCN, self).__init__()
        
        self.n_component = n_component
        self.h_hop = h_hop
        self.layer_L = layer_L
        self.bary_comp_method = bary_comp_para.method
        self.num_iters = bary_comp_para.num_iters
        self.reg = bary_comp_para.reg
        self.alpha = bary_comp_para.alpha
        self.num_projs = bary_comp_para.num_projs
    
    def compute_cost(self,cost_type):
        costMatrix = torch.zeros(self.n_component,self.n_component).float()
        for i in range(self.n_component):
            for j in range(self.n_component):
                if i != j:
                    costMatrix[i][j] = abs(self.S[i]-self.S[j])
        if cost_type == 'max':
            costMatrix = costMatrix /costMatrix.max()
        elif cost_type == 'normalize':
            costMatrix = normalize_features(costMatrix)   
        return costMatrix
    
    def wasserstein_propagation(self, dis_mx, data):
        output_gmm = torch.empty_like(dis_mx)
        if self.bary_comp_method == 'sliced_batch':
            for k in self.num_neighs_index.keys():
                dis_mx_batch = torch.stack([dis_mx[data.adj_list[i]] for i in self.num_neighs_index[k]])
                
                bary = sliced_wasserstein_barycenter_batch(dis_mx_batch, \
                    num_iterations=self.num_iters, num_projections=self.num_projs, lr=self.alpha)

                if torch.any(torch.isnan(bary)) or torch.any(torch.isinf(bary)):
                    raise ValueError("Cost matrix contains NaN or inf values.")
                output_gmm[self.num_neighs_index[k]] = bary
        elif self.bary_comp_method == 'ot':
            for i in range(dis_mx.shape[0]): 
                output_gmm[i] = ot.barycenter(A=dis_mx[data.adj_list[i]].T, M=self.costMatrix, \
                    weights=self.bary_weight_list[i], reg=self.reg, numItermax=self.num_iters)
        return output_gmm
    
    def forward(self, x, data):
        
        # self.degree, self.adj_list = get_degree_adj_list(edge_index, x.shape[0])
        self.U,self.S, self.base = reduction_transform(x, self.n_component)
        if self.bary_comp_method == 'ot':
            self.costMatrix = self.compute_cost(cost_type=None).to(x.device)
            self.bary_weight_list = [1/deg*torch.ones(deg.item()).cuda() for deg in data.degree]
        elif self.bary_comp_method == 'sliced_batch':
            num_neighs = set(data.degree.cpu().numpy())
            node_index = torch.LongTensor(range(len(data.degree)))
            num_neighs_index = {}
            for i in list(num_neighs):
                num_neighs_index[i] = node_index[data.degree==i]
            self.num_neighs_index = num_neighs_index
        dis_mx = self.U
        u= torch.zeros_like(dis_mx).to(x.device)
        for i in range(self.layer_L):
            dis_mx = normalize_features(torch.exp(dis_mx))
            for h in range(self.h_hop):
                dis_mx = self.wasserstein_propagation(dis_mx, data)
            dis_mx = Gram_Schmidt_ortho(torch.log(dis_mx))
            u += dis_mx
        
        u = u/self.layer_L
        x = inverse_transform(u,self.S,self.base)
        return x
        
        
def wgd(X, data, n_component, h_hop, layer_L, bary_comp_para):

    propagation_model = WGCN(n_component,h_hop,layer_L, bary_comp_para)
    return propagation_model(x=X, data=data)
