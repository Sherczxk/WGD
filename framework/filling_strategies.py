"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import torch
import torch_sparse

from feature_propagation import FeaturePropagation
from pcfi import pcfi
from wgd import wgd
from ginn import ginn
from utils import normalize_features, inverse_transform, reduction_transform

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer,IterativeImputer
from sklearn.utils.extmath import randomized_svd

def random_filling(X):
    return torch.randn_like(X)


def zero_filling(X):
    # return torch.zeros_like(X)
    mask = ~torch.isnan(X)
    X_filled = torch.nan_to_num(X, nan=0.0)
    return X_filled


def mean_filling(X, feature_mask):
    n_nodes = X.shape[0]
    return compute_mean(X, feature_mask).repeat(n_nodes, 1)


def neighborhood_mean_filling(edge_index, X, feature_mask):
    n_nodes = X.shape[0]
    X_zero_filled = X
    X_zero_filled[~feature_mask] = 0.0
    edge_values = torch.ones(edge_index.shape[1]).to(X.device)
    edge_index_mm = torch.stack([edge_index[1], edge_index[0]]).to(X.device)

    D = torch_sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, feature_mask.float())
    mean_neighborhood_features = torch_sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, X_zero_filled) / D
    # If a feature is not present on any neighbor, set it to 0
    mean_neighborhood_features[mean_neighborhood_features.isnan()] = 0

    return mean_neighborhood_features



def soft_threshold(X, threshold):
    """Soft thresholding function."""
    return torch.sign(X) * torch.relu(torch.abs(X) - threshold)

def softimpute_gpu(X, rank, lambda_, max_iter=100, tol=1e-5):
    """
    SoftImpute algorithm for matrix completion using PyTorch on GPU.

    Args:
        X (torch.Tensor): Input matrix with missing values (NaN).
        rank (int): Rank for the low-rank approximation.
        lambda_ (float): Regularization parameter.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        torch.Tensor: Completed matrix.
    """
    # Move input to GPU
    # X = torch.tensor(X, dtype=torch.float32).to(device)
    mask = ~torch.isnan(X)
    X_filled = torch.nan_to_num(X, nan=0.0)

    for i in range(max_iter):
        # SVD on the filled matrix
        U, S, V = torch.linalg.svd(X_filled, full_matrices=False)

        # Truncate SVD to the specified rank
        U = U[:, :rank]
        S = S[:rank]
        V = V[:rank, :]

        # Apply soft thresholding to singular values
        S_thresh = soft_threshold(S, lambda_)

        # Reconstruct the low-rank matrix
        X_prev = X_filled.clone()
        X_filled = U @ torch.diag(S_thresh) @ V

        # Fill in the observed values
        X_filled[mask] = X[mask]

        # Check for convergence
        diff = torch.norm(X_filled - X_prev) / torch.norm(X_prev)
        if diff < tol:
            print(f"Converged at iteration {i}")
            break

    return X_filled

def knn_impute(features):
    imputer = KNNImputer(n_neighbors=5)
    x = imputer.fit_transform(features.cpu().numpy())
    return torch.from_numpy(x).float().to(features.device)
        
        
def pca_impute(features,n_component):
    X = zero_filling(features)
    U,S,base = reduction_transform(X,n_component)
    x = inverse_transform(U,S,base)
    return x

def knn_impute_gpu(
    X: torch.Tensor,
    feature_mask: torch.Tensor,
    k: int = 5,
    metric: str = 'euclidean',
    aggregate: str = 'mean',
    device: str = 'cuda'
) -> torch.Tensor:
    """
    GPU-accelerated KNN Imputation for missing values.
    
    Args:
        X: Input feature matrix with NaN values indicating missing data
            shape: (num_samples, num_features)
        feature_mask: Boolean mask where True indicates observed values
            shape: (num_samples, num_features)
        k: Number of nearest neighbors to use
        metric: Distance metric ('euclidean' or 'cosine')
        aggregate: Aggregation method ('mean' or 'median')
        device: Device to use for computation ('cuda' or 'cpu')
    
    Returns:
        X_imputed: Imputed feature matrix
    """
    # Move data to device
    X = X.to(device)
    feature_mask = feature_mask.to(device)
    
    # Replace NaNs with 0 for computation
    X_zero = torch.where(feature_mask, X, torch.tensor(0., device=device))
    
    # 1. Compute pairwise distances
    if metric == 'euclidean':
        # Efficient masked Euclidean distance calculation
        dot_product = torch.mm(X_zero, X_zero.T)
        magnitude = (X_zero**2).sum(1)
        dist_matrix = magnitude.unsqueeze(1) + magnitude - 2 * dot_product
        
        # Apply masking
        valid_pairs = torch.mm(feature_mask.float(), feature_mask.float().T)
        dist_matrix = torch.sqrt(dist_matrix / (valid_pairs + 1e-8))
        dist_matrix[valid_pairs == 0] = float('inf')
        
    elif metric == 'cosine':
        # Masked cosine similarity
        norm = torch.norm(X_zero, dim=1, keepdim=True)
        X_normalized = X_zero / (norm + 1e-8)
        similarity = torch.mm(X_normalized, X_normalized.T)
        
        valid_pairs = torch.mm(feature_mask.float(), feature_mask.float().T)
        dist_matrix = 1 - similarity
        dist_matrix[valid_pairs == 0] = float('inf')
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # 2. Find k nearest neighbors (excluding self)
    _, topk_indices = torch.topk(dist_matrix, k+1, dim=1, largest=False)
    neighbor_indices = topk_indices[:, 1:k+1]  # Shape: (n_samples, k)

    # 3. Gather neighbor features and masks
    neighbor_features = X_zero[neighbor_indices]  # Shape: (n_samples, k, n_features)
    neighbor_masks = feature_mask[neighbor_indices]  # Shape: (n_samples, k, n_features)

    # 4. Aggregate values
    if aggregate == 'mean':
        # Masked mean aggregation
        sum_vals = (neighbor_features * neighbor_masks).sum(dim=1)
        valid_counts = neighbor_masks.sum(dim=1)
        imputed_vals = sum_vals / (valid_counts + 1e-8)
    elif aggregate == 'median':
        # Median aggregation (handling missing values)
        imputed_vals = torch.zeros_like(X)
        for feat_idx in range(X.shape[1]):
            feat_mask = feature_mask[:, feat_idx]
            missing_mask = ~feat_mask
            
            if not missing_mask.any():
                continue
                
            # Get neighbors for samples missing this feature
            neighbors = neighbor_features[missing_mask, :, feat_idx]  # (n_missing, k)
            neighbor_valids = neighbor_masks[missing_mask, :, feat_idx]  # (n_missing, k)
            
            # Filter valid neighbors
            valid_neighbors = [neighbors[i][neighbor_valids[i]] for i in range(neighbors.shape[0])]
            
            # Compute medians
            medians = torch.stack([
                torch.median(v) if len(v) > 0 else torch.tensor(0.)
                for v in valid_neighbors
            ]).to(device)
            
            imputed_vals[missing_mask, feat_idx] = medians
    else:
        raise ValueError(f"Unsupported aggregation: {aggregate}")

    # 5. Fill in missing values
    X_imputed = torch.where(feature_mask, X, imputed_vals)
    
    # Fallback to feature means if no neighbors available
    if aggregate == 'mean':
        feature_means = (X_zero * feature_mask).sum(0) / (feature_mask.sum(0) + 1e-8)
        missing_mask = (valid_counts == 0) & (~feature_mask)
        X_imputed[missing_mask] = feature_means[None, :].expand_as(X_imputed)[missing_mask]

    return X_imputed



def feature_propagation(edge_index, X, feature_mask, num_iterations):
    propagation_model = FeaturePropagation(num_iterations=num_iterations)

    return propagation_model.propagate(x=X, edge_index=edge_index, mask=feature_mask)


def filling(filling_method, edge_index, X, feature_mask, **kwargs):
    if filling_method == "random":
        X_reconstructed = random_filling(X)
    elif filling_method == "zero":
        X_reconstructed = zero_filling(X)
    elif filling_method == "mean":
        X_reconstructed = mean_filling(X, feature_mask)
    elif filling_method == "neighborhood_mean":
        X_reconstructed = neighborhood_mean_filling(edge_index, X, feature_mask)
    elif filling_method == "softimpute_gpu":
        X_reconstructed = softimpute_gpu(X, rank=kwargs["n_component"], lambda_=0.0)
    elif filling_method == "knn_impute":
        X_reconstructed = knn_impute(X)
        X_reconstructed = normalize_features(X_reconstructed)
    elif filling_method == "knn_impute_gpu":
        X_reconstructed = knn_impute_gpu(X, feature_mask)
    elif filling_method == "pca_impute":
        X_reconstructed = pca_impute(X, n_component=kwargs["n_component"]) 
    elif filling_method == "ginn":
        X_reconstructed =ginn(X,kwargs["missing_rate"],kwargs["seed"],train_size=0.5)
    elif filling_method == "feature_propagation":
        X_reconstructed = feature_propagation(edge_index, X, feature_mask, num_iterations=100)
    elif filling_method == 'pcfi':
        X_reconstructed = pcfi(edge_index, X, feature_mask, num_iterations=100, missing_type=kwargs["missing_type"], alpha=0.9, beta=1)
    elif filling_method == 'wgd':
        X[~feature_mask] = 0.0
        X_reconstructed = wgd(X, data=kwargs["data"], n_component=kwargs["n_component"], \
            h_hop=kwargs["h_hop"], layer_L=kwargs["layer_L"], bary_comp_para=kwargs["bary_comp_para"])
    else:
        raise ValueError(f"{filling_method} method not implemented")
    return X_reconstructed


def compute_mean(X, feature_mask):
    X_zero_filled = X
    X_zero_filled[~feature_mask] = 0.0
    num_of_non_zero = torch.count_nonzero(feature_mask, dim=0).to(X.device)
    mean_features = torch.sum(X_zero_filled, axis=0) / num_of_non_zero
    # If a feature is not present on any node, set it to 0
    mean_features[mean_features.isnan()] = 0

    return mean_features