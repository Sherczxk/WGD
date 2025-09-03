"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import numpy as np
from tqdm import tqdm
import argparse
import logging
import time
import yaml

from yacs.config import CfgNode as CN
import torch
from torch_geometric.data import NeighborSampler

from dataloader import load_data
from models import get_model
from filling_strategies import filling
from evaluation import test
from train import train
from utils import missing_feature, set_seed


def run():
    parser = argparse.ArgumentParser("GNN-Missing-Features")
    parser.add_argument('--config-file', type=str, default="",
                        help="path to config file", metavar="FILE")

    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[],
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    
    cfg = CN(new_allowed=True)
    
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    # logger.info(args)

    assert not (
        cfg.graph_sampling and cfg.model != "sage"
    ), f"{cfg.model} model does not support training with neighborhood sampling"
    assert not (cfg.graph_sampling and cfg.jk), "Jumping Knowledge is not supported with neighborhood sampling"

    device = torch.device(
        f"cuda:{cfg.gpu_idx}"
        if torch.cuda.is_available() and not (cfg.dataset_name == "OGBN-Products" and cfg.model == "lp")
        else "cpu"
    )

    test_accs, best_val_accs, train_times = [], [], []

    

    for seed in cfg.seeds[:cfg.n_runs]:
        set_seed(seed)
        data, evaluator = load_data(cfg.dataset_name, seed)
        n_nodes, n_features = data.x.shape
        train_loader = (
            NeighborSampler(
                data.edge_index,
                node_idx=data.train_mask,
                sizes=[15, 10, 5][: cfg.num_layers],
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=12,
            )
            if cfg.graph_sampling
            else None
        )
        # Setting `sizes` to -1 simply loads all the neighbors for each node. We can do this while evaluating
        # as we first compute the representation of all nodes after the first layer (in batches), then for the second layer, and so on
        inference_loader = (
            NeighborSampler(
                data.edge_index, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12,
            )
            if cfg.graph_sampling
            else None
        )
        num_classes = int(data.y.max()+1)
        
        data = data.to(device)
        train_start = time.time()
        if cfg.model == "lp":
            model = get_model(
                model_name=cfg.model,
                num_features=data.num_features,
                num_classes=num_classes,
                edge_index=data.edge_index,
                x=None,
                args=cfg,
            ).to(device)
            # logger.info("Starting Label Propagation")
            logits = model(y=data.y, edge_index=data.edge_index, mask=data.train_mask)
            (_, val_acc, test_acc), _ = test(model=None, x=None, data=data, logits=logits, evaluator=evaluator)
        else:
            x = data.x.clone()
            missing_x, missing_feature_mask = missing_feature(
                x, cfg.missing_rate, cfg.missing_type,
            ) # missing value is float("nan")
            
            # logger.debug("Starting feature filling")
            start = time.time()
            if cfg.filling_method == "pcfi":
                filled_features = filling(
                    cfg.filling_method, data.edge_index, missing_x, missing_feature_mask, missing_type=cfg.missing_type,
                    )
            elif cfg.filling_method == "ginn":
                filled_features = filling(
                    cfg.filling_method, data.edge_index, missing_x, missing_feature_mask, missing_rate=cfg.missing_rate, seed=seed
                    )
            elif cfg.filling_method == "wgd":
                filled_features = filling(
                    cfg.filling_method, data.edge_index, missing_x, missing_feature_mask, data=data,\
                        n_component=cfg.n_component, h_hop=cfg.h_hop, layer_L=cfg.layer_L, bary_comp_para=cfg.bary_comp_para,
                    )
            elif cfg.filling_method in ["softimpute_gpu","pca_impute"]:
                filled_features = filling(
                    cfg.filling_method, data.edge_index, missing_x, missing_feature_mask, \
                        n_component=cfg.n_component
                    )
            else:
                filled_features = filling(
                    cfg.filling_method, data.edge_index, missing_x, missing_feature_mask,
                    )
            
            if cfg.model in ["gcnmf", "pagnn"]:
                filled_features = torch.full_like(x, float("nan"))
            # print(f"Feature filling completed. It took: {time.time() - start:.2f}s")

            del missing_x, missing_feature_mask, x
            # print(time.time()-start)
            model = get_model(
                model_name=cfg.model, 
                num_features=data.num_features,
                num_classes=num_classes,
                edge_index=data.edge_index,
                args=cfg,
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(),lr= cfg.lr,weight_decay= cfg.weight_decay)
            critereon = torch.nn.NLLLoss()

            test_acc = 0
            val_accs = []
            for epoch in range(0, cfg.epochs):
                start = time.time()

                train(
                    model, filled_features, data, optimizer, critereon, train_loader=train_loader, device=device,
                )
                (train_acc, val_acc, tmp_test_acc), out = test(
                    model, x=filled_features, data=data, evaluator=evaluator, inference_loader=inference_loader, device=device,
                )
                if epoch == 0 or val_acc > max(val_accs):
                    test_acc = tmp_test_acc
                    y_soft = out.softmax(dim=-1)

                val_accs.append(val_acc)
                if epoch > cfg.patience and max(val_accs[-cfg.patience :]) <= max(val_accs[: -cfg.patience]):
                    break
                # logger.debug(
                #     f"Epoch {epoch + 1} - Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}, Test acc: {tmp_test_acc:.3f}. It took {time.time() - start:.2f}s"
                # )

            (_, val_acc, test_acc), _ = test(model, x=filled_features, data=data, logits=y_soft, evaluator=evaluator)
        best_val_accs.append(val_acc)
        test_accs.append(test_acc)
        train_times.append(time.time() - train_start)
        # print(f"Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}, Test acc: {tmp_test_acc:.3f}, Epoch: {epoch}")

    test_acc_mean, test_acc_std = np.mean(test_accs), np.std(test_accs)
    # print(f"Test Accuracy: {test_acc_mean * 100:.2f}% +- {test_acc_std * 100:.2f}")
    print(np.mean(train_times))
    print(test_acc_mean)

if __name__ == "__main__":
    run()