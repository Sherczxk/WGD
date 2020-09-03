import argparse
import numpy as np
import optuna
import random
import torch

from tqdm import tqdm

from utils import *

import pickle
import ot
import time
from utils import *
from dataloader import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora', choices=['cora', 'citeseer', 'amacomp', 'amaphoto'],
                    help='dataset name')
parser.add_argument('--type', default='whole', choices=['part', 'whole'],
                    help="part missing or whole missing")
parser.add_argument('--rate', default=0.1, help='missing rate')
parser.add_argument('--seed', default=17)

args = parser.parse_args()
dataset_str = args.dataset
missing_type = args.type
rate = float(args.rate)
SEED = int(args.seed)
TRIAL_SIZE = 100
TIMEOUT = 60 * 60 * 3

patience, epochs = 100, 10000
opt = 'Adam'
lr = 0.01

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

print(dataset_str, missing_type, rate)
print("patience:", patience)
print("epochs:", epochs)


def objective(trial):
    # Tune hyperparameters (dropout, weight decay, learning rate) using Optuna
    hidden_dim = trial.suggest_int('hidden_dim', 16, 40)
    n_component = trial.suggest_int('n_component', hidden_dim, 80)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-3, 1e-1)

    # prepare data and model
    data = load_data(dataset_str)
    class_num = int(torch.max(data.labels)) + 1

    # run model
    result = model(opt,data,hidden_dim,class_num, missing_type,n_component,rate,lr, weight_decay, epochs)
    return - result['val_acc']


def tune_hyperparams():
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE, timeout=TIMEOUT)
    return study.best_params


def evaluate_model(params):
    hidden_dim = params['hidden_dim']
    n_component = params['n_component'] 
    weight_decay = params['weight_decay']
    
    data = load_data(dataset_str)
    class_num = int(torch.max(data.labels)) + 1

    # run model
    result = model(opt,data,hidden_dim,class_num, missing_type,n_component,rate,lr, weight_decay, epochs)

    return result['test_acc']


def main():
    params = tune_hyperparams()
    result = evaluate_model(params)
    print('params',params)
    print('result',result)


if __name__ == '__main__':
    main()
