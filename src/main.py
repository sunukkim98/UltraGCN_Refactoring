
import torch
from pathlib import Path

from loguru import logger
from utils import set_random_seed
from utils import log_param
from data import MyDataset

"""
hyper parameter experiments in paper
ii_neighbor_num [5, 10, 20, 50]
gamma [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
lambda [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
"""
def main(model='mymodel',
         seed=-1: int,
         batch_size=1024: int,
         max_epoch=2000: int,
         early_stop_epoch=15: int,
         learning_rate=1e-3: float,

         ii_neighbor_num=5: int,
         gamma=1e-4: float,
         lambda_=2.75: float):
    """
    Handle user arguments of UltraGCN_Refactoring

    :param model: name of model to be trained and tested
    :param seed: random_seed (if -1, a default seed is used)
    :param batch_size: size of batch
    :param max_epoch: number of maximum training epochs
    :param early_stop_epoch: number of early stop epochs
    :param learning_rate: learning rate
    
    :param ii_neighbor_num: number of neighbors to consider for each item
    :param gamma: adjusts the weight in item-item graph learning
    :param lambda_: adjusts the weight in user-item graph learning
    """

    # Step 0. Initialization
    logger.info("The main procedure has started with the following parameters:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_random_seed(seed=seed, device=device)
    param = dict()
    param['model'] = model
    param['seed'] = seed
    param['device'] = device
    param['ii_neighbor_num'] = ii_neighbor_num
    param['gamma'] = gamma
    param['lambda'] = lambda_
    log_param(param)

    # Step 1. Load datasets
    data_path = Path(__file__).parents[1].absolute().joinpath("datasets")
    train_data = MyDataset()
