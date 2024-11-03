
import torch
from pathlib import Path

from loguru import logger
from utils import set_random_seed
from utils import log_param
from data import MyDataset

def main(model='mymodel',
         seed=-1):
    """
    Handle user arguments of UltraGCN_Refactoring

    :param ___: ___Explain_about_param
    :param seed: random_seed (if -1, a default seed is used)
    """

    # Step 0. Initialization
    logger.info("The main procedure has started with the following parameters:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_random_seed(seed=seed, device=device)
    param = dict()
    param['model'] = model
    param['seed'] = seed
    param['device'] = device
    log_param(param)

    # Step 1. Load datasets
    data_path = Path(__file__).parents[1].absolute().joinpath("datasets")
    train_data = MyDataset()
