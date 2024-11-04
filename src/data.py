#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torchvision
from torch.utils.data import Dataset
import warnings
from datasets import load_dataset
from torchvision import transforms
import torch
warnings.filterwarnings("ignore")


class MyDataset(Dataset):
    def __init__(self, 
                 data_path: str, # path to download dataset
                 hugging_address: str, # huggingface repository name 
                 dataset_type: str="train"):
        self.dataset_type = dataset_type
        self.data = load_dataset(hugging_address,
                                 cache_dir=data_path,
                                 split=dataset_type)