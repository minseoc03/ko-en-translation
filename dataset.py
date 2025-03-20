import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from omegaconf import DictConfig

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data.loc[idx, '원문'], self.data.loc[idx, '번역문']

def get_dataloader(cfg: DictConfig):
    data = pd.read_excel(cfg.data_path)
    custom_DS = CustomDataset(data)
    train_DS, val_DS, test_DS = torch.utils.data.random_split(custom_DS, [cfg.train_size, cfg.val_size, cfg.test_size])

    train_DL = torch.utils.data.DataLoader(train_DS, batch_size = cfg.batch_size, shuffle = True)
    val_DL = torch.utils.data.DataLoader(val_DS, batch_size = cfg.batch_size, shuffle = True)
    test_DL = torch.utils.data.DataLoader(test_DS, batch_size = cfg.batch_size, shuffle = True)

    return train_DL, val_DL, test_DL