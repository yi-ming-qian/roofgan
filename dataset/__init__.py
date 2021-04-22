from torch.utils.data import DataLoader
from dataset.dataset_house import HouseDataset
from dataset.dataset_house import pad_collate_fn_for_dict as pad_collate_fn_for_dict_house
import numpy as np


def get_dataloader(phase, config, use_all_points=False, is_shuffle=None):
    is_shuffle = phase == 'train' if is_shuffle is None else is_shuffle

    if config.module == 'house' or config.module == 'lvae' or config.module == "houseplus":
        dataset = HouseDataset(phase, config.data_root, config.exclude)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle,
                                num_workers=config.num_workers, collate_fn=pad_collate_fn_for_dict_house)
    else:
        raise NotImplementedError
    return dataloader
