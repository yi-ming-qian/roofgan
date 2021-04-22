import os
import numpy as np
from torch.utils.data import DataLoader
from config import get_config
from dataset import HouseDataset, pad_collate_fn_for_dict_house
from agent import WGANAgant
from util.utils import ensure_dir


def main():
    # create experiment config
    config = get_config('gan')()

    # create network and training agent
    tr_agent = WGANAgant(config)

    if config.is_train:
        # load from checkpoint if provided
        if config.cont:
            tr_agent.load_ckpt(config.ckpt)

        # create dataloader
        dataset = HouseDataset('train', config.data_root, config.exclude)
        
        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, collate_fn=pad_collate_fn_for_dict_house,
                                  drop_last=True)

        tr_agent.train(train_loader)
    else:
        tr_agent.load_ckpt(config.ckpt)

        # run generator
        tr_agent.generate(config)

if __name__ == '__main__':
    main()