import os
import numpy as np
import h5py
from torch.utils.data import DataLoader
from config import get_config
from dataset import HouseDataset, pad_collate_fn_for_dict_house
from agent import WGANAgant
from util.utils import ensure_dir


def main():
    # create experiment config
    config = get_config('lgan')()

    # create network and training agent
    tr_agent = WGANAgant(config)

    if config.is_train:
        # load from checkpoint if provided
        if config.cont:
            tr_agent.load_ckpt(config.ckpt)

        # create dataloader
        dataset = HouseDataset('train', config.data_root, config.exclude)
        # for i in range(5):
        #     dataset[i]
        # exit()
        # nums = np.zeros(8)
        # for i in range(len(dataset)):
        #     n = dataset[i]["num_blocks"]
        #     nums[n] += 1.
        # print(np.arange(8))
        # print(nums)
        # exit() #121. 231.  55.  19.   1, starting from 2
        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, collate_fn=pad_collate_fn_for_dict_house,
                                  drop_last=True)

        tr_agent.train(train_loader)
    else:
        # dataset = HouseDataset('test', config.data_root, config.max_n_parts)
        # for i in range(len(dataset)):
        #     dataset[i]
        # exit()
        # load trained weights
        tr_agent.load_ckpt(config.ckpt)

        # run generator
        generated_shape_codes = tr_agent.generate(config)

        # save generated z
        save_path = os.path.join(config.exp_dir, "results/fake_z_ckpt{}_num{}.h5".format(config.ckpt, config.n_samples))
        ensure_dir(os.path.dirname(save_path))
        with h5py.File(save_path, 'w') as fp:
            fp.create_dataset("zs", shape=generated_shape_codes.shape, data=generated_shape_codes)


if __name__ == '__main__':
    main()