from torch.utils.data import Dataset
import torch
import os
import json
import glob
import numpy as np
import math
import random
import time
import cv2
from dataset.house import House, symmetry_faceid

# Seq2Seq dataset
######################################################
class HouseDataset(Dataset):
    def __init__(self, phase, data_root, exclude):
        self.phase = phase
        self.categories = ["ii","iii","l","u","comp"]
        all_paths = []
        for c in self.categories:
            c_path = data_root + "/" + c + "/obj/*.txt"
            all_paths.extend(glob.glob(c_path))
        random.seed(0)
        random.shuffle(all_paths)
        all_paths = np.asarray(all_paths)

        num_blocks = []
        for path in all_paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    num_blocks.append(int(line))
                    break
        num_blocks = np.asarray(num_blocks) 

        self.transform = phase=="train"
        if exclude==0:
            if phase == "train":
                all_paths = all_paths[:-64]
                num_blocks = num_blocks[:-64]
            elif phase == "test":
                all_paths = all_paths[-64:]
            else:
                all_paths = all_paths[:-64]
        else:
            if phase == "train":
                flag = num_blocks!=exclude
                num_blocks = num_blocks[flag]
            elif phase == "test":
                flag = num_blocks==exclude
            else:
                raise NotImplementedError
            all_paths = all_paths[flag]

        self.all_paths = all_paths
        

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, index):
        t = int(time.time() * 1000000)
        random.seed(((t & 0xff000000) >> 24) +
                       ((t & 0x00ff0000) >> 8) +
                       ((t & 0x0000ff00) << 8) +
                       ((t & 0x000000ff) << 24))
        path = self.all_paths[index]
        tokens = [token for token in path.split('/')]
        category = tokens[-3]
        name = tokens[-1]

        block_types, founda_positions, roof_angles, roof_heights = [],[],[],[]
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i==0:
                    num_blocks = int(line)
                elif i==1:
                    founda_height = float(line)
                else:
                    tokens = [float(token) for token in line.split('\t') if token.strip() != '']
                    block_types.append(int(tokens[0]))
                    founda_positions.append(tokens[1:5])
                    #roof_angles.append([tokens[5], tokens[7]])
                    roof_angles.append([max(0.,math.cos(tokens[5])), max(0.,math.cos(tokens[7]))])

        block_types = np.asarray(block_types)
        founda_positions = np.asarray(founda_positions) # blocknum x 4
        roof_angles = np.asarray(roof_angles) # blocknum x 2
        if num_blocks != len(block_types):
            print("num blocks error")
            exit()
        # data augmentation
        if self.transform:
            # top bottom mirror
            if random.random()>0.5:
                for i in range(num_blocks):
                    tmp_pos = founda_positions[i,0]
                    founda_positions[i,0] = 63-founda_positions[i,1]
                    founda_positions[i,1] = 63-tmp_pos
            # left right mirror
            if random.random()>0.5:
                for i in range(num_blocks):
                    tmp_pos = founda_positions[i,2]
                    founda_positions[i,2] = 63-founda_positions[i,3]
                    founda_positions[i,3] = 63-tmp_pos
            # rotate 90
            if random.random()>0.5:
                for i in range(num_blocks):
                    tmp_pos = founda_positions[i,0]
                    founda_positions[i,0] = founda_positions[i,2]
                    founda_positions[i,2] = tmp_pos
                    tmp_pos = founda_positions[i,1]
                    founda_positions[i,1] = founda_positions[i,3]
                    founda_positions[i,3] = tmp_pos
                    block_types[i] = symmetry_faceid(block_types[i])
                    tmp_pos = roof_angles[i,0]
                    roof_angles[i,0] = roof_angles[i,1]
                    roof_angles[i,1] = tmp_pos

        # # shift to center
        top_bound = np.amin(founda_positions[:,0])
        bottom_bound = np.amax(founda_positions[:,1])
        left_bound = np.amin(founda_positions[:,2])
        right_bound = np.amax(founda_positions[:,3])
        founda_positions[:,0:2] += 31.5-(top_bound+bottom_bound)/2.
        founda_positions[:,2:4] += 31.5-(left_bound+right_bound)/2.

        house = House(block_types, founda_positions, founda_height, roof_angles)

        # check relationship
        middles = np.full_like(founda_positions, -1.)
        for i in range(num_blocks):
            middles[i,0] = (founda_positions[i,0]+founda_positions[i,1])/2.
            middles[i,1] = middles[i,0]
            middles[i,2] = (founda_positions[i,2]+founda_positions[i,3])/2.
            middles[i,3] = middles[i,2]
        colinear_onehots = [] # edge graph
        for i in range(1, num_blocks):
            for j in range(i):
                tmp_label = np.zeros(6)
                for f in range(4):
                    if abs(founda_positions[i,f]-founda_positions[j,f])<1.:
                        tmp_label[f] = 1.
                for f in range(2):
                    if np.dot(house.roof_normals[i,f],house.roof_normals[j,f])>0.94:
                        tmp_label[f+4] = 1.

                colinear_onehots.append(tmp_label)
        colinear_onehots = np.asarray(colinear_onehots)
        
        # roof
        roof_onehot_maps = np.zeros((num_blocks,3,32,32))
        roof_angle_maps = np.zeros((num_blocks,32,32))
        for i in range(num_blocks):
            t_onehot, t_angle = house.rasterize_block(i)
            t_onehot = cv2.resize(t_onehot, (32,32))
            t_angle = cv2.resize(t_angle, (32,32))
            t_onehot = np.transpose(t_onehot, (2,0,1))
            roof_onehot_maps[i], roof_angle_maps[i] = t_onehot, t_angle

        sample = {
            "path": path,
            "num_blocks": num_blocks,
            "founda_height": founda_height,
            #"founda_masks": torch.FloatTensor(founda_masks),
            "roof_onehot_maps": torch.FloatTensor(roof_onehot_maps),
            "roof_angle_maps": torch.FloatTensor(roof_angle_maps).unsqueeze(1),
            "colinear_onehots": torch.FloatTensor(colinear_onehots), # n(n-1)/2
            # "blockg_edge": torch.LongTensor(blockg_edge), # n(n-1)
            # "edgeg_node": torch.LongTensor(edgeg_node), # n(n-1)/2
            # "edgeg_edge": torch.LongTensor(edgeg_edge) # n(n-1)/2*(n-1)*2 = n(n-1)(n-2)
        }

        return sample

def pad_collate_fn_for_dict(batch):
    n_parts_batch = [d['num_blocks'] for d in batch]
    founda_height = [d['founda_height'] for d in batch]
    name_batch = [d['path'] for d in batch]

    colinear_onehots = [d['colinear_onehots'] for d in batch]
    colinear_onehots = torch.cat(colinear_onehots, dim=0)

    roof_onehot_maps = [d['roof_onehot_maps'] for d in batch]
    roof_onehot_maps = torch.cat(roof_onehot_maps, dim=0)

    roof_angle_maps = [d['roof_angle_maps'] for d in batch]
    roof_angle_maps = torch.cat(roof_angle_maps, dim=0)
    
    return {"path": name_batch,
            "num_blocks": torch.LongTensor(n_parts_batch),
            "founda_height": founda_height,
            "roof_onehot_maps": roof_onehot_maps,
            "roof_angle_maps": roof_angle_maps,
            "colinear_onehots": colinear_onehots
            }


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size).type_as(vec)], dim=dim)

if __name__ == "__main__":
    pass