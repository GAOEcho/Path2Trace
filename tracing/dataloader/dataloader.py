import pickle
import sys

from torch.utils.data import Dataset

sys.path.append('..')
from utils import generate_pos_neg_samples
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
from bidict import bidict


class path_data(Dataset):
    def __init__(self, data) -> None:
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_dataloader(config, net, old2new_case_id):
    batch_size = config.batch_size
    sample_type = config.sample_type

    if os.path.exists(
        f'dataset/case_idx_pair/all_idx_pair_{str(config.lbd)}_{sample_type}.pickle'
    ):

        with open(f'dataset/case_idx_pair/all_idx_pair_{str(config.lbd)}_{sample_type}.pickle', 'rb') as file:
            all_idx_pair = pickle.load(file=file)
            file.close()
    else:
        all_idx_pair = generate_pos_neg_samples(config, net, old2new_case_id)

    if os.path.exists(
        f'dataset/case_idx_pair/train_dataloader_{str(config.lbd)}_{sample_type}.pickle'
    ):

        with open(f'dataset/case_idx_pair/train_dataloader_{str(config.lbd)}_{sample_type}.pickle', 'rb') as file:
            train_dataloader = pickle.load(file=file)
            file.close()
        with open(f'dataset/case_idx_pair/test_dataloader_{str(config.lbd)}_{sample_type}.pickle', 'rb') as file:
            test_dataloader = pickle.load(file=file)
            file.close()
        with open(f'dataset/case_idx_pair/all_dataloader_{str(config.lbd)}_{sample_type}.pickle', 'rb') as file:
            all_dataloader = pickle.load(file=file)
            file.close()
    else:
        data = path_data(all_idx_pair)
        train_size = int(config.train_ratio * len(data))
        test_size = len(data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        with open(f'dataset/case_idx_pair/train_dataloader_{str(config.lbd)}_{sample_type}_{str(config.train_ratio)}.pickle', 'wb') as file:
            pickle.dump(train_dataloader, file, protocol=2)
            file.close()
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        with open(f'dataset/case_idx_pair/test_dataloader_{str(config.lbd)}_{sample_type}_{str(config.train_ratio)}.pickle', 'wb') as file:
            pickle.dump(test_dataloader, file, protocol=2)
            file.close()
        all_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
        with open(f'dataset/case_idx_pair/all_dataloader_{str(config.lbd)}_{sample_type}.pickle', 'wb') as file:
            pickle.dump(all_dataloader, file, protocol=2)
            file.close()

    return train_dataloader, test_dataloader, all_dataloader


def get_disjoint_dataloader(config, net, old2new_case_id):
    batch_size = config.batch_size
    sample_type = config.sample_type

    sub_nets = [net.subgraph(c).copy() for c in nx.connected_components(net)]
    full_idx = list(range(len(sub_nets)))
    np.random.shuffle(full_idx)
    sub_nets_shuff = [sub_nets[i] for i in full_idx]
    train_size = int(config.train_ratio * len(sub_nets))

    train_sub_nets = sub_nets_shuff[:train_size]
    test_sub_nets = sub_nets_shuff[train_size:]
    train_nodes = []
    for sub_net in train_sub_nets:
        train_nodes += sub_net.nodes()
    test_nodes = set(net.nodes()) - set(train_nodes)
    train_net = net.subgraph(train_nodes)
    test_net = net.subgraph(test_nodes)

    train_idx_pair = generate_pos_neg_samples(config, train_net, old2new_case_id)
    test_idx_pair = generate_pos_neg_samples(config, test_net, old2new_case_id)
    all_idx_pair = train_idx_pair + test_idx_pair

    all_dataset = path_data(all_idx_pair)
    dataset_size = len(all_dataset)
    indices = list(range(dataset_size))
    split = len(train_idx_pair)
    train_indices, test_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dataloader = torch.utils.data.DataLoader(all_dataset, batch_size=batch_size, sampler=train_sampler,
                                                   drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(all_dataset, batch_size=batch_size, sampler=test_sampler,
                                                  drop_last=True)
    all_dataloader = torch.utils.data.DataLoader(all_dataset, batch_size=batch_size, drop_last=True)

    return train_dataloader, test_dataloader, all_dataloader
