# -*- coding: utf-8 -*-
"""Data loader: contact map NPZ -> DGL graph, for training GARPepPI."""
import torch
import dgl
import scipy.sparse as spp
import numpy as np
import sys
import os
from torch.utils.data import DataLoader, Dataset

# Set these before training:
#   embed_data = np.load("path/to/embeddings.npz", allow_pickle=True)
#   cmaproot   = "path/to/contact_map/"
# device = torch.device('cuda')
device = torch.device('cuda')


def collate(samples):
    p1, p2, graphs1, dmaps1, graphs2, dmaps2, labels = map(list, zip(*samples))
    return p1, p2, graphs1, dmaps1, graphs2, dmaps2, torch.tensor(labels)


def default_loader(cpath, pid):
    cmap_data = np.load(cpath)
    nodenum = len(str(cmap_data['seq']))
    cmap = cmap_data['contact']

    g_embed = torch.tensor(embed_data[pid][:nodenum]).float().to(device)

    adj = spp.coo_matrix(cmap)
    G = dgl.DGLGraph(adj).to(device)
    G.ndata['feat'] = g_embed

    if nodenum > 1200:
        textembed = embed_data[pid][:1200]
    else:
        textembed = np.concatenate((embed_data[pid], np.zeros((1200 - nodenum, 1024))))
    textembed = torch.tensor(textembed).float().to(device)
    return G, textembed


class MyDataset(Dataset):
    def __init__(self, list1, list2, list3, transform=None, target_transform=None, loader=default_loader):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        p1 = self.list1[index]
        p2 = self.list2[index]
        label = self.list3[index]
        G1, embed1 = self.loader(cmaproot + p1 + '.npz', p1)
        G2, embed2 = self.loader(cmaproot + p2 + '.npz', p2)
        return p1, p2, G1, embed1, G2, embed2, label

    def __len__(self):
        return len(self.list1)


def pad_dmap(dmaplist):
    pad_dmap_tensors = torch.zeros((len(dmaplist), 1200, 1024)).float()
    for idx, d in enumerate(dmaplist):
        d = d.float().cpu()
        pad_dmap_tensors[idx] = torch.FloatTensor(d)
    pad_dmap_tensors = pad_dmap_tensors.unsqueeze(1).cuda()
    return pad_dmap_tensors
