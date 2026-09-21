# -*- coding: utf-8 -*-
"""Data utilities: SELFIES encoding, vocab mapping, contact map loading, padding."""
import torch
import re
import numpy as np
from selfies import encoder as smiles2selfies

# vocab file: one token per line (space-separated)
def getLetters(path):
    with open(path, 'r') as f:
        return f.read().split()


def get_selfies_list(smiles_list):
    return [smiles2selfies(smiles) for smiles in smiles_list]


def letterToIndex(letter, letters):
    return letters.index(letter)


def line2voc_arr(line, letters):
    regex = '(\[[^\[\]]{1,10}\])'
    char_list = re.split(regex, line)
    arr = []
    for char in char_list:
        if char.startswith('['):
            arr.append(letterToIndex(char, letters))
        else:
            for unit in char:
                arr.append(letterToIndex(unit, letters))
    return arr, len(arr)


def protein2contact_arr(protein_num, contactPath):
    contactMap = np.load(contactPath + protein_num + '.npz')['contact']
    return contactMap, contactMap.shape[0]


def getDataSet(FoldPath):
    with open(FoldPath, 'r') as f:
        cpi_list = f.read().strip().split('\n')
    cpi_list = [d for d in cpi_list if '.' not in d.strip().split()[0]]
    return [cpi.strip().split() for cpi in cpi_list]


def make_variables(lines, proteins, properties, letters, seq2numDic, contactPath):
    lines = get_selfies_list(lines)
    seq_and_len = [line2voc_arr(line, letters) for line in lines]
    vectorized = [sl[0] for sl in seq_and_len]
    seq_lengths = torch.LongTensor([sl[1] for sl in seq_and_len])
    contactMaps_and_sizes = [
        protein2contact_arr(str(seq2numDic[protein.encode('utf-8')]), contactPath)
        for protein in proteins
    ]
    contactMaps = [cm[0] for cm in contactMaps_and_sizes]
    contact_sizes = torch.LongTensor([cm[1] for cm in contactMaps_and_sizes])
    return pad_sequences(vectorized, seq_lengths, contactMaps, contact_sizes, properties)


def pad_sequences(vectorized_seqs, seq_lengths, contactMaps, contact_sizes, properties):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    contactMaps_tensor = torch.zeros(
        (len(contactMaps), contact_sizes.max(), contact_sizes.max())
    ).float()
    for idx, (con, con_size) in enumerate(zip(contactMaps, contact_sizes)):
        contactMaps_tensor[idx, :con_size, :con_size] = torch.FloatTensor(con)

    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    contactMaps_tensor = contactMaps_tensor[perm_idx]
    contact_sizes = contact_sizes[perm_idx]

    target = properties.double()
    if len(properties):
        target = target[perm_idx]

    contactMaps_tensor = contactMaps_tensor.unsqueeze(1)
    return seq_tensor, seq_lengths, contactMaps_tensor, contact_sizes, target
