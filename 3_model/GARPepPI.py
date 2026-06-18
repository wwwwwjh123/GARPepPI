# -*- coding: utf-8 -*-
"""GARPepPI: Graph Attention + TextCNN for Protein-Peptide Interaction Prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import TAGConv
from dgl.nn.pytorch.glob import MaxPooling


class ConvsLayer(nn.Module):
    """TextCNN: 3-layer 1D conv to extract sequence features. Output shape: (B, 128)."""
    def __init__(self, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(emb_dim, 128, kernel_size=3)
        self.mx1   = nn.MaxPool1d(3, stride=3)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3)
        self.mx2   = nn.MaxPool1d(3, stride=3)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3)
        self.mx3   = nn.MaxPool1d(130, stride=1)

    def forward(self, x):
        # Input: (B, 1, L, D) -> (B, D, L)
        x = x.squeeze(1).permute(0, 2, 1)
        x = F.relu(self.conv1(x));  x = self.mx1(x)
        x = F.relu(self.conv2(x));  x = self.mx2(x)
        x = F.relu(self.conv3(x));  x = self.mx3(x)
        return x.squeeze(2)  # (B, 128)


class GATPPI(nn.Module):
    """
    Predicts protein-peptide interaction probability.
    Fuses TAGConv graph features with TextCNN sequence features.
    """
    def __init__(self, args):
        super().__init__()
        torch.backends.cudnn.enabled = False
        self.emb_dim   = args['emb_dim']
        self.out_dim   = args['output_dim']

        self.gcn1      = TAGConv(self.emb_dim, self.emb_dim, k=2)
        self.fc_g       = nn.Linear(self.emb_dim, self.out_dim)
        self.maxpooling = MaxPooling()

        self.textcnn    = ConvsLayer(self.emb_dim)
        self.textflatten = nn.Linear(128, self.out_dim)

        self.w1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.gate_gc   = nn.Linear(self.out_dim * 2, self.out_dim)
        self.residual_fc = nn.Linear(self.out_dim * 2, self.out_dim * 2)

        self.fc1 = nn.Linear(self.out_dim * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, 1)

    def _encode(self, G, pad_dmap):
        g = F.relu(self.gcn1(G, G.ndata['feat']))
        g = g.reshape(-1, self.emb_dim)
        G.ndata['feat'] = g
        g = self.maxpooling(G, G.ndata['feat'])
        g = F.relu(self.fc_g(g))

        seq = F.relu(self.textflatten(self.textcnn(pad_dmap)))

        w1 = F.sigmoid(self.w1)
        return (1 - w1) * g + w1 * seq

    def forward(self, G1, pad_dmap1, G2, pad_dmap2):
        gc1 = self._encode(G1, pad_dmap1)
        gc2 = self._encode(G2, pad_dmap2)

        gate_in = torch.cat([gc1, gc2], dim=1)
        gate = torch.sigmoid(self.gate_gc(gate_in))
        gc = torch.cat([gate * gc1, (1 - gate) * gc2], dim=1)
        gc = gc + self.residual_fc(gate_in)

        gc = F.relu(self.bn1(self.fc1(gc)))
        gc = F.relu(self.bn2(self.fc2(gc)))
        return F.sigmoid(self.out(gc))
