# -*- coding: utf-8 -*-
"""Sequence to embedding vector (via pre-trained embedding dictionary)."""
import numpy as np


class s2t:
    """Load embedding dict from a TSV file (token \t vector), then embed sequences."""

    def __init__(self, filename):
        self.t2v = {}
        self.dim = None
        for line in open(filename):
            line = line.strip('\n').split('\t')
            t = line[0]
            v = np.array([float(x) for x in line[1].split()])
            if self.dim is None:
                self.dim = len(v)
            else:
                v = v[:self.dim]
            self.t2v[t] = v

    def embed(self, seq):
        """Return numpy array (L, D) of embedding vectors. Space-separated or char-by-char."""
        s = seq.strip().split() if seq.find(' ') > 0 else list(seq.strip())
        return np.array([self.t2v[x] for x in s if x in self.t2v])

    def embed_normalized(self, seq, length=1200):
        """Return fixed-shape (length, dim) tensor. Truncate or pad with zeros."""
        rst = self.embed(seq)
        if len(rst) > length:
            return rst[:length]
        if len(rst) < length:
            return np.concatenate((rst, np.zeros((length - len(rst), self.dim))))
        return rst
