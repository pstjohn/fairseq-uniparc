import numpy as np
import torch
from fairseq.data import FairseqDataset


class CSRLabelDataset(FairseqDataset):
    def __init__(self, csr_label_matrix):
        super().__init__()
        self.csr_label_matrix = csr_label_matrix

    def __getitem__(self, index):
        return self.csr_label_matrix[index].todense()

    def __len__(self):
        return self.csr_label_matrix.shape[0]

    def collater(self, samples):
        return torch.tensor(np.vstack(samples))