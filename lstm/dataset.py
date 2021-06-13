import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from constants import *

class NerDataset(Dataset):
    def __init__(self, filepath, vocab=None, tag_map=None):
        self.data = None
        self.vocab = vocab
        self.tag_map = tag_map

        self.read_data(filepath)

    def read_data(self, filepath):
        all_words = []
        all_ners = []

        sents = open(filepath, 'r', encoding='utf-8').read().split('\n\n')
        for sent in sents:
            items = sent.split('\n')
            for item in items:
                w, _, _, n = item.split('\t')
                # O, B-PER, I-PER, B-MISC, I-MISC, B-LOC, B-ORG, I-ORG, I-LOC
                if n[2:] in ['MISC', 'LOC', 'ORG', 'PER']:
                    n = n[2:]
                all_words.append(w)
                all_ners.append(n)
        if self.vocab is None or self.tag_map is None:
            self.vocab = Vocab(Counter(all_words), specials=(PAD, UNK))
            self.tag_map = Vocab(Counter(all_ners), specials=(PAD, UNK))

        X = []
        Y = []
        for sent in sents:
            x = []
            y = []
            items = sent.split('\n')
            for item in items:
                w, _, _, n = item.split('\t')
                if n[2:] in ['MISC', 'LOC', 'ORG', 'PER']:
                    n = n[2:]
                x.append(self.vocab.stoi[w])
                y.append(self.tag_map.stoi[n])
            X.append(torch.tensor(x))
            Y.append(torch.tensor(y))
        X = pad_sequence(X, batch_first=True, padding_value=PAD_IDX)
        Y = pad_sequence(Y, batch_first=True, padding_value=PAD_IDX)

        self.data = []
        for x, y in zip(X, Y):
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]