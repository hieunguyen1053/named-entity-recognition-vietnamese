import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from collections import Counter
from constants import *

class NerDataset(Dataset):
    def __init__(self, filepath, vocab=None, label=None):
        self.data = None
        self.vocab = vocab
        self.label = label

        self.read_data(filepath)

    def read_data(self, filepath):
        all_words = []
        all_ners = []

        sents = open(filepath, 'r', encoding='utf-8').read().split('\n\n')
        for sent in sents:
            items = sent.split('\n')
            for item in items:
                w, _, _, n = item.split('\t')
                if n[2:] in ['MISC', 'LOC', 'ORG', 'PER']:
                    n = n[2:]
                w = '_'.join(w.split())
                all_words.append(w)
                all_ners.append(n)
        if self.vocab is None or self.label is None:
            self.vocab = Vocab(Counter(all_words), specials=(PAD, UNK, BOS, EOS), specials_first=False)
            self.label = Vocab(Counter(all_ners), specials=(BOS, EOS, PAD), specials_first=False)
            self.vocab.unk_index = self.vocab.stoi[UNK]
            self.label.unk_index = self.label.stoi['O']

        X = []
        Y = []
        for sent in sents:
            x = [self.vocab.stoi[BOS]]
            y = [self.label.stoi[BOS]]
            items = sent.split('\n')
            for item in items:
                w, _, _, n = item.split('\t')
                w = '_'.join(w.split())
                if n[2:] in ['MISC', 'LOC', 'ORG', 'PER']:
                    n = n[2:]
                x.append(self.vocab.stoi[w])
                y.append(self.label.stoi[n])
            x.append(self.vocab.stoi[BOS])
            y.append(self.label.stoi[EOS])
            X.append(torch.tensor(x))
            Y.append(torch.tensor(y))

        self.data = []
        for x, y in zip(X, Y):
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]