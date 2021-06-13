import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from constants import *
from dataset import NerDataset
from train import train
from bi_lstm_crf import BiLSTM_CRF_NER

def collate_fn(samples):
    samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
    sentences = [x[0] for x in samples]
    tags = [x[1] for x in samples]
    return sentences, tags

train_dataset = NerDataset('/Users/hieunguyen/Desktop/NLP/Master/NER/data/vlsp2016/train.txt')
vocab = train_dataset.vocab
label = train_dataset.label
val_dataset = NerDataset('/Users/hieunguyen/Desktop/NLP/Master/NER/data/vlsp2016/dev.txt', vocab, label)

batch_size = 32

train_iter = DataLoader(train_dataset, batch_size, collate_fn=collate_fn)
val_iter = DataLoader(val_dataset, batch_size, collate_fn=collate_fn)

model = BiLSTM_CRF_NER(vocab, label)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

device = torch.device("cpu")
writer = SummaryWriter('runs/lstm_ner')
train(model, optimizer, writer, train_iter, val_iter, device, epochs=10, resume=False)
