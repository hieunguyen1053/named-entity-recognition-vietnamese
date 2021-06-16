import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from constants import *
from dataset import NerDataset
from bi_lstm import BiLSTM_NER
from train import train

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


model = BiLSTM_NER(vocab, label)
for name, param in model.named_parameters():
    if 'weight' in name:
        nn.init.normal_(param.data, 0, 0.01)
    else:
        nn.init.constant_(param.data, 0)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = criterion = nn.CrossEntropyLoss(ignore_index=model.tag_vocab.stoi[PAD])

device = torch.device("cpu")
writer = SummaryWriter('runs/lstm_ner')
train(model, optimizer, criterion, writer, train_iter, val_iter, device, epochs=20)
