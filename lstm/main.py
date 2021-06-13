import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from constants import *
from dataset import NerDataset
from lstm import LSTM_NER
from train import train

train_dataset = NerDataset('/Users/hieunguyen/Desktop/NLP/Master/NER/data/vlsp2016/train.txt')
vocab = train_dataset.vocab
label = train_dataset.label
print(label.stoi)
val_dataset = NerDataset('/Users/hieunguyen/Desktop/NLP/Master/NER/data/vlsp2016/dev.txt', vocab, label)
vocab_size = len(train_dataset.vocab)
output_size = len(train_dataset.label)
batch_size = 64

train_iter = DataLoader(train_dataset, batch_size)
val_iter = DataLoader(val_dataset, batch_size)

model = LSTM_NER(vocab_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

device = torch.device("cpu")
writer = SummaryWriter('runs/lstm_ner')
train(model, optimizer, criterion, writer, train_iter, val_iter, device, epochs=15, resume=True)
