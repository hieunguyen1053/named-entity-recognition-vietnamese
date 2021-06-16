import glob
import os.path as path

import pkbar
import torch

from constants import *


def padding(sents, pad_idx, device):
    lengths = [len(sent) for sent in sents]
    max_len = lengths[0]
    padded_data = []
    for s in sents:
        padded_data.append(s.tolist() + [pad_idx] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths


def train_epoch(model, optimizer, train_dataset, device, probar):
    model.train()
    model = model.to(device)

    epoch_loss = 0

    for idx, (sentences, tags) in enumerate(train_dataset):
        sentences, sent_lengths = padding(sentences, model.sent_vocab.stoi[PAD], device)
        tags, _ = padding(tags, model.tag_vocab.stoi[PAD], device)

        optimizer.zero_grad()
        batch_loss = model(sentences, tags, sent_lengths)
        loss = batch_loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        probar.update(idx, values=[('loss', loss),])

        epoch_loss += loss.item()
    return epoch_loss / len(train_dataset)

def evaluate(model, val_dataset, device):
    model.eval()
    model = model.to(device)

    losses = 0

    for idx, (sentences, tags) in enumerate(val_dataset):
        sentences, sent_lengths = padding(sentences, model.sent_vocab.stoi[PAD], device)
        tags, _ = padding(tags, model.tag_vocab.stoi[PAD], device)

        batch_loss = model(sentences, tags, sent_lengths)
        loss = batch_loss.mean()
        losses += loss.item()
    return losses / len(val_dataset)

def train(model, optimizer, writer, train_dataset, val_dataset, device, epochs, checkpoint_folder='./checkpoints', save_freq=1, resume=False):
    start_iter = 0
    if resume:
        model_list = glob.glob(path.join(checkpoint_folder, '*.pt'))
        if len(model_list) != 0:
            model_list.sort(reverse=True)
            start_iter = int(model_list[0].split('_')[-1].split('.')[0])
        model = model.load(path.join(checkpoint_folder, 'lstm_ner_%s.pt' % start_iter))

    batch_per_epoch = len(train_dataset)
    for epoch in range(start_iter+1, epochs+1):
        probar = pkbar.Kbar(target=batch_per_epoch, epoch=epoch-1, num_epochs=epochs, width=30, always_stateful=False)
        train_loss = train_epoch(model, optimizer, train_dataset, device, probar)
        val_loss = evaluate(model, val_dataset, device)

        probar.add(1, values=[('train_loss', train_loss), ('val_loss', val_loss),])
        writer.add_scalar('training loss',
                            train_loss,
                            epoch * len(train_dataset) + batch_per_epoch)
        writer.add_scalar('validation loss',
                            val_loss,
                            epoch * len(val_dataset) + batch_per_epoch)
        if epoch % save_freq == 0:
            model.save(path.join(checkpoint_folder, 'lstm_ner_%s.pt' % epoch))
