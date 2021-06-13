import glob
import os.path as path

import pkbar


def train_epoch(model, optimizer, criterion, train_iter, device, probar):
    model.train()
    model = model.to(device)

    max_len = len(train_iter.dataset[0][1])
    hidden = model.init_hidden(max_len)
    epoch_loss = 0

    for idx, (seq, label) in enumerate(train_iter):
        seq = seq.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred, hidden = model(seq, hidden)
        loss = criterion(pred.transpose(1, 2), label)

        hidden0, hidden1 = hidden
        hidden0 = hidden0.detach()
        hidden1 = hidden1.detach()
        hidden = (hidden0, hidden1)
        loss.backward()
        optimizer.step()

        probar.update(idx, values=[('loss', loss),])

        epoch_loss += loss.item()
    return epoch_loss / len(train_iter)

def evaluate(model, criterion, val_iter, device):
    model.eval()
    model = model.to(device)

    max_len = len(val_iter.dataset[0][1])
    hidden = model.init_hidden(max_len)
    losses = 0

    for idx, (seq, label) in enumerate(val_iter):
        seq = seq.to(device)
        label = label.to(device)

        pred, hidden = model(seq, hidden)
        loss = criterion(pred.transpose(1, 2), label)
        losses += loss.item()
    return losses / len(val_iter)

def train(model, optimizer, criterion, writer, train_iter, val_iter, device, epochs, checkpoint_folder='./checkpoints', save_freq=1, resume=False):
    start_iter = 0
    if resume:
        model_list = glob.glob(path.join(checkpoint_folder, '*.pt'))
        if len(model_list) != 0:
            model_list.sort(reverse=True)
            start_iter = int(model_list[0].split('_')[-1].split('.')[0])
        model = model.load(path.join(checkpoint_folder, 'lstm_ner_%s.pt' % start_iter))

    batch_per_epoch = len(train_iter)
    for epoch in range(start_iter+1, epochs+1):
        probar = pkbar.Kbar(target=batch_per_epoch, epoch=epoch-1, num_epochs=epochs, width=30, always_stateful=False)
        train_loss = train_epoch(model, optimizer, criterion, train_iter, device, probar)
        val_loss = evaluate(model, criterion, val_iter, device)
        probar.add(1, values=[('train_loss', train_loss), ('val_loss', val_loss),])
        writer.add_scalar('training loss',
                            train_loss,
                            epoch * len(train_iter) + batch_per_epoch)
        writer.add_scalar('validation loss',
                            val_loss,
                            epoch * len(val_iter) + batch_per_epoch)
        if epoch % save_freq == 0:
            model.save(path.join(checkpoint_folder, 'lstm_ner_%s.pt' % epoch))
