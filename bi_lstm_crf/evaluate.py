import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from bi_lstm_crf import BiLSTM_CRF_NER
from constants import *
from dataset import NerDataset


def _get_tags(sents):
    tags = []
    for sent_idx, iob_tags in enumerate(sents):
        curr_tag = {'type': None, 'start_idx': None,
                    'end_idx': None, 'sent_idx': None}
        for i, tag in enumerate(iob_tags):
            if tag == 'O' and curr_tag['type']:
                tags.append(tuple(curr_tag.values()))
                curr_tag = {'type': None, 'start_idx': None,
                            'end_idx': None, 'sent_idx': None}
            elif tag.startswith('B'):
                curr_tag['type'] = tag[2:]
                curr_tag['start_idx'] = i
                curr_tag['end_idx'] = i
                curr_tag['sent_idx'] = sent_idx
            elif tag.startswith('I'):
                curr_tag['end_idx'] = i
        if curr_tag['type']:
            tags.append(tuple(curr_tag.values()))
    tags = set(tags)
    return tags


def f_measure(y_true, y_pred):
    tags_true = _get_tags(y_true)
    tags_pred = _get_tags(y_pred)

    ne_ref = len(tags_true)
    ne_true = len(set(tags_true).intersection(tags_pred))
    ne_sys = len(tags_pred)
    if ne_ref == 0 or ne_true == 0 or ne_sys == 0:
        return 0
    p = ne_true / ne_sys
    r = ne_true / ne_ref
    f1 = (2 * p * r) / (p + r)

    return f1

def collate_fn(samples):
    samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
    sentences = [x[0] for x in samples]
    tags = [x[1] for x in samples]
    return sentences, tags

if __name__ == '__main__':
    torch.seed()
    train_dataset = NerDataset('/Users/hieunguyen/Desktop/NLP/Master/NER/data/vlsp2016/train.txt')
    vocab = train_dataset.vocab
    label = train_dataset.label

    test_dataset = NerDataset('/Users/hieunguyen/Desktop/NLP/Master/NER/data/vlsp2016/test.txt', vocab, label)
    batch_size = 1
    test_iter = DataLoader(test_dataset, batch_size, collate_fn=collate_fn)

    device = torch.device("cpu")
    model = BiLSTM_CRF_NER.load('/Users/hieunguyen/Desktop/NLP/Master/NER/bi_lstm_crf/checkpoints/lstm_ner_8.pt')

    y_test = []
    y_pred = []

    for idx, (sentences, tags) in tqdm(enumerate(test_iter), total=len(test_iter)):
        tags, _ = model.padding_sents(tags)

        pred_tags = model.predict(sentences)

        pred_tags = model.iob_tag(pred_tags[0][1:-1])
        tags = model.iob_tag(tags[0].tolist()[1:-1])

        y_pred.append(pred_tags)
        y_test.append(tags)

    print(f_measure(y_test, y_pred))
