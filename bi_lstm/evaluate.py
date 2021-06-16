import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bi_lstm import BiLSTM_NER
from constants import *
from dataset import NerDataset


def collate_fn(samples):
    samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
    sentences = [x[0] for x in samples]
    tags = [x[1] for x in samples]
    return sentences, tags


def padding(sents, pad_idx, device):
    lengths = [len(sent) for sent in sents]
    max_len = lengths[0]
    padded_data = []
    for s in sents:
        padded_data.append(s.tolist() + [pad_idx] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths

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


if __name__ == '__main__':
    params = torch.load('/Users/hieunguyen/Desktop/NLP/Master/NER/bi_lstm_crf/vocab_label.pt')
    vocab = params['vocab']
    label = params['label']
    test_dataset = NerDataset('/Users/hieunguyen/Desktop/NLP/Master/NER/data/vlsp2016/test.txt', vocab, label)
    batch_size = 1
    test_iter = DataLoader(test_dataset, batch_size, collate_fn=collate_fn)

    device = torch.device("cpu")
    model = BiLSTM_NER.load('/Users/hieunguyen/Desktop/NLP/Master/NER/lstm/ulstm_ner_8.pt')

    y_test = []
    y_pred = []

    for idx, (sentences, tags) in enumerate(tqdm(test_iter)):
        model.eval()
        sentences, sent_lengths = padding(sentences, model.tag_vocab.stoi[PAD], device)

        pred_tags = model(sentences, sent_lengths)
        pred_tags = torch.argmax(pred_tags.squeeze(0), dim=1)

        pred_tags = model.iob_tag(pred_tags.tolist()[1:-1])
        tags = model.iob_tag(tags[0].tolist()[1:-1])

        y_pred.append(pred_tags)
        y_test.append(tags)
    print(f_measure(y_test, y_pred))
