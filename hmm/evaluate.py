import torch
from tqdm import tqdm

from hmm import HMM_NER


def create_tags(sentences):
    tags = []
    for sentence_index, iob_tags in enumerate(sentences):
        i = 0
        current_tag = [None, None, None, None]
        while i < len(iob_tags):
            if iob_tags[i] == "O" and current_tag[0]:
                tags.append(tuple(current_tag))
                current_tag = [None, None, None, None]
            elif iob_tags[i].startswith("B"):
                current_tag[0] = iob_tags[i][2:]
                current_tag[1] = i
                current_tag[2] = i
                current_tag[3] = sentence_index
            elif iob_tags[i].startswith("I"):
                current_tag[2] = i
            i += 1
        if current_tag[0]:
            tags.append(tuple(current_tag))
    tags = set(tags)
    return tags


def iob_score(y_true, y_pred):
    tags_true = create_tags(y_true)
    tags_pred = create_tags(y_pred)
    ref = len(tags_true)
    true = len(set(tags_true).intersection(tags_pred))
    sys = len(tags_pred)
    if ref == 0 or true == 0 or sys == 0:
        return 0
    p = true / sys
    r = true / ref
    f1 = (2 * p * r) / (p + r)

    return f1

def evaluate(model, filepath):
    y_test = []
    y_pred = []

    data = open(filepath, 'r', encoding='utf-8').read()
    sents = data.split('\n\n')
    for sent in tqdm(sents):
        x = []
        y = []
        items = sent.split('\n')
        for item in items:
            w, _, _, n = item.split('\t')
            x.append(w)
            y.append(n)
        y_pred.append(model.viterbi(x))
        y_test.append(y)
        print(y_pred[-1])
        print(y_test[-1])

    # print(classification_report(y_test, y_pred, labels=['B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'O']))
    print(iob_score(y_test, y_pred))


if __name__ == '__main__':
    params = torch.load('./data.pt')
    vocab = params['vocab']
    label = params['label']
    emiss = params['emiss']
    trans = params['trans']
    model = HMM_NER(vocab, label, emiss, trans)
    evaluate(model, '/Users/hieunguyen/Desktop/NLP/Master/NER/data/vlsp2016/test.txt')
