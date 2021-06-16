from collections import Counter, defaultdict
from math import exp, log

import torch
from torchtext.vocab import Vocab

def build_vocab_and_label(sentences):
    words = []
    ners = []
    for sent in sentences:
        items = sent.split('\n')
        for item in items:
            word, _, _, tag = item.split('\t')
            word = '_'.join(word.split())
            words.append(word)
            ners.append(tag)
    return Vocab(Counter(words)), Vocab(Counter(ners), specials=('#'))

def build_emiss_matrix(sentences, vocab, label):
    tag_to_count = defaultdict(lambda: 0)
    emiss_to_count = defaultdict(lambda: 0)

    for sent in sentences:
        items = sent.split('\n')
        for item in items:
            word, _, _, tag = item.split('\t')
            word = '_'.join(word.split())
            word_idx = vocab.stoi[word]
            tag_idx = label.stoi[tag]
            tag_to_count[tag_idx] += 1
            emiss_to_count[(tag_idx, word_idx)] += 1

    emiss_matrix = torch.zeros((len(label)), len(vocab))
    for (tag, word) in emiss_to_count:
        count = emiss_to_count[(tag, word)]
        emiss_matrix[tag][word] = exp(log(count) - log(tag_to_count[tag]))
    return emiss_matrix

def build_trans_matrix(sentences, label):
    tag_to_count = defaultdict(lambda: 0)
    trans_to_count = defaultdict(lambda: 0)

    for sent in sentences:
        items = sent.split('\n')
        prev_tag = "#"
        prev_tag_idx = label.stoi[prev_tag]
        tag_to_count[prev_tag_idx] += 1
        for item in items:
            _, _, _, curr_tag = item.split('\t')
            curr_tag_idx = label.stoi[curr_tag]
            tag_to_count[curr_tag_idx] += 1
            trans_to_count[(prev_tag_idx, curr_tag_idx)] += 1
            prev_tag_idx = curr_tag_idx

    trans_matrix = torch.zeros((len(label)), (len(label)))
    for (pre_tag, curr_tag) in trans_to_count:
        count = trans_to_count[(pre_tag, curr_tag)]
        trans_matrix[pre_tag][curr_tag] = exp(log(count) - log(tag_to_count[pre_tag]))
    return trans_matrix

if __name__ == '__main__':
    data = open('/Users/hieunguyen/Desktop/NLP/Master/NER/data/vlsp2016/train.txt', 'r', encoding='utf-8').read()
    sents = data.split('\n\n')
    data = open('/Users/hieunguyen/Desktop/NLP/Master/NER/data/vlsp2016/dev.txt', 'r', encoding='utf-8').read()
    sents += data.split('\n\n')

    params = {}
    vocab, label = build_vocab_and_label(sents)
    emiss_matrix = build_emiss_matrix(sents, vocab, label)
    trans_matrix = build_trans_matrix(sents, label)
    params['vocab'] = vocab
    params['label'] = label
    params['emiss'] = emiss_matrix
    params['trans'] = trans_matrix
    torch.save(params, 'data.pt')
