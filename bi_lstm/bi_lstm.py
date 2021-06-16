import torch
import torch.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from constants import *


class BiLSTM_NER(nn.Module):
    def __init__(self, sent_vocab, tag_vocab, embed_dim=300, hidden_dim=300, num_layers=3):
        super(BiLSTM_NER, self).__init__()
        self.sent_vocab = sent_vocab
        self.tag_vocab = tag_vocab
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(len(sent_vocab), embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            bidirectional=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim * 2, len(tag_vocab))

    def forward(self, sentences, sent_lengths):
        sentences = sentences.transpose(0, 1)
        sentences = self.embedding(sentences)
        padded_sentences = pack_padded_sequence(sentences, sent_lengths, enforce_sorted=False)
        lstm_out, _ = self.lstm(padded_sentences)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        logits = self.linear(lstm_out)
        return logits

    def iob_tag(self, tags):
        tags = [self.tag_vocab.itos[tag] for tag in tags]
        prev_tag = 'O'
        for idx, curr_tag in enumerate(tags):
            if curr_tag != 'O' and prev_tag == 'O':
                tags[idx] = 'B-' + curr_tag
            if curr_tag == prev_tag and prev_tag != 'O':
                tags[idx] = 'I-' + curr_tag
            prev_tag = curr_tag
        return tags

    def save(self, filepath):
        params = {}
        params['sent_vocab'] = self.sent_vocab
        params['tag_vocab'] = self.tag_vocab
        params['embed_dim'] = self.embed_dim
        params['hidden_dim'] = self.hidden_dim

        params['embedding'] = self.embedding.state_dict()
        params['lstm'] = self.lstm.state_dict()
        params['linear'] = self.linear.state_dict()
        torch.save(params, filepath)

    @classmethod
    def load(cls, filepath):
        params = torch.load(filepath, map_location=torch.device('cpu'))
        sent_vocab = params['sent_vocab']
        tag_vocab = params['tag_vocab']
        embed_dim = params['embed_dim']
        hidden_dim = params['hidden_dim']

        model = cls(sent_vocab, tag_vocab, embed_dim, hidden_dim)
        model.embedding.load_state_dict(params['embedding'])
        model.lstm.load_state_dict(params['lstm'])
        model.linear.load_state_dict(params['linear'])
        return model

    @property
    def device(self):
        return self.embedding.weight.device
