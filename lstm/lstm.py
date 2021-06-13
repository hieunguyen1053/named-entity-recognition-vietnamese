import torch
import torch.functional as F
import torch.nn as nn

from constants import *


class LSTM_NER(nn.Module):
    def __init__(self, vocab_size, output_size, embed_size=128, hidden_size=128, num_layers=3):
        super(LSTM_NER, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size, PAD_IDX)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=0.2)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embed = self.embedding(input)
        output, hidden = self.lstm(embed, hidden)
        logits = self.linear(output)
        return logits, hidden

    def init_hidden(self, seq_len):
        return (torch.zeros(self.num_layers, seq_len, self.hidden_size),
                torch.zeros(self.num_layers, seq_len, self.hidden_size))

    def predict(self, input, tag_map):
        self.eval()
        hidden = self.init_hidden(input.size(0))
        pred, _ = self.forward(input.unsqueeze(0), hidden)
        tags = [tag_map.itos[torch.argmax(pred[0][i])] for i in range(pred.size(1))]
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
        params['vocab_size'] = self.vocab_size
        params['output_size'] = self.output_size
        params['embed_size'] = self.embed_size
        params['hidden_size'] = self.hidden_size
        params['num_layers'] = self.num_layers

        params['embedding'] = self.embedding.state_dict()
        params['lstm'] = self.lstm.state_dict()
        params['linear'] = self.linear.state_dict()
        torch.save(params, filepath)

    @classmethod
    def load(cls, filepath):
        params =  torch.load(filepath)
        vocab_size = params['vocab_size']
        output_size = params['output_size']
        embed_size = params['embed_size']
        hidden_size = params['hidden_size']
        num_layers = params['num_layers']

        model = cls(vocab_size, output_size, embed_size, hidden_size, num_layers)
        model.embedding.load_state_dict(params['embedding'])
        model.lstm.load_state_dict(params['lstm'])
        model.linear.load_state_dict(params['linear'])
        return model