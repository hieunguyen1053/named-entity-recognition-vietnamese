import torch
import torch.nn as nn

class HMM_NER(nn.Module):
    def __init__(self, vocab, tag_map, emiss, trans):
        super(HMM_NER, self).__init__()
        self.vocab = vocab
        self.tag_map = tag_map
        self.emiss = emiss
        self.trans = trans

    def forward(self, seq):
        first_word = seq[0]
        begin_tag = torch.tensor(0, dtype=torch.int32)

        emiss_0 = self.emiss.index_select(1, first_word).flatten()
        trans_0 = self.trans.index_select(0, begin_tag).flatten()
        prob_0 = emiss_0.mul(trans_0)
        probs = [prob_0]

        for word in seq[1:]:
            prev_prob = probs[-1]
            curr_prob = torch.zeros((1, emiss_0.size(0)), dtype=torch.float64).flatten()

            for i in range(emiss_0.size(0)):
                prob_i = torch.zeros((1, prob_0.size(0)), dtype=torch.float64).flatten()
                for j in range(prob_0.size(0)):
                    prob_i[j] = prev_prob[j] * self.trans[j, i] * self.emiss[i, word]
                curr_prob[i] = torch.max(prob_i)
            if torch.max(curr_prob) == 0:
                curr_prob[self.tag_map.stoi['O']] = 1.0
            probs.append(curr_prob)

        hidden_state_seq = []
        for prob in probs:
            state = prob.argmax()
            hidden_state_seq.append(state)
        return torch.tensor(hidden_state_seq, dtype=torch.int32)

    def viterbi(self, sent):
        seq = torch.tensor([self.vocab.stoi[word] for word in sent], dtype=torch.int32)
        hidden_state_seq = self.forward(seq)
        tags = [self.tag_map.itos[state] for state in hidden_state_seq]
        return tags