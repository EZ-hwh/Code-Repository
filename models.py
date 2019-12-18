import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embed_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embed_dim = self.embed_dim
        self.hidden_dim = self.hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        #Map the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tarset_size)

        #Matrix of transition parameters. Entry i,j is the score of 
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
