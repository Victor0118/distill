import torch
import torch.nn as nn
import torch.nn.functional as F

from util import init_embedding, fetch_embedding


class BiRNNModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        fc_size = config.fc_size
        self.rnn_type = config.rnn_type
        self.use_maxpool = config.use_maxpool
        init_embedding(self, config)

        if self.rnn_type.upper() == "LSTM":
            self.bi_rnn = nn.LSTM(self.embedding_dim, self.hidden_size, 1, batch_first=True, 
                bidirectional=True)
        elif self.rnn_type.upper() == "GRU":
            self.bi_rnn = nn.GRU(self.embedding_dim, self.hidden_size, 1, batch_first=True, bidirectional=True)
        else:
            raise ValueError("RNN type must be one of LSTM or GRU")
        self.fc1 = nn.Linear(2 * self.hidden_size, fc_size)
        #self.fc2 = nn.Linear(fc_size, config.dataset.N_CLASSES)
        #self.dropout = nn.Dropout(config.dropout)
        self.mode = config.mode

    def non_embedding_params(self):
        params = []
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                continue
            params.extend(p for p in m.parameters() if p.dim() == 2)
        return params

    def forward(self, x):
        x = fetch_embedding(self, self.mode, x, squash=True)
        rnn_seq, rnn_out = self.bi_rnn(x)
        if self.rnn_type.upper() == "LSTM":
            rnn_out = rnn_out[0]
        #if self.use_maxpool:
        #    rnn_out = rnn_seq.max(1)[0]

        # print(rnn_out.size())
        rnn_out = rnn_out.permute(1, 0, 2)
        rnn_out = rnn_out.contiguous().view(rnn_out.size()[0], rnn_out.size()[1] * rnn_out.size()[2])
        x = F.relu(self.fc1(rnn_out))
        #x = self.dropout(x)
        #logits = self.fc2(x)
        logits = x
        return logits
