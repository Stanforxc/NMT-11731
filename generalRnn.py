import torch.nn as nn

class BaseCoder(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_size, embedding_size, dropout, n_layers, rnn):
        super(BaseCoder, self).__init__()
        # init ...
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.n_layers = n_layers


        if rnn.lower() == "lstm":
            self.baseModel = nn.LSTM
        elif rnn.lower() == "gru":
            self.baseModel = nn.GRU
        else:
            ## raise error
            raise ValueError("No such cell!")
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError