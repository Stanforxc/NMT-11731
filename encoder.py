import torch
import torch.nn as nn
import generalRnn

class Encoder(generalRnn.BaseCoder):
    def __init__(self,vocab_size, hidden_size, embedding_size, dropout=0.0, n_layers=1, bidirectional=True,rnn="lstm"):
        super(Encoder, self).__init__(vocab_size, hidden_size,embedding_size,
                dropout, n_layers, rnn)
        self.embedding = nn.Embedding(vocab_size,embedding_size)

        self.rnn = self.baseModel(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers, 
                    batch_first=True,dropout=(0 if n_layers == 1 else dropout), bidirectional=bidirectional)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded)
        return output, hidden