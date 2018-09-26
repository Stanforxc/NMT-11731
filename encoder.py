import torch
import torch.nn as nn
import generalRnn

class Encoder(generalRnn.BaseCoder):
    def __init__(self,vocab_size, max_length, hidden_size, embedding_size, dropout=0.0, n_layers=1, bidirectional=True,rnn="lstm"):
        super(Encoder, self).__init__(vocab_size, max_length, hidden_size,embedding_size,
                dropout, n_layers, rnn)
        self.embedding = nn.Embedding(vocab_size,embedding_size)

        self.rnn = self.baseModel(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers, 
                    batch_first=True,dropout=(0 if n_layers == 1 else dropout), bidirectional=bidirectional)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded)
        # embedded = self.embedding(input_seq).view(1, batch_size,self.embed_size)
        # output = embedded
        # for i in range(self.num_layers):
        #     output, hidden = self.lstm(output,hidden)
        # packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # outputs, hidden = self.embedding(packed, hidden)
        # outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return output, hidden

    # def initHidden(self, batch_size):
    #     result = torch.zeros(1, batch_size, self.embed_size)
    #     return result