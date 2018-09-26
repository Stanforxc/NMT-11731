import torch
import torch.nn as nn
import torch.nn.functional as F
from generalRnn import BaseCoder

import numpy as np

class Decoder(BaseCoder):
    def __init__(self, vocab_size, hidden_size, embedding_size, dropout=0.0, n_layers=1, bidirectional=False,attention=False,rnn="lstm"):
        super(Decoder,self).__init__(vocab_size, hidden_size,embedding_size,
                dropout, n_layers, rnn)
        self.bidirectional = bidirectional
        self.rnn = self.baseModel(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers, 
                    batch_first=True,dropout=(0 if n_layers == 1 else dropout))
        self.output_size = vocab_size
        # self.max_length = max_length
        self.attention_usage = attention
        # self.start_of_sent = 1
        # self.end_of_sent = 2
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.wsm = nn.Linear(self.hidden_size, self.output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, encoder_hidden, encoder_outputs, func=F.log_softmax):
        batch_size = input_seq.size(0)
        max_length = input_seq.size(1)

        # using cuda or not
        inputs = input_seq
        
        # for bidrectional encoder
        # encoder_hidden: (num_layers * num_directions, batch_size, hidden_size)
        h = encoder_hidden
        decoder_hidden = tuple([torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) for h in encoder_hidden])

        outputs = []
        # lengths = np.array([max_length] * batch_size)

        prev = inputs[:, 0].unsqueeze(1)
        for i in range(max_length):
            softmax, decoder_hidden = self.forward_helper(prev, decoder_hidden,func)
            output_seq = softmax.squeeze(1) # batch * seq_length
            outputs.append(output_seq)
            # maybe beam search here
            prev = output_seq.topk(1)[1] # max probability index

        return outputs,decoder_hidden
            


    # no attention now
    # could insert one parameter like: src_matrix
    def forward_helper(self, decoder_input, decoder_hidden, func):
        batch_size = decoder_input.size(0)
        output_size = decoder_input.size(1)
        embedded = self.embedding(decoder_input)
        output,hidden = self.rnn(embedded, decoder_hidden)
        softmax = func(self.wsm(output.view(-1, self.hidden_size)), dim=1).view(batch_size,output_size,-1)
        return softmax, hidden
