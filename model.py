import numpy as np
import torch
import torch.utils.data
from torch import nn
import os
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import sys
import pickle
from utils import read_corpus
from vocab import Vocab, VocabEntry
from nltk.translate.bleu_score import corpus_bleu


# **Encoder**
# is a BiLSTM. Each has 256 hidden dimensions per direction.
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, attention_dim, value_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.BLSTM = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True,
                             num_layers=1)

        self.key_linear = nn.Linear(hidden_dim * 2, attention_dim)  # output from bLSTM
        self.value_linear = nn.Linear(hidden_dim * 2, value_dim)  # output from bLSTM

    def forward(self, input, src_lens):
        # print(input.size())
        # print(src_lens)
        embeddings = self.embedding(input)
        # embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, src_lens, batch_first=True)
        output, h = self.BLSTM(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # print(output.size())

        key = ApplyPerTime(self.key_linear, output).transpose(1, 2)
        value = ApplyPerTime(self.value_linear, output)  # (N, L, 128)
        return key, value


# **Decoder**
# - Single embedding layer from input characters to hidden dimension.
# - Input to LSTM cells is previous context, previous states, and current character.
# - 3 LSTM cells
# - On top is a linear layer for the query projection (done in Attention class)
# - The results of the query and the LSTM state are passed into a single hidden layer MLP for the character projection
# - The last layer of the character projection and the embedding layer have tied weights
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, attention_dim, value_dim, tf_rate):
        # assert decoder_hidden_dim == listener_hidden_dim
        super(Decoder, self).__init__()
        concat_dim = hidden_dim + value_dim

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.cell1 = nn.LSTMCell(input_size=concat_dim, hidden_size=hidden_dim)
        # self.cell2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        # self.cell3 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.attention = Attention(attention_dim, hidden_dim) # 128, 256

        # character projection
        self.mlp = nn.Linear(concat_dim, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.character_projection = nn.Linear(hidden_dim, vocab_size)

        # tie embedding and project weights
        self.character_projection.weight = self.embedding.weight

        self.softmax = nn.LogSoftmax(dim=-1)  # todo: remove??

        self.hidden_dim = hidden_dim

        # initial states
        self.h00 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, self.hidden_dim).type(torch.FloatTensor)), requires_grad=True)
        self.c00 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, self.hidden_dim).type(torch.FloatTensor)), requires_grad=True)
        self.tf_rate = tf_rate

    # listener_feature (N, T, 256)
    # Yinput (N, L )
    def forward(self, key, value, Yinput, max_len, mode, src_lens):

        # Create a binary mask for attention (N, L)
        src_lens = np.array(src_lens)
        attention_mask = np.zeros((len(src_lens), 1, np.max(src_lens)))  # N, 1, L
        for i in range(len(src_lens)):
            attention_mask[i, 0, :src_lens[i]] = np.ones(src_lens[i])
        attention_mask = to_variable(to_tensor(attention_mask))

        # INITIALIZATION
        batch_size = key.size()[0]  # train: N; test: 1

        _, context = self.attention(key, value, self.h00.expand(batch_size, self.hidden_dim).contiguous(), attention_mask)
        # common initial hidden and cell states for LSTM cells
        prev_h = self.h00.expand(batch_size, self.hidden_dim).contiguous()
        prev_c = self.c00.expand(batch_size, self.hidden_dim).contiguous()

        pred_seq = None
        pred_idx = to_variable(torch.zeros(batch_size).long())  # size [N] batch size = 1 for test
        # print(pred_idx.size())

        # if dont want to base on ref tgt
        if max_len is None:
            max_len = 100

        for step in range(max_len):

            # teacher force rate: prob to feed the ground truth as input
            teacher_force = True if np.random.random_sample() < self.tf_rate else False

            # label_embedding from Y input or previous prediction
            if mode == 'train' and teacher_force:
                label_embedding = self.embedding(Yinput[:, step])
            else:  # no teacher force for dev and test
                label_embedding = self.embedding(pred_idx)  # make sure size [N]
            # label_embedding = self.dropout(label_embedding)

            rnn_input = torch.cat([label_embedding, context], dim=-1)
            pred, context, attention, prev_h, prev_c = \
                self.forward_step(rnn_input, key, value, prev_h, prev_c, attention_mask)
            pred = pred.unsqueeze(1)

            # label index for the next loop
            pred_idx = torch.max(pred, dim=2)[1]  # argmax size [1, 1]
            pred_idx = pred_idx.squeeze(dim=1)

            # Checking eos for the whole batch matrix later
            # if not training and pred_idx.cpu().data.numpy()[0] == 2:
            #     break  # end of sentence

            # add to the prediction if not eos
            if pred_seq is None:
                pred_seq = pred
            else:
                pred_seq = torch.cat([pred_seq, pred], dim=1)

        return pred_seq

    def forward_step(self, concat, key, value, prev_h, prev_c, attention_mask):

        h1, c1 = self.cell1(concat, (prev_h, prev_c))

        attention, context = self.attention(key, value, h1, attention_mask)
        concat = torch.cat([c1, context], dim=1)  # (N, decoder_dim + values_dim)

        projection = self.character_projection(self.relu(self.mlp(concat)))
        pred = self.softmax(projection)

        return pred, context, attention, h1, c1


# helper function to apply layers for each timestep
def ApplyPerTime(input_module, input_x):

    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    reshaped_x = input_x.contiguous().view(-1, input_x.size(-1))
    output_x = input_module(reshaped_x)
    return output_x.view(batch_size,time_steps,-1)


# """
# - Mel vectors (N, L, 40)
# - Keys: (N, L, A)
# - Values: (N, L, B)
# - Decoder produces query (N, A)
# - Perform bmm( query(N, 1, A), keys(N, A, L)) = (N,1,L) this is the energy for each sample and each place in the utterance
# - Softmax over the energy (along the utterance length dimension) to create the attention over the utterance
# - Perform bmm( attention(N, 1, L), values(N,L,B)) = (N, 1, B) to produce the context (B = 256)
# - A: key/query length
# INPUTS:
# - encoder_feature: (N,L,B) where B=256 => keys(N, L, A) => keys(N, A, L)
# - decoder_state: (N, B) => query (N, A) unsqueeze=> (N, 1, A)
class Attention(nn.Module):
    def __init__(self, A=128, hidden_dim=256):
        super(Attention, self).__init__()
        # self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=-1)  # along the time dimension, which is the last one
        self.query_linear = nn.Linear(hidden_dim, A)

    def forward(self, key, value, decoder_state, attention_mask):

        query = self.query_linear(decoder_state).unsqueeze(1)  # query (N, A) => (N, 1, A)
        energy = torch.bmm(query, key)  # (N,1,L)
        attention = self.softmax(energy)  # (N,1,L)

        attention = attention * attention_mask
        attention = attention / torch.sum(attention, dim=-1).unsqueeze(2)  # (N,1,L) / (N, 1, 1) = (N,1,L)

        context = torch.bmm(attention, value)  # (N, 1, B) Eq. 5 in paper
        context = context.squeeze(dim=1)  # (N, B)
        # context = self.dropout(context)

        return attention, context


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_longtensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).long()


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
    # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)
