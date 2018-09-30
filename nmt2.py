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


'''
Dataset and Dataloader
'''
# Input: X, Yinput, Ytarget
def my_collate(batch):
    batch_size = len(batch)

    # sort by src sentence length
    tuples = [(tup[0].shape[0], tup[0], tup[1], tup[2]) for tup in batch]
    tuples.sort(key=lambda x: x[0], reverse=True)  # sort in descending order

    max_src_len = tuples[0][0]
    max_tgt_len = max([len(tup[1]) for tup in batch])

    padded_src_sents = np.zeros((batch_size, max_src_len))
    padded_Yinput = np.zeros((batch_size, max_tgt_len))
    padded_Ytarget = np.zeros((batch_size, max_tgt_len))

    src_lens = []
    tgt_lens = []

    for i in range(batch_size):
        src_sent, Yinput, Ytarget = tuples[i][1:]

        src_len = src_sent.shape[0]
        tgt_len = Yinput.shape[0]

        padded_src_sents[i, :src_len] = src_sent
        padded_Yinput[i, :tgt_len] = Yinput
        padded_Ytarget[i, : tgt_len] = Ytarget

        src_lens.append(src_len)
        tgt_lens.append(tgt_len)

    return to_longtensor(padded_src_sents), src_lens, \
           to_longtensor(np.array(padded_Yinput)), to_longtensor(np.array(padded_Ytarget)), tgt_lens


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, vocab):
        train_src = "data/train.de-en.de.wmixerprep"
        train_tgt = "data/train.de-en.en.wmixerprep"
        dev_src = "data/valid.de-en.de"
        dev_tgt = "data/valid.de-en.en"
        test_src = "data/test.de-en.de"
        test_tgt = "data/test.de-en.en"

        if dataset == 'train':
            src_sents = read_corpus(train_src, 'src')
            tgt_sents = read_corpus(train_tgt, 'tgt')
        elif dataset == 'dev':
            src_sents = read_corpus(dev_src, 'src')
            tgt_sents = read_corpus(dev_tgt, 'tgt')

        self.X = vocab.src.words2indices(src_sents)
        self.Y = vocab.tgt.words2indices(tgt_sents)

        assert len(self.X) == len(self.Y)

        self.Yinput = []
        self.Ytarget = []
        for i in range(len(self.Y)):
            # tgt seq input: <e> a b c
            # tgt seq target:    a b c  </e>
            input, target = self.Y[i][:-1], self.Y[i][1:]
            self.Yinput.append(input)
            self.Ytarget.append(target)

    def __getitem__(self, index):
        return np.array(self.X[index]), np.array(self.Yinput[index]), np.array(self.Ytarget[index])

    def __len__(self):
        return len(self.X)


# TODO: update for testing
class TestDataset(torch.utils.data.Dataset):

    def __init__(self):

        utterances = np.load(data_path + 'test.npy')
        self.X = utterances

    def __getitem__(self, index):
        eight_multiples = len(self.X[index]) // 8 * 8
        return self.X[index][:eight_multiples], eight_multiples

    def __len__(self):
        return len(self.X)


'''
Model: Encoder, Decoder, Attention
'''


# **Encoder**
# is a BiLSTM. Each has 256 hidden dimensions per direction.
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, attention_dim, value_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)
        self.BLSTM = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True,
                             num_layers=1)

        self.key_linear = nn.Linear(hidden_dim * 2, attention_dim)  # output from bLSTM
        self.value_linear = nn.Linear(hidden_dim * 2, value_dim)  # output from bLSTM

    def forward(self, input, src_lens):
        # print(input.size())
        # print(src_lens)
        embeddings = self.embedding(input)
        # print(embeddings.size())
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
    def forward(self, key, value, Yinput, max_len, training, src_lens):

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

        if not training:
            max_len = 500

        for step in range(max_len):

            # teacher force rate: prob to feed the ground truth as input
            teacher_force = True if np.random.random_sample() < self.tf_rate else False

            # label_embedding from Y input or previous prediction
            if training and teacher_force:
                label_embedding = self.embedding(Yinput[:, step])
            else:
                label_embedding = self.embedding(pred_idx.squeeze()) # make sure size [N]

            # print(label_embedding.size(), context.size())

            rnn_input = torch.cat([label_embedding, context], dim=-1)
            pred, context, attention, prev_h, prev_c = \
                self.forward_step(rnn_input, key, value, prev_h, prev_c, attention_mask)
            pred = pred.unsqueeze(1)

            # label index for the next loop
            pred_idx = torch.max(pred, dim=2)[1]  # argmax size [1, 1]
            if not training and pred_idx.cpu().data.numpy() == 2:  # TODO: 2 is the index for eos char
                break  # end of sentence

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
        self.softmax = nn.Softmax(dim=-1)  # along the time dimension, which is the last one
        self.query_linear = nn.Linear(hidden_dim, A)

    def forward(self, key, value, decoder_state, attention_mask):

        query = self.query_linear(decoder_state).unsqueeze(1)  # query (N, A) => (N, 1, A)
        energy = torch.bmm(query, key)  # (N,1,L)
        attention = self.softmax(energy)  # (N,1,L)

        attention = attention * attention_mask
        attention = attention / torch.sum(attention, dim=-1).unsqueeze(2)  # (N,1,L) / (N, 1, 1) = (N,1,L)

        context = torch.bmm(attention, value)  # (N, 1, B)
        context = context.squeeze(dim=1)  # (N, B)
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


'''
Train, validate and test routine
'''

import numpy as np
import torch
import torch.utils.data
from torch import nn
import os
from torch.autograd import Variable
import math

data_path = './data/'


def weights_init(layer):
    class_name = layer.__class__.__name__
    range = 0.1
    if class_name == 'LSTM':
        print(class_name)
        # Initialize LSTM weights
        # range = 1.0 / math.sqrt(hidden_dim)
        torch.nn.init.uniform(layer.weight_ih_l0, -range, range)
        torch.nn.init.uniform(layer.weight_hh_l0, -range, range)
    elif class_name == 'LSTMCell':
        print(class_name)
        torch.nn.init.uniform(layer.weight_ih, -range, range)
        torch.nn.init.uniform(layer.weight_hh, -range, range)


def compute_corpus_level_bleu_score(references, hypotheses):
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             hypotheses)

    return bleu_score

def train_model(batch_size, epochs, learn_rate, name, tf_rate, encoder_state, decoder_state):

    vocab = pickle.load(open(data_path + 'vocab.bin', 'rb'))
    tgt_vocab_size = len(vocab.tgt)

    train_dataset = MyDataset('train', vocab)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, collate_fn=my_collate)

    print(len(train_dataset))

    dev_dataset = MyDataset('dev', vocab)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=1,
                                                   shuffle=False, collate_fn=my_collate)

    # test_dataset = TestDataset()
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create the LAS network
    encoder = Encoder(vocab_size=len(vocab.src), hidden_dim=256, attention_dim=128, value_dim=256)
    decoder = Decoder(vocab_size=tgt_vocab_size, hidden_dim=256, attention_dim=128, value_dim=256, tf_rate=tf_rate)

    # Initialize weights
    # encoder.apply(weights_init)
    # decoder.apply(weights_init)

    # [optional] load state dicts
    if encoder_state and decoder_state:
        encoder.load_state_dict(torch.load(encoder_state))
        decoder.load_state_dict(torch.load(decoder_state))

    loss_fn = nn.CrossEntropyLoss(reduce=False)

    LAS_params = list(encoder.parameters()) + list(decoder.parameters())
    optim = torch.optim.Adam(LAS_params, lr=learn_rate, weight_decay=1e-5) # todo: change back to 1e-5
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.33)

    if torch.cuda.is_available():
        # Move the network and the optimizer to the GPU
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        loss_fn = loss_fn.cuda()

    for epoch in range(epochs):

        losses = []
        count = -1

        total = len(train_dataset) / batch_size
        interval = total // 1000

        for (src_sents, src_lens, Yinput, Ytarget, transcript_lens) in train_dataloader:

            # print(src_sents)

            actual_batch_size = len(src_lens)
            count += 1
            optim.zero_grad()  # Reset the gradients

            # forward
            key, value = encoder(to_variable(src_sents), src_lens)
            pred_seq = decoder(key, value, to_variable(Yinput), Yinput.size(-1), True, src_lens)
            # print('pred_seq', pred_seq.size())  # B, L, 33
            # print(pred_seq[0,0,:])
            # print('Ytarget', Ytarget.size())  # B, L

            pred_seq = pred_seq.resize(pred_seq.size(0) * pred_seq.size(1), tgt_vocab_size)

            # create the transcript mask
            transcript_mask = np.zeros((actual_batch_size, max(transcript_lens)))
            # print('max', max(transcript_lens))

            for i in range(actual_batch_size):
                transcript_mask[i, :transcript_lens[i]] = np.ones(transcript_lens[i])
            transcript_mask = to_variable(to_tensor(transcript_mask)).resize(actual_batch_size * max(transcript_lens))

            # loss
            loss = loss_fn(pred_seq, to_variable(Ytarget).resize(Ytarget.size(0) * Ytarget.size(1)))
            loss = torch.sum(loss * transcript_mask) / actual_batch_size

            # backword
            loss.backward()
            loss_np = loss.data.cpu().numpy()
            losses.append(loss_np)

            # clip gradients
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.)  # todo: tune???
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.)

            # UPDATE THE NETWORK!!!
            optim.step()
            scheduler.step()  # after train

            if count % interval == 0:
                print('Train Loss: %.2f  Progress: %d%%' % (np.asscalar(np.mean(losses)), count * 100 / total))

        print("### Epoch {} Loss: {:.4f} ###".format(epoch, np.asscalar(np.mean(losses))))

        torch.save(encoder.state_dict(), '%s-encoder-e%d' % (name, epoch))
        torch.save(decoder.state_dict(), '%s-decoder-e%d' % (name, epoch))

        # # validation
        for (src_sents, src_lens, Yinput, Ytarget, transcript_lens) in dev_dataloader:

            actual_batch_size = len(src_lens)
            assert actual_batch_size == 1

            # forward
            key, value = encoder(to_variable(src_sents), src_lens)
            pred_seq = decoder(key, value, to_variable(Yinput), Yinput.size(-1), False, src_lens)  # input prev pred
            prediction = torch.max(pred_seq, dim=2)[1].cpu().data.numpy()

            # print(prediction.shape)

        # TODO: convert prediction indices to word list and compute corpus BLUE score.
        bleu_score = 0

        print("Epoch {} validation BLUE score: {:.4f}".format(epoch, bleu_score))

    # # test
    # index = 0
    # fout = open(name+'.csv', 'w')
    # fout.write('Id,Predicted\n')
    # for src_sents, src_lens in test_dataloader:
    #     # input: np array
    #     # print('Yinput', Yinput.size())
    #
    #     # forward
    #     key, value = encoder(to_variable(src_sents), src_lens.numpy().tolist())
    #     pred_seq = decoder(key, value, None, None, False, src_lens.numpy().tolist())
    #     pred_seq = pred_seq.cpu().data.numpy()  # B, L, 33
    #
    #     for b in range(pred_seq.shape[0]):
    #         trans_dist = pred_seq[b,:,:]
    #
    #         transcript = ''.join(charlist[np.argmax(trans_dist[i, :])] for i in range(trans_dist.shape[0]))
    #
    #         fout.write('%d,%s\n' % (index, transcript))
    #         index += 1


encoder_state = None
decoder_state = None

if len(sys.argv) == 3:
    encoder_state = sys.argv[1]
    decoder_state = sys.argv[2]

train_model(batch_size=32, epochs=5, learn_rate=1e-2, name='beta0', tf_rate=0.5,
            encoder_state=encoder_state, decoder_state=decoder_state)
