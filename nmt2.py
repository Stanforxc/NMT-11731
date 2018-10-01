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
from dataloader import TrainDataset, DevDataset, my_collate, dev_collate
from model import Encoder, Decoder

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
    train_dataset = TrainDataset('train', vocab)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, collate_fn=my_collate)
    dev_dataset = DevDataset('dev', vocab)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=dev_collate)

    # Create the seq2seq network
    encoder = Encoder(vocab_size=len(vocab.src), hidden_dim=256, attention_dim=128, value_dim=256)
    decoder = Decoder(vocab_size=tgt_vocab_size, hidden_dim=256, attention_dim=128, value_dim=256, tf_rate=tf_rate)

    # Initialize weights
    encoder.apply(weights_init)
    decoder.apply(weights_init)

    # [optional] load state dicts
    if encoder_state and decoder_state:
        encoder.load_state_dict(torch.load(encoder_state))
        decoder.load_state_dict(torch.load(decoder_state))

    # Loss function and optimization
    loss_fn = nn.CrossEntropyLoss(reduce=False)
    model_params = list(encoder.parameters()) + list(decoder.parameters())
    optim = torch.optim.Adam(model_params, lr=learn_rate, weight_decay=1e-5) # todo: change back to 1e-5
    # optim = torch.optim.SGD(LAS_params, lr=learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.8)

    if torch.cuda.is_available():
        # Move the network and the optimizer to the GPU
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        loss_fn = loss_fn.cuda()

    best_bleu = 0
    # start training
    for epoch in range(epochs):
        losses = []
        tmp_losses = []
        count = -1

        total = len(train_dataset) / batch_size
        interval = total // 20

        scheduler.step()

        for (src_sents, src_lens, Yinput, Ytarget, tgt_lens) in train_dataloader:

            actual_batch_size = len(src_lens)
            count += 1
            optim.zero_grad()  # Reset the gradients

            # forward
            key, value = encoder(to_variable(src_sents), src_lens)
            pred_seq = decoder(key, value, to_variable(Yinput), Yinput.size(-1), 'train', src_lens)
            pred_seq = pred_seq.resize(pred_seq.size(0) * pred_seq.size(1), tgt_vocab_size)

            # create the tgt mask
            tgt_mask = np.zeros((actual_batch_size, max(tgt_lens)))
            # print('max', max(tgt_lens))

            for i in range(actual_batch_size):
                tgt_mask[i, :tgt_lens[i]] = np.ones(tgt_lens[i])
            tgt_num_words = tgt_mask.sum()
            tgt_mask = to_variable(to_tensor(tgt_mask)).resize(actual_batch_size * max(tgt_lens))

            # loss
            loss = loss_fn(pred_seq, to_variable(Ytarget).resize(Ytarget.size(0) * Ytarget.size(1)))
            loss = torch.sum(loss * tgt_mask) / actual_batch_size

            # backword
            loss.backward()
            loss_np = loss.data.cpu().numpy()
            losses.append(loss_np)
            tmp_losses.append(loss_np)

            ppl = loss_np * actual_batch_size / tgt_num_words

            # clip gradients
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.)

            # UPDATE THE NETWORK!!!
            optim.step()

            if count % interval == 0:
                print('Train Loss: %.2f Perplexity: %.2f Progress: %d%%'
                      % (np.asscalar(np.mean(tmp_losses)), np.asscalar(ppl), count * 100 / total))
                tmp_losses = []

        print("### Epoch {} Loss: {:.4f} ###".format(epoch, np.asscalar(np.mean(losses))))

        # # validation
        hyp_corpus = []
        ref_corpus = []
        for (src_sents, src_lens, Yinput, Ytarget, tgt_lens, tgt_sents_str, _) in dev_dataloader:

            actual_batch_size = len(src_lens)

            # forward
            key, value = encoder(to_variable(src_sents), src_lens)
            pred_seq = decoder(key, value, None, Yinput.size(-1), 'dev', src_lens)  # batch, sent_len, emb
            # print(pred_seq.size())

            # record word sequence
            ref_corpus.extend(tgt_sents_str)
            hyp_np = pred_seq.data.cpu().numpy()
            # print(hyp_np.shape)

            for b in range(hyp_np.shape[0]):
                word_seq = []
                for step in range(hyp_np.shape[1]):
                    pred_idx = np.argmax(hyp_np[b, step, :])
                    if pred_idx == vocab.tgt.word2id['</s>']:
                        break
                    word_seq.append(vocab.tgt.id2word[pred_idx])
                hyp_corpus.append(word_seq)

        count = 0
        for r, h in zip(ref_corpus, hyp_corpus):
            print(r)
            print(h)
            print()
            count += 1
            if count == 20:
                break

        bleu_score = compute_corpus_level_bleu_score(ref_corpus, hyp_corpus)

        if not best_bleu or bleu_score > best_bleu:
            best_bleu = bleu_score
            torch.save(encoder.state_dict(), '%s-encoder-e%d' % (name, epoch))
            torch.save(decoder.state_dict(), '%s-decoder-e%d' % (name, epoch))

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


def decode(encoder_state, decoder_state, mode, output_path):

    vocab = pickle.load(open(data_path + 'vocab.bin', 'rb'))
    tgt_vocab_size = len(vocab.tgt)
    decode_dataset = DevDataset(mode, vocab)
    decode_dataloader = torch.utils.data.DataLoader(decode_dataset, batch_size=16, shuffle=False, collate_fn=dev_collate)

    # Create the seq2seq network
    encoder = Encoder(vocab_size=len(vocab.src), hidden_dim=256, attention_dim=128, value_dim=256)
    decoder = Decoder(vocab_size=tgt_vocab_size, hidden_dim=256, attention_dim=128, value_dim=256, tf_rate=0)
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    encoder.load_state_dict(torch.load(encoder_state))
    decoder.load_state_dict(torch.load(decoder_state))

    # TODO: try
    # encoder.eval()
    # decoder.eval()

    hyp_corpus = []
    ref_corpus = []
    hyp_corpus_ordered = []
    for (src_sents, src_lens, Yinput, Ytarget, tgt_lens, tgt_sents_str, orig_indices) in decode_dataloader:

        # forward
        key, value = encoder(to_variable(src_sents), src_lens)
        pred_seq = decoder(key, value, None, Ytarget.size(-1), mode=mode, src_lens=src_lens)  # batch, sent_len, emb

        # record word sequence
        ref_corpus.extend(tgt_sents_str)
        hyp_np = pred_seq.data.cpu().numpy()
        batch_hyp_orderd = [None] * hyp_np.shape[0]

        for b in range(hyp_np.shape[0]):
            word_seq = []
            for step in range(hyp_np.shape[1]):
                pred_idx = np.argmax(hyp_np[b, step, :])
                if pred_idx == vocab.tgt.word2id['</s>']:
                    break
                word_seq.append(vocab.tgt.id2word[pred_idx])
            hyp_corpus.append(word_seq)
            batch_hyp_orderd[orig_indices[b]] = word_seq
        hyp_corpus_ordered.extend(batch_hyp_orderd)

    # count = 0
    # for r, h in zip(ref_corpus, hyp_corpus):
    #     print(r)
    #     print(h)
    #     print()
    #     count += 1
    #     if count == 20:
    #         break

    bleu_score = compute_corpus_level_bleu_score(ref_corpus, hyp_corpus)
    print(mode, 'BLEU Score: ', bleu_score)

    print("Writing to file...")
    with open(output_path, 'w') as f:
        for hyp in hyp_corpus_ordered:
            hyp_sent = ' '.join(hyp)
            f.write(hyp_sent + '\n')


if __name__ == '__main__':
    encoder_state = None
    decoder_state = None

    if len(sys.argv) == 3:
        encoder_state = sys.argv[1]
        decoder_state = sys.argv[2]

    train_model(batch_size=64, epochs=20, learn_rate=1e-3, name='try3', tf_rate=0.5,
                encoder_state=encoder_state, decoder_state=decoder_state)

    # decode(encoder_state, decoder_state, 'dev', 'decode-dev.txt')
    # decode(encoder_state, decoder_state, 'test', 'decode-test.text')
    """
    # try1. lr 1e-4, lr_decay 0.8 every two epochs
    # try2: attention_dim to 256 2
    """
