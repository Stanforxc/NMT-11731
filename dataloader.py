import numpy as np
import torch
import torch.utils.data
from utils import read_corpus


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


def dev_collate(batch):
    batch_size = len(batch)

    tuples = [(tup[0].shape[0], tup[0], tup[1], tup[2], tup[3]) for tup in batch]
    max_src_len = max([len(tup[0]) for tup in batch])
    max_tgt_len = max([len(tup[1]) for tup in batch])

    padded_src_sents = np.zeros((batch_size, max_src_len))
    padded_Yinput = np.zeros((batch_size, max_tgt_len))
    padded_Ytarget = np.zeros((batch_size, max_tgt_len))

    src_lens = []
    tgt_lens = []
    tgt_sents = []

    for i in range(batch_size):
        src_sent, Yinput, Ytarget, tgt_sent = tuples[i][1:]

        src_len = src_sent.shape[0]
        tgt_len = Yinput.shape[0]

        padded_src_sents[i, :src_len] = src_sent
        padded_Yinput[i, :tgt_len] = Yinput
        padded_Ytarget[i, : tgt_len] = Ytarget

        src_lens.append(src_len)
        tgt_lens.append(tgt_len)
        tgt_sents.append(tgt_sent)

    return to_longtensor(padded_src_sents), src_lens, \
           to_longtensor(np.array(padded_Yinput)), to_longtensor(np.array(padded_Ytarget)), tgt_lens, tgt_sents


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, vocab):
        train_src = "data/train.de-en.de.wmixerprep"
        train_tgt = "data/train.de-en.en.wmixerprep"

        src_sents = read_corpus(train_src, 'src')
        tgt_sents = read_corpus(train_tgt, 'tgt')

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


class DevDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, vocab):
        dev_src = dev_tgt = None
        if dataset == 'dev':
            dev_src = "data/valid.de-en.de.wmixerprep"
            dev_tgt = "data/valid.de-en.en.wmixerprep"
        elif dataset == 'test':
            dev_src = "data/test.de-en.de"
            dev_tgt = "data/test.de-en.en"

        src_sents = read_corpus(dev_src, 'src')
        tgt_sents = read_corpus(dev_tgt, 'tgt')

        self.X = vocab.src.words2indices(src_sents)
        self.Y = vocab.tgt.words2indices(tgt_sents)
        self.tgt_sents = tgt_sents

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
        return np.array(self.X[index]), np.array(self.Yinput[index]), np.array(self.Ytarget[index]), \
               self.tgt_sents[index]

    def __len__(self):
        return len(self.X)


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
