import torch
import numpy as np
import pickle as pkl
import os
import data.participle as participle


class DatasetIterator(object):
    """
    Data iterator, let classification model use
    """

    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # Records whether the number of batches is an integer
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # Length before pad
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index *
                                   self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index *
                                   self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, attribute, hyperparameter):
    iter = DatasetIterator(
        dataset, hyperparameter.batch_size, attribute.device)
    return iter


def load_pretrain_vector(attribute):
    """
    Use sogounews as the pretrained word vector
    Download link:https://github.com/Embedding/Chinese-Word-Vectors
    """
    
    pretrain_dir = "./data/sgns.sogounews.bigram-char"
    pretrain_save_name = "./data/embedding_sogounews"

    embedding_dimension = 300

    word_to_id = participle.load_vocab_file(attribute)

    if not os.path.exists(pretrain_save_name+".npz"):
        print("npz not exist ")
        embeddings = np.random.rand(len(word_to_id), embedding_dimension)
        with open(pretrain_dir, "r", encoding='UTF-8') as f:
            for i, line in enumerate(f.readlines()):
                lin = line.strip().split(" ")
                if lin[0] in word_to_id:
                    print("in")
                    idx = word_to_id[lin[0]]
                    emb = [float(x) for x in lin[1:301]]
                    embeddings[idx] = np.asarray(emb, dtype='float32')
                    if i == 0:
                        print(embeddings)
        np.savez_compressed(pretrain_save_name, embeddings=embeddings)
    else:
        print("npz exists")
    return word_to_id
