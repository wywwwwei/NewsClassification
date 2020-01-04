import torch
import numpy as np


class Attribute:
    def __init__(self, name, dataset='.', results_dir='./result/'):
        self.name = name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.class_list = [x.strip() for x in open(
            file=dataset+"/data/classlist.txt", encoding="utf8").readlines()]
        print(self.class_list)
        self.num_classes = len(self.class_list)
        self.model_save = dataset + self.name + ".ckpt"


class Hyperparameters:
    def __init__(self, embedding, dataset='.'):
        # Embedding layer
        self.embedding = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))
        self.embed_size = self.embedding.size(1)
        # Sample processing
        self.batch_size = 200
        self.pad_size = 32
        self.dropout = 0.5
        # Convolution layer
        self.filter_num = 256
        self.filter_shape = (2, 3, 4)
        # Gradient
        self.epoch = 20
        self.lr = 1e-3
