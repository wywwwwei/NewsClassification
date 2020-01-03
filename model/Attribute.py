import torch

class Attribute:
    def __init__(self,name,embedding,dataset='.',results_dir ='./result/'):
        self.name = name
        self.embedding = torch.tensor()
        self.embed_size = self.embedding.size(1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_list=[x.strip() for x in open(dataset+"/data/classlist.txt").readlines()]
        self.num_classes = len(self.class_list)
        self.model_save = dataset + self.name + ".ckpt"


class Hyperparameters:
    def __init__(self):
        #样本处理
        self.batch_size = 200
        self.pad_size = 32
        self.dropout =0.5
        self.require_improvement = 1000
        #卷积层
        self.filter_num = 256
        self.filter_shape =(2,3,4)
        #梯度
        self.epoch = 20
        self.lr = 1e-3