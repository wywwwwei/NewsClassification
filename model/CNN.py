import model.Attribute as attributes
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, attribute: attributes.Attribute, hyperparameter: attributes.Hyperparameters):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=hyperparameter.embedding, freeze=False)
        # Convolution layer
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=hyperparameter.filter_num, kernel_size=(
                i, hyperparameter.embed_size))for i in hyperparameter.filter_shape]
        )
        self.dropout = nn.Dropout(hyperparameter.dropout)  # Prevent overfitting
        # Fully-connected lyaer
        self.fc = nn.Linear(hyperparameter.filter_num *
                            len(hyperparameter.filter_shape), attribute.num_classes)

    def active_and_maxpooling(self, conv, x):
        out = F.relu(conv(x)).squeeze(3)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        return out

    # Prior to calculate
    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.active_and_maxpooling(conv, out)
                         for conv in self.convs], 1)
        out = self.dropout(out)
        return self.fc(out)
