import torch
import torch.nn as nn
import numpy as np


def weights_init(model):
    for name, w in model.named_parameters():
        if "embedding" not in name:
            if "weight" in name:
                nn.init.kaiming_normal_(w)


def train(model, attribute, hyperparameter, train_loader, validate_loader):
    model.train()
    # Instantiate an Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=attribute.lr)
    # User Cross-Entropy Loss Function.
    loss_func = nn.CrossEntropyLoss()

    total_batch = 0
    last_improve = 0
    flag = False

    for epoch in range(hyperparameter.epoch):
        for (XTrain, YTrain) in enumerate(train_loader):
            # Clear gradients
            optimizer.zero_grad()
            # Forward Propagation
            output = model(XTrain)
            # Calculate Loss: softmax cross-entropy loss
            loss = loss_func(output, YTrain)
            # Back Propagation
            loss.backward()
            # Update parameter
            optimizer.step()

        evaluate(model=model, attribute=attribute,
                 test_loader=validate_loader, train=True, epoch=epoch+1)
    torch.save(model.state_dict(), attribute.model_save)


def evaluate(model, attribute, test_loader, train=False, epoch=0):
    if(not train):
        model.load_state_dict(torch.load(attribute.save_path))
    model.eval()
    # Calculate Accuracy
    correct = 0
    total = 0

    with torch.no_grad():
        for (XTest, YTest) in test_loader:
            output = model(XTest)
            _, YPred = torch.max(output.data, 1)

            total += YTest.size(0)
            correct += (YPred == YTest).sum()

    print("Accuracy: {}" .format(100*correct/total))
