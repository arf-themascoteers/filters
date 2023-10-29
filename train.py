import analyze
from my_conv_net import MyConvNet
import torch
import torch.nn as nn
import constants
import cv2
import torch.nn.functional as F


def train(mode="normal"):
    data = cv2.imread("4.png")
    data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    data = torch.tensor(data, dtype=torch.float)
    analyze.plot_tensor(data,"others","tensor")
    images = data.reshape(1, 1, data.shape[0], data.shape[1])
    model = MyConvNet()
    filters = model.layer1[0].weight.data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyConvNet()

    if mode != "normal":
        filters = model.layer1[0].weight.data
        r = torch.rand(28, 28)
        filters[0, 0] = r
        filters[1, 0] = r

    analyze.analyze(model, mode, "before")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    for epoch in range(1000):
            optimizer.zero_grad()
            images = images.to(device)
            labels = torch.tensor([4])
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.nll_loss(outputs, labels)
            print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
            loss.backward()
            optimizer.step()


    torch.save(model, constants.DEFAULT_MODEL_PATH)
    analyze.analyze(model, mode, "after")
    return model