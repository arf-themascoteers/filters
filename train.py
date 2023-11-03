import torch
import cv2
import torch.nn.functional as F
from analyze import plot_tensor
from analyze import plot_filters
from analyze import plot_fms
from my_conv_net import MyConvNet


def train_it(model, data, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    NUM_EPOCHS = 1000
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    data = data.to(device)
    y_actual = torch.tensor([0]).to(device)
    for epoch  in range(0, NUM_EPOCHS):
        optimizer.zero_grad()
        y_pred = model(data)
        if epoch%10 == 0 and epoch < 100:
            plot_fms(model, mode, epoch)
        loss = F.nll_loss(y_pred, y_actual)
        loss.backward()
        optimizer.step()
        print(f'Epoch:{epoch + 1}, Loss:{loss.item()}')
    return model


def train(mode="normal"):
    data = cv2.imread("4.png")
    data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    data = torch.tensor(data, dtype=torch.float)
    plot_tensor(data,"others","main.png")
    data = data.reshape(1, 1, data.shape[0], data.shape[1])
    model = MyConvNet()
    if mode!="normal":

        filters = model.conv.weight.data
        lweights = model.linear.weight.data

        r = filters[0, 0]
        p = lweights[0,0]

        filters[1, 0] = r
        lweights[1, 0] = p

        if mode == "3same":
            filters[2, 0] = r
            lweights[2, 0] = p

    filters = model.conv.weight.data
    plot_filters(filters,mode,"before")
    train_it(model, data, mode)
    filters = model.conv.weight.data
    plot_filters(filters,mode,"after")