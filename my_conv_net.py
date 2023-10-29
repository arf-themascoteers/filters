import torch.nn as nn
import torch.nn.functional as F


class MyConvNet(nn.Module):

    def __init__(self):
        super(MyConvNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,3, (28,28)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(3, 10)
        )

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x, dim=1)

