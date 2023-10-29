import torch.nn as nn
import torch.nn.functional as F


class MyConvNet(nn.Module):

    def __init__(self):
        super(MyConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,100, (28,28)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        return F.log_softmax(x, dim=1)

