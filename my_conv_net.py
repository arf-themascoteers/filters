import torch.nn as nn
import torch.nn.functional as F


class MyConvNet(nn.Module):

    def __init__(self):
        super(MyConvNet, self).__init__()
        self.conv = nn.Conv2d(1,3, (28,28))
        self.relu = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3, 10)

        self.current_fms = None

    def forward(self, x):
        out = self.conv(x)
        self.current_fms = out
        out = self.relu(out)
        out = self.flatten(out)
        out = self.linear(out)
        return F.log_softmax(out, dim=1)

