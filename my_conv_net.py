import torch.nn as nn
import torch


class MyConvNet(nn.Module):

    def __init__(self):
        super(MyConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4)
        )

        self.fc1 = nn.Linear(108, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        return out

