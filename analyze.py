import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


def plot_tensor(tensor,folder, filename):
    mean = torch.mean(tensor)
    tensor = tensor.data.clone()
    # tensor[tensor >= mean] = 255
    # tensor[tensor < mean] = 0
    plt.imshow(tensor.cpu().numpy(), cmap="gray")
    plt.savefig(f"{folder}/{filename}")


def plot_filters(filters, folder, prefix):
    filters = filters.clone()
    filters = filters.reshape(filters.shape[0], filters.shape[2], filters.shape[3])
    for index,tensor in enumerate(filters[0:3]):
        plot_tensor(tensor, folder,f"{prefix}_{index}.png")



def analyze(model,folder,prefix):
    filters = model.layer1[0].weight.data
    plot_filters(filters,folder,prefix)



