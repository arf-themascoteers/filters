from train import train
from test import test


for mode in ["normal", "2same"]:
    model = train(mode)
    test()
