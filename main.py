from train import train


for mode in ["normal", "2same", "3same"]:
    model = train(mode)
