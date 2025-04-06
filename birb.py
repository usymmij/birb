import argparse
import torch
import timm
from dataUtils import SpectrogramDataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    parser = argparse.ArgumentParser(
                    prog='birb',
                    description='Identifies bird, amphibian, mammal, and insect species from the  Middle Magdalena Valley (Columbia)',
                    epilog='made by @usymmij')

    data = SpectrogramDataset("data/train_audio", "data/train.csv")
    # dataset is small
    trainlen = int(len(data) * 0.85)
    vallen = len(data) - trainlen 
    train, val = torch.utils.data.random_split(data, [trainlen, vallen])
    train_dl = DataLoader(train, batch_size=16, shuffle=True)
    val_dl = DataLoader(val, batch_size=16, shuffle=True)


if __name__ == "__main__":
    main()

