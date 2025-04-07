import torch
import torchaudio
import librosa
from torchaudio.transforms import Spectrogram as AudioSpectroTransform
from torchvision.transforms import Resize
import torchvision
from torchvision import io 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import os
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SpectrogramImageDataset(Dataset):
    def __init__(self, dir, ref_csv,):
        self.dir = dir
        self.labels = pd.read_csv(ref_csv)[["primary_label", "filename"]]
        self.classes = self.labels.primary_label.unique()
        
        self.target_transform = lambda l: torch.nn.functional.one_hot(
            torch.where(torch.Tensor(self.classes == l))[0],
            num_classes=206).type(torch.float16)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.labels.filename[idx]+".png")
        label = self.labels.primary_label[idx]
        #spec = io.decode_image(path, io.ImageReadMode.GRAY)
        spec = Image.open(path)
        if self.transform:
            spec = self.transform(spec)[0, :,:]
        if self.target_transform:
            label = self.target_transform(label)[0]
        return spec, label


"""
NOT USED
loading spectrograms individually is too slow
use the run with the '--process' flag below and save them as images first, then train on
the preprocessed images
"""
class SpectrogramDataset(Dataset):
    def __init__(self, dir, ref_csv,):
        self.dir = dir
        self.labels = pd.read_csv(ref_csv)[["primary_label", "filename"]]
        self.classes = self.labels.primary_label.unique()
        
        self.target_transform = lambda l: torch.nn.functional.one_hot(
            torch.where(torch.Tensor(self.classes == l))[0], num_classes=206).type(torch.float16)
        
        # transform stuff
        self.spec = AudioSpectroTransform(n_fft=800)
        self.resize = Resize((512, 64))

    def transform(self, audio):
        spec = librosa.amplitude_to_db(self.spec(audio))
        spec = torch.Tensor(spec)[:, 1:, :]
        #spec = torch.stack([spec, spec, spec], dim=1)
        spec = torch.movedim(spec, 1, 2)
        print(spec.size())
        spec = self.resize(spec)

        return spec

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audpath = os.path.join(self.dir, self.labels.filename[idx])
        label = self.labels.primary_label[idx]
        waveform, s_rate = torchaudio.load(audpath, normalize=True)
        if self.transform:
            spec = self.transform(waveform)
        if self.target_transform:
            label = self.target_transform(label)[0]
        return spec, label

if __name__ == "__main__":
    from pathlib import Path
    data = SpectrogramDataset("data/train_audio", "data/train.csv")

    print(data[10][0].size())
    id = 1420
    print(data.labels.filename[id])
    plt.imshow(data[id][0].T,  cmap='plasma')
    plt.show()

    process = False
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--process":
        for id in range(len(data)):
            path = "data/train_spectro/"
            Path(path+data.labels.filename[id].split("/")[0]).mkdir(parents=True, exist_ok=True)
            torchvision.utils.save_image(torch.squeeze(data[id][0]),
                                         path + data.labels.filename[id]+".png")
            if id % 20 == 0:
                print(id)
    #    plt.imshow(torch.squeeze(im.T))
    #    plt.show()


