import torch
import torchaudio
import librosa
from torchaudio.transforms import Spectrogram as AudioSpectroTransform
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SpectrogramDataset(Dataset):
    def __init__(self, dir, ref_csv,):
        self.dir = dir
        self.labels = pd.read_csv(ref_csv)[["primary_label", "filename"]]
        self.classes = self.labels.primary_label.unique()
        
        self.target_transform = lambda l: torch.nn.functional.one_hot(
            torch.where(torch.Tensor(self.classes == l))[0], num_classes=206).to(device)
        self.spec = AudioSpectroTransform(n_fft=800)

    def transform(self, audio):
        spec = librosa.amplitude_to_db(self.spec(audio))
        spec = torch.Tensor(spec).to(device)
        return spec

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audpath = os.path.join(self.dir, self.labels.filename[idx])
        label = self.labels.primary_label[idx]
        waveform, s_rate = torchaudio.load(audpath, normalize=True)
        if self.transform:
            spec =  self.transform(waveform)
        if self.target_transform:
            label = self.target_transform(label)[0]
        return spec, label

if __name__ == "__main__":
    data = SpectrogramDataset("data/train_audio", "data/train.csv")
