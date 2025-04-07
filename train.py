import argparse
import ast_model
import torch
from dataUtils import SpectrogramImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# seed so that the training split will be the same each time
torch.manual_seed(1024)
torch.cuda.manual_seed(1024) 

EPOCHS = 10

def main():
    tr_acc = []
    val_acc = []
    # running out of time, otherwise I would have preferred argparse
    #parser = argparse.ArgumentParser(
    #                prog='birb',
    #                description='Identifies bird, amphibian, mammal, and insect species from the  Middle Magdalena Valley (Columbia)',
    #                epilog='made by @usymmij')

    data = SpectrogramImageDataset("data/train_spectro", "data/train.csv")
    # dataset is small
    trainlen = int(len(data) * 0.75)
    vallen = int((len(data) - trainlen) * 0.5)
    testlen = len(data) - trainlen - vallen
    train, val, test = torch.utils.data.random_split(data, [trainlen, vallen, testlen])
    train_dl = DataLoader(train, batch_size=18, shuffle=True, num_workers=6)
    val_dl = DataLoader(val, batch_size=16, shuffle=True, num_workers=6)
    test_dl = DataLoader(test, batch_size=16, shuffle=True, num_workers=6)


    model = ast_model.ASTModel(input_tdim=512, input_fdim=64, label_dim=206,
                               audioset_pretrain=False).to(device)

    # second half of training
    version = "v1"
    if len(sys.argv) > 1 and sys.argv[1] == "--continue":
        version = "v2"
        model.load_state_dict(torch.load("/mnt/storage/bird_ckpts/v1/e9.h5"))

    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.999])
    loss_fn = torch.nn.CrossEntropyLoss()

    a=0
    for epoch in range(EPOCHS):
        accuracy = 0
        batches = 0
        for ims, labels in tqdm(train_dl):
            optimizer.zero_grad()

            ims = ims.to(device)
            labels = labels.to(device)
            out = model(torch.squeeze(ims))

            # Compute the loss and its gradients
            loss = loss_fn(out, labels)
            loss.backward()
            
            # Adjust learning weights
            optimizer.step()

            batches += len(out)
            accuracy += sum(torch.argmax(labels, axis=1) == torch.argmax(out, axis=1))

        print(f"train accuracy: {accuracy / batches}")
        tr_acc.append(accuracy / batches)
        accuracy = 0
        batches = 0

        for ims, labels in tqdm(val_dl): 
            ims = ims.to(device)
            labels = labels.to(device)
            out = model(torch.squeeze(ims))
            batches += len(out)
            accuracy += sum(torch.argmax(labels, axis=1) == torch.argmax(out, axis=1))

        print(f"validation accuracy: {accuracy / batches}")
        val_acc.append(accuracy / batches)

        if(epoch % 5 == 4):
            torch.save(model.state_dict(), "/mnt/storage/bird_ckpts/"+version+"/e"+str(epoch)+".h5")
    print("training accuracies:")
    print(tr_acc)
    print("validation accuracies")
    print(val_acc)


if __name__ == "__main__":
    main()

