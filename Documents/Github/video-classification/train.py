"""
    Usage: python train.py train
"""

import numpy as np
import tqdm

from data_loader.VideoDataset import VideoDataset
from models.network import VideoModel
from models.loss import video_classification_loss

from torch.utils.data import DataLoader
import fire
import os
import torch, torch.nn as nn

def train(info_file = '', num_epoch=25 ,val_size = 0.2, batch_size= 4, num_data_workers=1, use_gpu=True):  
    dataset = VideoDataset(info_file)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int((1-val_size)*len(dataset)), len(dataset) - int((1-val_size)*len(dataset))])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=num_data_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_data_workers)

    model = VideoModel(num_classes=3).float()

    if(use_gpu):
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), 
                    lr=0.01,
                    momentum=0.009,
                    weight_decay=0.0005,
                    nesterov=True)

    loss_func = nn.CrossEntropyLoss()

    for ep in range(1, num_epoch):
        print(f'Epoch {ep}', end=' ')

        losses = []
        for i, batch in tqdm.tqdm(enumerate(train_dataloader)):

            if(use_gpu):
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()

            optimizer.zero_grad()
            
            prediction = model(batch[0].float())
            loss = loss_func(prediction, batch[1].float())
            
            loss.backward()
            optimizer.step()

            if(use_gpu):
                loss = loss.detach().cpu()

            loss = loss.numpy()
            losses.append(loss)

            del loss, prediction

        print('Loss = ',np.average(losses))
        torch.cuda.empty_cache()

        # if(val_size is not None and ep%5 == 0):
        #     errors = []
        #     with torch.no_grad():
        #         for i, batch in tqdm.tqdm(enumerate(val_dataloader)):
        #             if(use_gpu):
        #                 batch[0] = batch[0].cuda()
        #                 batch[1] = batch[1].cuda()
        #             prediction = model(batch[0].float(), batch[1][:, 2:].float())
        #             err = torch.norm(prediction[:, :2] - batch[1][:, :2].float(), 1).detach().cpu().numpy()/batch[0].shape[0]
        #             errors.append(err)
        #     print(f'Validation loss = {np.average(errors)}')


if __name__ == "__main__":
    fire.Fire()