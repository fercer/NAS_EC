import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms, utils

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import loaders
import cnn


def main():
    
    composed = transforms.Compose([loaders.Rescale((224, 224)), loaders.RandomFlip(), loaders.RandomRotation(center_offset=True), loaders.ToTensor(input_datatype=np.float32, target_datatype=np.float32)])
    stare_ds = loaders.StareDiagnoses(root_dir=r'D:\Test_data\Retinal_images\STARE', transform=composed, dataset_size=128)

    stare_n_imgs = len(stare_ds)
    train_size = int(stare_n_imgs * 0.75)
    test_size = stare_n_imgs - train_size
    val_size = int(train_size * 0.25)
    train_size = train_size - val_size
    print('Available images:', stare_n_imgs, 'training:', train_size, 'validation:', val_size, 'testing', test_size)

    train_set, val_set, test_set = random_split(stare_ds, [train_size, val_size, test_size])

    train_dl = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=0)

    net = cnn.ResNet(3, 15)

    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = pl.Trainer(gpus=None, max_epochs=3, progress_bar_refresh_rate=20, logger=tb_logger)

    training_res = trainer.fit(net, train_dl, val_dl)
    print(training_res)

    testing_res = trainer.test(test_dataloaders=test_dl)
    print(testing_res)


if __name__ == '__main__':
    main()
