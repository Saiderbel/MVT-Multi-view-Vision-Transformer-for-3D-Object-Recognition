"""
PyTorch dataset for loading snapshots of ModelNet 3D meshes.
"""
import os
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import random

import torchvision.transforms as T
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

classes = {
  "bathtub": 0,
  "bed": 1,
  "chair": 2,
  "desk": 3,
  "dresser": 4,
  "monitor": 5,
  "night_stand": 6,
  "sofa": 7,
  "table": 8,
  "toilet": 9
}

class ModelNetSnapshot(torch.utils.data.Dataset):
    """
    Dataset for loading ModelNet 3D mesh snapshots.
    """
    dataset_path = ""
    
    def __init__(self, split, dataset_path='ModelNet10/', augment=True) -> None:
        super().__init__()
        assert split in ['train', 'test', 'validation']

        self.snapshots = []
        self.dataset_path = dataset_path
        self.augment=augment

        mean, std = torch.tensor([0.8953, 0.8935, 0.8942]), torch.tensor([0.0498, 0.0487, 0.0485])

        for root, dirs, files in os.walk(dataset_path):
            for dir in dirs:
                if 'snapshots' in dir and split in root:
                    self.snapshots.append(os.path.join(root, dir))
            '''
            if split in root:    
                for file in files:
                    if '.png' in file:
                        self.snapshots.append(os.path.join(root, file))
            '''
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        self.rand_trans = torch.nn.ModuleList([
            torch.nn.Sequential(T.CenterCrop(80), T.Resize(256)),
            torch.nn.Sequential(T.CenterCrop(150), T.Resize(256)),
            torch.nn.Sequential(T.CenterCrop(200), T.Resize(256)),
            T.RandomRotation((90, 90)),
            T.RandomRotation((180, 180)),
            T.RandomRotation((270, 270)),
        ])
        self.rand_trans_p =  0.3


    def __getitem__(self, index):

        # img_array = Image.open(self.snapshots[index])
        snapshot_path = self.snapshots[index]

        # Construct the filenames
        view_patches = []
        transform = random.choice(self.rand_trans) if (np.random.uniform() < self.rand_trans_p) else None
        for i in range(12):
            fn = os.path.join(snapshot_path, f'{str(i)}.png')
            if self.augment:
                view_patches.append(self.split_image(fn, transform=transform))
            else:
                view_patches.append(self.split_image(fn))


        # view_patches = np.array(view_patches)

        # Construct the patches for each view
        # patches = self.split_image()
        label = self.snapshots[index].split('/')[-3]

        # TODO: Do we want to do the image splits directly here or not?
        # view_patches = torch.tensor(view_patches, dtype=torch.float32) / 255
        view_patches = torch.stack(view_patches)

        return {
            # potentionally slow
            'view_patches': view_patches,
            'labels': classes[label]
        }

    def __len__(self):
        return len(self.snapshots)

    def split_image(self, img_path, patch_size=4, transform=None):
        im = Image.open(img_path)
        im = transform(self.normalize(im)) if transform else self.normalize(im)
        im = im.T
        assert im.shape[0] == im.shape[1]

        M = N = im.shape[0] // patch_size

        # Source: https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
        tiles = torch.stack([im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)])

        return tiles

## TODO: adapt code for our purposes
class MODELNETDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        #self.transforms = transforms.Compose(
        #    [transforms.ToTensor()]
        #)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 10

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ModelNetSnapshot(dataset_path=self.hparams.data_dir, split="train")
            self.data_test = ModelNetSnapshot(dataset_path=self.hparams.data_dir, split="test", augment=False)
            self.data_val = ModelNetSnapshot(dataset_path=self.hparams.data_dir, split="validation", augment=False)
            print(len(self.data_train))
            print(len(self.data_test))
            print(len(self.data_val))
            # dataset = ConcatDataset(datasets=[trainset, testset])
            # train_len = int(len(trainset)*0.8)
            # val_len = int((len(trainset)-train_len))
            # train_val_split = (train_len, val_len)
            # self.data_train, self.data_val= random_split(
            #     dataset=trainset,
            #     lengths=train_val_split,
            #     generator=torch.Generator().manual_seed(42),
            # )
            # self.data_test = testset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )


if __name__ == '__main__':
    ds = ModelNetSnapshot('train')
    print(ds[0]['view_patches'].shape)
    # print(ds[0]['patches'].shape)


