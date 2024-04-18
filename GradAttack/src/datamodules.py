from typing import Optional

import tensorflow as tf
import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from monai.apps import DecathlonDataset
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import Subset, DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from dataloader import DatasetGenerator
from settings import *

class MedDecathlonDataModule(LightningDataModule):
    #“Task01_BrainTumour”, “Task02_Heart”, “Task03_Liver”, “Task04_Hippocampus”, “Task05_Prostate”, “Task06_Lung”, “Task07_Pancreas”, “Task08_HepaticVessel”, “Task09_Spleen”, “Task10_Colon”).
    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        data_dir: str = DATA_PATH,
        num_workers: int = NUM_WORKERS,
        batch_sampler: Sampler = None,
        tune_on_val: float = 0,
        seed: int = RANDOM_SEED,
        task: str = TASK
    ):
        super().__init__()
        self._has_setup_attack = False

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = NUMBER_OUTPUT_CLASSES

        self.batch_sampler = batch_sampler
        self.tune_on_val = tune_on_val
        self.multi_class = False
        self.seed = seed
        self.task = task
        self.prepare_data()

    def prepare_data(self):
        DecathlonDataset(self.data_dir, task=self.task, section="training", download=True)
        DecathlonDataset(self.data_dir, task=self.task, section="test", download=True)

    def setup(self, stage: Optional[str] = None):
        """Initialize the dataset based on the stage option ('fit', 'test' or 'attack'):
        - if stage is 'fit', set up the training and validation dataset;
        - if stage is 'test', set up the testing dataset;
        - if stage is 'attack', set up the attack dataset (a subset of training images)

        Args:
            stage (Optional[str], optional): stage option. Defaults to None.
        """
        # medical-decathalon
        brats_data = DatasetGenerator(CROP_DIM)
        brats_data.print_info()
        ds_train = brats_data.get_train()
        ds_val = brats_data.get_validate()
        ds_test = brats_data.get_test()
        # #plot some training images
        # plt.figure(figsize=(20, 20))
        #
        # num_cols = 2
        # slice_num = 91
        #
        # msk_channel = 1
        # img_channel = 0
        #
        # for img, msk in ds_train.take(1):
        #     bs = img.shape[0]
        #
        #     for idx in range(bs):
        #         plt.subplot(bs, num_cols, idx * num_cols + 1)
        #         plt.imshow(img[idx, :, :, slice_num, img_channel], cmap="bone")
        #         plt.title("MRI {}".format(brats_data.input_channels[str(img_channel)]), fontsize=18)
        #         plt.subplot(bs, num_cols, idx * num_cols + 2)
        #         plt.imshow(msk[idx, :, :, slice_num, msk_channel], cmap="bone")
        #         plt.title("Tumor {}".format(brats_data.output_channels[str(msk_channel)]), fontsize=18)
        #         plt.show()
        #         break
        # print("Mean pixel value = {}".format(np.mean(img[0, :, :, :, 0])))
        if stage == "fit" or stage is None:
            self.train_set = ds_train
            if self.tune_on_val:
                self.val_set = ds_val
                #TODO is this redundant to dataloader??
                train_indices, val_indices = train_val_split(
                    len(self.train_set), self.tune_on_val)
                self.train_set = Subset(self.train_set, train_indices)
                self.val_set = Subset(self.val_set, val_indices)
            else:
                self.val_set = ds_val


        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = ds_test


        if stage == "attack":
            ori_train_set = ds_train
            self.attack_indices, self.class2attacksample = extract_attack_set(
                ori_train_set)
            self.train_set = Subset(ori_train_set, self.attack_indices)
            self.test_set = Subset(self.test_set, range(100))
        elif stage == "attack_mini":
            ori_train_set = ds_train
            self.attack_indices, self.class2attacksample = extract_attack_set(
                ori_train_set, sample_per_class=2)
            self.train_set = Subset(ori_train_set, self.attack_indices)
            self.test_set = Subset(self.test_set, range(100))
        elif stage == "attack_large":
            ori_train_set = ds_train
            self.attack_indices, self.class2attacksample = extract_attack_set(
                ori_train_set, sample_per_class=500)
            self.train_set = Subset(ori_train_set, self.attack_indices)
            self.test_set = Subset(self.test_set, range(100))

    def train_dataloader(self):
        if self.batch_sampler is None:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            )
        else:
            return DataLoader(
                self.train_set,
                batch_sampler=self.batch_sampler,
                num_workers=self.num_workers,
                shuffle=True,
            )

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


def train_val_split(dataset_size: int, val_train_split: float = 0.02):
    validation_split = int((1 - val_train_split) * dataset_size)
    train_indices = range(dataset_size)
    train_indices, val_indices = (
        train_indices[:validation_split],
        train_indices[validation_split:],
    )
    return train_indices, val_indices


def extract_attack_set(
    dataset: Dataset,
    sample_per_class: int = 5,
    multi_class=True,
    total_num_samples: int = 50,
    seed: int = None,
):
    if not multi_class:
        #######################################################################
        num_classes = len(dataset.classes)
        class2sample = {i: [] for i in range(num_classes)}
        select_indices = []
        if seed == None:
            index_pool = range(len(dataset))
        else:
            index_pool = np.random.RandomState(seed=seed).permutation(
                len(dataset))
        for i in index_pool:
            current_class = dataset[i][1]
            if len(class2sample[current_class]) < sample_per_class:
                class2sample[current_class].append(i)
                select_indices.append(i)
            elif len(select_indices) == sample_per_class * num_classes:
                break
        return select_indices, class2sample
    else:
        select_indices = range(total_num_samples)
        class2sample = None
        return select_indices, class2sample