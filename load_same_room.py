import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.audio_image_dir = self.root_dir

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        # image_name = os.path.join(self.root_dir,
        #                         self.key_pts_frame.iloc[idx, 0])
        #
        # image = mpimg.imread(image_name)

        numpy_image = os.path.join(self.audio_image_dir,
                                   self.key_pts_frame.iloc[idx, 0].split("/")[-2],
                                   self.key_pts_frame.iloc[idx, 0].split("/")[-1])
        image = np.load(numpy_image)

        DDR_each_band = self.key_pts_frame.iloc[idx, 1:31].values
        T60_each_band = self.key_pts_frame.iloc[idx, 61:91].values
        MeanT60_each_band = self.key_pts_frame.iloc[idx, -2:].values
        sample = {'image': image, 'ddr': DDR_each_band, 't60': T60_each_band, "MeanT60": MeanT60_each_band}
        # sample = {'image':image , 'ddr':np.asarray(DDR_each_band)}

        if self.transform:
            sample = self.transform(sample)

        return sample