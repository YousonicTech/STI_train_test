import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchvision.transforms import Normalize
import random
from torchvision import transforms, utils


class Dataset_dict(Dataset):

    def __init__(self, root_dir=None, transform=None, start_freq=0, end_freq=29, failed_file=".",
                 random_choose_slice=True, channel3=True):
        """
        数据集的类，继承自torch.utils.data.Dataset。

        Args:
            root_dir (str): 数据集所在的根目录。
            transform (callable, optional): 可调用对象，表示对样本进行的变换操作。默认为None。
            start_freq (int): 采样频率的起点。默认为0。
            end_freq (int): 采样频率的终点。默认为29。
            failed_file (str): 读取失败时使用的默认文件路径。默认为"."。
            random_choose_slice (bool): 是否随机选择一个切片。默认为True。
            channel3 (bool): 是否将通道数增加到3，以满足一些图像模型的3通道输入。默认为True。
        """
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.root_dir = root_dir
        self.transform = transform
        self.failed_file = failed_file
        self.normalize = Normalize(mean=[0.5], std=[0.5])
        self.random_choose_slice = random_choose_slice
        self.channel3 = channel3
        self.dict_data = []
        self.sample = []

        print("Root directory:", root_dir)

        for (root, dirs, files) in os.walk(root_dir):
            print("root:", root, "dirs:", dirs, "Number of files:", len(files))

            self.dict_data.extend([os.path.join(root, filen) for filen in files])
            self.root = root

    def __len__(self):
        """
               获取数据集的长度。
               Returns:
                   int: 数据集的长度。
        """
        return len(self.dict_data)

    def __getitem__(self, idx):
        images_list = []
        clean_list = []
        sti_list = []
        name_list = []

        dict_data_name = self.dict_data[idx]
        try:
            audio_image_dict = torch.load(dict_data_name)
        except:
            print("Failed to read file:", dict_data_name)
            audio_image_dict = torch.load(self.failed_file)

        dict_keys = list(audio_image_dict.keys())[0]
        dict_data = audio_image_dict[dict_keys]
        valid_info_count = 0

        for list_c in range(len(dict_data)):
            if dict_data[list_c] == 0:
                continue
            else:
                valid_info_count += 1
                if "image" in dict_data[list_c].keys():
                    temp_image = dict_data[list_c]['image']
                    temp_image = temp_image.unsqueeze(0)

                    temp_clean = dict_data[list_c]['clean']  # 干净语谱图
                    temp_clean = temp_clean.unsqueeze(0)

                    images_list.append(temp_image)
                    clean_list.append(temp_clean)

                    sti_list.append(dict_data[list_c]['t60'][10])
                    name_list.append(dict_data_name)

        images = torch.cat(images_list, dim=0).unsqueeze(1)  # [3, 1, 65, 175]
        cleans = torch.cat(clean_list, dim=0).unsqueeze(1)  # [3, 1, 65, 175]

        if self.transform:
            images = self.transform(images)  # images:[3, 1, 224, 224]
            cleans = self.transform(cleans)  # images:[3, 1, 224, 224]

            for i in range(images.shape[0]):
                # 先归一化到[0, 1], 相当于Totensor()
                temp_norm = images[i][0]
                temp_norm = (temp_norm - temp_norm.min()) / (temp_norm.max() - temp_norm.min())
                images[i][0] = temp_norm
            # Normalize到[-1, 1]
            images = self.normalize(images)

            for i in range(cleans.shape[0]):
                # 先归一化到[0, 1], 相当于Totensor()
                temp_norm = cleans[i][0]
                temp_norm = (temp_norm - temp_norm.min()) / (temp_norm.max() - temp_norm.min())
                cleans[i][0] = temp_norm
            # Normalize到[-1, 1]
            cleans = self.normalize(cleans)

        if self.random_choose_slice:
            slice_num = random.randint(0, int(images.shape[0]) - 1)
            images = images[slice_num].unsqueeze(0)
            cleans = cleans[slice_num].unsqueeze(0)
            sti_list = sti_list[slice_num].unsqueeze(0)
            valid_info_count = 1
            name_list = [name_list[slice_num]]

        if self.channel3:
            images = images.repeat(1, 3, 1, 1)
            cleans = cleans.repeat(1, 3, 1, 1)

        sample = {'image': images,
                  'sti': torch.Tensor(sti_list),
                  "validlen": valid_info_count,
                  'name': name_list,
                  'clean': cleans}

        return sample


padding_keys = []
stacking_keys = ['image','clean','sti']


def collate_fn(batch):
    keys = batch[0].keys()
    out = {k: [] for k in keys}
    
    for data in batch:
        for k, v in data.items():
            out[k].append(v)

    for k in stacking_keys:
        # print("key:",k)
        out[k] = torch.cat(out[k], dim=0)  # torch.stack(out[k],dim=0)

    for k in padding_keys:
        out[k] = pad_sequence(out[k], batch_first=True)
    return out


if __name__ == "__main__":
    train_dict_root = "/data/xbj/0929_STI_catTIMIT_withNOISE_DATA/train"
    data_transform = transforms.Compose([transforms.Resize([224, 224])])
    failed_file = "/data/xbj/0929_STI_catTIMIT_withNOISE_DATA/train$ ls Six_Config/Six_Config_BatteryBenson_left_Six_Config_DR4_FDKN0_TIMIT_S_1000dB-0.pt"

    train_transformed_dataset = Dataset_dict(root_dir=train_dict_root, transform=data_transform,
                                             start_freq=0, end_freq=29,
                                             failed_file=failed_file, random_choose_slice=False)
    train_loader = torch.utils.data.DataLoader(train_transformed_dataset,
                                               shuffle=False, num_workers=6,
                                               pin_memory=False,
                                               batch_size=4, drop_last=True,  # prefetch_factor=batch_size,
                                               collate_fn=collate_fn)


    # dict_keys=list(train_transformed_dataset.keys())[0]
    print(len(train_loader))

    for batch_i, datas in enumerate(train_loader):
        # print(datas['image'])
        print(datas['sti'].reshape(-1,1))

