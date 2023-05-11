import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import torch.nn as nn
criterion = nn.SmoothL1Loss()

class ValDataset(Dataset):
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

        val_data = {}
        numpy_image = os.path.join(self.audio_image_dir,
                                   self.key_pts_frame.iloc[idx, 0].split("/")[-2],
                                   self.key_pts_frame.iloc[idx, 0].split("/")[-1])
        image = np.load(numpy_image)
        # #加载图片后，判断该图片是否存在于val_data的关键字中
        # #不存在则要再加一个关键字，存在则添加在原来的关键字那里
        # #file_name_img = 音频名+通道数
        img_name = (numpy_image.split("/")[-1]).split("-")[0] + "-----"   + numpy_image.split("_")[-1].split(".")[0]
        # #判段file_name_img是否在val_data的keys中，不存在则添加，存在则添加在原来keys的append中
        # if file_name_img in val_data.keys():
        #     val_data[file_name_img] =


        # key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        DDR_each_band = self.key_pts_frame.iloc[idx, 1:31].values
        T60_each_band = self.key_pts_frame.iloc[idx, 61:91].values
        # print("numpy image:", numpy_image, "ddr:", DDR_each_band, "t60:", T60_each_band)


        sample = {'image': image, 'ddr': DDR_each_band, 't60': T60_each_band,'img_name':img_name}
        # sample = {'image':image , 'ddr':np.asarray(DDR_each_band)}

        # if self.transform:
        #     sample = self.transform(sample)
        totensor = ToTensor()
        sample = totensor(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image, ddr, t60,img_name = sample['image'], sample['ddr'], sample['t60'],sample["img_name"]
        image = np.expand_dims(image, 0)
        ddr = ddr.astype(float)
        t60 = t60.astype(float)
        # image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'ddr': torch.from_numpy(ddr),
                't60': torch.from_numpy(t60),
                "img_name":img_name}


class Val_meanT60():


    def __call__(self,epoch,net,val_loader,lr,device,save_name):
        # 因为我要在验证时，对同一段语音的同一通道数下的T60求mean
        # 所以这就需要让你的数据带有四个keys,要包含img_name(音频名字+通道数)
        import numpy as np

        count_num = 0
        Mse_all_img_way1 = 0
        Mse_all_img_way2 = 0
        Bias_all_img_way1 = 0
        Bias_all_img_way2 = 0
        relative_loss_way1 = 0
        relative_loss_way2 = 0
        # criterion = torch.nn.MSELoss(reduce=False,size_average=False)
        criterion = torch.nn.MSELoss()
        total_mean_loss = 0
        total_mean_bias = 0
        sameRoomMse = {}
        sameRoomBias = {}
        with torch.no_grad():
            for k, datas in val_loader.items():


                # 遍历同一个声源的内容，并作为一个batch_size送入网络
                input = []
                label = []

                for data in datas:
                    if data == 0:
                        continue
                    images = data['image']
                    ddr = data['ddr']
                    meanT60 = data['t60']
                    input.append(images)
                    label.append(meanT60)
                images = torch.stack(input, dim=0)
                meanT60 = torch.stack(label, dim=0)
                # num += images.shape[0]
                # total_num += images.shape[0]
                meanT60 = torch.tensor(meanT60.clone().detach().numpy(), dtype=torch.float32, device=device)
                images = torch.tensor(images.clone().detach().numpy(), dtype=torch.float32, device=device)
                # output_pts = net(images)
                # if epoch == 0:
                h_n = torch.randn((1, 98, 20), device=device, dtype=torch.float32)
                h_c = torch.randn((1, 98, 20), device=device, dtype=torch.float32)
                output_pts, h_n, h_c = net(images, h_n, h_c)
                loss = criterion(output_pts, meanT60)
                # 创建一个字典来存储T60,格式是{str:[Tensor]}
                count_num += 1
                #输出一个bias
                # bias = torch.sum((meanT60 - output_pts)) / output_pts.shape[0] / output_pts.shape[1]
                #输出多个bias
                loss = torch.sum(torch.square((meanT60 - output_pts)),dim=0) / output_pts.shape[0]
                bias = torch.sum((meanT60 - output_pts), dim=0) / output_pts.shape[0]
                total_mean_loss += loss
                # bias_scalars = [bias_i.item() for bias_i in bias]
                total_mean_bias += bias
                split_name = k.split("_")
                img_names = split_name[1] + "_" + split_name[2]
                if img_names in ["Lecture_Room","Meeting_Room"]:
                    img_names = img_names + "_" + split_name[3]
                #下面开始创建dict,存储不同房间的bias和mse
                if img_names in sameRoomBias.keys():
                    sameRoomBias[img_names][0] += bias
                    sameRoomBias[img_names][1] += 1
                else:
                    sameRoomBias[img_names] = [bias,0]
                #mse
                if img_names in sameRoomMse.keys():
                    sameRoomMse[img_names][0] += loss
                    sameRoomMse[img_names][1] += 1
                else:
                    sameRoomMse[img_names] = [loss,0]
            # mean_loss = total_mean_loss / count_num
            # mean_bias = total_mean_bias / count_num
            # print("in val,epoch {},mse is {},bias is {}".format(epoch, mean_loss, mean_bias))
            print("finish evaling")
            print("\n")
            print("starting saving")
            #要转为平均值
            for k,v in sameRoomMse.items():
                sameRoomMse[k] = sameRoomMse[k][0] / sameRoomMse[k][1]
            for k,v in sameRoomBias.items():
                sameRoomBias[k] = sameRoomBias[k][0] / sameRoomBias[k][1]
            if False:
                os.makedirs("/data2/cql/code/cnnLstmPredictT60/trainAndval/")
            txt_file = "/data2/cql/code/cnnLstmPredictT60/trainAndval/eval_AddTIMIT_fullT60_" + save_name+ "_meanT60.txt"
            with open(txt_file,"a") as f:
                f.write("\n")
                f.write("epoch:{}".format(epoch) + "    ")
                f.write("lr:{}".format(lr) + "    ")
                f.write("mse:{}".format(sameRoomMse) + "    ")
                f.write("\n")
                f.write("bias:{}".format(sameRoomBias) + "    ")
                f.write("\n")
                f.close()
            return sameRoomMse,sameRoomBias