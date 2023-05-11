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


    def __call__(self,writer,epoch,net,val_loader,lr,device):
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
                bias = torch.sum((meanT60 - output_pts)) / output_pts.shape[0] / output_pts.shape[1]
                total_mean_loss += loss.item()
                total_mean_bias += bias.item()
            mean_loss = total_mean_loss / count_num
            mean_bias = total_mean_bias / count_num
            print("in val,epoch {},mse is {},bias is {}".format(epoch, mean_loss, mean_bias))
            writer.add_scalar('val/mean_loss',mean_loss,epoch)
            writer.add_scalar('val/mean_bias',mean_bias,epoch)
            save_dir = "/data2/cql/code/cnnLstmPredictT60/trainAndval/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # with open("/data2/cql/code/cnnLstmPredictT60/trainAndval/autoLossWeight_eval_single_FullT60.txt", "a") as f:

            with open("/data2/cql/code/cnnLstmPredictT60/trainAndval/single_FullT60_rirTimitNoise.txt","a") as f:
                f.write("\n")
                f.write("epoch:{}".format(epoch) + "    ")
                f.write("lr:{}".format(lr) + "    ")
                f.write("mse:{}".format(mean_loss) + "    ")
                f.write("bias:{}".format(mean_bias) + "    ")
                f.write("\n")
                f.close()
                # f.write("mse_way1:{}".format(Mse_all_img_way1 / count_num) + "    ")
                # f.write("mse_way2:{}".format(Mse_all_img_way2 / count_num) + "    ")
                # f.write("\n")
                # f.write("bias_way1:{}".format(Bias_all_img_way1 / count_num) + "    ")
                # f.write("bias_way2:{}".format(Bias_all_img_way2 / count_num) + "    ")
                # f.write("\n")
                # f.write("relative_loss_way1:{}".format(relative_loss_way1 / count_num) + "    ")
                # f.write("relative_loss_way2:{}".format(relative_loss_way2 / count_num) + "    ")
                # f.write("\n")
                # f.close()



            return mean_loss,mean_bias


# import numpy as np
        # Mse_eval_all_batch = 0.0
        # Bias_eval_all_batch = 0.0
        # Correlation_eval_all_batch = 0.0
        # count_eval = 0
        # for i, data in enumerate(val_loader):
        #     images = data["image"]
        #     t60 = data["t60"]
        #     # ddr = torch.tensor(ddr, dtype=torch.float32, device=device)
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #
        #
        #     count_num += 1
        #     t60 = torch.tensor(t60, dtype=torch.float32, device=device)
        #     images = torch.tensor(images, dtype=torch.float32, device=device)
        #     output_pts = net(images)
        #     loss = criterion(output_pts, t60)
        #     device = images.device
        #     # [0,1]表示第0行第一列,这个数值才表示相关系数
        #     correlation_eval_one_batch = np.abs(np.corrcoef(output_pts.cpu().detach().numpy(),
        #                                                     t60.cpu().detach().numpy())[0, 1])
        #     Bias_eval_all_batch += (loss / t60.shape[0]).item()
        #     Correlation_eval_all_batch += correlation_eval_one_batch.item()
        #     Mse_eval_OneBatch = torch.sum(torch.square(t60 - output_pts)) / t60.shape[0]
        #     Mse_eval_all_batch += Mse_eval_OneBatch.item()
        #     print("in evaluation,at epoch:{},loss is {},correlation is {}.".format(epoch, loss, correlation_eval_one_batch))
        # print("after evaluating net,correlation is :{},Mse is: {},Bias is:{}".format(Correlation_eval_all_batch/count_num,
        #                                                                              Mse_eval_all_batch/count_num,
        #                                                                              Bias_eval_all_batch/count_num))
        #在每做完一次epoch后将mse,bias,correlation存储起来，一定要除以count_num