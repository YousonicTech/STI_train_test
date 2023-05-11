# -*- coding: utf-8 -*-
"""
@file      :  train_resnet50_500Hz.py
@Time      :  2022/8/19 15:27
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""
#screen -L Logfile train.log python new_train_FPN_STI.py
# CUDA_VISIBLE_DEVICES="2" python xxxx.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import cv2
import numpy as np
from tqdm import tqdm, trange
import time
import torch
from AutomaticWeightedLoss import AutomaticWeightedLoss

torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.tensorboard import SummaryWriter

import glob
from data_load_sti import Dataset_dict, collate_fn
from SSIMLoss import ssim

from model.attentionFPN import FPN
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import Rescale, RandomCrop
from torchvision.transforms import Normalize, ToTensor
import torch.optim as optim
import datetime

import argparse

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Whatever')
    parser.add_argument('--trained_epoch', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='/data/zdm/code/0406_awl_STI/save_model/0428_FPN')
    parser.add_argument('--pretrain_model_path', type=str,
                        default='/data/zdm/code/0406_awl_STI/save_model/0428_FPN/t60_predict_model_99_fullT60_rir_timit_noise.pt')
    parser.add_argument('--train_batch_size', type=int, default=7)
    parser.add_argument('--val_batch_size', type=int, default=7)

    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--ln_out', type=int, default=1)
    parser.add_argument('--start_freq', type=int, default=0)
    parser.add_argument('--end_freq', type=int, default=29)

    parser.add_argument('--SERVER', type=bool, default=True)  # 在服务器上运行记得改成True

    # parser.add_argument('--which_freq', type=int, default=4)  # 换倍频程
    args = parser.parse_args()
    return args

def load_checkpoint(checkpoint_path=None, trained_epoch=None, model=None, device=None):
    save_model = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(save_model['model'])
    trained_epoch_load = save_model['epoch']
    print('model loaded from %s' % checkpoint_path)
    if not trained_epoch is None:
        return_epoch = trained_epoch
    else:
        return_epoch = trained_epoch_load
    return model, return_epoch

def val_net(net, epoch, val_loader, writer):
    print('val')
    with torch.no_grad():
        net.eval()
        total_mse_loss = 0
        total_ssim_loss = 0
        total_mean_loss = 0
        total_mean_bias = 0
        progress_bar = tqdm(val_loader)
        for j, datas in enumerate(progress_bar):
            images_reshape = datas['image'].to(torch.float32).to(device)
            valid_len = datas['validlen']
            gt_sti = datas['sti'].reshape(-1,1).to(torch.float32).to(device)
            clean_reshape = datas['clean'].to(torch.float32).to(device)
            output_pts, dereverb_out = net(images_reshape, valid_len)

            mse_loss = criterion(output_pts, gt_sti)
            ssim_loss = (1 - ssim(dereverb_out, clean_reshape))
            loss = awl(mse_loss, ssim_loss)

            bias = torch.sum((gt_sti - output_pts)) / output_pts.shape[0]

            total_mean_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_ssim_loss += ssim_loss.item()
            total_mean_bias += bias.item()

        mean_loss = total_mean_loss / len(val_loader)
        mean_bias = float(total_mean_bias) / len(val_loader)
        mean_mse_loss = total_mse_loss / len(val_loader)
        mean_ssim_loss = total_ssim_loss / len(val_loader)
        writer.add_scalar('val/mean_loss', mean_loss, epoch)
        writer.add_scalar('val/mean_bias', mean_bias, epoch)
        writer.add_scalar('val/mean_mse_loss', mean_mse_loss, epoch)
        writer.add_scalar('val/mean_ssim_loss', mean_ssim_loss, epoch)

        print("Epoch %d Evaluation: mean loss:%f,mean bias:%f" % (epoch, mean_loss, mean_bias))


def train_net(start_epoch, n_epochs, train_loader, val_loader, args):
    model_dir = args.save_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = torch.optim.lr_scheduler.StepLR(optimizer,
                                         step_size=25,
                                         gamma=0.5, last_epoch=start_epoch)

    print("lr at beginning:", lr.get_last_lr())
    net.train()
    lr_list = list()
    print("the training process from epoch{}...".format(start_epoch))
    dt = datetime.datetime.now()

    log_dir = os.path.join("./log", dt.strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    for epoch in range(start_epoch, n_epochs):
        print("Training:epoch ", epoch, "lr:", lr.get_last_lr()[-1])
        lr_list.append(lr.get_last_lr())
        total_mse_loss = 0
        total_ssim_loss = 0
        total_mean_loss = 0
        total_mean_bias = 0
        progress_bar = tqdm(train_loader)

        # check val
        # val_net(net, epoch, val_loader, writer)

        for batch_i, datas in enumerate(progress_bar):
            # images_reshape = [batch, 3, 224, 224]
            images_reshape = datas['image'].to(torch.float32).to(device)

            clean_reshape = datas['clean'].to(torch.float32).to(device)
            valid_len = datas['validlen']
            gt_sti = datas['sti'].reshape(-1, 1).to(torch.float32).to(device) # [total_slices, 1]
            output_pts, dereverb_out = net(images_reshape, valid_len)  # [total_slice, 1], [total_slice, 3, 224, 224]

            mse_loss = criterion(output_pts, gt_sti)
            ssim_loss = (1 - ssim(dereverb_out, clean_reshape))

            loss = awl(mse_loss, ssim_loss)
            # print('output_pts', output_pts, 'loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=0.1, norm_type=2)

            optimizer.step()
            bias = torch.sum((gt_sti - output_pts)) / output_pts.shape[0]  # /sum(valid_len)
            total_mean_loss = total_mean_loss + loss.item()
            total_mean_bias = total_mean_bias + bias.item()
            total_mse_loss += mse_loss.item()
            total_ssim_loss += ssim_loss.item()

        mean_loss = total_mean_loss / len(train_loader)
        mean_bias = total_mean_bias / len(train_loader)
        mean_mse_loss = total_mse_loss / len(train_loader)
        mean_ssim_loss = total_ssim_loss / len(train_loader)
        writer.add_scalar('train/mean_loss', mean_loss, epoch)
        writer.add_scalar('train/mean_mse_loss', mean_mse_loss, epoch)
        writer.add_scalar('train/mean_ssim_loss', mean_ssim_loss, epoch)
        writer.add_scalar('train/mean_bias', mean_bias, epoch)
        writer.add_scalar('lr/lr', lr.get_last_lr()[-1], epoch)
        writer.add_scalar('lr/MSE param', awl.params[0], epoch)
        writer.add_scalar('lr/SSIM param', awl.params[1], epoch)

        print('in case no sti, last sti_gt is:', gt_sti)
        print("In training, epoch {},mse is {},bias is {}".format(epoch, mean_loss, mean_bias))
        lr.step()

        if epoch % 1 == 0:
            print("eval")
            # net.eval()
            # val_net(net, epoch, val_loader, writer)

        if True:
            # './Checkpoints/rir_timit_noise_new_0509_alldata/'
            model_name = 'sti_predict_model_%d_full_rir_timit_noise.pt' % (epoch)

            # after training, save your model parameters in the dir 'saved_models'
            state = {"model": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                     "lr": lr.state_dict(), "awl": awl.params}
            torch.save(state, os.path.join(model_dir, model_name))
            print('Finished Training')

    writer.close()

if __name__ == "__main__":

    args = parse_args()
    DEBUG = args.SERVER
    LOAD_PRETRAIN = args.load_pretrain  # False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_model_path = args.pretrain_model_path
    net = FPN(num_blocks=[2, 4, 23, 3], num_classes=3, back_bone="resnet50", pretrained=False)
    awl = AutomaticWeightedLoss(2)

    ### 这个判断的作用，用来计时
    if LOAD_PRETRAIN == True:
        start_time = time.time()
        net, trained_epoch = load_checkpoint(pretrain_model_path, 99, net, device)
        print('Successfully Loaded model: {}'.format(pretrain_model_path))
        print('Finished Initialization in {:.3f}s!!!'.format(
            time.time() - start_time))
    else:
        trained_epoch = 0
    net.to(device)

    criterion = torch.nn.MSELoss()
    # optimizer = optim.Adam([{'params': net.parameters(), 'initial_lr': 1e-5}], lr=1e-5, weight_decay=0.000001)
    optimizer = optim.Adam([
        {'params': net.parameters(), 'initial_lr': 1e-5},
        {'params': awl.parameters(), 'weight_decay': 0, 'initial_lr': 1e-5}
    ], lr=1e-5, weight_decay=0.000001)
    n_epochs = 100
    data_transform = transforms.Compose([transforms.Resize([224, 224])])


    # 网络参数数量的作用
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {"Total": total_num, "Trainable": trainable_num}


    print(get_parameter_number(net))

    if DEBUG == False:
        train_dict_root = "/Users/bajianxiang/Desktop/internship/STI_dataset"
        val_dict_root = "/Users/bajianxiang/Desktop/internship/STI_dataset"
        batch_size = 3
        val_batch_size = 3
        failed_file = "/Users/bajianxiang/Desktop/internship/STI_dataset/Nine_Config_Amaranth_left_Nine_Config_DR4_FKDW0_SI1891_TIMIT_S_1000dB-0.pt"
    else:
        train_dict_root = "/data/xbj/0929_STI_catTIMIT_withNOISE_DATA/train"
        # train_dict_root = "/mnt/sda/xbj/0815_GEN_DATASET/train"
        val_dict_root = "/data/xbj/0929_STI_catTIMIT_withNOISE_DATA/val"
        batch_size = args.train_batch_size
        val_batch_size = args.val_batch_size
        failed_file = "/data/xbj/0929_STI_catTIMIT_withNOISE_DATA/train/Six_Config/Six_Config_BatteryBenson_left_Six_Config_DR4_FDKN0_TIMIT_S_1000dB-0.pt"

    print("train_dir:", train_dict_root)

    train_transformed_dataset = Dataset_dict(root_dir=train_dict_root, transform=data_transform,
                                             start_freq=args.start_freq, end_freq=args.end_freq,
                                             failed_file=failed_file, random_choose_slice=False)

    print("len of train dataset:", len(train_transformed_dataset))

    val_transformed_dataset = Dataset_dict(root_dir=val_dict_root, transform=data_transform, start_freq=args.start_freq,
                                           end_freq=args.end_freq, failed_file=failed_file, random_choose_slice=False)

    print("len of val dataset:", len(val_transformed_dataset))

    if DEBUG == False:
        train_loader = torch.utils.data.DataLoader(train_transformed_dataset,
                                                   shuffle=False, num_workers=1,
                                                   batch_size=batch_size, drop_last=True,
                                                   collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_transformed_dataset, shuffle=False, num_workers=1,
                                                 batch_size=val_batch_size, drop_last=True,
                                                 collate_fn=collate_fn)
    else:
        train_loader = torch.utils.data.DataLoader(train_transformed_dataset,
                                                   shuffle=True, num_workers=6,
                                                   batch_size=batch_size, drop_last=True,  # prefetch_factor=batch_size,
                                                   collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_transformed_dataset,
                                                 shuffle=False, num_workers=1,
                                                 batch_size=val_batch_size, drop_last=True, prefetch_factor=100,
                                                 collate_fn=collate_fn)
    print("after train loader init")
    trained_epoch = args.trained_epoch

    # save test
    """ state = {"model": net.state_dict(), "optimizer": optimizer.state_dict(), "awl": awl.params}
    model_dir = args.save_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = 'test.pt'
    torch.save(state, os.path.join(model_dir, model_name)) """

    #train_net(trained_epoch, n_epochs, train_loader, val_loader, args)

    # val test
    log_dir = os.path.join("./log", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    val_net(net, n_epochs, val_loader, writer)
    # TODO 训练时要改dict，权重名字，验证时保存的文件名和路径

