import torch
import torchvision.transforms
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image


class Train_Valid_Dataset(Dataset):

    def __init__(self, csv_path, resize_height=224, resize_width=224, transform=None):
        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.to_tensor = transforms.ToTensor()  # 将数据转换成tensor形式
        self.transform = transform
        self.data_info = pd.read_csv(csv_path, header=None)  # header=None是去掉表头部分
        # 文件第一列包含图像文件名称
        self.image_arr = np.asarray(self.data_info.iloc[1:, 0])  # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始一直读取到最后一行
        # 第二列是图像的 label
        self.label_arr = np.asarray(self.data_info.iloc[1:, 2])
        # 计算 length
        self.data_len = len(self.data_info.index) - 1
        pass

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]
        img = Image.open(single_image_name)
        # 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        transform = transforms.Compose([
            transforms.Resize((self.resize_height, self.resize_width)),
            transforms.ToTensor()
        ])
        img = transform(img)
        # 得到图像的 label
        y_label = self.label_arr[index]
        y_label = np.array(y_label, np.dtype(np.int32))
        y_label = torch.from_numpy(y_label).type(torch.LongTensor)
        return img, y_label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.data_len
