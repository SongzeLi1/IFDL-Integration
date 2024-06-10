import random

from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
import numpy as np
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch

#############  dataset ##################
max_size_w = 512
max_size_h = 512


# max_size_w = 128
# max_size_h = 192
def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask = mask[:, :, 0:1]
    mask[mask <= 127.5] = 0.0
    mask[mask > 127.5] = 255.
    # mask[mask<=127.5] = 255.
    # mask[mask>127.5] = 0.0
    return mask


def random_transform(image):
    h, w, _ = image.shape
    # 随机选择小区域的大小和位置
    crop_width, crop_height = random.randint(10, 100), random.randint(10, 100)
    start_x = random.randint(0, w - crop_width)
    start_y = random.randint(0, h - crop_height)

    # 裁剪图像
    crop_img = image[start_y:start_y + crop_height, start_x:start_x + crop_width]

    # 随机翻转角度
    angles = [0, 10, 20, 30, 45, 60, 90, 120, 180, 200, 270]
    angle = random.choice(angles)
    M = cv2.getRotationMatrix2D((crop_width / 2, crop_height / 2), angle, 1)
    crop_img = cv2.warpAffine(crop_img, M, (crop_width, crop_height))

    # 随机压缩比例
    scale = random.uniform(0.7, 1.0)  # 压缩比例在50%到100%之间
    new_width = int(crop_width * scale)
    new_height = int(crop_height * scale)
    crop_img = cv2.resize(crop_img, (new_width, new_height))

    # 选择新的粘贴位置
    paste_x = random.randint(0, w - new_width)
    paste_y = random.randint(0, h - new_height)

    # 创建结果图像和二值图标签
    result_image = image.copy()
    binary_label = np.zeros((h, w), dtype=np.uint8)

    # 粘贴处理后的区域到新图像和标签图
    result_image[paste_y:paste_y + new_height, paste_x:paste_x + new_width] = crop_img
    binary_label[paste_y:paste_y + new_height, paste_x:paste_x + new_width] = 255
    # 将二值图标签维度变为[1, H, W]
    binary_label = binary_label[:, :, np.newaxis]

    return result_image, binary_label


class UNetDataset(Dataset):
    def  __init__(self, dirs_train, dirs_mask, train_transform=None, val_transform=None, mode='train', data_type=1):
        super(UNetDataset, self).__init__()
        self.mode = mode
        self.type = data_type
        self.dataTrain = []
        self.dataMask = []

        # # # # add # # # #
        # only tamper or orig
        if self.type == 0:
            for dir_train in dirs_train:

                # search for error path
                # for filename in os.listdir(dir_train):
                #     if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.tif'):
                #         pass
                #     else:
                #         print(os.path.join(dir_train, filename))

                self.dataTrain.extend([os.path.join(dir_train, filename)
                                       for filename in os.listdir(dir_train)
                                       if
                                       filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.tif')])
            self.dataTrain.sort()
            # self.dataTrain = self.dataTrain[:40]

            for dir_mask in dirs_mask:
                self.dataMask.extend([os.path.join(dir_mask, filename)
                                      for filename in os.listdir(dir_mask)
                                      if
                                      filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.tif')])
            self.dataMask.sort()
            # self.dataMask = self.dataMask[:40]
        #####

        # tamper + 30% orig
        else:
            # 处理每一对目录
            for i, (dir_train, dir_mask) in enumerate(zip(dirs_train, dirs_mask)):
                # 读取并处理训练目录的文件名
                train_files = [filename for filename in os.listdir(dir_train) if
                                   filename.endswith(('.jpg', '.png', '.tif'))]


                if i == 0:
                    # 对于第一组目录，使用所有匹配的文件
                    selected_files = train_files
                else:
                    # 对于其他组目录，随机选取30%的匹配文件
                    sample_size = int(len(train_files) * 0.5)
                    selected_files = random.sample(train_files, sample_size)

                # 将选定的文件路径添加到训练和掩码数据列表
                self.dataTrain.extend(os.path.join(dir_train, file) for file in selected_files)
                self.dataMask.extend(os.path.join(dir_mask, file.replace('ps_', 'ms_')) for file in selected_files)

        self.dataTrain.sort()
        self.dataMask.sort()

        # # # # # ----- # # # # #
        self.trainDataSize = len(self.dataTrain)
        self.maskDataSize = len(self.dataMask)

        self.transform1 = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.toTensor = A.Compose([ToTensorV2()])
        if train_transform is not None:
            self.train_transform = train_transform
        else:
            self.train_transform = A.Compose(
                [
                    A.Resize(max_size_h, max_size_w, p=1),
                    # A.RandomCrop(width=max_size, height=max_size,p=1),
                    A.VerticalFlip(p=0.2),  # p--> 1:0.2 2:0.3     (orig: p=0.2)
                    # A.RandomRotate90(p=0.2),    # 2:0.2           (orig: p=0.5)
                    A.HorizontalFlip(p=0.2),  # p--> 1:0.2 2:0.3   (orig: p=0.2)
                    # A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5),   #
                    # A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),  #
                    # A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), p=1),  # 2
                    ToTensorV2(),
                ]
            )
        if val_transform is not None:
            self.val_transform = val_transform
        else:
            self.val_transform = A.Compose(
                [
                    A.Resize(max_size_h, max_size_w, p=1),
                    # A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), p=1),
                    ToTensorV2(),
                ]
            )

        self.kernel = np.ones((4, 4), np.uint8)
        self.feature_chan = [128, 64, 32, 16]

    def __getitem__(self, index):
        assert self.trainDataSize == self.maskDataSize
        image_filename = self.dataTrain[index]
        image = cv2.imread(image_filename)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape

        mask_filename = self.dataMask[index]
        mask = cv2.imread(mask_filename)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        mask = preprocess_mask(mask)

        # # only orig
        # if self.mode == 'train':
        #     image, mask = random_transform(image)   # 应用随机图像处理

        if self.mode == 'train':
            transformed = self.train_transform(image=image, mask=mask)
            image = transformed["image"]
            final_mask = transformed["mask"]

        else:
            transformed = self.val_transform(image=image, mask=mask)
            image = transformed["image"]
            final_mask = transformed["mask"]

        image = image.float().div(255)
        final_mask = final_mask.float().div(255)
        final_mask = final_mask.permute(2, 0, 1)

        if self.mode == 'train':
            return image, final_mask, image_filename
        if self.mode == 'val':
            return image, final_mask, image_filename
        if self.mode == 'predict':
            return image, final_mask, w, h, image_filename

    def __len__(self):
        return self.trainDataSize
