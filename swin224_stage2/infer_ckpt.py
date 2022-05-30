import cv2
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_infer import Model, set_bn_eval

from albumentations.pytorch import ToTensorV2

import os

import albumentations as A
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class infer_dataset2(Dataset):
    def __init__(self, path):
        super().__init__()
        # self.transform = transforms.Compose([
        #     transforms.Resize(224),
        #     transforms.CenterCrop(224),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        self.transform =  A.Compose([A.Resize(224, 224),
                                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                            max_pixel_value=255.0, p=1.0), ToTensorV2()], p=1.)
        self.transform2 = A.Compose([
                A.OneOf([
                    A.Resize(60, 60, p=0.6),
                    A.Resize(60, 84, p=0.2),
                    A.Resize(47, 60, p=0.2),
                    ], p=0.45),
                A.OneOf([
                    A.Compose([
                        A.Resize(240, 240),
                        A.RandomCrop(224,224),
                        ], p=0.2),
                    A.Resize(224,224),
                    ], p=1.),
                #A.ImageCompression(quality_lower=80, quality_upper=100, p=0.75),
                #A.MotionBlur(blur_limit=40, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],max_pixel_value=255.0,p=1.0),ToTensorV2()], p=1.)
        # 读取标签文件
        self.train_file = pd.read_csv('../data/test/test_data.csv').values
        self.image_path = '../data/test/test/'

    def __len__(self):
        return len(self.train_file)

    # 返回狗的图片和狗的id
    def __getitem__(self, idx):
        img1, img2 = self.train_file[idx]
        #img1 = img1.replace('*','_')
        #img2 = img2.replace('*','_')
        # img1 = cv2.imdecode(np.fromfile(self.image_path+img1, dtype=np.uint8), 1)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # img1 = torch.from_numpy(img1).permute(2, 0, 1) / 255
        # img1, img1_r = self.transform(img1), self.transform2(img1)

        img = cv2.imdecode(np.fromfile(self.image_path+img1, dtype=np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = self.transform(image=img)['image']
        img1_r = self.transform2(image=img)['image']

        # img2 = cv2.imdecode(np.fromfile(self.image_path+img2, dtype=np.uint8), 1)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # img2 = torch.from_numpy(img2).permute(2, 0, 1) / 255
        # img2, img2_r = self.transform(img2), self.transform(img2)
        img = cv2.imdecode(np.fromfile(self.image_path+img2, dtype=np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = self.transform(image=img)['image']
        img2_r = self.transform2(image=img)['image']
        return img1, img2, img1_r, img2_r


batch_size = 72
# 加载数据集
dataset = infer_dataset2("train")
test_data = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda:0")
# 加载模型
model = Model("effb7", "SM", 512, num_classes=6000)
model.apply(set_bn_eval)
model.load_state_dict(torch.load('../ckpts/swin224_stage2.pth'),
                      strict=False)
model.to(device)
model.eval()
pred_list = []

with torch.no_grad():
    for i, (img1, img2, img1_r, img2_r) in tqdm(enumerate(test_data)):
        feature1 = model(img1.to(device))
        feature2 = model(img2.to(device))

        feature1_r = model(img1_r.to(device))
        feature2_r = model(img2_r.to(device))
        for x in range(len(feature1)):
            
            similarity = torch.cosine_similarity(feature1[x], feature2[x], dim=0)
            sim2 = torch.cosine_similarity(feature1_r[x], feature2_r[x], dim=0)
            # print((similarity.item()+sim2.item())/2.0)
            # exit(0)
            pred_list.append((similarity.item()+sim2.item())/2.0)
    val_pd = pd.read_csv('../data/test/test_data.csv')
    val_pd['prediction'] = pred_list
    val_pd.to_csv('results.csv', index=False)


