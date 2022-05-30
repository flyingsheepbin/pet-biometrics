import cv2
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_infer import Model, set_bn_eval


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class infer_dataset2(Dataset):
    def __init__(self, path):
        super().__init__()
        # self.transform = transforms.Compose([
        #     transforms.Resize(224),
        #     transforms.CenterCrop(224),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # 读取标签文件
        self.train_file = pd.read_csv('../data/validation/valid_data.csv').values
        self.image_path = '../data/validation/images/'

    def __len__(self):
        return len(self.train_file)

    # 返回狗的图片和狗的id
    def __getitem__(self, idx):
        img1, img2 = self.train_file[idx]

        img1 = cv2.imdecode(np.fromfile(self.image_path+img1, dtype=np.uint8), 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = torch.from_numpy(img1).permute(2, 0, 1) / 255
        img1 = self.transform(img1)

        img2 = cv2.imdecode(np.fromfile(self.image_path+img2, dtype=np.uint8), 1)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = torch.from_numpy(img2).permute(2, 0, 1) / 255
        img2 = self.transform(img2)
        return img1, img2


batch_size = 64
# 加载数据集
dataset = infer_dataset2("train")
test_data = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda:0")
# 加载模型
model = Model("effb7", "SM", 512, num_classes=6000)
model.apply(set_bn_eval)
model.load_state_dict(torch.load('results/car_uncropped_resnet50_SM_512_0.1_0.5_0.15_32_model.pth'),
                      strict=False)
model.to(device)
model.eval()
pred_list = []

with torch.no_grad():
    for i, (img1, img2) in tqdm(enumerate(test_data)):
        feature1 = model(img1.to(device))
        feature2 = model(img2.to(device))
        for x in range(len(feature1)):
            similarity = torch.cosine_similarity(feature1[x], feature2[x], dim=0)
            pred_list.append(similarity.item())
    print(len(pred_list))
    val_pd = pd.read_csv('../data/validation/valid_data.csv')
    val_pd['prediction'] = pred_list
    val_pd.to_csv('result.csv', index=False)


