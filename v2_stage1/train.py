import argparse
from utils import *
import pandas as pd
import torch
from thop import profile, clever_format
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from model_v2 import Model, set_bn_eval
from utils import recall, LabelSmoothingCrossEntropyLoss, BatchHardTripletLoss, ImageReader, MPerClassSampler
import os
from torch_ema import ExponentialMovingAverage
import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def copy_parameters_to_model(copy_of_model_parameters, model):
    for s_param, param in zip(copy_of_model_parameters, model.parameters()):
        if param.requires_grad:
            param.data.copy_(s_param.data)

def copy_parameters_from_model(model):
    copy_of_model_parameters = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    return copy_of_model_parameters 

def train(net, optim, ema):
    net.train()
    # fix bn on backbone network
    net.apply(set_bn_eval)
    total_loss, total_correct, total_num, data_bar = 0, 0, 0, tqdm(train_data_loader)
    for inputs, labels in data_bar:
        inputs, labels = inputs.cuda(), labels.cuda()
        #features, classes = net(inputs)
        features = net(inputs)

        feature_loss = feature_criterion(features, labels)
        # loss = class_loss + feature_loss*10.0
        loss = feature_loss
        total_loss += loss.item() * inputs.size(0)
        total_num += inputs.size(0)
        del inputs, labels, features
        torch.cuda.empty_cache()
        optim.zero_grad()
        loss.backward()
        optim.step()
   
        data_bar.set_description('Train Epoch {}/{} - Loss:{:.4f}%'
                                 .format(epoch, num_epochs, total_loss / total_num))
        if ema:
            ema.update(net.parameters())
    return total_loss / total_num, total_correct / total_num * 100


def test1(net, recall_ids):
    net.eval()
    with torch.no_grad():
        # obtain feature vectors for all data
        for key in eval_dict.keys():
            eval_dict[key]['features'] = []
            for inputs, labels in tqdm(eval_dict[key]['data_loader'], desc='processing {} data'.format(key)):
                inputs, labels = inputs.cuda(), labels.cuda()
                #features, classes = net(inputs)
                features = net(inputs)
                eval_dict[key]['features'].append(features)
                del inputs, labels, features
                torch.cuda.empty_cache()
            eval_dict[key]['features'] = torch.cat(eval_dict[key]['features'], dim=0)

        # compute recall metric
        acc_list = recall(eval_dict['test']['features'], test_data_set.labels, recall_ids)
    desc = 'Test Epoch {}/{} '.format(epoch, num_epochs)
    for index, rank_id in enumerate(recall_ids):
        desc += 'R@{}:{:.2f}% '.format(rank_id, acc_list[index] * 100)
        results['test_recall@{}'.format(rank_id)].append(acc_list[index] * 100)
    print(desc)
    return acc_list[0]
def test1_(net, recall_ids):
    net.eval()
    with torch.no_grad():
        # obtain feature vectors for all data
        total_correct,total_num=0,0
        total_loss=0
        for key in eval_dict.keys():
            eval_dict[key]['features'] = []
            for inputs, labels in tqdm(eval_dict[key]['data_loader'], desc='processing {} data'.format(key)):
                inputs, labels = inputs.cuda(), labels.cuda()
                features= net(inputs)
                total_num += inputs.size(0)

                supcon_loss = feature_criterion(features,labels)
                total_loss += supcon_loss.item() * inputs.size(0)
                del inputs, labels, features
                torch.cuda.empty_cache()
    print("epoch: {}    validaiton loss: {:4f} ".format(epoch,total_loss / total_num))
    return total_loss / total_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CGD')

    parser.add_argument('--feature_dim', default=512, type=int, help='feature dim')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    parser.add_argument('--batch_size', default=40, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=30, type=int, help='train epoch number')  # 50轮就差不多了
    parser.add_argument('--image_size', default=224, type=int, help='image size')
    opt = parser.parse_args()


    recalls, batch_size = [int(k) for k in opt.recalls.split(',')], opt.batch_size
    num_epochs = opt.num_epochs
    save_name_pre = ''.format('v2')

    results = {'train_loss': [], 'train_accuracy': []}
    for recall_id in recalls:
        results['test_recall@{}'.format(recall_id)] = []
    image_size=opt.image_size
    # dataset loader
    data_transforms = {
        "train": A.Compose([
            A.Resize(60, 60,p=0.45),
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.),
        "valid": A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.)
    }

    train_data_set = ImageReader(data_transforms["train"])
    train_sample = MPerClassSampler(train_data_set.labels, batch_size)
    train_data_loader = DataLoader(train_data_set, batch_sampler=train_sample, num_workers=4)

    test_data_set = ImageReader(data_transforms["valid"])
    test_data_loader = DataLoader(test_data_set, batch_size, shuffle=False, num_workers=4)
    eval_dict = {'test': {'data_loader': test_data_loader}}

    model = Model(512, num_classes=6000).cuda()
    iters = len(train_data_loader)
    ema_decay = 0.5**(1/iters) # 0.3 for middle/big dataset, increace when use low amount sample
    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    optimizer = Adam(model.parameters(), lr=5e-5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=29, eta_min=1e-6)
    feature_criterion = SupConLoss_clear()

    best_recall = 0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(model, optimizer, ema)
        copy_of_model_parameters = copy_parameters_from_model(model)
        ema.copy_to(model.parameters())
        rank = test1(model, recalls)
        lr_scheduler.step()

        data_base = {}
        if rank > best_recall:
            best_recall = rank
            torch.save(model.state_dict(), 'results/car_uncropped_resnet50_SM_512_0.1_0.5_0.15_32_model.pth')
        copy_parameters_to_model(copy_of_model_parameters, model)
