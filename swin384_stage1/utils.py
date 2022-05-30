import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import pandas as pd
import cv2
from torch.autograd import Variable
import albumentations as A
import os
import random

class ImageReader(Dataset):

    def __init__(self, data_path, data_name, data_type, crop_type):

        train_data = pd.read_csv('../data/train/train_data.csv')
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if data_type == 'train':
            self.transform = transforms.Compose([transforms.Resize((384, 384)), normalize])
        else:
            self.transform = transforms.Compose([transforms.Resize((384, 384)), normalize])
        self.imgs, self.labels = [], []
        for sample in train_data.values:
            self.imgs.append(sample[1])
            self.labels.append(sample[0])
        # 读取测试集一半的图片
        val_imgs = []
        val_labels = []
        count = 6000
        test_data = pd.read_csv('../pseudo_produce/pseudo.csv')
        for sample2 in test_data.values:
            # img, img2, pred
            # print(sample2)
            val_imgs += [sample2[0], sample2[1]]
            val_labels += [count, count]
            count += 1
            if count > 6499:
                break
        self.imgs += val_imgs
        self.labels += val_labels
        # tmp = np.sqrt(1 / np.sqrt(train_data['dog ID'].value_counts().sort_index().values))
        # self.margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

    def __getitem__(self, index):
        label = self.labels[index]
        imageName = self.imgs[index]
        name = '../data/train/images/' + imageName
        if not os.path.exists(name):
            name = '../data/validation/images/' + imageName
        # img = cv2.imdecode(np.fromfile('/home/kmyh/libin/dataset/pet_biometric_challenge_2022/train+test/images/'+imageName, dtype=np.uint8), 1)
        img = cv2.imdecode(np.fromfile(name, dtype=np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1) / 255
        if random.random() < 0.45:
            size = random.choice([50,60,70,80])
            img = transforms.Compose([transforms.Resize((size, size))])(img)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


def recall(feature_vectors, feature_labels, rank, gallery_vectors=None, gallery_labels=None):
    num_features = len(feature_labels)
    feature_labels = torch.tensor(feature_labels, device=feature_vectors.device)
    gallery_vectors = feature_vectors if gallery_vectors is None else gallery_vectors
    # 求出两两之间相似度
    dist_matrix = torch.cdist(feature_vectors.unsqueeze(0), gallery_vectors.unsqueeze(0)).squeeze(0)

    if gallery_labels is None:
        dist_matrix.fill_diagonal_(float('inf'))  # 对角线置inf
        gallery_labels = feature_labels
    else:
        gallery_labels = torch.tensor(gallery_labels, device=feature_vectors.device)

    idx = dist_matrix.topk(k=rank[-1], dim=-1, largest=False)[1]
    acc_list = []
    for r in rank:
        correct = (gallery_labels[idx[:, 0:r]] == feature_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        acc_list.append((torch.sum(correct) / num_features).item())
    return acc_list


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1, temperature=1.0):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature

    def forward(self, x, target):
        log_probs = F.log_softmax(x / self.temperature, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    @staticmethod
    def get_anchor_positive_triplet_mask(target):
        mask = torch.eq(target.unsqueeze(0), target.unsqueeze(1))
        mask.fill_diagonal_(False)
        return mask

    @staticmethod
    def get_anchor_negative_triplet_mask(target):
        labels_equal = torch.eq(target.unsqueeze(0), target.unsqueeze(1))
        mask = ~ labels_equal
        return mask

    def forward(self, x, target):
        pairwise_dist = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)

        mask_anchor_positive = self.get_anchor_positive_triplet_mask(target)
        anchor_positive_dist = mask_anchor_positive.float() * pairwise_dist
        hardest_positive_dist = anchor_positive_dist.max(1, True)[0]

        mask_anchor_negative = self.get_anchor_negative_triplet_mask(target)
        # make positive and anchor to be exclusive through maximizing the dist
        max_anchor_negative_dist = pairwise_dist.max(1, True)[0]
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative.float())
        hardest_negative_dist = anchor_negative_dist.min(1, True)[0]

        loss = (F.relu(hardest_positive_dist - hardest_negative_dist + self.margin))
        return loss.mean()


class MPerClassSampler(Sampler):
    def __init__(self, labels, batch_size, m=4):
        self.labels = np.array(labels)
        self.labels_unique = np.unique(labels)
        self.batch_size = batch_size
        self.m = m
        assert batch_size % m == 0, 'batch size must be divided by m'

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __iter__(self):
        for _ in range(self.__len__()):
            labels_in_batch = set()
            inds = np.array([], dtype=np.int)

            while inds.shape[0] < self.batch_size:
                sample_label = np.random.choice(self.labels_unique)
                if sample_label in labels_in_batch:
                    continue

                labels_in_batch.add(sample_label)
                sample_label_ids = np.argwhere(np.in1d(self.labels, sample_label)).reshape(-1)
                subsample = np.random.permutation(sample_label_ids)[:self.m]
                inds = np.append(inds, subsample)

            inds = inds[:self.batch_size]
            inds = np.random.permutation(inds)
            yield list(inds)


class SupConLoss_clear(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss_clear, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        single_samples = (mask.sum(1) == 0).float()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)

        # loss
        # filter those single sample
        loss = - mean_log_prob_pos*(1-single_samples)
        loss = loss.sum()/(loss.shape[0]-single_samples.sum())

        return loss


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def hard_aware_point_2_set_mining(dist_mat, labels, weighting='poly', coeff=10):
    """For each anchor, weight the positive and negative samples according to the paper:
    Yu, R., Dou, Z., Bai, S., Zhang, Z., Xu1, Y., & Bai, X. (2018). Hard-Aware Point-to-Set Deep Metric for Person Re-identification, ECCV 2018.
    Args:
      dist_mat: pytorch Variable, pairwise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N] size (N,1)
      weighting: str, weighting scheme, i.e., 'poly' or 'exp' => eq. (8) or (7) in the paper
      coefficient: float, corresponds to the std or alpha parameters used in the paper
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    N = dist_mat.size(0)
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # Exclude selfs for positive samples
    device = labels.device
    v = torch.zeros(N).to(device).type(is_pos.dtype)
    mask = torch.diag(torch.ones_like(v)).to(device).type(is_pos.dtype)
    # is_pos = mask * torch.diag(v) + (1. - mask) * is_pos  # 报错
    is_pos = mask * torch.diag(v) + (~mask) * is_pos

    # `dist_ap` means distance(anchor, positive)
    dist_ap = dist_mat[is_pos].contiguous().view(N, -1)
    # `dist_an` means distance(anchor, negative)
    dist_an = dist_mat[is_neg].contiguous().view(N, -1)
    # Weighting scheme
    if weighting == 'poly':
        w_ap = torch.pow(dist_ap + 1, coeff)
        w_an = torch.pow(dist_an + 1, -2 * coeff)
    else:
        w_ap = torch.exp(dist_ap / coeff)
        w_an = torch.exp(-dist_an / coeff)

    dist_ap = torch.sum(dist_ap * w_ap, dim=1) / torch.sum(w_ap, dim=1)
    dist_an = torch.sum(dist_an * w_an, dim=1) / torch.sum(w_an, dim=1)
    return dist_ap, dist_an


def euclidean_distance(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    #sqrt((x-y)^2)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    return dist.clamp(min=1e-12).sqrt()  # for numerical stability


class HAP2STripletLoss(nn.Module):
#"paper loss"
    def __init__(self, margin=1, coeff=10, weighting='poly'):
        super(HAP2STripletLoss, self).__init__()
        self.coeff = coeff
        self.weighting = weighting
        self.margin = margin
        if margin is None:
            self.ranking_loss = nn.SoftMarginLoss()
        else:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, targets):
    #"feats embedding immagine"
        # All pairwise distances
        D = euclidean_distance(feats,feats)

        # Compute hard aware point to set distances..
        d_ap, d_an = hard_aware_point_2_set_mining(D, targets, self.weighting, self.coeff)
        d_ap.requires_grad_()
        d_an.requires_grad_()

        # Compute loss
        Y = (d_an.data.new().resize_as_(d_an.data).fill_(1))
        Variable(Y,requires_grad=True)
        if self.margin is None:
            loss = self.ranking_loss(d_an-d_ap, Y)
        else:
            loss = self.ranking_loss(d_an, d_ap, Y)
        return loss
