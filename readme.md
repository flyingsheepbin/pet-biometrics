## 说明
本仓库是 [CVPR2022 Biometrics WorkshopPet Biometric Challenge](https://tianchi.aliyun.com/competition/entrance/531952/introduction) A榜TOP7 B榜TOP2实现代码
## 运行
我们的项目包含9个子项目，分三步运行

## 环境说明
GPU型号: RTX3090  
CUDA: 11.4  
Python: 3.7.11  
torch: 1.11.0+cu113  

## 解决方案描述

### 思路

#### Phase1

我们队伍在一开始首先想到的是训练一个类别为6000的分类模型，在推理阶段使用全连接层的上一层的输出作为图像的特征向量，使用余弦相似度作为相似度度量标准，但在实际测试中，通过图像分类的方式提取的图像特征不能很好的反应两张图片的相似程度，此时的最高的线上成绩是**0.66**

我们转而尝试用图像检索，参考了Kaggle平台上鲸鱼识别的比赛，选用模型结构为backbone+GEM pooling + Arcface，我们的训练集和验证集的数量比例为4：1，其中验证集由dog id样本数目大于3的样本组成，这个阶段最好成绩为**0.78**，使用的backbone为effnet_B7，我们还参考了谷歌地标建筑物检索比赛中夺冠模型DOLG，使用该模型结构且backbone为B7时取得了**0.82**的线上成绩。我们认为地标建筑物检索和本次狗鼻子检索任务还是有区别的，地标建筑物图片中往往没有很明显的特征，而像人脸识别这类任务中，被检索图片中有很明显的区域特征，我们认为dolg这类局部和全局特征相结合的思路可能并不适合本次竞赛。

我们之后参考了[Combination of Multiple Global Descriptors for Image Retrieve](https://arxiv.org/pdf/1903.10663v3.pdf) 论文中的模型结构，该网络通过拼接多种特征描述符来表示图片的特征向量，所使用的损失函数为**Triplet Loss** + **LabelSmooth Loss,** 评价指标是Recall 1，因为有的Dog ID只有两张图片，如果把这类图片划分到不同的集合里，那么就会出现图片完不成匹配的情形，我们索性不划分验证集，用全部图片当作训练集，全部图片当作验证集，此时backbone为b7的情况下，取得了线上**0.824**的成绩，为了继续缩小类内距离，扩大类间距离，我们继续使用arcface训练模型，但线上效果不明显。

我们参考了[Supervised Contrastive Learning](https://arxiv.org/pdf/2004.11362.pdf) 这篇论文，将Triplet loss换成了Supcon Loss，线上取得了**0.839**的成绩，我们又尝试完全使用Supcon Loss，此时线上结果为0.846，之后我们选用swin_base_224作为backbone，取得了84.7的成绩。

我们又尝试不同的图像增强策略，排除了几个图像增强策略后，分数又一次提高，并且经过数据分析后，发现验证集图像分辨率和清晰度小于训练集，我们采用以一定概率先resize60再resize224的策略缩小训练集和验证集的差距。

伪标签策略使用最好的单模预测验证集并进行排序，取前500类作为伪标签。

#### Phase2

Phase2公布测试集时，我们使用之前最好的模型提交后，发现成绩只有86分，分数降低了很多，后查看测试集，发现很多测试集图片质量更差，并且多了运动模糊，我们决定对一阶段模型进行微调，在原来训练集基础上增加了运动模糊、随即裁剪等图像增强方案，经过微调，最好的单模成绩反而是b7模型，单模线上88.04，后我们进行模型融合，取得了89分的最终成绩。

### 技术细节

#### Phase1

- Epoch 30
- lr 5e-5 with cosine weight decay
- AMP、EMA
- 图像增强
    
    ```python
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
    ```
    
- AMP、EMA

#### Phase2

- Epoch 20
- lr 3e-5 with cosine weight decay
- AMP、EMA
- 图像增强
    
    ```python
    data_transforms = {
            "train": A.Compose([
                    A.OneOf([
                    A.Resize(60, 60,p=0.6),
                    A.Resize(60, 84,p=0.2),
                    A.Resize(47, 60,p=0.2),
            ], p=0.45),
                    A.OneOf([
                        A.Compose([
                        A.Resize(240, 240),
                        A.RandomCrop(224, 224),
                    ],p=0.2),
                    A.Resize(224, 224),
            ], p=1),
    
                A.ImageCompression(quality_lower=90, quality_upper=100, p=0.75),
                A.MotionBlur(blur_limit=40, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0
                ),
                ToTensorV2()], p=1.),
            "valid": A.Compose([
                # A.Resize(60, 60,p=1),
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0
                ),
                ToTensorV2()], p=1.)
        }
    ```
