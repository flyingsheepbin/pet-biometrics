import torch
from torch import nn
from torch.nn import functional as F
import timm


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


class GlobalDescriptor(nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p

    def forward(self, x):
        assert x.dim() == 4, 'the input tensor of GlobalDescriptor must be the shape of [B, C, H, W]'
        if self.p == 1:
            return x.mean(dim=[-1, -2])
        elif self.p == float('inf'):
            return torch.flatten(F.adaptive_max_pool2d(x, output_size=(1, 1)), start_dim=1)
        else:
            sum_value = x.pow(self.p).mean(dim=[-1, -2])
            return torch.sign(sum_value) * (torch.abs(sum_value).pow(1.0 / self.p))

    def extra_repr(self):
        return 'p={}'.format(self.p)


class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.dim() == 2, 'the input tensor of L2Norm must be the shape of [B, C]'
        return F.normalize(x, p=2, dim=-1)


class Model(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        backbone = timm.create_model("tf_efficientnetv2_l",
                          pretrained=True,
                          num_classes=0,
                          global_pool="",
                          in_chans=3, features_only=False)
        self.backbone_out = backbone.feature_info[-1]['num_chs']

        gd_config="SM"
        self.features = []
        for name, module in backbone.named_children():
            if isinstance(module, nn.AdaptiveAvgPool1d) or isinstance(module, nn.Linear):
                continue
            self.features.append(module)
        self.features = nn.Sequential(*self.features)
        
        # Main Module
        n = len(gd_config)
        k = feature_dim // n
        assert feature_dim % n == 0, 'the feature dim should be divided by number of global descriptors'

        self.global_descriptors, self.main_modules = [], []
        for i in range(n):
            if gd_config[i] == 'S':
                p = 1
            elif gd_config[i] == 'M':
                p = float('inf')
            else:
                p = 3
            self.global_descriptors.append(GlobalDescriptor(p=p))
            self.main_modules.append(nn.Sequential(nn.Linear(1280, k, bias=False), L2Norm()))
        self.global_descriptors = nn.ModuleList(self.global_descriptors)
        self.main_modules = nn.ModuleList(self.main_modules)

    def forward(self, x):
        shared = self.features(x) #torch.Size([1, 2048, 14, 14]) #eff torch.Size([1, 2560, 7, 7])
        global_descriptors = []
        for i in range(len(self.global_descriptors)):
            global_descriptor = self.global_descriptors[i](shared)

            # global_descriptor = self.main_modules[i](global_descriptor)
            global_descriptors.append(global_descriptor)
        global_descriptors = F.normalize(torch.cat(global_descriptors, dim=-1), dim=-1)
        return global_descriptors
