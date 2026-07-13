import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models
from torchvision.models import ResNet34_Weights
import torch.utils.model_zoo as model_zoo
import math
from .meta_module import MetaModule, MetaLinear, MetaConv2d, MetaBatchNorm2d

from .cdac_loss import BCE_softlabels, advbce_unlabeled, sigmoid_rampup
#from evaluation import prediction

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': "https://download.pytorch.org/models/resnet34-b627a593.pth",
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

#'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth'

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)

# Taken from evaluation to avoid circular definition
def prediction(loader, model):
    model.eval()
    P, F = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.cuda().float()
            F.append(model.get_features(x))
            P.append(model.get_predictions(F[-1]))
    model.train()
    return torch.vstack(P), torch.vstack(F)

class ProtoClassifier(nn.Module):
    def __init__(self, size):
        super(ProtoClassifier, self).__init__()
        self.center = None
        self.label = None
        self.size = size

    def init(self, model, t_loader):
        t_pred, t_feat = prediction(t_loader, model)
        label = t_pred.argmax(dim=1)
        center = torch.nan_to_num(
            torch.vstack([t_feat[label == i].mean(dim=0) for i in range(self.size)])
        )
        invalid_idx = center.sum(dim=1) == 0
        if invalid_idx.any() and self.label is not None:
            old_center = torch.vstack(
                [t_feat[self.label == i].mean(dim=0) for i in range(self.size)]
            )
            center[invalid_idx] = old_center[invalid_idx]
        else:
            self.label = label
        self.center = center.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x, T=1.0):
        dist = torch.cdist(x, self.center)
        return F.softmax(-dist * T, dim=1)
    
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nobn=False):
        super(BasicBlock, self).__init__()
        self.eps = 1e-05
        self.momentum = 0.1
        self.affine = True
        self.track_running_stats = True
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = MetaBatchNorm2d(planes, self.eps, self.momentum, self.affine, self.track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = MetaBatchNorm2d(planes, self.eps, self.momentum, self.affine, self.track_running_stats)
        self.downsample = downsample
        self.stride = stride
        self.nobn = nobn

    def forward(self, x, source=True):

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class Meta_ResBase(MetaModule):
    def __init__(self, block, layers, backbone="resnet34", num_classes=1000):
        self.inplanes = 64
        super(Meta_ResBase, self).__init__()
        self.conv1 = MetaConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = MetaBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    padding=1, dilation=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = MetaLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        #if pretrained:
        self.load_state_dict(model_zoo.load_url(model_urls[backbone]))

    def _make_layer(self, block, planes, blocks, stride=1, nobn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MetaConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(planes * block.expansion, eps= 1e-05, momentum=0.1, affine=True, track_running_stats=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nobn=nobn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class Classifier_Meta(MetaModule):
    def __init__(self, in_dim, hidden_dim=512, num_classes=65, temp=0.05):
        super(Classifier_Meta, self).__init__()
        self.fc1 = MetaLinear(in_dim, hidden_dim)
        self.fc2 = MetaLinear(hidden_dim, num_classes, bias=False)
        self.temp = temp

    def forward(self, x, reverse=False):
        x = self.get_features(x, reverse=reverse)
        return self.get_predictions(x)

    def get_features(self, x, reverse=False):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x)
        return F.normalize(x) / self.temp

    def get_predictions(self, x):
        return self.fc2(x)

class ResModel(MetaModule):
    def __init__(
        self,
        backbone="resnet34",
        hidden_dim=512,
        output_dim=65,
        temp=0.05,
        pre_trained=True,
    ):
        super(ResModel, self).__init__()
        self.f = Meta_ResBase(BasicBlock, [3,4,6,3], backbone)
        self.c = Classifier_Meta(512, hidden_dim, output_dim, temp)#"""
        init_weights(self.c)

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.bce = BCE_softlabels()

    def forward(self, x, reverse=False):
        return self.c(self.f(x), reverse)

    def get_params(self, lr):
        params = []
        for k, v in dict(self.f.named_params()).items():                                    
            if v.requires_grad:
                if "classifier" not in k:
                    params += [{"params": [v], "base_lr": lr * 0.1 , "lr": lr * 0.1}]      
                else:
                    params += [{"params": [v], "base_lr": lr, "lr": lr}]
        params += [{"params": self.c.params(), "base_lr": lr, "lr": lr}]
        return params

    def get_features(self, x, reverse=False):
        return self.c.get_features(self.f(x), reverse=reverse)

    def get_predictions(self, x):
        return self.c.get_predictions(x)

    def base_loss(self, x, y):
        output = self.forward(x)
        return self.criterion(output, y).mean()

    def feature_base_loss(self, f, y):
        return self.criterion(self.get_predictions(f), y).mean()

    def sla_loss(self, f, y1, y2, alpha):
        out = self.get_predictions(f)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = self.criterion(out, y1)
        soft_loss = -(y2 * log_softmax_out).sum(axis=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()

    def nl_loss(self, f, y, alpha, T):
        out = self.get_predictions(f)
        y2 = F.softmax(out.detach() * T, dim=1)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = self.criterion(out, y)
        soft_loss = -(y2 * log_softmax_out).sum(dim=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()
    
    def mixup_loss(self, step, ux, x, y):
        u_prob = F.softmax(self.forward(ux), dim=1)
        u_pred = u_prob.max(1)
        u_mask = u_pred[0] >= 0.50              
        
        im_u_1 = ux[u_mask]
        psl_u = u_pred[1][u_mask]

        #mix_up
        alpha = 1
        lam = np.random.beta(alpha, alpha)
        psl_loss = 0
        # stream 1
        
        if im_u_1.size(0) > 0:
            size_1 = im_u_1.size(0)
            #print('stream 1: {}'.format(size_1))

            t_idx = torch.randperm(x.size(0))[0:size_1]
            mixed_x = lam * x[t_idx] + (1-lam) * im_u_1
            y_a, y_b = y[t_idx], psl_u

            out_mix = self.forward(mixed_x)
            psl_loss = lam * self.criterion(out_mix, y_a).mean() + (1-lam) * self.criterion(out_mix, y_b).mean()
        return psl_loss

    def mme_loss(self, _, x, lamda=0.1):
        out = self.forward(x, reverse=True)
        out = F.softmax(out, dim=1)
        return lamda * torch.mean(torch.sum(out * (torch.log(out + 1e-10)), dim=1))

    def cdac_loss(self, step, x, x1, x2):
        w_cons = 30 * sigmoid_rampup(step, 2000)
        f = self.f(x)
        f1 = self.f(x1)
        f2 = self.f(x2)

        out = self.c(f, reverse=True)
        out1 = self.c(f1, reverse=True)

        prob, prob1 = F.softmax(out, dim=1), F.softmax(out1, dim=1)
        aac_loss = advbce_unlabeled(
            target=None, f=f, prob=prob, prob1=prob1, bce=self.bce
        )

        out = self.c(f)
        out1 = self.c(f1)
        out2 = self.c(f2)

        prob, prob1, prob2 = (
            F.softmax(out, dim=1),
            F.softmax(out1, dim=1),
            F.softmax(out2, dim=1),
        )
        mp, pl = torch.max(prob.detach(), dim=1)
        mask = mp.ge(0.95).float()

        pl_loss = (F.cross_entropy(out2, pl, reduction="none") * mask).mean()
        con_loss = F.mse_loss(prob1, prob2)

        return aac_loss + pl_loss + w_cons * con_loss


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambd
        return output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)



import torch
import torch.nn as nn

# Import from your friend's repo (their MetaModule implementation)
# Make sure meta_module.py is in your PYTHONPATH (or same folder).
from .meta_module import MetaModule, MetaConv2d, MetaBatchNorm2d, MetaLinear


# -------------------------
# Meta-ResNet18 backbone
# -------------------------
def _conv3x3(in_planes, out_planes, stride=1):
    return MetaConv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=1, bias=False
    )

def _conv1x1(in_planes, out_planes, stride=1):
    return MetaConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MetaBasicBlock(MetaModule):
    """
    Minimal ResNet BasicBlock using Meta layers.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = MetaBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes, 1)
        self.bn2 = MetaBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class MetaResNet18Backbone(MetaModule):
    """
    ResNet18 trunk implemented with Meta layers.
    Outputs pooled features of shape (B, 512).
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.inplanes = 64

        self.conv1 = MetaConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = MetaBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet18 layers: [2, 2, 2, 2]
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * MetaBasicBlock.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * MetaBasicBlock.expansion, stride),
                MetaBatchNorm2d(planes * MetaBasicBlock.expansion),
            )

        layers = []
        layers.append(MetaBasicBlock(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * MetaBasicBlock.expansion

        for _ in range(1, blocks):
            layers.append(MetaBasicBlock(self.inplanes, planes, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (B, 512)
        return x


# -------------------------
# Meta PSNR/L1 head
# -------------------------
class MetaPNSRL1Head(MetaModule):
    """
    Equivalent to your PNSRL1Warpper head:
      feat(512) + [psnr, l1] -> MLP(64) -> logits(2)
    """
    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.fc1 = MetaLinear(feat_dim + 2, 64)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = MetaLinear(64, 2)

    def forward(self, feat, psnr, l1):
        if psnr.dim() == 1:
            psnr = psnr.unsqueeze(1)
        if l1.dim() == 1:
            l1 = l1.unsqueeze(1)

        # Keep dtype consistent with features
        psnr = psnr.to(dtype=feat.dtype)
        l1 = l1.to(dtype=feat.dtype)

        x = torch.cat([feat, psnr, l1], dim=1)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


class MetaLinearHead(MetaModule):
    """
    Simple head for non-psnr mode: feat(512) -> logits(2)
    """
    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.fc = MetaLinear(feat_dim, 2)

    def forward(self, feat):
        return self.fc(feat)


# -------------------------
# Final Meta Verifier Model
# -------------------------
class MetaVerifierModel(MetaModule):
    """
    Drop-in meta-learning compatible verifier.

    Key properties (to match your friend's algorithm):
      - Inherits MetaModule
      - Has named_params(), set_param(), copy()
      - Stores _meta_init_kwargs so trainer can re-instantiate new_model

    Forward compatibility:
      - Accepts both (pnsr, l1) like your current code
      - Also accepts keyword psnr=... (common spelling) as alias
    """
    def __init__(self, in_channels: int, include_psnr_l1: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.include_psnr_l1 = bool(include_psnr_l1)

        self.backbone = MetaResNet18Backbone(in_channels=self.in_channels)

        if self.include_psnr_l1:
            self.head = MetaPNSRL1Head(feat_dim=512)
        else:
            self.head = MetaLinearHead(feat_dim=512)

        # Used by the meta-trainer to rebuild the model per task
        self._meta_init_kwargs = dict(
            in_channels=self.in_channels,
            include_psnr_l1=self.include_psnr_l1,
        )

    def forward(self, x, pnsr=None, l1=None, psnr=None):
        """
        x: (B, C, H, W)
        pnsr/l1: (B,) or (B,1)
        psnr: alias for pnsr (spelling convenience)
        """
        feat = self.backbone(x)  # (B, 512)

        if self.include_psnr_l1:
            # Allow alias name "psnr"
            if psnr is not None and pnsr is None:
                pnsr = psnr
            assert pnsr is not None and l1 is not None, "Need (pnsr/psnr) and l1 when include_psnr_l1=True"
            return self.head(feat, pnsr, l1)

        return self.head(feat)


# Optional helper (use this instead of your old make_model when doing meta-learning)
def make_meta_verifier(in_channels: int, include_psnr_l1: bool = True) -> MetaVerifierModel:
    """
    Factory for meta verifier.
    """
    return MetaVerifierModel(in_channels=in_channels, include_psnr_l1=include_psnr_l1)