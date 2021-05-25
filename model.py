import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F
import timm

class enetv2(nn.Module):
    def __init__(self, enet_type, out_dim, pretrained=True):
        super(enetv2, self).__init__()
        self.enet = timm.create_model(enet_type, pretrained)
        n_ch = 4
        if ('efficientnet' in enet_type) or ('mixnet' in enet_type):
            self.enet.conv_stem.weight = nn.Parameter(self.enet.conv_stem.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.enet.classifier.in_features, out_dim)
            self.enet.classifier = nn.Identity()
        elif ('resnet' in enet_type or 'resnest' in enet_type) and 'vit' not in enet_type:
            self.enet.conv1[0].weight = nn.Parameter(self.enet.conv1[0].weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.enet.fc.in_features, out_dim)
            self.enet.fc = nn.Identity()
        elif 'rexnet' in enet_type or 'regnety' in enet_type or 'nf_regnet' in enet_type:
            self.enet.stem.conv.weight = nn.Parameter(self.enet.stem.conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.enet.head.fc.in_features, out_dim)
            self.enet.head.fc = nn.Identity()
        elif 'resnext' in enet_type:
            self.enet.conv1.weight = nn.Parameter(self.enet.conv1.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.enet.fc.in_features, out_dim)
            self.enet.fc = nn.Identity()
        elif 'hrnet_w32' in enet_type:
            self.enet.conv1.weight = nn.Parameter(self.enet.conv1.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.enet.classifier.in_features, out_dim)
            self.enet.classifier = nn.Identity()
        elif 'densenet' in enet_type:
            self.enet.features.conv0.weight = nn.Parameter(self.enet.features.conv0.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.enet.classifier.in_features, out_dim)
            self.enet.classifier = nn.Identity()
        elif 'ese_vovnet39b' in enet_type or 'xception41' in enet_type:
            self.enet.stem[0].conv.weight = nn.Parameter(self.enet.stem[0].conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.enet.head.fc.in_features, out_dim)
            self.enet.head.fc = nn.Identity()
        elif 'dpn' in enet_type:
            self.enet.features.conv1_1.conv.weight = nn.Parameter(self.enet.features.conv1_1.conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.enet.classifier.in_channels, out_dim)
            self.enet.classifier = nn.Identity()
        elif 'inception' in enet_type:
            self.enet.features[0].conv.weight = nn.Parameter(self.enet.features[0].conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.enet.last_linear.in_features, out_dim)
            self.enet.last_linear = nn.Identity()
        elif 'vit_base_resnet50' in enet_type:
            self.enet.patch_embed.backbone.stem.conv.weight = nn.Parameter(self.enet.patch_embed.backbone.stem.conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.enet.head.in_features, out_dim)
            self.enet.head = nn.Identity()
        else:
            raise

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.myfc(dropout(x))
            else:
                h += self.myfc(dropout(x))
        return h / len(self.dropouts)
