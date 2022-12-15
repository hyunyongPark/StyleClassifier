import os
import sys
#sys.path.append('/mnt/hdd1/wearly/kaggle/shopee/pytorch-image-models')
sys.path.append('../pytorch-image-models')

import timm
import torch
import torch.nn as nn
from config import CFG
import math

# Timm Library Model
# ====================================================
class KstyleNet(nn.Module):
    def __init__(self,
                 pretrained,
                 n_class=CFG.n_classes,
                 model_name=CFG.model_name):
        super(KstyleNet, self).__init__()

        self.n_classes = n_class
        self.model_name = model_name

        if 'efficientnet' in self.model_name:
            self.model = timm.create_model(self.model_name, pretrained=pretrained)
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()  # backbone 모델의 classifier 초기화
            self.model.classifier = nn.Linear(self.n_features, self.n_classes)
        if 'eca_nfnet' in self.model_name:
            self.model = timm.create_model(self.model_name, pretrained=pretrained)
            self.n_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Identity()  # backbone 모델의 classifier 초기화
            self.model.head.fc = nn.Linear(self.n_features, self.n_classes)
    
    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
    
    def forward(self, image):
        img_embedding = self.feature_extractor(image)
        return img_embedding  # shape(batch_size, class개수)

    def feature_extractor(self, x):
        x = self.model(x)
        return x

#     def freeze(self):
#         # To freeze the residual layers
#         for param in self.model.parameters():
#             param.requires_grad = False

#         if 'efficientnet' in self.model_name:
#             for param in self.model.classifier.parameters():
#                 param.requires_grad = True
#         if 'eca_nfnet' in self.model_name:
#             for param in self.model.head.fc.parameters():
#                 param.requires_grad = True

#     def unfreeze(self):
#         # Unfreeze all layers
#         for param in self.model.parameters():
#             param.requires_grad = True

