import os
import sys
sys.path.append('/mnt/hdd1/wearly/kaggle/shopee/pytorch-image-models')

import timm
import torch
import torch.nn as nn
from config import CFG
import math

# VGGNet11 Custom
# ====================================================            
# class VGG_11(nn.Module): # Paper model A
#     def __init__(self, 
#                  num_classes=CFG.n_classes, 
#                  init_weights=True):
#         super(VGG_11, self).__init__()
#         self.convnet = nn.Sequential(
#             # Input Channel (RGB: 3)
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2), 
            
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2), 
            
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2), 

#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2), 

#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2), 
#         )

#         self.fclayer = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, num_classes),
#         )
        
#         if init_weights:
#             self._initialize_weights()
    
    
#     # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
#     # filter의 값(가중치값)은 초기에 랜덤난수로서 뿌려짐 - 그렇기 때문에 torch.seed 를 고정해주는 것임.
#     # 필터를 정의할 때 초기 가중치값은 위 doc에서 설명함 
#     def _initialize_weights(self):
#         for m in self.modules(): # 모델 클래스에서 정의된 layer들을 iterable(ex : list, dict ...)한 객체로 반환
#             if isinstance(m, nn.Conv2d): # 위에서 정의된 m번째 layer가 nn.Conv2d함수에서 비롯된 인스턴스인가? (True or False) 
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels # 해당 filer의 모든 뉴런 개수
#                 # https://reniew.github.io/13/
#                 m.weight.data.normal_(0, math.sqrt(2. / n)) # He Initialization (relu에 적합한 편이라고 함)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d): # batchNorm layer의 경우
#                 m.weight.data.fill_(1) # weight값들은 전부 1로 초기화 
#                 m.bias.data.zero_() # bias는 0으로 초기화
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01) # 논문에 명시 (we sampled the weights from a normal distribution with the zero mean and 10−2 variance)
#                 m.bias.data.zero_()
    
#     def forward(self, x):
#         # batch_size = x.shape[0]
#         x = self.convnet(x)
#         x = torch.flatten(x, 1) # x = x.view(-1, .view(batch_size, -1))
#         x = self.fclayer(x)
#         return x


# # VGGNet19 Custom
# # ====================================================
# class VGG_19(nn.Module): # Paper model E
#     def __init__(self, 
#                  num_classes=CFG.n_classes, 
#                  init_weights=True):
#         super(VGG_19, self).__init__()
#         self.convnet = nn.Sequential(
#             # Input Channel (RGB: 3)
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2), 
            
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2), 
            
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2), 

#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2), 

#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2), 
#         )

#         self.fclayer = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, num_classes),
#         )
        
#         if init_weights:
#             self._initialize_weights()
    
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):  
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n)) # He Initialization
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1) 
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01) # 논문에 명시 (we sampled the weights from a normal distribution with the zero mean and 10−2 variance)
#                 m.bias.data.zero_()
    
#     def forward(self, x):
#         x = self.convnet(x)
#         x = torch.flatten(x, 1)
#         x = self.fclayer(x)
#         return x

    
# Pretrained VGGNet19 (Timm) 
# ====================================================
class Pretrained_VGG_19(nn.Module): # Paper model E
    def __init__(self,
                 pretrained,
                 n_classes=CFG.n_classes,
                 model_name=CFG.model_name):
        super(Pretrained_VGG_19, self).__init__()
        
        if 'vgg' in model_name:
            self.model = timm.create_model(model_name, pretrained=pretrained)
            self.n_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Identity()  # backbone 모델의 classifier 초기화
            self.model.head.fc = nn.Linear(self.n_features, n_classes)

    def forward(self, image):
        img_embedding = self.feature_extractor(image)
        return img_embedding  # shape(batch_size, class개수)
    
    def feature_extractor(self, x):
        x = self.model(x)
        return x
