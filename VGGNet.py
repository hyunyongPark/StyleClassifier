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
#     # filter??? ???(????????????)??? ????????? ?????????????????? ????????? - ????????? ????????? torch.seed ??? ??????????????? ??????.
#     # ????????? ????????? ??? ?????? ??????????????? ??? doc?????? ????????? 
#     def _initialize_weights(self):
#         for m in self.modules(): # ?????? ??????????????? ????????? layer?????? iterable(ex : list, dict ...)??? ????????? ??????
#             if isinstance(m, nn.Conv2d): # ????????? ????????? m?????? layer??? nn.Conv2d???????????? ????????? ??????????????????? (True or False) 
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels # ?????? filer??? ?????? ?????? ??????
#                 # https://reniew.github.io/13/
#                 m.weight.data.normal_(0, math.sqrt(2. / n)) # He Initialization (relu??? ????????? ???????????? ???)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d): # batchNorm layer??? ??????
#                 m.weight.data.fill_(1) # weight????????? ?????? 1??? ????????? 
#                 m.bias.data.zero_() # bias??? 0?????? ?????????
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01) # ????????? ?????? (we sampled the weights from a normal distribution with the zero mean and 10???2 variance)
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
#                 m.weight.data.normal_(0, 0.01) # ????????? ?????? (we sampled the weights from a normal distribution with the zero mean and 10???2 variance)
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
            self.model.head.fc = nn.Identity()  # backbone ????????? classifier ?????????
            self.model.head.fc = nn.Linear(self.n_features, n_classes)

    def forward(self, image):
        img_embedding = self.feature_extractor(image)
        return img_embedding  # shape(batch_size, class??????)
    
    def feature_extractor(self, x):
        x = self.model(x)
        return x
