# base
import os
import sys
sys.path.append('/mnt/hdd1/wearly/kaggle/shopee/pytorch-image-models')
from tqdm import tqdm
import numpy as np
import pandas as pd
import pymysql

# utils
from config import CFG
from model import *
from VGGNet import *
from datasets import *
from metrics import *

# torch
import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import urllib
import time
# https://computistics.tistory.com/13


def test_fn(data_loader, model, device):
    model.eval()
    loss_score = AverageMeter()
    total = 0
    to_num = 0
    aim_acc1 = 0.0
    aim_acc3 = 0.0
    aim_acc5 = 0.0
    aim_recall1 = 0.0
    aim_recall3 = 0.0
    aim_recall5 = 0.0

    tk0 = tqdm(enumerate(data_loader), total=len(data_loader))

    with torch.no_grad():
        for bi, d in tk0:
            batch_size = d[0].size()[0]

            image = d[0]
            targets = d[1]

            image = image.to(device)
            targets = targets.to(device)
            targets = targets.float()

            output = model(image)
            if CFG.loss_type == 'BCELoss':
                output = torch.sigmoid(output)
            _, predicted = output.max(1)
            total += targets.size(0)
            to_num += 1
            
            aim_multi_recall1 = top_n_recall(output, targets, n=1)
            aim_multi_recall3 = top_n_recall(output, targets, n=3)
            aim_multi_recall5 = top_n_recall(output, targets, n=5)
            aim_recall1 += aim_multi_recall1
            aim_recall3 += aim_multi_recall3
            aim_recall5 += aim_multi_recall5
            
            
            aim_multi_acc1, _, aim_multi_acc3 = aim_multi_label_acc(output, targets)
            aim_acc1 += aim_multi_acc1
            aim_acc3 += aim_multi_acc3
            
            tk0.set_postfix(
                Acc1 = aim_acc1 / total, Acc3 = aim_acc3 / total,
                Recall1 = aim_recall1/total, Recall3 = aim_recall3/total, Recall5 = aim_recall5/total,
            )
            #neptune.log_metric('Validation Loss', loss_score.avg)
            #neptune.log_metric('Validation Custom Top1 Accuracy', acc1 / total)
            #neptune.log_metric('Validation Top1 Accuracy', aim_acc1 / total)
            #neptune.log_metric('Validation Top2 Accuracy', aim_acc2 / total)
            #neptune.log_metric('Validation Top3 Accuracy', aim_acc3 / total)
    
    #print(f"Average Recall : Top1-Recall={}, Top2-Recall={}, Top3-Recall={}")
    
    #return loss_score, acc1/total, aim_acc1 / total, aim_acc2 / total, aim_acc3 / total


def running_process(st):
    
    ## Defining Model
    if 'vgg' in CFG.model_name:
        model = Pretrained_VGG_19(pretrained=False).to(CFG.device)
    else:
        model = KstyleNet(pretrained=False).to(CFG.device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(f'{CFG.model_path}/model.pth', map_location="cuda:0"))
    model.eval()

    ## Defining Decoder json
    style_json = { 0 : 'sophisticated', 1 : 'hiphop', 2 : 'sporty', 3 : 'tomboy', 4 : 'oriental',
                   5 : 'kitsch', 6 : 'sexy', 7 : 'street', 8 : 'manish', 9 : 'romantic',
                   10 : 'punk', 11 : 'country', 12 : 'preppy', 13 : 'hippy', 14 : 'avantgarde',
                   15 : 'western', 16 : 'classic', 17 : 'genderless', 18 : 'retro', 19 : 'military',
                   20 : 'resort', 21 : 'modern', 22 : 'feminine' }

    style_ls = []
    
    
    val_dataset = Style_dataset_sp(transform=get_valid_transforms(), mode='test', sty=st)
    
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CFG.Batch_size,
        num_workers=CFG.Num_worker,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    

    test_fn(valid_loader, model, CFG.device)
    
    #json_data = {}
    #print('Accuracy :  best model found for epoch {}'.format(epoch+1))
    #cust_acc1, acc1, acc2, acc3 = custom_acc, original_acc1, original_acc2, original_acc3

    

if __name__ == '__main__':
    style_csvlist = os.listdir("/mnt/hdd1/wearly/ethan/style classification/multi_style")
    style_csvlist.remove('multi_style_val.csv')
    style_csvlist.remove('multi_style_tr.csv')
    style_csvlist.remove('multi_style_tr_sample.csv')
    style_csvlist.remove('multi_style_test.csv')
    style_csvlist.remove('Data_checking.ipynb')
    style_csvlist.remove('.ipynb_checkpoints')
    for s in style_csvlist:
        print(f"--------------Starting {s}--------------")
        running_process(st=s)
    