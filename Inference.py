# base
import os
import sys
sys.path.append('/mnt/hdd1/wearly/kaggle/shopee/pytorch-image-models')
import tqdm
import numpy as np
import pandas as pd
import pymysql

# utils
from config import CFG
from model import *
from VGGNet import *
from datasets import get_valid_transforms

# torch
import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import urllib
import time
# https://computistics.tistory.com/13


def running_process():
    
    db = pymysql.connect(host='119.70.197.5',
                           port=3306,
                           user='root',
                           password='wearly123',
                           db='vogue_wearly', charset='utf8mb4')
    curs = db.cursor()
    sql = """select image_name,brand,date,text,color from vogue_wearly.vogue2"""
    curs.execute(sql)
    select = list(curs.fetchall())
    db.commit()
    sql_df = [list(select[i]) for i in range(len(select))]
    sql_df = pd.DataFrame(data=sql_df, columns=['image_name', 'brand', 'date', 'text', 'color'])
    curs.close()
    
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

    img_ls = sql_df['image_name'].values.tolist()
    ## Defining Entire roop
    style_ls = []
    for ipth in tqdm.tqdm(range(len(img_ls))):
        try:
            se_img = img_ls[ipth]
            ## Defining Dataset
            image = cv2.imread(f'/mnt/hdd1/wearly/vogue/vogue_images/{se_img}')  # 해당 index 에 대한 이미지 로드
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color 차원 부여

            ## Running augmnetation
            transforms = get_valid_transforms()
            augmented = transforms(image=image)
            image = augmented['image']

            ## Change Dimensions
            image = image.unsqueeze(0)  # 3D -> 4D

            ## Running inference
            with torch.no_grad():
                image = image.to(CFG.device)
                output = model(image)
                output = torch.sigmoid(output)
                output = output.cpu().detach().numpy()

            ## Generating CSV file
            main_style = np.argsort(np.max(output, axis=0))[-1]
            sub_style = np.argsort(np.max(output, axis=0))[-2]

            if output[0][main_style] < 0.5:
                main_style = None
            if output[0][sub_style] < 0.5:
                sub_style = None

            if main_style == None and sub_style == None:
                style_ls.append([se_img, sql_df['brand'][ipth], sql_df['date'][ipth], sql_df['text'][ipth], sql_df['color'][ipth], 
                                 main_style, sub_style])
            elif main_style != None and sub_style == None:
                style_ls.append([se_img, sql_df['brand'][ipth], sql_df['date'][ipth], sql_df['text'][ipth], sql_df['color'][ipth], 
                                 style_json[main_style], sub_style])
            else:
                style_ls.append([se_img, sql_df['brand'][ipth], sql_df['date'][ipth], sql_df['text'][ipth], sql_df['color'][ipth], 
                                 style_json[main_style], style_json[sub_style]])

        except:
            pass
    csv_data = pd.DataFrame(style_ls, columns=['image_name', 'brand', 'date', 'text', 'color', 'main','sub'])
    csv_data.to_csv(f'/mnt/hdd1/wearly/vogue/tables_sty.csv')
        

    return csv_data


if __name__ == '__main__':
    csv_data = running_process()
    print(csv_data)