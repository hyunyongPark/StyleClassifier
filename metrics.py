import numpy as np
import pandas as pd
import torch
from torchmetrics import Accuracy
from config import CFG

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): #mean , 128
        self.val = val
        self.sum += val * n # mini-batch loss sum 
        self.count += n # 
        self.avg = self.sum / self.count


# def multi_label_acc_V1(pred, label):
#     pred = pred.cpu().detach().numpy()
#     label = label.cpu().detach().numpy()

#     pred = [np.argsort(x)[::-1] for x in pred]
#     acc_1 = 0
#     for i in range(len(pred)):
#         if sum(label[i]) == 2:
#             if label[i][pred[i][0]] == 1 and label[i][pred[i][1]] == 1:
#                 acc_1 += 1
#         else:
#             if label[i][pred[i][0]] == 1:
#                 acc_1 += 1

#     multi_acc1 = acc_1

#     return multi_acc1



def aim_multi_label_acc(pred, label):
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    
    pred = [np.argsort(x)[::-1] for x in pred] # [3, 5, 0, 1 ,2, 4]
    
    acc_1 = 0
    for i in range(len(pred)):
        if label[i][pred[i][0]] == 1:
            acc_1 +=1

    acc_2 = 0
    for i in range(len(pred)):
        if label[i][pred[i][0]] or label[i][pred[i][1]]== 1:
            acc_2 +=1

    acc_3 = 0
    for i in range(len(pred)):
        if label[i][pred[i][0]] or label[i][pred[i][1]] or label[i][pred[i][2]] == 1:
            acc_3 += 1

    multi_acc1 = acc_1
    multi_acc2 = acc_2
    multi_acc3 = acc_3

    return multi_acc1, multi_acc2, multi_acc3


def top_n_recall(pred, label, n=1):
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    
    best_n = np.argsort(pred, axis=1)[:,-n:]
    ts = np.argmax(label, axis=1)
    successes = 0
    s_img_id = []
    w_img_id = []
    for i in range(ts.shape[0]):
        if ts[i] in best_n[i,:]:
            successes += 1
            s_img_id.append(i)
        else:
            w_img_id.append(i)

    return float(successes)#/ts.shape[0] #, s_img_id, w_img_id, best_n


# def multi_label_acc_V2(pred, label):
#     # https://torchmetrics.readthedocs.io/en/latest/
#     # https://stackoverflow.com/questions/61524717/pytorch-how-to-find-accuracy-for-multi-label-classification
#     pred = pred >= 0.5
#     pred = pred * 1
#     pred = pred.long()
#     label = label.long()
    
#     accuracy = Accuracy()
#     accuracy.to(CFG.device)
    
#     acc = []
#     for i in range(len(pred)): # 4
#         batch_acc = accuracy(pred[i], label[i]) # (,23)
#         batch_acc = batch_acc * pred[i].numel()
#         acc.append(batch_acc.cpu().detach().item())
    
# #     acc = torch.sum(pred == label)
# #     acc = acc.cpu().detach().item()
    
# #     return acc, pred.numel()
#     return sum(acc), pred.numel()