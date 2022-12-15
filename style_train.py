# !pip install ml_metrics
# !pip install albumentations==0.4.6


# base
import os
import sys
sys.path.append('../pytorch-image-models')
from tqdm import tqdm
import numpy as np

# utils
from config import CFG
from scheduler import *
from optimizer import *
from model import *
from VGGNet import *
from metrics import *
from datasets import Style_dataset, get_train_transforms, get_valid_transforms
from losses import FocalLoss
from losses import AsymmetricLoss
from losses import AsymmetricLossOptimized

# torch
import torch
from torch.utils.data import Dataset, DataLoader

# etc
#from knockknock import slack_sender
import neptune

def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    loss_score = AverageMeter()
    total = 0
    acc1 = 0.0
    aim_acc1 = 0.0
    aim_acc2 = 0.0
    aim_acc3 = 0.0
    aim_recall1 = 0.0
    aim_recall2 = 0.0
    aim_recall3 = 0.0
    
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for bi, d in tk0:
        batch_size = d[0].shape[0]

        images = d[0]
        targets = d[1]

        images = images.to(device)
        targets = targets.to(device)
        
        targets = targets.float()

        optimizer.zero_grad()

        output = model(images)
        if CFG.loss_type == 'BCELoss':
            output = torch.sigmoid(output)
            
        loss = criterion(output, targets)

        loss.backward() # 미분값
        optimizer.step() # 

        _, predicted = output.max(1)
        total += targets.size(0)
        
        aim_multi_recall1 = top_n_recall(output, targets, n=1)
        aim_multi_recall2 = top_n_recall(output, targets, n=2)
        aim_multi_recall3 = top_n_recall(output, targets, n=3)
        aim_recall1 += aim_multi_recall1
        aim_recall2 += aim_multi_recall2
        aim_recall3 += aim_multi_recall3
        
        aim_multi_acc1, aim_multi_acc2, aim_multi_acc3 = aim_multi_label_acc(output, targets)
        aim_acc1 += aim_multi_acc1
        aim_acc2 += aim_multi_acc2
        aim_acc3 += aim_multi_acc3
        
        loss_score.update(loss.detach().item(), batch_size) # 평균계산
        
        
        tk0.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch+1, LR=optimizer.param_groups[0]['lr'],
                        Top1_Recall = aim_recall1 / total, Top2_Recall = aim_recall2 / total, Top3_Recall = aim_recall3 / total,
                        Top1_Accuracy = aim_acc1 / total, Top2_Accuracy = aim_acc2 / total, Top3_Accuracy = aim_acc3 / total, 
                        )
        #neptune.log_metric('Training Loss', loss_score.avg)
        #neptune.log_metric('Training Top1 Accuracy', aim_acc1 / total)
        #neptune.log_metric('Training Top3 Accuracy', aim_acc1 / total)
        #neptune.log_metric('Training Top1 Recall', aim_recall1 / total)
        #neptune.log_metric('Training Top3 Recall', aim_recall3 / total)
        #neptune.log_metric('Learning Rate', optimizer.param_groups[0]['lr'])

    if CFG.scheduler_type == "OneCycleLR_PARAMS":
        scheduler.step()
    
    return loss_score


def eval_fn(data_loader, model, criterion, device, scheduler):
    model.eval()
    loss_score = AverageMeter()
    total = 0
    aim_acc1 = 0.0
    aim_acc2 = 0.0
    aim_acc3 = 0.0
    aim_recall1 = 0.0
    aim_recall2 = 0.0
    aim_recall3 = 0.0

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
                
            loss = criterion(output, targets)

            _, predicted = output.max(1)
            total += targets.size(0)
            
            aim_multi_recall1 = top_n_recall(output, targets, n=1)
            aim_multi_recall2 = top_n_recall(output, targets, n=2)
            aim_multi_recall3 = top_n_recall(output, targets, n=3)
            aim_recall1 += aim_multi_recall1
            aim_recall2 += aim_multi_recall2
            aim_recall3 += aim_multi_recall3
            
            aim_multi_acc1, aim_multi_acc2, aim_multi_acc3 = aim_multi_label_acc(output, targets)
            aim_acc1 += aim_multi_acc1
            aim_acc2 += aim_multi_acc2
            aim_acc3 += aim_multi_acc3
            
            tk0.set_postfix(Valid_Loss=loss_score.avg,
                        Top1_Recall = aim_recall1 / total, Top2_Recall = aim_recall2 / total, Top3_Recall = aim_recall3 / total,
                        Top1_Accuracy = aim_acc1 / total, Top2_Accuracy = aim_acc2 / total, Top3_Accuracy = aim_acc3 / total, 
                        )
            #neptune.log_metric('Validation Loss', loss_score.avg)
            #neptune.log_metric('Validation Top1 Accuracy', aim_acc1 / total)
            #neptune.log_metric('Validation Top3 Accuracy', aim_acc1 / total)
            #neptune.log_metric('Validation Top1 Recall', aim_recall1 / total)
            #neptune.log_metric('Validation Top3 Recall', aim_recall3 / total)
            
            
    if CFG.scheduler_type == "ReduceLROnPlateau":
        scheduler.step(loss_score.avg)
            
    return loss_score, aim_acc1 / total, aim_acc2 / total, aim_acc3 / total, aim_recall1 / total, aim_recall2 / total, aim_recall3 / total


#webhook_url = "https://hooks.slack.com/services/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" #wearly ml학습 channel
#@slack_sender(webhook_url=webhook_url, channel='#alarm', user_mentions=["<Ethan>","<harry>"])
def running_process():

#     neptune.init(
#         project_qualified_name = 'etotmetotm/style-classifier',
#         api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5OWY2NGNiMC01YTE5LTQ0M2MtODgzYi0yMjBhM2ViMDhiYjUifQ=='
#     )
    
    
    ## pass parameters to create experiment
    #neptune.create_experiment(params= None, name= f'{CFG.model_name}', tags= CFG.neptune_tags)
    
    ## Defining Dataset
    tr_dataset = Style_dataset(transform=get_train_transforms(), mode='train')
    val_dataset = Style_dataset(transform=get_valid_transforms(), mode='val')

    ## Defining Dataloader
    train_loader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=CFG.Batch_size,
        num_workers=CFG.Num_worker,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CFG.Batch_size,
        num_workers=CFG.Num_worker,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    ## Defining Model
    if 'vgg' in CFG.model_name:
        if CFG.pretrained == True:
            model = Pretrained_VGG_19(pretrained=CFG.pretrained)
        else:
            model = VGG_19_BN(pretrained=CFG.pretrained)
    else:
        model = KstyleNet(pretrained=CFG.pretrained).to(CFG.device)
    
    model = torch.nn.DataParallel(model)
    model.to(CFG.device)

    ## Defining Criterion
    if CFG.loss_type == 'BCELoss':
        criterion = torch.nn.BCELoss() # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py
    elif CFG.loss_type == 'FocalLoss':
        criterion = FocalLoss()      
    elif CFG.loss_type == 'AsymmetricLoss':
        criterion = AsymmetricLoss(gamma_neg=4,gamma_pos=0,clip=0.05, disable_torch_grad_focal_loss=True)
    elif CFG.loss_type == 'AsymmetricLossOptimized':
        criterion = AsymmetricLossOptimized(gamma_neg=4,gamma_pos=0,clip=0.05, disable_torch_grad_focal_loss=True)
    
    criterion.to(CFG.device)

    ## Defining Optimizer
    if CFG.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr_start)
    elif CFG.optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=CFG.lr_start, weight_decay=CFG.weight_decay)
    elif CFG.optimizer_type == 'ranger':
        optimizer = Ranger(model.parameters(), lr=CFG.lr_start)
    
    
    ## Defining Scheduler
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    if CFG.scheduler_type == 'ReduceLROnPlateau': 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **CFG.ReduceLROnPlateau_PARAMS)
    
    elif CFG.scheduler_type == 'Shopee_Custom_scheduler':
        scheduler = CustomScheduler(optimizer, **CFG.Shopee_PARAMS)
        
    elif CFG.scheduler_type == 'OneCycleLR':
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=steps_per_epoch, epochs=CFG.Epoch, **CFG.OneCycleLR_PARAMS)
    
    best_loss = 10000  # 초깃값
    best_acc = 0
    json_data = {}
    for epoch in range(CFG.Epoch):
        
        if CFG.scheduler_type == None:
            train_loss = train_fn(train_loader, model, criterion, optimizer, CFG.device, scheduler=None, epoch=epoch)
            valid_loss, acc1, acc2, acc3, recall1, recall2, recall3 = eval_fn(valid_loader, model, criterion, CFG.device, scheduler=None)
        else:
            train_loss = train_fn(train_loader, model, criterion, optimizer, CFG.device, scheduler=scheduler, epoch=epoch)
            valid_loss, acc1, acc2, acc3, recall1, recall2, recall3 = eval_fn(valid_loader, model, criterion, CFG.device, scheduler=scheduler)
        
        if recall1 > best_acc:
            if os.path.isdir(f'{CFG.save_path}/{CFG.pretrained}_{CFG.model_name}_{CFG.Epoch}epoch_{CFG.optimizer_type}_{CFG.loss_type}_recent') == False:
                os.mkdir(f'{CFG.save_path}/{CFG.pretrained}_{CFG.model_name}_{CFG.Epoch}epoch_{CFG.optimizer_type}_{CFG.loss_type}_recent')
            
            torch.save(model.state_dict(),
                       f'{CFG.save_path}/{CFG.pretrained}_{CFG.model_name}_{CFG.Epoch}epoch_{CFG.optimizer_type}_{CFG.loss_type}_recent/model.pth')
            
            print('Accuracy :  best model found for epoch {}'.format(epoch+1))
            
            acc_1, acc_2, acc_3, recall_1, recall_2, recall_3 = acc1, acc2, acc3, recall1, recall2, recall3
            
            json_data['Best Epoch'] = epoch+1
            json_data['Top1 Accuracy'] = np.round(acc_1, 3)
            json_data['Top2 Accuracy'] = np.round(acc_2, 3)
            json_data['Top3 Accuracy'] = np.round(acc_3, 3)
            json_data['Top1 Recall'] = np.round(recall_1, 3)
            json_data['Top2 Recall'] = np.round(recall_2, 3)
            json_data['Top3 Recall'] = np.round(recall_3, 3)
            
    json_data['Model name'] = CFG.model_name
    json_data['Image size'] = CFG.img_size
    json_data['Epoch'] = CFG.Epoch
    json_data['Batch size'] = CFG.Batch_size
    json_data['pretrained'] = CFG.pretrained
    json_data['Optimizer'] = CFG.optimizer_type
    json_data['Scheduler'] = CFG.scheduler_type
    json_data['Start learning rate'] = CFG.lr_start
    json_data['Loss'] = CFG.loss_type
    
    return json_data


if __name__ == '__main__':
    json_data = running_process()
    print('-------------Finished training process-------------')
    print(json_data)
    
    


