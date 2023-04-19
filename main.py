from utils.utils import AddParserManager, seed_everything, select_idx_for_train
import argparse
#import wandb
from core.function import train
from core.criterion import MultiClassDiceCELoss, FocalLoss, EntropyWithDistanceLoss
import numpy as np
from core.optimizer import CosineAnnealingWarmUpRestarts
from data.dataset import ImageDataset
from timm.models.layers import to_2tuple
from torch.utils.data import DataLoader
from models.DTFD_MIL import DTFD_MIL
import torch
import torch.nn as nn
import os
import albumentations
import albumentations.pytorch
from timm.scheduler.cosine_lr import CosineLRScheduler
import pandas as pd
import cv2
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=304, help='random seed')
parser.add_argument('--csv_path', type=str, required=True, metavar="FILE", help='path to CSV file')
parser.add_argument('--batch_size', type=int, default=1, help='Train Batch Size(MIL Model batch size 1, if you want other batchsize you have to make collapse_fn function)')
parser.add_argument('--workers', type=int, default=1, help='Num workers')
parser.add_argument('--shuffle', type=bool, default=True, help='Train dataloader shuffle')
parser.add_argument('--gpu', type=str, default="0",help='Number of GPU')
parser.add_argument('--in_channel', type=int, default=304, help='MIL model input channels')
parser.add_argument('--num_class', type=int, default=304, help='Number of Class')
parser.add_argument('--num_feed', type=int, default=1, help='Number of feedforward layer in dim reduction')
parser.add_argument('--num_attn', type=int, default=2, help='Number of attention, feedforward layer in attention')
parser.add_argument('--instance_per_group', type=int, default=128, help='Number of instance per group in tier1')
parser.add_argument('--optim', type=str, default='SGD', help='Type of Optimizer')
parser.add_argument('--scheduler', type=str, default='LambdaLR', help='Type of Scheduler')
parser.add_argument('--optim_eps', type=float, default=1e-8, help='AdamW optimizer parameters eps')
parser.add_argument('--optim_betas', default=[0.9, 0.999], help='AdamW optimizer parameters betas')
parser.add_argument('--lr', type=float, default=1e-4, help='Optimizer Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.95, help='Optimizer Weight decay')
parser.add_argument('--momentum', type=float, default=0.95, help='Optimizer momentum')
parser.add_argument('--scheduler_t', type=int, default=80, help='CosineAnnealing Warm Up restarts Scheduler parameters t0')
parser.add_argument('--scheduler_tmult', type=int, default=1, help='CosineAnnealing Warm Up restarts Scheduler parameters t mult')
parser.add_argument('--scheduler_eta', type=float, default=1.25e-3, help='CosineAnnealing Warm Up restarts Scheduler parameters eta max')
parser.add_argument('--scheduler_tup', type=int, default=8, help='CosineAnnealing Warm Up restarts Scheduler parameters t up')
parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='CosineAnnealing Warm Up restarts Scheduler parameters gamma')
parser.add_argument('--scheduler_lambda_weight', type=float, default=0.975, help='LambdaLR scheduler lr lamda weight parameter')
parser.add_argument('--scheduler_milestones', default=[100], help='MultiStemLR scheduler milestones')
parser.add_argument('--scheduler_multi_gamma', type=float, default=0.2, help='MultiStemLR scheduler gamma')
parser.add_argument('--model_path', type=str, default=None, help='Model path')
parser.add_argument('--model_save_path', type=str, default='./weight', help='Model save path')
parser.add_argument('--log_path', type=str, default='./log', help='Save log path')
parser.add_argument('--write_iter_num', type=int, default=10, help='Write Iter Number')
parser.add_argument('--epoch', type=int, default=100, help='Train Epoch')

def main():    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')
    print("Reading Data")
    train_csv_path = pd.read_csv(os.path.join(args.csv_path, 'train_data.csv'))
    valid_csv_path = pd.read_csv(os.path.join(args.csv_path, 'valid_data.csv'))    
    train_dataset = ImageDataset(Data_path=train_csv_path)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=config.WORKERS, pin_memory=True, shuffle=args.shuffle)
    valid_dataset = ImageDataset(Data_path=valid_csv_path)
    validloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=config.WORKERS, pin_memory=True, shuffle=False)   
    
    #Model 
    model = DTFD_MIL(in_channels=args.in_channel,
                         output_class=args.num_class, num_feed=args.num_feed, sub_attn=args.num_attn, instance_per_group=args.instance_per_group)    

    if args.optim=='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), eps=args.optim_eps, betas=args.optim_betas, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)    
    elif args.optim=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    if args.scheduler=='CAWUR':
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.scheduler_t, T_mult=args.scheduler_tmult, 
                                                     eta_max=args.scheduler_eta, T_up=args.scheduler_tup, gamma=args.scheduler_gamma)
    elif args.scheduler=='LambdaLR':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch:args.scheduler_lambda_weight**epoch)    
    elif args.scheduler=='MultiStemLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_multi_gamma)
        
    nSamples = [200, 200]
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    criterion = EntropyWithDistanceLoss(weights=0.7, entropy_weight=torch.Tensor(normedWeights), reduction='mean', sigmoid=False)

    if args.model_path:
        loc = {'cuda:{}':'cpu'.format(args.gpu)}
        checkpoint = torch.load(args.model_path, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    model = model.to(device)
    criterion = criterion.to(device)
    for epoch in range(start_epoch, args.epoch):
        is_best = False
        file = open(os.path.join(args.log_path, f'{epoch}_log.txt'), 'a')
        train(model=model, write_iter_num=args.write_iter_num, train_dataset=trainloader, optimizer=optimizer, 
                    device=device, criterion=criterion, epoch=epoch, file=file)
        accuracy = valid(model=model, write_iter_num=args.write_iter_num, valid_dataset=validloader, criterion=criterion, 
                               device=device, epoch=epoch, file=file)
        scheduler.step()
        is_best = accuracy > best_loss
        best_loss = max(best_loss, accuracy)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_loss,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best=is_best, path=args.model_save_path)
        file.close()

if __name__ == '__main__':
    main()
        
        