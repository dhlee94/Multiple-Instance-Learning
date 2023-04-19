import os
import yaml
from easydict import EasyDict
import random
import cv2
import numpy as np
import torch
import time
import logging
from pathlib import Path
from torch.nn import functional as F
import torch
import shutil
from collections import OrderedDict
from sklearn import metrics

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def classification_accruracy(predict, target, thresh=0.5, sigmoid=False):
    predict = torch.sigmoid(predict).detach().cpu().numpy() if sigmoid else predict
    target = target.detach().cpu().numpy()
    predict[predict>=thresh] = 1
    predict[predict<thresh] = 0
    accuracy = metrics.accuracy_score(target, predict)
    #precision = metrics.precision_score(target, predict)
    #recall = metrics.recall_score(target, predict)
    #f1 = metrics.f1_score(target, predict)    
    return accuracy, np.sum(predict, axis=0), np.sum(target, axis=0)

def classification_accruracy_multi(predict, target, softmax=True):
    predict = torch.softmax(predict, dim=-1).detach().cpu().numpy() if softmax else predict
    target = target.detach().cpu().numpy()
    predict = np.array(predict.argmax(axis=-1))
    target = np.array(target.argmax(axis=-1))
    accuracy = metrics.accuracy_score(target, predict)
    #precision = metrics.precision_score(target, predict)
    #recall = metrics.recall_score(target, predict)
    #f1 = metrics.f1_score(target, predict)   
    return accuracy

def classification_f1_multi(predict, target, softmax=True):
    predict = torch.softmax(predict, dim=-1).detach().cpu().numpy() if softmax else predict
    target = target.detach().cpu().numpy()
    predict = np.array(predict.argmax(axis=-1))
    target = np.array(target.argmax(axis=-1))
    #accuracy = metrics.accuracy_score(target, predict)
    precision = metrics.precision_score(target, predict)
    recall = metrics.recall_score(target, predict)
    f1 = metrics.f1_score(target, predict)   
    return f1, precision, recall

def specificity_and_sensitivity(predict, target, thresh, softmax=False, sigmoid=False):
    if sigmoid:
        predict = torch.sigmoid(predict).detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        predict[predict>=thresh] = 1
        predict[predict<thresh] = 0
    elif softmax:
        predict = torch.softmax(predict, dim=-1).detach().cpu().numpy()
        predict = np.array(predict.argmax(axis=-1))
        target = np.array(target.detach().cpu().numpy().argmax(axis=-1))
    cfx = metrics.confusion_matrix(target, predict)
    sensitivity = cfx[0,0]/(cfx[0,0]+cfx[0,1])
    specificity = cfx[1,1]/(cfx[1,0]+cfx[1,1])
    return specificity, sensitivity, cfx[0,0], cfx[1,1], cfx[1,0], cfx[0,1]

def specificity_and_sensitivity_per_class(predict, target, num_class):
    predict = torch.softmax(predict, dim=-1).detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    predict = predict.argmax(axis=-1)
    target = target.argmax(axis=-1)
    predict = np.where(predict=num_class)
    target = np.where(target=num_class)
    cfx = metrics.confusion_matrix(target, predict)
    sensitivity = cfx[0,0]/(cfx[0,0]+cfx[0,1])
    specificity = cfx[1,1]/(cfx[1,0]+cfx[1,1])
    return specificity, sensitivity, cfx[0,0], cfx[1,1], cfx[1,0], cfx[0,1]

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pth.tar'))
