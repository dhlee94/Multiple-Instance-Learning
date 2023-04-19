import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import os
import numpy as np
from tqdm import tqdm
from utils.utils import AverageMeter, classification_accruracy_multi, classification_f1_multi

def train(model=None, write_iter_num=5, train_dataset=None, optimizer=None, device=None, criterion=torch.nn.BCELoss(), epoch=None, file=None):
    best_loss = 0
    #scaler = torch.cuda.amp.GradScaler()
    assert train_dataset is not None, print("train_dataset is none")
    model.train()        
    ave_accuracy = AverageMeter()
    #scaler = torch.cuda.amp.GradScaler()
    for idx, train_batch in enumerate(tqdm(train_dataset)):
        Image, label, filename = train_batch
        Input = Variable(Image.squeeze(0).type(Tensor))
        Label = Variable(label.type(Tensor))
        prediction, sub_prediction, _, num_split = mil_model(Input, num_split=3)
        sub_label = label.repeat(num_split, 1).type(Tensor)
        optimizer_mil.zero_grad()
        loss_final = criterion(prediction, Label)
        loss_bag = criterion(sub_prediction, sub_label)
        whole_loss = loss_final + loss_bag
        whole_loss.backward()
        optimizer_mil.step()            
        #torch.nn.utils.clip_grad_norm_(mil_model.parameters(), grad_clipping)
        accuracy_mil = classification_accruracy_multi(sub_prediction, sub_label, softmax=True)            
        accuracy_final = classification_accruracy_multi(prediction, Label, softmax=True)
        whole_train_result.append(prediction)
        whole_train_label.append(Label)
        train_ave_accuracy.update(accuracy_final)
        if idx % write_iter_num == 0:
            tqdm.write(f'Epoch : {epoch + 1}/{epoch_num} {idx + 1}/{len(train_dataset)} '
                        f'Loss Bag : {loss_bag :.4f} '
                        f'Loss Final : {loss_final :.4f} '
                        f'Accuracy MIL : {accuracy_mil :.2f} '
                        f'Accuracy Final : {accuracy_final :.2f} '
                        f'Length of Data : {Input.shape[0]} '
                        f'File Name : {filename}')
        tqdm.write(f'Epoch : {epoch + 1}/{epoch_num} {idx + 1}/{len(train_dataset)} '
                    f'Loss Bag : {loss_bag :.4f} '
                    f'Loss Final : {loss_final :.4f} '
                    f'Accuracy MIL : {accuracy_mil :.2f} '
                    f'Accuracy Final : {accuracy_final :.2f} '
                    f'Length of Data : {Input.shape[0]} '
                    f'File Name : {filename}', file=file)
    tqdm.write(f'Epoch : {epoch + 1}/{epoch_num} '
            f'Train Average Accuracy : {train_ave_accuracy.average() :.2f} ' 
            f'Train F1 Score : {train_f1_score :.2f}', file=file)
    train_f1_score, _, _ = classification_f1_multi(torch.stack(whole_train_result, dim=0), torch.stack(whole_train_label, dim=0))
    print(f'Epoch : {epoch + 1}/{epoch_num} '
            f'Train Average Accuracy : {train_ave_accuracy.average() :.2f} ' 
            f'Train F1 Score : {train_f1_score :.2f} \n\n')
    file.close()

def valid(model=None, write_iter_num=5, valid_dataset=None, criterion=torch.nn.BCELoss(), device=None, epoch=None, file=None):
    ave_accuracy = AverageMeter()
    mil_model.eval()
    whole_valid_result = []
    whole_valid_label = []
    with torch.no_grad():
        for idx, valid_batch in enumerate(tqdm(valid_dataset)):
            #model input dat
            Image, label, filename = valid_batch
            Input = Variable(Image.squeeze(0).type(Tensor))
            Label = Variable(label.type(Tensor))
            ############ slide mil_model training
            prediction, sub_prediction, _, num_split = mil_model(Input, num_split=3)
            sub_label = label.repeat(num_split, 1).type(Tensor)
            ########## final mil_model training
            loss_bag = criterion(sub_prediction, sub_label)
            loss_final = criterion(prediction, Label)
            accuracy_mil = classification_accruracy_multi(sub_prediction, sub_label, softmax=True)            
            accuracy_final = classification_accruracy_multi(prediction, Label, softmax=True)
            
            whole_valid_result.append(prediction)
            whole_valid_label.append(Label)
            ave_accuracy.update(accuracy_final)
            if idx % write_iter_num == 0:
                tqdm.write(f'Epoch : {epoch + 1}/{epoch_num} {idx + 1}/{len(valid_dataset)} '
                        f'Validation Loss Bag: {loss_bag :.4f} '
                        f'Validation Loss Final: {loss_final :.4f} '
                        f'Validation Accuracy MIL: {accuracy_mil :.1f} '
                        f'Validation Accuracy Final: {accuracy_final :.1f} '
                        f'Best Accuracy : {best_val_loss :.2f} '
                        f'File Name : {filename}')
            tqdm.write(f'Epoch : {epoch + 1}/{epoch_num} {idx + 1}/{len(valid_dataset)} '
                        f'Validation Loss Bag: {loss_bag :.4f} '
                        f'Validation Loss Final: {loss_final :.4f} '
                        f'Validation Accuracy MIL: {accuracy_mil :.1f} '
                        f'Validation Accuracy Final: {accuracy_final :.1f} '
                        f'Best Accuracy : {best_val_loss :.2f} '
                        f'File Name : {filename}', file=file)
        valid_f1_score, _, _ = classification_f1_multi(torch.stack(whole_valid_result, dim=0), torch.stack(whole_valid_label, dim=0))
        print(f'Epoch : {epoch + 1}/{epoch_num} {idx + 1}/{len(valid_dataset)} '
                f'ValidAverage Accuracy : {ave_accuracy.average() :.2f} ' 
                f'Valid F1 Score : {valid_f1_score :.2f} \n\n') 