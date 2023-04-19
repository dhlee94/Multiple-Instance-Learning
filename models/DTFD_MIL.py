import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from einops import rearrange

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, act_layer=nn.GELU, drop_rate=0.0):
        super(FeedForward, self).__init__()
        if hidden_channels==None:
            hidden_channels = in_channels//4
        self.drop_rate = drop_rate
        if self.drop_rate != 0.0:
            self.drop_out = torch.nn.Dropout(p=self.drop_rate)   
        self.feed_layer = nn.Sequential(
                    nn.LayerNorm(in_channels, eps=1e-5),
                    nn.Linear(in_channels, hidden_channels, bias=False),
                    act_layer(),
                    nn.Linear(hidden_channels, in_channels, bias=False)
        )
    def forward(self, x):
        if self.drop_rate != 0.0:
            x = self.drop_rate(x)
        return self.feed_layer(x)

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, num=1):
        super(Attention, self).__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(out_channels, num)
    def forward(self, x, isNorm=True):
        V = self.attention_V(x)
        U = self.attention_U(x)
        A = self.attention_weights(V*U)
        A = torch.transpose(A, 1, 0)
        if isNorm:
            A = F.softmax(A, dim=1)
        return A


class classifier(nn.Module):
    def __init__(self, in_channels, n_classes, drop_rate=0.0):
        super(classifier, self).__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.fc = nn.Linear(in_channels, n_classes)
        self.drop_rate = drop_rate
        if self.drop_rate != 0.0:
            self.drop_out = torch.nn.Dropout(p=self.drop_rate)    
    def forward(self, x):
        if self.drop_rate != 0.0:
            x = self.drop_out(self.layernorm(x))
        return self.fc(self.layernorm(x))
    
class DimReduction(nn.Module):
    def  __init__(self, in_channels, hidden_channels, out_channels, num_feed=1, act_layer=nn.GELU, drop_rate=0.0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels, bias=False)
        self.act = act_layer()
        self.num_feed = num_feed
        if num_feed > 0:
            self.feedforward = nn.ModuleList()    
            for _ in range(num_feed):
                self.feedforward.append(FeedForward(in_channels=out_channels, hidden_channels=hidden_channels, drop_rate=drop_rate))
            self.feedforward.append(FeedForward(in_channels=out_channels, hidden_channels=hidden_channels, drop_rate=drop_rate))
            self.feedforward = nn.Sequential(*self.feedforward)
    def forward(self, x):
        x = self.act(self.fc1(x))
        if self.num_feed>0:
            x = self.feedforward(x)+x
        return x

class Uclassifier(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=128, num=1, sub_attn=1, num_cls=2, drop_rate=0.0):
        super(Uclassifier, self).__init__()
        self.layers = nn.ModuleList([])
        for i in range(sub_attn):
            self.layers.append(nn.ModuleList([Attention(in_channels=in_channels, out_channels=hidden_channels, num=num),
                                              FeedForward(in_channels=in_channels, hidden_channels=in_channels//4, drop_rate=drop_rate)]))
        self.classifier = classifier(in_channels=in_channels, n_classes=num_cls, drop_rate=drop_rate)
    def forward(self, x):
        for attn, ff in self.layers:
            A = attn(x)
            feat = torch.mm(A, x)
            x = ff(feat)+feat
        return self.classifier(x)
    
class DTFD_MIL(nn.Module):
    def __init__(self, in_channels, dim_out=1280, attn_out=320, num_feed=0, sub_attn=2, instance_per_group=3, output_class=2, drop_rate=0.0):
        super(DTFD_MIL, self).__init__()
        self.instance_per_group = instance_per_group
        self.dim_reduction = DimReduction(in_channels=in_channels, hidden_channels=in_channels//4, 
                                          out_channels=dim_out, num_feed=num_feed, act_layer=nn.GELU)
        self.layers = nn.ModuleList([])
        for _ in range(sub_attn):
            self.layers.append(nn.ModuleList([Attention(in_channels=dim_out, out_channels=attn_out, num=1),
                                              FeedForward(in_channels=dim_out, hidden_channels=dim_out//4, drop_rate=drop_rate)]))
        
        self.classifier = classifier(in_channels=dim_out, n_classes=output_class, drop_rate=drop_rate)
        self.Uclassifier = Uclassifier(in_channels=dim_out, hidden_channels=attn_out, num=1, num_cls=output_class, drop_rate=drop_rate)
    
    def forward(self, x, num_split, distill='MaxMinS'):
        feat_index = list(range(x.shape[0]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), num_split)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]
        
        device = x.get_device()
        device = f'cuda:{device}' if device!=-1 else 'cpu'
        slide_pseudo_feat = []
        slide_sub_preds = []
        instance_per_group = self.instance_per_group // num_split
        for index in index_chunk_list:
            index = torch.Tensor(index).type(torch.LongTensor).to(device)
            split_x = torch.index_select(x, dim=0, index=index)
            tmp_x = self.dim_reduction(split_x)
            tmp_sub_x = tmp_x.clone()
            for attn, ff in self.layers:
                tmp_attn = attn(tmp_x).squeeze(0)
                tmp_x = torch.einsum('ns,n->ns', tmp_x, tmp_attn)
                tmp_x = ff(tmp_x)+tmp_x
            
            tmp_xattn_tensor = torch.sum(tmp_x, dim=0).unsqueeze(0)
            tpredict = self.classifier(tmp_xattn_tensor)
            slide_sub_preds.append(tpredict)
            
            patch_pred_logits = get_cam_1d(self.classifier, tmp_x.unsqueeze(0)).squeeze(0)
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)
            
            _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            
            if distill == 'MaxMinS':
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                MaxMin_inst_feat = tmp_sub_x.index_select(dim=0, index=topk_idx)
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif distill == 'MaxS':
                max_inst_feat = tmp_sub_x.index_select(dim=0, index=topk_idx_max)
                slide_pseudo_feat.append(max_inst_feat)
            elif distill == 'AFS':
                af_inst_feat = tmp_xattn_tensor
                slide_pseudo_feat.append(af_inst_feat)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
        predict = self.Uclassifier(slide_pseudo_feat)
        
        return predict, slide_sub_preds, slide_pseudo_feat, len(index_chunk_list)