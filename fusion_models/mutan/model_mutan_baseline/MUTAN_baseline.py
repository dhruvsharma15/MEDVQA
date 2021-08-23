#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:10:16 2020

@author: dhruv
"""

import torch
import torch.nn as nn
from torch.nn import AvgPool2d
import torch.nn.functional as F

class MutanFusion(nn.Module):
    def __init__(self, input_dim=1024, out_dim=5000, num_layers=5):
        super(MutanFusion, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        hv = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)

            hv.append(nn.Sequential(do, lin, nn.Tanh()))
        #
        self.image_transformation_layers = nn.ModuleList(hv)
        #
        hq = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hq.append(nn.Sequential(do, lin, nn.Tanh()))
        #
        self.ques_transformation_layers = nn.ModuleList(hq)

    def forward(self, ques_emb, img_emb):
        # Pdb().set_trace()
        batch_size = img_emb.size()[0]
        x_mm = []
        for i in range(self.num_layers):
            x_hv = img_emb
            x_hv = self.image_transformation_layers[i](x_hv)

            x_hq = ques_emb
            x_hq = self.ques_transformation_layers[i](x_hq)
            x_mm.append(torch.mul(x_hq, x_hv))
        #
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        x_mm = F.tanh(x_mm)
        return x_mm

class MUTAN_baseline(nn.Module):
    def __init__(self, embedding_size, LSTM_units, LSTM_layers, feat_size,
                 batch_size, ans_vocab_size, global_avg_pool_size, 
                 dropout = 0.3, mutan_output_dim = 5000):
        super(MUTAN_baseline, self).__init__()
        self.batch_size = batch_size
        self.ans_vocab_size = ans_vocab_size
        self.mutan_output_dim = mutan_output_dim
        self.feat_size = feat_size
        self.mutan_out = 1000
        
        self.mutan = MutanFusion(LSTM_units, self.mutan_out)
        self.LSTM = nn.LSTM(input_size=embedding_size, hidden_size=LSTM_units, 
                            num_layers=LSTM_layers, batch_first=False)
        self.pool2d = AvgPool2d(global_avg_pool_size, stride=1)
        self.Dropout = nn.Dropout(p=dropout, )
        self.Linear_predict = nn.Linear(self.mutan_out, ans_vocab_size)
        self.Softmax = nn.Softmax()
        
    def forward(self, ques_embed, img_feat):
        
        # pooling image features
        img_feat_resh = img_feat.permute(0, 3, 1, 2)        # N x w x w x C -> N x C x w x w
        img_feat_pooled = self.pool2d(img_feat_resh)        # N x C x 1 x 1
        img_feat_sq = img_feat_pooled.squeeze()             # N x C
        
        # ques_embed                                         N x T x embedding_size
        ques_embed_resh = ques_embed.permute(1, 0, 2)       #T x N x embedding_size
        lstm_out, (hn, cn) = self.LSTM(ques_embed_resh)
        ques_lstm = lstm_out[-1]                            # N x lstm_units
        ques_lstm = self.Dropout(ques_lstm)

        iq_feat = self.mutan(img_feat_sq, ques_lstm)
        
        iq_sqrt = torch.sqrt(F.relu(iq_feat)) - torch.sqrt(F.relu(-iq_feat))
        iq_norm = F.normalize(iq_sqrt)
                
        pred = self.Linear_predict(iq_norm)
        pred = self.Softmax(pred)
        
        return pred