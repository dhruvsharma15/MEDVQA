#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:53:16 2020

@author: dhruv
"""

import torch
import torch.nn as nn
from torch.nn import AvgPool2d
import torch.nn.functional as F
from compact_bilinear_pooling import CompactBilinearPooling


class MCB_baseline(nn.Module):
    def __init__(self, embedding_size, LSTM_units, LSTM_layers, feat_size,
                 batch_size, ans_vocab_size, global_avg_pool_size, 
                 dropout = 0.3, MCB_output_dim = 5000):
        super(MCB_baseline, self).__init__()
        self.batch_size = batch_size
        self.ans_vocab_size = ans_vocab_size
        self.MCB_output_dim = MCB_output_dim
        self.feat_size = feat_size
        self.mcb_out = 1000
        self.mcb_factor = 5
        
        self.LSTM = nn.LSTM(input_size=embedding_size, hidden_size=LSTM_units, 
                            num_layers=LSTM_layers, batch_first=False)
        self.pool2d = AvgPool2d(global_avg_pool_size, stride=1)
        self.Dropout = nn.Dropout(p=dropout, )
        self.mcb = CompactBilinearPooling(self.feat_size, LSTM_units, self.MCB_output_dim)
        self.Linear_predict = nn.Linear(self.MCB_out, ans_vocab_size)
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
        
        iq_feat = self.mcb(img_feat_sq, ques_lstm)          # N x 5000
        iq_feat = self.Dropout(iq_feat)
        
        iq_resh = iq_feat.view(-1, 1, self.mcb_out, self.mcb_factor)     # N x 1 x 1000 x 5
        iq_sumpool = torch.sum(iq_resh, 3)                  # N x 1 x 1000 x 1
        iq_sumpool = torch.squeeze(iq_sumpool)              # N x 1000 
        
        iq_sqrt = torch.sqrt(F.relu(iq_sumpool)) - torch.sqrt(F.relu(-iq_sumpool))
        iq_norm = F.normalize(iq_sqrt)
                
        pred = self.Linear_predict(iq_norm)
        pred = self.Softmax(pred)
        
        return pred
        