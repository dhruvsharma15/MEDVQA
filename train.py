#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 20:02:42 2020

@author: dhruv
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

from model_mcb_baseline.MCB_baseline import MCB_baseline
from model_mcb_coatt.MCB_CoAtt import MCB_CoAtt

from model_mfb_baseline.MFB_baseline import MFB_baseline
from model_mfb_coatt.MFB_CoAtt import MFB_CoAtt

from model_mutan_baseline.MUTAN_baseline import MUTAN_baseline
from model_mutan_coatt.MUTAN_CoAtt import MUTAN_CoAtt

from data_loader import load_data
from utils import get_txt_file_content, make_answer_vocab
import argparse
from matplotlib import pyplot as plt
#from tensorboardX import SummaryWriter 

from sklearn.metrics import roc_auc_score

'''
python train.py --data_dir=../Data/ --train_data_path=../../MED-VQA/ImageClef-2019-VQA-Med-Training/QAPairsByCategory/C3_Organ_train.txt \ 
--val_data_path=../../MED-VQA/ImageClef-2019-VQA-Med-Validation/QAPairsByCategory/C3_Organ_val.txt \
--ques_model=bert --lstm_layers=2 --lstm_units=1024 --embed_size=1024 --pool_size=14 --feat_size=2048 --cat=cat3
'''

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../Data',
                        help='Data directory')
    parser.add_argument('--train_data_path', type=str, 
                        default='../MED-VQA/ImageClef-2019-VQA-Med-Training/All_QA_Pairs_train.txt',
                        help='path to training QA pairs')
    parser.add_argument('--val_data_path', type=str, 
                        default='../MED-VQA/ImageClef-2019-VQA-Med-Validation/All_QA_Pairs_val.txt',
                        help='path to validation QA pairs')
    parser.add_argument('--ques_model', type=str, default='bert',
                        help='Model for ques embedding')
    parser.add_argument('--cat', type=str, default='',
                        help='Category for QA')
    parser.add_argument('--lstm_layers', type=int, default=2,
                        help='No of LSTM layers')
    parser.add_argument('--lstm_units', type=int, default=1024,
                        help='No of LSTM layers')
    parser.add_argument('--embed_size', type=int, default=1024,
                        help='Embedding size of the question')
    parser.add_argument('--img_model', type=str, default='resnet152',
                        help='Model for image features')
    parser.add_argument('--feat_size', type=int, default=1024,
                        help='Feature size of the combined vector')
    parser.add_argument('--pool_size', type=int, default=8,
                        help='Global Avg Pool size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Epochs')
    parser.add_argument('--fusion', type=str, default='mcb',
                        help='Fusion Technique to be used')
    parser.add_argument('--att', type=str, default='yes',
                        help='Use attention module or not')
    
    
    args = parser.parse_args()
    print('attention:',args.att, type(args.att))
    if(args.fusion=='mcb'):
        if(args.att=='yes'):
            fusion_algo = MCB_CoAtt
        else:
            fusion_algo = MCB_baseline
    elif(args.fusion=='mfb'):
        if(args.att=='yes'):
            fusion_algo = MFB_CoAtt
        else:
            print(args.att)
            fusion_algo = MFB_baseline
    else:
        if(args.att=='yes'):
            fusion_algo = MUTAN_CoAtt
        else:
            print(args.att)
            fusion_algo = MUTAN_baseline
    
    print(fusion_algo)
    
    _, train_ques, train_ans = get_txt_file_content(args.train_data_path)
    _, val_ques, val_ans = get_txt_file_content(args.val_data_path)
    
    answers = train_ans + val_ans
    answer_to_ind_vocab, ind_to_answer_vocab = make_answer_vocab(answers)
    
    print(len(answer_to_ind_vocab))
    
    train_data = load_data(train_ans, answer_to_ind_vocab, args.data_dir, 
                           args.img_model, args.ques_model, 'train', args.cat)
    
    val_data = load_data(val_ans, answer_to_ind_vocab, args.data_dir, 
                         args.img_model, args.ques_model, 'val', args.cat)
    
    torch.cuda.set_device(0)
#    writer = SummaryWriter()
    
    print(train_data['answer'].shape)
    
    train_data_tensor = TensorDataset(torch.from_numpy(train_data['img_feat']),
                                      torch.from_numpy(train_data['ques_feat']),
                                      torch.from_numpy(train_data['answer']))
    val_data_tensor = TensorDataset(torch.from_numpy(val_data['img_feat']),
                                    torch.from_numpy(val_data['ques_feat']),
                                    torch.from_numpy(val_data['answer']))
    
    train_loader = DataLoader(train_data_tensor, batch_size = args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data_tensor, batch_size = args.batch_size, shuffle=True)
    
    model = fusion_algo(args.embed_size, args.lstm_units, args.lstm_layers, 
                        args.feat_size, args.batch_size, len(answer_to_ind_vocab),
                        args.pool_size)
    
    lr=0.001
    n_epochs = args.n_epochs
    print_every = 100

    criterion = nn.CrossEntropyLoss()   
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#    train_loss = np.zeros(n_epochs+1)
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    
    max_training_acc = 0
    max_val_acc = 0
    
    train_on_gpu = torch.cuda.is_available()
    if(train_on_gpu):
        model.cuda()
    
    for e in range(n_epochs):
        print('Epoch - ', e)
        running_acc = 0
        running_loss = 0
        counter = 0
        model.train()
        for img, ques, ans in train_loader:
            counter += 1
            
            img = img.cuda()
            ques = ques.float().cuda()
            ans = ans.cuda().long()
            
            optimizer.zero_grad()
            
            outputs = model(ques, img)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, ans)
            loss.backward()
            optimizer.step()
            
            acc = torch.sum(preds == ans).item()
            running_acc += acc
            running_loss += loss.item()
            
            if counter % print_every == 0:
                print('batch no.:', (counter*100)/len(train_loader), 
                      ' loss:', loss.item(),
                      ' acc:', acc/args.batch_size)
        
#            print(model.qatt_maps.shape, model.iatt_maps.shape)
        
        model.eval()
        val_running_loss = 0
        val_running_acc = 0
        for img, ques, ans in val_loader:
            img = img.cuda()
            ques = ques.cuda()
            ans = ans.cuda().long()
            
            val_output = model(ques, img)
            loss = criterion(val_output, ans)
            _, val_preds = torch.max(val_output, 1)
            val_running_loss += loss.item()
            val_running_acc += torch.sum(val_preds == ans).item()
        
        val_running_loss /= len(val_data['img_feat'])
        val_running_acc /= len(val_data['img_feat'])
        running_acc /= len(train_data['img_feat'])
        running_loss /= len(train_data['img_feat'])
        print('Validation loss:', val_running_loss,
              ' Validation acc:', val_running_acc,
              ' Training loss:', running_loss,
              ' Training acc:', running_acc)
        
        train_acc.append(running_acc)
        train_loss.append(running_loss)
        val_acc.append(val_running_acc)
        val_loss.append(val_running_loss)
        
        max_training_acc = max(max_training_acc, running_acc)
        max_val_acc = max(max_val_acc, val_running_acc)
        
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    ax1.plot(range(n_epochs), train_acc, 'r-x', label='train acc')
    ax1.plot(range(n_epochs), val_acc, 'b-o', label='val acc')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('accuracy')
    ax1.set_title('Epoch-Accuracy plot')
    ax1.legend()
    
    ax2.plot(range(n_epochs), train_loss, 'r-x', label='train loss')
    ax2.plot(range(n_epochs), val_loss, 'b-o', label='val loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('loss')
    ax2.set_title('Epoch-Loss plot')
    ax2.legend()
    
    print('attention:',args.att)
    if(args.att=='no'):
        coatt='baseline'
    else:
        coatt='coatt'

    file_name = os.path.join(os.path.join(args.data_dir, 'graphs'), args.cat+'_'+args.ques_model+'_'+args.img_model+'_'+args.fusion+'_'+coatt+'.png')
    fig.savefig(file_name)
    
    f = open(os.path.join(os.path.join(args.data_dir, 'logs'), args.cat+'_'+args.ques_model+'_'+args.img_model+'_'+args.fusion+'_'+coatt+'.txt'), 'w')
    f.write('category-'+args.cat+'\n')
    f.write('question embedding-'+ args.ques_model+'\n')
    f.write('image features-'+ args.img_model+'\n')
    f.write('model-'+args.fusion+'\n')
    f.write('training acc-'+ str(max_training_acc)+'\n')
    f.write('validation acc-'+ str(max_val_acc)+'\n')
    f.write('training accuracies-' + str(train_acc)+'\n')
    f.write('validation accuracies-' + str(val_acc)+'\n')
    
    f.close()
    
if __name__ == '__main__':
    main()