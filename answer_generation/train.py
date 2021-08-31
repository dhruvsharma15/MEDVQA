import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from model import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

from data_loader_gen import load_data
from utils import get_txt_file_content, make_word_map, encode_answers
import argparse
from matplotlib import pyplot as plt
#from tensorboardX import SummaryWriter 

from sklearn.metrics import roc_auc_score

encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?

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
    parser.add_argument('--decoder_att_size', type=int, default=1024,
                        help='Decoder attention size')
    parser.add_argument('--decoder_embed_dim', type=int, default=1024,
                        help='embedding size of decoder')
    parser.add_argument('--decoder_dim', type=int, default=1024,
                        help='dimension of decoder')
    
    
    args = parser.parse_args()
    
    _, train_ques, train_ans = get_txt_file_content(args.train_data_path)
    _, val_ques, val_ans = get_txt_file_content(args.val_data_path)
    
    answers = train_ans + val_ans
    word_to_num, num_to_word = make_word_map(answers)
        
    train_data = load_data(train_ans, word_to_num, args.data_dir, 
                           args.img_model, args.ques_model, 'train', args.cat)
    
    val_data = load_data(val_ans, word_to_num, args.data_dir, 
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
    
    encoder = Encoder(args.embed_size, args.lstm_units, args.lstm_layers, 
                        args.feat_size, args.batch_size, args.pool_size)
    decoder = DecoderWithAttention(args.decoder_att_size, args.decoder_embed_dim, 
                                    args.decoder_dim, len(word_to_num))
    
    lr=0.001
    n_epochs = args.n_epochs
    print_every = 100

    criterion = nn.CrossEntropyLoss()   
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=decoder_lr)
#    train_loss = np.zeros(n_epochs+1)
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []

    losses = AverageMeter()
    top5accs = AverageMeter()
    
    max_training_acc = 0
    max_val_acc = 0
    
    train_on_gpu = torch.cuda.is_available()
    if(train_on_gpu):
        encoder.cuda()
        decoder.cuda()
    
    for e in range(n_epochs):
        print('Epoch - ', e)
        running_acc = 0
        running_loss = 0
        counter = 0
        encoder.train()
        decoder.train()
        for img, ques, ans, ans_len in train_loader:
            counter += 1
            
            img = img.cuda()
            ques = ques.float().cuda()
            ans = ans.cuda().long()
            ans_len = ans_len.cuda().long()

            encoder_output = encoder(img, ques)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoder_output, ans, ans_len)

            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss.backward()

            # Update weights
            decoder_optimizer.step()
            encoder_optimizer.step()

            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))

            running_loss += loss.item()
            
            if counter % print_every == 0:
                print('batch no.:', (counter*100)/len(train_loader), 
                      ' loss:', loss.item())
        
#            print(model.qatt_maps.shape, model.iatt_maps.shape)
        
        encoder.eval()
        decoder.eval()

        val_running_loss = 0
        val_running_acc = 0
        for img, ques, ans, ans_len in val_loader:
            counter += 1
            
            img = img.cuda()
            ques = ques.float().cuda()
            ans = ans.cuda().long()
            ans_len = ans_len.cuda().long()

            encoder_output = encoder(img, ques)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoder_output, ans, ans_len)

            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

if __name__ == '__main__':
    main()