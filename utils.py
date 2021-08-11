#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:51:13 2020

@author: dhruv
"""

from os.path import join
import numpy as np
import h5py
from nltk import word_tokenize
from collections import Counter
import operator
import torch
from sklearn.metrics import f1_score

def get_txt_file_content(filepath):
    f = open(filepath, 'r')
    content = f.readlines()
    
    data = []
    questions = []
    answers = []
    for line in content:
        if(line[-1]=='\n'):    
            info = line[:-1].split('|')
        else:
            info = line.split('|')
        data.append({
            "image_id" : info[0],
            "question" : info[1],
            "answer" : info[2],
        })
        questions.append(info[1])
        answers.append(info[2])
    
    return data, questions, answers

def make_answer_vocab(answers):
	top_n = 2000
	answer_frequency = {} 
	for answer in answers:
		if answer in answer_frequency:
			answer_frequency[answer] += 1
		else:
			answer_frequency[answer] = 1

	answer_frequency_tuples = [ (-frequency, answer) for answer, frequency in answer_frequency.items()]
	answer_frequency_tuples.sort()
	answer_frequency_tuples = answer_frequency_tuples[0:top_n-1]

	answer_to_ind_vocab = {}
	ind_to_answer_vocab = {}

	for i, ans_freq in enumerate(answer_frequency_tuples):
		# print i, ans_freq
		ans = ans_freq[1]
		answer_to_ind_vocab[ans] = i
		ind_to_answer_vocab[i] = ans

	answer_to_ind_vocab['UNK'] = min(top_n - 1, len(answer_to_ind_vocab.keys()))
	ind_to_answer_vocab[answer_to_ind_vocab['UNK']] = 'UNK'
	return answer_to_ind_vocab, ind_to_answer_vocab

def load_image_features(data_dir='../Data', model='resnet152', split='Train'):
    fc7_features = None
    image_id_list = None
    with h5py.File( join( join(data_dir, model), (split + '_fc7.h5')),'r') as hf:
        fc7_features = np.array(hf.get('fc7_features'))
    with h5py.File( join( join(data_dir, model), (split + '_image_id_list.h5')),'r') as hf:
        image_id_list = np.array(hf.get('image_id_list'))
        image_id_list = [id.decode('UTF-8') for id in image_id_list]
        
    return fc7_features, image_id_list

def load_ques_embed(data_dir='../Data', model='bert', split='train', cat='allQA'):
    embeddings = None
    image_id_list = None
    cat_name = ""
    if(cat=='allQA'):
        cat_name = '_All_QA_Pairs_'
    elif(cat=='cat1'):
        cat_name = '_C1_Modality_'
    elif(cat=='cat2'):
        cat_name = '_C2_Plane_'
    elif(cat=='cat3'):
        cat_name = '_C3_Organ_'
    else:
        cat_name = '_C4_Abnormality_'
        
    root_path = join(join(join(data_dir, model), cat))
    with h5py.File( join( root_path, (split + cat_name +split+'_embed.h5')),'r') as hf:
        embeddings = np.array(hf.get('embedding'))
    with h5py.File( join( root_path, (split + cat_name +split+'_image_id_list.h5')),'r') as hf:
        image_id_list = np.array(hf.get('image_id_list'))
        image_id_list = [id.decode('UTF-8') for id in image_id_list]
        
    return embeddings, image_id_list

def list_to_dict(id_list):
    id_dict = {}
    count = 0
    for obj_id in id_list:
        id_dict[obj_id] = count
        count += 1
    
    return id_dict

def make_word_map(answers):
    """
    Makes word vocabulary to index words
    
    Args:
        answers: list of answers
    
    Returns:
        Dict: word to num map
        Dict: num to word map
    """
    word_to_num = dict()
    num_to_word = dict()
    words = list()
    
    for ans in answers:
        tokens = word_tokenize(ans)
        for t in tokens:
            words.append(t)
            
    counter = Counter(words)
    sorted_counter = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
    
    for ind, key in enumerate(sorted_counter):
        word_to_num[key[0]] = ind+1
        num_to_word[ind+1] = key[0]
        
    word_to_num['<start>'] = len(word_to_num)+1
    num_to_word[len(num_to_word)+1] = '<start>'
    
    word_to_num['<end>'] = len(word_to_num)+1
    num_to_word[len(num_to_word)+1] = '<end>'
    
    word_to_num['<unk>'] = len(word_to_num)+1
    num_to_word[len(num_to_word)+1] = '<unk>'
    
    word_to_num['<pad>'] = 0
    num_to_word[0] = '<pad>'
    
    return word_to_num, num_to_word

def encode_answers(answers, word_to_num):
    """
    This method is to encode the tokens in an answer by the corresponding number
    
    Args:
        answers: the list of answers
        word_to_num: the mapping of word to number
        
    Returns:
        List of nums: encoded answers
        List: of answer lengths
    """
    tokenized_answers = list()
    max_len = 0
    for ans in answers:
        tokens = word_tokenize(ans)
        max_len = max(max_len, len(tokens)+2)
        tokenized_answers.append(tokens)
    
    encoded_ans = list()
    ans_len = list()
    for ind, tokens in enumerate(tokenized_answers):
        enc_c = [word_to_num['<start>']] + [word_to_num.get(word, word_to_num['<unk>']) for word in tokens] + [
                        word_to_num['<end>']] + [word_to_num['<pad>']] * (max_len - len(tokens))
        encoded_ans.append(enc_c)
        ans_len.append(len(tokens)+2)
        
    encoded_ans = np.array(encoded_ans)
    ans_len = np.array(ans_len)
        
    return encoded_ans, ans_len
    
def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def save_checkpoint(data_dir, cat, epoch, encoder, decoder, encoder_optimizer, 
                    decoder_optimizer, bleu4):
    """
    Saves model checkpoint.
    :param data_dir: directory where the ckpt is to be stored
    :param cat: base name of processed dataset
    :param epoch: epoch number
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    """
    state = {'epoch': epoch,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + cat + '.pth.tar'
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    torch.save(state, join(data_dir, 'BEST_' + filename))
        
def unpack_seq(sequence, end_token, batch_size=32):
    """
    To convert the sequence vector to a list of seqs
    :sequence: the seq to be converted
    :end_token: the token that marks the end of the seq
    :batch_size: batch size
    :return the list of sequences:
    """
    seq_list = [[] for i in range(batch_size)]
    count = 0
    for i, num in enumerate(sequence):
        if(len(seq_list[count%batch_size])==0 or seq_list[count%batch_size][-1]!=end_token):
            seq_list[count%batch_size].append(num)
            count += 1
        elif(seq_list[count%batch_size][-1]==end_token):
            count = 0
            seq_list[count%batch_size].append(num)
            count += 1
    
    return seq_list

def write_output(output_path, input_path, run, answers):
    """
    To write the answer in the output file
    
    Args:
        output_path : path where the file needs to be written
        input_path : path for the test file
        run : the run ID
        answers : list of answers to be written
    """
    output_file = join(output_path, 'output_run_'+str(run)+'.txt')
    o = open(output_file, 'w')
    i= open(input_path, 'r')
    questions = i.readlines()
    for ind, ques in enumerate(questions):
        img = ques.split('|')[0]
        line = img+'|'+answers[ind]+'\n'
        o.write(line)
    
    i.close()
    o.close()
    
def f1(scores, targets):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :return: top-k f1 score
    """

    _, ind = scores.topk(1, 1, True, True)
#    print(targets.shape, ind.shape)
    ind = ind.squeeze()
    val = f1_score(targets, ind, average = 'macro')

    return val