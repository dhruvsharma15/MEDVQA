#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:07:42 2020

@author: dhruv
"""

from utils import list_to_dict, load_image_features, load_ques_embed
import numpy as np

def prepare_answers(answer_to_ind_vocab, answers):
    answers_ind = np.zeros(len(answers))
    for i in range(len(answers)):
        answers_ind[i] = answer_to_ind_vocab[answers[i]]
        
    return answers_ind

def prepare_ques_img_pair(img_id_list, img_feat, ques_id_list, ques_embed):
    data = dict()
    img_id_list_dict = list_to_dict(img_id_list)
    img_feat_data = list()
    
    for ques_id in ques_id_list:
        img_feat_data.append(img_feat[img_id_list_dict[ques_id]])
        
    img_feat_data = np.array(img_feat_data)
    
    data['img_feat'] = img_feat_data
    data['ques_feat'] = ques_embed
    
    return data

def load_data(answers, answers_vocab, data_dir, img_model, ques_model, split, ques_cat):
    answers_ind = prepare_answers(answers_vocab, answers)
    img_feat, image_id_list = load_image_features(data_dir, img_model, split)
    ques_embed, ques_id_list = load_ques_embed(data_dir, ques_model, split, ques_cat)
    
    data = prepare_ques_img_pair(image_id_list, img_feat, ques_id_list, ques_embed)
    data['answer'] = answers_ind
    
    print(data['img_feat'].shape)
    
    return data
    