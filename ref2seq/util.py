import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence
from masked_cross_entropy import *
import itertools
import random
import math
import sys
import os
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_token, EOS_token, PAD_token, UNK_token
from config import MAX_LENGTH, USE_CUDA, teacher_forcing_ratio, save_dir
from config import MAX_LENGTH, save_dir
import pickle
import logging
logging.basicConfig(level=logging.INFO)

#############################################
# generate file name for saving parameters
#############################################
def filename(reverse, obj):
	filename = ''
	if reverse:
		filename += 'reverse_'
	filename += obj
	return filename

#############################################
# Prepare Training Data
#############################################

    
# def indexesFromSentence(voc, sentence):
#     ids = []
#     for word in sentence:
#         word = word.lower()
#         if word in voc.word2idx:
#             ids.append(voc.word2idx[word])
#         else:
#             ids.append(UNK_token)
#     return ids
    # return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# batch_first: true -> false, i.e. shape: seq_len * batch
def zeroPadding(l, max_target_len, fillvalue=PAD_token):
    for line in l:
        while(len(line)<max_target_len):
            line.append(fillvalue)
    return l

def binaryMatrix(l, value=PAD_token):
    m = []
    for i in range(len(l)):
        m.append([])
        for j in range(len(l[i])):
            if l[i][j] == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1) # mask = 1 if not padding
    return m

# return attribute index and input pack_padded_sequence
def inputVar(data, input_length, evaluation=False):
    userVar = [d[0] for d in data]
    businessVar = [d[1] for d in data]
    user_length = [d[0] for d in input_length]
    business_length = [d[1] for d in input_length]
    user_padVar = Variable(torch.LongTensor(userVar), volatile=evaluation)
    business_padVar = Variable(torch.LongTensor(businessVar), volatile=evaluation)

    padVar = [user_padVar, business_padVar]
    length = [user_length, business_length]
    return padVar, length 

# convert to index, add EOS, zero padding
# return output variable, mask, max length of the sentences in batch
def outputVar(l):
    max_target_len = max([len(indexes) for indexes in l])
    padList = zeroPadding(l, max_target_len)
    mask = binaryMatrix(padList)
    mask = Variable(torch.ByteTensor(mask))
    padVar = Variable(torch.LongTensor(padList))
    return padVar, mask, max_target_len

# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by output length, reverse input
# return input, lengths for pack_padded_sequence, output_variable, mask
def batch2TrainData(user_review, user_length, business_review, business_length, pair_batch, evaluation=False, train_mask_idx=None):
    
    input_batch, output_batch = [], []
    input_length = []
    if train_mask_idx is not None:
        for i in range(len(pair_batch)):
            mask_idx = train_mask_idx[i]
            mu = mask_idx[0]
            mb = mask_idx[1]
            user_review_i = user_review[str(pair_batch[i][0])].copy()
            business_review_i = business_review[str(pair_batch[i][1])].copy()  
            user_length_i = user_length[str(pair_batch[i][0])].copy()  
            business_length_i = business_length[str(pair_batch[i][1])].copy()
   
            smax = len(user_review_i[0])
            if mu < len(user_review_i):
                user_review_i[mu] = [SOS_token] + [EOS_token] + [PAD_token for _ in range(smax-2)]
                user_length_i[mu] = 2
            if mb < len(business_review_i):
                business_review_i[mb] = [SOS_token] + [EOS_token] + [PAD_token for _ in range(smax-2)]
                business_length_i[mb] = 2
            
            input_batch.append([user_review_i, business_review_i])
            input_length.append([user_length_i, business_length_i])
            output_batch.append([SOS_token]+pair_batch[i][2][:MAX_LENGTH]+[EOS_token])        
        
    else:    
        for i in range(len(pair_batch)):
            input_batch.append([user_review[str(pair_batch[i][0])], business_review[str(pair_batch[i][1])]])
            input_length.append([user_length[str(pair_batch[i][0])], business_length[str(pair_batch[i][1])]])
            output_batch.append([SOS_token]+pair_batch[i][2][:MAX_LENGTH]+[EOS_token])
        
    ## input_batch size (batch_size, 2, dmax, smax)
    review_input, input_length = inputVar(input_batch, input_length, evaluation=evaluation)
    output, mask, max_target_len = outputVar(output_batch) # convert sentence to ids and padding
    return review_input, input_length, output, mask, max_target_len

