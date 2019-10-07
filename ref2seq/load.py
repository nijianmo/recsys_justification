import re
import os
import unicodedata
import pickle
import numpy as np
from config import MAX_LENGTH, save_dir
from utilities import *
# Our dataset are three dicts: user_review, business_review, user_business_EDU
# depends on the word_vocab file
PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3
class Voc:
    def __init__(self,env):
        self.index_word = env['index_word']
        self.word_index = env['word_index']
        self.n_words = len(self.word_index)

class Data:
    def __init__(self,env, dmax=10, smax = 20):
        self.dmax = dmax  #max number of edu
        self.smax = smax   #max number of token for a edu
        self.voc = Voc(env)
        self.train = env['train']
        self.train_mask_idx = env['train_mask_idx']
        self.dev = env['dev'] 
        self.test = env['test']
        self.user_text = env['user_text'] 
        self.item_text = env['item_text'] 
        self.user_index = env['user_index'] 
        self.item_index = env['item_index']
        
    

def pad_to_max(seq, seq_max, pad_token=0):
    while(len(seq)<seq_max): 
        seq.append(pad_token)
    return seq[:seq_max]

def prep_hierarchical_data_list(data, smax, dmax):
   
    all_data = dict()
    all_lengths = dict()
    for key, value in tqdm(data.items(), desc='building H-dict'):
        new_data = []
        data_lengths = []

        loop = range(0, len(value))
        for idx in loop:
            data_list = [SOS_token]+value[idx][:smax-2]+[EOS_token]
            sent_lens = len(data_list)
            if sent_lens==0:
                continue
            if sent_lens>smax:
                sent_lens = smax

            _data_list = pad_to_max(data_list, smax)
            new_data.append(_data_list)
            data_lengths.append(sent_lens)
            if len(new_data) >= dmax: # skip if already reach dmax
                break            
        new_data = pad_to_max(new_data, dmax, # dmax - early skip!
                            pad_token=[SOS_token]+[EOS_token]+[0 for _ in range(smax-2)])

        data_lengths = pad_to_max(data_lengths, dmax, pad_token=2)
        
        all_data[key] = new_data
        all_lengths[key] = data_lengths
    return all_data, all_lengths


def loadPrepareData(args):

    print("Start loading...")
    path = '{}/{}/env.json'.format(save_dir, args.corpus)
    print(path)
    env = dictFromFileUnicode(path)
    print('loading done...')
    ##prepare review data
    data = Data(env, args.dmax, args.smax)
    data.user_text, user_length = prep_hierarchical_data_list(data.user_text,  data.smax, data.dmax)
    data.item_text, item_length = prep_hierarchical_data_list(data.item_text, data.smax, data.dmax)
    length = [user_length, item_length]#, user_length2, item_length2]
    return data, length   #voc, user_review, business_review, user_business_EDU, train_pairs, valid_pairs, test_pairs

if __name__ == '__main__':
    loadPrepareData()
    
