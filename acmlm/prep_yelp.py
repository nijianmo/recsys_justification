import csv
from nltk import word_tokenize
import random
import os
import numpy as np
from tqdm import tqdm
from collections import Counter
#import cPickle as pickle
import string
import json
from collections import defaultdict
import sys
import argparse
from utilities import *

from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer

parser = argparse.ArgumentParser(description='Dataset Settings')
ps = parser.add_argument
ps('--dataset', dest='dataset')
args = parser.parse_args()
args.bert_model = 'bert-base-uncased'
args.do_lower_case = 'True'

tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab)


dataset = 'data/{}'.format(args.dataset)

#from utilities import *

def flatten(l):
    return [item for sublist in l for item in sublist]

def load_reviews(fp):
    with open('./{}/{}.json'.format(dataset, fp),'r') as f:
        data = json.load(f)
    return data

def load_set(fp):
    data= []
    with open('./{}/{}.txt'.format(dataset, fp),'r',encoding='utf8') as f:
      reader = csv.reader(f, delimiter='\t')
      for r in reader:
        data.append(r)
    return data

def sent2words(sent):
    #print("==========================")
    #print(sent)
    sent = sent.splitlines()
    sent = ' '.join(sent)
    sent = sent.lower()
    #_ sent =  sent.split(' ')
    
    _sent = []
    for token in sent.split():
        for sub_token in wordpiece_tokenizer.tokenize(token):
            _sent.append(sub_token)  
    
    return _sent

def review2words(review): # review -> list of reviews 
    return [sent2words(x) for x in review]

def get_words(data_dict):
    data_list = data_dict.items()
    reviews = [flatten(review2words(x[1])) for x in data_list]
    words = []
    for r in tqdm(reviews, desc='parsing words'):
        words += r
    return words

user_text = load_reviews('user_text')
item_text = load_reviews('item_text')

user_fa = load_reviews('user_fa')
item_fa = load_reviews('item_fa')

print("Number of Users={}".format(len(user_text)))
print("Number of Items={}".format(len(item_text)))


# load 
train  = load_set('train')
dev  = load_set('dev')
test  = load_set('test')
    
# check position of target edu
train_mask_idx = []
for d in train:
    user = d[0]
    item = d[1]
    texta = d[2]
    for idx, textb in enumerate(user_text[user]):
        if texta == textb:
            user_mask_idx = idx
            break
    for idx, textb in enumerate(item_text[item]):
        if texta == textb:
            item_mask_idx = idx
            break
    train_mask_idx.append((user_mask_idx, item_mask_idx))


# convert user/item index

user_ids = user_text.keys()
item_ids = item_text.keys()
#item_ids = list(item_text.keys())+list(item_text2.keys())

# numbering the keys
user_index = {key:index for index, key in enumerate(user_ids)}
item_index = {key:index for index, key in enumerate(item_ids)}

# for key, value in item_text.items():
#     if key == 'WzVc5o_bXS2C-ND7eEzwSg':
#         print(item_index[key])
#         print(value)

user_text = {user_index[key]:value for key, value in user_text.items()}
item_text = {item_index[key]:value for key, value in item_text.items()}

user_fa = {user_index[key]:value for key, value in user_fa.items()}
item_fa = {item_index[key]:value for key, value in item_fa.items()}        
            
        
# Preprocessing text

def preprocess_dict(data_dict):
    words = []
    for key, value in tqdm(data_dict.items(),desc='preprocessing'):
        # print("=============================")
        # print(value)
        new_val = review2words(value)
        # print(new_val)
        raw_words = flatten(new_val)
        # print(raw_words)
        words += raw_words # total words

        _str = [' '.join(x) for x in new_val] # join tokenized text
        
        data_dict[key] = _str
    return data_dict, words


user_text, words = preprocess_dict(user_text)
item_text, _ = preprocess_dict(item_text)

words = [x.lower() for x in words]


# print(user_negative.items()[:5])

print("Building Indexes")
word_index = tokenizer.vocab
index_word = {}
for k,v in word_index.items():
    index_word[v] = k

    
words = list(set(words))
#print(words[:50])

def word2id(word):
    try:
        return word_index[word]
    except:
        return 1 # 1 represents <unk>
    

def repr_convert(repr_dict, word_index):
    def sent2words(sent):
        sent = sent.split(' ')
        return [word2id(x.lower()) for x in sent]
    
    for key, value in tqdm(repr_dict.items(), desc='repr convert'):
        repr_dict[key] = [sent2words(x) for x in value]
    return repr_dict

user_text = repr_convert(user_text, word_index)
item_text = repr_convert(item_text, word_index)


def repr_convert_fa(repr_dict, word_index):
    def sent2words(sent):
        # sent is a dict
        sent = [s[0] for s in sent] # s[0] is fa, s[1] is its count    
        return sent
    
    for key, value in tqdm(repr_dict.items(), desc='repr convert'):
        repr_dict[key] = sent2words(value)
    return repr_dict

user_fa = repr_convert_fa(user_fa, word_index)
item_fa = repr_convert_fa(item_fa, word_index)

def process_set(d):
    try:
        '''
        #tokenize
        edu = d[2]  #process edu
        new_edu = review2words([edu])
        _edu = [word2id(x) for sub in new_edu for x in sub ] 
        '''
        
        edu = d[2]
        edu = tokenizer.tokenize(edu.lower())
        _edu = tokenizer.convert_tokens_to_ids(edu)
        
        fa2pos = eval(d[5])
        
        u = user_index[d[0]]
        i = item_index[d[1]]
        u_fa = user_fa[u]
        i_fa = item_fa[i]
        
        ui_fa = list(set(u_fa).intersection(set(i_fa)))
        for fa in i_fa:
            if len(ui_fa) >= 30:
                break
            if fa not in ui_fa:
                ui_fa.append(fa)

        return [[u, i, _edu, ui_fa, edu, fa2pos]]
    
    except:
        return []

train = [process_set(x) for x in train]
dev = [process_set(x) for x in dev]
test = [process_set(x) for x in test]

train =[x[0] for x in train if len(x)>0]
dev =[x[0] for x in dev if len(x)>0]
test =[x[0] for x in test if len(x)>0]

print('Train={} Dev={} Test={}'.format(len(train),len(dev),len(test)))

print('Vocab size = {}'.format(len(word_index)))
#print("Char Size ={}".format(len(char_index)))

fp = './{}/'.format(dataset)

if not os.path.exists(fp):
    os.makedirs(fp)

    
fa2idx, idx2fa = {}, {}
num_fa = 0
for k,v in user_fa.items():
    for vv in v:
        if vv not in fa2idx:
            fa2idx[vv] = num_fa
            idx2fa[num_fa] = vv
            num_fa += 1
for k,v in item_fa.items():
    for vv in v:
        if vv not in fa2idx:
            fa2idx[vv] = num_fa
            idx2fa[num_fa] = vv
            num_fa += 1
print("total number of fas = {}/{}".format(len(fa2idx), num_fa))


fa2tokids = {}
for fa in fa2idx:
    toks = tokenizer.tokenize(fa.lower())
    _toks = tokenizer.convert_tokens_to_ids(toks)
    #fa2tokids[fa] = _toks    
    fa2tokids[fa2idx[fa]] = _toks    
    
# convert fa in train,dev,test into fa index
for d in train:
    ui_fa = d[3]
    d[3] = [fa2idx[fa] for fa in ui_fa]
for d in dev:
    ui_fa = d[3]
    d[3] = [fa2idx[fa] for fa in ui_fa]
for d in test:
    ui_fa = d[3]
    d[3] = [fa2idx[fa] for fa in ui_fa]  
    
env = {
  'train':train,
  'train_mask_idx':train_mask_idx,
  'dev':dev,
  'test':test,
  'user_text':user_text,
  'item_text':item_text,
  'user_fa':user_fa,
  'item_fa':item_fa,
  'index_word':index_word,
  'word_index':word_index,
  'fa2idx':fa2idx,
  'idx2fa':idx2fa,
  'fa2tokids':fa2tokids,  
  'user_index':user_index,
  'item_index':item_index
}

dictToFile(env,'{}env.json'.format(fp))
