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
parser = argparse.ArgumentParser(description='Dataset Settings')
ps = parser.add_argument
ps('--dataset', dest='dataset')
args = parser.parse_args()


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
    #_ sent =  sent.split(' ')
    _sent = tylib_tokenize(sent, setting='nltk', lower=True)
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

# user_text2 = load_reviews('user_text2')
# item_text2 = load_reviews('item_text2')

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

# user_text2 = {user_index[key]:value for key, value in user_text2.items()}
# item_text2 = {item_index[key]:value for key, value in item_text2.items()}

# for key,value in item_text.items():
#     if key == 11345:
#         print(value)
        
            
        
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
        # print(_str)
        # for s in _str:
        #     print(s)
        data_dict[key] = _str
    return data_dict, words


user_text, words = preprocess_dict(user_text)
item_text, _ = preprocess_dict(item_text)
# print('*****')
# print(item_text[11345])
#print(user_text[1])

words = [x.lower() for x in words]

# user_dict = defaultdict(list)
# rating_dict = {}
# for t in tqdm(all_edus, desc='rebuilding user dict'):
#   user_dict[t[0]].append(t[1])
#   rating_dict[str(tuple([t[0],t[1]]))] = t[2]

# make ranking dictionary
# testing_users = [x[0] for x in test]
# testing_users += [x[0] for x in dev]

# testing_users = list(set(testing_users))
# print("Number of unique testing users={}".format(len(testing_users)))

# sample_count = 100

# user_negative = {}

# all_items = set([i for i in range(len(item_index))])

# for user in tqdm(testing_users, desc='build test'):
#   # Get ratings
#   ui = set(user_dict[user])
#   never_rated = list(all_items - ui)
#   _sample_count = min(len(never_rated), sample_count)
#   # print(len(never_rated))
#   sampled = random.sample(never_rated, _sample_count) # sample from never rated items
#   # print(sampled)
#   sampled = [str(x) for x in sampled]
#   user_negative[user] = ' '.join(sampled[:_sample_count])


# print(user_negative.items()[:5])

print("Building Indexes")
word_index, index_word = build_word_index(words,
                              # min_count=0,
                              min_count=5,
                              extra_words=['<pad>','<unk>','<sos>','<eos>'],
                              lower=True)

words = list(set(words))
#print(words[:50])

def word2id(word):
        try:
            return word_index[word]
        except:
            return 1 # 1 represents <unk>

def process_set(d):
    try:
        edu = d[2]  #process edu
        new_edu = review2words([edu])
        _edu = [word2id(x) for sub in new_edu for x in sub ] 
        return [[user_index[d[0]], item_index[d[1]], _edu]]
    except:
        return []

train = [process_set(x) for x in train]
dev = [process_set(x) for x in dev]
test = [process_set(x) for x in test]

train =[x[0] for x in train if len(x)>0]
dev =[x[0] for x in dev if len(x)>0]
test =[x[0] for x in test if len(x)>0]

print('Train={} Dev={} Test={}'.format(len(train),len(dev),len(test)))

all_edus = train + dev + test

def repr_convert(repr_dict, word_index):
    def sent2words(sent):
        sent = sent.split(' ')
        return [word2id(x.lower()) for x in sent]
    for key, value in tqdm(repr_dict.items(), desc='repr convert'):
        repr_dict[key] = [sent2words(x) for x in value]
    return repr_dict

user_text = repr_convert(user_text, word_index)
item_text = repr_convert(item_text, word_index)
# item_text2 = repr_convert(item_text2, word_index)
# user_text2 = repr_convert(user_text2, word_index)
# print('**')
# print(item_text[11345])
# print("Collecting Characters..")
# chars = []
# for t in tqdm(words, desc='Collecting Chars'):
#     for c in t:
#         chars += c


# char_index, index_char = build_word_index(chars,
#                             min_count=0,
#                             extra_words=['<pad>','<unk>','<br>'],
#                             lower=False)


print('Vocab size = {}'.format(len(word_index)))
#print("Char Size ={}".format(len(char_index)))

fp = './{}/'.format(dataset)

if not os.path.exists(fp):
    os.makedirs(fp)

# build_embeddings(word_index, index_word, out_dir=fp,
#   init_type='uniform', init_val=0.01,
#   normalize=False, emb_types=[('glove',50)])
#
# print("Saved Glove")

env = {
  'train':train,
  'train_mask_idx':train_mask_idx,
  'dev':dev,
  'test':test,
  'user_text':user_text,
  'item_text':item_text,
  #'user_text2':user_text2,
  #'item_text2':item_text2,
  'index_word':index_word,
  'word_index':word_index,
  #'char_index':char_index,
  'user_index':user_index,
  'item_index':item_index
}

dictToFile(env,'{}env.json'.format(fp))
