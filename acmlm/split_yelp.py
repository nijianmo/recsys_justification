import json
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
import re
import gzip
#from keras.preprocessing import sequence
import numpy as np
#import codecs
import operator
import datetime
import csv
import argparse
import gzip
import string
import random
random.seed(999)

from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer

parser = argparse.ArgumentParser(description='Dataset Settings')
ps = parser.add_argument
ps('--dataset', dest='dataset') # category
args = parser.parse_args()
args.bert_model = 'bert-base-uncased'
args.do_lower_case = 'True'

#main = ''
in_fp = './data/{}/{}_filter_flat_positive.large.json'.format(args.dataset, args.dataset)

tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab)


limit = 99999999999
min_reviews = 5
max_words = 1500

user_count = Counter()
item_count = Counter()
users = defaultdict(list)
items = defaultdict(list)

pairs = set()
interactions = defaultdict(list)

print("Building count dictionary first to save memory...")
#with gzip.open(in_fp, 'r') as f:
with open(in_fp, 'r') as f:
    for i, line in tqdm(enumerate(f), desc = "1st pass of reviews"):
        if(i>limit):
            break
        d = json.loads(line)
        user = d['user_id']
        item = d['business_id']
        fas = d['fine_aspect']
        if len(fas) <= 0:
            continue
        if (user, item) in pairs:
            continue
        pairs.add((user, item)) # only keep one edu per (u,i) pair
        user_count[user] += 1
        item_count[item] += 1

pairs = set()
#with gzip.open(in_fp, 'r') as f:
with open(in_fp, 'r') as f:
    for i,line in tqdm(enumerate(f), desc = "2nd pass of reviews"):
        if(i>limit):
            break
        d = json.loads(line)

        user = d['user_id']
        item = d['business_id']
        #rating = d['stars']
        time = d['date']
        ts = time
        # ts = int(datetime.datetime.strptime(time,
        #                     '%Y-%m-%d').strftime("%s"))
        
        fas = d['fine_aspect']
        if len(fas) <= 0:
            continue
        if (user, item) in pairs:
            continue
        pairs.add((user, item)) # only keep one edu per (u,i) pair
        
        if(user_count[user] < min_reviews or item_count[item] < min_reviews):
            continue
        
        text = d['edu'].strip(string.punctuation+' '+'\n')   #remove punctuation space \n
        interactions[user].append([item, ts, text, fas])

print("Number of users={}".format(len(interactions)))
# Filter interactions 2nd time

new_interactions = defaultdict(list)
new_items = []
for key, value in interactions.items():
    # if(len(value)<min_reviews):
    #     continue
    # else:
    new_interactions[key] = value
    new_items += [x[0] for x in value]

print('Filtered Users={}'.format(len(new_interactions)))

new_items_dict = dict(Counter(new_items)) # count item occurrence
new_interactions2 = defaultdict(list)
new_items2 = []

for key, value in new_interactions.items():
    new_v = [x for x in value if new_items_dict[x[0]]>=min_reviews]
    if(len(new_v)<min_reviews):
        continue
    else:
        new_interactions2[key] = new_v
        new_items2 += [x[0] for x in new_v]

num_items = len(list(set(new_items2)))

print("Filtered Users={}".format(len(new_interactions2)))
print("Final number of items={}".format(num_items))

interactions = new_interactions2

# split train/dev/test
import random

train = defaultdict(list)
dev = defaultdict(list)
test = defaultdict(list)

user_repr = defaultdict(list)
item_repr = defaultdict(list)

user_repr_fa = defaultdict(dict)
item_repr_fa = defaultdict(dict)

interaction_list = []

train, dev, test = [], [],[]

# make reviews
for user, items in tqdm(interactions.items(), desc='make interactions'):
    if(len(items)<2):
        continue
    #sorted_items = sorted(items, key=operator.itemgetter(2)) # sort based on time
    random.shuffle(items)
    
    train +=  [[user,x[0],x[-2],x[-1]] for x in items[:-2]] 
    dev += [[user, items[-2][0], items[-2][-2], items[-2][-1]]]
    test += [[user, items[-1][0], items[-1][-2], items[-1][-1]]]
    # hold the most recent two as dev/test

# user fa representation
for t in train:
    user_repr[t[0]].append(t[-2]) 
    item_repr[t[1]].append(t[-2])
    for fa in t[-1]:
        if fa in user_repr_fa[t[0]]:
            user_repr_fa[t[0]][fa] += 1
        else:
            user_repr_fa[t[0]][fa] = 1
        if fa in item_repr_fa[t[1]]:
            item_repr_fa[t[1]][fa] += 1
        else:
            item_repr_fa[t[1]][fa] = 1    

for t in test:
    if t[1] not in item_repr:
        print('****{} is not in train***'.format(t[1]))
        item_repr[t[1]].append('')
        item_repr_fa[t[1]]['<pad>'] = 1

for t in dev:
    if t[1] not in item_repr:
        print('****{} is not in train***'.format(t[1]))
        item_repr[t[1]].append('')
        item_repr_fa[t[1]]['<pad>'] = 1

print(list(user_repr.items())[0])


###
### add skeleton and subword position
for d in train:
    edu = d[2]
    fas = d[3]
    idx2fa = {}
    for k,v in fas.items():
        for vv in v:
            idx2fa[vv] = k
    _edu = []
    fa2pos={}
    for idx, token in enumerate(edu.lower().split()):
        sub_tokens = wordpiece_tokenizer.tokenize(token)
        pos = []
        curr_idx = len(_edu)
        for sub_token in sub_tokens:
            _edu.append(sub_token)
            pos.append(curr_idx)
            curr_idx += 1
        if idx in idx2fa:
            fa = idx2fa[idx]
            fa2pos[fa] = pos
    d.append(_edu)
    d.append(fa2pos)        

for d in dev:
    edu = d[2]
    fas = d[3]
    idx2fa = {}
    for k,v in fas.items():
        for vv in v:
            idx2fa[vv] = k
    _edu = []
    fa2pos={}
    for idx, token in enumerate(edu.lower().split()):
        sub_tokens = wordpiece_tokenizer.tokenize(token)
        pos = []
        curr_idx = len(_edu)
        for sub_token in sub_tokens:
            _edu.append(sub_token)
            pos.append(curr_idx)
            curr_idx += 1
        if idx in idx2fa:
            fa = idx2fa[idx]
            fa2pos[fa] = pos
    d.append(_edu)
    d.append(fa2pos)        

for d in test:
    edu = d[2]
    fas = d[3]
    idx2fa = {}
    for k,v in fas.items():
        for vv in v:
            idx2fa[vv] = k
    _edu = []
    fa2pos={}
    for idx, token in enumerate(edu.lower().split()):
        sub_tokens = wordpiece_tokenizer.tokenize(token)
        pos = []
        curr_idx = len(_edu)
        for sub_token in sub_tokens:
            _edu.append(sub_token)
            pos.append(curr_idx)
            curr_idx += 1
        if idx in idx2fa:
            fa = idx2fa[idx]
            fa2pos[fa] = pos
    d.append(_edu)
    d.append(fa2pos)        
    

print("==========================")
print("Set Stats")
print(len(train))
print(len(dev))
print(len(test))
print("==========================")


print(train[:10])
print(dev[:10])

def write_interactions(fp, data, mode='json'):
    with open(fp, 'w+',encoding='utf8') as f:
        if(mode=='csv'):
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(data)
        else:
            json.dump(data, f, indent=4)

print("Finished running file..")

fp = './data/{}/'.format(args.dataset)
import os

if not os.path.exists(fp):
    os.makedirs(fp)

write_interactions('{}train.txt'.format(fp), train, mode='csv')
write_interactions('{}dev.txt'.format(fp), dev, mode='csv')
write_interactions('{}test.txt'.format(fp), test, mode='csv')

write_interactions('{}user_text.json'.format(fp), user_repr)
write_interactions('{}item_text.json'.format(fp), item_repr)

# sort
for u in user_repr_fa:
    temp = sorted(user_repr_fa[u].items(), key=lambda x:x[1], reverse=True) #list
    user_repr_fa[u] = temp
for i in item_repr_fa:
    temp = sorted(item_repr_fa[i].items(), key=lambda x:x[1], reverse=True) #list
    item_repr_fa[i] = temp
    
write_interactions('{}user_fa.json'.format(fp), user_repr_fa)
write_interactions('{}item_fa.json'.format(fp), item_repr_fa)
# write_interactions('{}user_text2.json'.format(fp), user_repr2)
# write_interactions('{}item_text2.json'.format(fp), item_repr2)

print("Finished running file..")
