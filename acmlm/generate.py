### generate

from __future__ import absolute_import, division, print_function, unicode_literals
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from collections import defaultdict

import argparse
import logging
import os
import random
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.modeling import BertForPreTraining
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

import math
import sys
import time
import datetime
import itertools
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_token, EOS_token, PAD_token, UNK_token
from model import *
from bert_decoder import *


parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--corpus",
                    default='yelp',
                    type=str,
                    help="The input corpus.")
parser.add_argument("--bert_model", default='bert-base-uncased', type=str, 
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
parser.add_argument("--output_dir",
                    default='tmp',
                    type=str,
                    help="The output directory where the model checkpoints will be written.")

## Other parameters
parser.add_argument("--max_seq_length",
                    default=24,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    action='store_true',
                    help="Whether to run evaluation.")
parser.add_argument("--train_batch_size",
                    default=16,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=16,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--learning_rate",
                    default=3e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--on_memory",
                    action='store_true',
                    help="Whether to load train samples into memory or use disk")
parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumualte before performing a backward/update pass.")
parser.add_argument('--fp16',
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale',
                    type = float, default = 0,
                    help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                    "0 (default value): dynamic loss scaling.\n"
                    "Positive power of 2: static loss scaling value.\n")
parser.add_argument('-dm', '--dmax', type=int, default=5, help='Max number of documents')
parser.add_argument('-sm', '--smax', type=int, default=20, help='Max number of words in each document')
parser.add_argument('-K', '--K', type=int, default=40, help='Max num aspect kept')
parser.add_argument("--model_file", default='tmp/pytorch_model_0.bin', type=str, 
                    help="Pretrained fine-tuned model.")

args = parser.parse_args()
print(args)

# Load pre-trained model (weights)
model_version = 'bert-base-uncased'
model = BertMLMDecoder.from_pretrained(model_version)
model_file = args.model_file
model.load_state_dict(torch.load(model_file))
model.eval()
cuda = torch.cuda.is_available()
if cuda:
    model = model.cuda()

data, length = loadPrepareData(args)
user_length, item_length = length #, user_length2, item_length2 = length

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=model_version.endswith("uncased"))

def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]

def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]

def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'
mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]


# generate skeleton
user2skeleton = defaultdict(list)
item2skeleton = defaultdict(list)
for d in data.train:  
    user, business = d[0], d[1]
    tokens = d[2]
    fa2pos = d[5]
    '''for fa in fa2pos:
        pos = set(fa2pos[fa])
        for i in pos:
            if i < len(tokens):
                token = tokens[i] # original token
                tokens[i] = tokenizer.vocab["[MASK]"]
    '''
    k = min(int(len(tokens) * 0.2), len(fa2pos)) # 20% to mask
    fas = random.sample(list(fa2pos.keys()), k=k) # sample w/o replacement
    for fa in fas:
        pos = set(fa2pos[fa])
        for i in pos:
            if i < len(tokens):
                token = tokens[i] # original token
                tokens[i] = tokenizer.vocab["[MASK]"]

    user2skeleton[user].append(tokens)
    item2skeleton[business].append(tokens)
    
    
fa2tokids=data.fa2tokids
K = args.K


### test
n_samples = 5
batch_size = 1
max_len = 20
top_k = 100
temperature = 1.0
generation_mode = "parallel-sequential"
leed_out_len = 5 # max_len
burnin = 250
sample = True
max_iter = 500

def get_init_text_ids(seed_text, max_len, batch_size = 1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    
    seed_text = [cls_id] + seed_text + [sep_id] 
    input_mask = [1] * len(seed_text)    
    # Zero-pad up to the sequence length.
    while len(seed_text) < max_len:
        seed_text.append(0)
        input_mask.append(0)
        
    batch = []
    batch_mask = []
    for _ in range(batch_size):
        batch.append(seed_text)
        batch_mask.append(input_mask)
    
    return batch, batch_mask


def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
    """ Generate a word from from out[gen_idx]
    
    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k 
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx
  
# make batch input
def get_init_text(seed_text, max_len, batch_size = 1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    #if rand_init:
    #    for ii in range(max_len):
    #        init_idx[seed_len+ii] = np.random.randint(0, len(tokenizer.vocab))
    
    return tokenize_batch(batch)

def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))


# Generation modes as functions
import math
import time

def parallel_sequential_generation(seed_text, positions, fa_ids, fa_mask, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300, burnin=200,
                                   cuda=False, print_every=10, verbose=True):
    """ Generate for one random position at a timestep
        fa_ids, fa_mask - batch_size equals to args batch_size
    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    seed_len = 1
    batch, batch_mask = get_init_text_ids(seed_text, max_len, batch_size)
    
    fa_ids = torch.tensor(fa_ids).cuda() if cuda else torch.tensor(fa_ids)
    fa_mask = torch.tensor(fa_mask).cuda() if cuda else torch.tensor(fa_mask)
    if len(fa_ids.shape) == 1:
        fa_ids = fa_ids.unsqueeze(0)
        fa_mask = fa_mask.unsqueeze(0)
    
    for ii in range(max_iter):
        if positions is not None:
            kk = random.sample(positions, k=1)[0]
        else:
            kk = np.random.randint(0, len(seed_text))
        for jj in range(batch_size):
            batch[jj][seed_len+kk] = mask_id # add 1 to skip cls
        #print("***")
        #print(kk)
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        inp_mask = torch.tensor(batch_mask).cuda() if cuda else torch.tensor(batch_mask)

        sequence_output, pooled_output = model.bert(inp, attention_mask=inp_mask, output_all_encoded_layers=False)
        fa_emb = model.bert.embeddings.word_embeddings(fa_ids)

        _fa_mask = fa_mask.unsqueeze(dim=2).repeat(1,1,fa_emb.size(-1)).float()
        fa_emb_mask = fa_emb * _fa_mask 
        out = model.cls(sequence_output, fa_emb_mask)
        
        topk = top_k if (ii >= burnin) else 0
        idxs = generate_step(out, gen_idx=1+kk, top_k=topk, temperature=temperature, sample=(ii < burnin))
        
        for jj in range(batch_size):
            if type(idxs) == list:
                batch[jj][seed_len+kk] = idxs[jj]
            else:
                batch[jj][seed_len+kk] = idxs
            
        if verbose and np.mod(ii+1, print_every) == 0:
            for_print = tokenizer.convert_ids_to_tokens(batch[0])
            for_print = for_print[:seed_len+kk+1] + ['(*)'] + for_print[seed_len+kk+1:]
            print("iter", ii+1, " ".join(for_print))
            
    return untokenize_batch(batch)

def parallel_generation(seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300, sample=True, 
                        cuda=False, print_every=10, verbose=True):
    """ Generate for all positions at each time step """
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)
    
    for ii in range(max_iter):
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        for kk in range(max_len):
            idxs = generate_step(out, gen_idx=seed_len+kk, top_k=top_k, temperature=temperature, sample=sample)
            for jj in range(batch_size):
                batch[jj][seed_len+kk] = idxs[jj]
            
        if verbose and np.mod(ii, print_every) == 0:
            print("iter", ii+1, " ".join(tokenizer.convert_ids_to_tokens(batch[0])))
    
    return untokenize_batch(batch)
            
def sequential_generation(seed_text, batch_size=10, max_len=15, leed_out_len=15, 
                          top_k=0, temperature=None, sample=True, cuda=False):
    """ Generate one word at a time, in L->R order """
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)
    
    for ii in range(max_len):
        inp = [sent[:seed_len+ii+leed_out_len]+[sep_id] for sent in batch]
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        idxs = generate_step(out, gen_idx=seed_len+ii, top_k=top_k, temperature=temperature, sample=sample)
        for jj in range(batch_size):
            batch[jj][seed_len+ii] = idxs[jj]
        
    return untokenize_batch(batch)


def generate(n_samples, seed_text="[CLS]", batch_size=10, max_len=25, 
             generation_mode="parallel-sequential",
             sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500,
             cuda=False, print_every=1):
    # main generation function to call
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    for batch_n in range(n_batches):
        if generation_mode == "parallel-sequential":
            batch = parallel_sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k,
                                                   temperature=temperature, burnin=burnin, max_iter=max_iter, 
                                                   cuda=cuda, verbose=True)
        elif generation_mode == "sequential":
            batch = sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k, 
                                          temperature=temperature, leed_out_len=leed_out_len, sample=sample,
                                          cuda=cuda)
        elif generation_mode == "parallel":
            batch = parallel_generation(seed_text, batch_size=batch_size,
                                        max_len=max_len, top_k=top_k, temperature=temperature, 
                                        sample=sample, max_iter=max_iter, 
                                        cuda=cuda, verbose=False)
        
        if (batch_n + 1) % print_every == 0:
            print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
            start_time = time.time()
        
        sentences += batch
    return sentences


f1 = open('metrics/ref-raw.txt', 'w')
f2 = open('metrics/tst-raw.txt', 'w')

for item in tqdm(range(len(data.test))):
    ref = data.test[item][4]
    f1.write(" ".join(ref) + '\n')
    
    i = data.test[item][1]
    skeletons = item2skeleton[i] # random select one from all justifications write about that item
    if len(skeletons) == 0: # item has not appear in train set
        u = data.test[item][0]
        skeletons = user2skeleton[u] 
        
    tokens_a = random.sample(skeletons, k=1)[0]
    tokens_b = None
    #is_next_label = None

    fa2pos = data.test[item][-1] # fa position - decide which words are target words 
    fa_ids = data.test[item][3]
    _fa_ids = [idx for fa in fa_ids for idx in fa2tokids[str(fa)] ] # flatten subwords!!!
    fa_ids = _fa_ids[:K] 
    fa_mask = [1 for _ in range(len(fa_ids))]
    while len(fa_ids) < K: # pad
        fa_ids.append(PAD_token) # add [PAD]
        fa_mask.append(0)
    # print(tokenizer.convert_ids_to_tokens(fa_ids))
    seed_text = tokens_a    
    # seed_text = tokens_a
    positions = [idx for idx, tok in enumerate(seed_text) if tok == 103 ]
    # test 
    top_k = 5
    ret = parallel_sequential_generation(seed_text, None, fa_ids, fa_mask, batch_size=1, max_len=max_len, top_k=top_k,
                                   temperature=0.5, burnin=len(seed_text), max_iter=3*len(seed_text), 
                                   cuda=cuda, verbose=True, print_every=10)
    f2.write(" ".join(ret[0]) + '\n')

f1.close()
f2.close()
    
