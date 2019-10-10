# raise ValueError("deal with Variable requires_grad, and .cuda()")
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence

import itertools
import random
import math
import sys
import os
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_token, EOS_token, PAD_token
from model import *#EncoderRNN, LuongAttnDecoderRNN, Attn
from config import MAX_LENGTH, USE_CUDA, teacher_forcing_ratio, save_dir
from util import *
import time
from masked_cross_entropy import *
cudnn.benchmark = True

import datetime

#############################################
# Training
#############################################

def maskNLLLoss(input, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(input, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.cuda() if USE_CUDA else loss
    return loss, nTotal.data[0]

def train(user_input_variable, business_input_variable, user_lengths, business_lengths,target_variable, mask, 
        max_target_len, encoderU, encoderB, decoder, embedding, encoderU_optimizer, encoderB_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH):

    encoderU_optimizer.zero_grad()
    encoderB_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    if USE_CUDA:
        user_input_variable = user_input_variable.cuda()
        business_input_variable = business_input_variable.cuda()
        target_variable = target_variable.cuda()
        mask = mask.cuda()

    loss = 0
    print_losses = []

    outU, hiddenU = encoderU(user_input_variable, user_lengths, None)
    outB, hiddenB = encoderB(business_input_variable, business_lengths, None)

    target_variable = target_variable.transpose(0,1)  ## size (batch_size, seq_len)->(seq_len, batch_size)
    mask = mask.transpose(0,1)
    decoder_input = target_variable[:-1]
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoder_hidden = hiddenU[:decoder.n_layers] + hiddenB[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    '''
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        decoder_output, decoder_hidden, user_decoder_attn, business_decoder_attn = decoder(
            decoder_input, decoder_hidden, outU, outB
        )
        
        mask_loss = masked_cross_entropy(decoder_output, target_variable[1:], mask[1:])
        loss += mask_loss
        print_losses.append(mask_loss.data.item())
    '''
    # Run through decoder one time step at a time
    decoder_output, user_decoder_attn, business_decoder_attn = [], [], []
    for i in range(len(decoder_input)):
        use_teacher_forcing = True if i == 0 or random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            decoder_output_i, decoder_hidden, user_decoder_attn_i, business_decoder_attn_i = decoder(
                decoder_input[i:i+1], decoder_hidden, outU, outB # 1 x B
            )
        else:
            topv, topi = decoder_output_i.data.topk(1)
            topi = topi.squeeze(-1) # 1 x B
            decoder_output_i, decoder_hidden, user_decoder_attn_i, business_decoder_attn_i = decoder(
                topi, decoder_hidden, outU, outB
            )
                   
        decoder_output.append(decoder_output_i)
        user_decoder_attn.append(user_decoder_attn_i)
        business_decoder_attn.append(business_decoder_attn_i)
    decoder_output = torch.stack(decoder_output)
    user_decoder_attn = torch.stack(user_decoder_attn)
    business_decoder_attn = torch.stack(business_decoder_attn)
    mask_loss = masked_cross_entropy(decoder_output, target_variable[1:], mask[1:])
    loss += mask_loss
    print_losses.append(mask_loss.data.item())

    loss.backward()

    clip = 5.0
    ecU = torch.nn.utils.clip_grad_norm_(encoderU.parameters(), clip)
    ecB = torch.nn.utils.clip_grad_norm_(encoderB.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoderU_optimizer.step()
    encoderB_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) 

def evaluate(user_input_variable, business_input_variable, user_lengths, business_lengths, target_variable, mask, max_target_len, encoderU, encoderB,
             decoder, embedding, encoderU_optimizer, encoderB_optimizer, decoder_optimizer, max_length=MAX_LENGTH):

    encoderU.eval()
    encoderB.eval()
    decoder.eval()

    if USE_CUDA:
        user_input_variable = user_input_variable.cuda()
        business_input_variable = business_input_variable.cuda()
        target_variable = target_variable.cuda()
        mask = mask.cuda()

    loss = 0
    print_losses = []

    outU, hiddenU = encoderU(user_input_variable, user_lengths, None)
    outB, hiddenB = encoderB(business_input_variable, business_lengths, None)

    target_variable = target_variable.transpose(0,1)  ## size (batch_size, seq_len)->(seq_len, batch_size)
    mask = mask.transpose(0,1)
    decoder_input = target_variable[:-1]
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoder_hidden = hiddenU[:decoder.n_layers] + hiddenB[:decoder.n_layers]

    '''
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        decoder_output, decoder_hidden, user_decoder_attn, business_decoder_attn = decoder(
            decoder_input, decoder_hidden, outU, outB
        )
        
        mask_loss = masked_cross_entropy(decoder_output, target_variable[1:], mask[1:])
        loss += mask_loss
        print_losses.append(mask_loss.data.item())
    '''
    # Run through decoder one time step at a time
    decoder_output, user_decoder_attn, business_decoder_attn = [], [], []
    for i in range(len(decoder_input)):
        use_teacher_forcing = True if i == 0 or random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            decoder_output_i, decoder_hidden, user_decoder_attn_i, business_decoder_attn_i = decoder(
                decoder_input[i:i+1], decoder_hidden, outU, outB # 1 x B
            )
        else:
            topv, topi = decoder_output_i.data.topk(1)
            topi = topi.squeeze(-1) # 1 x B
            decoder_output_i, decoder_hidden, user_decoder_attn_i, business_decoder_attn_i = decoder(
                topi, decoder_hidden, outU, outB
            )
                   
        decoder_output.append(decoder_output_i)
        user_decoder_attn.append(user_decoder_attn_i)
        business_decoder_attn.append(business_decoder_attn_i)
    decoder_output = torch.stack(decoder_output)
    user_decoder_attn = torch.stack(user_decoder_attn)
    business_decoder_attn = torch.stack(business_decoder_attn)
    mask_loss = masked_cross_entropy(decoder_output, target_variable[1:], mask[1:])
    loss += mask_loss
    print_losses.append(mask_loss.data.item())
    
    return sum(print_losses) 

def batchify(pairs, user_review, user_length, business_review, business_length, bsz, evaluation=False, train_mask_idx=None, shuffle=False):
    # Work out how cleanly we can divide the dataset into bsz parts.
    if shuffle:
        random.shuffle(pairs)
    nbatch = len(pairs) // bsz
    print("num of batch: ", nbatch)
    data = []
    if train_mask_idx is not None:
        print("Include train mask...")
        for i in range(nbatch):
            data.append(batch2TrainData(user_review, user_length, business_review, business_length, pairs[i * bsz: i * bsz + bsz], evaluation, train_mask_idx[i * bsz: i * bsz + bsz]))
        if len(pairs) % nbatch != 0: # last batch
            data.append(batch2TrainData(user_review, user_length, business_review, business_length, pairs[nbatch * bsz: len(pairs)], evaluation, train_mask_idx[nbatch * bsz: len(pairs)]))
    else:    
        for i in range(nbatch):
            data.append(batch2TrainData(user_review, user_length, business_review, business_length, pairs[i * bsz: i * bsz + bsz], evaluation))
        if len(pairs) % nbatch != 0: # last batch
            data.append(batch2TrainData(user_review, user_length, business_review, business_length, pairs[nbatch * bsz: len(pairs)], evaluation))
    return data

def trainIters(args, corpus, reverse, n_epoch, learning_rate, batch_size, n_layers, hidden_size, 
                print_every, loadFilename=None, attn_model='dot', decoder_learning_ratio=1.0):

    print(args)

    currentDT = datetime.datetime.now()
    directory = os.path.join(save_dir, args.corpus, 'model', '{}_{}_{}'.format(n_layers, hidden_size, currentDT.strftime('%Y-%m-%d-%H:%M:%S')))
    print(directory)

    print("corpus: {}, reverse={}, n_epoch={}, learning_rate={}, batch_size={}, n_layers={}, hidden_size={}, decoder_learning_ratio={}".format(corpus, reverse, n_epoch, learning_rate, batch_size, n_layers, hidden_size, decoder_learning_ratio))

    data, length = loadPrepareData(args)
    print('load data...')
    
    print(len(data.train))
    print(len(data.dev))
    print(len(data.test))
    exit(0)


    user_length, item_length = length #, user_length2, item_length2 = length
    train_batches = batchify(data.train, data.user_text, user_length, data.item_text, item_length, batch_size, train_mask_idx=data.train_mask_idx, shuffle=True)
    val_batches = batchify(data.dev, data.user_text, user_length, data.item_text, item_length, batch_size)
    test_batches = batchify(data.test, data.user_text, user_length, data.item_text, item_length, batch_size)
    
    # model
    checkpoint = None 
    print('Building encoder and decoder ...')
    embedding = nn.Embedding(data.voc.n_words, hidden_size)
    encoderU = EncoderRNNlinear(data.voc.n_words, hidden_size, embedding, data.dmax, n_layers, args.encoder_dropout)
    encoderB = EncoderRNNlinear(data.voc.n_words, hidden_size, embedding, data.dmax, n_layers, args.encoder_dropout)

    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, data.voc.n_words, n_layers, args.decoder_dropout)
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoderU.load_state_dict(checkpoint['enU'])
        encoderB.load_state_dict(checkpoint['enB'])
        decoder.load_state_dict(checkpoint['de'])
    # use cuda
    if USE_CUDA:
        encoderU = encoderU.cuda()
        encoderB = encoderB.cuda()
        decoder = decoder.cuda()

    # optimizer
    print('Building optimizers ...')
    encoderU_optimizer = optim.Adam(encoderU.parameters(), lr=learning_rate)
    encoderB_optimizer = optim.Adam(encoderB.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoderU_optimizer.load_state_dict(checkpoint['enU_opt'])
        encoderB_optimizer.load_state_dict(checkpoint['enB_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])

    # initialize
    print('Initializing ...')
    start_epoch = 0
    perplexity = []
    best_val_loss = None
    print_loss = 0
    if loadFilename:
        start_epoch = checkpoint['epoch'] + 1
        perplexity = checkpoint['plt']

    for epoch in range(start_epoch, n_epoch):
        epoch_start_time = time.time()
        # train epoch
        encoderU.train()
        encoderB.train()
        decoder.train()

        print_loss = 0
        start_time = time.time()
        for batch, training_batch in enumerate(train_batches):
            input_variable, lengths, target_variable, mask, max_target_len = training_batch
            user_input_variable, business_input_variable = input_variable
            user_lengths, business_lengths = lengths
            if batch+5 % 1000 == 5:
                print("user_lengths: ", user_lengths)

            loss = train(user_input_variable, business_input_variable, user_lengths, business_lengths,target_variable, mask, 
                        max_target_len, encoderU, encoderB, decoder, embedding, encoderU_optimizer, encoderB_optimizer, decoder_optimizer, batch_size)
            print_loss += loss
            perplexity.append(loss)
            #print("batch {} loss={}".format(batch, loss))
            if batch % print_every == 0 and batch > 0:
                cur_loss = print_loss / print_every
                elapsed = time.time() - start_time

                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_batches), learning_rate,
                        elapsed * 1000 / print_every, cur_loss, math.exp(cur_loss)))

                print_loss = 0
                start_time = time.time()
        
        # evaluate
        val_loss = 0
        for val_batch in val_batches:
            input_variable, lengths, target_variable, mask, max_target_len = val_batch
            user_input_variable, business_input_variable = input_variable
            user_lengths, business_lengths = lengths

            loss = evaluate(user_input_variable, business_input_variable, user_lengths, business_lengths, target_variable, mask, max_target_len, encoderU, encoderB,
                             decoder, embedding, encoderU_optimizer, encoderB_optimizer, decoder_optimizer, batch_size)
            val_loss += loss
        val_loss /= len(val_batches)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save({
                'epoch': epoch,
                'enU': encoderU.state_dict(),
                'enB': encoderB.state_dict(),
                'de': decoder.state_dict(),
                'enU_opt': encoderU_optimizer.state_dict(),
                'enB_opt': encoderB_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(epoch, filename(reverse, 'expansion_model'))))
            best_val_loss = val_loss
     
            # Run on test data.
            test_loss = 0
            for test_batch in test_batches:
                input_variable, lengths, target_variable, mask, max_target_len = test_batch
                user_input_variable, business_input_variable = input_variable
                user_lengths, business_lengths = lengths

                loss = evaluate(user_input_variable, business_input_variable, user_lengths, business_lengths, target_variable, mask, max_target_len, encoderU, encoderB,
                          decoder, embedding, encoderU_optimizer, encoderB_optimizer, decoder_optimizer, batch_size)
                test_loss += loss
            test_loss /= len(test_batches)
            print('-' * 89)
            print('| test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
            print('-' * 89)

        if val_loss > best_val_loss: # early stop
            break
