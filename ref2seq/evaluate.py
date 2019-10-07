import torch
from torch.autograd import Variable
import random
from model import *
from util import *
from config import USE_CUDA
import sys
import os
from config import MAX_LENGTH, USE_CUDA, teacher_forcing_ratio, save_dir
from masked_cross_entropy import *
import itertools
import random
import math
from tqdm import tqdm
from load import SOS_token, EOS_token, PAD_token, UNK_token
from model import EncoderRNN, LuongAttnDecoderRNN, Attn
import pickle
import logging
logging.basicConfig(level=logging.INFO)
import torch.nn.functional as F
 
class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden # decoder hidden so far
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        return sum(self.sentence_scores) / len(self.sentence_scores)
        # return mean of sentence_score

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topi = topi.squeeze(0)
        topv = topv.squeeze(0)
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size): 
            ni = topi[0][i] 
            ni = int(ni.cpu())
            if ni == EOS_token:
                terminates.append(([voc.index_word[str(idx)] for idx in self.sentence_idxes] + ['<eos>'], 
                                   self.avgScore())) # tuple(word_list, score_float) 
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(ni)
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, ni, idxes, scores))
        return terminates, sentences

    def toWordScore(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<eos>')
            else:
                words.append(voc.index_word[str(self.sentence_idxes[i])])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<eos>')
        return (words, self.avgScore())

def beam_decode(decoder, decoder_hidden, outU, outB, voc, beam_size, max_length=MAX_LENGTH):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for t in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = Variable(torch.LongTensor([[sentence.last_idx]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

            decoder_output, decoder_hidden, user_decoder_attn, business_decoder_attn = decoder(
            decoder_input, sentence.decoder_hidden, outU, outB
        ) # feed sentence decoder hidden
            
            decoder_output = F.softmax(decoder_output, dim=-1)
            topv, topi = decoder_output.data.topk(beam_size)
            term, top = sentence.addTopk(topi, topv, decoder_hidden, beam_size, voc)
            terminal_sentences.extend(term)
            next_top_sentences.extend(top)

        next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
        prev_top_sentences = next_top_sentences[:beam_size]
        next_top_sentences = []

    terminal_sentences += [sentence.toWordScore(voc) for sentence in prev_top_sentences]
    terminal_sentences.sort(key=lambda x: x[1], reverse=True)

    n = min(len(terminal_sentences), 3) # top 3?
    return terminal_sentences[:n]



def beam_decode_batch(decoder, decoder_hidden, outU, outB, voc, beam_size, batch_size, max_length=MAX_LENGTH):
    # these list should include all samples in the batch
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    for i in range(batch_size):
        terminal_sentences.append([])
        next_top_sentences.append([])
        prev_top_sentences.append([Sentence(decoder_hidden[:,i])])
    
    for t in tqdm(range(max_length)):
        beam_sizes = [len(s) for s in prev_top_sentences]
        for j in range(max(beam_sizes)):
            decoder_input = []
            sentence_decoder_hidden = []

            for i in range(batch_size):
                sentence = prev_top_sentences[i][j]
                decoder_input.append(torch.LongTensor([sentence.last_idx]))
                sentence_decoder_hidden.append(sentence.decoder_hidden)

            decoder_input = torch.stack(decoder_input, 1)
            sentence_decoder_hidden = torch.stack(sentence_decoder_hidden, 1)
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input        

            decoder_output, decoder_hidden, user_decoder_attn, business_decoder_attn = decoder(
                decoder_input, sentence_decoder_hidden, outU, outB
            )

            decoder_output = F.softmax(decoder_output, dim=-1)
            topv, topi = decoder_output.data.topk(beam_size)

            for i in range(batch_size):
                sentence = prev_top_sentences[i][j]
                term, top = sentence.addTopk(topi[:,i:i+1], topv[:,i:i+1], decoder_hidden[:,i], beam_size, voc)
                terminal_sentences[i].extend(term)
                next_top_sentences[i].extend(top)

        for i in range(batch_size):
            # after adding all beams, keep beam top
            next_top_sentences[i].sort(key=lambda s: s.avgScore(), reverse=True)
            prev_top_sentences[i] = next_top_sentences[i][:beam_size]
            next_top_sentences[i] = []   

    for i in range(batch_size):
        terminal_sentences[i] += [sentence.toWordScore(voc) for sentence in prev_top_sentences[i]]
        terminal_sentences[i].sort(key=lambda x: x[1], reverse=True)

        n = min(len(terminal_sentences), 3) # keep top 3?
        terminal_sentences[i] = terminal_sentences[i][:n]
    
    return terminal_sentences


def decode(decoder, decoder_hidden, outU, outB, voc, batch_size, max_length=MAX_LENGTH):

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden, user_decoder_attn, business_decoder_attn = decoder(
            decoder_input, decoder_hidden, outU, outB
        )
        topv, topi = decoder_output.data.topk(3)
        # squeeze when batch_size=1?
        topi = topi.squeeze(0)
        topv = topv.squeeze(0)
        ni = topi[0][0] # most possible word
        if ni == EOS_token:
            decoded_words.append('<eos>')
            break
        else:
            decoded_words.append(voc.idx2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    return decoded_words


def evaluate(encoderU, encoderB, decoder, voc, \
             user_input_variable, business_input_variable, \
             user_lengths, business_lengths, target_variable, mask, max_target_len, \
             beam_size, batch_size, max_length=MAX_LENGTH):
    
    if USE_CUDA:
        user_input_variable = user_input_variable.cuda()
        business_input_variable = business_input_variable.cuda()
        target_variable = target_variable.cuda()
        mask = mask.cuda()
    
    outU, hiddenU = encoderU(user_input_variable, user_lengths, None)
    outB, hiddenB = encoderB(business_input_variable, business_lengths, None)
    decoder_hidden = hiddenU[:decoder.n_layers] + hiddenB[:decoder.n_layers]
    
    if beam_size == 1:
        return decode(decoder, decoder_hidden, outU, outB, voc, batch_size)
    else:
        if batch_size == 1:
            return beam_decode(decoder, decoder_hidden, outU, outB, voc, beam_size)
        else:
            return beam_decode_batch(decoder, decoder_hidden, outU, outB, voc, beam_size, batch_size)

def evaluateRandomly(encoderU, encoderB, decoder, voc, \
                     input_index, user_input_variable, business_input_variable, \
                     user_lengths, business_lengths, target_variable, \
                     mask, max_target_len, reverse, beam_size):
    path = "./metrics/"
    batch_size = len(input_index) # batch_size
    
    user_index = [inp[0] for inp in input_index]
    business_index = [inp[1] for inp in input_index]
    
    target_words_list = [[voc.index_word[str(x)] for x in inp] for inp in target_variable.tolist()] # consider each sample in the batch
    for target_words in target_words_list:
        print(" ".join(target_words[1:]))
        
    if beam_size == 1:
        output_words = evaluate(encoderU, encoderB, decoder, voc, \
                                user_input_variable, business_input_variable, \
                                ser_lengths, business_lengths, target_variable, \
                                mask, max_target_len, beam_size, batch_size)
        print(" ".join(output_words[:-1]))
    else:
        output_words_list = evaluate(encoderU, encoderB, decoder, voc, \
                                user_input_variable, business_input_variable, \
                                user_lengths, business_lengths, target_variable, \
                                mask, max_target_len, beam_size, batch_size)
        for output_words in output_words_list:
            # take the top candidate of the beam 
            output_words, score = output_words[0] 
            #f2.write(" ".join(output_words[:-1]) + "\n")
            print(" ".join(output_words[:-1]))
 

# do not need mask since no edu will appear 
def batchify(pairs, user_review, user_length, business_review, business_length, bsz, evaluation=False):
    # Work out how cleanly we can divide the dataset into bsz parts.
    #random.shuffle(pairs)
    nbatch = len(pairs) // bsz
    print("num of batch: ", nbatch)
    data = []
    for i in range(nbatch):
        data.append(batch2TrainData(user_review, user_length, business_review, business_length, pairs[i * bsz: i * bsz + bsz], evaluation))
    if len(pairs) % nbatch != 0: # last batch
        print(len(pairs) - nbatch * bsz)
        data.append(batch2TrainData(user_review, user_length, business_review, business_length, pairs[nbatch * bsz: len(pairs)], evaluation))
    return data

def batch2TrainData(user_review, user_length, business_review, business_length, pair_batch, evaluation=False):
    
    input_batch, output_batch = [], []
    input_index = []
    input_length = []
    for i in range(len(pair_batch)):
        input_index.append((str(pair_batch[i][0]), str(pair_batch[i][1])))
        input_batch.append([user_review[str(pair_batch[i][0])], business_review[str(pair_batch[i][1])]])
        input_length.append([user_length[str(pair_batch[i][0])], business_length[str(pair_batch[i][1])]])
        output_batch.append([SOS_token]+pair_batch[i][2]+[EOS_token])
    review_input, input_length = inputVar(input_batch, input_length, evaluation=evaluation)
    output, mask, max_target_len = outputVar(output_batch) # convert sentence to ids and padding
    return input_index, review_input, input_length, output, mask, max_target_len
    
    
def runTest(args, n_layers, hidden_size, reverse, modelFile, beam_size, batch_size, input, corpus):

    data, length = loadPrepareData(args)
    voc = data.voc
    print('load data...')
    user_length, item_length = length #, user_length2, item_length2 = length
    # train_batches = batchify(data.train, data.user_text, user_length, data.item_text, item_length, batch_size)
    # val_batches = batchify(data.dev, data.user_text, user_length, data.item_text, item_length, batch_size)
    test_batches = batchify(data.test, data.user_text, user_length, data.item_text, item_length, batch_size)  
    
    print('Building encoder and decoder ...')
    
    embedding = nn.Embedding(data.voc.n_words, hidden_size)
    encoderU = EncoderRNNlinear(data.voc.n_words, hidden_size, embedding, data.dmax, n_layers)
    encoderB = EncoderRNNlinear(data.voc.n_words, hidden_size, embedding, data.dmax, n_layers)

    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, data.voc.n_words, n_layers)
    
    # load model 
    checkpoint = torch.load(modelFile)        
    encoderU.load_state_dict(checkpoint['enU'])
    encoderB.load_state_dict(checkpoint['enB'])
    decoder.load_state_dict(checkpoint['de'])
    
    # train mode set to false, effect only on dropout, batchNorm
    encoderU.train(False)
    encoderB.train(False)
    decoder.train(False)

    if USE_CUDA:
        encoderU = encoderU.cuda()
        encoderB = encoderB.cuda()
        decoder = decoder.cuda()    
        
    if not args.sample:
        # evaluate on test
        for test_batch in tqdm(test_batches):

            input_index, input_variable, lengths, target_variable, mask, max_target_len = test_batch
            user_input_variable, business_input_variable = input_variable
            user_lengths, business_lengths = lengths

            # evaluate on train
            evaluateRandomly(encoderU, encoderB, decoder, voc, \
                             input_index, user_input_variable, business_input_variable, \
                             user_lengths, business_lengths, \
                             target_variable, mask, max_target_len, reverse, beam_size)
    else:
        # evaluate using sample 
        sample(encoderU, encoderB, decoder, voc, test_batches, reverse)
        

# top-k sample 
def sample(encoderU, encoderB, decoder, voc, test_batches, reverse, n_words=MAX_LENGTH):
    
    word2idx = voc.word_index
    idx2word = voc.index_word
    START_TOKEN = idx2word[str(SOS_token)]
    STOP_TOKEN = idx2word[str(EOS_token)]
    
    path = "./metrics/"
    f1 = open(path + "ref-new.txt",'w')
    f2 = open(path + "tst-new.txt",'w')
    
    # Here is how to use this function for top-p sampling
    temperature = 1.0
    top_k = 0
    top_p = 0.9

    for test_batch in tqdm(test_batches):
        
        input_index, input_variable, lengths, target_variable, mask, max_target_len = test_batch
        user_input_variable, business_input_variable = input_variable
        user_lengths, business_lengths = lengths

        target_words_list = [[voc.index_word[str(x)] for x in inp] for inp in target_variable.tolist()] # consider each sample in the batch
        for target_words in target_words_list:
            f1.write(" ".join(target_words[1:]) + "\n")
            #print(" ".join(target_words[1:]))
        
        # sample
        if USE_CUDA:
            user_input_variable = user_input_variable.cuda()
            business_input_variable = business_input_variable.cuda()
            target_variable = target_variable.cuda()
            mask = mask.cuda()

        outU, hiddenU = encoderU(user_input_variable, user_lengths, None)
        outB, hiddenB = encoderB(business_input_variable, business_lengths, None)
        decoder_hidden = hiddenU[:decoder.n_layers] + hiddenB[:decoder.n_layers]
        
        batch_size = outU.size(1) # TxBxH
        word_idx = int(word2idx[START_TOKEN])
        decoder_input = torch.rand(1, batch_size).mul(word_idx).long() # 1xB
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
        
        sentences = []
        for i in range(n_words):
            
            decoder_output, decoder_hidden, user_decoder_attn, business_decoder_attn = decoder(
                decoder_input, decoder_hidden, outU, outB
            )
            
            logits = decoder_output.transpose(0, 1) # 1xBxV -> Bx1xV
            logits = logits / temperature
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probabilities = F.softmax(filtered_logits, dim=-1)
            probabilities = probabilities.squeeze(1)
            next_token = torch.multinomial(probabilities, 1)  

            decoder_input = next_token.transpose(0, 1)
            sentences.append(next_token.squeeze().cpu().data)

        _sentences = [[] for _ in range(batch_size)]
        isFinished = [False for _ in range(batch_size)] 
        for sentence in sentences:
            for i, word_idx in enumerate(sentence.tolist()):
                if not isFinished[i]:
                    if word_idx == EOS_token:
                        isFinished[i] = True
                    else:
                        _sentences[i].append(idx2word[str(word_idx)])
        
        
        for _sentence in _sentences:
            sentence = " ".join(_sentence)
            #print(_sentence)
            f2.write(sentence + "\n")
        
    f1.close()
    f2.close()        
        

        
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            logits {torch.Tensor} -- logits (in batch), size B x 1 x vocab_size
            
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:  
        values = torch.topk(logits, top_k)[0] # Bxtop_k
        batch_mins = values[:, :, -1].expand_as(logits.squeeze(1)).unsqueeze(1)
        logits = torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)
        
    if top_p > 0.0 and top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)

        cumprobs = sorted_probs.cumsum(dim=-1)
        # Create mask for all cumulative probabilities less than p
        mask = cumprobs < top_p
        # First mask must always be pickable
        mask = F.pad(mask[:, :, :-1], (1, 0, 0, 0), value=1)

        masked_probs = torch.where(mask, sorted_probs, torch.tensor(float('inf')).to(probs))

        batch_mins = masked_probs.min(dim=-1, keepdim=True)[0].expand_as(logits)

        # Mask out all logits (tail) that are too small
        logits = torch.where(probs < batch_mins, torch.tensor(float('-inf')).to(logits), logits)
    
    return logits

# # Here is how to use this function for top-p sampling
# temperature = 1.0
# top_k = 0
# top_p = 0.9

# # Get logits with a forward pass in our model (input is pre-defined)
# logits = model(input)

# # Keep only the last token predictions, apply a temperature coefficient and filter
# logits = logits[..., -1, :] / temperature
# filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

# # Sample from the filtered distribution
# probabilities = F.softmax(filtered_logits, dim=-1)
# next_token = torch.multinomial(probabilities, 1)        
        
        
