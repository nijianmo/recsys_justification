import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

class Attn(nn.Module):
    def __init__(self, method='dot', hidden_size=None):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        # end of update

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (N,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,N,T)
        '''
        max_len = encoder_outputs.size(0) # T
        seq_len = hidden.size(0) # N
        this_batch_size = encoder_outputs.size(1)
        
        H = hidden.repeat(max_len,1,1,1) # [T*N*B*H]
        encoder_outputs = encoder_outputs.repeat(seq_len,1,1,1).transpose(0,1) # [N*T*B*H] -> [T*N*B*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score [B*N*T]
        return F.softmax(attn_energies.view(-1,max_len)).view(this_batch_size,-1,max_len) # normalize with softmax on T axis

    def score(self, hidden, encoder_outputs):
        energy = self.attn(torch.cat([hidden, encoder_outputs], 3)) # [T*N*B*2H]->[T*N*B*H]
        energy = energy.view(-1, self.hidden_size) # [T*N*B,H]
        v = self.v.unsqueeze(1) #[1*H]
        #print(energy.size())
        #print(v.size())
        energy = energy.mm(v) # [T*N*B,H] * [H,1] -> [T*N*B,1]
        att_energies = energy.view(-1,hidden.size(1),hidden.size(2)) # [T*N*B] 
        att_energies = att_energies.transpose(0, 2).contiguous() # [B*N*T]
        return att_energies

