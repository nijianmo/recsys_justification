import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from config import USE_CUDA
from load import PAD_token

class EncoderRNN(nn.Module):  #max-pool
    def __init__(self, input_size, hidden_size, embedding, n_layers=1, dropout=0.3):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)


    def forward(self, x, x_len, hidden=None):
        '''
        :param x1, x2: 
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param x1_len, x2_len:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        # x size (batch_size, dmax, smax)
        # x_len size(batch_size, dmax)
        # sort x
        batch_size, dmax, smax = x.size()
        x = x.view(-1,smax)  ## x size (batch_size*dmax, smax)
        x = x.transpose(0,1)  ## x size (smax, batch_size*dmax)
        x_len = np.array(x_len).reshape(-1)

        x_sort_idx = np.argsort(-np.array(x_len)).tolist()
        x_unsort_idx = np.argsort(x_sort_idx).tolist()
       
        x_len = np.array(x_len)[x_sort_idx].tolist()
        x = x[:, x_sort_idx]
 
        x_emb = self.embedding(x)

        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x_emb, x_len)

        out_p, hidden = self.gru(x_emb_p, hidden) 

        out, out_len = torch.nn.utils.rnn.pad_packed_sequence(out_p)
        # unsort x2
        out = out[:, x_unsort_idx, :] # index batch axis
        # output: (seq_len_1, batch*dmax, hidden*n_dir)
        #hidden size (num_layers * num_directions, batch*dmax, hidden_size)
        out = out[:, :, :self.hidden_size] + out[:, : ,self.hidden_size:] # Sum bidirectional outputs (seq_len_1, batch*dmax, hidden)
        out = out.view(-1, batch_size, dmax, self.hidden_size)
        out,_ = torch.max(out, 2)

        hidden = hidden.view(-1, batch_size, dmax, self.hidden_size)
        hidden,_ = torch.max(hidden, 2)

        # output: (seq_len_1, batch, hidden)
        return out, hidden


class EncoderRNNlinear(nn.Module):  #Linear
    def __init__(self, input_size, hidden_size, embedding, dmax, n_layers=1, dropout=0.3):
        super(EncoderRNNlinear, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.dmax = dmax
        self.fc = nn.Linear(dmax, 1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)       


    def forward(self, x, x_len, hidden=None):
        '''
        :param x1, x2: 
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param x1_len, x2_len:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        # x size (batch_size, dmax, smax)
        # x_len size(batch_size, dmax)
        # sort x
        batch_size, dmax, smax = x.size()
        x = x.view(-1,smax)  ## x size (batch_size*dmax, smax)
        x = x.transpose(0,1)  ## x size (smax, batch_size*dmax)
        x_len = np.array(x_len).reshape(-1)

        x_sort_idx = np.argsort(-np.array(x_len)).tolist()
        x_unsort_idx = np.argsort(x_sort_idx).tolist()
       
        x_len = np.array(x_len)[x_sort_idx].tolist()
        x = x[:, x_sort_idx]
 
        x_emb = self.embedding(x)

        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x_emb, x_len)

        out_p, hidden = self.gru(x_emb_p, hidden) 

        out, out_len = torch.nn.utils.rnn.pad_packed_sequence(out_p, padding_value=PAD_token)

        # unsort x2
        out = out[:, x_unsort_idx, :] # index batch axis
        # output: (seq_len_1, batch*dmax, hidden*n_dir)

        out = self.dropout(out)
        hidden = self.dropout(hidden)

        #hidden size (num_layers * num_directions, batch*dmax, hidden_size)
        out = out[:, :, :self.hidden_size] + out[:, : ,self.hidden_size:] # Sum bidirectional outputs (seq_len_1, batch*dmax, hidden)
        out = out.view(-1, batch_size, dmax, self.hidden_size)
        out = out.transpose(-2,-1)
        out = self.fc(out)
        out = out.squeeze(-1) # only last dim

        hidden = hidden.view(-1, batch_size, dmax, self.hidden_size)
        hidden = hidden.transpose(-2,-1)
        hidden = self.fc(hidden)
        hidden = hidden.squeeze(-1)

        # output: (seq_len_1, batch, hidden)
        return out, hidden

class EncoderRNNaspect(nn.Module):  #max-pool
    def __init__(self, input_size, hidden_size, output_size, embedding, n_layers=1, dropout=0.2):
        super(EncoderRNNaspect, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.fc = nn.Linear(hidden_size, output_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)


    def forward(self, x, x_len, hidden=None):
        '''
        :param x1, x2: 
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param x1_len, x2_len:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        # x size (batch_size, dmax, smax)
        # x_len size(batch_size, dmax)
        # sort x
        batch_size, dmax, smax = x.size()
        x = x.view(-1,smax)  ## x size (batch_size*dmax, smax)
        x = x.transpose(0,1)  ## x size (smax, batch_size*dmax)
        x_len = np.array(x_len).reshape(-1)

        x_sort_idx = np.argsort(-np.array(x_len)).tolist()
        x_unsort_idx = np.argsort(x_sort_idx).tolist()
       
        x_len = np.array(x_len)[x_sort_idx].tolist()
        x = x[:, x_sort_idx]
 
        x_emb = self.embedding(x)

        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x_emb, x_len)

        out_p, hidden = self.gru(x_emb_p, hidden) 

        out, out_len = torch.nn.utils.rnn.pad_packed_sequence(out_p)
        # unsort x2
        out = out[:, x_unsort_idx, :] # index batch axis
        # output: (seq_len_1, batch*dmax, hidden*n_dir)
        #hidden size (num_layers * num_directions, batch*dmax, hidden_size)
        out = out[:, :, :self.hidden_size] + out[:, : ,self.hidden_size:] # Sum bidirectional outputs (seq_len_1, batch*dmax, hidden)
        out = out.view(-1, batch_size, dmax, self.hidden_size)
        out,_ = torch.max(out, 2)
        # output: (seq_len_1, batch, hidden)
        hidden = hidden.view(-1, batch_size, dmax, self.hidden_size)
        hidden,_ = torch.max(hidden, 2)

        ## full connect for aspect classification
        fc_input = torch.max(out,0)
        fc_output = self.fc(fc_input)
        ## fc_output size (batch_size, class_num)
        return out, hidden, fc_output

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
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

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, output_size, n_layers=1, dropout_p=0.2):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        '''
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.data.shape[0], -1) # (1,B,N)
        word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N)->(B,N)
        context = context.squeeze(0)
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        # Return final output, hidden state
        return output, hidden

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.3):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 3, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn_dropout = nn.Dropout(dropout)       
 
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, encoderB_outputs):
        # Note: we run all steps at one pass

        # Get the embedding of all input words
        '''
        :param input_seq:
            word input for all time steps, in shape (N*B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        '''
        
        batch_size = input_seq.size(1) # B
        seq_len = input_seq.size(0) # N
        embedded = self.embedding(input_seq) # [N*B*H]
        embedded = self.embedding_dropout(embedded)
     
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden) 

        rnn_output = self.gru_dropout(rnn_output)
        hidden = self.gru_dropout(hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        # Need to feed in hidden rather than rnn_output
        user_attn_weights = self.attn(rnn_output, encoder_outputs) # [N*B*H] x [T*B*H] -> [B*N*T]
        user_context = user_attn_weights.bmm(encoder_outputs.transpose(0, 1)) # [B*N*T] x [B*T*H] -> [B*N*H]
        
        business_attn_weights = self.attn(rnn_output, encoderB_outputs) # [N*B*H] x [T*B*H] -> [B*N*T]
        business_context = business_attn_weights.bmm(encoderB_outputs.transpose(0, 1)) # [B*N*T] x [B*T*H] -> [B*N*H]
        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        # rnn_output = rnn_output.squeeze(0) # [1*B*H] -> [B*H]
        # context = context.squeeze(1)       # [B*1*H] -> [B*H]
        user_context = user_context.transpose(0, 1)        # [B*N*H] -> [N*B*H]
        business_context = business_context.transpose(0, 1)
        #context = self.attn_dropout(context)
        concat_input = torch.cat((rnn_output, user_context,business_context), 2) # [N*B*H] + [N*B*H] -> [N*B*2H]
        concat_output = F.tanh(self.concat(concat_input)) 
        
        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        #output = F.softmax(output) # masked_cross_entroy already uses softmax
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, user_attn_weights, business_attn_weights

