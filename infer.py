import torchaudio
import torch
import random
import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch import distributions

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

BATCH_SIZE = 256
NUM_EPOCHS = 35
N_MELS     = 40
IN_SIZE = 40
HIDDEN_SIZE = 128
KERNEL_SIZE = (20, 5)
STRIDE = (8, 2)
GRU_NUM_LAYERS = 2
NUM_DIRS = 2
NUM_CLASSES = 2
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

def sepconv(in_size, out_size, kernel_size, stride=1, dilation=1, padding=0):
    return nn.Sequential(
        torch.nn.Conv1d(in_size, in_size, kernel_size[1], 
                        stride=stride[1], dilation=dilation, groups=in_size,
                        padding=padding),
        
        torch.nn.Conv1d(in_size, out_size, kernel_size=1, 
                        stride=stride[0], groups=int(in_size/kernel_size[0])),
    )

class CRNN(nn.Module):
    def __init__(self, in_size, hidden_size, kernel_size, stride, gru_nl, ):
        super(CRNN, self).__init__()
          
        self.sepconv = sepconv(in_size=in_size, out_size=hidden_size, kernel_size=kernel_size, stride=stride)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=gru_nl, dropout=0.1, bidirectional=True)
        self.init_weights()
        

    def init_weights(self):
        pass

    
    def forward(self, x, hidden):
        x = self.sepconv(x)
        
        # (BS, HS, ?) -> (HS, BS, ?) ->(seq_len, BS, HS)
        x = x.transpose(0, 1).transpose(0, 2)
        
        x, hidden = self.gru(x, hidden)
        # x : (seq_len, BS, HS * num_dirs)
        # hidden : (num_layers * num_dirs, BS, HS)
                        
        return x, hidden

class ApplyAttn(nn.Module):
    def __init__(self, in_size, num_classes):
        super(ApplyAttn, self).__init__()
        self.U = nn.Linear(in_size, num_classes, bias=False)
        
    
    def init_weights(self):
        pass
    
    
    def forward(self, e, data):
        data = data.transpose(0, 1)           # (BS, seq_len, hid_size*num_dirs)
        a = F.softmax(e, dim=-1).unsqueeze(1)
        c = torch.bmm(a, data).squeeze()
        Uc = self.U(c)        
        return F.log_softmax(Uc, dim=-1)

class FullModel(nn.Module):
    def __init__(self, CRNN_model, attn_layer, apply_attn):
        super(FullModel, self).__init__()
        
        self.CRNN_model = CRNN_model
        self.attn_layer = attn_layer
        self.apply_attn = apply_attn

        
    def forward(self, batch, hidden):
        output, hidden = self.CRNN_model(batch, hidden)
        # output: (seq_len, BS, hidden*num_dir)
        
        e = []
        for el in output:
            e_t = self.attn_layer(el)       # -> (BS, 1)
            e.append(e_t)
        e = torch.cat(e, dim=1)        # -> (BS, seq_len)
        
        probs = self.apply_attn(e, output)
        return probs

class AttnMech(nn.Module):
    def __init__(self, lin_size):
        super(AttnMech, self).__init__()
        
        self.Wx_b = nn.Linear(lin_size, lin_size)
        self.Vt   = nn.Linear(lin_size, 1, bias=False)
        
        
    def init_weights(self):
        pass
    
    
    def forward(self, x):
        x = torch.tanh(self.Wx_b(x))
        e = self.Vt(x)
        return e

if __name__ == '__main__':
    CRNN_model = CRNN(IN_SIZE, HIDDEN_SIZE, KERNEL_SIZE, STRIDE, GRU_NUM_LAYERS)
    attn_layer = AttnMech(HIDDEN_SIZE * NUM_DIRS)
    apply_attn = ApplyAttn(HIDDEN_SIZE * 2, NUM_CLASSES)

    full_model = FullModel(CRNN_model, attn_layer, apply_attn)
    print("hi")
    import os
    cws = os.getcwd()
    print(cws)
    while True:
        test_audio = torchaudio.load('0a9f9af7_nohash_0.wav')[0].squeeze().to(device)
        checkpoint = torch.load("checkpoint.pth", map_location=torch.device('cpu'))#['model_state_dict']
        full_model = FullModel(CRNN_model, attn_layer, apply_attn).to(device)

        full_model.load_state_dict(checkpoint['model_state_dict'])
        melspec_test = torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_mels=N_MELS).to(device)
        full_model.eval()
        with torch.no_grad():
            test_audio_mel = torch.log(melspec_test(test_audio) + 1e-9).unsqueeze(0).to(device)


            # define frist hidden with 0
            hidden = torch.zeros(GRU_NUM_LAYERS*2, 1, HIDDEN_SIZE).to(device)    # (num_layers*num_dirs,  BS, HS)
            # run model
            probs = full_model(test_audio_mel, hidden)
            print(probs)
            break
