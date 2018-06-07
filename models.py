# -*- coding: utf-8 -*-
"""
Created on Wed May 30 13:41:15 2018

@author: mzw06
"""
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container a recurrent module."""

    def __init__(self, rnn_type, ninp, ntag, nhid, nlayers, dropout=0):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntag)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class CNNModel(nn.Module):
    """Container of a CNN module. """
    
    def __init__(self, activation=F.relu, num_classes=3, dropout=0.5):
        super(CNNModel, self).__init__()
        self.activation = activation
        # convolution layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=4)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv1d(8, 8, kernel_size=4)
        self.bn2 = nn.BatchNorm1d(8)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=4)
        self.bn3 = nn.BatchNorm1d(16)
        # fully connected layers
        self.fc1 = nn.Linear(16*22, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, num_classes)
        # max pooling
        self.pool = nn.MaxPool1d(2, 2)
        # dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.activation(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.bn2(self.conv2(x))
        x = self.pool(self.activation(x))
        x = self.bn3(self.conv3(x))
        x = self.pool(self.activation(x))
        x = x.view(x.size(0), -1)
        x = self.bn5(self.fc1(x))
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
