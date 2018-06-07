#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:03:04 2018

@author: Jack
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import cohen_kappa_score, classification_report

import models
import data

parser = argparse.ArgumentParser(description='LOB CNN Model: Main Function')
parser.add_argument('--data', type=str, default='./data',
                    help='location of the market data')
parser.add_argument('--symbol', type=str, default='a',
                    help='symbol of asset (a, b)')
parser.add_argument('--delta', type=int, default=1,
                    help='horizon of price change')
parser.add_argument('--alpha', type=float, default=0,
                    help='threshold for stationary')
parser.add_argument('--ninp', type=int, default=4,
                    help='size of input vector')
parser.add_argument('--ntag', type=int, default=3,
                    help='size of target')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--decay', type=float, default=1.5,
                    help='learning rate decay')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--bsz', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=100,
                    help='sequence length')
parser.add_argument('--nsample', type=int, default=150000,
                    help='size of training set after subsample')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=2222,
                    help='random seed')
parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='cnn_model.pt',
                    help='path to save the final model')
args = parser.parse_args()


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

###############################################################################
# Load data
###############################################################################

dataset = data.Market(args.data, symbol=args.symbol, delta=args.delta, alpha=args.alpha, scaler=MinMaxScaler)
train_data = dataset.train
valid_data = dataset.valid
test_data = dataset.test

# Build a matrix of size num_batch * args.bsz containing the index of observation.
np.random.seed(args.seed)
index = data.subsample_index(train_data[1], args.bptt, args.nsample)
train_batch = data.batch_index(index, args.bsz) 
valid_batch = data.batch_index(np.arange(args.bptt - 1, len(valid_data[1])), args.bsz)  
test_batch = data.batch_index(np.arange(args.bptt - 1, len(test_data[1])), args.bsz)

classes = ['Downward', 'Stationary', 'Upward']

###############################################################################
# Build the model
###############################################################################

model = models.CNNModel(activation=F.relu, num_classes=args.ntag, dropout=args.dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
model.to(device)

###############################################################################
# Training code
###############################################################################

def get_batch(source, source_batch, i):
    """Construct the input  and target data of the model, with batch. """
    data = torch.zeros(args.bsz, 1, args.bptt, args.ninp)
    target = torch.zeros(args.bsz, dtype=torch.long)
    batch_index = source_batch[i]
    for j in range(args.bsz):
        data[j, 0, :, :] = torch.from_numpy(source[0][batch_index[j] - args.bptt + 1: batch_index[j] + 1]).float()
        target[j] = int(source[1][batch_index[j]])
    return data.to(device), target.to(device)
    
def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    y_true = []  # true labels
    y_pred = []  # predicted labels
    start_time = time.time()
    for batch, i in enumerate(range(len(train_batch))):
        data, targets = get_batch(train_data, train_batch, i)  
        model.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.data

        _, predicted = torch.max(outputs, 1)
        y_true.extend(targets.tolist())
        y_pred.extend(predicted.tolist())        
        
        if (batch + 1) % args.log_interval == 0:
            cur_loss = total_loss.item() / (batch + 1)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                epoch, batch + 1, len(train_batch), lr,
                elapsed * 1000 / args.log_interval, cur_loss))
            start_time = time.time()
    # compute Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    return total_loss.item() / (batch + 1), kappa
            
def evaluate(source, source_batch):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    y_true = []  # true labels
    y_pred = []  # predicted labels
    for i in range(len(source_batch)):      
        data, targets = get_batch(source, source_batch, i)
        outputs = model(data)
        total_loss += len(targets) * criterion(outputs, targets).data
        _, predicted = torch.max(outputs, 1)
        y_true.extend(targets.tolist())
        y_pred.extend(predicted.tolist())
    val_loss = total_loss.item() / np.size(source_batch) 
    # Make report for the classfier
    report = classification_report(y_true, y_pred, target_names=classes)
    kappa = cohen_kappa_score(y_true, y_pred)
    return val_loss, kappa, report

# Loop over epochs
epochs = args.epochs
lr = args.lr
best_val_loss = None
Loss = np.zeros((epochs + 1, 2))
Kappa = np.zeros((epochs + 1, 2))
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        tra_loss, tra_kappa = train()
        val_loss, val_kappa, _ = evaluate(valid_data, valid_batch)
        Loss[epoch] = [tra_loss, val_loss]
        Kappa[epoch] = [tra_kappa, val_kappa]
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'
              .format(epoch, (time.time() - epoch_start_time), val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open('rnn_model.pt', 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= args.decay
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open('rnn_model.pt', 'rb') as f:
    model = torch.load(f)    
    
# Run on test data
test_loss, kappa, report = evaluate(test_data, test_batch)
print('=' * 89)
print('| End of training | test loss {:5.2f} '.format(test_loss))
print(report)
print('Cohen Kappa Score: {:.2f}'.format(kappa))
print('=' * 89)          
 
# Plot the loss of train and valid set after each epoch
plt.plot(Loss[1:])
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(('Train', 'Valid'))
plt.show()
# Plot the Kappa of train and valid set after each epoch    
plt.plot(Kappa[1:])
plt.xlabel('epoch')
plt.ylabel('kappa')
plt.legend(('Train', 'Valid'))
plt.show()  






