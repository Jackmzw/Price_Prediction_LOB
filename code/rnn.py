# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:25:27 2018

@author: mzw06
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import cohen_kappa_score, classification_report

import models
import data

parser = argparse.ArgumentParser(description='LOB RNN/LSTM Model: Main Function')
parser.add_argument('--data', type=str, default='../data',
                    help='location of market data')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--symbol', type=str, default='a',
                    help='symbol of asset (a, b)')
parser.add_argument('--delta', type=int, default=1,
                    help='prediction horizon by event')
parser.add_argument('--alpha', type=float, default=0,
                    help='threshold for stationary')
parser.add_argument('--ninp', type=int, default=4,
                    help='size of input vector')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer')
parser.add_argument('--ntag', type=int, default=3,
                    help='size of target')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--decay', type=float, default=1.5,
                    help='learning rate decay')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
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
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='rnn_model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
valid_batch = data.batch_index(np.arange(args.bptt-1, len(valid_data[1])), args.bsz)
test_batch = data.batch_index(np.arange(args.bptt-1, len(test_data[1])), args.bsz)

classes = ['Downward', 'Stationary', 'Upward']

###############################################################################
# Build the model
###############################################################################

model = models.RNNModel(args.model, args.ninp, args.ntag, args.nhid, args.nlayers, args.dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, source_batch, i):
    """Construct the input and target data of the model, with batch. """
    data = torch.zeros(args.bptt, args.bsz, args.ninp)
    target = torch.zeros(args.bsz, dtype=torch.long)
    batch_index = source_batch[i]
    for j in range(args.bsz):
        data[:, j, :] = torch.from_numpy(source[0][batch_index[j] - args.bptt + 1: batch_index[j] + 1]).float()
        target[j] = int(source[1][batch_index[j]])
    return data.to(device), target.to(device)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    y_true = []  # true labels
    y_pred = []  # predicted labels
    start_time = time.time()
    hidden = model.init_hidden(args.bsz)
    for batch, i in enumerate(range(len(train_batch))):

        data, targets = get_batch(train_data, train_batch, i)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output[-1], targets)
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.data

        _, predicted = torch.max(output[-1], 1)
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
    hidden = model.init_hidden(args.bsz)
    for i in range(len(source_batch)):

        data, targets = get_batch(source, source_batch, i)
        output, hidden = model(data, hidden)
        total_loss += len(targets) * criterion(output[-1], targets).data
        _, predicted = torch.max(output[-1], 1)
        y_true.extend(targets.tolist())
        y_pred.extend(predicted.tolist())
        hidden = repackage_hidden(hidden)
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

sns.set()
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
