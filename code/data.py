# -*- coding: utf-8 -*-
"""
Created on Wed May 30 21:39:43 2018

@author: Jack
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def creat_label(df, delta=1, alpha=0):
    """Given a dataframe, construct the labels for each timestamps. """
    ts = (df['bid_price'] + df['ask_price']) / 2
    pct = (ts.shift(-delta) - ts) / ts  # forward return
    pct = pct.dropna() 
    label = (pct >= -alpha).astype(int) + (pct > alpha).astype(int)
    label = label.values
    value = df.values[:-delta]
    return value, label

def subsample_index(label, bptt, nsample):
    """Subsample the observations to produce a balanced training set, return a list of index. """
    index = [None]*3
    for i in range(3):
        # Get index of category i
        index[i] = np.where(label == i)[0]
        index[i] = index[i][index[i] >= bptt - 1] 
        if i == 1:
            # Under sample the Stationary category
            index[i] = np.random.choice(index[i], size=int(nsample) - 2 * int(nsample / 5), replace=False)
        else:
            # Over sample the Downward and Upward categories
            index[i] = np.random.choice(index[i], size=int(nsample / 5), replace=True)
    subsample = np.concatenate(index)
    subsample = np.random.permutation(subsample)
    return subsample

def batch_index(index, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(index) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    remainder = index[:nbatch * bsz]
    # Evenly divide the data across the bsz batches.
    remainder = remainder.reshape(bsz, -1).T
    return remainder

class Market(object):
    """Create dataset for training, include train and test set. """
    def __init__(self, path, symbol='a', delta=1, alpha=0, scaler=MinMaxScaler):
        self.path = path
        self.symbol = symbol
        self.delta = delta
        self.alpha = alpha
        self.scaler = scaler
        self.raw_data = self.read_data()
        self.train, self.valid, self.test = self.process_data()
    
    def read_data(self):
        data = []
        for i in range(1, 6):
            path = os.path.join(self.path, self.symbol+'_'+str(i)+'.txt')
            assert os.path.exists(path)
            df = pd.read_csv(path, header=None, index_col=0, parse_dates=True,
                             names=['Time', 'bid_price', 'ask_price', 'bid_size', 'ask_size'],
                             dtype = np.float64)
            date = str(df.index[0]).split()[0]            
            df1 = df[date+" 21:20:00": date+" 23:30:00"]
            if len(df1) > 0:
                data.append(df1)
            df2 = df[date+" 09:05:00": date+" 15:00:00"]
            if len(df2) > 0:
                data.append(df2)
        return data
                
    def process_data(self):
        values = []
        labels = []
        n = len(self.raw_data)
        for i in range(n):
            x, y = creat_label(df=self.raw_data[i], delta=self.delta, alpha=self.alpha)
            labels.append(y)
            if i == 0:
                Scaler = self.scaler()
                value = Scaler.fit_transform(x)
                values.append(value)
            else:
                value = Scaler.transform(x)
                Scaler = self.scaler()
                Scaler.fit(x)
                values.append(value)
        train_x = np.concatenate(values[:6], axis=0)
        train_y = np.concatenate(labels[:6])
        valid_x = np.concatenate(values[6:8], axis=0)
        valid_y = np.concatenate(labels[6:8])
        test_x = values[-1]
        test_y = labels[-1]
        return [train_x, train_y], [valid_x, valid_y], [test_x, test_y]
        