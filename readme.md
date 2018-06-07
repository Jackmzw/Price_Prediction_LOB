# Price Trend Prediction Using Deep Learning

This software implements the Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) to predict the price movement using high frequency limit order data. For details, please refer to my report [Price Trend Prediction Using Deep Learning](./report.pdf).

## Requirements
- The program is written in Python, and uses [pytorch](http://pytorch.org/), [scikit-learn](http://scikit-learn.org/stable/index.html), [pandas](https://pandas.pydata.org/) and [numpy](http://www.numpy.org/).
- If necessary, run `pip install -r requirements.txt`.
- A GPU is not necessary, but can provide a significant speed up especially for training a new model.

## Usage

### RNN model

This example trains a multi-layer RNN (Basic or LSTM) on a price movement prediction task.

The code is tested under Windows 10 Anaconda 3 and reproducible.

```bash
python rnn.py         # Train a two layers LSTM on asset a’s LOB data, running default epoch of 50
python rnn.py --model=RNN_TANH  # Train a two layers basic RNN model with tanh activation function
python rnn.py --symbol=b --epochs=30    # Train a two layers LSTM on asset b’s LOB data, running epoch of 30
```

After training, it will print out the performance measures in test set, as well as the plots of loss and kappa in both train and valid set after each epoch.

During training, if a keyboard interrupt (Ctrl-C) is received, training is stopped and the current model is evaluated against the test dataset.

The `rnn.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the market data
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --symbol SYMBOL    symbol of asset (a, b)
  --nhid NHID        number of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  —-decay DECAY      learning rate dacay
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --bsz BSZ          batch size
  --bptt BPTT        sequence length
  --nsample NSAMPLE  size of training set after subsample
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --seed SEED        random seed
  --log-interval N   report interval
  --save SAVE        path to save the final model
```

### CNN model

Training CNN model is quiet similar to training a RNN model.

Train model
```bash
python cnn.py         # Train a CNN model on asset a’s LOB data, running default epoch of 50
```

The `cnn.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the market data
  --symbol SYMBOL    symbol of asset (a, b)s
  --lr LR            initial learning rate
  —-decay DECAY      learning rate dacay
  --epochs EPOCHS    upper epoch limit
  --bsz N            batch size
  --bptt BPTT        sequence length
  --nsample NSAMPLE  size of training set after subsample
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --seed SEED        random seed
  --log-interval N   report interval
  --save SAVE        path to save the final model
```
