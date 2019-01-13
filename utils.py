import codecs
import collections
import os
import pickle

import numpy as np


class TextLoader(object):
    def __init__(self, data_dir, batch_size, seq_length):
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not os.path.exists(tensor_file):
            print("reading text file")
            self.preprocess(input_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, tensor_file):
        with codecs.open(input_file, "r") as f:
            data = f.read()
        self.tensor = np.array(list(map(ord, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, tensor_file):
        self.tensor = np.load(tensor_file)
        self.num_batches = self.tensor.size // (self.batch_size * self.seq_length)

    def create_batches(self):
        self.num_batches = self.tensor.size // (self.batch_size * self.seq_length)
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

        validation_batches = int(self.num_batches * .2)
        self.val_batches = zip(self.x_batches[-validation_batches:], self.y_batches[-validation_batches:])
        self.x_batches = self.x_batches[:-validation_batches]
        self.y_batches = self.y_batches[:-validation_batches]
        self.num_batches -= validation_batches

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0


def visualize_result():
    import pandas as pd
    import matplotlib.pyplot as plt

    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    files = [('GridGRU, 3 layers', 'save_gridgru3layers/log.csv'),
             # ('GridGRU, 6 layers', 'save_gridgru6layers/log.csv'),
             ('GridLSTM, 3 layers', 'save_gridlstm3layers/log.csv'),
             ('GridLSTM, 6 layers', 'save_gridlstm6layers/log.csv'),
             ('Stacked GRU, 3 layers', 'save_gru3layers/log.csv'),
             # ('Stacked GRU, 6 layers', 'save_gru6layers/log.csv'),
             ('Stacked LSTM, 3 layers', 'save_lstm3layers/log.csv'),
             ('Stacked LSTM, 6 layers', 'save_lstm6layers/log.csv'),
             ('Stacked RNN, 3 layers', 'save_rnn3layers/log.csv'),
             ('Stacked RNN, 6 layers', 'save_rnn6layers/log.csv')]
    for i, (k, v) in enumerate(files):
        train_loss = pd.read_csv('./save/tinyshakespeare/{}'.format(v)).groupby('epoch').mean()['train_loss']
        plt.plot(train_loss.index.tolist(), train_loss.tolist(), label=k, lw=2, color=tableau20[i * 2])
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Average training loss')
    plt.show()
