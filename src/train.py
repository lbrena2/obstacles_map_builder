#!/usr/bin/python

"""Train the neural network model using the given training set."""

import os
import torch
import argparse
import numpy as np
import pandas as pd
from torch import optim
from utils import RangeUnion
from tqdm import tqdm, trange
from datetime import datetime
import matplotlib.pyplot as plt
from pytorchutils import MaskedLoss
from torch.utils.data import DataLoader, Subset
from dataset import get_dataset, get_dataset_ranges

# todo: to be fixed


def train():
    """Train the neural network model, save weights and plot loss over time."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='name of the model',
                        default=f'model_{str(datetime.now())}')
    parser.add_argument('-f', '--filename', type=str,
                        help='name of the dataset (.h5 file)', default='/home/mirko/Documents/Data/data_bo.h5')
    parser.add_argument('-e', '--epochs', type=int,
                        help='number of epochs of the training phase', default=600)
    parser.add_argument('-bs', '--batch-size', type=int,
                        help='size of the batches of the training data', default=1024)
    parser.add_argument('-lr', '--learning-rate', type=float,
                        help='learning rate used for the training phase', default=1e-4)
    args = parser.parse_args()

    name = args.name
    filename = args.filename
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    max_patience = 30
    validation_freq = 1
    out_channels = 65 * 5
    model_path = f'model/{name}'
    output_path = f'{model_path}/output'
    checkpoint_path = f'{model_path}/checkpoints'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for k, v in vars(args).items():
        print(f'{k} = "{v}"')
    print(f'device = "{device}"')

    # Dataset

    input_cols = ['bag0_x', 'bag1_x', 'bag2_x', 'bag3_x', 'bag4_x',
                  'bag5_x', 'bag6_x', 'bag7_x', 'bag8_x', 'bag9_x', 'bag10_x']

    target_cols = ['bag0_y', 'bag1_y', 'bag2_y', 'bag3_y', 'bag4_y',
                   'bag5_y', 'bag6_y', 'bag7_y', 'bag8_y', 'bag9_y', 'bag10_y']

    dataset = get_dataset(filename, device=device, augment=True,
                          input_cols=input_cols, target_cols=target_cols)

    bag_index_ranges = get_dataset_ranges(dataset)

    print(f'bagfiles = "{len(bag_index_ranges)}"')

    # note: use only to show dataset ranges
    print(bag_index_ranges)
    exit()

    # todo: below
    train_split = xrange(0, int(n_bagfiles * 0.6))
    val_split = xrange(int(n_bagfiles * 0.6), int(n_bagfiles * 0.8))

    train_ranges = [r for i in train_split for r in bag_index_ranges[i]]
    val_ranges = [r for i in val_split for r in bag_index_ranges[i]]

    train_dataset = Subset(dataset, RangeUnion(train_ranges))
    val_dataset = Subset(dataset, RangeUnion(val_ranges))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NN(in_channels=3, out_channels=len(coords)).to(device)
    model.summary((3, 64, 80), device)

    # Optimizer & Loss

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = MaskedLoss(torch.nn.MSELoss)

    # Training

    history = pd.DataFrame()

    epochs_logger = trange(epochs, desc='epoch')
    for epoch in epochs_logger:
        steps_logger = tqdm(train_loader, desc='step')
        for step, batch in enumerate(steps_logger):
            x, y = batch
            mask = (y > 0).to(y.dtype)

            optimizer.zero_grad()

            preds = model(x)

            loss = loss_function(preds, y, mask)
            loss.backward()
            optimizer.step()

            loss = loss.item()

            if step % validation_freq == 0:
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_x, val_y = val_batch
                        val_mask = (val_y > 0).to(val_y.dtype)

                        val_preds = model(val_x)

                        val_loss = loss_function(
                            val_preds, val_y, val_mask).item()

            history = history.append({
                'loss': loss,
                'val_loss': val_loss,
            }, ignore_index=True)

            mean_values = metrics.query('epoch == ' + str(epoch)).mean(axis=0)
            mean_loss = mean_values['loss']
            mean_val_loss = mean_values['val_loss']

            log_str = 'loss: %.5f, val_loss: %.5f' % (mean_loss, mean_val_loss)
            epochss_logger.set_postfix_str(log_str)
            steps_logger.set_postfix_str(log_str)

        checkpoint_name = '%d_%.5f_state_dict.pth' % (epoch, mean_val_loss)
        torch.save(model.state_dict(), checkpoint_path + '/' + checkpoint_name)

    # keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    def plot_lines(df, columns, colors, ax, alpha=0.25, show_range=False, window_size=1):
        for color, column in zip(colors, columns):
            agg_df = df.groupby('epoch')[column]

            if window_size > 1:
                agg_df = agg_df.rolling(window_size)

            means = agg_df.mean()
            ax.plot(np.arange(len(means)), means, c=color)

            if show_range:
                mins = agg_df.min()
                maxs = agg_df.max()
                ax.fill_between(x=np.arange(len(means)),
                                y1=mins, y2=maxs, alpha=alpha)

        ax.legend(columns)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_lines(history, ['loss', 'val_loss'], ['blue', 'orange'], ax)
    plt.show()


if __name__ == '__main__':
    train()
