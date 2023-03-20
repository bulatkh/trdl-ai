import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import scipy

import pytorch_lightning as pl
from torch import nn
import torch



class ReconstructionVis(pl.Callback):
    """
    A callback which logs one or more classifier-specific metrics at the end of each
    validation and test epoch, to all available loggers.
    The available metrics are: accuracy, precision, recall, F1-score.
    """

    def __init__(self, id_to_act=None):
        self.epoch = 0
        self._reset_state()
        self.id_to_act = id_to_act

    def _reset_state(self):
        self.inputs = []
        self.noised = []
        self.reconstructed = []
        self.gt = []
        self.pairwise_loss = []

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.epoch += 1
        self._reset_state()

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._reset_state()

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx) -> None:
        self.inputs = outputs['x']
        self.noised = outputs['noised']
        self.reconstructed = outputs['reconstructed']
        self.gt = outputs['y']
    
    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx, dataloader_idx) -> None:
        self.inputs.extend(outputs['x'].cpu().detach().numpy())
        self.reconstructed.extend(outputs['reconstructed'].cpu().detach().numpy())
        self.gt.extend(outputs['y'].cpu().detach().numpy())
        self.pairwise_loss.extend(outputs['pairwise_loss'].cpu().detach().numpy())

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        idx = random.randint(0, len(self.inputs) - 1)
        inp = self.inputs[idx].cpu().detach().numpy()
        noised = self.noised[idx].cpu().detach().numpy()
        rec = self.reconstructed[idx].cpu().detach().numpy()
        visualize_reconstruction(inp, rec, f'epoch_{self.epoch}', noised, 'results/reconstruction/train')

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        gt = np.array(self.gt)
        pairwise_loss = np.array(self.pairwise_loss)
        visualize_loss_distribution(gt, pairwise_loss, 'loss_dist', save_path='results/reconstruction/test', id_to_act=self.id_to_act)
        
        idxs = np.random.randint(0, len(self.inputs) - 1, 20)
        for i, idx in enumerate(idxs): 
            inp = self.inputs[idx]
            rec = self.reconstructed[idx]
            gt = self.gt[idx]
            visualize_reconstruction(inp, rec, f'test_label_{gt}_{i}', save_path='results/reconstruction/test')


def visualize_loss_distribution(gt, pairwise_loss, name, save_path='results/reconstruction/test', id_to_act=None, threshold_save_path=None):
    os.makedirs(save_path, exist_ok=True)
    unique_act = np.unique(gt)
    if id_to_act is None:
        id_to_act = {}
        for act in unique_act:
            id_to_act[act] = act
    df_list = []
    lowest_other = 1e8
    for act in unique_act:
        idx_act = np.where(gt == act)
        loss_act = pairwise_loss[idx_act]
        conf_int = mean_confidence_interval(loss_act)
        print(f'Mean loss {id_to_act[act]}', *conf_int)
        if id_to_act[act] == 'CYCLING':
            cycling_upper = conf_int[-1]
        else:
            if conf_int[1] < lowest_other:
                lowest_other = conf_int[1]
        if len(df_list) == 0:
            df_list = [(loss, id_to_act[act]) for loss in loss_act] 
        else:
            df_list.extend([(loss, id_to_act[act]) for loss in loss_act])
    df = pd.DataFrame(df_list, columns=['loss', 'label'])
    print('Suggested threshold:', (lowest_other + cycling_upper) / 2)
    sns.displot(data=df, x='loss', hue='label', kind='kde', common_norm=False)
    plt.show()
    plt.savefig(f'{save_path}/{name}.png')
    plt.close()


def visualize_reconstruction(inp, rec, name, nsd=None, save_path='results/reconstruction/train'):
    os.makedirs(save_path, exist_ok=True)
    num_channels = inp.shape[0]
    fig, axes = plt.subplots(num_channels, 1, figsize=(6, 25))
    x = np.arange(0, inp.shape[1])
    for i in range(num_channels):
        ax = axes[i]
        if nsd is None:
            df = pd.DataFrame(data=[x, inp[i],rec[i]]).T
            df.columns = ['x', 'inp', 'rec']
        else:
            df = pd.DataFrame(data=[x, inp[i],rec[i], nsd[i]]).T
            df.columns = ['x', 'inp', 'rec', 'nsd']
        sns.lineplot(data=df, x='x', y='inp', ax=ax, label='input')
        sns.lineplot(data=df,  x='x', y='rec', ax=ax, label='reconstructed')
        if nsd is not None:
            sns.lineplot(data=df,  x='x', y='nsd', ax=ax, label='noised')
    fig.tight_layout()
    plt.show()
    plt.savefig(f'{save_path}/{name}.png')
    plt.close()


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m, 3), round(m-h, 3), round(m+h, 3)