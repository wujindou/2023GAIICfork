import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import csv

from utils import to_device, Checkpoint, Step, Smoother, Logger, EMA, FGM
from models import TranslationModel
from dataset import TranslationDataset
from config import Config
from losses import CE

from evaluate import CiderD

def compute_batch(model, source, targets, verbose = False, optional_ret = []):
    source = to_device(source, 'cuda:0')
    targets = to_device(targets, 'cuda:0')
    losses = {}
    pred = model(source[:, :conf['input_l']], targets[:, :conf['output_l']])
    losses['loss_g'] = CE(pred[:, :-1], targets[:, 1:])
    return losses, pred

def array2str(arr):
    out = ''
    for i in range(len(arr)):
        if arr[i]==conf['pad_id'] or arr[i]==conf['eos_id']:
            break
        if arr[i]==conf['sos_id']:
            continue
        out += str(int(arr[i])) + ' '
    if len(out.strip())==0:
        out = '0'
    return out.strip()


def evaluate(model, loader, output_file=None, beam=1, n=-1):
    metrics = Smoother(100)
    res, gts = [], {}
    tot = 0
    for (source, targets) in tqdm(loader):
        if n>0 and tot>n:
            break
        source = to_device(source, 'cuda:0')
        pred = model(source, beam=beam)
        pred = pred.cpu().numpy()
        #print(pred.shape)
        for i in range(pred.shape[0]):
            res.append({'image_id':tot, 'caption': [array2str(pred[i])]})
            gts[tot] = [array2str(targets[i])]
            tot += 1
    CiderD_scorer = CiderD(df='corpus', sigma=15)
    cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
    metrics.update(cider = cider_score)
    print(metrics.value())
    return metrics


def get_model():
    return TranslationModel(conf['input_l'], conf['output_l'], conf['n_token'],
                            encoder_layer=conf['n_layer'], decoder_layer=conf['n_layer'])

start_epoch=0
def train():
    train_data = TranslationDataset(conf['train_file'], conf['input_l'], conf['output_l'])
    valid_data = TranslationDataset(conf['valid_file'], conf['input_l'], conf['output_l'])

    train_loader = DataLoader(train_data, batch_size=conf['batch'], shuffle=True, num_workers=12, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=conf['valid_batch'], shuffle=True, num_workers=12, drop_last=False)

    for epoch in range(start_epoch, conf['n_epoch']):
        for (source, targets) in tqdm(train_loader):
            print(source)
            print(targets)

version = 1
conf = Config(version)
train()
