WANDB = False
import wandb
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import csv

from utils import to_device, Checkpoint, Step, Smoother, Logger, EMA, FGM, AWP
from models_bart import CustomBartModel
from dataset import BartDataset
from config_bart import Config
from losses import CE

from transformers import BartConfig, BartForConditionalGeneration

from evaluate import CiderD

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


def evaluate(model, loader, output_file=None, n=-1):
    metrics = Smoother(100)
    res, gts = [], {}
    tot = 0
    for (source, mask, targets) in tqdm(loader):
        if n>0 and tot>n:
            break
        source = to_device(source, 'cuda:0')
        mask = to_device(mask, 'cuda:0')
        pred = model(source, mask)
        pred = pred.cpu().numpy()
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
    return CustomBartModel()
    # return BartForConditionalGeneration.from_pretrained('facebook/bart-base')

def grid_params():
    if WANDB:
        wandb.init(
                project="2023GAIIC",
                name="grid",
        )

    valid_data = BartDataset(conf['valid_file'])

    valid_loader = DataLoader(valid_data, batch_size=conf['valid_batch'], shuffle=True, num_workers=12, drop_last=False)

    model = get_model()
    checkpoint = Checkpoint(model = model)
    model = torch.nn.DataParallel(model)
    model.to('cuda:0')

    checkpoint.resume(file_path="/root/2023GAIIC/checkpoint/2/model_6.pt")
    logger = Logger(conf['grid_dir']+'/log%d.txt'%version, 'a')
    logger.log(conf)
    writer = SummaryWriter(conf['grid_dir'])
    
    model.eval()
    best = 0.0
    Path(conf['grid_dir']).mkdir(exist_ok=True, parents=True)
    metrics = evaluate(model, valid_loader)
    logger.log('valid', metrics.value())
    if WANDB:
        wandb.log({
            'valid_metric': metrics.value(),
            'top_p':top_p,
            'temperature':temperature,
            'length_penalty':length_penalty,
            })
    writer.add_scalars('valid metric', metrics.value())
    if best < metrics.value()['cider']:
        best = metrics.value()['cider']
        if WANDB:
            wandb.log({
                'best_valid_metric': metrics.value(),
                'best_top_p':top_p,
                'best_temperature':temperature,
                'best_length_penalty':length_penalty,
                })

    logger.close()
    writer.close()
    
def inference(model_file, data_file):
    test_data = BartDataset(data_file)
    test_loader = DataLoader(test_data, batch_size=conf['valid_batch'], shuffle=False, num_workers=12, drop_last=False)

    model = get_model()
    checkpoint = Checkpoint(model = model)
    checkpoint.resume(model_file)
    
    model = nn.DataParallel(model)
    model.to('cuda:0')
    model.eval()
    
    fp = open('pred.csv', 'w', newline='')
    writer = csv.writer(fp)
    tot = 0
    for (source, mask) in tqdm(test_loader):
        source = to_device(source, 'cuda:0')
        mask = to_device(mask, 'cuda:0')
        pred = model(source, mask, infer=True)
        pred = np.array(pred)
        for i in range(pred.shape[0]):
            writer.writerow([tot, pred[i]])
            tot += 1
    fp.close()

version = 1
conf = Config(version)

grid_params()
# inference('checkpoint/%d/model_cider.pt'%version, conf['test_file'])
