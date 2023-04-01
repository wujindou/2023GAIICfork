WANDB = True
import wandb
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import csv

from utils import to_device, Checkpoint, Step, Smoother, Logger, EMA, FGM
from models_bart import BartModel2
from dataset import BartDataset
from config_bart import Config
from losses import CE

from transformers import BartConfig, BartForConditionalGeneration

from evaluate import CiderD

def compute_batch(model, source, mask, targets, verbose = False, optional_ret = []):
    source = to_device(source, 'cuda:0')
    mask = to_device(mask, 'cuda:0')
    targets = to_device(targets, 'cuda:0')
    output = model(source, mask, targets)
    return output.loss

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
            res.append({'image_id':tot, 'caption': [array2str(pred[i][1:-1])]})
            gts[tot] = [array2str(targets[i])]
            tot += 1
    CiderD_scorer = CiderD(df='corpus', sigma=15)
    cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
    metrics.update(cider = cider_score)
    print(metrics.value())
    return metrics

def get_model():
    return BartModel2(n_token=conf['n_token'])
    # return BartForConditionalGeneration.from_pretrained('facebook/bart-base')

def train():
    if WANDB:
        wandb.init(
                project="2023GAIIC",
                name="bart",
        )

    train_data = BartDataset(conf['train_file'])
    valid_data = BartDataset(conf['valid_file'])

    train_loader = DataLoader(train_data, batch_size=conf['batch'], shuffle=True, num_workers=12, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=conf['valid_batch'], shuffle=True, num_workers=12, drop_last=False)

    model = get_model()
    step = Step()
    checkpoint = Checkpoint(model = model, step = step)
    model = torch.nn.DataParallel(model)
    ema = EMA(model, 0.999, device="cuda:0")
    ema.register()
    fgm = FGM(model)
    model.to('cuda:0')

    optimizer = torch.optim.AdamW(model.parameters(), lr=conf['lr'])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 12, 24, 40], gamma=0.8)

    start_epoch = 0
    
    logger = Logger(conf['model_dir']+'/log%d.txt'%version, 'a')
    logger.log(conf)
    writer = SummaryWriter(conf['model_dir'])
    
    Path(conf['model_dir']).mkdir(exist_ok=True, parents=True)
    for epoch in range(start_epoch, conf['n_epoch']):
        print('epoch', epoch)
        logger.log('new epoch', epoch)
        for (source, mask, targets) in tqdm(train_loader):
            step.forward(source.shape[0])
            
            loss = compute_batch(model, source, mask, targets)
            loss = loss.mean()
            loss.backward()

            fgm.attack()
            adv_loss = compute_batch(model, source, mask, targets)
            adv_loss = adv_loss.mean()
            adv_loss.backward()
            fgm.restore()

            optimizer.step() #优化一次
            ema.update()
            optimizer.zero_grad() #清空梯度

            if step.value%100==0:
                logger.log(step.value, loss.item())
                if WANDB:
                    wandb.log({'step': step.value})
                    wandb.log({'train_loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
        ema.apply_shadow()
        if (epoch%3==0 and epoch >= 24) or epoch%6==0:
            checkpoint.save(conf['model_dir']+'/model_%d.pt'%epoch)
            model.eval()
            metrics = evaluate(model, valid_loader)
            logger.log('valid', step.value, metrics.value())
            if WANDB:
                wandb.log({'valid_metric': metrics.value()})
            writer.add_scalars('valid metric', metrics.value(), step.value)
            checkpoint.update(conf['model_dir']+'/model.pt', metrics = metrics.value())
            model.train()

        # scheduler.step()
        if WANDB:
            wandb.log({'epoch': epoch})
        ema.restore()
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

version = 2
conf = Config(version)

# train()
inference('checkpoint/%d/model_cider.pt'%version, conf['test_file'])
