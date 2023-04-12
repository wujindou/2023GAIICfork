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
        pred = model(source, mask, infer=True)
        pred = pred.cpu().numpy()
        # pred = np.array(pred)
        for i in range(pred.shape[0]):
            print(targets[i])
            res.append({'image_id':tot, 'caption': [array2str(pred[i][1:-1])]})
            print(array2str(pred[i][1:-1]))
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

def train():
    if WANDB:
        wandb.init(
                project="2023GAIIC",
                name="bart",
        )

    train_data = BartDataset(conf['train_file'])
    valid_data = BartDataset(conf['valid_file'])

    train_loader = DataLoader(train_data, batch_size=conf['batch'], shuffle=True, num_workers=12, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=conf['valid_batch'], shuffle=True, num_workers=12, drop_last=False)

    model = get_model()
    step = Step()
    checkpoint = Checkpoint(model = model, step = step)
    model = torch.nn.DataParallel(model)
    model.to('cuda:0')
    # ema = EMA(model, 0.999, device="cuda:0")
    # ema.register()
    fgm = FGM(model)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=conf['lr'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf['lr'])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 6, 12, 18, 24, 35], gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0008, epochs=conf['n_epoch'], steps_per_epoch=min(500, len(train_loader)), pct_start=0.05)

    awp = AWP(model, optimizer, adv_lr=0.1, adv_eps=0.001)

    # checkpoint.resume(file_path="./pretrain/2/model_loss_0.3469.pt")
    start_epoch = 0
    # checkpoint.resume(file_path="./checkpoint/2/model_9.pt")

    logger = Logger(conf['model_dir']+'/log%d.txt'%version, 'a')
    logger.log(conf)
    writer = SummaryWriter(conf['model_dir'])
    
    Path(conf['model_dir']).mkdir(exist_ok=True, parents=True)
    for epoch in range(start_epoch, conf['n_epoch']):
        print('epoch', epoch)
        logger.log('new epoch', epoch)
        for (source, mask, targets) in tqdm(train_loader):
            source = to_device(source, 'cuda:0')
            mask = to_device(mask, 'cuda:0')
            targets = to_device(targets, 'cuda:0')
            step.forward(source.shape[0])

            loss = model(source, mask, targets).loss
            
            loss = loss.mean()
            loss.backward()
            
            if conf['awp_start'] <= epoch:
                loss = awp.attack_backward(source, mask, targets)
                loss.backward()
                awp._restore()

            fgm.attack()
            adv_loss = model(source, mask, targets).loss
            adv_loss = adv_loss.mean()
            adv_loss.backward()
            fgm.restore()

            optimizer.step()
            # scheduler.step()
            # ema.update()
            optimizer.zero_grad()

            if step.value%100==0:
                logger.log(step.value, loss.item())
                if WANDB:
                    wandb.log({'step': step.value})
                    wandb.log({'train_loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
        # ema.apply_shadow()
        # if epoch > 100:
        if epoch%1==0:
            checkpoint.save(conf['model_dir']+'/model_%d.pt'%epoch)
            model.eval()
            metrics = evaluate(model, valid_loader)
            logger.log('valid', step.value, metrics.value())
            if WANDB:
                wandb.log({'valid_metric': metrics.value()})
            writer.add_scalars('valid metric', metrics.value(), step.value)
            checkpoint.update(conf['model_dir']+'/model.pt', metrics = metrics.value())
            model.train()

        if WANDB:
            wandb.log({'epoch': epoch})
        # ema.restore()
    logger.close()
    writer.close()
    
def inference(model_file, data_file):
    test_data = BartDataset(data_file)
    test_loader = DataLoader(test_data, batch_size=conf['valid_batch'], shuffle=False, num_workers=12, drop_last=False)

    model = get_model()
    averaged_weights = torch.load('./averaged_model_weights.pt')
    # model.load_state_dict(averaged_weights)
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

train()
# inference('checkpoint/%d/model_39.pt'%version, conf['test_file'])
