WANDB = False
import wandb
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from pathlib import Path
from tqdm import tqdm

from utils import to_device, Checkpoint, Step, Logger
from models_bart import PretrainBartModel
from dataset import NgramData
from config_bart import Config

def get_model():
    return PretrainBartModel(n_token=conf['n_token'])

def train():
    if WANDB:
        wandb.init(
                project="2023GAIIC",
                name="pre_bart_ngram",
        )

    train_data = NgramData(conf['pretrain_file'])
    val_data = NgramData(conf['preval_file'])
    train_loader = DataLoader(train_data, batch_size=conf['batch'], shuffle=True, num_workers=12, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=conf['valid_batch'], shuffle=True, num_workers=12, drop_last=False)

    model = get_model()
    step = Step()
    checkpoint = Checkpoint(model = model, step = step)
    model = torch.nn.DataParallel(model)
    accumulation_steps = 4.
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf['lr'])
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, epochs=conf['n_epoch'], steps_per_epoch=min(500, int(len(train_loader)/accumulation_steps)), pct_start=0.05)
    scaler = GradScaler()

    # checkpoint.resume(file_path="./pretrain/model_loss_0.3173.pt")
    start_epoch = 0
    best_loss = 100.

    logger = Logger(conf['pre_model_dir']+'/log%d.txt'%version, 'a')
    logger.log(conf)
    writer = SummaryWriter(conf['pre_model_dir'])
    
    Path(conf['pre_model_dir']).mkdir(exist_ok=True, parents=True)
    for epoch in range(start_epoch, conf['pre_n_epoch']):
        print('epoch', epoch)
        logger.log('new epoch', epoch)
        for i, (source, targets) in enumerate(tqdm(train_loader)):
            source = to_device(source, 'cuda')
            targets = to_device(targets, 'cuda')
            step.forward(source.shape[0])
            with autocast():
                loss = model(source, targets).loss
                loss = loss.mean() / accumulation_steps
            # loss.backward()
            scaler.scale(loss).backward()
            if ((i+1) % accumulation_steps)==0:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
                # optimizer.step()
                # scheduler.step()
                # optimizer.zero_grad()
                scaler.step(optimizer)
                scaler.update()

            if step.value%50==0:
                logger.log(step.value, loss.item()*accumulation_steps)
                if WANDB:
                    wandb.log({'step': step.value})
                    wandb.log({'train_loss': loss.item()*accumulation_steps, 'lr': optimizer.param_groups[0]['lr']})

        if epoch%10==0:
            checkpoint.save(conf['pre_model_dir']+'/model_%d.pt'%epoch)

        if WANDB:
            wandb.log({'epoch': epoch})
    logger.close()
    writer.close()

version = 2
conf = Config(version)

train()
