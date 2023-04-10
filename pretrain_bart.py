WANDB = True
import wandb
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as autocast
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
                name="pre_bart_large",
        )

    train_data = NgramData(conf['pretrain_file'])
    val_data = NgramData(conf['preval_file'])
    train_loader = DataLoader(train_data, batch_size=conf['batch'], shuffle=True, num_workers=12, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=conf['valid_batch'], shuffle=True, num_workers=12, drop_last=False)

    model = get_model()
    step = Step()
    checkpoint = Checkpoint(model = model, step = step)
    model = torch.nn.DataParallel(model)
    model.to('cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=conf['lr'])

    start_epoch = 0
    best_loss = 100.
    accumulation_steps = 1.

    # checkpoint.resume(file_path="./pretrain/1/model_145.pt")
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
            loss.backward()
            if ((i+1) % accumulation_steps)==0:
                optimizer.step()
                optimizer.zero_grad()

            if step.value%100==0:
                logger.log(step.value, loss.item()*accumulation_steps)
                if WANDB:
                    wandb.log({'step': step.value})
                    wandb.log({'train_loss': loss.item()*accumulation_steps, 'lr': optimizer.param_groups[0]['lr']})

        if epoch%5==0:
            # checkpoint.save(conf['pre_model_dir']+'/model_%d.pt'%epoch)
            model.eval()
            val_losses = []
            for (val_source, val_targets) in tqdm(val_loader):
                val_source = to_device(val_source, 'cuda')
                val_targets = to_device(val_targets, 'cuda')
                val_loss = model(val_source, val_targets).loss
                val_loss = val_loss.mean()
                val_losses.append(val_loss.item())
            val_losses = np.array(val_losses).mean()
            logger.log(val_losses)
            print("valid loss", val_losses)
            if WANDB:
                wandb.log({'preval_loss': val_losses})
            if best_loss>val_losses:
                print("Saving model...")
                best_loss=val_losses
                checkpoint.save(conf['pre_model_dir']+'/model_loss_%.4f.pt'%val_losses)
            model.train()

        if WANDB:
            wandb.log({'epoch': epoch})
    logger.close()
    writer.close()

version = 1
conf = Config(version)

train()
