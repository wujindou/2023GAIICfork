WANDB = True
import wandb
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
                name="pre_bart",
        )

    train_data = NgramData(conf['pretrain_file'])
    train_loader = DataLoader(train_data, batch_size=conf['batch'], shuffle=True, num_workers=12, drop_last=False)

    model = get_model()
    step = Step()
    checkpoint = Checkpoint(model = model, step = step)
    model = torch.nn.DataParallel(model)
    model.to('cuda:0')

    optimizer = torch.optim.AdamW(model.parameters(), lr=conf['lr'])

    start_epoch = 145
    
    checkpoint.resume(file_path="./pretrain/1/model_145.pt")
    logger = Logger(conf['pre_model_dir']+'/log%d.txt'%version, 'a')
    logger.log(conf)
    writer = SummaryWriter(conf['pre_model_dir'])
    
    Path(conf['pre_model_dir']).mkdir(exist_ok=True, parents=True)
    for epoch in range(start_epoch+1, conf['pre_n_epoch']):
        print('epoch', epoch)
        logger.log('new epoch', epoch)
        for (source, targets) in tqdm(train_loader):
            source = to_device(source, 'cuda:0')
            targets = to_device(targets, 'cuda:0')
            step.forward(source.shape[0])

            loss = model(source, targets).loss
            
            loss = loss.mean()
            loss.backward()
            optimizer.step() #优化一次
            optimizer.zero_grad() #清空梯度

            if step.value%100==0:
                logger.log(step.value, loss.item())
                if WANDB:
                    wandb.log({'step': step.value})
                    wandb.log({'train_loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        if epoch%5==0:
            checkpoint.save(conf['pre_model_dir']+'/model_%d.pt'%epoch)

        # scheduler.step()
        if WANDB:
            wandb.log({'epoch': epoch})
    logger.close()
    writer.close()

version = 1
conf = Config(version)

train()
