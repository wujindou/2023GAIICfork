
import torch
import torch.nn as nn
from pathlib import Path
import time
import json
import math
from bisect import bisect
import numpy as np
import random
import cv2
import copy
from collections import OrderedDict
from torch.nn.functional import normalize


def get_parameters(model, pars):
    ret = [{'params': getattr(model, x).parameters()} for x in pars]
    print(ret)
    return ret

def output_tensor(x, precision=3):
    print(np.round(x.detach().cpu().numpy(), precision))

def to_device(data, device):
  if isinstance(data, torch.Tensor):
    data = data.to(device)
  elif isinstance(data, np.ndarray):
    data = to_device(torch.from_numpy(data), device)
  elif isinstance(data, tuple):
    data = tuple(to_device(item,device) for item in data)
  elif isinstance(data, list):
    data = list(to_device(item,device) for item in data)
  elif isinstance(data, dict):
    data = dict((k,to_device(v,device)) for k,v in data.items())
  else:
    raise TypeError('Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.', type(data), data)
  return data

class Smoother():
    def __init__(self, window):
        self.window = window
        self.num = {}
        self.sum = {}
    def update(self, **kwargs):
        """
        为了调用方便一致，支持kwargs中有值为None的，会被忽略
        kwargs中一些值甚至可以为dict，也就是再套一层。
        示例: update(a=1, b=2, c={'c':1, 'd':3})，相当于update(a=1, b=2, c=1, d=3)
        如果值为参数的None的话忽略
        """
        values = {}
        for key in kwargs:
            if isinstance(kwargs[key], dict):
                for x in kwargs[key]:
                    values[x] = kwargs[key][x] #有可能会覆盖，如update(a=1,b={'a':2})
            else:
                values[key] = kwargs[key]
        for key in values:
            if values[key] is None:
                continue
            if key not in self.num:
                self.num[key] = []
                self.sum[key] = 0
            self.num[key].append(values[key])
            self.sum[key] += values[key]

            if len(self.num[key])>self.window:
                self.sum[key] -= self.num[key][-self.window-1]
            if len(self.num[key])>self.window*2:
                self.clear(key)
        pass
    def clear(self, key):
        del self.num[key][:-self.window]
    def value(self, key = None, mean=True):
        if mean:
            if key is None:
                return {key: self.sum[key] / min(len(self.num[key]),self.window) for key in self.num}
            return self.sum[key] / min(len(self.num[key]),self.window)
        if key is None:
            return {key: np.array(self.num[key]) for key in self.num}
        return np.array(self.sum[key])
    def keys(self):
        return self.num.keys()

class Step():
    def __init__(self):
        self.step = 0
        self.round = {}
    def clear(self):
        self.step = 0
        self.round = {}
    def forward(self, x):
        self.step += x
    def reach_cycle(self, mod, ignore_zero = True):
        now = self.step // mod
        if now==0 and ignore_zero:
            return False
        if mod not in self.round or self.round[mod]!=now: #新过了一个或多个cycle
            self.round[mod] = now
            return True
        return False
    def state_dict(self):
        return {'step': self.step, 'round':self.round}
    def load_state_dict(self, state):
        self.step = state['step']
        self.round = state['round']
    @property
    def value(self):
        return self.step

class Logger():
    def __init__(self, file_name, mode = 'w', buffer = 100):
        (Path(file_name).parent).mkdir(exist_ok = True, parents = True)
        self.file_name = file_name
        self.fp = open(file_name, mode)
        self.cnt = 0
        self.stamp = time.time()
        self.buffer = buffer
    def log(self, *args, end='\n'):
        for x in args:
            if isinstance(x, dict):
                for y in x:
                    self.fp.write(str(y)+':'+str(x[y])+' ')
            else:
                self.fp.write(str(x)+' ')
        self.fp.write(end)
        self.cnt += 1
        if self.cnt>=self.buffer or time.time()-self.stamp>5:
            self.cnt = 0
            self.stamp = time.time()
            self.fp.close()
            self.fp = open(self.file_name, 'a')
        pass
    def close(self):
        self.fp.close()

class Checkpoint():
    def __init__(self, **contents):
        """
        contents每个元素都需要有load_state_dict()方法
        """
        self.contents = contents
        self.contents['best_metrics'] = {}
    def update(self, file_path, logger = None, **kwargs):
        """
        根据metrics选择性地更新保存当前最好模型
        metrics: {metric_name: float 或 None}，越大越好。None的话忽略
        file_path: 保存文件名，*.pt
        """
        metrics = {}
        for key in kwargs:
            if isinstance(kwargs[key], dict):
                for x in kwargs[key]:
                    metrics[x] = kwargs[key][x] #有可能会覆盖，如update(a=1,b={'a':2})
            else:
                metrics[key] = kwargs[key]
        for metric in metrics:
            if metrics[metric] is None:
                continue
            if metric not in self.contents['best_metrics'] or metrics[metric]>self.contents['best_metrics'][metric]:
                self.contents['best_metrics'][metric] = metrics[metric]
                torch.save(self._get_contents(), file_path[:-3]+'_%s.pt'%metric)
                #torch.save(self.contents['optimizer'].state_dict(), file_path[:-3]+'_%s.pt'%metric)
                print('new best metric', metric, metrics[metric])
                if logger is not None:
                    logger.log('new best metric', metric, metrics[metric])
        pass
    def _get_contents(self):
        ret = {}
        for key in self.contents:
            if isinstance(self.contents[key], nn.DataParallel):
                ret[key] = self.contents[key].module.state_dict()
            elif hasattr(self.contents[key], 'state_dict'):
                ret[key] = self.contents[key].state_dict()
            else:
                ret[key] = self.contents[key]
        return ret
    def save(self, file_path):
        torch.save(self._get_contents(), file_path)
        
    def resume(self, file_path):
        memory = torch.load(file_path)
        self.contents['best_metrics'] = memory.pop('best_metrics')
        for key in memory:
            if key not in self.contents:
                print('loaded key not in contents:', key)
                continue
            if isinstance(self.contents[key], nn.DataParallel):
                self.contents[key].module.load_state_dict(memory[key])
            else:
                self.contents[key].load_state_dict(memory[key])
        pass

class EMA:
    def __init__(self, model, decay, device=None):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.model.to(device=self.device)

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if self.device is not None:
                    self.shadow[name].to(device=self.device)

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGM:
    def __init__(self, model: torch.nn.Module, eps=1.):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {
    }

    # only attack word embedding
    def attack(self, emb_name='embed_tokens'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embed_tokens'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {
    }

class AWP:
    """
    Implements weighted adverserial perturbation
    adapted from: https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1, adv_eps=0.0001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs, mask, targets):
        if self.adv_lr == 0:
            return
        self._save()
        self._attack_step()

        loss = self.model(inputs, mask, targets).loss
        loss = loss.mean()

        self.optimizer.zero_grad()
        return loss

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 在损失函数之前获得梯度
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


def truncate(a:list,b:list,maxLen):
    maxLen -= 3  # 空留给cls sep sep
    assert maxLen >= 0
    len2 = maxLen // 2  # 若为奇数，更长部分给左边
    len1 = maxLen - len2
    # 一共就a超长与否，b超长与否，组合的四种情况
    if len(a) + len(b) > maxLen:  # 需要截断
        if len(a) <= len1 and len(b) > len2:
            b = b[:maxLen - len(a)]
        elif len(a) > len1 and len(b) <= len2:
            a = a[:maxLen - len(b)]
        elif len(a) > len1 and len(b) > len2:
            a = a[:len1]
            b = b[:len2]
    return a, b


def paddingList(ls: list, val, returnTensor=False):
    ls = ls[:]  # 不要改变了原list尺寸
    maxLen = max([len(i) for i in ls])
    for i in range(len(ls)):
        ls[i] = ls[i] + [val] * (maxLen - len(ls[i]))
    return torch.tensor(ls, device='cuda') if returnTensor else ls


class NGramMaskGenerator:
    """
    Mask ngram tokens
    https://github.com/zihangdai/xlnet/blob/0b642d14dd8aec7f1e1ecbf7d6942d5faa6be1f0/data_utils.py
    """

    def __init__(self, tokenizer, mask_lm_prob=0.15, max_seq_len=512, max_preds_per_seq=None, max_gram=1, keep_prob=0.1,
                 mask_prob=0.8, **kwargs):
        self.tokenizer = tokenizer
        self.mask_lm_prob = mask_lm_prob
        self.keep_prob = keep_prob
        self.mask_prob = mask_prob
        assert self.mask_prob + self.keep_prob <= 1, f'The prob of using [MASK]({mask_prob}) and the prob of using original token({keep_prob}) should between [0,1]'
        self.max_preds_per_seq = max_preds_per_seq
        if max_preds_per_seq is None:
            self.max_preds_per_seq = math.ceil(max_seq_len * mask_lm_prob / 10) * 10

        self.max_gram = max(max_gram, 1)
        self.mask_window = int(1 / mask_lm_prob)  # make ngrams per window sized context
        # self.vocab_words = list(tokenizer.vocab.keys())
        self.vocab_words = list(tokenizer.get_vocab().keys())

    def mask_tokens(self, tokens, rng, **kwargs):
        special_tokens = ['<mask>', '<s>', '</s>', '<pad>', '<unk>']  # + self.tokenizer.tokenize(' ')
        indices = [i for i in range(len(tokens)) if tokens[i] not in special_tokens]
        ngrams = np.arange(1, self.max_gram + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, self.max_gram + 1)
        pvals /= pvals.sum(keepdims=True)

        unigrams = []
        for id in indices:
            if self.max_gram > 1 and len(unigrams) >= 1 and self.tokenizer.part_of_whole_word(tokens[id]):
                unigrams[-1].append(id)
            else:
                unigrams.append([id])

        num_to_predict = min(self.max_preds_per_seq, max(1, int(round(len(tokens) * self.mask_lm_prob))))
        mask_len = 0
        offset = 0
        mask_grams = np.array([False] * len(unigrams))
        while offset < len(unigrams):
            n = self._choice(rng, ngrams, p=pvals)
            ctx_size = min(n * self.mask_window, len(unigrams) - offset)
            m = rng.randint(0, ctx_size - 1)
            s = offset + m
            e = min(offset + m + n, len(unigrams))
            offset = max(offset + ctx_size, e)
            mask_grams[s:e] = True

        target_labels = [None] * len(tokens)
        w_cnt = 0
        for m, word in zip(mask_grams, unigrams):
            if m:
                for idx in word:
                    label = self._mask_token(idx, tokens, rng, self.mask_prob, self.keep_prob)
                    target_labels[idx] = label
                    w_cnt += 1
                if w_cnt >= num_to_predict:
                    break

        target_labels = [self.tokenizer.get_vocab()[x] if x else 0 for x in target_labels]
        return tokens, target_labels

    def _choice(self, rng, data, p):
        cul = np.cumsum(p)
        x = rng.random() * cul[-1]
        id = bisect(cul, x)
        return data[id]

    def _mask_token(self, idx, tokens, rng, mask_prob, keep_prob):
        label = tokens[idx]
        mask = '<mask>'
        rand = rng.random()
        if rand < mask_prob:
            new_label = mask
        elif rand < mask_prob + keep_prob:
            new_label = label
        else:
            new_label = rng.choice(self.vocab_words)

        tokens[idx] = new_label

        return label
