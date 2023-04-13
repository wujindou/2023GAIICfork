import csv
import traceback

import pandas as pd
from torch.utils.data import Dataset
from transformers import BartTokenizer

from utils import *


class BaseDataset(Dataset):
    def _try_getitem(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        wait = 0.1
        while True:
            try:
                ret = self._try_getitem(idx)
                return ret
            except KeyboardInterrupt:
                break
            except (Exception, BaseException) as e:
                exstr = traceback.format_exc()
                print(exstr)
                print('read error, waiting:', wait)
                time.sleep(wait)
                wait = min(wait * 2, 1000)


class TranslationDataset(BaseDataset):
    def __init__(self, data_file, input_l=150, output_l=80, sos_id=1, eos_id=2, pad_id=0):
        with open(data_file, 'r') as fp:
            reader = csv.reader(fp)
            self.samples = [row for row in reader]
            self.input_l = input_l
            self.output_l = output_l
            self.sos_id = sos_id
            self.pad_id = pad_id
            self.eos_id = eos_id

    def __len__(self):
        return len(self.samples)

    def _try_getitem(self, idx):
        source = [int(x) for x in self.samples[idx][1].split()]
        if len(source) < self.input_l:
            source.extend([self.pad_id] * (self.input_l - len(source)))
        if len(self.samples[idx]) < 3:
            return np.array(source)[:self.input_l]
        target = [self.sos_id] + [int(x) for x in self.samples[idx][2].split()] + [self.eos_id]
        if len(target) < self.output_l:
            target.extend([self.pad_id] * (self.output_l - len(target)))
        return np.array(source)[:self.input_l], np.array(target)[:self.output_l]


class BartDataset(BaseDataset):
    def __init__(self, data_file):
        with open(data_file, 'r') as fp:
            reader = csv.reader(fp)
            self.samples = [row for row in reader]
            # self.tokenizer = BartTokenizer.from_pretrained('./custom_pretrain_cn')
            self.input_l = 150
            self.output_l = 80
            self.sos_id = 0
            self.pad_id = 1
            self.eos_id = 2

    def __len__(self):
        return len(self.samples)

    def _try_getitem(self, idx):
        # source = self.samples[idx][1]
        # source_ids = self.tokenizer(source, max_length=150, padding='max_length', truncation=True)
        # try:
        #     target = self.samples[idx][2]
        # except:
        #     return torch.LongTensor(source_ids['input_ids']), torch.LongTensor(source_ids['attention_mask'])
        # target_ids = self.tokenizer(target, max_length=80, padding='max_length', truncation=True)
        # print(source_ids['input_ids'])
        # print(source_ids['attention_mask'])
        # print(target_ids['input_ids'])
        # return torch.LongTensor(source_ids['input_ids']), torch.LongTensor(
        #     source_ids['attention_mask']), torch.LongTensor(target_ids['input_ids'])
        source = [self.sos_id] + [int(x) for x in self.samples[idx][1].split()] + [self.eos_id]
        if len(source) < self.input_l:
            source.extend([self.pad_id] * (self.input_l - len(source)))
        if len(self.samples[idx]) < 3:
            return np.array(source)[:self.input_l]
        target = [self.sos_id] + [int(x) for x in self.samples[idx][2].split()] + [self.eos_id]
        # target = [int(x) for x in self.samples[idx][2].split()]
        if len(target) < self.output_l:
            target.extend([self.pad_id] * (self.output_l - len(target)))
        input_ids = np.array(source)[:self.input_l]
        attention_mask = np.array([1] * len(input_ids) + [0] * (self.input_l - len(input_ids)))
        target_ids = np.array(target)[:self.output_l]
        assert(len(input_ids)==len(attention_mask))
        return input_ids, attention_mask, target_ids

class NgramData(BaseDataset):
    #传入句子对列表
    def __init__(self, path):
        super().__init__()
        self.samples = pd.read_csv(path,header=None)
        # with open(path,'r') as f:
        #     self.data = f.readlines()
        self.tk = BartTokenizer.from_pretrained('./custom_pretrain_cn')
        self.spNum=len(self.tk.all_special_tokens)
        self.tkNum=self.tk.vocab_size
        self.input_l = 224
        self.output_l= 80
        self.sos_id = 0
        self.pad_id = 1
        self.eos_id = 2
        self.mask_token_id = 4

    def __len__(self):
        return len(self.samples)

    def _try_getitem(self, idx):
        text1 = self.samples.iloc[idx, 1]
        text2 = self.samples.iloc[idx, 2]

        # if pd.isna(text2):
        #     text1 = [int(x) for x in text1.split()]
        #     input_ids, out1_ids = self.random_mask(text1)
        #     input_ids = [self.sos_id] + text1 + [self.eos_id]
        #     labels = [-100] + out1_ids + [-100]
        #     if len(input_ids) < self.input_l:
        #         input_ids.extend([self.pad_id] * (self.input_l - len(input_ids)))
        #     if len(labels) < self.input_l:
        #         labels.extend([-100] * (self.input_l - len(labels)))
        #     assert len(input_ids)==len(labels)
        #     return torch.LongTensor(input_ids), torch.LongTensor(labels)
        text1 = [int(x) for x in text1.split()]
        text2 = [int(x) for x in text2.split()]
        if random.random()>0.5:
            text1, text2 = text2, text1 
        text1, out1_ids = self.random_mask(text1)
        text2, out2_ids = self.random_mask(text2)
        input_ids = [self.sos_id] + text1 + [self.eos_id] + text2 + [self.eos_id]
        labels = [-100] + out1_ids + [-100] + out2_ids + [-100]
        if len(input_ids) < self.input_l:
            input_ids.extend([self.pad_id] * (self.input_l - len(input_ids)))
        if len(labels) < self.input_l:
            labels.extend([-100] * (self.input_l - len(labels)))
        assert len(input_ids)==len(labels)
        return torch.LongTensor(input_ids), torch.LongTensor(labels)

    def random_mask(self,text_ids):
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        idx=0
        while idx<len(rands):
            if rands[idx]<0.15:#需要mask
                ngram=np.random.choice([1,2,3], p=[0.7,0.2,0.1])#若要mask，进行x_gram mask的概率
                if ngram==3 and len(rands)<7:#太大的gram不要应用于过短文本
                    ngram=2
                if ngram==2 and len(rands)<4:
                    ngram=1
                L=idx+1
                R=idx+ngram#最终需要mask的右边界（开）
                while L<R and L<len(rands):
                    rands[L]=np.random.random()*0.15#强制mask
                    L+=1
                idx=R
                if idx<len(rands):
                    rands[idx]=1#禁止mask片段的下一个token被mask，防止一大片连续mask
            idx+=1

        for r, i in zip(rands, text_ids):
            if r < 0.15 * 0.8:
                input_ids.append(self.mask_token_id)
                output_ids.append(i)#mask预测自己
            elif r < 0.15 * 0.9:
                input_ids.append(i)
                output_ids.append(i)#自己预测自己
            elif r < 0.15:
                input_ids.append(np.random.randint(self.spNum,self.tkNum))
                output_ids.append(i)#随机的一个词预测自己，随机词不会从特殊符号中选取，有小概率抽到自己
            else:
                input_ids.append(i)
                output_ids.append(-100)#保持原样不预测

        return input_ids, output_ids
