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
    def __init__(self, data_file, input_l, output_l, sos_id=1, eos_id=2, pad_id=0):
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
    def __init__(self, data_file, sos_id=0, eos_id=2, pad_id=1):
        with open(data_file, 'r') as fp:
            reader = csv.reader(fp)
            self.samples = [row for row in reader]
            self.sos_id = sos_id
            self.pad_id = pad_id
            self.eos_id = eos_id
            self.tokenizer = BartTokenizer.from_pretrained('./custom_bart')

    def __len__(self):
        return len(self.samples)

    def _try_getitem(self, idx):
        source = self.samples[idx][1]
        source_ids = self.tokenizer(source, max_length=150, padding='max_length', truncation=True)
        try:
            target = self.samples[idx][2]
        except:
            return torch.LongTensor(source_ids['input_ids']), torch.LongTensor(source_ids['attention_mask'])
        target_ids = self.tokenizer(target, max_length=80, padding='max_length', truncation=True)
        return torch.LongTensor(source_ids['input_ids']), torch.LongTensor(
            source_ids['attention_mask']), torch.LongTensor(target_ids['input_ids'])


# class NgramData(BaseDataset):
#     def __init__(self, path: str):
#         super().__init__()
#         self.data = []
#         with open(path) as f:
#             for line in f:
#                 line = line.strip('\n').replace('\n','')
#                 self.data.append(line)
#         self.max_len = 150
#         self.tokenizer = BartTokenizer.from_pretrained('./custom_bart')
#         self.mask_gen = NGramMaskGenerator(self.tokenizer, max_gram=4, max_seq_len=150)
#
#     def _try_getitem(self, idx):
#         data = self.data[idx]
#         data_token = self.tokenizer.tokenize(data)
#         data_token, lm_labels = self.mask_gen.mask_tokens(data_token, random)
#         data_ids = self.tokenizer.convert_tokens_to_ids(data_token)
#         features = OrderedDict(input_ids=data_ids,
#                                input_mask=[1] * len(data_ids),
#                                labels=lm_labels)
#
#         torch.set_printoptions(profile="full")
#         for f in features:
#             features[f] = torch.LongTensor(features[f] + [0] * (self.max_len - len(data_ids)))
#         return features["input_ids"], features["input_mask"], features["labels"]

class NgramData(BaseDataset):
    #传入句子对列表
    def __init__(self, path):
        super().__init__()
        self.data = pd.read_csv(path,header=None)
        self.test_text = self.data.iloc[:, 1].isnull()
        self.tk = BartTokenizer.from_pretrained('./custom_bart')
        self.maxLen = 150
        self.spNum=len(self.tk.all_special_tokens)
        self.tkNum=self.tk.vocab_size

    def __len__(self):
        return len(self.data)
    def _try_getitem(self, idx):
        text1, text2, = self.data.iloc[idx]
        if random.random()>0.5:
            text1,text2=text2,text1
        text1,text2=truncate(text1,text2,self.maxLen)
        text1_ids,text2_ids = self.tk.convert_tokens_to_ids(text1),self.tk.convert_tokens_to_ids(text2)
        text1_ids, out1_ids = self.random_mask(text1_ids)
        text2_ids, out2_ids = self.random_mask(text2_ids)
        input_ids = [self.tk.cls_token_id] + text1_ids + [self.tk.sep_token_id] + text2_ids + [self.tk.sep_token_id]
        token_type_ids=[0]*(len(text1_ids)+2)+[1]*(len(text2_ids)+1)
        labels = [-100] + out1_ids + [-100] + out2_ids + [-100]
        assert len(input_ids)==len(token_type_ids)==len(labels)
        return input_ids, token_type_ids, labels

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
                input_ids.append(self.tk.mask_token_id)
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

