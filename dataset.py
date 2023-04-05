import csv
import traceback

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


class NgramData(BaseDataset):
    def __init__(self, path: str):
        super().__init__()
        self.data = []
        with open(path) as f:
            for line in f:
                line = line.strip('\n').replace('\n','')
                self.data.append(line)
        self.max_len = 150
        self.tokenizer = BartTokenizer.from_pretrained('./custom_bart')
        self.mask_gen = NGramMaskGenerator(self.tokenizer, max_gram=4, max_seq_len=150)

    def _try_getitem(self, idx):
        data = self.data[idx]
        data_token = self.tokenizer.tokenize(data)
        data_token, lm_labels = self.mask_gen.mask_tokens(data_token, random)
        data_ids = self.tokenizer.convert_tokens_to_ids(data_token)
        features = OrderedDict(input_ids=data_ids,
                               input_mask=[1] * len(data_ids),
                               labels=lm_labels)

        torch.set_printoptions(profile="full")
        for f in features:
            features[f] = torch.LongTensor(features[f] + [0] * (self.max_len - len(data_ids)))
        return features["input_ids"], features["input_mask"], features["labels"]

    def __len__(self):
        return len(self.data)
