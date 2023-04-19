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
            self.tokenizer = BartTokenizer.from_pretrained('./custom_pretrain')
            self.input_l = 150
            self.output_l = 80
            self.sos_id = 0
            self.pad_id = 1
            self.eos_id = 2

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
        # print(source_ids['input_ids'],source_ids['attention_mask'],target_ids['input_ids'])
        return torch.LongTensor(source_ids['input_ids']), torch.LongTensor(source_ids['attention_mask']), torch.LongTensor(target_ids['input_ids'])
        # source = [self.sos_id] + [int(x) for x in self.samples[idx][1].split()] + [self.eos_id]
        # if len(source) < self.input_l:
        #     source.extend([self.pad_id] * (self.input_l - len(source)))
        # if len(self.samples[idx]) < 3:
        #     input_ids = np.array(source)[:self.input_l]
        #     attention_mask = np.array([1] * len(input_ids) + [0] * (self.input_l - len(input_ids)))
        #     assert(len(input_ids)==len(attention_mask))
        #     return input_ids, attention_mask
        # target = [self.sos_id] + [int(x) for x in self.samples[idx][2].split()] + [self.eos_id]
        # # target = [int(x) for x in self.samples[idx][2].split()]
        # if len(target) < self.output_l:
        #     target.extend([self.pad_id] * (self.output_l - len(target)))
        # input_ids = np.array(source)[:self.input_l]
        # attention_mask = np.array([1] * len(input_ids) + [0] * (self.input_l - len(input_ids)))
        # target_ids = np.array(target)[:self.output_l]
        # assert(len(input_ids)==len(attention_mask))
        # return input_ids, attention_mask, target_ids

class NgramData(BaseDataset):
    #传入句子对列表
    def __init__(self, path):
        super().__init__()
        self.samples = pd.read_csv(path,header=None)
        # with open(path,'r') as f:
        #     self.data = f.readlines()
        self.tk = BartTokenizer.from_pretrained('./custom_pretrain')
        self.spNum=len(self.tk.all_special_tokens)
        self.vocab_size=self.tk.vocab_size
        self.input_l = 150
        self.output_l= 80
        self.sos_id = 0
        self.pad_id = 1
        self.eos_id = 2
        self.mask_token_id = 4

    def __len__(self):
        return len(self.samples)

    def _try_getitem(self, idx):
        # if random.random()>0.5:
        text1 = self.samples.iloc[idx, 0]
        text1 = self.tk(text1, max_length=150, truncation=True)['input_ids'][1:-1]
        text1, out1_ids = self.random_mask(text1)
        input_ids = [self.sos_id] + text1 + [self.eos_id]
        labels = [-100] + out1_ids + [-100]
        if len(input_ids) < self.input_l:
            input_ids.extend([self.pad_id] * (self.input_l - len(input_ids)))
        if len(labels) < self.input_l:
            labels.extend([-100] * (self.input_l - len(labels)))
        assert len(input_ids)==len(labels)
        return torch.LongTensor(input_ids), torch.LongTensor(labels)
        # else:
        #     text1, text2 = self.samples.iloc[idx, 0], self.samples.iloc[idx, 1]
        #     if pd.isna(text2):
        #         text1 = self.tk(text1, max_length=self.input_l, truncation=True)['input_ids'][1:-1]
        #         text1, out1_ids = self.random_mask(text1)
        #         input_ids = [self.sos_id] + text1 + [self.eos_id]
        #         labels = [-100] + out1_ids + [-100]
        #         if len(input_ids) < self.input_l:
        #             input_ids.extend([self.pad_id] * (self.input_l - len(input_ids)))
        #         if len(labels) < self.input_l:
        #             labels.extend([-100] * (self.input_l - len(labels)))
        #         assert len(input_ids)==len(labels)
        #         return torch.LongTensor(input_ids), torch.LongTensor(labels)
        #     text1 = self.tk(text1, max_length=self.input_l, truncation=True)['input_ids'][1:-1]
        #     text2 = self.tk(text2, max_length=self.input_l, truncation=True)['input_ids'][1:-1]
        #     if random.random()>0.5:
        #         text1, text2 = text2, text1 
        #     text1, out1_ids = self.random_mask(text1)
        #     text2, out2_ids = self.random_mask(text2)
        #     input_ids = [self.sos_id] + text1 + [self.eos_id] + text2 + [self.eos_id]
        #     labels = [-100] + out1_ids + [-100] + out2_ids + [-100]
        #     if len(input_ids) < self.input_l:
        #         input_ids.extend([self.pad_id] * (self.input_l - len(input_ids)))
        #     if len(labels) < self.input_l:
        #         labels.extend([-100] * (self.input_l - len(labels)))
        #     assert len(input_ids)==len(labels)
        #     return torch.LongTensor(input_ids), torch.LongTensor(labels)

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
                input_ids.append(np.random.randint(self.spNum,self.vocab_size))
                output_ids.append(i)#随机的一个词预测自己，随机词不会从特殊符号中选取，有小概率抽到自己
            else:
                input_ids.append(i)
                output_ids.append(-100)#保持原样不预测

        return input_ids, output_ids

class DAEData(BaseDataset):
    #传入句子对列表
    def __init__(self, path):
        super().__init__()
        self.samples = pd.read_csv(path,header=None)
        self.tk = BartTokenizer.from_pretrained('./custom_pretrain')
        self.vocab_size = self.tk.vocab_size
        self.vocab_id_to_token_dict = {v: k for k, v in self.tk.get_vocab().items()}
        self.spNum=len(self.tk.all_special_tokens)
        self.vocab_size=self.tk.vocab_size
        self.input_l = 260
        self.sos_id = 0
        self.pad_id = 1
        self.eos_id = 2
        self.mask_token_id = 4
        self.seg_token_ids=[0,1,2,3,4]

        # Denoising ratios
        self.permute_sentence_ratio = 1.0
        self.mask_ratio = 0.15
        self.random_ratio = 0.1
        self.insert_ratio = 0.05
        self.rotate_ratio = 0.05
        self.item_transform_func = None

        self.mask_span_distribution = None

    def __len__(self):
        return len(self.samples)

    def _try_getitem(self, idx):
        if random.random()>0.5:
            text1 = self.samples.iloc[idx, 0]
            text1 = self.tk(text1, max_length=150, truncation=True)['input_ids'][1:-1]
            result = self.denoising_autoencoder(text1, self.input_l)
            return result['source'], result['target'], result['loss_mask']
        else:
            text1, text2 = self.samples.iloc[idx, 0], self.samples.iloc[idx, 1]
            if pd.isna(text2):
                text1 = self.tk(text1, max_length=150, truncation=True)['input_ids'][1:-1]
                result = self.denoising_autoencoder(text1, self.input_l)
                return result['source'], result['target'], result['loss_mask']
            text1 = self.tk(text1, max_length=150, truncation=True)['input_ids'][1:-1]
            text2 = self.tk(text2, max_length=80, truncation=True)['input_ids'][1:-1]
            if random.random()>0.5:
                text1, text2 = text2, text1
            input_ids = text1 + text2
            result = self.denoising_autoencoder(input_ids, self.input_l)
            return result['source'], result['target'], result['loss_mask']

    def denoising_autoencoder(self, source, max_seq_length):
        """Biuld training sample.

        Arguments:
            sample: A list of sentences in which each sentence is a list token ids.
            max_seq_length: Desired sequence length.
            np_rng: Random number genenrator. Note that this rng state should be
                numpy and not python since python randint is inclusive for
                the opper bound whereas the numpy one is exclusive.
        """
        tokens = [self.sos_id]
        for num in source:
            tokens.append(num)
            if num == 264:
                tokens.append(self.eos_id)

        # if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]
        tokens[-1] = self.eos_id
        tokens = torch.LongTensor(tokens)
        full_stops = (tokens == self.eos_id).long()
        assert (max_seq_length - tokens.shape[0]) >= 0, (tokens.size(), tokens[-1], max_seq_length)

        source, target = tokens, tokens.clone()
        use_decoder = 1
        # if torch.rand(1).item() < 0.5:
        #     use_decoder = 0
        if self.permute_sentence_ratio > 0.0 and use_decoder == 1:
            source = self.permute_sentences(source, full_stops, self.permute_sentence_ratio)

        if self.mask_ratio > 0.0:
            source = self.random_mask(source)

        if self.insert_ratio > 0.0:
            # raise NotImplementedError
            source = self.add_insertion_noise(source, self.insert_ratio)

        if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
        #     raise NotImplementedError
            source = self.add_rolling_noise(source)

        # there can additional changes to make:
        if self.item_transform_func is not None:
            source, target = self.item_transform_func(source, target)

        assert (source >= 0).all()
        # assert (source[1:-1] >= 1).all()
        assert (source <= self.vocab_size).all()
        assert source[0] == self.sos_id
        assert source[-1] == self.eos_id

        # tokenizer = get_tokenizer()
        # print(' '.join(tokenizer.tokenizer.convert_ids_to_tokens(source)))
        # print(tokenizer.detokenize(target))
        # print(tokenizer.detokenize(source))
        # print()

        prev_output_tokens = torch.zeros_like(target)
        prev_output_tokens[0] = self.eos_id  # match the preprocessing in fairseq
        prev_output_tokens[1:] = target[:-1]

        # src_padding_length = max_seq_length - source.shape[0]
        # tgt_padding_length = max_seq_length - target.shape[0]
        # assert src_padding_length >= 0, (source.size(), source[-1], max_seq_length)
        # assert tgt_padding_length >= 0, (target.size(), target[-1], max_seq_length)
        source_ = torch.full((max_seq_length,), self.pad_id, dtype=torch.long)
        source_[:source.shape[0]] = source
        target_ = torch.full((max_seq_length,), self.pad_id, dtype=torch.long)
        target_[:target.shape[0]] = target
        prev_output_tokens_ = torch.full((max_seq_length,), self.pad_id, dtype=torch.long)
        prev_output_tokens_[:prev_output_tokens.shape[0]] = prev_output_tokens


        return {
            "source": source_,
            "target": target_,
            "prev_output_tokens": prev_output_tokens_,
            "attn_mask": (source_ != self.pad_id).long(),
            "loss_mask": (target_ != self.pad_id).long() if use_decoder else (target_ != source_).long(),
            "use_decoder": torch.tensor(use_decoder).long()
        }

    def permute_sentences(self, source, full_stops, p=1.0):
        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero(as_tuple=False) + 2
        result = source.clone()

        num_sentences = sentence_ends.size(0)
        num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
        substitutions = torch.randperm(num_sentences)[:num_to_permute]
        ordering = torch.arange(0, num_sentences)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        # Ignore <bos> at start
        index = 1
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i > 0 else 1) : sentence_ends[i]]
            result[index : index + sentence.size(0)] = sentence
            index += sentence.size(0)
        return result

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_token_id
        result[noise_indices[:num_random]] = torch.randint(
            low=1, high=self.vocab_size, size=(num_random,)
        )

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

    def add_rolling_noise(self, tokens):
        offset = np.random.randint(1, max(1, tokens.size(-1) - 1) + 1)
        tokens = torch.cat(
            (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),
            dim=0,
        )
        return tokens
    
    def random_mask(self,text_ids):
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        idx=0
        while idx<len(rands):
            if rands[idx]<0.15:#需要mask
                ngram=np.random.choice([1,2,3], p=[0.7,0.2,0.1])
                if ngram==3 and len(rands)<7:
                    ngram=2
                if ngram==2 and len(rands)<4:
                    ngram=1
                L=idx+1
                R=idx+ngram
                while L<R and L<len(rands):
                    rands[L]=np.random.random()*0.15
                    L+=1
                idx=R
                if idx<len(rands):
                    rands[idx]=1
            idx+=1

        for r, i in zip(rands, text_ids):
            if r < 0.15 * 0.8 and i not in [0,1,2,3,4]:
                input_ids.append(self.mask_token_id)
            # elif r < 0.15:
            #     input_ids.append(np.random.randint(self.spNum,self.vocab_size))
            else:
                input_ids.append(i)

        return torch.LongTensor(input_ids)
