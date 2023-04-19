import csv
import traceback

import pandas as pd
from torch.utils.data import Dataset
from transformers import BartTokenizer
import jieba_fast

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
        self.zh_tokenizer = jieba_fast.lcut
        self.input_l = 150
        self.output_l= 80
        self.sos_id = 0
        self.pad_id = 1
        self.eos_id = 2
        self.mask_token_id = 4
        self.seg_token_ids=[0,1,2,3,4]

        # Denoising ratios
        self.permute_sentence_ratio = 1.0
        self.mask_ratio = 0.15
        self.random_ratio = 0.1
        self.insert_ratio = 0.0
        self.rotate_ratio = 0.0
        self.mask_whole_word = 1
        self.item_transform_func = None

        self.mask_span_distribution = None

    def __len__(self):
        return len(self.samples)

    def _try_getitem(self, idx):
        # if random.random()>0.5:
        text1 = self.samples.iloc[idx, 0]
        text1 = self.tk(text1, max_length=150, truncation=True)['input_ids'][1:-1]
        text1 = [self.sos_id] + text1 + [self.eos_id]
        result = self.denoising_autoencoder(text1, self.input_l)
        """
        {
            "source": source_,
            "target": target_,
            "prev_output_tokens": prev_output_tokens_,
            "attn_mask": (source_ != self.pad_id).long(),
            "loss_mask": (target_ != self.pad_id).long() if use_decoder else (target_ != source_).long(),
            "use_decoder": torch.tensor(use_decoder).long()
        }
        """
        print(result['source'], result['target'], result['loss_mask'])
        return result['source'], result['target'], result['loss_mask']
        # else:
        #     text1, text2 = self.samples.iloc[idx, 0], self.samples.iloc[idx, 1]
        #     if pd.isna(text2):
        #         text1 = self.tk(text1, max_length=self.input_l, truncation=True)['input_ids'][1:-1]
        #         text1, out1_ids = self.denoising_autoencoder(text1, self.input_l)
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
        #     text1, out1_ids = self.denoising_autoencoder(text1, self.input_l)
        #     text2, out2_ids = self.denoising_autoencoder(text2, self.input_l)
        #     input_ids = [self.sos_id] + text1 + [self.eos_id] + text2 + [self.eos_id]
        #     labels = [-100] + out1_ids + [-100] + out2_ids + [-100]
        #     if len(input_ids) < self.input_l:
        #         input_ids.extend([self.pad_id] * (self.input_l - len(input_ids)))
        #     if len(labels) < self.input_l:
        #         labels.extend([-100] * (self.input_l - len(labels)))
        #     assert len(input_ids)==len(labels)
        #     return torch.LongTensor(input_ids), torch.LongTensor(labels)

    def denoising_autoencoder(self, tokens, max_seq_length):
        """Biuld training sample.

        Arguments:
            sample: A list of sentences in which each sentence is a list token ids.
            max_seq_length: Desired sequence length.
            np_rng: Random number genenrator. Note that this rng state should be
                numpy and not python since python randint is inclusive for
                the opper bound whereas the numpy one is exclusive.
        """
        if len(tokens) > max_seq_length:
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
            replace_length = 1 if use_decoder else -1
            mask_ratio = self.mask_ratio * 2 if use_decoder else self.mask_ratio
            source = self.add_whole_word_mask(source, mask_ratio, replace_length)

        if self.insert_ratio > 0.0:
            raise NotImplementedError
            source = self.add_insertion_noise(source, self.insert_ratio)

        if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
            raise NotImplementedError
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

    def add_whole_word_mask(self, source, p, replace_length=1):
        is_word_start, word_starts = self.word_starts(source)
        num_to_mask_word = int(math.ceil(word_starts.size(0) * p))
        num_to_mask_char = int(math.ceil(word_starts.size(0) * p * 0.1))
        num_to_mask = num_to_mask_word + num_to_mask_char
        if num_to_mask > word_starts.size(0):
            word_starts = is_word_start.nonzero(as_tuple=False)
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio
        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            # print(source.size(), word_starts.size(), indices.size(), mask_random.size())
            source[indices] = self.mask_token_id
            source[indices[mask_random]] = torch.randint(
                1, self.vocab_size, size=(mask_random.sum(),)
            )
            # sorted_indices = torch.sort(indices)[0]
            # continue_mask_pos = ((sorted_indices + 1)[:-1] == sorted_indices[1:])
            # continue_mask_indices = sorted_indices[1:][continue_mask_pos]
            # to_keep[continue_mask_indices] = 0

        # for char indices, we already masked, the following loop handles word mask
        indices = indices[:num_to_mask_word]
        mask_random = mask_random[:num_to_mask_word]
        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_token_id
                    source[indices[mask_random]] = torch.randint(
                        1, self.vocab_size, size=(mask_random.sum(),)
                    )
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_token_id
                    source[indices[mask_random]] = torch.randint(
                        1, self.vocab_size, size=(mask_random.sum(),)
                    )

                assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def word_starts(self, source):
        if self.mask_whole_word is None:
            is_word_start = torch.ones(source.size())
            is_word_start[0] = 0
            is_word_start[-1] = 0
            return is_word_start
        raw_tokens = [self.vocab_id_to_token_dict[i] for i in source.tolist()]
        words = [raw_tokens[0]] + self.zh_tokenizer(''.join(raw_tokens[1:-1]), HMM=True) + [raw_tokens[-1]]

        def _is_chinese_char(c):
            """Checks whether CP is the codepoint of a CJK character."""
            # This defines a "chinese character" as anything in the CJK Unicode block:
            #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
            #
            # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
            # despite its name. The modern Korean Hangul alphabet is a different block,
            # as is Japanese Hiragana and Katakana. Those alphabets are used to write
            # space-separated words, so they are not treated specially and handled
            # like the all of the other languages.
            if len(c) > 1:
                return all([_is_chinese_char(c_i) for c_i in c])
            cp = ord(c)
            if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                    (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
                return True

            return False

        def align_linear(atokens, btokens):
            a2c = []
            c2b = []
            a2b = []
            length = 0
            for tok in atokens:
                a2c.append([length + i for i in range(len(tok))])
                length += len(tok)
            for i, tok in enumerate(btokens):
                c2b.extend([i for _ in range(len(tok))])

            for i, amap in enumerate(a2c):
                bmap = [c2b[ci] for ci in amap]
                a2b.append(list(set(bmap)))
            return a2b
        
        raw_to_word_align = align_linear(raw_tokens, words)
        is_word_start = torch.zeros(source.size())
        word_starts = []
        skip_cur_word = True
        for i in range(1, len(raw_to_word_align)):
            if raw_to_word_align[i-1] == raw_to_word_align[i]:
                # not a word start, as they align to the same word
                if not skip_cur_word and not _is_chinese_char(raw_tokens[i]):
                    word_starts.pop(-1)
                    skip_cur_word = True
                continue
            else:
                is_word_start[i] = 1
                if _is_chinese_char(raw_tokens[i]):
                    word_starts.append(i)
                    skip_cur_word = False
        is_word_start[0] = 0
        is_word_start[-1] = 0
        word_starts = torch.tensor(word_starts).long().view(-1, 1)
        return is_word_start, word_starts

