import os
import csv
import numpy as np
from tokenizers import ByteLevelBPETokenizer
from transformers import BartTokenizer, BartForConditionalGeneration, BertTokenizer

path = './custom_pretrain'
os.mkdir(path)
samples = []
with open("../data/raw.csv",'r') as fp:
    reader = csv.reader(fp)
    sample = [row for row in reader]
    for i in sample:
        samples.append(i[1])
        samples.append(i[2])

with open("../data/preliminary_a_test.csv",'r') as fp:
    reader = csv.reader(fp)
    sample = [row for row in reader]
    for i in sample:
        samples.append(i[1])

with open("data.txt", 'w') as f:
    for i in samples:
        f.write(i + '\n')

def build_vocab(vocab_file = './vocab.txt'):
    init_list = [x for x in range(1300)]
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    all_special_ids, all_special_tokens = zip(*sorted(zip(tokenizer.all_special_ids, tokenizer.all_special_tokens)))
    print(f'insert {all_special_ids} {all_special_tokens}')
    for i in range(len(all_special_ids)):
        init_list.insert(all_special_ids[i], all_special_tokens[i])

    with open(vocab_file, 'w') as fp:
        for i in init_list:
            print(i)
            fp.write(f'{i}\n')

tokenizer = ByteLevelBPETokenizer(lowercase=True, add_prefix_space=True)
tokenizer.train(files='./data.txt',  special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])
# tokenizer.add_special_tokens(['10','11'])
tokenizer.save_model(path)
# build_vocab(vocab_file="./vocab.txt")
tokenizer = BartTokenizer.from_pretrained(path)
tokenizer.save_pretrained(path)

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
# model.encoder.resize_embeddings(tokenizer.vocab_size)
# model.decoder.resize_embeddings(tokenizer.vocab_size)
model.resize_token_embeddings(tokenizer.vocab_size)
model.save_pretrained(path)
