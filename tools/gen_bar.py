import os
import csv
import numpy as np
from tokenizers import ByteLevelBPETokenizer
from transformers import BartTokenizer, BartForConditionalGeneration

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

tokenizer = ByteLevelBPETokenizer(lowercase=True, add_prefix_space=True)
tokenizer.train(files='./data.txt', 
                min_frequency=2, 
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

tokenizer.save_model(path)
tokenizer = BartTokenizer.from_pretrained(path)
tokenizer.save_pretrained(path)

model = BartForConditionalGeneration.from_pretrained("/root/autodl-tmp/pretrain")
model.resize_token_embeddings(tokenizer.vocab_size)
model.save_pretrained(path)
