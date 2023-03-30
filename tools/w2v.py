import csv
import numpy as np
from gensim.models import word2vec

with open("../data/raw.csv",'r') as fp:
    reader = csv.reader(fp)
    sample = [row for row in reader]

samples = []
for i in sample:
    samples.append([int(x) for x in i[1].split()])
    samples.append([int(x) for x in i[2].split()])

model = word2vec.Word2Vec(samples, vector_size=512, min_count=1, workers=12)

model.save('w2v.model')
model.wv.save_word2vec_format('w2v.vectors')
