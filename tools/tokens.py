from tokenizers import ByteLevelBPETokenizer
import csv
with open("../data/raw.csv",'r') as fp:
    reader = csv.reader(fp)
    sample = [row for row in reader]

samples = []
for i in sample:
    samples.append(i[1])
    samples.append(i[2])

tokenizer = ByteLevelBPETokenizer()

tokenizer.train_from_iterator(iterator=samples, vocab_size=5000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model(".", "custom_bart")
