from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

# Load the vocabulary and merges files
vocab_file = "vocab.json"
merges_file = "merges.txt"

# Initialize the BPE model
bpe = BPE(vocab_file, merges_file)

# Initialize the tokenizer with the BPE model
tokenizer = Tokenizer(bpe)

# Set the pre-tokenizer to whitespace
tokenizer.pre_tokenizer = Whitespace()

# Define the text to tokenize
text = "Hello, world! This is a test."

# Tokenize the text
tokens = tokenizer.encode(text)

# Print the tokens
print(tokens.tokens)

