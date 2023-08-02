from transformers import AutoTokenizer
import transformers
from signjoey.vocabulary import TextVocabulary

# Specify the name of the pre-trained BPE tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
TextVocabulary(file="/Users/eluzzon/efrat_private/fpt4slt/data/vocab.txt", mBartVocab=False)
sentence = "Your input sentence goes here doe's"
encoded = tokenizer.encode(sentence)
tokenizer.tokenize("I have a new GPU!")
decoded = tokenizer.decode(encoded)

x = 1
