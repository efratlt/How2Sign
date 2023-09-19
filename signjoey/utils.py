from collections import defaultdict
from transformers import AutoTokenizer
import sentencepiece as spm
import torch
def get_tokenizer(data_cfg):
    tokenizer_type = data_cfg["tokenizer"]
    tokenizer = None
    level = data_cfg["level"]
    if tokenizer_type == "transformer":
        if level == "sentencepiece":
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.Load(data_cfg["tokenize_model"])
        else:
            tokenizer = AutoTokenizer.from_pretrained(level)
    return tokenizer


def save_vocab(vocab, name, dir_path):
    # dir_path = "/Users/eluzzon/efrat_private/How2Sign/data/vocab_pickle"
    torch.save(vocab, f"{dir_path}/{name}_vocab.pt", _use_new_zipfile_serialization=False)


def load_vocab(name, dir_path):
    return torch.load(f"{dir_path}/{name}_vocab.pt")