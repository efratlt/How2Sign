# coding: utf-8
import numpy as np

from collections import defaultdict, Counter
from typing import List
from torchtext.data import Dataset
from transformers import AutoTokenizer

def get_special_tokens(data_cfg=None, tokenizer=None):
    if data_cfg and data_cfg["tokenizer"] == "transformer":
        if data_cfg["level"] == "sentencepiece":
            special_tokens = [UNK_TOKEN(data_cfg=data_cfg, tokenizer=tokenizer),
                              PAD_TOKEN(data_cfg=data_cfg, tokenizer=tokenizer),
                              BOS_TOKEN(data_cfg=data_cfg, tokenizer=tokenizer),
                              EOS_TOKEN(data_cfg=data_cfg, tokenizer=tokenizer)]
        else:
            special_tokens = tokenizer.all_special_tokens
    else:
        special_tokens = [UNK_TOKEN(), PAD_TOKEN(), BOS_TOKEN(), EOS_TOKEN()]
    return special_tokens

def get_unk_token_id(data_cfg=None, tokenizer=None):

    if data_cfg and data_cfg["tokenizer"] == "transformer":
        if data_cfg["level"] == "sentencepiece":
            id = tokenizer.unk_id()
        else:
            id = tokenizer.unk_token_id
    else:
        assert 0
    return id
def SIL_TOKEN(data_cfg=None, tokenizer=None):
    sil = "<si>"
    if data_cfg and data_cfg["tokenizer"] == "transformer":
        if data_cfg["level"] == "sentencepiece":
            sil = "<unk>"
        elif tokenizer is not None:
            sil = tokenizer.unk_token
    return sil

def UNK_TOKEN(data_cfg=None, tokenizer=None):
    unk = "<unk>"
    if data_cfg and data_cfg["tokenizer"] == "transformer":
        if data_cfg["level"] == "sentencepiece":
            unk = "<unk>"
        elif tokenizer is not None:
            unk = tokenizer.unk_token
    return unk

def PAD_TOKEN(data_cfg=None, tokenizer=None):
    pad = "<pad>"
    if data_cfg and data_cfg["tokenizer"] == "transformer":
        if data_cfg["level"] == "sentencepiece":
            pad = "<pad>"
        elif tokenizer is not None:
            pad = tokenizer.unk_token
    return pad

def BOS_TOKEN(data_cfg=None, tokenizer=None):
    bos = "<s>"
    if data_cfg and data_cfg["tokenizer"] == "transformer":
        if data_cfg["level"] == "sentencepiece":
            bos = "<s>"
        elif tokenizer is not None:
            bos = tokenizer.unk_token
    return bos

def EOS_TOKEN(data_cfg=None, tokenizer=None):
    eos = "</s>"
    if data_cfg and data_cfg["tokenizer"] == "transformer":
        if data_cfg["level"] == "sentencepiece":
            eos = "</s>"
        elif tokenizer is not None:
            eos = tokenizer.unk_token
    return eos
"""
SIL_TOKEN = "<si>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
"""

class Vocabulary:
    """ Vocabulary represents mapping between tokens and indices. """

    def __init__(self):
        # don't rename stoi and itos since needed for torchtext
        # warning: stoi grows with unknown tokens, don't use for saving or size
        self.specials = []
        self.itos = []
        self.stoi = None
        self.DEFAULT_UNK_ID = None

    def _from_list(self, tokens: List[str] = None, fast: bool = False):
        """
        Make vocabulary from list of tokens.
        Tokens are assumed to be unique and pre-selected.
        Special symbols are added if not in list.

        :param tokens: list of tokens
        :param fast: check duplicate check if true
        """
        self.add_tokens(tokens=self.specials + tokens, fast=fast)
        assert len(self.stoi) == len(self.itos)

    def _from_file(self, file: str):
        """
        Make vocabulary from contents of file.
        File format: token with index i is in line i.

        :param file: path to file where the vocabulary is loaded from
        """
        print(f'Loading vocabulary from {file}...')
        tokens = []
        with open(file, "r", encoding="utf-8") as open_file:
            lines = open_file.readlines()
            print(f'Vocabulary contains {len(lines)} entries')
            for line in lines:
                tokens.append(line.strip("\n"))
        self._from_list(tokens, fast=True)
        print('Vocabulary loaded')

    def __str__(self) -> str:
        return self.stoi.__str__()

    def to_file(self, file: str):
        """
        Save the vocabulary to a file, by writing token with index i in line i.

        :param file: path to file where the vocabulary is written
        """
        with open(file, "w", encoding="utf-8") as open_file:
            for t in self.itos:
                open_file.write("{}\n".format(t))

    def add_tokens(self, tokens: List[str], fast: bool = False):
        """
        Add list of tokens to vocabulary

        :param tokens: list of tokens to add to the vocabulary
        :param fast: check duplicate check if true
        """
        for t in tokens:
            new_index = len(self.itos)
            # add to vocab if not already there
            if fast or t not in self.itos:
                self.itos.append(t)
                self.stoi[t] = new_index

    def is_unk(self, token: str) -> bool:
        """
        Check whether a token is covered by the vocabulary

        :param token:
        :return: True if covered, False otherwise
        """
        return self.stoi[token] == self.DEFAULT_UNK_ID()

    def __len__(self) -> int:
        return len(self.itos)


class TextVocabulary(Vocabulary):
    def __init__(self, tokens: List[str] = None, file: str = None, mBartVocab: bool = False, tokenizer=None, data_cfg=None):
        """
        Create vocabulary from list of tokens or file.

        Special tokens are added if not already in file or list.
        File format: token with index i is in line i.

        :param tokens: list of tokens
        :param file: file to load vocabulary from
        :param mBartVocab: if true, will use special tokens from the mBART vocabulary
        """
        super().__init__()
        #self.tokenizer = None

        if tokenizer is not None:
            vocab = {}
            if data_cfg["level"] == "sentencepiece":
                vocab_size = tokenizer.get_piece_size()
                for i in range(vocab_size):
                    vocab[tokenizer.id_to_piece(i)] = i
            else:
                vocab = tokenizer.get_vocab()
            assert len(vocab) != 0

            self.itos = list(dict(sorted(vocab.items(), key=lambda x: x[1])).keys())
            self.stoi = defaultdict()
            for i, token in enumerate(self.itos):
                self.stoi[token] = i
            self.specials = get_special_tokens(data_cfg=data_cfg, tokenizer=tokenizer)
            self.DEFAULT_UNK_ID = get_unk_token_id(data_cfg=data_cfg, tokenizer=tokenizer)
            #self.tokenizer = tokenizer
        else:
            if not mBartVocab:
                self.specials = [UNK_TOKEN(), PAD_TOKEN(), BOS_TOKEN(), EOS_TOKEN()]
            else:
                self.specials = []  # Special tokens already present in text vocabulary
                assert file is not None
            self.DEFAULT_UNK_ID = lambda: 0 if not mBartVocab else 3  # mBART <unk> is at 3
            self.stoi = defaultdict(self.DEFAULT_UNK_ID)

            if tokens is not None:
                self._from_list(tokens)
            elif file is not None:
                self._from_file(file)

    def array_to_sentence(self, array: np.array, cut_at_eos=True, tokens_to_remove=None, level=None, tokenizer=None) -> List[str]:
        """
        Converts an array of IDs to a sentence, optionally cutting the result
        off at the end-of-sequence token.

        :param array: 1D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of strings (tokens)
        """

        if tokenizer is not None:
            if level == "sentencepiece":
                sentence = tokenizer.DecodeIds([int(a) for a in array])
            else:
                sentence = tokenizer.decode(list(array))
            first_token_to_remove = len(sentence)
            if tokens_to_remove is not None:
                for token_to_remove in tokens_to_remove:
                    index = sentence.find(token_to_remove)
                    if index != -1 and first_token_to_remove > index:
                        first_token_to_remove = index
                if first_token_to_remove != len(sentence):
                    sentence = sentence[:first_token_to_remove]
        else:
            sentence = []
            for i in array:
                s = self.itos[i]
                if cut_at_eos and s == EOS_TOKEN:
                    break
                sentence.append(s)
        return sentence

    def arrays_to_sentences(self, arrays: np.array, cut_at_eos=True, tokens_to_remove=None, level=None, tokenizer=None) -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their
        sentences, optionally cutting them off at the end-of-sequence token.

        :param arrays: 2D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of list of strings (tokens)
        """
        sentences = []
        for array in arrays:
            sentences.append(self.array_to_sentence(array=array, cut_at_eos=cut_at_eos, tokens_to_remove=tokens_to_remove, level=level, tokenizer=tokenizer))
        return sentences


class GlossVocabulary(Vocabulary):
    def __init__(self, tokens: List[str] = None, file: str = None, data_cfg=None, tokenizer=None):
        """
        Create vocabulary from list of tokens or file.

        Special tokens are added if not already in file or list.
        File format: token with index i is in line i.

        :param tokens: list of tokens
        :param file: file to load vocabulary from
        """
        super().__init__()
        self.specials = [SIL_TOKEN(), UNK_TOKEN(), PAD_TOKEN()]
        self.DEFAULT_UNK_ID = get_unk_token_id(data_cfg=data_cfg, tokenizer=tokenizer)
        self.stoi = defaultdict()
        #self.stoi = defaultdict(self.DEFAULT_UNK_ID)

        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

        # TODO (Cihan): This bit is hardcoded so that the silence token
        #   is the first label to be able to do CTC calculations (decoding etc.)
        #   Might fix in the future.
        self.stoi = defaultdict()
        for i, token in enumerate(self.itos):
            self.stoi[token] = i
        assert self.stoi[SIL_TOKEN()] == 0

    def arrays_to_sentences(self, arrays: np.array) -> List[List[str]]:
        gloss_sequences = []
        for array in arrays:
            sequence = []
            for i in array:
                sequence.append(self.itos[i])
            gloss_sequences.append(sequence)
        return gloss_sequences


def filter_min(counter: Counter, minimum_freq: int):
    """ Filter counter by min frequency """
    filtered_counter = Counter({t: c for t, c in counter.items() if c >= minimum_freq})
    return filtered_counter


def sort_and_cut(counter: Counter, limit: int):
    """ Cut counter to most frequent,
    sorted numerically and alphabetically"""
    # sort by frequency, then alphabetically
    tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]
    return vocab_tokens


def build_vocab(
    field: str, max_size: int, min_freq: int, dataset: Dataset, vocab_file: str = None, mBartVocab: bool = False, bertTokenizer=None, data_cfg=None
) -> Vocabulary:
    """
    Builds vocabulary for a torchtext `field` from given`dataset` or
    `vocab_file`.

    :param field: attribute e.g. "src"
    :param max_size: maximum size of vocabulary
    :param min_freq: minimum frequency for an item to be included
    :param dataset: dataset to load data for field from
    :param vocab_file: file to store the vocabulary,
        if not None, load vocabulary from here
    :param mBartVocab: if true, will use special tokens from the mBART vocabulary
    :return: Vocabulary created from either `dataset` or `vocab_file`
    """
    if bertTokenizer is not None and field == "txt":
        vocab = TextVocabulary(file=vocab_file, mBartVocab=mBartVocab, tokenizer=bertTokenizer, data_cfg=data_cfg)
    else:
        if vocab_file is not None:
            # load it from file
            if field == "gls":
                vocab = GlossVocabulary(file=vocab_file, data_cfg=data_cfg, tokenizer=bertTokenizer)
            elif field == "txt":
                vocab = TextVocabulary(file=vocab_file, mBartVocab=mBartVocab,  data_cfg=data_cfg, tokenizer=bertTokenizer)
            else:
                raise ValueError("Unknown vocabulary type")
        else:
            tokens = []
            for i in dataset.examples:
                if field == "gls":
                    tokens.extend(i.gls)
                elif field == "txt":
                    tokens.extend(i.txt)
                else:
                    raise ValueError("Unknown field type")

            counter = Counter(tokens)
            if min_freq > -1:
                counter = filter_min(counter, min_freq)
            vocab_tokens = sort_and_cut(counter, max_size)
            assert len(vocab_tokens) <= max_size

            if field == "gls":
                vocab = GlossVocabulary(tokens=vocab_tokens, data_cfg=data_cfg, tokenizer=bertTokenizer)
            elif field == "txt":
                vocab = TextVocabulary(tokens=vocab_tokens, mBartVocab=mBartVocab, data_cfg=data_cfg, tokenizer=bertTokenizer)
            else:
                raise ValueError("Unknown vocabulary type")

            assert len(vocab) <= max_size + len(vocab.specials)
            #assert vocab.itos[vocab.DEFAULT_UNK_ID()] == UNK_TOKEN()
        """
        for i, s in enumerate(vocab.specials):
            if i != vocab.DEFAULT_UNK_ID():
                assert not vocab.is_unk(s)
        """
    return vocab
