import re
from typing import List

import torch
from torch.utils.data.dataset import Dataset, T_co

from src.data.Vocab import Vocab


class CoNLLDataset(Dataset):
    def __init__(self, padding_size: int, dataset_path: str):
        self.words: List[List[str]] = []
        self.pos: List[List[str]] = []
        self.ner_labels: List[List[str]] = []
        self.parse_dataset(dataset_path)
        self.len_dataset: int = len(self.words)
        self.vocab_words: Vocab = self.build_vocab(self.words)
        self.vocab_pos: Vocab = self.build_vocab(self.pos)
        self.vocab_label_ner: Vocab = self.build_vocab(self.ner_labels, is_label=True)
        self.padding_size = padding_size
        self.dataset = self.build_dataset()
        print("LEN VOCAB WORDS", len(self.vocab_words))
        print("LEN VOCAB vocab", len(self.vocab_label_ner))

    def build_dataset(self) -> List[dict]:
        result = []
        for id_sentence in range(self.len_dataset):
            encoded_words = self.encode_tokens(self.words[id_sentence], self.vocab_words)
            encoded_pos = self.encode_tokens(self.pos[id_sentence], self.vocab_pos)
            encoded_ner_labels = self.encode_tokens(self.ner_labels[id_sentence], self.vocab_label_ner)
            sample = {
                "id_sentence": id_sentence,
                "words": encoded_words,
                "pos": encoded_pos,
                "labels": encoded_ner_labels,
            }
            result.append(sample)
        return result

    def __getitem__(self, index) -> T_co:
        return self.dataset[index]

    def __len__(self):
        return self.len_dataset

    def build_vocab(
        self,
        token_list: List[List[str]],
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        freq_to_drop: int = 0,
        is_label: bool = False,
    ) -> Vocab:
        """
        build a vocabulary for the roles label vector
        """
        vocab = Vocab(pad_token=pad_token, unk_token=unk_token, is_label=is_label)
        for sentence_id in range(self.len_dataset):
            for token in token_list[sentence_id]:
                if token != pad_token:
                    vocab.add_token(token)
        vocab.drop_frequency(freq_to_drop=freq_to_drop)
        return vocab

    def encode_tokens(self, sentence: List[str], vocab: Vocab) -> torch.LongTensor:
        result: List[int] = [
            vocab.token_to_id(token) if vocab.is_present(token) else vocab.unk_id for token in sentence
        ]
        len_sentence = len(sentence)
        if len_sentence < self.padding_size:
            padding: List[int] = [vocab.pad_id] * (self.padding_size - len_sentence)
            result.extend(padding)
        return torch.LongTensor(result)

    def parse_dataset(self, dataset_path):
        dataset = []
        with open(dataset_path) as f:
            next(f)
            next(f)
            raw_text = f.read()
            sentences: List[str] = raw_text.split("\n\n")
            sentences: List[List[str]] = [sentence.split("\n") for sentence in sentences][:-1]
            # sentence:List[List[List[str]]] = [row.split(" ") for sentence in sentences for row in sentence]

            for sent in sentences:
                words = []
                pos_tags = []
                ner_labels = []
                for row in sent:
                    (word, pos_tag, _, ner_label) = row.split(" ")
                    words.append(word)
                    pos_tags.append(pos_tag)
                    ner_labels.append(ner_label)
                self.words.append(words)
                self.pos.append(pos_tags)
                self.ner_labels.append(ner_labels)

    def process_list_literal(self, literal):
        literal = self.remove_brackets(literal)
        literal = self.trim_literal(literal)
        return literal

    def parse_label_list(self, literal):
        literal = self.process_list_literal(literal)
        return self.parse_list(literal)

    def parse_tokens_list(self, literal):
        literal = self.process_list_literal(literal)
        return self.parse_tokens(literal)

    def trim_literal(self, literal):
        return re.sub("\\s+", " ", literal).strip()

    def remove_brackets(self, literal):
        return literal.replace("[", "").replace("]", "")

    def parse_list(self, list_literal):
        return [int(label) for label in list_literal.split(" ")]

    def parse_tokens(self, tokens_literal):
        # slicing is to remove ' ' around tokens
        return [tok[1:-1] for tok in tokens_literal.split(" ")]
