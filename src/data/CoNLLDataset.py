from torch.utils.data.dataset import Dataset, T_co
import csv
import re


class CoNLLDataset(Dataset):

    def __init__(self, dataset_path):
        self.dataset = self.parse_dataset(dataset_path)
        print()

    def __getitem__(self, index) -> T_co:
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def parse_dataset(self, dataset_path):
        dataset = []
        with open(dataset_path) as f:
            # creating a csv reader object
            csvreader = csv.reader(f)

            # extracting field names through first row
            next(csvreader)

            # extracting each data row one by one
            for i, row in enumerate(csvreader):
                ner_tags = self.parse_label_list(row[3])
                pos_tags = self.parse_label_list(row[4])
                tokens = self.parse_tokens_list(row[5])
                dataset.append({'tokens': tokens, 'pos_tags': pos_tags, 'ner_tags': ner_tags})
        return dataset

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
        return literal.replace("[", "").replace("]","")

    def parse_list(self, list_literal):
        return [int(label) for label in list_literal.split(" ")]

    def parse_tokens(self, tokens_literal):
        # slicing is to remove ' ' around tokens
        return [tok[1:-1] for tok in tokens_literal.split(" ")]
