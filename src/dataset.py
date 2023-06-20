from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
import pandas as pd


class TweetDataset(Dataset):
    def __init__(self, filename, maxlen):
        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter=",")

        self.classes = ["negative", "neutral", "positive"]
        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, "text"]
        label = torch.tensor(
            self.classes.index(self.df.loc[index, "sentiment"]), dtype=torch.long
        )

        # Preprocessing the text to be suitable for BERT
        input_ids = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.maxlen,
        )["input_ids"][0]
        attention_mask = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.maxlen,
        )["attention_mask"][0]

        return {
            "sentence": sentence,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
        }
