import json
import torch

from torch.utils.data import Dataset
from transformers import LayoutLMTokenizer


class SROIEDataset(Dataset):

    def __init__(self, jsonl_path, label2id):

        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.tokenizer = LayoutLMTokenizer.from_pretrained(
            "microsoft/layoutlm-base-uncased"
        )

        self.label2id = label2id
        self.max_length = 512

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        words = sample["words"]
        boxes = sample["boxes"]
        labels = sample["ner_tags"]

        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        word_ids = tokenized.word_ids()

        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            else:
                label_ids.append(
                    self.label2id[labels[word_idx]]
                )

        bbox = []

        for word_idx in word_ids:

            if word_idx is None:
                bbox.append([0, 0, 0, 0])

            else:
                bbox.append(boxes[word_idx])

        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "bbox": torch.tensor(bbox),
            "labels": torch.tensor(label_ids)
        }