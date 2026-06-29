import json
import numpy as np
import torch

from torch.utils.data import Dataset

from transformers import (
    LayoutLMTokenizerFast,
    LayoutLMForTokenClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import precision_recall_fscore_support


# ==========================================================
# LABELS
# ==========================================================

LABELS = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-DATE",
    "I-DATE",
    "B-TOTAL",
    "I-TOTAL"
]

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

MAX_LENGTH = 512


# ==========================================================
# DATASET
# ==========================================================

class SROIEDataset(Dataset):

    def __init__(self, jsonl_file):

        self.samples = []

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.tokenizer = LayoutLMTokenizerFast.from_pretrained(
            "microsoft/layoutlm-base-uncased"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        words = sample["words"]
        boxes = sample["boxes"]
        labels = sample["ner_tags"]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)

        label_ids = []
        bbox = []

        for word_idx in word_ids:

            if word_idx is None:

                label_ids.append(-100)
                bbox.append([0, 0, 0, 0])

            else:

                label_ids.append(
                    label2id[labels[word_idx]]
                )

                bbox.append(
                    boxes[word_idx]
                )

        item = {
            "input_ids":
                encoding["input_ids"].squeeze(),

            "attention_mask":
                encoding["attention_mask"].squeeze(),

            "bbox":
                torch.tensor(bbox, dtype=torch.long),

            "labels":
                torch.tensor(label_ids, dtype=torch.long)
        }

        return item


# ==========================================================
# METRICS
# ==========================================================

def compute_metrics(pred):

    predictions = pred.predictions.argmax(-1)
    labels = pred.label_ids

    true_labels = []
    true_predictions = []

    for prediction, label in zip(predictions, labels):

        for p, l in zip(prediction, label):

            if l != -100:
                true_predictions.append(p)
                true_labels.append(l)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        true_predictions,
        average="weighted",
        zero_division=0
    )

    accuracy = (
        np.array(true_predictions)
        ==
        np.array(true_labels)
    ).mean()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading datasets...")

train_dataset = SROIEDataset(
    "processed/sroie/train.jsonl"
)

test_dataset = SROIEDataset(
    "processed/sroie/test.jsonl"
)

print("Train size:", len(train_dataset))
print("Test size :", len(test_dataset))


# ==========================================================
# MODEL
# ==========================================================

print("Loading LayoutLM model...")

model = LayoutLMForTokenClassification.from_pretrained(
    "microsoft/layoutlm-base-uncased",
    num_labels=len(LABELS),
    label2id=label2id,
    id2label=id2label
)


# ==========================================================
# TRAINING ARGS
# ==========================================================

training_args = TrainingArguments(
    output_dir="./layoutlm_invoice_model",

    num_train_epochs=10,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    learning_rate=5e-5,

    weight_decay=0.01,

    logging_steps=20,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    save_total_limit=2,

    report_to="none"
)


# ==========================================================
# TRAINER
# ==========================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# ==========================================================
# TRAIN
# ==========================================================

print("\nStarting training...\n")

trainer.train()

print("\nTraining complete!\n")


# ==========================================================
# EVALUATE
# ==========================================================

results = trainer.evaluate()

print("\nFinal Evaluation Results:")
print(results)


# ==========================================================
# SAVE MODEL
# ==========================================================

trainer.save_model("./layoutlm_invoice_model")

print("\nModel saved successfully.")