from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import numpy as np

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import ast
import os
import wandb
import time
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.global_parameters import ANTISEMITIC_PREFIXES


class BinaryClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class AntisemitismClassifier:
    def __init__(self):
        self.model = None
        self.train_classifier()

    def extract_bin_categories(self, category_list, as_categories):
        count_as = 0
        for cat in category_list:
            for as_c in as_categories:
                if cat.startswith(as_c):
                    count_as += 1
        return 1 if count_as > len(category_list) / 2 else 0

    def binary_preprocess(self, new_df):
        # הכנת דאטה
        def safe_literal_eval(val):
            if isinstance(val, str):
                return ast.literal_eval(val)
            return val

        new_df["extracted_subcategories"] = new_df["extracted_subcategories"].apply(safe_literal_eval)

        new_df["binary_label_strict"] = new_df["extracted_subcategories"].apply(
            lambda cats: self.extract_bin_categories(cats, ANTISEMITIC_PREFIXES)
        )

        texts = new_df["clean_extracted_text"].tolist()
        labels = new_df["binary_label_strict"].tolist()

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)

        acc = accuracy_score(labels, preds)
        precision_micro = precision_score(labels, preds, average='micro', zero_division=0)
        recall_micro = recall_score(labels, preds, average='micro', zero_division=0)
        f1_micro = f1_score(labels, preds, average='micro', zero_division=0)

        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)

        return {
            'accuracy': acc,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro
        }

    def train_classifier(self):
        train_dataset = BinaryClassificationDataset(train_texts, train_labels, tokenizer)
        val_dataset = BinaryClassificationDataset(val_texts, val_labels, tokenizer)

        model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)

        training_args = TrainingArguments(
            output_dir="./bert_binary_classifier",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
