from pathlib import Path

from transformers import Trainer, AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
import ast
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments

from transformers import AutoModel
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

from models.binary_dataset import BinaryClassificationDataset
from src.global_parameters import ANTISEMITIC_PREFIXES

ROOT_PATH = Path(__file__).parent.parent.parent
RESOURCE_PATH = ROOT_PATH / "resources"


def extract_bin_categories(category_list, as_categories):
    count_as = 0
    for cat in category_list:
        for as_c in as_categories:
            if cat.startswith(as_c):
                count_as += 1
    return 1 if count_as > len(category_list) / 2 else 0


def binary_preprocess(combined_df_cleaned):
    def safe_literal_eval(val):
        if isinstance(val, str):
            return ast.literal_eval(val)
        return val

    combined_df_cleaned["extracted_subcategories"] = combined_df_cleaned["extracted_subcategories"].apply(
        safe_literal_eval)

    combined_df_cleaned["binary_label_strict"] = combined_df_cleaned["extracted_subcategories"].apply(
        lambda cats: extract_bin_categories(cats, ANTISEMITIC_PREFIXES)
    )

    texts = combined_df_cleaned["clean_extracted_text"].tolist()
    labels = combined_df_cleaned["binary_label_strict"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )


def compute_metrics(eval_pred):
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


def compute_metrics_binary(eval_pred):
    import torch
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


class WeightedBinaryClassifier(nn.Module):
    def __init__(self, base_model_name, pos_weight):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :]).squeeze(-1)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits.unsqueeze(-1),
        )


class BinaryAntisemitismClassifier:
    def __init__(self, texts, labels, tokenizer, max_length=128, train_texts=None,
                 train_labels=None, val_texts=None, val_labels=None):

        self.train_texts = train_texts
        self.train_labels = train_labels
        self.val_texts = val_texts
        self.val_labels = val_labels
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = torch.tensor(labels, dtype=torch.long)

        self.train_dataset = BinaryClassificationDataset(train_texts, train_labels, tokenizer)
        self.val_dataset = BinaryClassificationDataset(val_texts, val_labels, tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        assert set(train_labels).issubset({0, 1})
        assert set(val_labels).issubset({0, 1})

        # Calculating pos_weight according to class distribution
        neg = (torch.tensor(train_labels) == 0).sum()
        pos = (torch.tensor(train_labels) == 1).sum()
        self.pos_weight = torch.tensor([neg / pos]).to("cuda")
        self.model = WeightedBinaryClassifier("cardiffnlp/twitter-roberta-base-sentiment", self.pos_weight)

        self.trainer = None
        self.train_classifier()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def train_classifier(self):
        training_args = TrainingArguments(
            output_dir="./bert_binary_classifier",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            logging_dir="./logs",
            logging_steps=10,
            report_to="none"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics_binary,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        self.trainer.train()

    def predict(self):
        # prediction in validation set
        predictions = self.trainer.predict(self.val_dataset)

        logits = predictions.predictions
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs >= 0.5).astype(int).flatten()

        labels = predictions.label_ids

        # Classification report
        report = classification_report(labels, preds, target_names=["Negative", "Positive"])
        print(report)

        # Save model
        model_path = RESOURCE_PATH / "binary_model_weighted"
        self.trainer.save_model(str(model_path))
        self.tokenizer.save_pretrained(model_path)
