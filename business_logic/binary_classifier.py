from transformers import Trainer, TrainingArguments
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import ast
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from business_logic.preprocessing import preprocess_dataframe
from models.binary_dataset import BinaryClassificationDataset
from src.global_parameters import ANTISEMITIC_PREFIXES


def extract_bin_categories(category_list, as_categories):
    count_as = 0
    for cat in category_list:
        for as_c in as_categories:
            if cat.startswith(as_c):
                count_as += 1
    return 1 if count_as > len(category_list) / 2 else 0


def binary_preprocess(combined_df_cleaned):
    # הכנת דאטה
    def safe_literal_eval(val):
        if isinstance(val, str):
            return ast.literal_eval(val)
        return val

    combined_df_cleaned["extracted_subcategories"] = combined_df_cleaned["extracted_subcategories"].apply(safe_literal_eval)

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


class BinaryAntisemitismClassifier:
    def __init__(self, train_texts, train_labels, val_texts, val_labels):
        self.model = None
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.val_texts = val_texts
        self.val_labels = val_labels
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

        self.combined_df_cleaned, self.categories = preprocess_dataframe()

        self.train_classifier()

    def train_classifier(self):
        train_dataset = BinaryClassificationDataset(self.train_texts, self.train_labels, self.tokenizer)
        val_dataset = BinaryClassificationDataset(self.val_texts, self.val_labels, self.tokenizer)

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
            compute_metrics=compute_metrics
        )

        trainer.train()

        # ניבוי על סט האימות
        predictions = trainer.predict(val_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids

        # טבלת ביצועים
        report = classification_report(labels, preds, digits=4)
        print(report)

        model.save_pretrained("saved_model_bert_binary")
        self.tokenizer.save_pretrained("saved_model_bert_binary")

        trainer.model.save_pretrained("best_model_binary")
        self.tokenizer.save_pretrained("best_model_binary")

        model.save_pretrained("/content/drive/MyDrive/best_model_binary2204")
        self.tokenizer.save_pretrained("/content/drive/MyDrive/best_model_binary2204")

        # חישוב המטריצה
        cm = confusion_matrix(labels, preds)

        # ציור
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Antisemitic', 'Antisemitic'],
                    yticklabels=['Not Antisemitic', 'Antisemitic'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def create_df_peds(self):
        # טוקניזציה מהירה לכל הדאטה
        tokenizer = BertTokenizer.from_pretrained("saved_model_bert_binary")  # או התיקייה שבה שמרת את המודל
        model = BertForSequenceClassification.from_pretrained("saved_model_bert_binary")
        model.eval()

        # texts: רשימת טקסטים מדויקים מתוך df (clean_extracted_text)
        texts = self.combined_df_cleaned["clean_extracted_text"].tolist()

        # יצירת predictions על כל הדאטה
        all_preds = []
        batch_size = 32

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # שלבי טוקניזציה
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

                inputs = {key: val.to(device) for key, val in inputs.items()}

                # הרצה
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().tolist())  # חשוב להחזיר ל-CPU כדי לא לקרוס ב-pandas

        # הוספת עמודת תחזית לדאטה המקורי
        self.combined_df_cleaned["binary_prediction"] = all_preds

        # סינון רשומות אנטישמיות לפי המודל
        df_predicted_antisemitic = self.combined_df_cleaned[self.combined_df_cleaned["binary_prediction"] == 1].copy()
        df_predicted_antisemitic.to_csv('df_predicted_antisemitic.csv', encoding='utf-8-sig', index=False)
        return df_predicted_antisemitic
