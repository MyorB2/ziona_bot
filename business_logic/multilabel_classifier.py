import warnings

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from sklearn.utils import resample
import torch, numpy as np, pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
import joblib, os

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")
# Ensure device is set to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics_multilabel(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.3).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0)
    }


class AntisemitismClassifier:
    def __init__(self):
        self.mlb = None
        self.val_labels = None
        self.train_labels = None
        self.val_texts = None
        self.train_texts = None
        self.val_dataset_aug = None
        self.train_dataset_aug = None
        self.val_dataset = None
        self.train_dataset = None
        self.model = None
        self.df = pd.read_csv('/content/df_predicted_antisemitic.csv')
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.category_classes = None
        self.train_classifier()

    def multi_preprocess(self):
        # --- שלב 1: הכנת דאטה ---
        # df = df_predicted_antisemitic.copy()
        self.df["extracted_categories"] = self.df["extracted_categories"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        texts = self.df["clean_extracted_text"].tolist()
        labels = self.df["extracted_categories"].tolist()

        cleaned_data = [(t, l) for t, l in zip(texts, labels) if
                        isinstance(l, list) and all(isinstance(x, str) for x in l)]
        texts, labels = zip(*cleaned_data)

        texts = list(texts)
        labels = list(labels)

        self.mlb = MultiLabelBinarizer()
        labels_bin = self.mlb.fit_transform(labels)
        self.category_classes = self.mlb.classes_

        save_path = "/content/drive/MyDrive/Models_2204"
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(self.mlb, f"{save_path}/saved_self.mlb.pkl")
        print("self.mlb saved to:", f"{save_path}/saved_self.mlb.pkl")

        # --- Oversampling ---
        flat_labels = [label for sublist in labels for label in sublist]
        label_counts = Counter(flat_labels)
        rare_labels = [label for label, count in label_counts.items() if count < 500]

        df_ml = pd.DataFrame({"text": texts, "labels": labels})
        df_rare = df_ml[df_ml["labels"].apply(lambda x: any(label in rare_labels for label in x))]

        df_oversampled = pd.concat([
            df_ml,
            resample(df_rare, replace=True, n_samples=len(df_rare) * 3, random_state=42)
        ])

        # --- ניקוי נוסף אחרי האיחוד ---
        df_oversampled = df_oversampled[
            df_oversampled["labels"].apply(lambda l: isinstance(l, list) and all(isinstance(x, str) for x in l))]

        texts = df_oversampled["text"].tolist()
        labels = df_oversampled["labels"].tolist()
        labels_bin = self.mlb.transform(labels)

        # --- שלב 2: טקסטים עם Hate Score + לוגיטים מ-HateBERT ---
        hate_tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")
        hate_model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english").to(
            "cuda")
        hate_model.eval()

        hate_scores = []
        texts_augmented = []
        hate_logits_all = []

        for text in tqdm(texts):
            inputs = hate_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cuda")
            with torch.no_grad():
                outputs = hate_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                hate_logits_all.append(outputs.logits.detach().cpu().numpy())
                score = probs[0][1].item()
                texts_augmented.append(f"[HATE_SCORE={round(score, 2)}] {text}")

        hate_logits_all = np.vstack(hate_logits_all)

        stratify_labels = labels_bin.argmax(axis=1)

        self.train_texts, self.val_texts, self.train_labels, self.val_labels = train_test_split(
            texts, labels_bin, test_size=0.2, stratify=stratify_labels, random_state=42
        )

        train_texts_aug, val_texts_aug = train_test_split(texts_augmented, test_size=0.2, random_state=42)
        hate_logits_train, hate_logits_val = train_test_split(hate_logits_all, test_size=0.2, random_state=42)

        self.train_dataset = MultiLabelDataset(self.train_texts, self.train_labels, self.tokenizer)
        self.val_dataset = MultiLabelDataset(self.val_texts, self.val_labels, self.tokenizer)
        self.train_dataset_aug = MultiLabelDataset(train_texts_aug, self.train_labels, self.tokenizer)
        self.val_dataset_aug = MultiLabelDataset(val_texts_aug, self.val_labels, self.tokenizer)

    def train_classifier(self):
        try:
            print("Start training AntisemitismClassifier model")
            model1 = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base",
                                                                        num_labels=len(self.category_classes),
                                                                        problem_type="multi_label_classification").to(
                "cuda")
            model2 = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base",
                                                                        num_labels=len(self.category_classes),
                                                                        problem_type="multi_label_classification").to(
                "cuda")

            training_args = TrainingArguments(
                output_dir="./ensemble_models",
                eval_strategy="epoch",
                save_strategy="epoch",
                num_train_epochs=1,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                learning_rate=2e-5,
                weight_decay=0.01,
                logging_dir="./logs_ensemble",
                logging_steps=10,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="none"
            )

            trainer1 = CombinedLossTrainer(model=model1, args=training_args, train_dataset=self.train_dataset,
                                           eval_dataset=self.val_dataset, compute_metrics=compute_metrics_multilabel)
            trainer2 = CombinedLossTrainer(model=model2, args=training_args, train_dataset=self.train_dataset_aug,
                                           eval_dataset=self.val_dataset_aug, compute_metrics=compute_metrics_multilabel)
            # trainer3 = CombinedLossTrainer(model=model3, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=compute_metrics_multilabel)

            trainer1.train()
            trainer2.train()
            # trainer3.train()

            # --- שלב 5: שילוב התחזיות ---
            pred1 = trainer1.predict(self.val_dataset).predictions
            pred2 = trainer2.predict(self.val_dataset_aug).predictions
            # pred3 = trainer3.predict(val_dataset).predictions

            avg_logits = (torch.tensor(pred1) + torch.tensor(pred2)) / 2
            preds = (torch.sigmoid(avg_logits) > 0.3).int().numpy()
            true_labels = self.val_labels

            print(classification_report(true_labels, preds, target_names=self.mlb.classes_))

            trainer1.model.save_pretrained("saved_model_multilabel_1")
            trainer2.model.save_pretrained("saved_model_multilabel_2")
            # trainer3.model.save_pretrained("saved_model_multilabel_3")

            # שמירת הטוקנייזר התואם (משותף לשלושתם)
            self.tokenizer.save_pretrained("saved_model_multilabel_1")

            # --- Save in Drive ---
            trainer1.model.save_pretrained("/content/drive/MyDrive/Models_2204/saved_model_multilabel_1")
            trainer2.model.save_pretrained("/content/drive/MyDrive/Models_2204/saved_model_multilabel_2")
            # trainer3.model.save_pretrained("/content/drive/MyDrive/saved_model_multilabel_3")

            self.tokenizer.save_pretrained("/content/drive/MyDrive/Models_2204/saved_model_multilabel_1")
        except Exception as e:
            raise Exception(f"Error while training model: {e}")

    def predict(self, text):
        # --- טען מודלים שמורים ---
        binary_tokenizer = BertTokenizer.from_pretrained("/content/drive/MyDrive/Models_2204/best_model_binary2204")
        binary_model = BertForSequenceClassification.from_pretrained(
            "/content/drive/MyDrive/Models_2204/best_model_binary2204")
        binary_model.eval().to("cuda")

        multi_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        multi_model1 = AutoModelForSequenceClassification.from_pretrained(
            "/content/drive/MyDrive/Models_2204/saved_model_multilabel_1").to("cuda")
        multi_model2 = AutoModelForSequenceClassification.from_pretrained(
            "/content/drive/MyDrive/Models_2204/saved_model_multilabel_2").to("cuda")
        # multi_model3 = AutoModelForSequenceClassification.from_pretrained("saved_model_multilabel_3").to("cuda")

        #  שלב 1: ניבוי בינארי
        binary_inputs = binary_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(
            "cuda")
        with torch.no_grad():
            logits = binary_model(**binary_inputs).logits
            probs = torch.softmax(logits, dim=1)
            pred_bin = torch.argmax(probs).item()

        if pred_bin == 0:
            return {"binary_prediction": "Not Antisemitic", "categories": []}

        #  שלב 2: ניבוי קטגוריות אם אנטישמי
        multi_inputs = multi_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(
            "cuda")
        with torch.no_grad():
            logits1 = multi_model1(**multi_inputs).logits
            logits2 = multi_model2(**multi_inputs).logits
            # logits3 = multi_model3(**multi_inputs).logits

        avg_logits = (logits1 + logits2) / 2
        probs = torch.sigmoid(avg_logits).cpu().numpy()[0]

        #  מיפוי לקטגוריות
        threshold = 0.3
        pred_indices = np.where(probs > threshold)[0]
        pred_categories = self.mlb.classes_[pred_indices]

        return {
            "binary_prediction": "Antisemitic",
            "categories": pred_categories.tolist()[0]
        }
