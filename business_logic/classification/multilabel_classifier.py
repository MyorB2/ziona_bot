import os
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, hamming_loss, classification_report,
    precision_recall_curve
)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from textblob import TextBlob
from torch.nn.functional import sigmoid
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from models.custom_loss_model import CustomLossModel
from models.multi_dataset import MultiLabelDataset
from models.save_model_callback import SaveBestModelCallback

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

ROOT_PATH = Path(__file__).parent.parent.parent
RESOURCE_PATH = ROOT_PATH / "resources"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultilabelClassifier(Dataset):
    def __init__(self, texts, labels, tokenizer, dataset_path):
        self.X_meta_scaled = None
        self.labels_bin = None
        self.texts = None
        self.labels_val = None
        self.labels_train = None
        self.texts_val = None
        self.texts_train = None
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels
        self.df = pd.read_csv(dataset_path)
        self.df['updated_mapped_categories'] = self.df['updated_mapped_categories'].apply(self.clean_category_list)
        self.filtered_df = self.df[self.df["binary_category"] == 1][["clean_text", "updated_mapped_categories"]].reset_index(drop=True)
        self.filtered_df['category_count'] = self.filtered_df['updated_mapped_categories'].apply(
            lambda x: len(x) if isinstance(x, list) else 0)
        self.save_model_dir = "/content/drive/MyDrive/Project/Models/Multi_model/1805_1/final_multilabel_model"
        self.root_save_dir = "/content/drive/MyDrive/Project/Models/Multi_model/1805_1"
        self.trainer = None
        self.mlb = MultiLabelBinarizer()
        self.mlb_path = "/content/drive/MyDrive/Project/Models/Multi_model/1805_1/saved_mlb.pkl"
        self.tokenizers = {}
        self.models = {}

        self.train()

    @staticmethod
    def clean_category_list(cat_list):
        if not isinstance(cat_list, list):
            return cat_list
        # Remove duplicates
        cat_list = list(set(cat_list))
        # Remove [0] category
        if 0 in cat_list and len(cat_list) > 1:
            cat_list = [x for x in cat_list if x != 0]
        return sorted(cat_list)

    @staticmethod
    def compute_metrics_multi(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds_bin = (probs >= 0.5).astype(int)

        return {
            "f1_micro": f1_score(labels, preds_bin, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, preds_bin, average="macro", zero_division=0),
            "precision_micro": precision_score(labels, preds_bin, average="micro", zero_division=0),
            "recall_micro": recall_score(labels, preds_bin, average="micro", zero_division=0),
            "accuracy_samples": accuracy_score(labels, preds_bin),
            "hamming_loss": hamming_loss(labels, preds_bin)
        }

    def prepare_multilabel_data(self, df, label_column="updated_mapped_categories", text_column="clean_text",
                                label_to_remove=0, oversample_factor=1.5, rare_thresh_ratio=0.10,
                                mlb_save_path="./saved_mlb.pkl"):

        category_mapping = {
            1: "Antisemitic Ideology",
            2: "Stereotypes and Dehumanization",
            3: "Antisemitism Against Israel/Zionism",
            4: "Holocaust Denial/Distortion",
            5: "Fundamental/Modern Antisemitism"
        }

        texts = df[text_column].tolist()
        labels = [[category_mapping[l] for l in label_list if l != label_to_remove]
                  for label_list in df[label_column]]

        # one-hot vector
        labels_bin_full = self.mlb.fit_transform(labels)
        classes = list(self.mlb.classes_)
        print("Labels after filtering:", classes)

        # save mlb
        joblib.dump(self.mlb, mlb_save_path)
        print(f"Saved updated MLB to: {mlb_save_path}")

        # DataFrame
        df_aug = pd.DataFrame({"text": texts, "labels": labels})

        # Calculate rare labels
        flat_labels = [label for sublist in labels for label in sublist]
        label_counts = Counter(flat_labels)
        threshold = len(labels) * rare_thresh_ratio
        rare_labels = [label for label, count in label_counts.items() if count < threshold]
        print(f"Rare labels (under {rare_thresh_ratio * 100:.0f}%): {rare_labels}")

        # Oversampling for rare labels
        if rare_labels:
            df_rare = df_aug[df_aug["labels"].apply(lambda x: any(label in rare_labels for label in x))]
            df_rare_aug = df_rare.copy()
            df_rare_aug["text"] += " [AUG]"
            df_aug = pd.concat([
                df_aug,
                resample(df_rare_aug, replace=True, n_samples=int(len(df_rare) * oversample_factor), random_state=42)
            ])
            df_aug = df_aug.sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"After oversampling: {len(df_aug)} samples")
        else:
            print("No rare labels found. Skipping oversampling.")

        final_texts = df_aug["text"].tolist()
        final_labels = df_aug["labels"].tolist()
        final_labels_bin = self.mlb.transform(final_labels)

        return final_texts, final_labels, final_labels_bin, classes

    @staticmethod
    def check_label_distribution(labels):
        flat_labels_post = [label for sublist in labels for label in sublist]
        label_counts_post = Counter(flat_labels_post)
        total_post = len(labels)

        print("\nDistribution after oversampling:")
        for label, count in label_counts_post.items():
            print(f"Label {label}: {count} samples, ratio: {count / total_post:.2%}")

    def compute_and_save_thresholds_and_report(self, trainer, val_dataset, save_dir, file_prefix=""):
        os.makedirs(save_dir, exist_ok=True)
        mlb = joblib.load(self.mlb_path)

        # predict on validation set
        outputs = trainer.predict(val_dataset)
        logits = outputs.predictions
        labels_val = outputs.label_ids
        probs = sigmoid(torch.tensor(logits)).numpy()

        # opt threshold
        def get_optimal_thresholds_by_f1(y_true, y_probs):
            thresholds = []
            for i in range(y_true.shape[1]):
                precision, recall, thr = precision_recall_curve(y_true[:, i], y_probs[:, i])
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_thr = thr[np.argmax(f1)] if len(thr) > 0 else 0.5
                thresholds.append(best_thr)
            return np.array(thresholds)

        optimal_thresholds = get_optimal_thresholds_by_f1(labels_val, probs)

        # binary predictions
        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        thresholds_tensor = torch.tensor(optimal_thresholds, dtype=torch.float32)
        preds_bin = (probs_tensor >= thresholds_tensor).int().numpy()

        # classification report
        target_names = [str(c) for c in self.mlb.classes_]
        metrics_report = classification_report(labels_val, preds_bin, target_names=target_names, digits=4)
        print(metrics_report)

        # saving
        thresholds_path = os.path.join(save_dir, f"{file_prefix}optimal_thresholds.pkl")
        report_path = os.path.join(save_dir, f"{file_prefix}classification_report.txt")

        joblib.dump(optimal_thresholds, thresholds_path)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(metrics_report)

        print("Successfully saved thresholds and reports")
        print(f"Thresholds: {thresholds_path}")
        print(f"ReportsL {report_path}")

        return optimal_thresholds, metrics_report

    def save_custom_model(self, model, tokenizer, pos_weight, save_dir):
        mlb = joblib.load(self.mlb_path)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(save_dir, "custom_model_weights.pt"))

        tokenizer.save_pretrained(save_dir)

        config = model.base_model.config if hasattr(model, "base_model") else model.config
        config.save_pretrained(save_dir)

        joblib.dump({
            "pos_weight": pos_weight,
            "mlb": mlb,
            "category_classes": list(mlb.classes_)
        }, os.path.join(save_dir, "extras.pkl"))

        print("Model, tokenizer, config saved successfully.")

    @staticmethod
    def load_custom_model(save_dir):
        print(f"Loading model from: {save_dir}")
        tokenizer = AutoTokenizer.from_pretrained(save_dir, local_files_only=True)
        config = AutoConfig.from_pretrained(save_dir, local_files_only=True)

        extras_path = os.path.join(save_dir, "extras.pkl")
        extras = joblib.load(extras_path)
        mlb = extras["mlb"]
        pos_weight = torch.tensor(extras["pos_weight"], dtype=torch.float32)

        config.num_labels = len(mlb.classes_)
        config.problem_type = "multi_label_classification"

        base_model = AutoModelForSequenceClassification.from_config(config)
        model = CustomLossModel(base_model, pos_weight)
        model_weights_path = os.path.join(save_dir, "custom_model_weights.pt")
        model.load_state_dict(torch.load(model_weights_path, map_location="cuda"))

        print("Model, tokenizer, and extras loaded successfully.")
        return model.to("cuda"), tokenizer, mlb

    @staticmethod
    def get_logits(txt, tokenizer, model):
        inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            return torch.sigmoid(model(**inputs).logits).cpu().numpy()

    def build_x_meta(self, models, tokenizers):
        logits_deberta_all, logits_twitter_all, hate_scores, text_lengths, sentiments = [], [], [], [], []

        for text in tqdm(self.texts):
            # Hate score
            hate_inputs = tokenizers["hatebert"](text, return_tensors="pt", truncation=True, padding=True)
            hate_inputs = {k: v.to(device) for k, v in hate_inputs.items()}
            with torch.no_grad():
                hate_prob = torch.softmax(models["hatebert"](**hate_inputs).logits, dim=1)[0][1].item()
            hate_scores.append([hate_prob])

            logits_deberta = self.get_logits(text, tokenizers["deberta"], models["deberta"])
            logits_twitter = self.get_logits(text, tokenizers["twitter"], models["twitter"])
            logits_deberta_all.append(logits_deberta)
            logits_twitter_all.append(logits_twitter)

            text_lengths.append([len(text), len(text.split())])
            sentiments.append([TextBlob(text).sentiment.polarity])

        # X_meta
        X_meta = np.hstack([
            np.vstack(logits_deberta_all),
            np.vstack(logits_twitter_all),
            np.array(hate_scores),
            np.array(text_lengths),
            np.array(sentiments),
            np.abs(np.vstack(logits_deberta_all) - 0.5)
        ])

        return X_meta

    @staticmethod
    def find_best_global_threshold_by_macro_f1(y_true, y_probs, thresholds=np.linspace(0.1, 0.9, 81)):
        best_threshold = 0.5
        best_f1 = 0.0
        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_threshold = thresh
        return best_threshold, best_f1

    def train_and_save_meta_model_with_thresholds(
            self,
            models,
            tokenizers,
            save_dir,
            mlb,
            thresholds_path=None
    ):
        os.makedirs(save_dir, exist_ok=True)

        print("Building meta features...")
        X_meta = self.build_x_meta(models, tokenizers)

        print("Scaling...")
        scaler = StandardScaler()
        X_meta_scaled = scaler.fit_transform(X_meta)

        print("Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(X_meta_scaled, self.labels_bin, test_size=0.2,
                                                          random_state=42)

        print("Starting GridSearchCV...")
        param_grid = {
            "estimator__hidden_layer_sizes": [(32,), (64,), (128,)],
            "estimator__alpha": [0.0001, 0.001, 0.01],
            "estimator__learning_rate": ["constant", "adaptive"]
        }

        base_mlp = MLPClassifier(max_iter=500, random_state=42)
        meta_model = MultiOutputClassifier(base_mlp)

        grid_search = GridSearchCV(
            estimator=meta_model,
            param_grid=param_grid,
            cv=3,
            scoring="f1_macro",
            verbose=2,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        print("Best Params:", grid_search.best_params_)
        print("Best F1-macro:", grid_search.best_score_)

        y_pred_default = grid_search.predict(X_val)
        print("\nReport with default threshold (0.5):")
        print(classification_report(y_val, y_pred_default, target_names=[str(c) for c in mlb.classes_], digits=3,
                                    zero_division=0))

        y_probs = grid_search.predict_proba(X_val)
        y_probs_matrix = np.stack([
            cls_probs[:, 1] if cls_probs.ndim > 1 else cls_probs
            for cls_probs in y_probs
        ], axis=1)

        joblib.dump(grid_search.best_estimator_, os.path.join(save_dir, "meta_model_best.pkl"))
        joblib.dump(scaler, os.path.join(save_dir, "meta_scaler.pkl"))
        np.save(os.path.join(save_dir, "meta_model_val_probs.npy"), y_probs_matrix)

        if thresholds_path and os.path.exists(thresholds_path):
            print(f"\n📥 Using optimal thresholds from: {thresholds_path}")
            optimal_thresholds = joblib.load(thresholds_path) if thresholds_path.endswith(".pkl") else np.load(
                thresholds_path)
            print("First few thresholds:", np.round(optimal_thresholds[:5], 3))

            y_pred_optimal = (y_probs_matrix >= optimal_thresholds).astype(int)
            print("\nReport with optimal thresholds:")
            print(classification_report(y_val, y_pred_optimal, target_names=[str(c) for c in mlb.classes_], digits=3,
                                        zero_division=0))
            np.save(os.path.join(save_dir, "meta_model_val_preds_opt.npy"), y_pred_optimal)

        print("\nFinding best global threshold by F1_macro...")
        best_global_thresh, best_macro_f1 = self.find_best_global_threshold_by_macro_f1(y_val, y_probs_matrix)
        y_pred_global = (y_probs_matrix >= best_global_thresh).astype(int)

        print(f"Best global threshold: {best_global_thresh:.3f} | F1_macro: {best_macro_f1:.3f}")
        print("\nReport with global threshold:")
        print(classification_report(y_val, y_pred_global, target_names=[str(c) for c in mlb.classes_], digits=3,
                                    zero_division=0))

        with open(os.path.join(save_dir, "meta_model_report_global_thresh.txt"), "w") as f:
            f.write(classification_report(y_val, y_pred_global, target_names=[str(c) for c in mlb.classes_], digits=3,
                                          zero_division=0))

        print("Meta-model, scaler, thresholds, and reports saved to:", save_dir)
        return grid_search.best_estimator_

    def predict(
            self,
            texts,
            save_dir,
            model_paths,
            thresholds_type="default",
            thresholds_path=None
    ):
        # Load components
        try:
            meta_model = joblib.load(os.path.join(save_dir, "meta_model_best.pkl"))
            scaler = joblib.load(os.path.join(save_dir, "meta_scaler.pkl"))
            mlb = joblib.load(os.path.join(save_dir, "saved_mlb.pkl"))
        except FileNotFoundError as e:
            print(f"Error loading model or mlb: {e}")
            print("Please ensure the files exist at the specified paths.")
            # You might want to exit or handle this error appropriately
            exit()

        # Load base models
        models = {}
        tokenizers = {}
        for name, path in model_paths.items():
            if name == "deberta":
                model, tokenizer_loaded, _ = self.load_custom_model(path)
            else:
                tokenizer_loaded = AutoTokenizer.from_pretrained(path)
                model = AutoModelForSequenceClassification.from_pretrained(path)
            model.to(device)
            models[name] = model.eval()
            tokenizers[name] = tokenizer_loaded

        # Build features
        print("Building X_meta for inference...")
        X_meta = self.build_x_meta(models, tokenizers)
        X_meta_scaled = scaler.transform(X_meta)

        # Predict probabilities
        y_probs = meta_model.predict_proba(X_meta_scaled)
        y_probs_matrix = np.stack([
            cls[:, 1] if cls.ndim > 1 else cls for cls in y_probs
        ], axis=1)

        # Load thresholds
        if thresholds_type == "per_class" and thresholds_path:
            thresholds = joblib.load(thresholds_path) if thresholds_path.endswith(".pkl") else np.load(thresholds_path)
        elif thresholds_type == "global" and thresholds_path:
            thresholds = np.full(y_probs_matrix.shape[1], float(np.load(thresholds_path)))  # scalar → vector
        else:
            thresholds = np.full(y_probs_matrix.shape[1], 0.5)  # default

        # Decode predictions
        class_names = list(mlb.classes_)
        results = []
        for i, text in enumerate(texts):
            pred = (y_probs_matrix[i] >= thresholds).astype(int)
            pred_labels = [class_names[j] for j in range(len(class_names)) if pred[j] == 1]
            prob_dict = dict(zip(class_names, y_probs_matrix[i]))

            results.append({
                "text": text,
                "predicted_labels": pred_labels,
                "probabilities": prob_dict
            })

        return results

    def train(self):
        self.texts, labels, self.labels_bin, category_classes = self.prepare_multilabel_data(
            self.filtered_df,
            mlb_save_path="/content/drive/MyDrive/Project/Models/Multi_model/1805/saved_mlb222.pkl"
        )
        # pos_weight
        label_counts = self.labels_bin.sum(axis=0)
        total_samples = self.labels_bin.shape[0]
        neg_counts = total_samples - label_counts
        pos_weight = neg_counts / (label_counts + 1e-6)

        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to("cuda")

        # Model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-large",
            num_labels=5,
            problem_type="multi_label_classification")

        custom_model = CustomLossModel(base_model, pos_weight_tensor).to("cuda")

        # Dataset
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        dataset = MultiLabelDataset(self.texts, self.labels_bin, tokenizer)

        # train val split
        self.texts_train, self.texts_val, self.labels_train, self.labels_val = train_test_split(
            self.texts, self.labels_bin, test_size=0.2, random_state=42
        )

        train_dataset = MultiLabelDataset(self.texts_train, self.labels_train, tokenizer)
        val_dataset = MultiLabelDataset(self.texts_val, self.labels_val, tokenizer)

        # training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.save_model_dir}/checkpoints",

            # train
            learning_rate=1e-5,
            num_train_epochs=6,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,

            # evaluation
            eval_strategy="steps",
            eval_steps=1200,
            metric_for_best_model="f1_macro",
            load_best_model_at_end=True,
            greater_is_better=True,

            # saving
            save_strategy="steps",
            save_steps=1200,
            save_total_limit=3,

            fp16=True,
            gradient_checkpointing=True,

            # logs
            logging_dir="/content/drive/MyDrive/Project/Models/Multi_model/1805_1/logs",
            logging_steps=50,
            report_to="none")

        # New trainer
        self.trainer = Trainer(
            model=custom_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics_multi,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                SaveBestModelCallback(
                    dest_dir=self.save_model_dir)
            ]
        )

        # Start Training
        self.trainer.train()

        # Save threshold
        optimal_thresholds, metrics_report = self.compute_and_save_thresholds_and_report(
            trainer=self.trainer,
            val_dataset=val_dataset,
            save_dir=self.save_model_dir,
            file_prefix=""
        )

        # Save full model
        custom_model = custom_model
        tokenizer = tokenizer
        pos_weight_tensor = pos_weight_tensor
        save_dir = f"{self.save_model_dir}/saving"
        self.save_custom_model(custom_model, tokenizer, pos_weight_tensor, save_dir)

        self.metamodel()

    def metamodel(self):
        mlb = joblib.load(self.mlb_path)
        model_paths = {
            "deberta": f"{self.save_model_dir}/saving",
            "hatebert": "Hate-speech-CNERG/dehatebert-mono-english",
            "twitter": "cardiffnlp/twitter-roberta-base",
        }

        for name, path in model_paths.items():
            if name == "deberta":
                model, tokenizer_loaded, _ = self.load_custom_model(path)
                model.to(device)
            else:
                tokenizer_loaded = AutoTokenizer.from_pretrained(path)
                model = AutoModelForSequenceClassification.from_pretrained(path)
                model.to(device)

            self.models[name] = model.eval()
            self.tokenizers[name] = tokenizer_loaded
        X_meta = self.build_x_meta(self.models, self.tokenizers)
        scaler = StandardScaler()
        self.X_meta_scaled = scaler.fit_transform(X_meta)
        X_train, X_val, y_train, y_val = train_test_split(self.X_meta_scaled, self.labels_bin, test_size=0.2,
                                                          random_state=42)
        meta_model = MultiOutputClassifier(
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
        )
        meta_model.fit(X_train, y_train)

        y_pred = meta_model.predict(X_val)
        print(classification_report(y_val, y_pred, target_names=[str(c) for c in mlb.classes_], digits=3))

        meta_model = self.train_and_save_meta_model_with_thresholds(
            models=self.models,
            tokenizers=self.tokenizers,
            save_dir="/content/drive/MyDrive/Project/Models/Multi_model/1805_1",
            mlb=mlb,
            thresholds_path=f"{self.save_model_dir}/optimal_thresholds.pkl"
        )

        self.post_metamodel()

    def post_metamodel(self):
        thresholds_path = os.path.join(self.save_model_dir, "optimal_thresholds.pkl")
        optimal_thresholds = joblib.load(thresholds_path)
        print("Optimal thresholds")
        print(np.round(optimal_thresholds, 4))

        report_path = os.path.join(self.save_model_dir, "classification_report.txt")
        with open(report_path, "r", encoding="utf-8") as f:
            metrics_report = f.read()

        print(metrics_report)
        save_dir = f"{self.save_model_dir}/saving"
        # save_dir = "/content/drive/MyDrive/Project/Models/Multi_model/1705_1/final_multilabel_model"

        model, tokenizer, mlb = self.load_custom_model(save_dir)
        val_dataset = MultiLabelDataset(self.texts_val, self.labels_val, tokenizer)

        inference_args = TrainingArguments(
            output_dir="./tmp_eval",
            per_device_eval_batch_size=8,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=inference_args,
            tokenizer=tokenizer
        )

        outputs = trainer.predict(val_dataset)
        logits = outputs.predictions
        labels_val = outputs.label_ids

        outputs = trainer.predict(val_dataset)
        logits = outputs.predictions
        labels_val = outputs.label_ids

        probs = sigmoid(torch.tensor(logits)).numpy()
        num_classes = probs.shape[1]

        thresholds_tensor = torch.tensor(optimal_thresholds[:num_classes], dtype=torch.float32, device=probs.device)
        probs_tensor = torch.tensor(probs, dtype=torch.float32, device=probs.device)  # Convert probs to tensor
        preds_bin = (probs_tensor >= thresholds_tensor).int()  # Use probs_tensor for comparison

        labels_val = outputs.label_ids
        preds_bin = (probs >= optimal_thresholds).astype(int)
        relevant_indices = np.arange(0, labels_val.shape[1])  # Get indices from 1 to the end

        labels_val_multi = labels_val[:, relevant_indices]
        preds_bin_multi = preds_bin[:, relevant_indices]
        target_names_multi = [str(name) for name in mlb.classes_[relevant_indices]]
        print(classification_report(labels_val_multi, preds_bin_multi, target_names=target_names_multi, digits=4))
