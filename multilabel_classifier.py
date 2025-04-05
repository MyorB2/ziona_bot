import os
import torch
import numpy as np
import warnings
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from sklearn.model_selection import train_test_split, MultilabelStratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from preprocessing import prepare_dataset, remove_non_antisemitic
from utils import compute_metrics, compute_final_metrics, test_model
from hate_speech_dataset_class import HateSpeechDataset

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")
# Ensure device is set to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "GroNLP/hateBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class AntisemitismClassifier:
    def __init__(self):
        self.model = None

    def train_classifier(self):
        combined_df_cleaned, categories = prepare_dataset()
        combined_df_cleaned = combined_df_cleaned.dropna(subset=["clean_extracted_text"]).reset_index(drop=True)
        combined_df_cleaned = remove_non_antisemitic(combined_df_cleaned)

        labels = np.array(combined_df_cleaned["one_hot_sub_cat"].tolist())
        print(f"\nchecking labels shape: {labels.shape}\n")
        num_labels = len(categories)

        texts = combined_df_cleaned["clean_extracted_text"].tolist()
        labels = np.array(combined_df_cleaned["one_hot_bin_cat"].tolist())

        # Train, Validation and Test split
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, test_size=0.2, random_state=42, stratify=train_val_labels
        )

        # Datasets
        train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer)
        val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer)
        test_dataset = HateSpeechDataset(test_texts, test_labels, tokenizer)

        # Split data using MultilabelStratifiedKFold
        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        for train_idx, val_idx in mskf.split(combined_df_cleaned["clean_extracted_text"], labels):
            train_texts, val_texts = combined_df_cleaned["clean_extracted_text"].iloc[train_idx].tolist(), combined_df_cleaned["clean_extracted_text"].iloc[val_idx].tolist()
            train_labels, val_labels = labels[train_idx], labels[val_idx]

            train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer)
            val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer)

            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)

            training_args = TrainingArguments(
                output_dir="./results",
                evaluation_strategy="epoch",
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=3,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
                save_strategy="epoch",
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics
            )

            trainer.train()

            # Save the trained model
            model.save_pretrained("saved_hateBERT_model")
            tokenizer.save_pretrained("saved_hateBERT_model")

        self.model = model
        # Get total metrics after training
        final_metrics = compute_final_metrics()
        print("Final Averaged Metrics:", final_metrics)
        #  Evaluate Model on Test Set
        test_metrics = test_model(model, device, test_dataset)
        print("\nFinal Test Set Metrics:", test_metrics)

    def predict(self, text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        text_list = [text]
        test_dataset = HateSpeechDataset(text_list, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available())

        all_preds = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Predicting"):
                batch = {key: val.to(device) for key, val in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
                all_preds.extend(preds)

        # Interpret the prediction. The label '1' from this model represents hate speech.
        if all_preds[0][0] == 1:
            return "Hate Speech"
        else:
            return "Not Hate Speech"
