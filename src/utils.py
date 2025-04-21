import nltk
import os
import torch
import numpy as np
import json
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")


def compute_metrics(pred, fold_number=None):
    preds = torch.sigmoid(torch.tensor(pred.predictions)) > 0.5  # Convert logits to binary
    labels = pred.label_ids

    # Compute per-label metrics
    precision_per_label = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_label = recall_score(labels, preds, average=None, zero_division=0)
    f1_per_label = f1_score(labels, preds, average=None, zero_division=0)

    # Store per-label metrics in a dictionary
    label_metrics = {
        f"label_{i}": {
            "precision": precision_per_label[i],
            "recall": recall_per_label[i],
            "f1_score": f1_per_label[i]
        } for i in range(len(precision_per_label))
    }

    # Compute overall metrics
    overall_metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
    }

    metrics_file = "evaluation_metrics_per_label.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {"folds": []}

    fold_results = {
        "fold": fold_number,
        "overall": overall_metrics,
        "per_label": label_metrics
    }
    all_metrics["folds"].append(fold_results)

    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=4)

    return overall_metrics


def compute_final_metrics():
    metrics_file = "evaluation_metrics_per_label.json"

    with open(metrics_file, "r") as f:
        all_metrics = json.load(f)

    num_folds = len(all_metrics["folds"])

    # Extract overall metrics from all folds
    overall_metrics_list = [fold["overall"] for fold in all_metrics["folds"]]

    # Compute the average for each overall metric across folds
    final_overall_metrics = {
        metric: np.mean([fold[metric] for fold in overall_metrics_list])
        for metric in overall_metrics_list[0]
    }

    # Extract per-label metrics
    per_label_metrics = {}
    for fold in all_metrics["folds"]:
        for label, label_scores in fold["per_label"].items():
            if label not in per_label_metrics:
                per_label_metrics[label] = {
                    "precision": [],
                    "recall": [],
                    "f1_score": []
                }
            per_label_metrics[label]["precision"].append(label_scores["precision"])
            per_label_metrics[label]["recall"].append(label_scores["recall"])
            per_label_metrics[label]["f1_score"].append(label_scores["f1_score"])

    # Compute the average for each label across folds
    final_per_label_metrics = {
        label: {
            "precision": np.mean(scores["precision"]),
            "recall": np.mean(scores["recall"]),
            "f1_score": np.mean(scores["f1_score"])
        }
        for label, scores in per_label_metrics.items()
    }

    # Save final averaged metrics
    all_metrics["final"] = {
        "overall": final_overall_metrics,
        "per_label": final_per_label_metrics
    }

    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=4)

    return all_metrics["final"]


def test_model(model, device, test_dataset, batch_size=8):

    model.eval()
    model.to(device)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing Model"):
            batch = {key: val.to(device) for key, val in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits

            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_metrics = compute_metrics(SimpleNamespace(predictions=all_preds, label_ids=all_labels))

    print("\nTest Set Metrics:", test_metrics)
    return test_metrics
