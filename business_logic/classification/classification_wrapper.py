import os
import pickle
from pathlib import Path
from safetensors.torch import load_file as load_safetensors
import torch
import numpy as np
import joblib
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    PreTrainedModel
)
from transformers.modeling_outputs import SequenceClassifierOutput

# Constants
BASE_DIR = Path(__file__).absolute().parent.parent.parent
BIN_MODEL_PATH = os.path.join(BASE_DIR, 'resources', 'binary_model_files')
ML_MODEL_PATH = os.path.join(BASE_DIR, 'resources', 'multilabel_model_files')

# Mapping for multi-label categories
code_to_name = {
    1: "Antisemitic Ideology",
    2: "Stereotypes and Dehumanization",
    3: "Antisemitism Against Israel/Zionism",
    4: "Holocaust Denial/Distortion",
    5: "Indirect Antisemitism",
}
# Create a reverse mapping for easy lookup
name_to_code = {name: code for code, name in code_to_name.items()}


def load_model_config(model_path):
    try:
        # Check if path exists and contains config.json
        config_file = os.path.join(model_path, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"config.json not found in {model_path}")

        # Load config
        config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True  # Ensure it doesn't try to download from HuggingFace
        )
        return config

    except Exception as e:
        print(f"Error loading config from {model_path}: {e}")
        return None


class DebertaBinaryClassifier(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = "encoder"

    def __init__(self, config, pos_weight=None):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config._name_or_path, config=config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else None
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls).squeeze(-1)
        loss = None
        if labels is not None and self.loss_fn is not None:
            loss = self.loss_fn(logits, labels.float())
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits.unsqueeze(-1),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CustomLossModel(nn.Module):
    def __init__(self, base_model, pos_weight):
        super().__init__()
        self.base_model = base_model
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels.float()) if labels is not None else None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )


def load_binary_model(device=torch.device('cpu')):
    """Load binary classifier entirely from local files, supporting .safetensors or .bin weights."""
    # load config
    config = load_model_config(BIN_MODEL_PATH)
    # instantiate model architecture
    model = DebertaBinaryClassifier(config)
    # locate weights: search for any .safetensors file
    safetensors_files = [f for f in os.listdir(BIN_MODEL_PATH) if f.endswith('.safetensors')]
    if safetensors_files:
        weights_path = os.path.join(BIN_MODEL_PATH, safetensors_files[0])
        try:
            # Pass device as a torch.device object
            state_dict = load_safetensors(weights_path, device=str(device))
        except ImportError:
            raise ImportError("safetensors library not installed; cannot load safetensors weights")
    else:
        # fallback to pytorch_model.bin
        bin_path = os.path.join(BIN_MODEL_PATH, "pytorch_model.bin")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"No weights file found in {BIN_MODEL_PATH}: looked for '*.safetensors' or pytorch_model.bin"
            )
        state_dict = torch.load(bin_path, map_location=device)  # Fixed: use device parameter

    # Remove unexpected keys from the state dictionary
    unexpected_keys = ["loss_fn.pos_weight"]
    for key in unexpected_keys:
        if key in state_dict:
            del state_dict[key]

    # load state dict
    model.load_state_dict(state_dict)
    model.to(device).eval()
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BIN_MODEL_PATH, local_files_only=True)
    return model, tokenizer, device


def load_multilabel_model(device=torch.device('cpu')):
    """Load multi-label model entirely from local files."""
    # load config
    config = AutoConfig.from_pretrained(ML_MODEL_PATH, local_files_only=True)


    # Temporarily set the default tensor type to CPU before loading pickle files
    # This ensures any tensors in the pickle files are created on CPU
    torch.set_default_tensor_type('torch.FloatTensor')  # CPU tensors

    try:
        # load extras & classes
        extras = joblib.load(os.path.join(ML_MODEL_PATH, "extras.pkl"))

        # Ensure pos_weight is moved to the correct device
        if isinstance(extras["pos_weight"], torch.Tensor):
            pos_w = extras["pos_weight"].to(device)
        else:
            pos_w = torch.tensor(extras["pos_weight"], dtype=torch.float32, device=device)

    finally:
        # Restore original tensor type
        orig = torch.get_default_dtype()
        torch.set_default_dtype(orig)

    try:
        mlb = joblib.load(os.path.join(ML_MODEL_PATH, "updated_mlb.pkl"))

    except Exception as e:
        raise Exception(e)

    # with open(os.path.join(ML_MODEL_PATH, "extras.pkl"), "rb") as f:
    #     extras = torch.load(f, map_location=torch.device("cpu"))
    #     if isinstance(extras["pos_weight"], torch.Tensor):
    #         extras["pos_weight"] = extras["pos_weight"].cpu()
    # mlb = joblib.load(os.path.join(ML_MODEL_PATH, "updated_mlb.pkl"))

    # adjust config
    config.num_labels = len(mlb.classes_)
    config.problem_type = "multi_label_classification"
    # instantiate base model
    base_model = AutoModelForSequenceClassification.from_config(config)
    model = CustomLossModel(base_model, pos_w)
    # load weights
    weights_file = os.path.join(ML_MODEL_PATH, "custom_model_weights.pt")
    state_dict = torch.load(weights_file, map_location=device,
                            weights_only=False)  # Fixed: use device parameter and weights_only=False
    model.load_state_dict(state_dict)
    model.to(device).eval()
    # tokenizer & thresholds
    tokenizer = AutoTokenizer.from_pretrained(ML_MODEL_PATH, local_files_only=True)
    thresholds = np.load(os.path.join(ML_MODEL_PATH, "deberta_thresholds_optimal.npy"))
    return model, tokenizer, mlb, thresholds, device


class ClassificationModel:
    """
    End-to-end predictor using fixed model paths.
    Returns 0 if binary prediction is False, otherwise returns the numerical code for the multi-label prediction.
    """

    def __init__(self, device=None):
        # Auto-detect device if not provided
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        self.device = device
        print(f"Using device: {self.device}")

        self.bin_model, self.bin_tokenizer, self.device = load_binary_model(self.device)
        (self.multi_model,
         self.multi_tokenizer,
         self.mlb,
         self.thresholds,
         self.device) = load_multilabel_model(self.device)

    def predict(self, texts, batch_size=8, bin_threshold=0.5):
        """
        Predict on a list of texts.
        Returns 0 if binary prediction is False, otherwise returns the numerical code for the multi-label prediction.
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # binary stage
        enc = self.bin_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            bin_logits = self.bin_model(**enc).logits.squeeze(-1)
            bin_probs = torch.sigmoid(bin_logits).cpu().numpy()
            bin_preds = bin_probs >= bin_threshold

        # Check binary prediction for the first text
        if not bin_preds[0]:
            return 0
        else:
            # multi-label stage
            text = texts[0]  # Assuming prediction is for a single text
            enc2 = self.multi_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            enc2 = {k: v.to(self.device) for k, v in enc2.items()}  # Fixed: properly move to device
            with torch.no_grad():
                multi_logits = self.multi_model(**enc2).logits.cpu().numpy()[0]
                multi_probs = 1 / (1 + np.exp(-multi_logits))
                # Find the label with the highest probability
                best_label_index = np.argmax(multi_probs)
                predicted_label = self.mlb.classes_[best_label_index]
                # Get the numerical code for the predicted label
                return name_to_code.get(predicted_label, 0)  # Return 0 if label not found in mapping
