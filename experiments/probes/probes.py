import json
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

import scripts.probes.extract_activations as extract_activations

PROBE_WEIGHTS_DIR = Path(__file__).parent / "probe_weights"
CLASSIFICATION_THRESHOLD = 0.5


class Probe:
    """Pure model wrapper around sklearn Ridge regression for probing."""

    def __init__(self, reg_strength: float = 1.0):
        self.reg_strength = reg_strength
        self.model = Ridge(alpha=reg_strength, max_iter=1000)
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean((preds > CLASSIFICATION_THRESHOLD) == (y > CLASSIFICATION_THRESHOLD)))

    def save(self, path: Union[str, Path]):
        if not self.is_trained:
            raise ValueError("Cannot save probe that hasn't been trained.")
        torch.save({"model": self.model, "reg_strength": self.reg_strength}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Probe":
        data = torch.load(path, map_location="cpu")
        probe = cls(reg_strength=data["reg_strength"])
        probe.model = data["model"]
        probe.is_trained = True
        return probe


class ProbeTrainer:
    """Orchestrates probe training: data loading, training, evaluation, logging."""

    def __init__(
        self,
        experiment_name: str,
        layers: list[int],
        reg_strength: float = 1.0,
        label_key: str = "flag",
        activation_source: str = "response",
        logger: Optional[Callable[[dict], None]] = None,
    ):
        self.experiment_name = experiment_name
        if not layers:
            raise ValueError("layers must contain at least one layer index.")
        self.layers = layers
        self.reg_strength = reg_strength
        self.label_key = label_key
        valid_sources = {"response", "response_last", "intervention", "intervention_last"}
        if activation_source not in valid_sources:
            raise ValueError(f"activation_source must be one of {valid_sources}.")
        self.activation_source = activation_source
        self.logger = logger
        self.probe = Probe(reg_strength=reg_strength)

    def load_activations_and_labels(
        self,
        split: str = "train",
        include_topics: Optional[list[str]] = None,
        exclude_topics: Optional[list[str]] = None,
    ):
        split_dir = (extract_activations.ACTIVATIONS_OUTPUT_DIR / self.experiment_name) / split
        metadata_path = split_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        df = pd.DataFrame(metadata["samples"])
        if df.empty:
            raise ValueError(f"No samples in {metadata_path}. Did you run extraction?")

        if include_topics is not None:
            mask = df["topic"].isin(include_topics)
        elif exclude_topics is not None:
            mask = ~df["topic"].isin(exclude_topics)
        else:
            mask = pd.Series(True, index=df.index)

        indices = df.index[mask].tolist()
        df = df.loc[mask]

        activations_key = f"{self.activation_source}_activations"
        layer_activations = []
        for layer in self.layers:
            layer_file = split_dir / f"activations_layer_{layer}.pt"
            layer_payload = torch.load(layer_file, map_location="cpu")
            if activations_key not in layer_payload:
                raise KeyError(
                    f"{activations_key} missing from {layer_file}. "
                    f"Available keys: {list(layer_payload.keys())}"
                )
            layer_tensor = layer_payload[activations_key].to(torch.float32)[indices]
            layer_activations.append(layer_tensor)

        activations = (
            torch.cat(layer_activations, dim=-1)
            if len(layer_activations) > 1
            else layer_activations[0]
        )

        labels = self._extract_labels(df)

        if activations.shape[0] != len(labels):
            raise ValueError(
                f"Activation/label mismatch: {activations.shape[0]} vs {len(labels)}"
            )

        return activations.numpy(), labels

    def train(
        self,
        include_topics: Optional[list[str]] = None,
        exclude_topics: Optional[list[str]] = None,
        save: bool = True,
    ):
        train_acts, train_labels = self.load_activations_and_labels(
            "train", include_topics=include_topics, exclude_topics=exclude_topics
        )

        self.probe.fit(train_acts, train_labels)
        acc = self.probe.score(train_acts, train_labels)

        if save:
            self._save_probe()

        self._log({
            "train_acc_linear": acc,
            "layer": self.layers,
            "reg_strength": self.reg_strength,
        })

        return acc

    def evaluate(
        self,
        include_topics: Optional[list[str]] = None,
        exclude_topics: Optional[list[str]] = None,
    ):
        test_acts, test_labels = self.load_activations_and_labels(
            "val", include_topics=include_topics, exclude_topics=exclude_topics
        )

        acc = self.probe.score(test_acts, test_labels)

        self._log({
            "test_acc_linear": acc,
            "layer": self.layers,
            "reg_strength": self.reg_strength,
        })

        return acc

    def _save_probe(self, path: Optional[Union[str, Path]] = None):
        if path is None:
            PROBE_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"linear_probe_{self.experiment_name}_{self.layers}_{self.reg_strength}.pkl"
            path = PROBE_WEIGHTS_DIR / filename

        self.probe.save(path)
        print(f"Saved probe to {path}")

    def _log(self, data: dict):
        if self.logger:
            self.logger(data)

    def _extract_labels(self, df: pd.DataFrame) -> np.ndarray:
        if self.label_key not in df.columns:
            raise KeyError(
                f"Label '{self.label_key}' not found. Available: {list(df.columns)}"
            )

        col = df[self.label_key]

        if self.label_key == "flag":
            return (col == "PASS").astype(np.float32).values

        if col.dtype == bool:
            return col.astype(np.float32).values

        if np.issubdtype(col.dtype, np.number):
            return col.astype(np.float32).values

        raise ValueError(
            f"Unsupported label type for '{self.label_key}': {col.dtype}. "
            "Only 'flag' (PASS/FAIL), booleans, and numeric types are supported."
        )
