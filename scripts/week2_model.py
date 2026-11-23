"""RoBERTa window classification with optional additive year and chamber effects."""

import math
import statistics
from functools import partial

import torch
from torch import nn

from week2_text_pilot import make_model, speech_log_probs


FEATURES = {"text": (), "year": ("year",), "chamber": ("chamber",),
            "both": ("year", "chamber")}
CHAMBER = {"H": 0.0, "S": 1.0}


class SpeechClassifier(nn.Module):
    """Add a learned class-specific offset for each metadata feature to the text logits."""

    def __init__(self, encoder, variant, metadata_stats):
        super().__init__()
        self.encoder = encoder
        self.variant = variant
        self.features = FEATURES[variant]
        if list(self.features) != metadata_stats["features"]:
            raise ValueError("Checkpoint features do not match the requested variant.")
        if "year" in self.features:
            mean, scale = metadata_stats["year_mean"], metadata_stats["year_scale"]
            if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
                raise ValueError("The year standardization parameters are invalid.")
        if "chamber" in self.features and metadata_stats["chamber_encoding"] != CHAMBER:
            raise ValueError("Unexpected chamber encoding in checkpoint.")
        self.metadata_stats = metadata_stats
        # Zero initialization gives the paired text and metadata models identical initial
        # predictions and preserves the RNG state for matched dropout on the first batch.
        self.metadata_weight = (nn.Parameter(torch.zeros(len(self.features), 2))
                                if self.features else None)

    def forward(self, input_ids, attention_mask, metadata=None):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.metadata_weight is not None:
            output.logits = output.logits.float() + (metadata @ self.metadata_weight).float()
        return output


def build_model(variant, train_speeches):
    """Initialize the verified pretrained encoder and fit metadata transforms on training only."""
    features = FEATURES[variant]
    stats = {"features": list(features)}
    if "year" in features:
        years = [s.year for s in train_speeches]
        stats["year_mean"] = statistics.fmean(years)
        stats["year_scale"] = statistics.pstdev(years) or 1.0
    if "chamber" in features:
        stats["chamber_encoding"] = CHAMBER.copy()
    return SpeechClassifier(make_model(False), variant, stats)


def build_for_reload(variant, metadata_stats):
    """Create the exact local architecture for a strict load_state_dict of a saved run."""
    return SpeechClassifier(make_model(True), variant, metadata_stats)


def log_probs(model, speeches, device, bf16):
    """Return one log-probability vector per speech, averaging probabilities across windows."""
    forward = model
    if model.features:
        rows = []
        for speech in speeches:
            features = []
            if "year" in model.features:
                features.append((speech.year - model.metadata_stats["year_mean"])
                                / model.metadata_stats["year_scale"])
            if "chamber" in model.features:
                features.append(CHAMBER[speech.chamber])
            rows.extend([features] * len(speech.windows))
        metadata = torch.tensor(rows, dtype=torch.float32, device=device)
        forward = partial(model, metadata=metadata)
    return speech_log_probs(forward, speeches, device, bf16)
