"""
YAMNet Audio Event Classifier — TensorFlow Hub edition.

Loads the official pre-trained YAMNet model from:
    https://tfhub.dev/google/yamnet/1

Public API
----------
predict_audio_events(waveform, sample_rate)
    → dict with top_class, top_score, top_dict, all_events

The model is loaded **once** at import time and cached for the lifetime of
the process, so subsequent calls are fast (~15 ms per 1-second clip on CPU).
"""

from __future__ import annotations

import csv
import io
import numpy as np
from pathlib import Path

# ---------- TensorFlow / Hub imports ----------
import tensorflow as tf
import tensorflow_hub as hub

# ------------------------------------------------------------------ #
# 1.  Load the YAMNet model from TensorFlow Hub (cached after first   #
#     download — subsequent runs are instant).                         #
# ------------------------------------------------------------------ #
print("⏳ Loading YAMNet model from TensorFlow Hub …")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("✅ YAMNet model loaded successfully!")

# ------------------------------------------------------------------ #
# 2.  Load the class-name mapping.                                     #
#     The CSV ships inside the hub module itself, so we read it from   #
#     there instead of requiring a separate file.                      #
# ------------------------------------------------------------------ #
def _load_class_names() -> list[str]:
    """
    Return the 521 YAMNet class names in index order.

    Strategy:
    1. Try to read 'yamnet_class_map.csv' bundled inside the hub module.
    2. Fall back to a local copy at  backend/models/yamnet_class_map.csv.
    3. If neither is available, return generic labels ("class_0", …).
    """
    # --- attempt 1: CSV shipped with the hub model -------------------
    try:
        # The hub module exposes the CSV path as an asset
        class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [row["display_name"] for row in reader]
    except Exception:
        pass

    # --- attempt 2: local fallback -----------------------------------
    local_csv = Path(__file__).parents[1] / "models" / "yamnet_class_map.csv"
    if local_csv.is_file():
        with open(local_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [row["display_name"] for row in reader]

    # --- attempt 3: generic placeholder labels -----------------------
    print("⚠  Could not locate yamnet_class_map.csv — using generic labels.")
    return [f"class_{i}" for i in range(521)]


CLASS_NAMES: list[str] = _load_class_names()
print(f"   ↳ {len(CLASS_NAMES)} YAMNet class names loaded.")


# ------------------------------------------------------------------ #
# 3.  Waveform preparation helpers                                     #
# ------------------------------------------------------------------ #
def _ensure_mono_16k(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Convert an arbitrary waveform to the format YAMNet expects:
      • mono channel
      • 16 000 Hz sample rate
      • float32 in the range [-1.0, 1.0]
    """
    # Mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # float32
    waveform = waveform.astype(np.float32)

    # Resample to 16 kHz if necessary
    if sample_rate != 16000:
        target_len = int(len(waveform) * 16000 / sample_rate)
        waveform = np.interp(
            np.linspace(0, len(waveform), target_len, endpoint=False),
            np.arange(len(waveform)),
            waveform,
        ).astype(np.float32)

    # Normalise to [-1, 1]
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak

    return waveform


# ------------------------------------------------------------------ #
# 4.  Public inference function                                        #
# ------------------------------------------------------------------ #
def predict_audio_events(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    top_n: int = 5,
) -> dict:
    """
    Run YAMNet inference on a raw waveform.

    Parameters
    ----------
    waveform : np.ndarray
        1-D or 2-D NumPy array of audio samples.
    sample_rate : int
        Sample rate of *waveform* (will be resampled to 16 kHz internally).
    top_n : int
        How many top-scoring classes to include in the response.

    Returns
    -------
    dict
        {
            "top_class":  str,          # e.g. "Screaming"
            "top_score":  float,        # e.g. 0.87
            "top_dict":   {str: float}, # top-N classes with their scores
            "all_events": [str],        # list of all class names above 0.1
        }
    """
    wav = _ensure_mono_16k(waveform, sample_rate)

    # YAMNet returns:
    #   scores      – (n_frames, 521) softmax probabilities
    #   embeddings  – (n_frames, 1024) feature vectors
    #   spectrogram – the log-mel spectrogram (unused here)
    scores, embeddings, spectrogram = yamnet_model(wav)
    scores_np: np.ndarray = scores.numpy()          # (n_frames, 521)

    # Average across all frames
    mean_scores = scores_np.mean(axis=0)             # (521,)

    # Top class
    top_idx = int(mean_scores.argmax())
    top_label = CLASS_NAMES[top_idx]
    top_score = float(mean_scores[top_idx])

    # Top-N dictionary
    top_idxs = mean_scores.argsort()[-top_n:][::-1]
    top_dict = {CLASS_NAMES[i]: round(float(mean_scores[i]), 4) for i in top_idxs}

    # Every class that scored > 0.1 (useful for multi-label scenarios)
    all_events = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) if mean_scores[i] > 0.1]

    return {
        "top_class": top_label,
        "top_score": round(top_score, 4),
        "top_dict": top_dict,
        "all_events": all_events,
    }
