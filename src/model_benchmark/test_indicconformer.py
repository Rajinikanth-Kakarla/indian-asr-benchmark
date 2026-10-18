
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
# AUDIO_PATH = Path("data/audio/hindi_audio.mp3")
# LANGUAGE = "hi"

# AUDIO_PATH = Path("data/audio/telugu_audio.mp3")
# LANGUAGE = "te"

AUDIO_PATH = Path("data/audio/hyderabadi.mp3")
LANGUAGE = "te"

# 1. Load the pretrained model
print("Loading IndicConformer...")

model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)
model.eval()

# 2. Load and resample audio to 16 kHz mono
print(f"Loading audio: {AUDIO_PATH}")

waveform, _ = librosa.load(
    str(AUDIO_PATH),
    sr=16000,
    mono=True,
)

waveform = torch.from_numpy(
    np.asarray(waveform, dtype=np.float32)
).unsqueeze(0)

print("Sample rate: 16000")
print(
    "Audio duration:",
    round(waveform.numel() / 16000, 2),
    "seconds",
)

# 3. Run inference
print("Running transcription...")

with torch.inference_mode():
    prediction = model(waveform, LANGUAGE, "ctc")

# 4. Inspect the returned prediction
print("\nRaw model output:")
print(prediction)
print("\nOutput type:", type(prediction).__name__)

print(
    "Audio duration:",
    round(waveform.shape[-1] / 16000, 2),
    "seconds",
)