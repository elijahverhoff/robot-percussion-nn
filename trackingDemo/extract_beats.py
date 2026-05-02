"""
Extract predicted beat times from an audio file using the trained causal TCN.
Saves a JSON file with the list of beat times in seconds.

Usage:  python extract_beats.py songs/your_song.mp3
Outputs: songs/your_song.beats.json
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from scipy.signal import find_peaks


# --- Audio / feature parameters (must match training) ---
SR     = 22050
HOP    = 220
N_FFT  = 2048
N_MELS = 81
FPS    = SR / HOP  # 100.0

# --- Inference parameters ---
THRESHOLD       = 0.40   # peak-picking threshold tuned on val set
MIN_INTERVAL_S  = 0.15   # rules out tempos > 400 BPM
CHECKPOINT_PATH = 'best_v1.pt'


# --- Model architecture (must match training) ---
class DilatedConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, causal=True):
        super().__init__()
        self.causal = causal
        pad_total = (kernel_size - 1) * dilation
        if causal:
            self.left_pad, self.right_pad = pad_total, 0
        else:
            self.left_pad = pad_total // 2
            self.right_pad = pad_total - self.left_pad
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.left_pad, self.right_pad))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.1, causal=True):
        super().__init__()
        self.conv1 = DilatedConv1d(channels, channels, kernel_size, dilation, causal)
        self.conv2 = DilatedConv1d(channels, channels, kernel_size, dilation, causal)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.drop(y)
        y = F.relu(self.norm2(self.conv2(y)))
        y = self.drop(y)
        return x + y


class BeatTCN(nn.Module):
    def __init__(self, n_mels=81, channels=64, kernel_size=5,
                 dilations=(1, 2, 4, 8, 16, 32), dropout=0.1, causal=True):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((3, 1)),
            nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((3, 1)),
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((3, 1)),
        )
        self.channel_proj = nn.Conv1d(64 * 3, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            TCNBlock(channels, kernel_size, d, dropout, causal) for d in dilations
        ])
        self.head = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.frontend(x)
        B, C, F_dim, T = x.shape
        x = x.reshape(B, C * F_dim, T)
        x = self.channel_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(1)


def extract_beats(audio_path):
    print(f"Loading audio: {audio_path}")
    audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    duration = len(audio) / SR
    print(f"  duration: {duration:.1f}s")

    print("Computing log-mel spectrogram...")
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel_t = torch.from_numpy(log_mel).float().unsqueeze(0)  # [1, N_MELS, T]

    print("Loading model...")
    model = BeatTCN()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    model.eval()

    print("Running inference...")
    with torch.no_grad():
        logits = model(log_mel_t)
        activation = torch.sigmoid(logits).squeeze(0).numpy()

    print("Peak-picking...")
    min_dist = int(MIN_INTERVAL_S * FPS)
    peaks, _ = find_peaks(activation, height=THRESHOLD, distance=min_dist)
    beat_times = (peaks / FPS).tolist()

    print(f"Found {len(beat_times)} beats")
    if len(beat_times) > 1:
        intervals = np.diff(beat_times)
        bpm = 60.0 / np.median(intervals)
        print(f"Estimated tempo: {bpm:.1f} BPM")

    return beat_times


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_beats.py <audio_file>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    beat_times = extract_beats(audio_path)

    out_path = audio_path.with_suffix('.beats.json')
    with open(out_path, 'w') as f:
        json.dump({'beats': beat_times, 'audio_file': str(audio_path)}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()