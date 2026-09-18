"""Run the original FP32 LeeNet11 checkpoint on the project's audio inputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import wave
from pathlib import Path

import numpy as np
import torch

from leenet11 import LeeNet11


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(__file__).with_name("weights") / "leenet11.pth"
DEFAULT_LABELS = REPOSITORY_ROOT / "PC_UI" / "class_labels_indices.csv"
SAMPLE_RATE = 32000
SAMPLE_COUNT = 320000
CLASS_COUNT = 527


def load_model(checkpoint_path: Path) -> tuple[LeeNet11, int | None]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = LeeNet11(
        sample_rate=SAMPLE_RATE, window_size=1024, hop_size=320,
        mel_bins=64, fmin=50, fmax=14000, classes_num=CLASS_COUNT,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint.get("iteration")


def load_audio(path: Path) -> torch.Tensor:
    """Read exactly 10 seconds of mono PCM16; raw PCM is little-endian 32 kHz."""
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as stream:
            actual = (
                stream.getnchannels(), stream.getsampwidth(),
                stream.getframerate(), stream.getnframes(), stream.getcomptype(),
            )
            expected = (1, 2, SAMPLE_RATE, SAMPLE_COUNT, "NONE")
            if actual != expected:
                raise ValueError(f"Expected mono PCM16 32 kHz / 10 s WAV: {path}")
            pcm = stream.readframes(SAMPLE_COUNT)
    elif path.suffix.lower() == ".pcm":
        pcm = path.read_bytes()
    else:
        raise ValueError(f"Expected a .wav or .pcm input: {path}")

    if len(pcm) != SAMPLE_COUNT * 2:
        raise ValueError(f"Expected {SAMPLE_COUNT * 2} PCM bytes: {path}")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(samples).unsqueeze(0)


def load_labels(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        labels = {
            int(row["index"]): row["display_name"]
            for row in csv.DictReader(stream)
        }
    if set(labels) != set(range(CLASS_COUNT)):
        raise ValueError("The label file must cover all 527 AudioSet classes")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="+", help="32 kHz mono PCM16 WAV/PCM files, 10 s each")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if not 1 <= args.top_k <= CLASS_COUNT:
        parser.error("--top-k must be between 1 and 527")
    if args.threads < 1:
        parser.error("--threads must be positive")

    torch.set_num_threads(args.threads)
    model, iteration = load_model(args.checkpoint)
    labels = load_labels(args.labels)
    with torch.inference_mode():
        for path in args.audio:
            probabilities = model(load_audio(path))["clipwise_output"][0]
            scores, indices = torch.topk(probabilities, args.top_k)
            result = {
                "audio": str(path),
                "model": "LeeNet11 FP32",
                "checkpoint_iteration": iteration,
                "score_type": "independent sigmoid scores",
                "top_events": [
                    {"class_index": index, "label": labels[index], "score": score}
                    for score, index in zip(scores.tolist(), indices.tolist())
                ],
            }
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
