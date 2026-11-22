"""
audio_decoder.py - Multi-format audio file loader for NeuralCompress.

Supports WAV, FLAC, OGG (Vorbis), and MP3 formats.
Decodes any supported format to a float32 PCM waveform tensor.

Backend priority:
  1. soundfile  - fast and stable for WAV / FLAC / OGG
  2. torchaudio - fallback for MP3 and other formats (requires ffmpeg or sox)
"""

from pathlib import Path
from typing import Tuple

import torch


# Supported audio file extensions (lowercase and uppercase)
SUPPORTED_EXTENSIONS: set[str] = {
    ".wav", ".WAV",
    ".flac", ".FLAC",
    ".ogg", ".OGG",
    ".mp3", ".MP3",
}

# Supported text file extensions (e.g. BMS, BME, BML, PMS, BMSON chart files)
SUPPORTED_TEXT_EXTENSIONS: set[str] = {
    ".bms", ".BMS",
    ".bme", ".BME",
    ".bml", ".BML",
    ".pms", ".PMS",
    ".bmson", ".BMSON",
    ".txt", ".TXT",
}


def get_supported_extensions() -> set[str]:
    """Return the set of supported audio file extensions."""
    return SUPPORTED_EXTENSIONS


def get_supported_text_extensions() -> set[str]:
    """Return the set of supported text file extensions."""
    return SUPPORTED_TEXT_EXTENSIONS


def load_audio_file(path: Path) -> Tuple[torch.Tensor, int]:
    """
    Load any supported audio file and return a float32 PCM waveform tensor.

    Attempts to decode using soundfile first (fast, WAV/FLAC/OGG).
    Falls back to torchaudio for formats soundfile cannot handle (e.g. MP3).

    Args:
        path: Path to the audio file (.wav / .flac / .ogg / .mp3)

    Returns:
        waveform: torch.Tensor of shape [channels, samples], dtype=float32
        sample_rate: int

    Raises:
        RuntimeError: If the file cannot be decoded by any available backend.
        ValueError:   If the file extension is not in SUPPORTED_EXTENSIONS.
    """
    path = Path(path)

    if path.suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: '{path.suffix}'. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # --- Backend 1: soundfile ------------------------------------------------
    try:
        import soundfile as sf
        audio_data, sample_rate = sf.read(str(path), dtype="float32")
        # soundfile returns [samples] (mono) or [samples, channels] (multi)
        waveform = torch.from_numpy(audio_data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)   # [1, samples]
        else:
            waveform = waveform.t().contiguous()  # [channels, samples]
        return waveform, int(sample_rate)
    except Exception as sf_err:
        # soundfile may not support MP3; try torchaudio next
        pass

    # --- Backend 2: torchaudio -----------------------------------------------
    try:
        import torchaudio
        # torchaudio.load returns ([channels, samples], sample_rate) already
        waveform, sample_rate = torchaudio.load(str(path))
        waveform = waveform.to(torch.float32)
        return waveform, int(sample_rate)
    except Exception as ta_err:
        raise RuntimeError(
            f"Failed to decode '{path.name}'.\n"
            f"  soundfile error : {sf_err}\n"
            f"  torchaudio error: {ta_err}\n"
            f"For MP3 support, ensure ffmpeg is installed (e.g. `sudo apt install ffmpeg`)."
        ) from ta_err


def collect_audio_files(directory: Path, recursive: bool = True) -> list[Path]:
    """
    Collect all supported audio files from a directory.

    Args:
        directory: Path to the directory to scan.
        recursive: If True (default), descend into subdirectories recursively.

    Returns:
        Sorted list of Path objects for supported audio files.
    """
    if recursive:
        files = [
            f for f in directory.rglob("*")
            if f.is_file() and f.suffix in SUPPORTED_EXTENSIONS
        ]
    else:
        files = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix in SUPPORTED_EXTENSIONS
        ]
    return sorted(files)


def collect_text_files(directory: Path, recursive: bool = True) -> list[Path]:
    """
    Collect all supported text files (BMS/BME/BML/PMS/BMSON) from a directory.

    Args:
        directory: Path to the directory to scan.
        recursive: If True (default), descend into subdirectories recursively.

    Returns:
        Sorted list of Path objects for supported text files.
    """
    valid_exts = {ext.lower() for ext in SUPPORTED_TEXT_EXTENSIONS}
    if recursive:
        files = [
            f for f in directory.rglob("*")
            if f.is_file() and f.suffix.lower() in valid_exts
        ]
    else:
        files = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in valid_exts
        ]
    return sorted(files)

