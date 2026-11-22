"""
Base interface and common chunked streaming logic for neural audio codecs.
"""

import abc
from typing import Optional
import torch
import torchaudio

# Compatibility patch for NumPy 2.x and legacy dependencies (e.g. DAC tensorboard)
try:
    import numpy as np
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_
except ImportError:
    pass


class BaseCodecCompressor(abc.ABC):
    """Abstract base class for all neural audio codecs."""

    name: str = "base"

    def __init__(self, device: torch.device, **kwargs):
        self.device = device
        self._load_model()

    @abc.abstractmethod
    def _load_model(self):
        """Initialize and load the model weights."""
        pass

    @abc.abstractmethod
    def target_sample_rate(self) -> int:
        """Required audio sampling rate for this codec."""
        pass

    @abc.abstractmethod
    def bits_per_token(self) -> int:
        """Number of bits used to represent a single codebook index."""
        pass

    @abc.abstractmethod
    def encode_chunk(self, chunk_wav: torch.Tensor) -> torch.Tensor:
        """
        Encode a single short audio chunk.

        Args:
            chunk_wav: Tensor of shape [1, 1, samples] at target_sample_rate()
        Returns:
            Tensor of codes of shape [codebooks, frames]
        """
        pass

    @abc.abstractmethod
    def decode_chunk(self, chunk_codes: torch.Tensor) -> torch.Tensor:
        """
        Decode a single chunk of discrete codes back to raw waveform.

        Args:
            chunk_codes: Tensor of shape [codebooks, frames]
        Returns:
            Tensor of waveform of shape [1, samples] at target_sample_rate()
        """
        pass

    def prepare_audio(self, waveform: torch.Tensor, orig_sr: int, mono: bool = False) -> torch.Tensor:
        """Utility to resample and channel-convert input audio."""
        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        target_sr = self.target_sample_rate()
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            waveform = resampler(waveform)
        return waveform

    def encode(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        chunk_sec: float,
        mono: bool = False,
    ) -> torch.Tensor:
        """
        Encode audio in streaming chunks to maintain bounded memory consumption O(1).
        If chunk_sec <= 0, processes the entire audio in a single pass without chunking.

        Args:
            waveform: Tensor of shape [channels, samples]
            sample_rate: sampling rate of input waveform
            chunk_sec: chunk size in seconds (<= 0 for no chunking)
            mono: If True, mixdown to mono before encoding
        Returns:
            Tensor of codes of shape [channels, codebooks, total_frames]
        """
        wav = self.prepare_audio(waveform, sample_rate, mono=mono)
        num_channels = wav.shape[0]
        total_samples = wav.shape[-1]

        if chunk_sec is None or chunk_sec <= 0:
            # Whole audio encoding in a single pass
            ch_codes = []
            for ch in range(num_channels):
                ch_wav = wav[ch : ch + 1].unsqueeze(0).to(self.device)  # [1, 1, samples]
                codes = self.encode_chunk(ch_wav)
                ch_codes.append(codes.cpu())
            return torch.stack(ch_codes, dim=0)

        target_sr = self.target_sample_rate()
        chunk_samples = int(chunk_sec * target_sr)

        ch_codes = []
        for ch in range(num_channels):
            ch_wav = wav[ch : ch + 1]  # [1, samples]
            all_codes = []
            for start_idx in range(0, total_samples, chunk_samples):
                end_idx = min(start_idx + chunk_samples, total_samples)
                chunk = ch_wav[:, start_idx:end_idx].unsqueeze(0).to(self.device)  # [1, 1, samples]
                codes = self.encode_chunk(chunk)
                all_codes.append(codes.cpu())
            ch_codes.append(torch.cat(all_codes, dim=-1))  # [codebooks, total_frames]

        return torch.stack(ch_codes, dim=0)  # [channels, codebooks, total_frames]

    def decode(
        self,
        codes: torch.Tensor,
        chunk_frames: int,
        samples: Optional[int] = None,
        overlap_frames: int = 32,
    ) -> torch.Tensor:
        """
        Decode codes back to audio waveform in chunks to prevent OOM.
        Uses Overlap-Add (OLA) with crossfading to eliminate chunk boundary click noise.
        If chunk_frames <= 0, decodes the entire sequence in a single pass.

        Args:
            codes: Tensor of shape [channels, codebooks, total_frames] or [codebooks, total_frames]
            chunk_frames: Number of frames per chunk during decode (<= 0 for no chunking)
            samples: Optional target number of audio samples (at target_sample_rate).
                     If specified, cuts off trailing padding introduced by chunking or codec internal padding.
            overlap_frames: Number of frames to overlap between adjacent chunks for crossfading.
        Returns:
            Tensor of waveform of shape [channels, samples]
        """
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)  # [1, codebooks, total_frames]

        num_channels = codes.shape[0]
        total_frames = codes.shape[-1]
        decoded_channels = []

        if chunk_frames is None or chunk_frames <= 0 or total_frames <= chunk_frames:
            # Whole audio decoding in a single pass
            for ch in range(num_channels):
                single_ch_codes = codes[ch].to(self.device)
                decoded_chunk = self.decode_chunk(single_ch_codes.long())
                decoded_channels.append(decoded_chunk.squeeze(0).cpu())
        else:
            step = chunk_frames
            overlap = min(overlap_frames, step // 4) if overlap_frames > 0 else 0

            for ch in range(num_channels):
                single_ch_codes = codes[ch]  # [codebooks, total_frames]

                if overlap <= 0:
                    # Fallback to simple concatenation if overlap is disabled
                    all_wavs = []
                    for start_idx in range(0, total_frames, step):
                        end_idx = min(start_idx + step, total_frames)
                        chunk = single_ch_codes[:, start_idx:end_idx].to(self.device)
                        decoded_chunk = self.decode_chunk(chunk.long())
                        all_wavs.append(decoded_chunk.squeeze(0).cpu())
                    decoded_channels.append(torch.cat(all_wavs, dim=-1))
                    continue

                # Estimate hop length from first chunk
                sample_chunk = single_ch_codes[:, :min(total_frames, 10)].to(self.device)
                sample_dec = self.decode_chunk(sample_chunk.long())
                hop = max(1, sample_dec.shape[-1] // sample_chunk.shape[-1])

                est_total_samples = total_frames * hop
                out_wav = torch.zeros(est_total_samples, dtype=torch.float32)
                weights = torch.zeros(est_total_samples, dtype=torch.float32)

                n_chunks = (total_frames + step - 1) // step
                for k in range(n_chunks):
                    s_k = k * step
                    e_k = min((k + 1) * step, total_frames)

                    ctx_left = overlap if k > 0 else 0
                    ctx_right = overlap if e_k < total_frames else 0

                    chunk_start_frame = s_k - ctx_left
                    chunk_end_frame = e_k + ctx_right
                    chunk = single_ch_codes[:, chunk_start_frame:chunk_end_frame].to(self.device)
                    decoded_chunk = self.decode_chunk(chunk.long()).squeeze(0).cpu()
                    chunk_samples = decoded_chunk.shape[-1]

                    w = torch.ones(chunk_samples, dtype=torch.float32)
                    if k > 0:
                        fade_in_len = min(2 * overlap * hop, chunk_samples)
                        w[:fade_in_len] = torch.linspace(0.0, 1.0, fade_in_len)
                    if e_k < total_frames:
                        fade_out_len = min(2 * overlap * hop, chunk_samples)
                        w[-fade_out_len:] = torch.linspace(1.0, 0.0, fade_out_len)

                    start_samp = chunk_start_frame * hop
                    end_samp = start_samp + chunk_samples

                    if end_samp > out_wav.shape[-1]:
                        pad_len = end_samp - out_wav.shape[-1]
                        out_wav = torch.cat([out_wav, torch.zeros(pad_len)])
                        weights = torch.cat([weights, torch.zeros(pad_len)])

                    out_wav[start_samp:end_samp] += decoded_chunk * w
                    weights[start_samp:end_samp] += w

                # Normalize by weights (in non-zero areas)
                valid_mask = weights > 1e-4
                out_wav[valid_mask] /= weights[valid_mask]
                decoded_channels.append(out_wav)

        wav = torch.stack(decoded_channels, dim=0)  # [channels, samples]
        if samples is not None and samples > 0:
            wav = wav[..., :samples]
        return wav
