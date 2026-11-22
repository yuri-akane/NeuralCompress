"""
SNAC (Multi-Scale Neural Audio Codec) implementation.
Supports 24kHz, 32kHz, and 44kHz models by Hubert Siuzdak.
"""

import math
from typing import List, Optional
import torch
from .base import BaseCodecCompressor


class SNACCompressor(BaseCodecCompressor):
    def __init__(
        self,
        device: torch.device,
        model_type: str = "32khz",
        **kwargs,
    ):
        self.model_type = model_type.lower()
        if "24" in self.model_type:
            self._sr = 24000
            self.hf_model_id = "hubertsiuzdak/snac_24khz"
            self.name = "SNAC_24khz"
        elif "44" in self.model_type:
            self._sr = 44100
            self.hf_model_id = "hubertsiuzdak/snac_44khz"
            self.name = "SNAC_44khz"
        else:
            self._sr = 32000
            self.hf_model_id = "hubertsiuzdak/snac_32khz"
            self.name = "SNAC_32khz"

        super().__init__(device)

    def _load_model(self):
        from snac import SNAC

        self.model = SNAC.from_pretrained(self.hf_model_id).to(self.device)
        self.model.eval()

        # Dynamic token rates based on vq_strides
        # e.g., 24kHz: [4, 2, 1] -> rates: [1, 2, 4] (sum=7 tokens/frame)
        # e.g., 32kHz/44kHz: [8, 4, 2, 1] -> rates: [1, 2, 4, 8] (sum=15 tokens/frame)
        base_stride = self.model.vq_strides[0]
        self.rates: List[int] = [base_stride // s for s in self.model.vq_strides]
        self.total_tokens_per_frame: int = sum(self.rates)

        # Compute padding alignment unit: hop_length * lcm(vq_strides[0], attn_window_size or 1)
        lcm = math.lcm(self.model.vq_strides[0], self.model.attn_window_size or 1)
        self.pad_to = self.model.hop_length * lcm

    def target_sample_rate(self) -> int:
        return self._sr

    def bits_per_token(self) -> int:
        # Codebook vocabulary size is 4096 (12 bits)
        return 12

    @torch.no_grad()
    def encode_chunk(self, chunk_wav: torch.Tensor) -> torch.Tensor:
        """
        Encode audio chunk into multi-scale packed 2D tensor.
        Output shape: [total_tokens_per_frame, T] where tokens per time frame T
        are temporally interleaved according to self.rates:
          - 24kHz (3 levels): 7 tokens/frame (rates 1, 2, 4)
          - 32kHz / 44kHz (4 levels): 15 tokens/frame (rates 1, 2, 4, 8)
        """
        codes: List[torch.Tensor] = self.model.encode(chunk_wav)
        T = codes[0].shape[-1]

        level_tensors = []
        for c, r in zip(codes, self.rates):
            if r == 1:
                level_tensors.append(c.unsqueeze(1))
            else:
                level_tensors.append(c.reshape(1, T, r).permute(0, 2, 1))

        packed = torch.cat(level_tensors, dim=1).squeeze(0)  # [total_tokens_per_frame, T]
        return packed

    @torch.no_grad()
    def decode_chunk(self, chunk_codes: torch.Tensor) -> torch.Tensor:
        """
        Unpack [total_tokens_per_frame, T] temporally interleaved tensor back to
        multi-scale code levels and decode.
        """
        packed = chunk_codes.unsqueeze(0).to(self.device)  # [1, total_tokens_per_frame, T]
        T = packed.shape[-1]

        unpacked_codes = []
        idx = 0
        for r in self.rates:
            if r == 1:
                u = packed[:, idx : idx + 1, :].reshape(1, T)
            else:
                u = packed[:, idx : idx + r, :].permute(0, 2, 1).reshape(1, r * T)
            unpacked_codes.append(u)
            idx += r

        rec = self.model.decode(unpacked_codes)  # [1, 1, samples]
        return rec.squeeze(0)

    def encode(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        chunk_sec: float,
        mono: bool = False,
    ) -> torch.Tensor:
        """
        Encode audio with chunk sample count aligned to pad_to (e.g. 12288 samples)
        to prevent intermediate 0-padding corruptions at chunk boundaries.
        """
        if chunk_sec is None or chunk_sec <= 0:
            return super().encode(waveform, sample_rate, chunk_sec=0, mono=mono)

        target_sr = self.target_sample_rate()
        chunk_samples = int(chunk_sec * target_sr)
        if hasattr(self, "pad_to") and self.pad_to > 0:
            chunk_samples = max(self.pad_to, (chunk_samples // self.pad_to) * self.pad_to)
        aligned_chunk_sec = chunk_samples / target_sr

        return super().encode(waveform, sample_rate, chunk_sec=aligned_chunk_sec, mono=mono)
