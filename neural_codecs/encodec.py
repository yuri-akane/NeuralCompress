"""
Meta EnCodec implementation.
"""

import torch
from .base import BaseCodecCompressor


class EncodecCompressor(BaseCodecCompressor):
    def __init__(
        self,
        device: torch.device,
        bandwidth: float = 24.0,
        model_name: str = "facebook/encodec_24khz",
        **kwargs,
    ):
        self.bandwidth = float(bandwidth)
        self.model_name = model_name
        bw_str = int(self.bandwidth) if self.bandwidth.is_integer() else self.bandwidth
        self.name = f"EnCodec_{bw_str}kbps"
        super().__init__(device)

    def _load_model(self):
        from transformers import EncodecModel

        self.model = EncodecModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def target_sample_rate(self) -> int:
        return 24000

    def bits_per_token(self) -> int:
        return 10

    @torch.no_grad()
    def encode_chunk(self, chunk_wav: torch.Tensor) -> torch.Tensor:
        encoded_frames = self.model.encode(chunk_wav, bandwidth=self.bandwidth)
        codes = encoded_frames.audio_codes  # shape: [batch, n_codebooks, n_frames]
        return codes.squeeze(0).squeeze(0) if codes.ndim > 2 and codes.shape[0] == 1 else codes.squeeze(0)

    @torch.no_grad()
    def decode_chunk(self, chunk_codes: torch.Tensor) -> torch.Tensor:
        codes = chunk_codes.unsqueeze(0).unsqueeze(0).to(self.device)
        output = self.model.decode(codes, [None])
        return output.audio_values.squeeze(0)
