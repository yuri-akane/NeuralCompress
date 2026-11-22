"""
Descript Audio Codec (DAC) implementation.
"""

import torch
from .base import BaseCodecCompressor


class DACCompressor(BaseCodecCompressor):
    def __init__(
        self,
        device: torch.device,
        n_quantizers: int = 9,
        model_type: str = "44khz",
        **kwargs,
    ):
        self.n_quantizers = int(n_quantizers)
        self.model_type = model_type
        self.name = f"DAC_{self.model_type}_q{self.n_quantizers}"
        super().__init__(device)

    def _load_model(self):
        import dac

        model_path = dac.utils.download(model_type=self.model_type)
        self.model = dac.DAC.load(model_path).to(self.device)
        self.model.eval()

    def target_sample_rate(self) -> int:
        if self.model_type == "44khz":
            return 44100
        elif self.model_type == "24khz":
            return 24000
        elif self.model_type == "16khz":
            return 16000
        return 44100

    def bits_per_token(self) -> int:
        return 10

    @torch.no_grad()
    def encode_chunk(self, chunk_wav: torch.Tensor) -> torch.Tensor:
        wav = self.model.preprocess(chunk_wav, self.target_sample_rate())
        _, codes, _, _, _ = self.model.encode(wav, n_quantizers=self.n_quantizers)
        codes = codes.squeeze(0)
        # Slicing to desired number of codebooks
        if codes.shape[0] > self.n_quantizers:
            codes = codes[: self.n_quantizers, :]
        return codes

    @torch.no_grad()
    def decode_chunk(self, chunk_codes: torch.Tensor) -> torch.Tensor:
        codes = chunk_codes.unsqueeze(0).to(self.device)
        z = self.model.quantizer.from_codes(codes)[0]
        audio = self.model.decode(z)
        return audio.squeeze(0)
