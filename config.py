"""
Configuration module for NeuralCompress.
Provides unified configuration loaded from constants.py (default values)
and overridden by CLI options (highest priority).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import constants


@dataclass
class GeneralConfig:
    input_dir: str = constants.input_dir
    output_csv: str = constants.output_csv
    compressed_dir: str = constants.compressed_dir
    reconstructed_dir: str = constants.reconstructed_dir
    reconstruct: bool = constants.reconstruct
    save_nc: bool = constants.save_nc
    force_mono: bool = constants.force_mono
    enabled_codecs: List[str] = field(default_factory=lambda: list(constants.enabled_codecs))
    chunk_sec: float = constants.chunk_sec
    chunk_frames: int = constants.chunk_frames
    overlap_frames: int = constants.overlap_frames


@dataclass
class ContainerConfig:
    compress_nc: bool = constants.compress_nc
    lzma_preset: int = constants.lzma_preset


@dataclass
class AppConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    container: ContainerConfig = field(default_factory=ContainerConfig)
    codecs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        k: dict(v) for k, v in getattr(constants, "codec_params", {}).items()
    })

    def get_codec_params(self, codec_name: str) -> Dict[str, Any]:
        """Returns codec-specific parameters dict."""
        return self.codecs.get(codec_name.lower(), {})


def resolve_codec_params(codec_name: str, args: Any = None) -> Dict[str, Any]:
    """
    Resolve codec-specific parameters:
    1. CLI options (highest priority)
    2. constants.py codec_params (defaults)
    """
    name = codec_name.lower()
    base_params = dict(getattr(constants, "codec_params", {}).get(name, {}))

    if args is not None:
        if name == "encodec":
            bw = getattr(args, "encodec_bw", None)
            if bw is not None:
                base_params["bandwidth"] = bw
        elif name == "dac":
            nq = getattr(args, "dac_nq", None)
            if nq is not None:
                base_params["n_quantizers"] = nq
        elif name == "snac":
            model = getattr(args, "snac_model", None)
            if model is not None:
                base_params["model_type"] = model

    return base_params


def load_config(args: Any = None) -> AppConfig:
    """
    Load AppConfig with priority:
    1. CLI options (if provided via args)
    2. constants.py (default values)
    """
    config = AppConfig()

    if args is not None:
        # Override general settings if present in CLI args
        input_target = getattr(args, "input_dir", None) or getattr(args, "input", None)
        if input_target is not None:
            config.general.input_dir = str(input_target)
        if getattr(args, "output_csv", None) is not None:
            config.general.output_csv = args.output_csv
        
        codecs_arg = getattr(args, "codecs", None)
        if codecs_arg is None and getattr(args, "codec", None) is not None:
            codecs_arg = args.codec
        if codecs_arg is not None:
            from neural_codecs import normalize_codecs
            try:
                parsed_codecs = normalize_codecs(codecs_arg)
                if parsed_codecs:
                    config.general.enabled_codecs = parsed_codecs
            except ValueError as e:
                import sys
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(2)

        if getattr(args, "force_mono", None) is not None:
            config.general.force_mono = bool(args.force_mono)
        elif getattr(args, "mono", None) is not None:
            config.general.force_mono = bool(args.mono)

        # Container settings
        if getattr(args, "raw", False):
            config.container.compress_nc = False
        if getattr(args, "lzma_preset", None) is not None:
            config.container.lzma_preset = args.lzma_preset

        # Codec params overrides
        for c_name in list(config.codecs.keys()):
            config.codecs[c_name] = resolve_codec_params(c_name, args)

    return config
