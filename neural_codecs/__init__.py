"""
Neural Audio Codecs package.
Provides registry and lazy factory function for supported audio codecs.
"""

from typing import Any, Dict, List

# Registry mapping: codec identifier -> (module_path, class_name)
CODEC_REGISTRY: Dict[str, str] = {
    "encodec": "neural_codecs.encodec.EncodecCompressor",
    "dac": "neural_codecs.dac.DACCompressor",
    "snac": "neural_codecs.snac.SNACCompressor",
}


def get_available_codecs() -> List[str]:
    """Returns list of all registered codec names."""
    return list(CODEC_REGISTRY.keys())


def normalize_codecs(codecs_input: Any) -> List[str]:
    """
    Normalize codec names from user inputs:
    - comma-separated string: 'snac,dac' or 'snac, dac'
    - list of strings: ['snac,dac', 'encodec']
    - handles case-insensitivity (e.g., 'SNAC' -> 'snac')
    - validates against CODEC_REGISTRY and removes duplicates while preserving order
    """
    if not codecs_input:
        return []

    if isinstance(codecs_input, str):
        raw_items = [part.strip() for part in codecs_input.split(",") if part.strip()]
    else:
        raw_items = []
        for item in codecs_input:
            if isinstance(item, str):
                raw_items.extend([part.strip() for part in item.split(",") if part.strip()])
            else:
                raw_items.append(str(item).strip())

    available = get_available_codecs()
    available_map = {c.lower(): c for c in available}
    result = []
    for item in raw_items:
        clean = item.strip().lower()
        if not clean:
            continue
        if clean not in available_map:
            raise ValueError(f"Unknown codec: '{item}'. Available codecs: {', '.join(available)}")
        canonical = available_map[clean]
        if canonical not in result:
            result.append(canonical)
    return result


def create_codec(name: str, device: Any, **kwargs) -> Any:
    """
    Factory function to instantiate a codec compressor by registry name.
    Dynamically loads the codec module on demand to avoid unnecessary imports.

    Args:
        name: Name of codec registered in CODEC_REGISTRY (e.g., 'encodec', 'dac', 'snac')
        device: torch.device (cpu or cuda)
        **kwargs: Model-specific parameters (e.g., bandwidth, n_quantizers, etc.)
    """
    name_lower = name.lower()
    if name_lower not in CODEC_REGISTRY:
        raise ValueError(f"Unknown codec: '{name}'. Available codecs: {list(CODEC_REGISTRY.keys())}")

    target = CODEC_REGISTRY[name_lower]
    mod_name, cls_name = target.rsplit(".", 1)

    import importlib
    mod = importlib.import_module(mod_name)
    codec_cls = getattr(mod, cls_name)
    return codec_cls(device=device, **kwargs)


__all__ = [
    "CODEC_REGISTRY",
    "get_available_codecs",
    "normalize_codecs",
    "create_codec",
]
