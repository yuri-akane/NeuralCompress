"""
Compression and bitrate metrics calculation module for NeuralCompress (.nc).
Provides theoretical bounds and practical container (.nc) compression metrics.
"""

from typing import Any, Dict


def calculate_compression_metrics(
    codes: Any,
    bits_per_token: int,
    original_size_bytes: int,
    duration_sec: float,
    compressed_size_bytes: int,
) -> Dict[str, Any]:
    """
    Computes theoretical metrics and practical .nc container compression metrics.

    Args:
        codes: Tensor or array of discrete token indices
        bits_per_token: Codebook index representation bits (e.g., 10 for Encodec/DAC, 12 for SNAC)
        original_size_bytes: Size of original uncompressed input file in bytes
        duration_sec: Audio duration in seconds
        compressed_size_bytes: Size of the actual compressed .nc container in bytes

    Returns:
        metrics dict containing tokens, theoretical_bytes, theoretical_kbps, packed_bytes, packed_kbps, ratio_pct, savings_pct
    """
    total_tokens = codes.numel() if hasattr(codes, "numel") else len(codes)

    # Theoretical size based purely on bits/token (information entropy lower bound without container overhead)
    theoretical_bits = total_tokens * bits_per_token
    theoretical_bytes = int((theoretical_bits + 7) // 8)
    theo_bitrate_kbps = (theoretical_bytes * 8) / (duration_sec * 1000) if duration_sec > 0 else 0.0

    # Practical storage size (.nc container with LZMA compression)
    packed_bytes = compressed_size_bytes
    bitrate_kbps = (packed_bytes * 8) / (duration_sec * 1000) if duration_sec > 0 else 0.0

    # Compression ratio: (Compressed / Original) * 100%
    ratio_pct = (packed_bytes / original_size_bytes * 100) if original_size_bytes > 0 else 0.0
    savings_pct = 100.0 - ratio_pct

    return {
        "tokens": total_tokens,
        "theoretical_bytes": theoretical_bytes,
        "theoretical_kbps": round(theo_bitrate_kbps, 2),
        "packed_bytes": packed_bytes,
        "packed_kbps": round(bitrate_kbps, 2),
        "ratio_pct": round(ratio_pct, 2),
        "savings_pct": round(savings_pct, 2),
    }
