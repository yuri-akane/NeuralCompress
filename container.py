"""
NeuralCompress container packaging module (.nc files).
Supports multi-file (solid archive) packaging.

Design:
- Raw .nc contains uncompressed UTF-8 JSON metadata and raw int16 token chunks.
- When compressed (either in Python via lzma or externally via `xz -9e`), the entire
  .nc container (including metadata) is compressed into a standard XZ stream.
- unpack_nc automatically detects XZ magic bytes (b"\\xfd7zXZ\\x00") and decompresses
  transparently, allowing both uncompressed .nc and XZ-compressed .nc files.
"""

import json
import lzma
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import constants

if TYPE_CHECKING:
    import torch

MAGIC_HEADER = b"NCMP"  # Neural CoMPress magic bytes
XZ_MAGIC = b"\xfd7zXZ\x00"  # Standard XZ stream magic bytes
FORMAT_VERSION = 4


def pack_nc(
    files_data: List[Dict[str, Any]],
    codec_name: str,
    codec_params: Dict[str, Any],
    bits_per_token: int,
    sample_rate: int,
    folder_name: Optional[str] = None,
    text_files_data: Optional[List[Dict[str, Any]]] = None,
) -> bytes:
    """
    Pack audio token tensors, optional text files, and metadata into a single uncompressed .nc binary payload.
    No LZMA compression is applied inside this function.
    
    Args:
        files_data: List of dicts, each containing:
            - 'filename': str
            - 'codes': torch.Tensor (discrete tokens)
            - 'original_sr': int
            - 'channels': int
            - 'original_samples': int
        codec_name: str (e.g. 'snac', 'dac', 'encodec')
        codec_params: dict
        bits_per_token: int
        sample_rate: int (codec native sample rate)
        folder_name: Optional[str] (original folder name if compressed from a directory)
        text_files_data: Optional list of dicts, each containing:
            - 'filename': str
            - 'data': bytes
    """
    import torch

    files_metadata = []
    raw_tokens_chunks = bytearray()

    for item in files_data:
        codes = item["codes"]
        shape = list(codes.shape)
        numel = int(codes.numel())

        files_metadata.append({
            "filename": item.get("filename", "audio.wav"),
            "original_sr": item.get("original_sr", sample_rate),
            "channels": item.get("channels", shape[0] if len(shape) > 1 else 1),
            "original_samples": item.get("original_samples", 0),
            "tokens_shape": shape,
            "tokens_numel": numel,
        })

        # Append int16 raw bytes directly without compression
        chunk_bytes = codes.to(torch.int16).cpu().numpy().tobytes()
        raw_tokens_chunks.extend(chunk_bytes)

    # Process text files if any
    text_files_metadata = []
    raw_text_chunks = bytearray()
    if text_files_data:
        for t_item in text_files_data:
            t_data = t_item["data"]
            if isinstance(t_data, str):
                t_data = t_data.encode("utf-8")
            text_files_metadata.append({
                "filename": t_item.get("filename", "unknown.txt"),
                "size": len(t_data),
            })
            raw_text_chunks.extend(t_data)

    metadata = {
        "format_version": FORMAT_VERSION,
        "codec_name": codec_name,
        "codec_params": codec_params,
        "bits_per_token": bits_per_token,
        "sample_rate": sample_rate,
        "files": files_metadata,
    }
    if text_files_metadata:
        metadata["text_files"] = text_files_metadata
    if folder_name:
        metadata["folder_name"] = str(folder_name)

    meta_json = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    meta_len = len(meta_json)

    packed_data = bytearray()
    packed_data.extend(MAGIC_HEADER)
    packed_data.extend(struct.pack("<B", FORMAT_VERSION))
    packed_data.extend(struct.pack("<I", meta_len))
    packed_data.extend(meta_json)
    packed_data.extend(raw_tokens_chunks)
    packed_data.extend(raw_text_chunks)

    return bytes(packed_data)


def unpack_nc(data: bytes) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Unpack .nc binary payload.
    Supports both raw .nc and transparently decompresses XZ-compressed .nc payloads.
    
    Returns:
        (unpacked_audio_files, unpacked_text_files, container_metadata)
        unpacked_audio_files: list of dicts:
            [{
                'filename': str,
                'codes': torch.Tensor,
                'original_sr': int,
                'channels': int,
                'original_samples': int,
                'tokens_shape': list
            }, ...]
        unpacked_text_files: list of dicts:
            [{
                'filename': str,
                'data': bytes
            }, ...]
        container_metadata: dict
    """
    import torch

    # Check if the entire payload is compressed with XZ/LZMA
    if data.startswith(XZ_MAGIC):
        data = lzma.decompress(data)

    if len(data) < 9:
        raise ValueError("Invalid .nc file: file too short.")

    magic = data[:4]
    if magic != MAGIC_HEADER:
        raise ValueError(f"Invalid file format: expected header {MAGIC_HEADER}, got {magic}")

    version = struct.unpack("<B", data[4:5])[0]
    meta_len = struct.unpack("<I", data[5:9])[0]

    meta_end = 9 + meta_len
    if len(data) < meta_end:
        raise ValueError("Corrupt file: header length exceeds file size.")

    meta_json = data[9:meta_end].decode("utf-8")
    metadata = json.loads(meta_json)

    raw_payload_bytes = data[meta_end:]
    unpacked_audio_files = []

    # Calculate total elements of tokens
    audio_files_meta = metadata.get("files", [])
    total_tokens_elements = sum(f["tokens_numel"] for f in audio_files_meta)
    total_tokens_bytes = total_tokens_elements * 2  # int16 is 2 bytes

    tokens_bytes = raw_payload_bytes[:total_tokens_bytes]
    if total_tokens_elements > 0:
        all_tokens_tensor = torch.frombuffer(bytearray(tokens_bytes), dtype=torch.int16)
        offset_elements = 0
        for file_meta in audio_files_meta:
            numel = file_meta["tokens_numel"]
            shape = file_meta["tokens_shape"]
            slice_tensor = all_tokens_tensor[offset_elements : offset_elements + numel].clone()
            codes = slice_tensor.reshape(shape)
            offset_elements += numel

            unpacked_audio_files.append({
                "filename": file_meta.get("filename", "audio.wav"),
                "codes": codes,
                "original_sr": file_meta.get("original_sr", metadata.get("sample_rate", 24000)),
                "channels": file_meta.get("channels", 1),
                "original_samples": file_meta.get("original_samples", 0),
                "tokens_shape": shape,
            })

    # Unpack text files if any
    raw_text_bytes = raw_payload_bytes[total_tokens_bytes:]
    unpacked_text_files = []
    text_files_meta = metadata.get("text_files", [])
    text_offset = 0
    for t_meta in text_files_meta:
        sz = t_meta["size"]
        t_data = raw_text_bytes[text_offset : text_offset + sz]
        text_offset += sz
        unpacked_text_files.append({
            "filename": t_meta.get("filename", "unknown.txt"),
            "data": bytes(t_data),
        })

    return unpacked_audio_files, unpacked_text_files, metadata


def create_nc_payload(
    compress: Optional[bool] = None,
    preset: Optional[int] = None,
    **kwargs,
) -> bytes:
    """
    Pack metadata, audio tokens, and text into .nc payload, optionally compressing with LZMA (XZ).
    """
    data = pack_nc(**kwargs)
    should_compress = compress if compress is not None else constants.compress_nc
    if should_compress:
        lzma_preset = preset if preset is not None else constants.lzma_preset
        data = lzma.compress(data, preset=lzma_preset)
    return data


def save_nc_archive(
    filepath: Path | str,
    compress: Optional[bool] = None,
    preset: Optional[int] = None,
    **kwargs,
) -> int:
    """
    Pack and write archive to disk.
    
    Args:
        filepath: Destination path (.nc).
        compress: Whether to compress the whole container with LZMA (XZ).
                  Defaults to constants.compress_nc.
        preset: LZMA compression preset (0-9). Defaults to constants.lzma_preset.
        **kwargs: Arguments passed to pack_nc().
    """
    data = create_nc_payload(compress=compress, preset=preset, **kwargs)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def load_nc_archive(filepath: Path | str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Read archive from disk and unpack (handles both raw and XZ-compressed .nc)."""
    path = Path(filepath)
    with open(path, "rb") as f:
        data = f.read()
    return unpack_nc(data)
