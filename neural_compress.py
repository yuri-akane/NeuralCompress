#!/usr/bin/env python3
"""
NeuralCompress: Neural Audio Compression Suite
CLI utility for encoding, decoding, and benchmarking audio using neural audio codecs.
Supports single-file compression and multi-file solid archives (.nc).
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1" #offlineのときは行頭#をはずす


from pathlib import Path
import sys
import time
from typing import List, Optional

from audio_decoder import (
    collect_audio_files,
    collect_text_files,
    get_supported_extensions,
    get_supported_text_extensions,
    load_audio_file,
)
from config import AppConfig, load_config, resolve_codec_params
from container import load_nc_archive, save_nc_archive
from neural_codecs import CODEC_REGISTRY, create_codec, get_available_codecs, normalize_codecs
from constants import chunk_sec, chunk_frames
from cli import run_cli

def get_torch_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Encode Command
# ---------------------------------------------------------------------------

def cmd_encode(args, config: AppConfig):
    try:
        import torch
    except ImportError as e:
        print(f"Error: Missing dependency ({e}). Please install: pip install torch torchaudio soundfile")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input '{input_path}' not found.")
        sys.exit(1)

    device = get_torch_device()
    raw_codec = args.codec or (config.general.enabled_codecs[0] if config.general.enabled_codecs else "encodec")
    try:
        norm_codecs = normalize_codecs(raw_codec)
        codec_name = norm_codecs[0] if norm_codecs else "encodec"
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)

    # Codec params: CLI option > constants.py default
    params = resolve_codec_params(codec_name, args)

    # Collect audio and text files
    valid_text_exts = {ext.lower() for ext in get_supported_text_extensions()}
    if input_path.is_file():
        if input_path.suffix.lower() in valid_text_exts:
            files = []
            text_files = [input_path]
        else:
            files = [input_path]
            text_files = []
        folder_name = None
    else:
        files = collect_audio_files(input_path)
        text_files = collect_text_files(input_path)
        folder_name = input_path.name

    if not files and not text_files:
        print(f"No supported audio or text files found in '{input_path}'.")
        return

    print(f"Initializing codec '{codec_name}' on {device}...")
    compressor = create_codec(codec_name, device=device, **params)

    output_arg = Path(args.output) if args.output else None

    # Determine archiving mode
    # If folder or multiple files, default to solid archive
    total_file_count = len(files) + len(text_files)
    is_multi_file = total_file_count > 1 or bool(text_files) or input_path.is_dir()
    archive_mode = is_multi_file and (args.archive or (output_arg and output_arg.suffix.lower() == ".nc") or (not args.separate))

    if archive_mode:
        if output_arg and output_arg.suffix.lower() == ".nc":
            archive_path = output_arg
        elif output_arg:
            output_arg.mkdir(parents=True, exist_ok=True)
            archive_path = output_arg / f"{input_path.stem}_{compressor.name}.nc"
        else:
            archive_path = input_path.parent / f"{input_path.name}_{compressor.name}.nc" if input_path.is_dir() else input_path.with_suffix(f".{compressor.name}.nc")

        print(f"\n[Archive Encode] Packaging {len(files)} audio file(s) and {len(text_files)} text file(s) into single archive: {archive_path.name}")
        
        # Read text files (raw binary preserved)
        text_files_data = []
        total_text_size = 0
        for t_file in text_files:
            t_data = t_file.read_bytes()
            t_size = len(t_data)
            total_text_size += t_size
            rel_path = t_file.relative_to(input_path) if input_path.is_dir() else Path(t_file.name)
            print(f" -> Including text: {rel_path.as_posix()} ({t_size / 1024:.1f} KB)")
            text_files_data.append({
                "filename": rel_path.as_posix(),
                "data": t_data,
            })

        encoded_files_data = []
        total_orig_size = total_text_size
        total_duration = 0.0
        t_start = time.time()

        for audio_file in files:
            orig_size = audio_file.stat().st_size
            waveform, orig_sr = load_audio_file(audio_file)
            orig_samples = waveform.shape[-1]
            channels = waveform.shape[0]
            duration = orig_samples / orig_sr
            total_orig_size += orig_size
            total_duration += duration

            rel_path = audio_file.relative_to(input_path) if input_path.is_dir() else Path(audio_file.name)
            print(f" -> Encoding audio: {rel_path.as_posix()} ({duration:.2f}s, {channels}ch, {orig_size / 1024:.1f} KB)...")
            codes = compressor.encode(
                waveform,
                orig_sr,
                chunk_sec=config.general.chunk_sec,
                mono=args.mono if args.mono is not None else config.general.force_mono,
            )

            encoded_files_data.append({
                "filename": rel_path.as_posix(),
                "codes": codes,
                "original_sr": orig_sr,
                "channels": codes.shape[0],
                "original_samples": orig_samples,
            })

        t_encode = time.time() - t_start
        archive_size = save_nc_archive(
            filepath=archive_path,
            files_data=encoded_files_data,
            codec_name=codec_name,
            codec_params=params,
            bits_per_token=compressor.bits_per_token(),
            sample_rate=compressor.target_sample_rate(),
            folder_name=folder_name,
            text_files_data=text_files_data,
            compress=not getattr(args, "raw", False),
            preset=getattr(args, "lzma_preset", None),
        )

        ratio = (archive_size / total_orig_size) * 100 if total_orig_size > 0 else 0
        bitrate = (archive_size * 8) / (total_duration * 1000) if total_duration > 0 else 0
        print(f"\n[Complete] Archive saved: {archive_path.resolve()}")
        print(f" -> Audio: {len(files)}, Text: {len(text_files)}, Total Original Size: {total_orig_size / 1024:.1f} KB")
        print(f" -> Archive Size: {archive_size / 1024:.1f} KB (Ratio: {ratio:.2f}%, Overall Bitrate: {bitrate:.1f} kbps, Time: {t_encode:.2f}s)")

    else:
        # Separate 1-to-1 encoding
        for audio_file in files:
            if output_arg and input_path.is_file():
                out_file = output_arg
            elif output_arg and input_path.is_dir():
                rel_subdir = audio_file.relative_to(input_path).parent
                sub_out_dir = output_arg / rel_subdir
                sub_out_dir.mkdir(parents=True, exist_ok=True)
                out_file = sub_out_dir / f"{audio_file.stem}_{compressor.name}.nc"
            else:
                out_file = audio_file.with_suffix(f".{compressor.name}.nc")

            orig_size = audio_file.stat().st_size
            waveform, orig_sr = load_audio_file(audio_file)
            orig_samples = waveform.shape[-1]
            channels = waveform.shape[0]
            duration = orig_samples / orig_sr

            print(f"\n[Encode] {audio_file.name} ({duration:.2f}s, {channels}ch, {orig_size / 1024:.1f} KB)")
            t0 = time.time()
            codes = compressor.encode(
                waveform,
                orig_sr,
                chunk_sec=config.general.chunk_sec,
                mono=args.mono if args.mono is not None else config.general.force_mono,
            )
            encode_time = time.time() - t0

            packed_size = save_nc_archive(
                filepath=out_file,
                files_data=[{
                    "filename": audio_file.name,
                    "codes": codes,
                    "original_sr": orig_sr,
                    "channels": codes.shape[0],
                    "original_samples": orig_samples,
                }],
                codec_name=codec_name,
                codec_params=params,
                bits_per_token=compressor.bits_per_token(),
                sample_rate=compressor.target_sample_rate(),
                folder_name=folder_name,
                compress=not getattr(args, "raw", False),
                preset=getattr(args, "lzma_preset", None),
            )

            ratio = (packed_size / orig_size) * 100 if orig_size > 0 else 0
            bitrate = (packed_size * 8) / (duration * 1000) if duration > 0 else 0
            print(f" -> Saved to: {out_file.resolve()}")
            print(f" -> Size: {packed_size / 1024:.1f} KB (Ratio: {ratio:.2f}%, Bitrate: {bitrate:.1f} kbps, Time: {encode_time:.2f}s)")


# ---------------------------------------------------------------------------
# Decode Command
# ---------------------------------------------------------------------------

def cmd_decode(args, config: AppConfig):
    try:
        import soundfile as sf
        import torch
    except ImportError as e:
        print(f"Error: Missing dependency ({e}). Please install: pip install torch torchaudio soundfile")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input '{input_path}' not found.")
        sys.exit(1)

    device = get_torch_device()
    files = [input_path] if input_path.is_file() else sorted(list(input_path.glob("*.nc")))
    if not files:
        print(f"No .nc files found in '{input_path}'.")
        return

    output_arg = Path(args.output) if args.output else None
    compressor_cache = {}

    for nc_file in files:
        unpacked_audio_files, unpacked_text_files, metadata = load_nc_archive(nc_file)
        raw_codec_name = metadata.get("codec_name", "encodec").lower()
        codec_name = raw_codec_name if raw_codec_name in CODEC_REGISTRY else raw_codec_name.split("_")[0]
        codec_params = metadata.get("codec_params", {})
        folder_name = metadata.get("folder_name")
        total_audio = len(unpacked_audio_files)
        total_text = len(unpacked_text_files)
        is_multi_in_archive = (total_audio + total_text) > 1

        # Determine target directory
        if output_arg and not is_multi_in_archive and output_arg.suffix.lower() == ".wav" and not folder_name:
            # Single file explicit output path
            target_dir = output_arg.parent
        elif output_arg:
            base_dir = output_arg
            target_dir = base_dir / folder_name if folder_name else base_dir
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            base_dir = nc_file.parent
            if folder_name:
                target_dir = base_dir / folder_name
            elif is_multi_in_archive:
                # Fallback to avoid scattering when multiple tracks exist in legacy archive
                target_dir = base_dir / nc_file.stem
            else:
                target_dir = base_dir
            target_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Decode Archive] {nc_file.name} ({total_audio} audio track(s), {total_text} text file(s), Codec: {codec_name}) -> Output dir: {target_dir}")

        # 1. Restore text files
        for t_item in unpacked_text_files:
            track_rel = Path(t_item["filename"])
            out_text_path = target_dir / track_rel
            out_text_path.parent.mkdir(parents=True, exist_ok=True)
            out_text_path.write_bytes(t_item["data"])
            print(f" -> Restored text: {track_rel.as_posix()} ({len(t_item['data']) / 1024:.1f} KB)")

        # 2. Decode audio files
        if unpacked_audio_files:
            cache_key = (codec_name, str(sorted(codec_params.items())))
            if cache_key not in compressor_cache:
                print(f"Initializing codec '{codec_name}' from container metadata on {device}...")
                compressor_cache[cache_key] = create_codec(codec_name, device=device, **codec_params)
            compressor = compressor_cache[cache_key]

            for item in unpacked_audio_files:
                track_name = item["filename"]
                codes = item["codes"]
                original_samples = item.get("original_samples", 0)
                original_sr = item.get("original_sr", compressor.target_sample_rate())

                target_samples = (
                    round(
                        original_samples
                        * compressor.target_sample_rate()
                        / original_sr
                    )
                    if original_samples > 0 and original_sr > 0 else None
                )

                if output_arg and not is_multi_in_archive and output_arg.suffix.lower() in [".wav", ".ogg", ".flac"] and not folder_name:
                    out_audio_path = output_arg
                else:
                    track_rel = Path(track_name)
                    out_sub_dir = target_dir / track_rel.parent
                    out_sub_dir.mkdir(parents=True, exist_ok=True)
                    out_audio_path = out_sub_dir / f"{track_rel.stem}.ogg"

                t0 = time.time()
                rec_wav = compressor.decode(
                    codes,
                    chunk_frames=config.general.chunk_frames,
                    samples=target_samples,
                    overlap_frames=config.general.overlap_frames,
                )
                decode_time = time.time() - t0

                # Ensure C-contiguous memory layout
                audio_data = (
                    rec_wav.squeeze(0).cpu().numpy()
                    if rec_wav.shape[0] == 1
                    else rec_wav.t().contiguous().cpu().numpy()
                )
                channels = rec_wav.shape[0]
                sr = compressor.target_sample_rate()

                if out_audio_path.suffix.lower() == ".ogg":
                    # Stream in blocks (10s) to prevent libsndfile buffer overflow / Segfault on long audio
                    chunk_samples = sr * 10
                    with sf.SoundFile(
                        str(out_audio_path),
                        mode="w",
                        samplerate=sr,
                        channels=channels,
                        format="OGG",
                        compression_level=0.3,
                    ) as f:
                        for start_idx in range(0, len(audio_data), chunk_samples):
                            f.write(audio_data[start_idx : start_idx + chunk_samples])
                else:
                    sf.write(str(out_audio_path), audio_data, sr)

                print(f" -> Restored audio: {out_audio_path.name} ({decode_time:.2f}s)")


# ---------------------------------------------------------------------------
# Benchmark Command
# ---------------------------------------------------------------------------

def cmd_bench(args, config: AppConfig):
    from benchmark import run_benchmark
    run_benchmark(config=config)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    handlers = {
        "encode": cmd_encode,
        "decode": cmd_decode,
        "bench": cmd_bench,
    }
    run_cli(handlers)


if __name__ == "__main__":
    main()
