#!/usr/bin/env python3
"""
Neural Audio Codec Compression Benchmark & Audio Reconstruction Tool.
Main entry point for running compression and quality evaluations.
Integrates with .nc containers for realistic, fully decodable compression benchmarks.
"""

import csv
from pathlib import Path
import sys
import time
from typing import List, Optional

from neural_codecs import CODEC_REGISTRY, create_codec, get_available_codecs, normalize_codecs
from config import AppConfig, load_config
from audio_decoder import collect_audio_files, collect_text_files, load_audio_file
from container import create_nc_payload


def run_benchmark(
    config: Optional[AppConfig] = None,
    input_dir_override: Optional[str] = None,
    output_csv_override: Optional[str] = None,
    codecs_override: Optional[List[str]] = None,
    reconstruct_override: Optional[bool] = None,
    save_nc_override: Optional[bool] = None,
    force_mono_override: Optional[bool] = None,
    encodec_bw_override: Optional[float] = None,
    dac_nq_override: Optional[int] = None,
    snac_model_override: Optional[str] = None,
):
    try:
        import torch
        from metrics import calculate_compression_metrics
    except ImportError as e:
        print(f"Error: Missing required library ({e}). Please install requirements: pip install torch torchaudio soundfile")
        sys.exit(1)

    if config is None:
        config = load_config()

    # Resolve settings (CLI overrides defaults from constants.py)
    gen = config.general
    input_dir = input_dir_override or gen.input_dir
    output_csv = output_csv_override or gen.output_csv
    raw_codecs = codecs_override if codecs_override is not None else gen.enabled_codecs
    selected_codecs = normalize_codecs(raw_codecs)
    reconstruct = reconstruct_override if reconstruct_override is not None else gen.reconstruct
    save_nc = save_nc_override if save_nc_override is not None else getattr(gen, "save_nc", True)
    force_mono = force_mono_override if force_mono_override is not None else gen.force_mono

    # Codec parameters
    encodec_params = dict(config.get_codec_params("encodec"))
    if encodec_bw_override is not None:
        encodec_params["bandwidth"] = encodec_bw_override

    dac_params = dict(config.get_codec_params("dac"))
    if dac_nq_override is not None:
        dac_params["n_quantizers"] = dac_nq_override

    snac_params = dict(config.get_codec_params("snac"))
    if snac_model_override is not None:
        snac_params["model_type"] = snac_model_override

    codec_params_map = {
        "encodec": encodec_params,
        "dac": dac_params,
        "snac": snac_params,
    }

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input path '{input_dir}' not found.")
        sys.exit(1)

    if input_path.is_file():
        audio_files = [input_path]
        text_files = []
        folder_name = None
    else:
        audio_files = collect_audio_files(input_path)
        text_files = collect_text_files(input_path)
        folder_name = input_path.name

    if not audio_files and not text_files:
        print(f"No supported audio or text files found in '{input_dir}'.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    total_files_msg = f"{len(audio_files)} audio file(s)"
    if text_files:
        total_files_msg += f", {len(text_files)} text file(s)"
    print(f"Input path: {input_path.resolve()} ({total_files_msg})")
    print(f"Target codecs: {', '.join(selected_codecs)}")
    print(
        f"Settings: EnCodec={encodec_params.get('bandwidth', 24.0)}kbps, "
        f"DAC={dac_params.get('n_quantizers', 9)} quantizers, "
        f"SNAC={snac_params.get('model_type', '44khz')} model"
    )
    compress_status = f"LZMA preset {config.container.lzma_preset}" if config.container.compress_nc else "Raw (Uncompressed)"
    print(f"Container Compression: .nc ({compress_status})")
    print(f"Audio Channels Mode: {'Forced Mono' if force_mono else 'Stereo / Passthrough (Dual-Mono per-channel encode)'}")
    print(f"Compressed Binary (.nc) Output: {'Enabled' if save_nc else 'Disabled'}")
    print(f"Reconstructed Audio Output: {'Enabled' if reconstruct else 'Disabled'}\n")

    reconstruct_dir = Path(gen.reconstructed_dir)
    if reconstruct:
        reconstruct_dir.mkdir(parents=True, exist_ok=True)

    compressed_dir = Path(gen.compressed_dir)
    if save_nc:
        compressed_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate compressors
    compressors = []
    for c_name in selected_codecs:
        c_name_lower = c_name.lower()
        if c_name_lower in CODEC_REGISTRY:
            params = codec_params_map.get(c_name_lower, config.get_codec_params(c_name_lower))
            print(f"Loading {c_name} model with params: {params}...")
            comp = create_codec(c_name_lower, device=device, **params)
            comp.codec_id = c_name_lower
            compressors.append(comp)
        else:
            print(f"Warning: Codec '{c_name}' is not registered. Skipping.")

    if not compressors:
        print("No valid codecs selected. Exiting.")
        return

    print("\n" + "=" * 90)
    print(f"{'Filename':<22} | {'Codec':<14} | {'Orig Size':<10} | {'NC Size':<10} | {'Ratio':<7} | {'Bitrate':<9}")
    print("=" * 90)

    summary_records = []
    codec_encoded_files = {comp.name: [] for comp in compressors}

    for audio_file in audio_files:
        orig_size = audio_file.stat().st_size
        try:
            waveform, sr = load_audio_file(audio_file)
        except Exception as e:
            print(f"Warning: Skipping '{audio_file.name}': {e}")
            continue
        duration = waveform.shape[-1] / sr

        for comp in compressors:
            t0 = time.time()
            codes = comp.encode(waveform, sr, chunk_sec=config.general.chunk_sec, mono=force_mono)
            encode_time = time.time() - t0

            # Prepare metadata for single file container
            file_meta = {
                "filename": audio_file.name,
                "codes": codes,
                "original_sr": sr,
                "channels": codes.shape[0],
                "original_samples": waveform.shape[-1],
            }
            codec_encoded_files[comp.name].append(file_meta)

            codec_id = getattr(comp, "codec_id", comp.name.lower().split("_")[0])
            codec_p = codec_params_map.get(codec_id, config.get_codec_params(codec_id))
            nc_payload = create_nc_payload(
                files_data=[file_meta],
                codec_name=codec_id,
                codec_params=codec_p,
                bits_per_token=comp.bits_per_token(),
                sample_rate=comp.target_sample_rate(),
                compress=config.container.compress_nc,
                preset=config.container.lzma_preset,
            )

            metrics = calculate_compression_metrics(
                codes=codes,
                bits_per_token=comp.bits_per_token(),
                original_size_bytes=orig_size,
                duration_sec=duration,
                compressed_size_bytes=len(nc_payload),
            )

            # Determine relative subdirectory if nested
            rel_subdir = (
                audio_file.relative_to(input_path).parent
                if input_path.is_dir() and audio_file.is_relative_to(input_path)
                else Path(".")
            )

            # Save decodable compressed .nc container
            compressed_file_str = "None"
            if save_nc:
                sub_comp_dir = compressed_dir / rel_subdir
                sub_comp_dir.mkdir(parents=True, exist_ok=True)
                comp_file_path = sub_comp_dir / f"{audio_file.stem}_{comp.name}.nc"
                with open(comp_file_path, "wb") as f_nc:
                    f_nc.write(nc_payload)
                compressed_file_str = comp_file_path.name

            # Decode / Reconstruct to audio for subjective listening test
            reconstruct_file_str = "None"
            if reconstruct:
                target_samples = (
                    round(waveform.shape[-1] * comp.target_sample_rate() / sr)
                    if waveform.shape[-1] > 0 and sr > 0 else None
                )
                rec_wav = comp.decode(
                    codes,
                    chunk_frames=config.general.chunk_frames,
                    samples=target_samples,
                    overlap_frames=config.general.overlap_frames,
                )  # [channels, samples]
                import soundfile as sf
                sub_rec_dir = reconstruct_dir / rel_subdir
                sub_rec_dir.mkdir(parents=True, exist_ok=True)
                rec_path = sub_rec_dir / f"{audio_file.stem}_{comp.name}.wav"
                rec_audio_data = rec_wav.squeeze(0).cpu().numpy() if rec_wav.shape[0] == 1 else rec_wav.t().cpu().numpy()
                sf.write(str(rec_path), rec_audio_data, comp.target_sample_rate())
                reconstruct_file_str = rec_path.name

            print(
                f"{audio_file.name[:20]:<22} | "
                f"{comp.name:<14} | "
                f"{orig_size / 1024:<8.1f}KB | "
                f"{metrics['packed_bytes'] / 1024:<8.1f}KB | "
                f"{metrics['ratio_pct']:>6.2f}% | "
                f"{metrics['packed_kbps']:>6.1f}kbps"
            )

            summary_records.append({
                "filename": audio_file.name,
                "duration_sec": round(duration, 2),
                "original_bytes": orig_size,
                "codec": comp.name,
                "tokens": metrics["tokens"],
                "theoretical_bytes": metrics["theoretical_bytes"],
                "theoretical_kbps": metrics["theoretical_kbps"],
                "nc_packed_bytes": metrics["packed_bytes"],
                "nc_packed_kbps": metrics["packed_kbps"],
                "compression_ratio_pct": metrics["ratio_pct"],
                "size_reduction_pct": metrics["savings_pct"],
                "encode_time_sec": round(encode_time, 2),
                "compressed_nc_file": compressed_file_str,
                "reconstructed_wav": reconstruct_file_str,
            })

    print("=" * 90)

    # Multi-file Folder Solid Archive Summary
    if len(audio_files) > 1 or text_files:
        # Load text files if present (BMS/bmson charts etc.)
        text_files_data = []
        total_text_size = 0
        for t_file in text_files:
            t_data = t_file.read_bytes()
            t_size = len(t_data)
            total_text_size += t_size
            rel_path = t_file.relative_to(input_path) if input_path.is_dir() else Path(t_file.name)
            text_files_data.append({
                "filename": rel_path.as_posix(),
                "data": t_data,
            })

        total_audio_size = sum(f.stat().st_size for f in audio_files)
        total_orig = total_audio_size + total_text_size

        archive_label = input_path.name if input_path.is_dir() else input_path.stem
        info_str = f"{len(audio_files)} audio file(s)"
        if text_files:
            info_str += f", {len(text_files)} text file(s)"
        print("\n" + "=" * 90)
        print(f"Folder Solid Archive Summary: '{archive_label}' ({info_str}, Total Orig: {total_orig / 1024:.1f} KB)")
        print(f"{'Codec':<14} | {'Orig Total':<12} | {'Solid .nc':<12} | {'Solid Ratio':<12} | {'Separate .nc Sum':<16}")
        print("-" * 90)

        for comp in compressors:
            codec_id = getattr(comp, "codec_id", comp.name.lower().split("_")[0])
            codec_p = codec_params_map.get(codec_id, config.get_codec_params(codec_id))
            solid_nc_payload = create_nc_payload(
                files_data=codec_encoded_files[comp.name],
                codec_name=codec_id,
                codec_params=codec_p,
                bits_per_token=comp.bits_per_token(),
                sample_rate=comp.target_sample_rate(),
                folder_name=folder_name,
                text_files_data=text_files_data if text_files_data else None,
                compress=config.container.compress_nc,
                preset=config.container.lzma_preset,
            )
            solid_size = len(solid_nc_payload)
            solid_ratio = (solid_size / total_orig * 100) if total_orig > 0 else 0
            separate_sum = sum(
                r["nc_packed_bytes"] for r in summary_records if r["codec"] == comp.name
            )
            separate_ratio = (separate_sum / total_orig * 100) if total_orig > 0 else 0

            print(
                f"{comp.name:<14} | "
                f"{total_orig / 1024:<10.1f}KB | "
                f"{solid_size / 1024:<10.1f}KB | "
                f"{solid_ratio:>10.2f}% | "
                f"{separate_sum / 1024:<10.1f}KB ({separate_ratio:.2f}%)"
            )

            if save_nc:
                solid_archive_path = compressed_dir / f"{archive_label}_{comp.name}.nc"
                with open(solid_archive_path, "wb") as f_nc:
                    f_nc.write(solid_nc_payload)

        print("=" * 90)

    # Save CSV
    if summary_records:
        csv_path = Path(output_csv)
        fieldnames = list(summary_records[0].keys())
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_records)
        print(f"\nBenchmark results saved to: {csv_path.resolve()}")
        if save_nc:
            print(f"Compressed .nc files saved to: {compressed_dir.resolve()}/")
        if reconstruct:
            print(f"Reconstructed WAVs saved to: {reconstruct_dir.resolve()}/")


if __name__ == "__main__":
    print("Direct execution of benchmark.py is no longer supported.")
    print("Please use: python3 neural_compress.py bench [options]")
    sys.exit(1)
