# NeuralCompress

this is an experimental project. not suitable for product use.
これは実験的プロジェクトです。そのままでの製品等での使用はお控えください。

**NeuralCompress** is a compression suite and benchmark toolkit powered by neural audio codecs and entropy coding (LZMA / XZ preset 9).

It provides both an extreme high-ratio audio archiver (.nc container) and a benchmark runner to compare compression ratios, bitrates, and subjective reconstruction fidelity across models.

---

## Key Features

- 3 Neural Audio Codecs
  - SNAC (Hubert Siuzdak, 2024): Multi-scale hierarchical RVQ (24kHz / 32kHz / 44.1kHz). Minimum rate: under 1kbps.
  - DAC (Descript Audio Codec, 2023): 44.1kHz full-band.
  - EnCodec (Meta, 2022): Versatile multi-bandwidth benchmark baseline (1.5 ~ 24 kbps).
- `.nc` Solid Container:
  - Compress individual audio tracks or pack entire folders/albums into a single solid `.nc` archive.
  - Solid compression shares dictionary history across tracks, yielding even higher compression ratios.
  - Self-contained metadata: Decompresses automatically without needing to specify the original codec or sample rate.
- Memory-Bounded Streaming (OOM Safe):
  - 10-second (you can set duration) chunked streaming encoding/decoding keeps RAM/VRAM footprint bounded to ~2–3 GB even for hour-long files.
- Stereo Support:
  - Encodes multi-channel audio with independent left/right streams to preserve spatial imaging and true bitrate accuracy.
- License
  - All models are licensed under MIT license.

---

## Supported Codecs & Licenses

| Codec | Creator / Year | Sampling Rate | Typical Bitrate | License |
| :--- | :--- | :--- | :--- | :--- |
| **SNAC** | Hubert Siuzdak (2024) | 24 / 32 / 44.1 kHz | 0.98 ~ 2.6 kbps | MIT |
| **DAC** | Descript (2023) | 44.1 kHz | ~8.0 kbps | MIT |
| **EnCodec** | Meta (2022) | 24 kHz | 1.5 ~ 24 kbps | MIT |

---

## Installation

### 1. Requirements
- Python 3.10+ (tested on Python 3.11, 3.12, and 3.15)
- PyTorch & Torchaudio (GPU/CUDA recommended for fastest encoding)

```bash
# Core dependencies
pip install torch torchaudio soundfile transformers huggingface_hub

# Codec packages
pip install descript-audio-codec snac
```

> **Note on NumPy 2.x:**
> NeuralCompress includes an automatic compatibility patch for NumPy 2.x and legacy tensorboard dependencies inside DAC, so NumPy downgrading is not required.

---

## Quick Start & CLI Usage

neural_compress.py provides whole tools.

### 1. Encode Audio to `.nc` Compressed File

#### Compress a single file:
```bash
# Compress using SNAC (default 32kHz, ~1.9 kbps)
python3 neural_compress.py encode input.wav -o music.nc --codec snac
```

#### Archive an entire album/song folder (audio + charts) into a single `.nc` file:
```bash
# Solid archive: Packs audio (WAV/FLAC/OGG/MP3) via neural codec + charts (BMS/BME/BML/PMS/BMSON) as raw text into one LZMA/XZ compressed archive
python3 neural_compress.py encode ./bms_folder/ -o ./compressed_files/ --codec snac
```

#### Batch compress a folder into separate individual `.nc` files:
```bash
python3 neural_compress.py encode ./wav_folder/ -o ./compressed/ --separate --codec dac
```

---

### 2. Decode `.nc` Back to OGG Audio

Decompressing requires no codec flags. All configuration is stored inside the `.nc` header:

#### Decompress a single file:
```bash
python3 neural_compress.py decode music.nc -o restored.ogg
```

#### Decompress an entire multi-track archive:
```bash
# Restores all files into the output folder with original filenames and sample rates
python3 neural_compress.py decode my_album.nc -o ./restored_album/
```

---

### 3. Run Compression Benchmark Across Models

Benchmark all configured codecs on a folder of `.wav` files:

```bash
# Run benchmark and generate comparison CSV + binary tokens + reconstructed audio
python3 neural_compress.py bench test_audio/

# Or use the legacy entry point directly:
python3 benchmark.py test_audio/
```

#### Output Artifacts:
- `compression_summary.csv`: Table containing file sizes, token counts, compression ratios (%), theoretical bitrates, and real .nc bitrates (kbps).
- `compressed_files/`: Generated `.nc` container files (fully decodable standalone and folder solid archives).
- `reconstructed_audio/`: Decoded WAV files for listening and blind A/B testing.

## Project Structure

```text
compress/
├── constants.py           # Central configuration
├── neural_compress.py     # Main unified CLI (encode, decode, bench)
├── benchmark.py           # Benchmark runner entry point
├── config.py              # settings loader with fallbacks
├── container.py           # .nc container (solid archive / unpacker)
├── metrics.py             # XZ/LZMA compression and bitrate metrics
├── neural_codecs/         # Codec implementations
│   ├── __init__.py        # Dynamic codec factory and registry
│   ├── base.py            # BaseCodecCompressor (streaming, stereo dual-mono)
│   ├── snac.py            # SNAC wrapper (multi-scale RVQ interleaving)
│   ├── dac.py             # DAC wrapper
│   └── encodec.py         # EnCodec wrapper
├── test_audio/            # Test WAV audio files
├── compressed_files/      # Output compressed packages
└── reconstructed_audio/   # Output reconstructed WAV audio
```

---

## License

This repository is licensed under the MIT License. Please refer to individual model source repositories for pretrained model weights licenses (see table above).
