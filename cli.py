"""
CLI argument parsing and dispatching module for NeuralCompress.
Handles command-line arguments for encode, decode, and bench subcommands.
"""

import argparse
import sys
from typing import Callable, Dict, Optional, Sequence

from config import AppConfig, load_config
from neural_codecs import get_available_codecs


def build_parser() -> argparse.ArgumentParser:
    """Build and return top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="neural_compress",
        description="NeuralCompress: Neural Audio Compression Suite",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (legacy/optional, constants.py is used directly as defaults)",
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Encode subcommand
    p_enc = subparsers.add_parser("encode", help="Encode audio and text files into .nc compressed container or archive")
    p_enc.add_argument("input", type=str, help="Input audio file, or folder containing audio (WAV/FLAC/OGG/MP3) and text files (BMS/BME/BML/PMS/BMSON)")
    p_enc.add_argument("-o", "--output", type=str, default=None, help="Output destination .nc file or folder")
    p_enc.add_argument("--codec", type=str, help=f"Codec name (default: from constants.py, choices: {', '.join(get_available_codecs())})")
    p_enc.add_argument("--archive", action="store_true", help="Pack multiple files into a single solid .nc archive (default for folders)")
    p_enc.add_argument("--separate", action="store_true", help="Save each file as an individual .nc file instead of a single archive")
    p_enc.add_argument("--mono", action="store_true", default=None, help="Force mixdown to mono")
    p_enc.add_argument("--raw", "--no-compress", dest="raw", action="store_true", help="Output uncompressed raw .nc container without LZMA compression")
    p_enc.add_argument("--lzma-preset", type=int, choices=list(range(0, 10)), default=None, help="LZMA compression preset (0-9, default: constants.lzma_preset)")
    p_enc.add_argument("--encodec_bw", type=float, choices=[1.5, 3.0, 6.0, 12.0, 24.0], help="EnCodec bandwidth (kbps)")
    p_enc.add_argument("--dac_nq", type=int, choices=list(range(1, 10)), help="DAC quantizers (1-9)")
    p_enc.add_argument("--snac_model", type=str, choices=["24khz", "32khz", "44khz"], help="SNAC model type")

    # Decode subcommand
    p_dec = subparsers.add_parser("decode", help="Decode .nc compressed container/archive back to audio (WAV) and text files")
    p_dec.add_argument("input", type=str, help="Input .nc file or folder")
    p_dec.add_argument("-o", "--output", type=str, default=None, help="Output destination folder or WAV file")

    # Bench subcommand
    p_bench = subparsers.add_parser("bench", help="Run compression benchmark across multiple codecs")
    p_bench.add_argument(
        "input_dir",
        type=str,
        nargs="?",
        default=None,
        help="Input audio file, or folder containing audio files (default: from constants.py)",
    )
    p_bench.add_argument(
        "--codecs",
        "--codec",
        dest="codecs",
        type=str,
        default=None,
        help=f"Codecs to evaluate (comma-separated, e.g. 'snac,dac', choices: {', '.join(get_available_codecs())})",
    )
    p_bench.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output summary CSV path (overrides constants.py)",
    )
    p_bench.add_argument(
        "--mono",
        dest="mono",
        action="store_true",
        default=None,
        help="Force mixdown audio to mono before encoding",
    )
    p_bench.add_argument(
        "--raw",
        "--no-compress",
        dest="raw",
        action="store_true",
        help="Output uncompressed raw .nc container without LZMA compression",
    )
    p_bench.add_argument(
        "--lzma-preset",
        type=int,
        choices=list(range(0, 10)),
        default=None,
        help="LZMA compression preset (0-9, default: constants.lzma_preset)",
    )
    p_bench.add_argument(
        "--encodec_bw",
        type=float,
        default=None,
        choices=[1.5, 3.0, 6.0, 12.0, 24.0],
        help="EnCodec target bandwidth in kbps (choices: 1.5, 3.0, 6.0, 12.0, 24.0)",
    )
    p_bench.add_argument(
        "--dac_nq",
        type=int,
        default=None,
        choices=list(range(1, 10)),
        help="DAC number of codebooks/quantizers (1 to 9)",
    )
    p_bench.add_argument(
        "--snac_model",
        type=str,
        default=None,
        choices=["24khz", "32khz", "44khz"],
        help="SNAC model resolution (choices: 24khz, 32khz, 44khz)",
    )

    return parser


def run_cli(
    handlers: Dict[str, Callable[[argparse.Namespace, AppConfig], None]],
    argv: Optional[Sequence[str]] = None,
) -> None:
    """Parse CLI arguments, load config, and dispatch to appropriate subcommand handler."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    config = load_config(args)
    handler = handlers.get(args.subcommand)
    if handler:
        handler(args, config)
    else:
        parser.print_help()
        sys.exit(1)
