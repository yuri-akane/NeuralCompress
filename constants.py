"""
Constants and Configuration for NeuralCompress.
Direct configuration values used across the project.
"""

# OOM防止のためのエンコード分割チャンク長 (秒)
chunk_sec = 20.0

# OOM防止のためのデコード分割フレーム数
#メモリがすぐ足りなくなるので適宜減らしてください
chunk_frames = 1024

# チャンク境界のクリックノイズを防止するクロスフェード重複フレーム数 (DAC 44.1kHz時: 32フレーム ≒ 約0.37秒)
# 公式の#PR96と同じ方針で実装
overlap_frames = 32

# ==============================================================================
# 一般設定 (General Settings)
# ==============================================================================
input_dir = "test_audio"
output_csv = "compression_summary.csv"
compressed_dir = "compressed_files"
reconstructed_dir = "reconstructed_audio"

# 出力オプション@benchmark
reconstruct = True    # 聴き比べ用復元 WAV を保存
save_nc = True        # benchmark時に.ncを保存するかどうか

# 音声処理設定
force_mono = False    # True: モノラル強制変換, False: ステレオ/マルチチャンネル維持

# 評価対象コデック一覧 (登録名: encodec, dac, snac など)
enabled_codecs = ["encodec", "dac", "snac"]

# ==============================================================================
# コンテナ設定 (.nc Container Settings)
# ==============================================================================
# .nc 保存時に全体を LZMA(XZ) 圧縮するかどうか (True: xz圧縮, False: 無圧縮Raw)
compress_nc = False
# LZMA 圧縮プリセット (0-9, 9が最高圧縮)
lzma_preset = 9

# ==============================================================================
# コデック固有設定 (Codec-Specific Settings)
# ==============================================================================
codec_params = {
    "encodec": {
        "bandwidth": 24.0,
        "model_name": "facebook/encodec_24khz",
    },
    "dac": {
        "n_quantizers": 9,
        "model_type": "44khz",
    },
    "snac": {
        "model_type": "44khz",
    },
}
