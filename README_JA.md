# NeuralCompress

this is an experimental project. not suitable for product use.
これは実験的プロジェクトです。そのままでの製品等での使用はお控えください。

**NeuralCompress**は、ニューラルオーディオコデック（離散トークン型コデック）とエントロピー可逆圧縮（LZMA / XZ preset 9）を組み合わせた圧縮・性能比較ベンチマークツールです。

超高圧縮オーディオアーカイバ（.ncコンテナ）としての機能と、各コデックの圧縮率・実効ビットレート・復元音質（WAV）を客観的・主観的に評価するベンチマークの両方を提供します。

---

## 主な特徴

- 対応コデック
  - SNAC (Hubert Siuzdak, 2024): マルチスケール階層RVQ（24kHz / 32kHz / 44.1kHz）。最低音質なら1kbps未満
  - DAC (Descript, 2023): 44.1kHz フルバンド対応、音質重視。
  - EnCodec (Meta, 2022): 第1世代ニューラルコデックの基準（比較用）。可変ビットレート（1.5〜24 kbps）。
- `.nc` ソリッドコンテナ
  - 単一の音声ファイル圧縮に加え、フォルダ内の複数曲および関連譜面テキスト（BMS/BME/BML/PMS/BMSON）を 1 本の `.nc` ファイルにまとめてアーカイブ。
  - 音声ファイル（WAV/FLAC/OGG/MP3）はニューラルコデックで高圧縮し、テキストファイルはそのまま格納。
  - ファイル間でLZMAの辞書を共有するソリッド圧縮により、1ファイルずつ個別圧縮するよりもさらに高圧縮に。
  - メタデータ自己内包型: 展開時はモデル名や周波数を指定する必要がなく、フォルダ構成や譜面テキストを含めて自動で判別・復元。
- OOM対策ストリーミング:
  - 10秒単位（設定可）のチャンク分割ストリーミングにより、長尺音源でもRAM/VRAM消費を一定に維持。
- ステレオ（マルチチャンネル）対応:
  - L/Rを独立にエンコードし、音場や広がりを損なわずに実効ビットレートを正確に反映。
- ライセンス:
  - 全モデルMIT Licenseです。

---

## サポートコデックとライセンス

| コデック名 | 開発元 / 年 | サンプリング周波数 | 標準ビットレート | ライセンス |
| :--- | :--- | :--- | :--- | :--- |
| **SNAC** | Hubert Siuzdak (2024) | 24 / 32 / 44.1 kHz | 0.98 〜 2.6 kbps | MIT |
| **DAC** | Descript (2023) | 44.1 kHz | 約 8.0 kbps | MIT |
| **EnCodec** | Meta (2022) | 24 kHz | 1.5 〜 24 kbps | MIT |

---

## インストール

### 1. 動作要件
- Python 3.10 以上 (Python 3.11, 3.12, 3.15 で動作確認済み)
- PyTorch & Torchaudio（GPU / CUDA 推奨）

```bash
# 基本ライブラリのインストール
pip install torch torchaudio soundfile transformers huggingface_hub

# 各コデックライブラリのインストール
pip install descript-audio-codec snac
```

> **NumPy 2.x 互換パッチについて:**
> NumPy 2.x 環境で DAC の依存関係がエラーを起こす問題は、スクリプト側で自動互換パッチを適用しているため、NumPy をダウングレードする必要はありません。

---

## 使い方(CLI)

neural_compress.pyからすべての機能を利用できます。

### 1. 音声を圧縮する (`encode`)

#### ① 1ファイルを単一の `.nc` ファイルに圧縮:
```bash
# SNAC（デフォルト: 32kHz, 約1.9kbps）で圧縮
python3 neural_compress.py encode input.wav -o music.nc --codec snac
```

#### ② フォルダ内の全楽曲・譜面テキストを「1本の `.nc` アーカイブ」にまとめて圧縮:
```bash
# BMS/bmson 楽曲フォルダ丸ごと 1 つのファイルにソリッド圧縮（音声はニューラル圧縮、テキストは生データ保持、全体をLZMA/XZ圧縮）
python3 neural_compress.py encode ./bms_folder/ -o ./compressed_files/ --codec snac
```

#### ③ フォルダ内の全音声を「1曲ずつ個別の `.nc`」として一括変換:
```bash
python3 neural_compress.py encode ./wav_folder/ -o ./compressed/ --separate --codec dac
```

---

### 2. 圧縮ファイルを復元する (`decode`)

展開時はコデックの指定が不要です（`.nc` ファイル内のヘッダーから自動認識されます）。

#### ① 1ファイルを展開:
```bash
python3 neural_compress.py decode music.nc -o restored.ogg
```

#### ② まとめてアーカイブされた `.nc` から全曲を一括展開:
```bash
# アーカイブ内の全ファイルを元のファイル名・サンプリング周波数で復元
python3 neural_compress.py decode my_album.nc -o ./restored_album/
```

---

### 3. コデック間の圧縮率比較ベンチマークを実行する (`bench`)

フォルダ内の WAV 音源に対して、設定されたすべてのコデックで一括処理を行い、圧縮率と復元音声を出力します。

```bash
# ベンチマーク実行
python3 neural_compress.py bench test_audio/

# または従来のスクリプトを直接実行
python3 benchmark.py test_audio/
```

#### 出力生成物:
- `compression_summary.csv`: ファイルサイズ、トークン数、圧縮率（%）、理論ビットレート、.ncコンテナ実効ビットレート（kbps）の一覧表。
- `compressed_files/`: 実際に生成された各モデルの `.nc` コンテナファイル（単体およびフォルダ全体のソリッドアーカイブ。そのままdecode可能）。
- `reconstructed_audio/`: 人間の耳で試聴・A/Bテスト比較するための復元 WAV ファイル群。

## ディレクトリ構成

```text
compress/
├── constants.py           # モデル設定、出力先、ビットレート等を一括管理
├── neural_compress.py     # 総合エントリポイント (encode, decode, bench)
├── benchmark.py           # ベンチマーク専用エントリポイント
├── config.py              # 設定の読み込みとパース
├── container.py           # .nc コンテナフォーマットの pack / unpack (ソリッドアーカイブ対応)
├── metrics.py             # XZ可逆圧縮、ビットレート、圧縮率計算
├── neural_codecs/         # コデック実装パッケージ
│   ├── __init__.py        # レジストリ・遅延ファクトリ関数
│   ├── base.py            # BaseCodecCompressor（共通ストリーミング・ステレオ処理）
│   ├── snac.py            # SNAC 実装（マルチスケールRVQパッキング）
│   ├── dac.py             # DAC 実装
│   └── encodec.py         # EnCodec 実装
├── test_audio/            # テスト音源フォルダ
├── compressed_files/      # 出力：圧縮バイナリ保存先
└── reconstructed_audio/   # 出力：復元WAV音声保存先
```

---

## ライセンス

本リポジトリのコードはMIT Licenseで提供されています。各モデルの事前学習済み重みのライセンスについては、上記「サポートコデックとライセンス一覧」をご参照ください。
