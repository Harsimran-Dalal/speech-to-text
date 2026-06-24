# Models Used in builderr Local Dictation Engine

This document details the local models used by the transcription engine, their licenses, and file sizes to satisfy local-only and offline licensing requirements.

## 1. model list

| Model ID | Source Hugging Face Repo | License | Size (Quantized) | Description / Role |
|---|---|---|---|---|
| `faster-whisper-base-int8` | [Systran/faster-whisper-base](https://huggingface.co/Systran/faster-whisper-base) | MIT | ~140 MB | Fast-pass transcription, language detection, and low-latency streaming drafts. |
| `faster-whisper-small-int8` | [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) | MIT | ~460 MB | High-quality final transcription for English, Indian-English, and Hinglish escalation. |

## 2. Quantitative Details

- **Total Weights Size on Disk**: ~600 MB (well below the 5 GB maximum limit).
- **Quantization**: `int8` quantization applied on CPU for fast execution and low memory footprint.
- **CPU Resource Bounds**: Quantized execution is limited to 4 threads to prevent CPU thrashing/contention.
- **Internet Usage**: Completely local, with network calls blocked during both batch and streaming runs (models pre-warmed offline).
