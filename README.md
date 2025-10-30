# mlx-transcribe

Audio transcription with speaker diarization, optimized for Apple Silicon using MLX Whisper.

## Overview

mlx-transcribe records audio in segments, transcribes it using MLX Whisper, and identifies different speakers using pyannote.audio. All processing is optimized for Apple Silicon hardware.

## Features

- Real-time audio recording and transcription
- Speaker diarization (identifies different speakers)
- Apple Silicon GPU acceleration (MPS)
- Continuous recording with configurable segment duration
- Outputs formatted transcripts with speaker identification
- JSON export for advanced analysis

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.12 or higher
- HuggingFace account (free) for diarization models

## Installation

Install system dependencies:

```bash
brew install portaudio ffmpeg
```

Install Python dependencies:

```bash
uv add mlx-whisper pyaudio pyannote.audio pydub torch python-dotenv soundfile onnxruntime
```

## Setup

1. Get a HuggingFace token: https://huggingface.co/settings/tokens
2. Accept model conditions:
   - https://huggingface.co/pyannote/speaker-diarization-3.0
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/embedding
3. Create a `.env` file in the project root:

```
HF_TOKEN=your_token_here
```

## Usage

```bash
uv run src/transcript-with-diarization.py
```

Select the Whisper model size (tiny/base/small/medium/large) and segment duration when prompted. Press Ctrl+C to stop recording.

## Output

Transcriptions are saved in `transcriptions/session_YYYYMMDD_HHMMSS/`:

- `transcription_complete.txt` - Formatted transcript with speaker labels
- `segment_NNN.wav` - Audio segments
- `segment_NNN.json` - Segment data in JSON format

## License

See LICENSE file for details.
