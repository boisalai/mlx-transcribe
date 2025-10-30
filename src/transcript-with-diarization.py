"""
Audio Transcription with Speaker Diarization (Apple Silicon Optimized)

This script records audio in segments, transcribes them using MLX Whisper (optimized
for Apple Silicon), and identifies different speakers using pyannote.audio.

Installation:
$ brew install portaudio ffmpeg
$ uv add mlx-whisper pyaudio pyannote.audio pydub torch python-dotenv soundfile onnxruntime

Setup:
1. Get a free HuggingFace token: https://huggingface.co/settings/tokens
2. Accept conditions: https://huggingface.co/pyannote/speaker-diarization-3.0
4. Accept conditions: https://huggingface.co/pyannote/segmentation-3.0
5. Accept conditions: https://huggingface.co/pyannote/embedding
6. Create a .env file with: HF_TOKEN=your_token_here

Usage:
$ uv run transcript-with-diarization.py
Press CTRL+C to stop recording.
"""

import mlx_whisper
import pyaudio
import wave
from datetime import datetime
import os
import threading
import time
from pyannote.audio import Pipeline
import json
from dotenv import load_dotenv
import soundfile as sf
import torch

# Load environment variables from .env file
load_dotenv()

def record_and_transcribe_with_diarization(model_size="base", segment_minutes=5):
    """
    Records and transcribes continuously with speaker identification
    Combines MLX Whisper (fast) with pyannote (speaker detection)
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    SEGMENT_SECONDS = segment_minutes * 60
    
    print(f"🔄 Using MLX Whisper '{model_size}' (Apple Silicon optimized)...")
    mlx_model_path = f"mlx-community/whisper-{model_size}-mlx"
    
    print("🔄 Loading diarization model...")
    
    # Get token from environment variable
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    if not HF_TOKEN:
        print("\n❌ ERROR: HuggingFace token not found!")
        print("   1. Get token: https://huggingface.co/settings/tokens")
        print("   2. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   3. Create a .env file in your project root with:")
        print("      HF_TOKEN=your_token_here\n")
        return
    
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        token=HF_TOKEN
    )

    # Use MPS (Metal Performance Shaders) for Apple Silicon
    try:
        import torch
        if torch.backends.mps.is_available():
            diarization_pipeline.to(torch.device("mps"))
            print("✅ Using Apple Silicon GPU (MPS) for diarization")
    except Exception as e:
        print(f"⚠️  Running diarization on CPU: {e}")
    
    if not os.path.exists("transcriptions"):
        os.makedirs("transcriptions")
    
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"transcriptions/session_{session_timestamp}"
    os.makedirs(session_dir)
    
    master_file = f"{session_dir}/transcription_complete.txt"
    
    # Initialize master file
    with open(master_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"TRANSCRIPTION SESSION WITH SPEAKER DIARIZATION\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_size} (MLX Whisper + Pyannote)\n")
        f.write("=" * 70 + "\n\n")
    
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print(f"\n🎤 Recording in progress ({segment_minutes} min segments)...")
    print("💡 Press Ctrl+C to stop\n")
    
    segment_num = 1
    is_recording = True
    
    def transcribe_with_speakers(audio_file, segment_num):
        """Transcribes a segment with speaker identification"""
        print(f"\n📝 Analyzing segment {segment_num}...")

        # Load audio in memory (workaround for torchcodec issues)
        waveform, sample_rate = sf.read(audio_file)
        # Convert to torch tensor and add channel dimension if needed
        waveform_tensor = torch.from_numpy(waveform).float()
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)  # Add channel dimension

        audio_dict = {
            "waveform": waveform_tensor,
            "sample_rate": sample_rate
        }

        # 1. Diarization (who speaks when)
        print(f"   👥 Identifying speakers...")
        diarization = diarization_pipeline(audio_dict)
        
        # 2. Transcription with MLX Whisper (fast!)
        print(f"   🗣️  Transcribing audio (MLX)...")
        result = mlx_whisper.transcribe(
            audio_file,
            path_or_hf_repo=mlx_model_path,
            language="fr",
            word_timestamps=True  # Important for syncing with diarization
        )
        
        # 3. Merge diarization and transcription
        print(f"   🔗 Merging data...")
        transcript_with_speakers = []
        
        for segment in result["segments"]:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            
            # Find dominant speaker for this segment
            speaker = get_speaker_for_time(diarization, start_time, end_time)
            
            transcript_with_speakers.append({
                "speaker": speaker,
                "start": start_time,
                "end": end_time,
                "text": text.strip()
            })
        
        # 4. Format and save
        formatted_text = format_transcript(transcript_with_speakers, segment_num)
        
        with open(master_file, "a", encoding="utf-8") as f:
            f.write(formatted_text)
        
        # Also save as JSON for later analysis
        json_file = f"{session_dir}/segment_{segment_num:03d}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(transcript_with_speakers, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Segment {segment_num} completed")
        print_preview(transcript_with_speakers)
    
    def get_speaker_for_time(diarization, start, end):
        """Finds the main speaker for a time interval"""
        speaker_time = {}

        # pyannote.audio 4.0+ DiarizeOutput has .speaker_diarization attribute
        for turn, speaker in diarization.speaker_diarization:
            overlap_start = max(turn.start, start)
            overlap_end = min(turn.end, end)

            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                speaker_time[speaker] = speaker_time.get(speaker, 0) + duration

        if not speaker_time:
            return "Unknown Speaker"

        return max(speaker_time, key=speaker_time.get)
    
    def format_transcript(segments, segment_num):
        """Formats transcription with speakers"""
        output = f"\n{'='*70}\n"
        output += f"SEGMENT {segment_num}\n"
        output += f"{'='*70}\n\n"
        
        current_speaker = None
        
        for seg in segments:
            speaker = seg["speaker"]
            text = seg["text"]
            timestamp = format_time(seg["start"])
            
            # New line if speaker changes
            if speaker != current_speaker:
                if current_speaker is not None:
                    output += "\n"
                output += f"[{timestamp}] {speaker}:\n"
                current_speaker = speaker
            
            output += f"  {text}\n"
        
        output += "\n"
        return output
    
    def format_time(seconds):
        """Formats time as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def print_preview(segments):
        """Displays preview of first 3 sentences"""
        preview = segments[:3] if len(segments) > 3 else segments
        for seg in preview:
            print(f"   {seg['speaker']}: {seg['text'][:60]}...")
    
    # Main recording loop
    try:
        while is_recording:
            frames = []
            segment_start = time.time()
            
            print(f"\n🔴 Recording segment {segment_num}...")
            
            while time.time() - segment_start < SEGMENT_SECONDS and is_recording:
                data = stream.read(CHUNK)
                frames.append(data)
            
            if not frames:
                break
            
            # Save segment
            audio_file = f"{session_dir}/segment_{segment_num:03d}.wav"
            wf = wave.open(audio_file, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Transcribe with diarization in separate thread
            transcribe_thread = threading.Thread(
                target=transcribe_with_speakers,
                args=(audio_file, segment_num)
            )
            transcribe_thread.start()
            
            segment_num += 1
            
    except KeyboardInterrupt:
        print("\n✅ Recording completed")
        is_recording = False
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    print(f"\n✨ Session completed!")
    print(f"📁 Directory: {session_dir}")
    print(f"📄 Transcription: {master_file}")
    print(f"💡 JSON files available for advanced analysis")

if __name__ == "__main__":
    print("=" * 70)
    print("   TRANSCRIBER WITH SPEAKER IDENTIFICATION (MLX)")
    print("=" * 70)
    
    print("\n⚠️  PREREQUISITES:")
    print("1. HuggingFace token (free): https://huggingface.co/settings/tokens")
    print("2. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("3. Create .env file with: HF_TOKEN=your_token_here")
    
    model = input("\nWhisper model (tiny/base/small/medium/large) [medium]: ").strip() or "medium"
    segment = input("Segment duration in minutes [5]: ").strip()
    segment = int(segment) if segment else 5
    
    try:
        record_and_transcribe_with_diarization(model, segment)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
