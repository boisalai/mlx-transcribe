"""
Audio Transcription with Segmented Recording

This script records audio in segments and transcribes them immediately using MLX Whisper,
optimized for Apple Silicon. It's designed for long recording sessions where you want
real-time transcription feedback without losing data if something goes wrong.

Key features:
- Progressive transcription (see results as they come in)
- Memory efficient (processes one segment at a time)
- Crash-resistant (only lose the current segment if interrupted)
- All segments combined in a final master file
- Optimized for Apple Silicon with MLX

Installation:
$ brew install portaudio
$ uv add mlx-whisper pyaudio
$ uv run src/whisper_project/transcript-by-segment.py

Usage:
Press CTRL+C to stop recording.
"""

import mlx_whisper
import pyaudio
import wave
from datetime import datetime
import os
import threading
import time

def record_and_transcribe_continuous(model_size="base", segment_minutes=5):
    """
    Records and transcribes continuously in segments using MLX Whisper
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    SEGMENT_SECONDS = segment_minutes * 60
    
    print(f"Using MLX Whisper '{model_size}' (Apple Silicon optimized)...")
    model_path = f"mlx-community/whisper-{model_size}-mlx"
    
    if not os.path.exists("transcriptions"):
        os.makedirs("transcriptions")
    
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"transcriptions/session_{session_timestamp}"
    os.makedirs(session_dir)
    
    master_file = f"{session_dir}/transcription_complete.txt"
    
    # Initialize master file
    with open(master_file, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f"TRANSCRIPTION SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_size} (MLX - Apple Silicon)\n")
        f.write("=" * 50 + "\n\n")
    
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print("\n🎤 Recording in progress ({} minute segments)...".format(segment_minutes))
    print("💡 Press Ctrl+C to stop\n")
    
    segment_num = 1
    is_recording = True
    
    def transcribe_segment(audio_file, segment_num, model_path):
        """Transcribes a segment in a separate thread"""
        print(f"\n📝 Transcribing segment {segment_num} (MLX)...")
        
        result = mlx_whisper.transcribe(
            audio_file,
            path_or_hf_repo=model_path,
            language="fr"
        )
        
        # Append to master file
        with open(master_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- SEGMENT {segment_num} ---\n")
            f.write(result["text"] + "\n")
        
        print(f"✅ Segment {segment_num} transcribed")
        print(f"   Text: {result['text'][:100]}...")
    
    try:
        while is_recording:
            frames = []
            segment_start = time.time()
            
            print(f"\n🔴 Recording segment {segment_num}...")
            
            # Record for segment duration
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
            
            # Transcribe in a separate thread to avoid blocking recording
            transcribe_thread = threading.Thread(
                target=transcribe_segment, 
                args=(audio_file, segment_num, model_path)
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
    print(f"📄 Complete transcription: {master_file}")

if __name__ == "__main__":
    print("=" * 60)
    print("   CONTINUOUS AUDIO TRANSCRIBER (MLX)")
    print("=" * 60)
    
    model = input("\nModel size (tiny/base/small/medium/large) [base]: ").strip() or "base"
    segment = input("Segment duration in minutes [5]: ").strip()
    segment = int(segment) if segment else 5
    
    try:
        record_and_transcribe_continuous(model, segment)
    except Exception as e:
        print(f"\n❌ Error: {e}")