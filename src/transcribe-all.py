"""
This code records audio from your microphone and automatically transcribes it to 
text using MLX Whisper (optimized for Apple Silicon), saving both the audio file 
and timestamped transcription.

Installation:
$ brew install portaudio
$ uv add mlx-whisper pyaudio

Usage:
$ uv run transcribe-all.py
Press CTRL+C to stop recording.
"""
import mlx_whisper
import pyaudio
import wave
from datetime import datetime
import os

def record_and_transcribe(model_size="base"):
    """
    Records audio until the user presses Ctrl+C
    """
    # Audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    print(f"Loading MLX Whisper model '{model_size}'...")
    # MLX Whisper charge automatiquement le modèle optimisé
    
    # Create folder
    if not os.path.exists("transcriptions"):
        os.makedirs("transcriptions")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = f"transcriptions/audio_{timestamp}.wav"
    output_file = f"transcriptions/transcription_{timestamp}.txt"
    
    # Recording
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print("\n🎤 Recording in progress...")
    print("💡 Press Ctrl+C to stop\n")
    
    frames = []
    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        print("\n✅ Recording completed")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save audio
    print("💾 Saving audio...")
    wf = wave.open(audio_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    # Transcribe with MLX Whisper
    print("📝 Transcription in progress (using Apple Silicon acceleration)...")
    result = mlx_whisper.transcribe(
        audio_file,
        path_or_hf_repo=f"mlx-community/whisper-{model_size}-mlx",
        language="fr"
    )
    
    # Save transcription
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f"TRANSCRIPTION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(result["text"] + "\n\n")
        
        f.write("-" * 50 + "\n")
        f.write("DETAILS WITH TIMESTAMPS:\n")
        f.write("-" * 50 + "\n\n")
        
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            f.write(f"[{start:.2f}s - {end:.2f}s] {text}\n")
    
    print(f"\n✅ Transcription saved: {output_file}")
    print(f"\n📄 Transcribed text:\n{'-'*50}\n{result['text']}\n{'-'*50}")

if __name__ == "__main__":
    print("=" * 60)
    print("   AUDIO TRANSCRIBER (MLX - Apple Silicon Optimized)")
    print("=" * 60)
    
    model = input("\nModel size (tiny/base/small/medium/large) [base]: ").strip() or "base"
    
    try:
        record_and_transcribe(model)
        print("\n✨ Completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")