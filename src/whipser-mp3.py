"""
Transcribes an audio file in French using the MLX Whisper model and saves the 
result to a text file.
See https://github.com/ml-explore/mlx-examples/tree/main/whisper
See also https://simonwillison.net/2024/Aug/13/mlx-whisper/

Installation:
$ brew install ffmpeg
$ uv add mlx-whisper
"""
import time
from pathlib import Path
import mlx_whisper

start_time = time.time()

model = "mlx-community/whisper-large-v3-mlx"
downloads_folder = Path.home() / "Downloads"
audio_file_path = str(downloads_folder / "Recording.mp3")
txt_file_path = str(downloads_folder / "transcription.txt")

print(audio_file_path)

try:
    result = mlx_whisper.transcribe(
        audio_file_path,
        path_or_hf_repo=model,
        language="fr", 
        word_timestamps=True
    )

    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(result['text'])

    print(f"Transcription terminée. Voir le fichier '{txt_file_path}'.")
except FileNotFoundError:
    print(f"Le fichier {audio_file_path} n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur s'est produite lors de la transcription : {e}")

end_time = time.time()
print(f"Execution time: {(end_time - start_time)/60:.1f} minutes")