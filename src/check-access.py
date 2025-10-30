from dotenv import load_dotenv
from huggingface_hub import HfApi
import os

load_dotenv()
token = os.getenv("HF_TOKEN")
print(f"Token found: {'✅ Yes' if token else '❌ No'}")

api = HfApi()

models_to_check = [
    "pyannote/speaker-diarization-3.0",
    "pyannote/segmentation-3.0",
    "pyannote/embedding"
]

for model in models_to_check:
    try:
        info = api.model_info(model, token=token)
        print(f"✅ {model} - Access granted")
    except Exception as e:
        print(f"❌ {model} - Access denied or not accepted")