# Local Models

Place pre-downloaded HuggingFace model directories here to avoid downloading at container startup.

The app checks this folder first. If a model directory is found, it is used directly.
If the folder is empty, the app downloads the configured model from HuggingFace on first run.

## How to download a model locally

```bash
pip install huggingface_hub

# Recommended lightweight model (~2 GB, good for CPU)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', local_dir='models/TinyLlama-1.1B-Chat-v1.0')
"

# Default model (heavier, ~14 GB, better quality)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('HuggingFaceH4/zephyr-7b-beta', local_dir='models/zephyr-7b-beta')
"
```

## Supported models

| Model | Size | RAM needed | Quality | Best for |
|-------|------|-----------|---------|----------|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2 GB | ~4 GB | Good | CPU / dev |
| `mistralai/Mistral-7B-Instruct-v0.2` | ~14 GB | ~16 GB | Excellent | GPU |
| `HuggingFaceH4/zephyr-7b-beta` | ~14 GB | ~16 GB | Excellent | GPU |

After placing a downloaded model here, rebuild the Docker image so it gets copied in:

```bash
docker build -t rag-app .
docker run -p 8000:8000 -p 8501:8501 rag-app
```
