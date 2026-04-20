"""LLM Engine - loads model and generates text

Uses HuggingFace transformers library for inference.
"""
from transformers import pipeline
import torch


class LLMEngine:
    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-beta"):
        # Store model identifier
        self.model_name = model_name
        
        # Select device - GPU is faster but CPU works too
        if torch.cuda.is_available():
            self.device = "cuda"
            print("Using GPU for faster inference")
        else:
            self.device = "cpu"
            print("Using CPU for inference")

        print(f"Loading model: {model_name}")

        # Load with HuggingFace pipeline
        # Pipeline handles tokenization + model inference automatically
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device=0 if self.device == "cuda" else -1,
        )

    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate text from prompt using the LLM
        
        Parameters:
        - prompt: Input text to continue
        - max_length: Max tokens to generate (constraints output length)
        - temperature: Controls randomness
          * Lower (e.g., 0.3) = more deterministic/focused
          * Higher (e.g., 1.0) = more creative/diverse
        """
        try:
            outputs = self.pipe(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=0.95,  # Nucleus sampling for coherence
                do_sample=True,  # Use sampling instead of greedy
            )
            return outputs[0]["generated_text"]
        except Exception as e:
            print(f"Error generating: {e}")
            return f"Error: {str(e)}"

    def get_model_info(self) -> dict:
        """Return model info"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "family": self._get_model_family(),
        }

    def _get_model_family(self) -> str:
        """Figure out what model family this is"""
        name = os.path.basename(self.model_name).lower()
        if "llama" in name:
            return "Llama"
        elif "mistral" in name:
            return "Mistral"
        elif "tinyllama" in name:
            return "TinyLlama"
        elif "zephyr" in name:
            return "Zephyr"
        else:
            return "Unknown"
