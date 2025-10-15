"""Model inference for language models."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List


# Model configurations
MODELS = {
    'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.1',
    'solar-10.7b': 'upstage/SOLAR-10.7B-Instruct-v1.0',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'vicuna-13b': 'lmsys/vicuna-13b-v1.5',
    'mixtral-8x7b': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'wizardlm-13b': 'WizardLM/WizardLM-13B-V1.2',
}


class LLMInference:
    """Simple LLM inference wrapper."""

    def __init__(self, model_name: str, device: str = 'cuda'):
        """Initialize model.

        Args:
            model_name: Short model name (e.g., 'llama2-7b')
            device: Device to run on ('cuda' or 'cpu')
        """
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}")

        self.model_name = model_name
        self.model_path = MODELS[model_name]
        self.device = device

        print(f"Loading {model_name} from {self.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' if device == 'cuda' else None,
            trust_remote_code=True
        )

        if device == 'cpu':
            self.model = self.model.to(device)

        self.model.eval()
        print(f"Model loaded successfully")

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_k=60,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated part (skip input)
        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated.strip()

    def construct_rag_prompt(self, retrieved_docs: List[str], query: str) -> str:
        """Construct RAG prompt with retrieved documents.

        Args:
            retrieved_docs: List of retrieved document strings
            query: Attack query/instruction

        Returns:
            Full prompt
        """
        # Join documents with double newline
        docs_str = '\n\n'.join(retrieved_docs)

        # Construct prompt: documents + query
        prompt = f"{docs_str}\n\n{query}"

        return prompt


if __name__ == '__main__':
    # Test with a small model (if available)
    print("Model inference module loaded")
    print(f"Available models: {list(MODELS.keys())}")
