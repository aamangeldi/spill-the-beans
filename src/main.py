"""Main experiment runner for RAG privacy attack."""
import argparse
import json
import os
from datetime import datetime
import random
import gc
import torch
import re
from transformers import AutoTokenizer

# Silence HuggingFace download progress bars
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

from data_loader import load_dataset
from retrieval import BM25Retriever
from model import LLMInference, MODELS
from evaluator import evaluate
from wikiqa_loader import load_wikiqa_questions


def run_experiment(
    model_name: str,
    num_samples: int = 100,
    k_retrieval: int = 1,
    device: str = 'auto',
):
    """Run privacy attack experiment on a model.

    Args:
        model_name: Model to test (e.g., 'llama2-7b')
        num_samples: Number of queries to test
        k_retrieval: Number of documents to retrieve per query
        device: Device to run on
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {model_name}")
    print(f"{'='*60}\n")

    # Load dataset
    print("Loading dataset...")
    text = load_dataset('data', only_file='wiki_newest.txt')

    # Build or load retrieval index with tokenizer-based chunking (256 subword tokens, 128 stride)
    chunk_tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        trust_remote_code=True
    )
    retriever = BM25Retriever(tokenizer=chunk_tokenizer, max_chunk_length=256, stride=128)
    index_path = 'data/retrieval_index_llama2_256_128.pkl'
    if os.path.exists(index_path):
        print("Loading chunked BM25 index...")
        retriever.load(index_path)
    else:
        print("Building chunked BM25 index...")
        retriever.build_index(text)
        retriever.save(index_path)

    # Load WikiQA questions for attack (following paper's methodology)
    print("Loading WikiQA questions...")
    wikiqa_questions = load_wikiqa_questions(min_length=50, max_questions=230)
    print(f"Loaded {len(wikiqa_questions)} long questions from WikiQA")

    # Sample questions for this experiment
    random.seed(42)
    sampled_questions = random.sample(wikiqa_questions, min(num_samples, len(wikiqa_questions)))

    # Load model
    model = LLMInference(model_name, device=device)

    # Run experiments
    predictions = []
    references = []

    for i, question in enumerate(sampled_questions):
        print(f"\nProcessing {i+1}/{len(sampled_questions)}...")

        # Use WikiQA question for retrieval (anchor query)
        # This simulates an adversary with NO prior knowledge of the datastore
        query = question

        # Retrieve chunks from Wikipedia datastore
        # Paper uses num_document=1, which retrieves k=1 chunk of 256 tokens
        retrieved_chunks = retriever.retrieve(query, k=k_retrieval)

        # Select an in-document anchor sentence from the top retrieved chunk (anchor-based attack)
        def select_anchor_sentence(doc: str) -> str:
            # Simple sentence split; prefer a mid-length sentence
            sentences = re.split(r'(?<=[.!?])\s+', doc.strip())
            # filter out very short or extremely long sentences
            candidates = [s.strip() for s in sentences if 40 <= len(s.strip()) <= 200]
            if not candidates and sentences:
                # fallback to the longest sentence
                candidates = sorted([s.strip() for s in sentences], key=len, reverse=True)
            return candidates[0] if candidates else doc[:150]

        anchor_query = select_anchor_sentence(retrieved_chunks[0]) if retrieved_chunks else question

        # Construct RAG prompt with anchor-based attack
        prompt = model.construct_rag_prompt(retrieved_chunks, anchor_query)

        # Generate output
        try:
            output = model.generate(prompt)
            predictions.append(output)

            # Reference is the retrieved chunks
            reference = '\n\n'.join(retrieved_chunks)
            references.append(reference)

            print(f"Question: {query[:80]}...")
            print(f"Retrieved {len(retrieved_chunks)} chunk(s), ~{sum(len(c.split()) for c in retrieved_chunks)} tokens total")
            print(f"Generated ({len(output)} chars): {output[:100]}...")
        except Exception as e:
            print(f"Error generating for query '{query[:80]}...': {e}")
            continue

    # Evaluate
    print(f"\n{'='*60}")
    print(f"Evaluating {len(predictions)} predictions...")
    print(f"{'='*60}\n")

    metrics = evaluate(predictions, references)

    # Print results
    print(f"\nResults for {model_name}:")
    print(f"  ROUGE-L:   {metrics['rouge_l']:.4f}")
    print(f"  BLEU:      {metrics['bleu']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  BERTScore: {metrics['bertscore']:.4f}")

    # Save results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"{results_dir}/{model_name}_{timestamp}.json"

    results_data = {
        'model_name': model_name,
        'model_path': MODELS[model_name],
        'num_samples': len(predictions),
        'k_retrieval': k_retrieval,
        'metrics': metrics,
        'timestamp': timestamp,
        'predictions': predictions[:5],  # Save first 5 for inspection
        'references': references[:5]
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Clean up model from memory before loading next model
    print(f"Cleaning up {model_name} from memory and disk cache...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clear HuggingFace cache to free disk space (important for Colab)
    # This deletes the downloaded model files from ~/.cache/huggingface/
    try:
        import shutil
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if cache_dir.exists():
            # Find model cache directories
            model_cache_dirs = list(cache_dir.glob(f"models--*{MODELS[model_name].replace('/', '--')}*"))
            for cache_path in model_cache_dirs:
                shutil.rmtree(cache_path, ignore_errors=True)
                print(f"  Cleared cache: {cache_path.name}")
    except Exception as e:
        print(f"  Warning: Could not clear cache ({e})")

    print("Memory and disk cleanup complete\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run RAG privacy attack experiments')
    parser.add_argument(
        '--models',
        nargs='+',
        default=['llama2-7b'],
        choices=list(MODELS.keys()),
        help='Models to test'
    )
    parser.add_argument('--num-samples', type=int, default=100, help='Number of test samples')
    parser.add_argument('--k-retrieval', type=int, default=1, help='Number of chunks to retrieve (paper uses 1)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to run on (auto=best available)')

    args = parser.parse_args()

    print("RAG Privacy Attack Experiment")
    print(f"Models: {args.models}")
    print(f"Samples: {args.num_samples}")
    print(f"Device: {args.device}")

    # Run experiments for each model
    all_results = {}
    for model_name in args.models:
        try:
            metrics = run_experiment(
                model_name,
                num_samples=args.num_samples,
                k_retrieval=args.k_retrieval,
                device=args.device,
            )
            all_results[model_name] = metrics
        except Exception as e:
            print(f"\nError running experiment for {model_name}: {e}")
            continue

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}\n")
    print(f"{'Model':<20} {'ROUGE-L':<10} {'BLEU':<10} {'F1':<10} {'BERTScore':<10}")
    print("-" * 60)
    for model_name, metrics in all_results.items():
        print(f"{model_name:<20} "
              f"{metrics['rouge_l']:<10.4f} "
              f"{metrics['bleu']:<10.4f} "
              f"{metrics['f1']:<10.4f} "
              f"{metrics['bertscore']:<10.4f}")


if __name__ == '__main__':
    main()
