"""Evaluation metrics for measuring text overlap."""
from typing import List, Dict
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score_func
import numpy as np
import re
import string


def _normalize_text(text: str) -> str:
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # normalize whitespace
    text = " ".join(text.split())
    return text


def compute_token_f1(prediction: str, reference: str) -> float:
    """Compute SQuAD-style token-level F1 (bag-of-words with multiplicity)."""
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Count overlap with multiplicity
    from collections import Counter

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    common = pred_counts & ref_counts

    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def evaluate(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        predictions: List of generated texts
        references: List of reference texts (retrieved documents)

    Returns:
        Dict with 'rouge_l', 'bleu', 'f1', 'bertscore' keys
    """
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge_scores.append(scores['rougeL'].fmeasure)
    rouge_l = np.mean(rouge_scores)

    # BLEU - wrap each reference in a list to match predictions element-wise
    bleu = BLEU()
    bleu_score = bleu.corpus_score(predictions, [[ref] for ref in references]).score / 100.0  # Normalize to [0, 1]

    # Token F1
    f1_scores = [compute_token_f1(pred, ref) for pred, ref in zip(predictions, references)]
    f1 = np.mean(f1_scores)

    # BERTScore
    P, R, F1 = bert_score_func(
        predictions,
        references,
        lang='en',
        model_type='bert-base-uncased',
        verbose=False
    )
    bertscore = F1.mean().item()

    return {
        'rouge_l': rouge_l,
        'bleu': bleu_score,
        'f1': f1,
        'bertscore': bertscore
    }


if __name__ == '__main__':
    # Test metrics
    predictions = [
        "The quick brown fox jumps over the lazy dog",
        "Hello world this is a test"
    ]
    references = [
        "The quick brown fox jumps over the lazy dog",  # Perfect match
        "Goodbye world this was a test"  # Partial match
    ]

    metrics = evaluate(predictions, references)
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
