"""Evaluation metrics for measuring text overlap."""
from typing import List, Dict
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score_func
import numpy as np


def compute_token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score.

    Args:
        prediction: Generated text
        reference: Reference text (retrieved documents)

    Returns:
        F1 score
    """
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    overlap = pred_tokens & ref_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
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

    # Leakage metrics: longest common contiguous word substring (LCS as substring)
    def longest_common_substring_word_len(a: str, b: str) -> int:
        a_words = a.split()
        b_words = b.split()
        if not a_words or not b_words:
            return 0
        # Use DP for longest common substring over words
        # Only keep previous row to save memory
        prev = [0] * (len(b_words) + 1)
        best = 0
        for i in range(1, len(a_words) + 1):
            curr = [0] * (len(b_words) + 1)
            aw = a_words[i - 1]
            for j in range(1, len(b_words) + 1):
                curr[j] = (prev[j - 1] + 1) if aw == b_words[j - 1] else 0
                if curr[j] > best:
                    best = curr[j]
            prev = curr
        return best

    lcs_lengths = [longest_common_substring_word_len(pred, ref) for pred, ref in zip(predictions, references)]
    pred_lengths = [max(1, len(pred.split())) for pred in predictions]
    lcs_mean = float(np.mean(lcs_lengths)) if lcs_lengths else 0.0
    lcs_max = float(np.max(lcs_lengths)) if lcs_lengths else 0.0
    lcs_mean_frac = float(np.mean([l / pl for l, pl in zip(lcs_lengths, pred_lengths)])) if lcs_lengths else 0.0
    leakage_rate_8 = float(np.mean([1.0 if l >= 8 else 0.0 for l in lcs_lengths])) if lcs_lengths else 0.0
    leakage_rate_13 = float(np.mean([1.0 if l >= 13 else 0.0 for l in lcs_lengths])) if lcs_lengths else 0.0

    return {
        'rouge_l': rouge_l,
        'bleu': bleu_score,
        'f1': f1,
        'bertscore': bertscore,
        'leakage_lcs_mean': lcs_mean,
        'leakage_lcs_max': lcs_max,
        'leakage_rate_8': leakage_rate_8,
        'leakage_rate_13': leakage_rate_13
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
