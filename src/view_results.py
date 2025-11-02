"""View and compare experiment results.

Usage:
    python src/view_results.py --results-dir <path_to_results_directory>

Defaults to 'results/final' if --results-dir is not specified.
"""

import json
import os
import glob
import argparse
from collections import defaultdict


def view_results(results_dir='results'):
    """Load and display all experiment results."""
    if not os.path.exists(results_dir):
        print(f"No results directory found at {results_dir}")
        return

    # Find all result files
    result_files = glob.glob(f"{results_dir}/*.json")

    if not result_files:
        print(f"No result files found in {results_dir}")
        return

    print(f"Found {len(result_files)} result files\n")

    # Group by model
    model_results = defaultdict(list)

    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            model_name = data['model_name']
            model_results[model_name].append(data)

    # Display results table
    print(f"{'='*80}")
    print("EXPERIMENT RESULTS")
    print(f"{'='*80}\n")
    print(f"{'Model':<20} {'Samples':<10} {'ROUGE-L':<12} {'BLEU':<12} {'F1':<12} {'BERTScore':<12} {'Leak@8':<10} {'Leak@13':<10} {'SP':<6}")
    print("-" * 80)

    for model_name in sorted(model_results.keys()):
        results_list = model_results[model_name]
        # Show most recent result
        latest = max(results_list, key=lambda x: x['timestamp'])
        metrics = latest['metrics']

        leak8 = metrics.get('leakage_rate_8', 0.0)
        leak13 = metrics.get('leakage_rate_13', 0.0)
        sp = 'anti' if latest.get('system_prompt') else 'none'
        print(f"{model_name:<20} "
              f"{latest['num_samples']:<10} "
              f"{metrics['rouge_l']:<12.4f} "
              f"{metrics['bleu']:<12.4f} "
              f"{metrics['f1']:<12.4f} "
              f"{metrics['bertscore']:<12.4f} "
              f"{leak8:<10.4f} "
              f"{leak13:<10.4f} "
              f"{sp:<6}")

    print()

    # Show sample outputs
    print(f"\n{'='*80}")
    print("SAMPLE OUTPUTS (from most recent run)")
    print(f"{'='*80}\n")

    for model_name in sorted(model_results.keys()):
        latest = max(model_results[model_name], key=lambda x: x['timestamp'])

        if 'predictions' in latest and latest['predictions']:
            print(f"\n{model_name}:")
            print(f"  Reference (first 150 chars): {latest['references'][0][:150]}...")
            print(f"  Prediction (first 150 chars): {latest['predictions'][0][:150]}...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="View aggregated experiment results")
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="results/final", 
        help="Directory containing model result JSON files (default: results/final)"
    )
    args = parser.parse_args()
    view_results(args.results_dir)
