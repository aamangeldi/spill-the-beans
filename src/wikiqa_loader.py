"""Load WikiQA dataset for attack queries."""
from datasets import load_dataset


def load_wikiqa_questions(min_length: int = 50, max_questions: int = 230) -> list[str]:
    """Load long questions from WikiQA dataset.

    Args:
        min_length: Minimum character length for "long" questions
        max_questions: Maximum number of questions to return

    Returns:
        List of question strings
    """
    # Load WikiQA from HuggingFace
    dataset = load_dataset("microsoft/wiki_qa", split="train")

    # Extract unique questions (dataset has question-answer pairs)
    unique_questions = set()
    for item in dataset:
        question = item['question'].strip()
        if question:
            unique_questions.add(question)

    # Filter for long questions
    long_questions = [q for q in unique_questions if len(q) >= min_length]

    # Sort for reproducibility
    long_questions = sorted(long_questions)

    # Return up to max_questions
    return long_questions[:max_questions]


if __name__ == '__main__':
    # Test the loader
    questions = load_wikiqa_questions()
    print(f"Loaded {len(questions)} long questions from WikiQA")
    print(f"\nFirst 5 questions:")
    for i, q in enumerate(questions[:5], 1):
        print(f"{i}. {q}")
