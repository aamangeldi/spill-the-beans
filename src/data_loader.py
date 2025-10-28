"""Load dataset."""
import os


def load_dataset(data_dir: str = 'data') -> str:
    """Load all .txt files and concatenate into one continuous text stream.

    This matches the paper's approach in modules/Index.py:
    - Reads all .txt files from directory
    - Joins them into one big string
    - Chunks are created later based on tokens, not articles

    Args:
        data_dir: Directory containing .txt files

    Returns:
        Concatenated text from all .txt files
    """
    all_texts = []

    # Read all .txt files
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(data_dir, filename)
        print(f"Reading {filepath}...")

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            all_texts.append(text)

    # Concatenate all texts with space separator
    full_text = " ".join(all_texts)

    print(f"Loaded {len(all_texts)} file(s)")
    print(f"Total text length: {len(full_text)} chars, {len(full_text.split())} words")

    return full_text


if __name__ == '__main__':
    # Test the data loader
    text = load_dataset('data')
    print(f"\nFirst 200 chars: {text[:200]}")
    print(f"\nLast 200 chars: {text[-200:]}")
