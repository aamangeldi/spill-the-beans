"""Load dataset."""
import os


def load_dataset(data_dir: str = 'data', only_file: str | None = 'wiki_newest.txt') -> str:
    """Load all .txt files and concatenate into one continuous text stream.

    Args:
        data_dir: Directory containing .txt files

    Returns:
        Concatenated text from all .txt files
    """
    all_texts = []

    # Read the specified file only (to match the paper), fallback to all .txt if missing
    if only_file is not None:
        filepath = os.path.join(data_dir, only_file) if os.path.isdir(data_dir) else data_dir
        if os.path.isdir(data_dir) and os.path.exists(filepath):
            print(f"Reading {filepath}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                all_texts.append(text)
        else:
            # Fallback: if the specified file isn't found, read all .txt files in the directory
            print(f"Warning: {filepath} not found. Falling back to reading all .txt files in {data_dir}")
            for filename in sorted(os.listdir(data_dir)):
                if not filename.endswith('.txt'):
                    continue
                filepath_i = os.path.join(data_dir, filename)
                print(f"Reading {filepath_i}...")
                with open(filepath_i, 'r', encoding='utf-8') as f:
                    text = f.read()
                    all_texts.append(text)
    else:
        # Explicitly requested to read all .txt files
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
