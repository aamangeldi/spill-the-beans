"""Load Wikipedia dataset."""

def load_wiki_dataset(file_path: str) -> list[dict]:
    """Load Wikipedia articles from text file.

    Args:
        file_path: Path to wiki_newest.txt

    Returns:
        List of dicts with 'title' and 'text' keys
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to separate articles
    articles = content.split('\n\n')

    # Parse into structured format
    parsed = []
    for article in articles:
        article = article.strip()
        if not article:
            continue

        lines = article.split('\n', 1)
        if len(lines) == 2:
            title = lines[0].strip()
            text = lines[1].strip()
            parsed.append({'title': title, 'text': text})
        elif len(lines) == 1:
            # Single line articles
            parsed.append({'title': lines[0].strip(), 'text': lines[0].strip()})

    return parsed


if __name__ == '__main__':
    # Test the data loader
    articles = load_wiki_dataset('data/wiki_newest.txt')
    print(f"Loaded {len(articles)} articles")
    print(f"\nFirst article:")
    print(f"Title: {articles[0]['title']}")
    print(f"Text preview: {articles[0]['text'][:200]}...")
