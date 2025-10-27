"""Load Wikipedia dataset."""
import re


def strip_latex(text: str) -> str:
    r"""Remove LaTeX markup from Wikipedia text.

    Wikipedia articles often contain LaTeX math notation like:
    {\displaystyle W} or {\displaystyle \Delta U=Q-W}

    This function removes these to make chunks more readable.

    Args:
        text: Raw Wikipedia text with LaTeX markup

    Returns:
        Cleaned text without LaTeX
    """
    # Remove {\displaystyle ...} blocks - these are common in math articles
    # Need to handle nested braces carefully
    text = re.sub(r'\{\s*\\displaystyle[^}]*\}', '', text)

    # Remove other common LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)  # \command{arg}
    text = re.sub(r'\\[a-zA-Z]+', '', text)  # \command

    # Remove standalone LaTeX symbols that might remain
    text = re.sub(r'[δΔ]\s*[A-Z]\s*=', lambda m: m.group(0).replace('δ', '').replace('Δ', ''), text)

    # Clean up excessive whitespace caused by removal
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines -> double newline
    text = re.sub(r' +', ' ', text)  # Multiple spaces -> single space

    return text.strip()


def load_wiki_dataset(file_path: str, min_words: int = 100) -> list[dict]:
    """Load Wikipedia articles from text file.

    Following the paper: filter articles with < 100 words.
    Combines article intro with all its sections.

    Args:
        file_path: Path to wiki_newest.txt
        min_words: Minimum word count (paper uses 100)

    Returns:
        List of dicts with 'title' and 'text' keys
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines
    sections = content.split('\n\n')

    parsed = []
    current_article = None
    current_sections = []

    for section in sections:
        # Strip only newlines, preserve leading spaces
        section_stripped = section.strip('\n')
        if not section_stripped:
            continue

        # Check if this is a section (starts with space) or new article
        if section_stripped.startswith(' '):
            # This is a section within an article
            if current_article is not None:
                current_sections.append(section_stripped.strip())
        else:
            # This is a new article - save previous article if exists
            if current_article is not None:
                # Combine intro + all sections
                full_text = current_article['intro'] + '\n\n' + '\n\n'.join(current_sections)
                full_text = full_text.strip()

                # Strip LaTeX markup to make text more readable
                full_text = strip_latex(full_text)

                word_count = len(full_text.split())

                # Filter by min_words
                if word_count >= min_words:
                    parsed.append({
                        'title': current_article['title'],
                        'text': full_text
                    })

            # Start new article
            section_clean = section_stripped.strip()
            lines = section_clean.split('\n', 1)

            if len(lines) == 2:
                current_article = {
                    'title': lines[0].strip(),
                    'intro': lines[1].strip()
                }
            elif len(lines) == 1:
                # No intro text, just title - set intro to empty string to avoid duplication
                current_article = {
                    'title': lines[0].strip(),
                    'intro': ''
                }

            current_sections = []

    # Don't forget the last article
    if current_article is not None:
        full_text = current_article['intro'] + '\n\n' + '\n\n'.join(current_sections)
        full_text = full_text.strip()

        # Strip LaTeX markup to make text more readable
        full_text = strip_latex(full_text)

        word_count = len(full_text.split())

        if word_count >= min_words:
            parsed.append({
                'title': current_article['title'],
                'text': full_text
            })

    return parsed


if __name__ == '__main__':
    # Test the data loader
    articles = load_wiki_dataset('data/wiki_newest.txt')
    print(f"Loaded {len(articles)} articles")
    print(f"\nFirst article:")
    print(f"Title: {articles[0]['title']}")
    print(f"Text preview: {articles[0]['text'][:200]}...")
