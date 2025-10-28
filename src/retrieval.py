"""BM25 document retrieval."""
import pickle
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np


class BM25Retriever:
    """BM25 retriever with token-based chunking."""

    def __init__(self, tokenizer=None, max_chunk_length: int = 256, stride: int = 128):
        """Initialize retriever.

        Args:
            tokenizer: HuggingFace tokenizer (optional, uses whitespace if None)
            max_chunk_length: Maximum tokens per chunk (paper uses 256)
            stride: Stride for overlapping chunks (paper uses 128)
        """
        self.tokenizer = tokenizer
        self.max_chunk_length = max_chunk_length
        self.stride = stride
        self.bm25 = None
        self.chunks = None  # List of chunk strings

    def build_index(self, text: str):
        """Build BM25 index from continuous text stream.

        1. Tokenize all text
        2. Split into chunks (256 tokens, 128 stride)
        3. Build BM25 index

        Args:
            text: Concatenated text from all documents
        """
        print(f"Tokenizing text ({len(text)} chars, {len(text.split())} words)...")

        if self.tokenizer:
            # Use HuggingFace tokenizer
            # Tokenize in chunks to avoid memory issues
            all_tokens = []
            words = text.split()
            step_size = 1024

            print("Tokenizing in chunks...")
            chunks_to_tokenize = [words[i:i + step_size] for i in range(0, len(words), step_size)]
            chunks_to_tokenize = [" ".join(chunk) for chunk in chunks_to_tokenize]

            for chunk_text in chunks_to_tokenize:
                tokens = self.tokenizer(chunk_text)['input_ids']
                all_tokens.extend(tokens)

            all_tokens = np.array(all_tokens)
            print(f"Total tokens: {len(all_tokens)}")

            # Split into chunks
            print(f"Creating chunks (max_length={self.max_chunk_length}, stride={self.stride})...")
            token_chunks = self._get_token_chunks(
                all_tokens,
                pad_token=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
            )

            # Decode chunks back to text
            self.chunks = []
            for token_chunk in token_chunks:
                chunk_text = self.tokenizer.decode(token_chunk, skip_special_tokens=True)
                self.chunks.append(chunk_text)

        else:
            # Simple whitespace tokenization (faster, good enough for BM25)
            print("Using whitespace tokenization...")
            words = text.split()
            print(f"Total words: {len(words)}")

            # Split into chunks
            print(f"Creating chunks (max_length={self.max_chunk_length}, stride={self.stride})...")
            self.chunks = []
            for start in range(0, len(words), self.stride):
                end = min(start + self.max_chunk_length, len(words))
                chunk_words = words[start:end]
                chunk_text = ' '.join(chunk_words)
                self.chunks.append(chunk_text)

                if end == len(words):
                    break

        print(f"Created {len(self.chunks)} chunks")
        print(f"Average chunk size: {sum(len(c.split()) for c in self.chunks) / len(self.chunks):.1f} words")

        # Build BM25 index
        print("Building BM25 index...")
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        print("Index built successfully")

    def _get_token_chunks(self, tokens: np.ndarray, pad_token: int) -> np.ndarray:
        """Split tokens into overlapping chunks.

        Args:
            tokens: Flat array of token IDs
            pad_token: Token ID for padding

        Returns:
            Array of token chunks, each of length max_chunk_length
        """
        assert tokens.ndim == 1, "Tokens should be flattened first!"
        num_tokens = len(tokens)
        tokens_as_chunks = []

        for begin_loc in range(0, num_tokens, self.stride):
            end_loc = min(begin_loc + self.max_chunk_length, num_tokens)
            token_chunk = tokens[begin_loc:end_loc].copy()

            # Pad last chunk if needed
            if end_loc == num_tokens and len(token_chunk) < self.max_chunk_length:
                pads = np.array([pad_token] * (self.max_chunk_length - len(token_chunk)))
                token_chunk = np.concatenate([token_chunk, pads])

            assert len(token_chunk) == self.max_chunk_length
            tokens_as_chunks.append(token_chunk)

        return np.stack(tokens_as_chunks)

    def retrieve(self, query: str, k: int = 1) -> List[str]:
        """Retrieve top-k chunks for a query.

        Args:
            query: Search query
            k: Number of chunks to retrieve

        Returns:
            List of retrieved chunk texts
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]

        # Return chunks
        return [self.chunks[idx] for idx in top_k_indices]

    def save(self, path: str):
        """Save index to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'chunks': self.chunks,
                'max_chunk_length': self.max_chunk_length,
                'stride': self.stride
            }, f)
        print(f"Index saved to {path}")

    def load(self, path: str):
        """Load index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunks = data['chunks']
            self.max_chunk_length = data['max_chunk_length']
            self.stride = data['stride']
        print(f"Index loaded from {path}")
        print(f"Loaded {len(self.chunks)} chunks")


if __name__ == '__main__':
    # Test the retriever
    from data_loader import load_dataset
    import os

    print("Loading dataset...")
    text = load_dataset('data')

    # Build or load index
    retriever = BM25Retriever(max_chunk_length=256, stride=128)
    index_path = 'data/retrieval_index_chunked.pkl'

    if os.path.exists(index_path):
        print("\nLoading existing index...")
        retriever.load(index_path)
    else:
        print("\nBuilding new index...")
        retriever.build_index(text)
        print("Saving index...")
        retriever.save(index_path)

    # Test retrieval
    print("\n" + "="*60)
    query = "How are epithelial tissues joined together?"
    print(f"Query: {query}")
    chunks = retriever.retrieve(query, k=1)

    print(f"\nRetrieved {len(chunks)} chunk(s):")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({len(chunk.split())} tokens) ---")
        print(chunk[:500] + ("..." if len(chunk) > 500 else ""))
