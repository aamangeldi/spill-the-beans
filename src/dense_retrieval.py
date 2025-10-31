"""Dense retrieval using sentence embeddings."""
import pickle
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """Dense retriever using sentence embeddings with token-based chunking."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', tokenizer=None,
                 max_chunk_length: int = 256, stride: int = 128):
        """Initialize retriever.

        Args:
            model_name: Sentence transformer model name (default: all-MiniLM-L6-v2)
            tokenizer: HuggingFace tokenizer (optional, uses whitespace if None)
            max_chunk_length: Maximum tokens per chunk (paper uses 256)
            stride: Stride for overlapping chunks (paper uses 128)
        """
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_chunk_length = max_chunk_length
        self.stride = stride
        self.model = None
        self.embeddings = None  # Numpy array of chunk embeddings
        self.chunks = None  # List of chunk strings

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            print(f"Loading sentence transformer model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully")

    def build_index(self, text: str):
        """Build dense index from continuous text stream.

        1. Tokenize all text
        2. Split into chunks (256 tokens, 128 stride) - same as BM25
        3. Encode chunks to embeddings
        4. Normalize for cosine similarity

        Args:
            text: Concatenated text from all documents
        """
        print(f"Tokenizing text ({len(text)} chars, {len(text.split())} words)...")

        if self.tokenizer:
            # Use HuggingFace tokenizer - same logic as BM25Retriever
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
            # Simple whitespace tokenization - same as BM25Retriever
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

        # Encode chunks to embeddings
        print(f"Encoding {len(self.chunks)} chunks with {self.model_name}...")
        self._load_model()

        # Encode in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Encoded {i + len(batch)}/{len(self.chunks)} chunks...")

        self.embeddings = np.vstack(all_embeddings)

        # Normalize embeddings for cosine similarity (dot product = cosine for normalized vectors)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms

        print(f"Index built successfully: {self.embeddings.shape}")

    def _get_token_chunks(self, tokens: np.ndarray, pad_token: int) -> np.ndarray:
        """Split tokens into overlapping chunks.

        Same implementation as BM25Retriever for consistency.

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
        """Retrieve top-k chunks for a query using cosine similarity.

        Args:
            query: Search query
            k: Number of chunks to retrieve

        Returns:
            List of retrieved chunk texts
        """
        if self.embeddings is None or self.chunks is None:
            raise ValueError("Index not built. Call build_index() first.")

        self._load_model()

        # Encode query
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]

        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Compute cosine similarity (dot product with normalized vectors)
        scores = np.dot(self.embeddings, query_embedding)

        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]

        # Return chunks
        return [self.chunks[idx] for idx in top_k_indices]

    def save(self, path: str):
        """Save index to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model_name': self.model_name,
                'embeddings': self.embeddings,
                'chunks': self.chunks,
                'max_chunk_length': self.max_chunk_length,
                'stride': self.stride
            }, f)
        print(f"Index saved to {path}")

    def load(self, path: str):
        """Load index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model_name = data['model_name']
            self.embeddings = data['embeddings']
            self.chunks = data['chunks']
            self.max_chunk_length = data['max_chunk_length']
            self.stride = data['stride']
        print(f"Index loaded from {path}")
        print(f"Loaded {len(self.chunks)} chunks with {self.embeddings.shape[1]}-dim embeddings")


if __name__ == '__main__':
    # Test the retriever
    from data_loader import load_dataset
    import os

    print("Loading dataset...")
    text = load_dataset('data')

    # Build or load index
    retriever = DenseRetriever(model_name='all-MiniLM-L6-v2', max_chunk_length=256, stride=128)
    index_path = 'data/retrieval_index_dense_all-MiniLM-L6-v2.pkl'

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
        print(f"\n--- Chunk {i+1} ({len(chunk.split())} words) ---")
        print(chunk[:500] + ("..." if len(chunk) > 500 else ""))
