"""BM25 document retrieval with chunking strategy from the paper."""
import pickle
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import numpy as np


def chunk_document(text: str, max_length: int = 256, stride: int = 128) -> List[str]:
    """Chunk document into overlapping segments.

    Following paper's approach:
    - max_retrieval_seq_length: 256 tokens
    - stride: 128 tokens (50% overlap)

    Args:
        text: Document text to chunk
        max_length: Maximum tokens per chunk
        stride: Number of tokens to stride between chunks

    Returns:
        List of text chunks
    """
    # Simple whitespace tokenization (approximates real tokens)
    tokens = text.split()

    # Handle empty documents
    if len(tokens) == 0:
        return []

    if len(tokens) <= max_length:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(' '.join(chunk_tokens))

        # If we've reached the end, break
        if end == len(tokens):
            break

        # Move forward by stride
        start += stride

    return chunks


class BM25Retriever:
    """BM25 retriever with document chunking."""

    def __init__(self, max_chunk_length: int = 256, stride: int = 128):
        """Initialize retriever.

        Args:
            max_chunk_length: Maximum tokens per chunk (paper uses 256)
            stride: Stride for overlapping chunks (paper uses 128)
        """
        self.max_chunk_length = max_chunk_length
        self.stride = stride
        self.bm25 = None
        self.chunks = None  # List of (chunk_text, source_doc_title, chunk_idx)

    def build_index(self, documents: List[Dict[str, str]]):
        """Build BM25 index from documents with chunking.

        Args:
            documents: List of dicts with 'title' and 'text' keys
        """
        print(f"Chunking {len(documents)} documents (max_length={self.max_chunk_length}, stride={self.stride})...")

        all_chunks = []
        chunk_metadata = []

        for doc in documents:
            # Combine title and text
            full_text = f"{doc['title']}\n{doc['text']}"

            # Chunk the document
            doc_chunks = chunk_document(full_text, self.max_chunk_length, self.stride)

            # Store chunks with metadata
            for i, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'title': doc['title'],
                    'chunk_idx': i,
                    'total_chunks': len(doc_chunks),
                    'text': chunk
                })

        self.chunks = chunk_metadata

        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        print(f"Average chunks per document: {len(all_chunks)/len(documents):.1f}")

        # Tokenize chunks for BM25 (simple whitespace tokenization)
        tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]

        # Build BM25 index
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_chunks)
        print("Index built successfully")

    def retrieve(self, query: str, k: int = 1) -> List[str]:
        """Retrieve top-k chunks for a query.

        Args:
            query: Search query
            k: Number of chunks to retrieve (paper uses num_document=1)

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
        retrieved = []
        for idx in top_k_indices:
            chunk_info = self.chunks[idx]
            retrieved.append(chunk_info['text'])

        return retrieved

    def save(self, path: str):
        """Save index to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'chunks': self.chunks,
                'max_chunk_length': self.max_chunk_length,
                'stride': self.stride
            }, f)

    def load(self, path: str):
        """Load index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunks = data['chunks']
            self.max_chunk_length = data['max_chunk_length']
            self.stride = data['stride']


if __name__ == '__main__':
    # Test the retriever
    import os
    from data_loader import load_wiki_dataset

    print("Loading dataset...")
    articles = load_wiki_dataset('data/wiki_newest.txt')
    print(f"Loaded {len(articles)} articles")

    # Build or load index
    retriever = BM25Retriever(max_chunk_length=256, stride=128)
    index_path = 'data/retrieval_index_chunked.pkl'

    if os.path.exists(index_path):
        print("\nLoading existing index...")
        retriever.load(index_path)
    else:
        print("\nBuilding new index...")
        retriever.build_index(articles)
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
