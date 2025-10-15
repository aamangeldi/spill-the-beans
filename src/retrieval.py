"""Simple BM25-style document retrieval using sklearn."""
import pickle
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class BM25Retriever:
    """Simple TF-IDF retriever (approximates BM25)."""

    def __init__(self):
        """Initialize retriever."""
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.doc_vectors = None
        self.documents = None

    def build_index(self, documents: List[Dict[str, str]]):
        """Build retrieval index from documents.

        Args:
            documents: List of dicts with 'title' and 'text' keys
        """
        self.documents = documents

        # Combine title and text for indexing
        corpus = [f"{doc['title']} {doc['text']}" for doc in documents]

        print(f"Building index from {len(corpus)} documents...")
        self.doc_vectors = self.vectorizer.fit_transform(corpus)
        print(f"Index built with vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve top-k documents for a query.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of retrieved document texts (title + text)
        """
        if self.doc_vectors is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Vectorize query
        query_vector = self.vectorizer.transform([query])

        # Compute similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]

        # Return full documents
        retrieved = []
        for idx in top_k_indices:
            doc = self.documents[idx]
            full_text = f"{doc['title']}\n{doc['text']}"
            retrieved.append(full_text)

        return retrieved

    def save(self, path: str):
        """Save index to disk."""
        with open(path, 'wb') as f:
            pickle.dump((self.vectorizer, self.doc_vectors, self.documents), f)

    def load(self, path: str):
        """Load index from disk."""
        with open(path, 'rb') as f:
            self.vectorizer, self.doc_vectors, self.documents = pickle.load(f)


if __name__ == '__main__':
    # Test the retriever
    import os
    from data_loader import load_wiki_dataset

    print("Loading dataset...")
    articles = load_wiki_dataset('data/wiki_newest.txt')
    print(f"Loaded {len(articles)} articles")

    # Build or load index
    retriever = BM25Retriever()
    index_path = 'data/retrieval_index.pkl'

    if os.path.exists(index_path):
        print("Loading existing index...")
        retriever.load(index_path)
    else:
        retriever.build_index(articles)
        print("Saving index...")
        retriever.save(index_path)

    # Test retrieval
    query = "British actor awards Emmy BAFTA"
    print(f"\nQuery: {query}")
    docs = retriever.retrieve(query, k=3)
    print(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(doc[:300] + "...")
