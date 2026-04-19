"""RAG Pipeline Implementation

Core Algorithm:
1. Load documents (TXT, PDF, MD) from disk
2. Chunk documents with sliding window (overlap preserves context)
3. Encode chunks to embeddings using sentence-transformers
4. Store embeddings in FAISS index for similarity search
5. On query: encode query, find k-NN in FAISS, return chunks
"""
import os
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader


class RAGPipeline:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        # Load embedding model - converts text to vector space
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Document chunking parameters
        self.chunk_size = 400  # words per chunk
        self.chunk_overlap = 50  # word overlap (sliding window)

        # FAISS index - efficient similarity search (L2 distance)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Store chunk texts and metadata
        self.documents: List[str] = []
        self.metadata: List[Dict] = []

    def _load_pdf(self, filepath: str) -> str:
        """Extract text from a PDF file"""
        text = ""
        try:
            with open(filepath, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF {filepath}: {e}")
        return text

    def _chunk_document(self, text: str, doc_id: int) -> List[Tuple[str, Dict]]:
        """Split document using sliding window overlapping chunks
        
        Algorithm: Sliding window with overlap
        - Window size: chunk_size words
        - Stride: chunk_size - chunk_overlap words
        - Overlap preserves context at boundaries
        """
        chunks = []
        words = text.split()
        stride = self.chunk_size - self.chunk_overlap  # how many words to advance

        # Create overlapping chunks using sliding window
        for i in range(0, len(words), stride):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            if len(chunk_text.strip()) > 0:
                meta = {
                    "doc_id": doc_id,
                    "chunk_index": len(chunks),
                    "text": chunk_text,
                }
                chunks.append((chunk_text, meta))

        return chunks

    def ingest_documents(self, file_paths: List[str]) -> Dict:
        """Load documents, chunk, embed, and store in FAISS
        
        Steps:
        1. For each file: read text (handle PDF and text files)
        2. Normalize whitespace
        3. Split into overlapping chunks
        4. For each chunk: encode to embedding vector
        5. Add embedding to FAISS index
        6. Store chunk text and metadata
        """
        total_chunks = 0

        for doc_id, filepath in enumerate(file_paths):
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                continue

            print(f"Loading: {filepath}")

            # Step 1: Read file based on type
            if filepath.lower().endswith('.pdf'):
                text = self._load_pdf(filepath)
            else:
                # Read as text file (TXT, MD, etc.)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()

            # Step 2: Normalize whitespace
            text = " ".join(text.split())
            
            # Step 3: Break into chunks
            chunks = self._chunk_document(text, doc_id)

            # Steps 4-6: Encode and store each chunk
            for chunk_text, meta in chunks:
                # Encode to embedding vector
                embedding = self.embedding_model.encode(chunk_text, convert_to_numpy=True)
                embedding = np.array([embedding], dtype=np.float32)

                # Add to FAISS
                self.index.add(embedding)

                # Store the text and metadata
                self.documents.append(chunk_text)
                self.metadata.append(meta)
                total_chunks += 1

        print(f"Ingested {total_chunks} chunks total")
        return {
            "status": "success",
            "documents": len(file_paths),
            "chunks": total_chunks,
        }

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Find top-k most similar chunks to query using semantic similarity
        
        Algorithm (k-NN search):
        1. Encode query to embedding vector (same space as chunks)
        2. Search FAISS for k nearest neighbors
        3. Return chunks with similarity scores (L2 distances)
        
        Lower L2 distance = more similar
        """
        if len(self.documents) == 0:
            return []

        # Step 1: Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = np.array([query_embedding], dtype=np.float32)

        # Step 2: Find k-NN in FAISS index
        k = min(k, len(self.documents))  # Can't return more than we have
        distances, indices = self.index.search(query_embedding, k)

        # Step 3: Format results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1:  # Valid result
                result = {
                    "chunk": self.documents[int(idx)],
                    "metadata": self.metadata[int(idx)],
                    "distance": float(distance),  # L2 distance (lower = better match)
                }
                results.append(result)

        return results

    def get_index_stats(self) -> Dict:
        """Return info about what's stored"""
        return {
            "total_chunks": len(self.documents),
            "embedding_dim": self.embedding_dim,
            "embedding_model": "all-MiniLM-L6-v2",
        }
