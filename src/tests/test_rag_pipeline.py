"""
Test script for the complete RAG pipeline.
Demonstrates document processing, embedding generation, and similarity search.
"""

import sys
import os
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.document_processor import DocumentProcessor
from src.core.embedding_engine import EmbeddingEngine

def test_rag_pipeline():
    """Test the complete RAG pipeline with the rabbit care guide."""
    
    print("Testing RAG Pipeline with Rabbit Care Guide")
    print("=" * 50)
    
    # Process document (adjust path since we're in tests/ folder)
    print("1. Processing document...")
    processor = DocumentProcessor()
    result = processor.process_document("../test_document.pdf")
    print(f"   Created {len(result['chunks'])} chunks")
    
    # Create embeddings
    print("\n2. Generating embeddings...")
    engine = EmbeddingEngine()
    chunk_texts = [chunk['text'] for chunk in result['chunks']]
    embeddings = engine.encode_texts(chunk_texts)
    print(f"   Generated embeddings: {embeddings.shape}")
    
    # Test queries
    queries = [
        "How much hay should I feed my rabbit?",
        "What vegetables are safe for rabbits?",
        "How big should a rabbit enclosure be?",
        "When should I spay my rabbit?"
    ]
    
    print("\n3. Testing similarity search...")
    for query in queries:
        print(f"\nQuery: {query}")
        query_embedding = engine.encode_texts([query])
        indices, scores = engine.find_most_similar(query_embedding[0], embeddings, top_k=2)
        
        for i, (idx, score) in enumerate(zip(indices, scores)):
            print(f"  Match {i+1} (score: {score:.3f}):")
            print(f"  {chunk_texts[idx][:150]}...")
            print()

if __name__ == "__main__":
    test_rag_pipeline()