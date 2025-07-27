"""
Test script for the complete RAG pipeline.
Document processing, embedding generation, and similarity search.
"""

import sys
import os
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.document_processor import DocumentProcessor
from core.embedding_engine import EmbeddingEngine
from core.vector_store import VectorStore

def test_rag_pipeline():
    """Test the complete RAG pipeline with the rabbit care guide."""
    
    print("Testing RAG Pipeline with Rabbit Care Guide")
    print("=" * 50)

    # Debug paths
    current_dir = os.path.dirname(__file__)
    project_root = os.path.join(current_dir, "..", "..")
    pdf_path = os.path.join(project_root, "test_document.pdf")
    abs_pdf_path = os.path.abspath(pdf_path)
    
    print(f"Current directory: {current_dir}")
    print(f"PDF path: {abs_pdf_path}")
    print(f"PDF exists: {os.path.exists(abs_pdf_path)}")
    
    # Process document
    print("\n1. Processing document...")
    processor = DocumentProcessor()
    result = processor.process_document(abs_pdf_path)
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
    
    # Test in-memory similarity search
    print("\n3. Testing in-memory similarity search...")
    for query in queries:
        print(f"\nQuery: {query}")
        query_embedding = engine.encode_texts([query])
        indices, scores = engine.find_most_similar(query_embedding[0], embeddings, top_k=2)
        
        for i, (idx, score) in enumerate(zip(indices, scores)):
            print(f"  Match {i+1} (score: {score:.3f}):")
            print(f"  {chunk_texts[idx][:150]}...")
            print()

    # Test vector store integration
    print("\n4. Testing vector store integration...")
    
    # Initialize vector store
    vector_store = VectorStore(
        backend="chromadb",
        collection_name="rabbit_care_guide",
        persist_directory="./rabbit_vector_db"
    )
    
    # Store embeddings in vector database
    print("   Storing embeddings in vector database...")
    chunk_metadata = [{"chunk_id": i, "document": "rabbit_care_guide"} for i in range(len(chunk_texts))]
    vector_ids = vector_store.add_vectors(embeddings, chunk_texts, chunk_metadata)
    print(f"   Stored {len(vector_ids)} vectors in ChromaDB")
    
    # Test persistent search
    print("\n   Testing persistent vector search...")
    for query in queries[:2]:  # Test with first 2 queries
        print(f"\n   Query: {query}")
        query_embedding = engine.encode_texts([query])
        results = vector_store.search(query_embedding[0], k=2)
        
        for i, result in enumerate(results):
            print(f"     Match {i+1} (score: {result.get('score', 'N/A'):.3f}):")
            print(f"     {result['text'][:100]}...")
    
    # Show collection stats
    stats = vector_store.get_collection_stats()
    print(f"\nVector store stats: {stats}")
    
    print("\nFull RAG pipeline with vector store completed!")

if __name__ == "__main__":
    test_rag_pipeline()