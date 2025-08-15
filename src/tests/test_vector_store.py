"""
Test script for vector store implementation.
Tests ChromaDB and Pinecone vector storage, retrieval, and similarity search.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.vector_store.vector_store import VectorStore


def test_vector_store():
    """Test the vector store implementation."""
    
    # Test data
    test_embeddings = np.random.random((5, 384)).astype(np.float32)
    test_texts = [
        "This is the first test document about machine learning.",
        "Second document discusses natural language processing.",
        "Third document covers computer vision topics.",
        "Fourth document is about data science methodologies.",
        "Fifth document explores deep learning architectures."
    ]
    
    test_metadata = [
        {"category": "ML", "topic": "general"},
        {"category": "NLP", "topic": "processing"},
        {"category": "CV", "topic": "vision"},
        {"category": "DS", "topic": "methods"},
        {"category": "DL", "topic": "architecture"}
    ]
    
    # Test ChromaDB
    print("Testing ChromaDB Vector Store")
    print("=" * 40)
    
    try:
        # Initialize vector store
        vector_store = VectorStore(
            backend="chromadb",
            collection_name="test_collection",
            persist_directory="./test_vector_db"
        )
        
        # Add vectors
        print("Adding test vectors...")
        ids = vector_store.add_vectors(test_embeddings, test_texts, test_metadata)
        print(f"Added {len(ids)} vectors")
        
        # Get stats
        stats = vector_store.get_collection_stats()
        print(f"Collection stats: {stats}")
        
        # Test search
        print("\nTesting similarity search...")
        query_embedding = np.random.random((1, 384)).astype(np.float32)
        results = vector_store.search(query_embedding, k=3)
        
        print(f"Found {len(results)} similar vectors:")
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result.get('score', 'N/A'):.3f}")
            print(f"   Text: {result['text'][:50]}...")
            print(f"   Metadata: {result['metadata']}")
        
        # Test export
        print("\nExporting vectors...")
        if vector_store.export_vectors("test_export.json"):
            print("Export successful")
        
        # Test delete
        print("\nTesting vector deletion...")
        if len(ids) > 0:
            delete_ids = ids[:2]  # Delete first 2 vectors
            if vector_store.delete_vectors(delete_ids):
                print(f"Successfully deleted {len(delete_ids)} vectors")
                
                # Verify deletion by checking stats
                new_stats = vector_store.get_collection_stats()
                print(f"Vectors after deletion: {new_stats.get('total_vectors', 'N/A')}")
        
        # Test clear collection
        print("\nTesting collection clearing...")
        if vector_store.clear_collection():
            print("Collection cleared successfully")
            
            # Verify clearing
            final_stats = vector_store.get_collection_stats()
            print(f"Final vector count: {final_stats.get('total_vectors', 'N/A')}")
        
        print("\nChromaDB vector store test completed successfully!")
        
    except Exception as e:
        print(f"ChromaDB test failed: {str(e)}")
        print("Note: Make sure ChromaDB is installed: pip install chromadb")
    
    print("\n" + "=" * 50)
    
    # Test Pinecone (if available)
    print("Testing Pinecone Vector Store")
    print("=" * 40)
    
    try:
        # Check if Pinecone API key is available
        if not os.getenv('PINECONE_API_KEY'):
            print("Skipping Pinecone test - no API key found")
            print("Set PINECONE_API_KEY environment variable to test Pinecone")
            return
        
        # Initialize Pinecone vector store
        pinecone_store = VectorStore(
            backend="pinecone",
            collection_name="test-index",
            embedding_dimension=384
        )
        
        # Add vectors
        print("Adding test vectors to Pinecone...")
        pinecone_ids = pinecone_store.add_vectors(test_embeddings, test_texts, test_metadata)
        print(f"Added {len(pinecone_ids)} vectors to Pinecone")
        
        # Get stats
        pinecone_stats = pinecone_store.get_collection_stats()
        print(f"Pinecone stats: {pinecone_stats}")
        
        # Test search
        print("\nTesting Pinecone similarity search...")
        query_embedding = np.random.random((1, 384)).astype(np.float32)
        pinecone_results = pinecone_store.search(query_embedding, k=3)
        
        print(f"Found {len(pinecone_results)} similar vectors in Pinecone:")
        for i, result in enumerate(pinecone_results):
            print(f"{i+1}. Score: {result.get('score', 'N/A'):.3f}")
            print(f"   Text: {result['text'][:50]}...")
            print(f"   Metadata: {result['metadata']}")
        
        print("\nPinecone vector store test completed successfully!")
        
    except Exception as e:
        print(f"Pinecone test failed: {str(e)}")
        print("Note: Make sure Pinecone is installed and API key is set")
        print("Install: pip install pinecone-client")
        print("API Key: export PINECONE_API_KEY=your_key_here")


def test_vector_store_integration():
    """Test vector store integration with different embedding dimensions."""
    
    print("\n" + "=" * 60)
    print("Testing Vector Store Integration with Different Dimensions")
    print("=" * 60)
    
    # Test different embedding dimensions
    dimensions = [128, 256, 512, 768]
    
    for dim in dimensions:
        print(f"\nTesting with {dim}-dimensional embeddings...")
        
        try:
            # Create test data for this dimension
            test_embeddings = np.random.random((3, dim)).astype(np.float32)
            test_texts = [
                f"Test document 1 with {dim}D embeddings",
                f"Test document 2 with {dim}D embeddings", 
                f"Test document 3 with {dim}D embeddings"
            ]
            
            # Initialize vector store
            vector_store = VectorStore(
                backend="chromadb",
                collection_name=f"test_collection_{dim}d",
                persist_directory=f"./test_vector_db_{dim}d",
                embedding_dimension=dim
            )
            
            # Add and search
            ids = vector_store.add_vectors(test_embeddings, test_texts)
            query_embedding = np.random.random((1, dim)).astype(np.float32)
            results = vector_store.search(query_embedding, k=2)
            
            print(f"{dim}D: Added {len(ids)} vectors, found {len(results)} similar")
            
        except Exception as e:
            print(f"{dim}D: Failed - {str(e)}")
    
    print("\nIntegration testing completed!")


def benchmark_vector_store():
    """Benchmark vector store performance."""
    
    print("\n" + "=" * 60)
    print("Benchmarking Vector Store Performance")
    print("=" * 60)
    
    import time
    
    # Test with different data sizes
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\nBenchmarking with {size} vectors...")
        
        try:
            # Generate test data
            embeddings = np.random.random((size, 384)).astype(np.float32)
            texts = [f"Test document {i} for benchmarking" for i in range(size)]
            
            # Initialize vector store
            vector_store = VectorStore(
                backend="chromadb",
                collection_name=f"benchmark_{size}",
                persist_directory=f"./benchmark_db_{size}"
            )
            
            # Benchmark adding vectors
            start_time = time.time()
            ids = vector_store.add_vectors(embeddings, texts)
            add_time = time.time() - start_time
            
            # Benchmark search
            query_embedding = np.random.random((1, 384)).astype(np.float32)
            start_time = time.time()
            results = vector_store.search(query_embedding, k=10)
            search_time = time.time() - start_time
            
            print(f"{size} vectors:")
            print(f"Add time: {add_time:.3f}s ({size/add_time:.1f} vectors/sec)")
            print(f"Search time: {search_time:.3f}s")
            print(f"Found {len(results)} results")
            
        except Exception as e:
            print(f"Benchmark failed for {size} vectors: {str(e)}")
    
    print("\nBenchmarking completed!")


def main():
    """Run all vector store tests."""
    print("Starting Vector Store Test Suite")
    print("=" * 60)
    
    # Run main functionality tests
    test_vector_store()
    
    # Run integration tests
    test_vector_store_integration()
    
    # Run performance benchmarks
    benchmark_vector_store()
    
    print("\nAll vector store tests have been completed")
    print("=" * 60)


if __name__ == "__main__":
    main()