"""
Vector store implementation for RAG system.
Supports ChromaDB and Pinecone for persistent vector storage and similarity search.
"""

import os
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime
import json

# Vector database imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available. Install with: pip install chromadb")

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone not available. Install with: pip install pinecone")
except Exception as e:
    PINECONE_AVAILABLE = False
    logging.warning(f"Pinecone import failed: {str(e)}")

# Utilities
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Unified vector store interface supporting multiple backends.
    Handles vector storage, retrieval, and similarity search.
    """
    
    def __init__(self, 
                 backend: str = "chromadb",
                 collection_name: str = "document_chunks",
                 persist_directory: str = "./vector_db",
                 embedding_dimension: int = 384,
                 distance_metric: str = "cosine",
                 **kwargs):
        """
        Initialize vector store.
        
        Args:
            backend: Vector database backend ("chromadb" or "pinecone")
            collection_name: Name of the collection/index
            persist_directory: Directory for persistent storage (ChromaDB)
            embedding_dimension: Dimension of embeddings
            distance_metric: Distance metric for similarity ("cosine", "euclidean", "dot")
            **kwargs: Additional backend-specific arguments
        """
        self.backend = backend
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric
        
        # Initialize backend
        self.client = None
        self.collection = None
        self.index = None
        
        # Create persist directory
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize the selected backend
        self._initialize_backend(**kwargs)
        
        logger.info(f"Vector store initialized with {backend} backend")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Embedding dimension: {embedding_dimension}")
    
    def _initialize_backend(self, **kwargs):
        """Initialize the selected vector database backend."""
        if self.backend == "chromadb":
            self._initialize_chromadb(**kwargs)
        elif self.backend == "pinecone":
            self._initialize_pinecone(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _initialize_chromadb(self, **kwargs):
        """Initialize ChromaDB client and collection."""
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=None  # We'll provide embeddings manually
                )
                logger.info(f"Connected to existing ChromaDB collection: {self.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,
                    metadata={"hnsw:space": self.distance_metric}
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def _initialize_pinecone(self, **kwargs):
        """Initialize Pinecone client and index."""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install with: pip install pinecone-client")
        
        # Get API key from environment or kwargs
        api_key = kwargs.get('api_key') or os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY environment variable.")
        
        try:
            # Initialize Pinecone
            pinecone.init(
                api_key=api_key,
                environment=kwargs.get('environment') or os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')
            )
            
            # Check if index exists
            if self.collection_name not in pinecone.list_indexes():
                # Create index
                pinecone.create_index(
                    name=self.collection_name,
                    dimension=self.embedding_dimension,
                    metric=self.distance_metric
                )
                logger.info(f"Created new Pinecone index: {self.collection_name}")
            else:
                logger.info(f"Connected to existing Pinecone index: {self.collection_name}")
            
            # Connect to index
            self.index = pinecone.Index(self.collection_name)
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def add_vectors(self, 
                   embeddings: np.ndarray,
                   texts: List[str],
                   metadata: Optional[List[Dict[str, Any]]] = None,
                   ids: Optional[List[str]] = None) -> List[str]:
        """
        Add vectors to the store.
        
        Args:
            embeddings: Array of embeddings to store
            texts: Corresponding text content
            metadata: Optional metadata for each vector
            ids: Optional custom IDs (will generate UUIDs if not provided)
            
        Returns:
            List of vector IDs
        """
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings must match number of texts")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        
        # Prepare metadata
        if metadata is None:
            metadata = [{}] * len(embeddings)
        
        # Add timestamps and text to metadata
        current_time = datetime.now().isoformat()
        for i, meta in enumerate(metadata):
            meta.update({
                'text': texts[i],
                'created_at': current_time,
                'text_hash': hashlib.md5(texts[i].encode()).hexdigest()
            })
        
        # Store vectors using the appropriate backend
        if self.backend == "chromadb":
            self._add_vectors_chromadb(embeddings, ids, metadata)
        elif self.backend == "pinecone":
            self._add_vectors_pinecone(embeddings, ids, metadata)
        
        logger.info(f"Added {len(embeddings)} vectors to {self.backend}")
        return ids
    
    def _add_vectors_chromadb(self, embeddings: np.ndarray, ids: List[str], metadata: List[Dict]):
        """Add vectors to ChromaDB."""
        try:
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=[meta['text'] for meta in metadata],
                metadatas=metadata,
                ids=ids
            )
        except Exception as e:
            logger.error(f"Failed to add vectors to ChromaDB: {str(e)}")
            raise
    
    def _add_vectors_pinecone(self, embeddings: np.ndarray, ids: List[str], metadata: List[Dict]):
        """Add vectors to Pinecone."""
        try:
            # Prepare vectors for Pinecone
            vectors = [
                (ids[i], embeddings[i].tolist(), metadata[i])
                for i in range(len(embeddings))
            ]
            
            # Upsert in batches (Pinecone has batch limits)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                
        except Exception as e:
            logger.error(f"Failed to add vectors to Pinecone: {str(e)}")
            raise
    
    def search(self, 
              query_embedding: np.ndarray,
              k: int = 5,
              filter_dict: Optional[Dict[str, Any]] = None,
              include_distances: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_dict: Optional metadata filter
            include_distances: Whether to include similarity scores
            
        Returns:
            List of search results with metadata and optionally distances
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if self.backend == "chromadb":
            return self._search_chromadb(query_embedding, k, filter_dict, include_distances)
        elif self.backend == "pinecone":
            return self._search_pinecone(query_embedding, k, filter_dict, include_distances)
        
        return []
    
    def _search_chromadb(self, query_embedding: np.ndarray, k: int, 
                        filter_dict: Optional[Dict], include_distances: bool) -> List[Dict]:
        """Search ChromaDB collection."""
        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k,
                where=filter_dict,
                include=['documents', 'metadatas', 'distances'] if include_distances else ['documents', 'metadatas']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i]
                }
                
                if include_distances and 'distances' in results:
                    # Convert distance to similarity score (higher is better)
                    distance = results['distances'][0][i]
                    if self.distance_metric == "cosine":
                        similarity = 1 - distance
                    else:
                        similarity = 1 / (1 + distance)  # Convert distance to similarity
                    result['score'] = similarity
                
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {str(e)}")
            return []
    
    def _search_pinecone(self, query_embedding: np.ndarray, k: int,
                        filter_dict: Optional[Dict], include_distances: bool) -> List[Dict]:
        """Search Pinecone index."""
        try:
            results = self.index.query(
                vector=query_embedding[0].tolist(),
                top_k=k,
                filter=filter_dict,
                include_metadata=True,
                include_values=False
            )
            
            # Format results
            formatted_results = []
            for match in results['matches']:
                result = {
                    'id': match['id'],
                    'text': match['metadata'].get('text', ''),
                    'metadata': match['metadata']
                }
                
                if include_distances:
                    result['score'] = match['score']
                
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Pinecone search failed: {str(e)}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            True if successful
        """
        try:
            if self.backend == "chromadb":
                self.collection.delete(ids=ids)
            elif self.backend == "pinecone":
                self.index.delete(ids=ids)
            
            logger.info(f"Deleted {len(ids)} vectors from {self.backend}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            return False
    
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a vector.
        
        Args:
            id: Vector ID
            metadata: New metadata
            
        Returns:
            True if successful
        """
        try:
            if self.backend == "chromadb":
                # ChromaDB doesn't support direct metadata updates
                # Would need to delete and re-add
                logger.warning("ChromaDB doesn't support metadata updates without re-adding")
                return False
            elif self.backend == "pinecone":
                self.index.update(id=id, metadata=metadata)
                return True
                
        except Exception as e:
            logger.error(f"Failed to update metadata: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            if self.backend == "chromadb":
                count = self.collection.count()
                return {
                    'backend': self.backend,
                    'collection_name': self.collection_name,
                    'total_vectors': count,
                    'embedding_dimension': self.embedding_dimension
                }
            elif self.backend == "pinecone":
                stats = self.index.describe_index_stats()
                return {
                    'backend': self.backend,
                    'collection_name': self.collection_name,
                    'total_vectors': stats.get('total_vector_count', 0),
                    'embedding_dimension': self.embedding_dimension,
                    'index_fullness': stats.get('index_fullness', 0)
                }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all vectors from the collection."""
        try:
            if self.backend == "chromadb":
                # Reset collection
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,
                    metadata={"hnsw:space": self.distance_metric}
                )
            elif self.backend == "pinecone":
                self.index.delete(delete_all=True)
            
            logger.info(f"Cleared all vectors from {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False
    
    def export_vectors(self, output_path: str) -> bool:
        """Export all vectors and metadata to a file."""
        try:
            if self.backend == "chromadb":
                # Get all data from ChromaDB
                results = self.collection.get(include=['documents', 'metadatas', 'embeddings'])
                
                export_data = {
                    'backend': self.backend,
                    'collection_name': self.collection_name,
                    'embedding_dimension': self.embedding_dimension,
                    'vectors': {
                        'ids': results['ids'],
                        'embeddings': [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in results['embeddings']],
                        'documents': results['documents'],
                        'metadatas': results['metadatas']
                    }
                }
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                logger.info(f"Exported {len(results['ids'])} vectors to {output_path}")
                return True
            else:
                logger.warning("Export not implemented for Pinecone backend")
                return False
                
        except Exception as e:
            logger.error(f"Failed to export vectors: {str(e)}")
            return False


def main():
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
        
        print("\nVector store test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")


if __name__ == "__main__":
    main()