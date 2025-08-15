"""
Base vector store class providing unified interface for multiple backends.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
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
        
        # Backend implementation instance
        self.backend_impl = None
        
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
            from .chromadb_backend import ChromaDBBackend
            self.backend_impl = ChromaDBBackend(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                distance_metric=self.distance_metric,
                **kwargs
            )
        elif self.backend == "pinecone":
            from .pinecone_backend import PineconeBackend
            self.backend_impl = PineconeBackend(
                collection_name=self.collection_name,
                embedding_dimension=self.embedding_dimension,
                distance_metric=self.distance_metric,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
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
        
        # Delegate to backend implementation
        self.backend_impl.add_vectors(embeddings, ids, metadata)
        
        logger.info(f"Added {len(embeddings)} vectors to {self.backend}")
        return ids
    
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
        
        return self.backend_impl.search(query_embedding, k, filter_dict, include_distances)
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            True if successful
        """
        return self.backend_impl.delete_vectors(ids)
    
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a vector.
        
        Args:
            id: Vector ID
            metadata: New metadata
            
        Returns:
            True if successful
        """
        return self.backend_impl.update_metadata(id, metadata)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        stats = self.backend_impl.get_collection_stats()
        stats.update({
            'backend': self.backend,
            'collection_name': self.collection_name,
            'embedding_dimension': self.embedding_dimension
        })
        return stats
    
    def clear_collection(self) -> bool:
        """Clear all vectors from the collection."""
        return self.backend_impl.clear_collection()
    
    def export_vectors(self, output_path: str) -> bool:
        """Export all vectors and metadata to a file."""
        return self.backend_impl.export_vectors(output_path)