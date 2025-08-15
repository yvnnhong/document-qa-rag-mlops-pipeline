"""
Utility functions for vector store operations.
Provides helper functions for validation, conversion, and common operations.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class VectorStoreUtils:
    """
    Utility class providing helper functions for vector store operations.
    """
    
    @staticmethod
    def validate_inputs(embeddings: np.ndarray, texts: List[str], metadata: Optional[List[Dict]] = None) -> bool:
        """
        Validate inputs for vector store operations.
        
        Args:
            embeddings: Array of embeddings
            texts: List of text content
            metadata: Optional metadata list
            
        Returns:
            True if inputs are valid
            
        Raises:
            ValueError: If inputs are invalid
        """
        if len(embeddings) != len(texts):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of texts ({len(texts)})")
        
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError(f"Number of metadata items ({len(metadata)}) must match number of texts ({len(texts)})")
        
        if len(embeddings) == 0:
            raise ValueError("Cannot add empty list of embeddings")
        
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings must be a numpy array")
        
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        
        return True
    
    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length for cosine similarity.
        
        Args:
            embeddings: Input embeddings array
            
        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    @staticmethod
    def compute_similarity_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray, metric: str = "cosine") -> np.ndarray:
        """
        Compute similarity matrix between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            metric: Similarity metric ("cosine", "dot", "euclidean")
            
        Returns:
            Similarity matrix
        """
        if metric == "cosine":
            # Normalize and compute dot product
            norm1 = VectorStoreUtils.normalize_embeddings(embeddings1)
            norm2 = VectorStoreUtils.normalize_embeddings(embeddings2)
            return np.dot(norm1, norm2.T)
        elif metric == "dot":
            return np.dot(embeddings1, embeddings2.T)
        elif metric == "euclidean":
            # Compute negative euclidean distance (higher = more similar)
            from sklearn.metrics.pairwise import euclidean_distances
            return -euclidean_distances(embeddings1, embeddings2)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    @staticmethod
    def create_text_hash(text: str) -> str:
        """
        Create a hash for text content for deduplication.
        
        Args:
            text: Input text
            
        Returns:
            MD5 hash of the text
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def enrich_metadata(texts: List[str], metadata: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Enrich metadata with timestamps and text hashes.
        
        Args:
            texts: List of text content
            metadata: Optional existing metadata
            
        Returns:
            Enriched metadata list
        """
        if metadata is None:
            metadata = [{}] * len(texts)
        
        current_time = datetime.now().isoformat()
        
        enriched_metadata = []
        for i, (text, meta) in enumerate(zip(texts, metadata)):
            enriched_meta = meta.copy()
            enriched_meta.update({
                'text': text,
                'created_at': current_time,
                'text_hash': VectorStoreUtils.create_text_hash(text),
                'char_count': len(text),
                'word_count': len(text.split())
            })
            enriched_metadata.append(enriched_meta)
        
        return enriched_metadata
    
    @staticmethod
    def filter_results_by_score(results: List[Dict], min_score: float = 0.0) -> List[Dict]:
        """
        Filter search results by minimum similarity score.
        
        Args:
            results: List of search results
            min_score: Minimum similarity score threshold
            
        Returns:
            Filtered results
        """
        return [result for result in results if result.get('score', 0) >= min_score]
    
    @staticmethod
    def deduplicate_results(results: List[Dict], key: str = 'text_hash') -> List[Dict]:
        """
        Remove duplicate results based on a key in metadata.
        
        Args:
            results: List of search results
            key: Key to use for deduplication (default: 'text_hash')
            
        Returns:
            Deduplicated results (keeps first occurrence)
        """
        seen = set()
        deduplicated = []
        
        for result in results:
            # Get the dedup key from metadata
            dedup_value = result.get('metadata', {}).get(key)
            
            if dedup_value is None:
                # If key doesn't exist, keep the result
                deduplicated.append(result)
            elif dedup_value not in seen:
                seen.add(dedup_value)
                deduplicated.append(result)
        
        return deduplicated
    
    @staticmethod
    def format_search_results(results: List[Dict], include_text_preview: bool = True, preview_length: int = 100) -> List[Dict]:
        """
        Format search results for display.
        
        Args:
            results: Raw search results
            include_text_preview: Whether to include text preview
            preview_length: Length of text preview
            
        Returns:
            Formatted results
        """
        formatted = []
        
        for i, result in enumerate(results):
            formatted_result = {
                'rank': i + 1,
                'id': result.get('id', 'unknown'),
                'score': result.get('score', 0.0),
                'metadata': result.get('metadata', {})
            }
            
            if include_text_preview:
                text = result.get('text', '')
                if len(text) > preview_length:
                    formatted_result['text_preview'] = text[:preview_length] + '...'
                else:
                    formatted_result['text_preview'] = text
            
            formatted.append(formatted_result)
        
        return formatted
    
    @staticmethod
    def batch_process_embeddings(embeddings: np.ndarray, batch_size: int = 100) -> List[np.ndarray]:
        """
        Split embeddings into batches for processing.
        
        Args:
            embeddings: Input embeddings array
            batch_size: Size of each batch
            
        Returns:
            List of embedding batches
        """
        batches = []
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    @staticmethod
    def estimate_memory_usage(num_vectors: int, embedding_dim: int, metadata_size_per_vector: int = 500) -> Dict[str, float]:
        """
        Estimate memory usage for vector storage.
        
        Args:
            num_vectors: Number of vectors
            embedding_dim: Dimension of embeddings
            metadata_size_per_vector: Estimated metadata size per vector in bytes
            
        Returns:
            Dictionary with memory estimates in MB
        """
        # Embedding storage (float32 = 4 bytes per dimension)
        embedding_size_mb = (num_vectors * embedding_dim * 4) / (1024 * 1024)
        
        # Metadata storage
        metadata_size_mb = (num_vectors * metadata_size_per_vector) / (1024 * 1024)
        
        # Index overhead (estimated at 20% of embedding size)
        index_overhead_mb = embedding_size_mb * 0.2
        
        total_mb = embedding_size_mb + metadata_size_mb + index_overhead_mb
        
        return {
            'embeddings_mb': round(embedding_size_mb, 2),
            'metadata_mb': round(metadata_size_mb, 2),
            'index_overhead_mb': round(index_overhead_mb, 2),
            'total_estimated_mb': round(total_mb, 2)
        }
    
    @staticmethod
    def export_results_to_json(results: List[Dict], output_path: str, include_metadata: bool = True) -> bool:
        """
        Export search results to JSON file.
        
        Args:
            results: Search results to export
            output_path: Path to save the JSON file
            include_metadata: Whether to include full metadata
            
        Returns:
            True if successful
        """
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'total_results': len(results),
                'results': []
            }
            
            for result in results:
                export_result = {
                    'id': result.get('id'),
                    'score': result.get('score'),
                    'text': result.get('text', '')
                }
                
                if include_metadata:
                    export_result['metadata'] = result.get('metadata', {})
                
                export_data['results'].append(export_result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(results)} results to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")
            return False
    
    @staticmethod
    def validate_backend_compatibility(backend: str, operation: str) -> bool:
        """
        Check if a backend supports a specific operation.
        
        Args:
            backend: Backend name ("chromadb" or "pinecone")
            operation: Operation name
            
        Returns:
            True if operation is supported
        """
        compatibility_matrix = {
            'chromadb': {
                'add_vectors': True,
                'search': True,
                'delete_vectors': True,
                'update_metadata': False,  # Not supported
                'export_vectors': True,
                'clear_collection': True
            },
            'pinecone': {
                'add_vectors': True,
                'search': True,
                'delete_vectors': True,
                'update_metadata': True,
                'export_vectors': False,  # Not supported
                'clear_collection': True
            }
        }
        
        return compatibility_matrix.get(backend, {}).get(operation, False)