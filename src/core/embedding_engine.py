"""
Embedding engine for converting text into vector representations.
Supports multiple embedding models using TensorFlow and sentence-transformers.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import pickle
from pathlib import Path

# Core ML libraries
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import torch

# Data processing
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Configuration and utilities
import yaml
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class EmbeddingEngine:
    """
    Advanced embedding engine supporting multiple models and backends.
    Handles text vectorization for semantic search and retrieval.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 backend: str = "sentence-transformers",
                 cache_dir: str = "./models",
                 batch_size: int = 32,
                 max_length: int = 512):
        """
        Initialize embedding engine.
        
        Args:
            model_name: Name/path of embedding model
            backend: Backend to use ("sentence-transformers", "tensorflow", "huggingface")
            cache_dir: Directory to cache models
            batch_size: Batch size for processing
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model and tokenizer placeholders
        self.model = None
        self.tokenizer = None
        self.embedding_dim = None
        
        # Load model
        self._load_model()
        
        logger.info(f"Embedding engine initialized with {backend} backend")
        logger.info(f"Model: {model_name}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _load_model(self):
        """Load embedding model based on specified backend."""
        try:
            if self.backend == "sentence-transformers":
                self._load_sentence_transformer()
            elif self.backend == "tensorflow":
                self._load_tensorflow_model()
            elif self.backend == "huggingface":
                self._load_huggingface_model()
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to sentence-transformers
            if self.backend != "sentence-transformers":
                logger.info("Falling back to sentence-transformers backend")
                self.backend = "sentence-transformers"
                self._load_sentence_transformer()
            else:
                raise
    
    def _load_sentence_transformer(self):
        """Load sentence transformer model."""
        logger.info(f"Loading sentence transformer: {self.model_name}")
        
        self.model = SentenceTransformer(
            self.model_name,
            cache_folder=str(self.cache_dir)
        )
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Configure device
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            logger.info("Using CUDA for sentence transformers")
        else:
            logger.info("Using CPU for sentence transformers")
    
    def _load_tensorflow_model(self):
        """Load TensorFlow-based embedding model."""
        logger.info(f"Loading TensorFlow model: {self.model_name}")
        
        try:
            # Try loading as TensorFlow Hub model
            import tensorflow_hub as hub
            self.model = hub.load(self.model_name)
            
            # Test embedding dimension
            test_embedding = self.model(["test text"])
            self.embedding_dim = test_embedding.shape[-1]
            
        except ImportError:
            logger.warning("TensorFlow Hub not available, using sentence-transformers with TF backend")
            # Use sentence-transformers but specify TensorFlow backend
            os.environ['SENTENCE_TRANSFORMERS_BACKEND'] = 'tf'
            self._load_sentence_transformer()
        except Exception as e:
            logger.warning(f"Could not load TF model: {str(e)}, falling back to sentence-transformers")
            self._load_sentence_transformer()
    
    def _load_huggingface_model(self):
        """Load HuggingFace transformer model with TensorFlow backend."""
        try:
            from transformers import TFAutoModel, AutoTokenizer
            
            logger.info(f"Loading HuggingFace model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir)
            )
            
            self.model = TFAutoModel.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                from_tf=True
            )
            
            # Get embedding dimension
            config = self.model.config
            self.embedding_dim = config.hidden_size
            
        except ImportError:
            logger.warning("Transformers library not properly configured, using sentence-transformers")
            self._load_sentence_transformer()
        except Exception as e:
            logger.warning(f"Could not load HuggingFace model: {str(e)}")
            self._load_sentence_transformer()
    
    def encode_texts(self, 
                    texts: Union[str, List[str]], 
                    normalize_embeddings: bool = True,
                    show_progress: bool = True) -> np.ndarray:
        """
        Convert texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize_embeddings: Whether to normalize embeddings
            show_progress: Show progress bar for large batches
            
        Returns:
            Numpy array of embeddings
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        logger.info(f"Encoding {len(texts)} texts")
        
        try:
            if self.backend == "sentence-transformers":
                embeddings = self._encode_sentence_transformers(texts, show_progress)
            elif self.backend == "tensorflow":
                embeddings = self._encode_tensorflow(texts, show_progress)
            elif self.backend == "huggingface":
                embeddings = self._encode_huggingface(texts, show_progress)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
            
            # Normalize embeddings if requested
            if normalize_embeddings:
                embeddings = normalize(embeddings, norm='l2')
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise
    
    def _encode_sentence_transformers(self, texts: List[str], show_progress: bool) -> np.ndarray:
        """Encode using sentence transformers."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False  # We handle normalization separately
        )
        return embeddings
    
    def _encode_tensorflow(self, texts: List[str], show_progress: bool) -> np.ndarray:
        """Encode using TensorFlow model."""
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), 
                     disable=not show_progress, 
                     desc="Encoding"):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                batch_embeddings = self.model(batch_texts)
                
                # Handle different TF model outputs
                if hasattr(batch_embeddings, 'numpy'):
                    batch_embeddings = batch_embeddings.numpy()
                elif isinstance(batch_embeddings, dict):
                    # Some models return dict with pooler_output or last_hidden_state
                    if 'pooler_output' in batch_embeddings:
                        batch_embeddings = batch_embeddings['pooler_output'].numpy()
                    else:
                        batch_embeddings = batch_embeddings['last_hidden_state'].numpy()
                        # Mean pooling for sequence models
                        batch_embeddings = np.mean(batch_embeddings, axis=1)
                
                embeddings.append(batch_embeddings)
                
            except Exception as e:
                logger.warning(f"Error processing batch {i}: {str(e)}")
                # Use zero embeddings as fallback
                fallback_embeddings = np.zeros((len(batch_texts), self.embedding_dim))
                embeddings.append(fallback_embeddings)
        
        return np.vstack(embeddings)
    
    def _encode_huggingface(self, texts: List[str], show_progress: bool) -> np.ndarray:
        """Encode using HuggingFace transformers."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), 
                     disable=not show_progress, 
                     desc="Encoding"):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='tf'
            )
            
            # Get embeddings
            outputs = self.model(**inputs)
            
            # Pool embeddings (mean pooling)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                batch_embeddings = outputs.pooler_output
            else:
                # Mean pooling over sequence length
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                
                # Expand attention mask for broadcasting
                input_mask_expanded = tf.cast(
                    tf.expand_dims(attention_mask, -1), tf.float32
                )
                
                # Apply mask and mean pool
                masked_embeddings = token_embeddings * input_mask_expanded
                summed = tf.reduce_sum(masked_embeddings, axis=1)
                summed_mask = tf.reduce_sum(input_mask_expanded, axis=1)
                batch_embeddings = summed / tf.maximum(summed_mask, 1e-9)
            
            embeddings.append(batch_embeddings.numpy())
        
        return np.vstack(embeddings)
    
    def compute_similarity(self, 
                          embeddings1: np.ndarray, 
                          embeddings2: np.ndarray,
                          metric: str = "cosine") -> np.ndarray:
        """
        Compute similarity between embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            metric: Similarity metric ("cosine", "dot", "euclidean")
            
        Returns:
            Similarity matrix
        """
        if metric == "cosine":
            return cosine_similarity(embeddings1, embeddings2)
        elif metric == "dot":
            return np.dot(embeddings1, embeddings2.T)
        elif metric == "euclidean":
            from sklearn.metrics.pairwise import euclidean_distances
            return -euclidean_distances(embeddings1, embeddings2)  # Negative for similarity
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def find_most_similar(self, 
                         query_embedding: np.ndarray,
                         candidate_embeddings: np.ndarray,
                         top_k: int = 5,
                         metric: str = "cosine") -> Tuple[np.ndarray, np.ndarray]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query embedding (1D array)
            candidate_embeddings: Candidate embeddings (2D array)
            top_k: Number of results to return
            metric: Similarity metric
            
        Returns:
            Tuple of (indices, similarity_scores)
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute similarities
        similarities = self.compute_similarity(
            query_embedding, 
            candidate_embeddings, 
            metric=metric
        ).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def save_embeddings(self, 
                       embeddings: np.ndarray, 
                       metadata: Dict[str, Any],
                       filepath: str):
        """
        Save embeddings and metadata to disk.
        
        Args:
            embeddings: Embedding array
            metadata: Associated metadata
            filepath: Path to save file
        """
        save_data = {
            'embeddings': embeddings,
            'metadata': metadata,
            'model_name': self.model_name,
            'backend': self.backend,
            'embedding_dim': self.embedding_dim
        }
        
        filepath = Path(filepath)
        
        if filepath.suffix == '.npz':
            # Save as compressed numpy
            np.savez_compressed(filepath, **save_data)
        elif filepath.suffix == '.pkl':
            # Save as pickle
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
        else:
            # Default to numpy
            np.savez_compressed(filepath.with_suffix('.npz'), **save_data)
        
        logger.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load embeddings and metadata from disk.
        
        Args:
            filepath: Path to embedding file
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.npz':
            data = np.load(filepath, allow_pickle=True)
            embeddings = data['embeddings']
            metadata = data['metadata'].item()
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            embeddings = data['embeddings']
            metadata = data['metadata']
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Embeddings loaded from {filepath}")
        return embeddings, metadata
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'backend': self.backend,
            'embedding_dim': self.embedding_dim,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'cache_dir': str(self.cache_dir)
        }
    
    def benchmark_performance(self, 
                            test_texts: List[str],
                            iterations: int = 3) -> Dict[str, float]:
        """
        Benchmark embedding performance.
        
        Args:
            test_texts: List of texts for benchmarking
            iterations: Number of iterations to average
            
        Returns:
            Performance metrics
        """
        import time
        
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            embeddings = self.encode_texts(test_texts, show_progress=False)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        texts_per_second = len(test_texts) / avg_time
        
        return {
            'avg_time_seconds': avg_time,
            'texts_per_second': texts_per_second,
            'total_texts': len(test_texts),
            'embedding_dim': self.embedding_dim,
            'backend': self.backend
        }


def main():
    """Test the embedding engine."""
    
    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing helps computers understand human language.",
        "Vector embeddings capture semantic meaning in numerical form.",
        "Document retrieval systems use similarity search."
    ]
    
    # Initialize embedding engine
    engine = EmbeddingEngine(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        backend="sentence-transformers"
    )
    
    # Test encoding
    print("Testing text encoding...")
    embeddings = engine.encode_texts(test_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Test similarity search
    print("\nTesting similarity search...")
    query = "How do computers process language?"
    query_embedding = engine.encode_texts([query])
    
    indices, scores = engine.find_most_similar(
        query_embedding[0], 
        embeddings, 
        top_k=3
    )
    
    print(f"Query: {query}")
    print("Most similar texts:")
    for i, (idx, score) in enumerate(zip(indices, scores)):
        print(f"{i+1}. Score: {score:.3f} - {test_texts[idx]}")
    
    # Benchmark performance
    print("\nBenchmarking performance...")
    benchmark_results = engine.benchmark_performance(test_texts)
    for key, value in benchmark_results.items():
        print(f"{key}: {value}")
    
    # Test save/load
    print("\nTesting save/load...")
    engine.save_embeddings(embeddings, {"texts": test_texts}, "test_embeddings.npz")
    loaded_embeddings, loaded_metadata = engine.load_embeddings("test_embeddings.npz")
    print(f"Loaded embeddings shape: {loaded_embeddings.shape}")
    print(f"Embeddings match: {np.allclose(embeddings, loaded_embeddings)}")


if __name__ == "__main__":
    main()