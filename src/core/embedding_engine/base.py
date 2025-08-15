#base.py
#Base embedding engine class providing unified interface for multiple backends.

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

# Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingEngine:
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 backend: str = "sentence-transformers",
                 cache_dir: str = "./models",
                 batch_size: int = 32,
                 max_length: int = 512):
        
        self.model_name = model_name
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Backend implementation instance
        self.backend_impl = None
        self.embedding_dim = None
        
        # Load model
        self._initialize_backend()
        
        logger.info(f"Embedding engine initialized with {backend} backend")
        logger.info(f"Model: {model_name}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _initialize_backend(self):
        try:
            if self.backend == "sentence-transformers":
                from .sentence_transformer_backend import SentenceTransformerBackend
                self.backend_impl = SentenceTransformerBackend(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                    batch_size=self.batch_size
                )
            elif self.backend == "tensorflow":
                from .tensorflow_backend import TensorFlowBackend
                self.backend_impl = TensorFlowBackend(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                    batch_size=self.batch_size
                )
            elif self.backend == "huggingface":
                from .huggingface_backend import HuggingFaceBackend
                self.backend_impl = HuggingFaceBackend(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                    batch_size=self.batch_size,
                    max_length=self.max_length
                )
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
            
            # Get embedding dimension from backend
            self.embedding_dim = self.backend_impl.get_embedding_dimension()
                
        except Exception as e:
            logger.error(f"Error loading {self.backend} backend: {str(e)}")
            # Fallback to sentence-transformers
            if self.backend != "sentence-transformers":
                logger.info("Falling back to sentence-transformers backend")
                self.backend = "sentence-transformers"
                self._initialize_backend()
            else:
                raise
    
    def encode_texts(self, 
                    texts: Union[str, List[str]], 
                    normalize_embeddings: bool = True,
                    show_progress: bool = True) -> np.ndarray:
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        logger.info(f"Encoding {len(texts)} texts")
        
        try:
            embeddings = self.backend_impl.encode_texts(texts, show_progress)
            
            # Normalize embeddings if requested
            if normalize_embeddings:
                from sklearn.preprocessing import normalize
                embeddings = normalize(embeddings, norm='l2')
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise
    
    def compute_similarity(self, 
                          embeddings1: np.ndarray, 
                          embeddings2: np.ndarray,
                          metric: str = "cosine") -> np.ndarray:
        from .similarity import SimilarityComputer
        return SimilarityComputer.compute_similarity(embeddings1, embeddings2, metric)
    
    def find_most_similar(self, 
                         query_embedding: np.ndarray,
                         candidate_embeddings: np.ndarray,
                         top_k: int = 5,
                         metric: str = "cosine") -> Tuple[np.ndarray, np.ndarray]:
        from .similarity import SimilarityComputer
        return SimilarityComputer.find_most_similar(
            query_embedding, candidate_embeddings, top_k, metric
        )
    
    def save_embeddings(self, 
                       embeddings: np.ndarray, 
                       metadata: Dict[str, Any],
                       filepath: str):
        from .storage import EmbeddingStorage
        
        save_data = {
            'embeddings': embeddings,
            'metadata': metadata,
            'model_name': self.model_name,
            'backend': self.backend,
            'embedding_dim': self.embedding_dim
        }
        
        EmbeddingStorage.save_embeddings(save_data, filepath)
    
    def load_embeddings(self, filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        from .storage import EmbeddingStorage
        return EmbeddingStorage.load_embeddings(filepath)
    
    def get_model_info(self) -> Dict[str, Any]:
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