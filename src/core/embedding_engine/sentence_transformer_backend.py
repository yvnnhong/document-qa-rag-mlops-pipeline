"""
Sentence Transformer backend for embedding generation.
Handles sentence-transformers library integration.
"""

import logging
import numpy as np
from typing import List
from pathlib import Path

# Core ML libraries
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logger = logging.getLogger(__name__)

#Handles sentence-transformers specific operations
class SentenceTransformerBackend:
    def __init__(self, model_name: str, cache_dir: Path, batch_size: int):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        
        # Model placeholder
        self.model = None
        self.embedding_dim = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
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
    
    def encode_texts(self, texts: List[str], show_progress: bool) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False  # We handle normalization separately
        )
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim