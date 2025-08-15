#tensorflow_backend.py
#TensorFlow backend for embedding generation.
#Handles TensorFlow Hub and TensorFlow-based models.

import os
import logging
import numpy as np
from typing import List
from pathlib import Path
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

#Handles TensorFlow specific operations for embeddings
class TensorFlowBackend:
    def __init__(self, model_name: str, cache_dir: Path, batch_size: int):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        
        # Model placeholder
        self.model = None
        self.embedding_dim = None
        
        # Load model
        self._load_model()
    
    #Load TensorFlow-based embedding model
    def _load_model(self):
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
            # Fallback to sentence transformers with TF backend
            os.environ['SENTENCE_TRANSFORMERS_BACKEND'] = 'tf'
            self._fallback_to_sentence_transformers()
        except Exception as e:
            logger.warning(f"Could not load TF model: {str(e)}, falling back to sentence-transformers")
            self._fallback_to_sentence_transformers()
    
    def _fallback_to_sentence_transformers(self):
        from .sentence_transformer_backend import SentenceTransformerBackend
        
        # Create sentence transformer backend
        st_backend = SentenceTransformerBackend(
            model_name=self.model_name,
            cache_dir=self.cache_dir,
            batch_size=self.batch_size
        )
        
        # Copy over the model and dimension
        self.model = st_backend.model
        self.embedding_dim = st_backend.embedding_dim
        self._is_fallback = True
    
    def encode_texts(self, texts: List[str], show_progress: bool) -> np.ndarray:
        # If using fallback, delegate to sentence transformers
        if hasattr(self, '_is_fallback') and self._is_fallback:
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=False
            )
        
        # Use TensorFlow Hub model
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
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim