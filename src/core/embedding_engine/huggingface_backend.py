#huggingface_backend.py
#huggingFace backend for embedding generation.
#Handles HuggingFace transformers library integration.

import logging
import numpy as np
from typing import List
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf

# Configure logging
logger = logging.getLogger(__name__)

#Handles HuggingFace transformers specific operations
class HuggingFaceBackend:
    def __init__(self, model_name: str, cache_dir: Path, batch_size: int, max_length: int):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Model and tokenizer placeholders
        self.model = None
        self.tokenizer = None
        self.embedding_dim = None
        
        # Load model
        self._load_model()
    
    #Load HuggingFace transformer model with TensorFlow backend
    def _load_model(self):
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
            self._fallback_to_sentence_transformers()
        except Exception as e:
            logger.warning(f"Could not load HuggingFace model: {str(e)}")
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
        
        # Use HuggingFace transformers
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
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim