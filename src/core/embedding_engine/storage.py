#storage.py
#Embedding storage utilities for saving and loading embeddings.
#Handles different file formats and serialization methods.

import logging
import pickle
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

#Handles saving and loading of embeddings to/from disk
class EmbeddingStorage:
    @staticmethod
    def save_embeddings(save_data: Dict[str, Any], filepath: str):
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
    
    @staticmethod
    def load_embeddings(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
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