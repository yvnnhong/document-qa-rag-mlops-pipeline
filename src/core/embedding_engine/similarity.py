#similarity.py
#Similarity computation utilities for embeddings.
#Handles different similarity metrics and search operations.

import logging
import numpy as np
from typing import Tuple

# Configure logging
logger = logging.getLogger(__name__)

#Handles similarity computation between embeddings
class SimilarityComputer:
    @staticmethod
    def compute_similarity(embeddings1: np.ndarray, 
                          embeddings2: np.ndarray,
                          metric: str = "cosine") -> np.ndarray:
        if metric == "cosine":
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(embeddings1, embeddings2)
        elif metric == "dot":
            return np.dot(embeddings1, embeddings2.T)
        elif metric == "euclidean":
            from sklearn.metrics.pairwise import euclidean_distances
            return -euclidean_distances(embeddings1, embeddings2)  # Negative for similarity
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    @staticmethod
    def find_most_similar(query_embedding: np.ndarray,
                         candidate_embeddings: np.ndarray,
                         top_k: int = 5,
                         metric: str = "cosine") -> Tuple[np.ndarray, np.ndarray]:
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute similarities
        similarities = SimilarityComputer.compute_similarity(
            query_embedding, 
            candidate_embeddings, 
            metric=metric
        ).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores