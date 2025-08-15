#Pinecone backend implementation for vector storage.
#Handles Pinecone-specific operations for cloud-based vector storage.

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

# Pinecone imports
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone not available. Install with: pip install pinecone-client")
except Exception as e:
    PINECONE_AVAILABLE = False
    logging.warning(f"Pinecone import failed: {str(e)}")

# Configure logging
logger = logging.getLogger(__name__)


class PineconeBackend:
    #uses similarity search similar to ChromaDB backend 
    def __init__(self, 
                 collection_name: str,
                 embedding_dimension: int,
                 distance_metric: str = "cosine",
                 **kwargs):
        
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric
        
        # Pinecone components
        self.index = None
        
        # Initialize Pinecone
        self._initialize_pinecone(**kwargs)
    
    def _initialize_pinecone(self, **kwargs):
        #Initialize Pinecone client and index
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
    
    def add_vectors(self, embeddings: np.ndarray, ids: List[str], metadata: List[Dict]):
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
            
            logger.debug(f"Added {len(embeddings)} vectors to Pinecone")
                
        except Exception as e:
            logger.error(f"Failed to add vectors to Pinecone: {str(e)}")
            raise
    
    def search(self, 
              query_embedding: np.ndarray, 
              k: int, 
              filter_dict: Optional[Dict], 
              include_distances: bool) -> List[Dict]:

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
        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors from Pinecone")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors from Pinecone: {str(e)}")
            return False
    
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        try:
            self.index.update(id=id, metadata=metadata)
            logger.info(f"Updated metadata for vector {id} in Pinecone")
            return True
        except Exception as e:
            logger.error(f"Failed to update metadata in Pinecone: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'index_fullness': stats.get('index_fullness', 0),
                'backend_type': 'pinecone'
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone index stats: {str(e)}")
            return {'total_vectors': 0, 'backend_type': 'pinecone'}
    
    def clear_collection(self) -> bool:
        try:
            self.index.delete(delete_all=True)
            logger.info(f"Cleared all vectors from Pinecone index: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear Pinecone index: {str(e)}")
            return False
    
    def export_vectors(self, output_path: str) -> bool:
        logger.warning("Export not implemented for Pinecone backend - service limitation")
        return False