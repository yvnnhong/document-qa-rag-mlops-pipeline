#ChromaDB backend implementation for vector storage.
#Handles ChromaDB-specific operations for persistent vector storage.

import logging
import json
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

# ChromaDB imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available. Install with: pip install chromadb")

# Configure logging
logger = logging.getLogger(__name__)

class ChromaDBBackend:
    #Provides persistent local vector storage with similarity search.
    def __init__(self, 
                 collection_name: str,
                 persist_directory: Path,
                 distance_metric: str = "cosine",
                 **kwargs):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric
        
        # ChromaDB components
        self.client = None
        self.collection = None
        
        # Initialize ChromaDB
        self._initialize_chromadb(**kwargs)
    
    def _initialize_chromadb(self, **kwargs):
        #Initialize ChromaDB client and collection#
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=None  # We'll provide embeddings manually
                )
                logger.info(f"Connected to existing ChromaDB collection: {self.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,
                    metadata={"hnsw:space": self.distance_metric}
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def add_vectors(self, embeddings: np.ndarray, ids: List[str], metadata: List[Dict]):
        try:
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=[meta['text'] for meta in metadata],
                metadatas=metadata,
                ids=ids
            )
            logger.debug(f"Added {len(embeddings)} vectors to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to add vectors to ChromaDB: {str(e)}")
            raise
    
    def search(self, 
              query_embedding: np.ndarray, 
              k: int, 
              filter_dict: Optional[Dict], 
              include_distances: bool) -> List[Dict]:
        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k,
                where=filter_dict,
                include=['documents', 'metadatas', 'distances'] if include_distances else ['documents', 'metadatas']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i]
                }
                
                if include_distances and 'distances' in results:
                    # Convert distance to similarity score (higher is better)
                    distance = results['distances'][0][i]
                    if self.distance_metric == "cosine":
                        similarity = 1 - distance
                    else:
                        similarity = 1 / (1 + distance)  # Convert distance to similarity
                    result['score'] = similarity
                
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {str(e)}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors from ChromaDB: {str(e)}")
            return False
    
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        logger.warning("ChromaDB doesn't support metadata updates without re-adding")
        return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {
                'total_vectors': count,
                'backend_type': 'chromadb'
            }
        except Exception as e:
            logger.error(f"Failed to get ChromaDB collection stats: {str(e)}")
            return {'total_vectors': 0, 'backend_type': 'chromadb'}
    
    def clear_collection(self) -> bool:
        try:
            # Reset collection by deleting and recreating
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": self.distance_metric}
            )
            logger.info(f"Cleared all vectors from ChromaDB collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB collection: {str(e)}")
            return False
    
    def export_vectors(self, output_path: str) -> bool:
        try:
            # Get all data from ChromaDB
            results = self.collection.get(include=['documents', 'metadatas', 'embeddings'])
            
            embeddings_list = []
            for emb in results['embeddings']:
                if hasattr(emb, 'tolist'):
                    embeddings_list.append(emb.tolist())
                else:
                    embeddings_list.append(emb)

            export_data = {
                'backend': 'chromadb',
                'collection_name': self.collection_name,
                'vectors': {
                    'ids': results['ids'],
                    'embeddings': embeddings_list,
                    'documents': results['documents'],
                    'metadatas': results['metadatas']
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(results['ids'])} vectors from ChromaDB to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export ChromaDB vectors: {str(e)}")
            return False