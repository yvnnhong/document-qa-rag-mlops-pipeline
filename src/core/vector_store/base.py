#this file, base.py, creates a singular interface that works with multiple 
#backends (chromadb, pinecone)
import logging #for debug/info msgs
import uuid #to generate unique IDs for vectors 
from typing import List, Dict, Any, Optional #for type hints 
import numpy as np #used for handling embedding arrays 
from datetime import datetime #track timestamps for metadata 
from pathlib import Path #file + directory operations 
import hashlib #create text hashes 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    #Handles vector storage, retrieval, and similarity search.
    def __init__(self, 
                 backend: str = "chromadb", #default to chromadb bcuz it's free + no internet connection required
                 collection_name: str = "document_chunks",
                 persist_directory: str = "./vector_db", #where to save chromadb files
                 embedding_dimension: int = 384,
                 distance_metric: str = "cosine",
                 **kwargs):
        
        self.backend = backend
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric
        
        # Backend implementation instance 
        # (initialize backend placeholder that will 
        # hold actual chromadb/pinecone implementation)
        self.backend_impl = None
        
        # Create persist directory -- create folder for storing chromadb files if it doesnt
        #already exist 
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize the selected backend
        self._initialize_backend(**kwargs)
        
        logger.info(f"Vector store initialized with {backend} backend")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Embedding dimension: {embedding_dimension}")
    
    def _initialize_backend(self, **kwargs):
        #Initialize the selected vector database backend
        if self.backend == "chromadb":
            from .chromadb_backend import ChromaDBBackend
            self.backend_impl = ChromaDBBackend(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                distance_metric=self.distance_metric,
                **kwargs
            )
        elif self.backend == "pinecone":
            from .pinecone_backend import PineconeBackend
            self.backend_impl = PineconeBackend(
                collection_name=self.collection_name,
                embedding_dimension=self.embedding_dimension,
                distance_metric=self.distance_metric,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def add_vectors(self, 
                   embeddings: np.ndarray,
                   texts: List[str],
                   metadata: Optional[List[Dict[str, Any]]] = None,
                   ids: Optional[List[str]] = None) -> List[str]:
    
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings must match number of texts")
        
        # Generate IDs if not provided
        if ids is None:
            ids = []
            for i in range(len(embeddings)):
                new_id = str(uuid.uuid4())
                ids.append(new_id)
        
        # Prepare metadata
        if metadata is None:
            metadata = [{}] * len(embeddings)
        
        # Add timestamps and text to metadata
        current_time = datetime.now().isoformat()
        for i, meta in enumerate(metadata):
            meta.update({
                'text': texts[i],
                'created_at': current_time,
                'text_hash': hashlib.md5(texts[i].encode()).hexdigest()
            })
        
        # Delegate to backend implementation
        self.backend_impl.add_vectors(embeddings, ids, metadata)
        
        logger.info(f"Added {len(embeddings)} vectors to {self.backend}")
        return ids
    
    #note: include_distances means whether or not to include similarity scores
    def search(self, 
              query_embedding: np.ndarray,
              k: int = 5,
              filter_dict: Optional[Dict[str, Any]] = None,
              include_distances: bool = True) -> List[Dict[str, Any]]:
        
        #reshape to 2d array if 1d
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        return self.backend_impl.search(query_embedding, k, filter_dict, include_distances)
    
    def delete_vectors(self, ids: List[str]) -> bool:
        return self.backend_impl.delete_vectors(ids)
    
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        return self.backend_impl.update_metadata(id, metadata)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        #Get statistics about the collection
        stats = self.backend_impl.get_collection_stats()
        stats.update({
            'backend': self.backend,
            'collection_name': self.collection_name,
            'embedding_dimension': self.embedding_dimension
        })
        return stats
    
    def clear_collection(self) -> bool:
        #Clear all vectors from the collection
        return self.backend_impl.clear_collection()
    
    def export_vectors(self, output_path: str) -> bool:
        #Export all vectors and metadata to a file
        return self.backend_impl.export_vectors(output_path)