#chunkers.py
#Text chunking strategies for document processing.
#Implements multiple approaches for splitting text into semantic units.

import logging
from typing import List, Dict, Any

# NLP libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Try to load spaCy model
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    nlp = None
    logging.warning("spaCy model not available. Semantic chunking will fall back to sentence-aware.")

# Configure logging
logger = logging.getLogger(__name__)

#Handles different text chunking strategies
class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, method: str = "sentence_aware") -> List[Dict[str, Any]]:
        if method == "sentence_aware":
            return self._chunk_sentence_aware(text)
        elif method == "fixed_size":
            return self._chunk_fixed_size(text)
        elif method == "semantic":
            return self._chunk_semantic(text)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
    
    #Chunk text while preserving sentence boundaries
    def _chunk_sentence_aware(self, text: str) -> List[Dict[str, Any]]:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk.strip(),
                    'char_count': len(current_chunk),
                    'sentence_count': len(sent_tokenize(current_chunk)),
                    'word_count': len(word_tokenize(current_chunk))
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence
                chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'char_count': len(current_chunk),
                'sentence_count': len(sent_tokenize(current_chunk)),
                'word_count': len(word_tokenize(current_chunk))
            })
        
        return chunks
    
    def _chunk_fixed_size(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text into fixed-size pieces."""
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            if chunk_text:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'char_count': len(chunk_text),
                    'start_pos': i,
                    'end_pos': i + len(chunk_text)
                })
                chunk_id += 1
        return chunks
    
    #Chunk text based on semantic boundaries using spaCy
    def _chunk_semantic(self, text: str) -> List[Dict[str, Any]]:
        if nlp is None:
            logger.warning("spaCy model not available, falling back to sentence-aware chunking")
            return self._chunk_sentence_aware(text)
        
        doc = nlp(text)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            
            if len(current_chunk) + len(sentence_text) > self.chunk_size and current_chunk:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk.strip(),
                    'char_count': len(current_chunk),
                    'entities': self._extract_entities(nlp(current_chunk))
                })
                
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence_text
                chunk_id += 1
            else:
                current_chunk += " " + sentence_text if current_chunk else sentence_text
        
        if current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'char_count': len(current_chunk),
                'entities': self._extract_entities(nlp(current_chunk))
            })
        
        return chunks
    
    #Get overlap text from end of current chunk
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        if len(text) <= overlap_size:
            return text
        return text[-overlap_size:]
    
    #Extract named entities from spaCy doc
    def _extract_entities(self, doc) -> List[Dict[str, str]]:
        return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]

 #Factory class for different chunking strategies
class ChunkingStrategies:
    @staticmethod
    def get_chunker(strategy: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> TextChunker:
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Validate strategy
        valid_strategies = ["sentence_aware", "fixed_size", "semantic"]
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Valid options: {valid_strategies}")
        return chunker