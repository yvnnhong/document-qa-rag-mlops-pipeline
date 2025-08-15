#document_processor.py: 
#takes documents (PDFs, text files) and converts them into small, searchable
#chunks for the RAG system 

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Document processing libraries
import PyPDF2
import pdfplumber
from io import BytesIO

# NLP libraries
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Data processing
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles document ingestion, text extraction, cleaning, and chunking.
    Supports PDF and text files with multiple extraction methods.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.stop_words = set(stopwords.words('english'))
        
    def extract_text_from_pdf(self, file_path: str, method: str = "pdfplumber") -> str:
        """
        Extract text from PDF using specified method.
        
        Args:
            file_path: Path to PDF file
            method: Extraction method ("pdfplumber" or "pypdf2")
            
        Returns:
            Extracted text content
        """
        try:
            if method == "pdfplumber":
                return self._extract_with_pdfplumber(file_path)
            elif method == "pypdf2":
                return self._extract_with_pypdf2(file_path)
            else:
                raise ValueError(f"Unknown extraction method: {method}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            # Try fallback method
            fallback_method = "pypdf2" if method == "pdfplumber" else "pdfplumber"
            logger.info(f"Trying fallback method: {fallback_method}")
            try:
                return self.extract_text_from_pdf(file_path, fallback_method)
            except Exception as fallback_error:
                logger.error(f"Fallback extraction failed: {str(fallback_error)}")
                return ""
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber (better for complex layouts)."""
        text_content = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                        logger.debug(f"Extracted text from page {page_num + 1}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
                    
        return "\n\n".join(text_content)
    
    def _extract_with_pypdf2(self, file_path: str) -> str:
        """Extract text using PyPDF2 (faster, simpler)."""
        text_content = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                        logger.debug(f"Extracted text from page {page_num + 1}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
                    
        return "\n\n".join(text_content)
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from various file formats.
        
        Args:
            file_path: Path to file
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(str(file_path))
        elif file_extension in ['.txt', '.md']:
            return self._extract_from_text_file(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _extract_from_text_file(self, file_path: str) -> str:
        """Extract text from plain text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file: {file_path}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']', '', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
        
        # Remove extra spaces
        text = text.strip()
        
        return text
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from document.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary containing metadata
        """
        file_path = Path(file_path)
        metadata = {
            'filename': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_extension': file_path.suffix.lower(),
            'created_date': file_path.stat().st_ctime,
            'modified_date': file_path.stat().st_mtime,
        }
        
        # PDF-specific metadata
        if file_path.suffix.lower() == '.pdf':
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    metadata.update({
                        'num_pages': len(pdf_reader.pages),
                        'pdf_info': pdf_reader.metadata if pdf_reader.metadata else {}
                    })
            except Exception as e:
                logger.warning(f"Could not extract PDF metadata: {str(e)}")
                
        return metadata
    
    def chunk_text(self, text: str, method: str = "sentence_aware") -> List[Dict[str, Any]]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Text to chunk
            method: Chunking method ("sentence_aware", "fixed_size", or "semantic")
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if method == "sentence_aware":
            return self._chunk_sentence_aware(text)
        elif method == "fixed_size":
            return self._chunk_fixed_size(text)
        elif method == "semantic":
            return self._chunk_semantic(text)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
    
    def _chunk_sentence_aware(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text while preserving sentence boundaries."""
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
    
    def _chunk_semantic(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text based on semantic boundaries using spaCy."""
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
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from end of current chunk."""
        if len(text) <= overlap_size:
            return text
        return text[-overlap_size:]
    
    def _extract_entities(self, doc) -> List[Dict[str, str]]:
        """Extract named entities from spaCy doc."""
        return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
    
    def process_document(self, file_path: str, chunking_method: str = "sentence_aware") -> Dict[str, Any]:
        """
        Complete document processing pipeline.
        
        Args:
            file_path: Path to document
            chunking_method: Method for text chunking
            
        Returns:
            Dictionary with processed document data
        """
        logger.info(f"Processing document: {file_path}")
        
        # Extract text
        raw_text = self.extract_text_from_file(file_path)
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Extract metadata
        metadata = self.extract_metadata(file_path)
        
        # Chunk text
        chunks = self.chunk_text(cleaned_text, method=chunking_method)
        
        # Compile results
        result = {
            'metadata': metadata,
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'chunks': chunks,
            'processing_stats': {
                'total_chars': len(cleaned_text),
                'total_chunks': len(chunks),
                'avg_chunk_size': sum(chunk['char_count'] for chunk in chunks) / len(chunks) if chunks else 0,
                'chunking_method': chunking_method
            }
        }
        
        logger.info(f"Document processed: {len(chunks)} chunks created")
        return result


def main():
    """Test the document processor."""
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    
    # Test with a sample text
    sample_text = """
    This is a sample document for testing the document processor.
    It contains multiple sentences and paragraphs to demonstrate
    the chunking functionality. The processor should handle this
    text appropriately and create meaningful chunks.
    
    This is a second paragraph to test paragraph boundaries.
    The system should maintain context while creating chunks.
    """
    
    # Test chunking
    chunks = processor.chunk_text(sample_text)
    
    print(f"Created {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"Chunk {chunk['chunk_id']}: {chunk['char_count']} chars")
        print(f"Text: {chunk['text'][:100]}...")
        print("---")


if __name__ == "__main__":
    main()