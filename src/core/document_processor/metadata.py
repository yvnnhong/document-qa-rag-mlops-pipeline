#metadata.py
#Metadata extraction utilities for document processing.
#Handles file-level and document-specific metadata extraction.

import logging
from pathlib import Path
from typing import Dict, Any

# PDF processing for metadata
import PyPDF2

# Configure logging
logger = logging.getLogger(__name__)

#Handles extraction of document metadata and file information
class MetadataExtractor:
    @staticmethod
    def extract_file_metadata(file_path: str) -> Dict[str, Any]:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        
        metadata = {
            'filename': file_path.name,
            'file_size': stat.st_size,
            'file_extension': file_path.suffix.lower(),
            'created_date': stat.st_ctime,
            'modified_date': stat.st_mtime,
            'file_path': str(file_path.absolute())
        }
        
        return metadata
    
    @staticmethod
    def extract_pdf_metadata(file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                pdf_metadata = {
                    'num_pages': len(pdf_reader.pages),
                    'pdf_info': {}
                }
                
                # Extract PDF document info if available
                if pdf_reader.metadata:
                    pdf_info = {}
                    for key, value in pdf_reader.metadata.items():
                        # Clean up the key (remove /prefix)
                        clean_key = key.lstrip('/')
                        pdf_info[clean_key] = value
                    pdf_metadata['pdf_info'] = pdf_info
                
                return pdf_metadata
                
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {str(e)}")
            return {
                'num_pages': 0,
                'pdf_info': {}
            }
    
    @staticmethod
    def extract_text_metadata(text: str) -> Dict[str, Any]:
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'line_count': 0,
                'paragraph_count': 0
            }
        
        # Basic text statistics
        char_count = len(text)
        word_count = len(text.split())
        line_count = len(text.splitlines())
        
        # Count paragraphs (double newlines)
        paragraphs = text.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'line_count': line_count,
            'paragraph_count': paragraph_count
        }
    
    @staticmethod
    def extract_processing_metadata(chunks: list, chunking_method: str) -> Dict[str, Any]:
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'chunking_method': chunking_method
            }
        
        total_chunks = len(chunks)
        total_chars = sum(chunk.get('char_count', 0) for chunk in chunks)
        avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
        
        return {
            'total_chunks': total_chunks,
            'avg_chunk_size': avg_chunk_size,
            'chunking_method': chunking_method,
            'total_processed_chars': total_chars
        }
    
    @staticmethod
    def extract_all_metadata(file_path: str, text: str = None, chunks: list = None, 
                           chunking_method: str = None) -> Dict[str, Any]:
        metadata = {}
        
        # File metadata
        metadata.update(MetadataExtractor.extract_file_metadata(file_path))
        
        # PDF-specific metadata
        if Path(file_path).suffix.lower() == '.pdf':
            pdf_metadata = MetadataExtractor.extract_pdf_metadata(file_path)
            metadata.update(pdf_metadata)
        
        # Text content metadata
        if text is not None:
            text_metadata = MetadataExtractor.extract_text_metadata(text)
            metadata['text_stats'] = text_metadata
        
        # Processing metadata
        if chunks is not None and chunking_method is not None:
            processing_metadata = MetadataExtractor.extract_processing_metadata(chunks, chunking_method)
            metadata['processing_stats'] = processing_metadata
        
        return metadata