#pipeline.py
#Main document processing pipeline.
#Orchestrates the complete document processing workflow.

import logging
from typing import Dict, Any

# Download required NLTK data
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Import processing components
from .extractors import FileExtractor
from .cleaners import TextCleaner
from .chunkers import TextChunker
from .metadata import MetadataExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.text_cleaner = TextCleaner()
        self.text_chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.metadata_extractor = MetadataExtractor()
        
        # Initialize stopwords for potential future use
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words('english'))
        
        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def extract_text_from_file(self, file_path: str) -> str:
        return FileExtractor.extract_text_from_file(file_path)
    
    def clean_text(self, text: str) -> str:
        return self.text_cleaner.clean_text(text)
    
    def chunk_text(self, text: str, method: str = "sentence_aware") -> list:
        return self.text_chunker.chunk_text(text, method=method)
    
    def extract_metadata(self, file_path: str, text: str = None, chunks: list = None, 
                        chunking_method: str = None) -> Dict[str, Any]:
        return self.metadata_extractor.extract_all_metadata(
            file_path=file_path,
            text=text,
            chunks=chunks,
            chunking_method=chunking_method
        )
    
    def process_document(self, file_path: str, chunking_method: str = "sentence_aware") -> Dict[str, Any]:
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Step 1: Extract text
            logger.debug("Step 1: Extracting text from file")
            raw_text = self.extract_text_from_file(file_path)
            
            if not raw_text:
                logger.warning(f"No text extracted from {file_path}")
                return self._create_empty_result(file_path, chunking_method)
            
            # Step 2: Clean text
            logger.debug("Step 2: Cleaning text")
            cleaned_text = self.clean_text(raw_text)
            
            # Step 3: Chunk text
            logger.debug(f"Step 3: Chunking text using {chunking_method} method")
            chunks = self.chunk_text(cleaned_text, method=chunking_method)
            
            # Step 4: Extract metadata
            logger.debug("Step 4: Extracting metadata")
            metadata = self.extract_metadata(
                file_path=file_path,
                text=cleaned_text,
                chunks=chunks,
                chunking_method=chunking_method
            )
            
            # Compile results
            result = {
                'metadata': metadata,
                'raw_text': raw_text,
                'cleaned_text': cleaned_text,
                'chunks': chunks,
                'processing_stats': metadata.get('processing_stats', {})
            }
            
            logger.info(f"Document processed successfully: {len(chunks)} chunks created")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def _create_empty_result(self, file_path: str, chunking_method: str) -> Dict[str, Any]:
        """Create empty result structure for failed processing."""
        metadata = self.extract_metadata(file_path, "", [], chunking_method)
        
        return {
            'metadata': metadata,
            'raw_text': "",
            'cleaned_text': "",
            'chunks': [],
            'processing_stats': {
                'total_chars': 0,
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'chunking_method': chunking_method
            }
        }
    
    def get_supported_formats(self) -> list:
        return ['.pdf', '.txt', '.md']
    
    def validate_file(self, file_path: str) -> bool:
        from pathlib import Path
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False
            
        if file_path.suffix.lower() not in self.get_supported_formats():
            return False
            
        return True


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