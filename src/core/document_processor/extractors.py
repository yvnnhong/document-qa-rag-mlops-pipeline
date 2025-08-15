#extractors.py
#Text extraction modules for different file formats.
#Handles PDF and text file extraction with multiple methods.

import logging
from pathlib import Path
from typing import List

# Document processing libraries
import PyPDF2
import pdfplumber

# Configure logging
logger = logging.getLogger(__name__)


class PDFExtractor:
    #Handles PDF text extraction with multiple methods
    
    @staticmethod
    def extract_with_pdfplumber(file_path: str) -> str:
        #Extract text using pdfplumber (better for complex layouts)
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
    
    @staticmethod
    def extract_with_pypdf2(file_path: str) -> str:
        #Extract text using PyPDF2 (faster, simpler)
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
    
    @staticmethod
    def extract_text(file_path: str, method: str = "pdfplumber") -> str:
        #Extract text from PDF using specified method with fallback.
        try:
            if method == "pdfplumber":
                return PDFExtractor.extract_with_pdfplumber(file_path)
            elif method == "pypdf2":
                return PDFExtractor.extract_with_pypdf2(file_path)
            else:
                raise ValueError(f"Unknown extraction method: {method}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            # Try fallback method
            fallback_method = "pypdf2" if method == "pdfplumber" else "pdfplumber"
            logger.info(f"Trying fallback method: {fallback_method}")
            try:
                return PDFExtractor.extract_text(file_path, fallback_method)
            except Exception as fallback_error:
                logger.error(f"Fallback extraction failed: {str(fallback_error)}")
                return ""


class TextExtractor:
    #Handles plain text file extraction with encoding detection#
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """Extract text from plain text files with encoding fallback."""
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


class FileExtractor:
    #Main file extraction interface that routes to appropriate extractor

    @staticmethod
    def extract_text_from_file(file_path: str) -> str:
        #Extract text from various file formats.
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        file_extension = file_path.suffix.lower()
        if file_extension == '.pdf':
            return PDFExtractor.extract_text(str(file_path))
        elif file_extension in ['.txt', '.md']:
            return TextExtractor.extract_text(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")