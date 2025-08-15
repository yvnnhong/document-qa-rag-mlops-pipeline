#Text cleaning utilities for document processing.
#Handles text normalization and common PDF extraction issues.

import re
import logging

# Configure logging
logger = logging.getLogger(__name__)


class TextCleaner:
    #Handles text cleaning and normalization operations
    
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']', '', text)
        
        # Fix common PDF extraction issues
        text = TextCleaner._fix_pdf_issues(text)
        
        # Remove extra spaces
        text = text.strip()
        
        return text
    
    @staticmethod
    def _fix_pdf_issues(text: str) -> str:
        # Fix hyphenated words that got split across lines
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Multiple newlines to single
        text = re.sub(r'\n+', '\n', text)
        
        return text
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph breaks)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        import unicodedata
        
        # Normalize to NFC form (canonical composition)
        text = unicodedata.normalize('NFC', text)
        
        # Replace common unicode quotes and dashes
        replacements = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '—': '-',
            '–': '-',
            '…': '...'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text