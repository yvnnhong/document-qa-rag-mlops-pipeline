#llm_integration/response_processing.py
#Response processing utilities for LLM integration.
#Handles response cleaning, formatting, and source citation.

import re
import logging
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)


class ResponseProcessor:
    
    @staticmethod
    def clean_response(response: str, question: str) -> str:
        # Remove common artifacts
        response = response.strip()
        
        # Remove repeated question
        if response.startswith(question):
            response = response[len(question):].strip()
        
        # Remove "Answer:" prefix if present
        if response.lower().startswith("answer:"):
            response = response[7:].strip()
        
        # Remove excessive newlines
        response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)
        
        # Ensure proper sentence endings
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response
    
    @staticmethod
    def format_sources(retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        
        for i, chunk in enumerate(retrieved_chunks):
            source = {
                'index': i + 1,
                'text_preview': chunk.get('text', '')[:100] + '...',
                'score': chunk.get('score', 0),
                'metadata': chunk.get('metadata', {})
            }
            sources.append(source)
        
        return sources
    
    @staticmethod
    def extract_confidence_score(response: str) -> float:
        # Simple heuristic based on uncertainty indicators
        uncertainty_phrases = [
            "i'm not sure", "i don't know", "unclear", "uncertain",
            "might be", "could be", "possibly", "perhaps", "maybe",
            "i cannot", "not enough information", "insufficient context"
        ]
        
        confidence_phrases = [
            "according to", "specifically", "clearly states", "explicitly",
            "the document shows", "the text indicates", "based on the context"
        ]
        
        response_lower = response.lower()
        
        # Count uncertainty indicators
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in response_lower)
        
        # Count confidence indicators
        confidence_count = sum(1 for phrase in confidence_phrases if phrase in response_lower)
        
        # Simple scoring mechanism
        if uncertainty_count > confidence_count:
            return max(0.3, 1.0 - (uncertainty_count * 0.2))
        elif confidence_count > 0:
            return min(0.9, 0.6 + (confidence_count * 0.1))
        else:
            return 0.7  # Default moderate confidence
    
    @staticmethod
    def validate_response_quality(response: str, context: str) -> Dict[str, Any]:
        metrics = {
            'length_appropriate': 20 <= len(response.split()) <= 200,
            'context_referenced': any(word in response.lower() for word in context.lower().split()[:20]),
            'proper_ending': response.endswith(('.', '!', '?')),
            'no_hallucination_indicators': not any(phrase in response.lower() for phrase in [
                "i think", "i believe", "in my opinion", "generally speaking"
            ]),
            'confidence_score': ResponseProcessor.extract_confidence_score(response)
        }
        
        # Overall quality score
        quality_score = sum([
            metrics['length_appropriate'],
            metrics['context_referenced'],
            metrics['proper_ending'],
            metrics['no_hallucination_indicators']
        ]) / 4.0
        
        metrics['overall_quality'] = quality_score
        
        return metrics