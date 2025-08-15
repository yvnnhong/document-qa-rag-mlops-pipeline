#llm_integration/base.py
#Base LLM integration class providing unified interface for multiple backends.

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMIntegration:
    def __init__(self,
                 backend: str = "openai",
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 max_tokens: int = 500,
                 temperature: float = 0.7,
                 system_prompt: Optional[str] = None,
                 **kwargs):
    
        self.backend = backend
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Backend implementation instance
        self.backend_impl = None
        
        # Default system prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Initialize backend
        self._initialize_backend(api_key, **kwargs)
        
        logger.info(f"LLM integration initialized with {backend} backend")
        logger.info(f"Model: {model_name}")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for RAG responses."""
        return """You are a helpful AI assistant that answers questions based on provided context. 

Instructions:
- Answer questions using ONLY the information provided in the context
- If the context doesn't contain enough information to answer the question, say so clearly
- Be accurate, concise, and helpful
- Cite specific details from the context when relevant
- If asked about something not in the context, politely explain that you don't have that information
- Format your response clearly and naturally"""
    
    def _initialize_backend(self, api_key: Optional[str], **kwargs):
        """Initialize the selected LLM backend."""
        if self.backend == "openai":
            from .openai_backend import OpenAIBackend
            self.backend_impl = OpenAIBackend(
                model_name=self.model_name,
                api_key=api_key,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **kwargs
            )
        elif self.backend == "huggingface":
            from .huggingface_backend import HuggingFaceBackend
            self.backend_impl = HuggingFaceBackend(
                model_name=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **kwargs
            )
        elif self.backend == "local":
            logger.warning("Local model backend not yet implemented")
            raise NotImplementedError("Local model backend not yet implemented")
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def prepare_context(self, 
                       retrieved_chunks: List[Dict[str, Any]], 
                       max_context_length: int = 2000) -> str:

        from .prompt_engineering import PromptEngineer
        return PromptEngineer.prepare_context(retrieved_chunks, max_context_length)
    
    def create_prompt(self, 
                     question: str, 
                     context: str, 
                     include_instructions: bool = True) -> str:
        from .prompt_engineering import PromptEngineer
        return PromptEngineer.create_prompt(question, context, include_instructions)
    
    def generate_response(self, 
                         question: str,
                         retrieved_chunks: List[Dict[str, Any]],
                         include_sources: bool = True,
                         **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        # Prepare context
        context = self.prepare_context(retrieved_chunks)
        
        # Create prompt
        prompt = self.create_prompt(question, context)
        
        # Generate response based on backend
        try:
            response_text = self.backend_impl.generate_response(prompt, self.system_prompt, **kwargs)
            
            # Clean up response
            from .response_processing import ResponseProcessor
            cleaned_response = ResponseProcessor.clean_response(response_text, question)
            
            # Prepare result
            result = {
                'answer': cleaned_response,
                'question': question,
                'context': context,
                'chunks_used': len(retrieved_chunks),
                'generation_time': time.time() - start_time,
                'model': self.model_name,
                'backend': self.backend,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add source citations if requested
            if include_sources:
                result['sources'] = ResponseProcessor.format_sources(retrieved_chunks)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return {
                'answer': f"I apologize, but I encountered an error while generating a response: {str(e)}",
                'question': question,
                'context': context,
                'error': str(e),
                'generation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'backend': self.backend,
            'model_name': self.model_name,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'system_prompt': self.system_prompt[:100] + '...' if len(self.system_prompt) > 100 else self.system_prompt
        }