#llm_integration/openai_backend.py
#OpenAI backend for LLM integration.
#Handles OpenAI API communication and response generation.

import os
import logging
from typing import Optional

# OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install with: pip install openai")

# Configure logging
logger = logging.getLogger(__name__)


class OpenAIBackend:
    
    def __init__(self, model_name: str, api_key: Optional[str], max_tokens: int, 
                 temperature: float, **kwargs):
    
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs
        
        # Initialize client
        self.client = None
        self._initialize_client(api_key)
    
    def _initialize_client(self, api_key: Optional[str]):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with: pip install openai")
        
        # Get API key
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        try:
            # Initialize OpenAI client
            self.client = openai.OpenAI(api_key=api_key)
            
            # Test the connection
            self.client.models.list()
            logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    
    def generate_response(self, prompt: str, system_prompt: str, **kwargs) -> str:
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                top_p=kwargs.get('top_p', 1.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                presence_penalty=kwargs.get('presence_penalty', 0.0)
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise