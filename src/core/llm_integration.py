"""
LLM integration for RAG system.
Supports multiple LLM backends including OpenAI, Hugging Face, and local models.
Handles context preparation, prompt engineering, and response generation.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import time
from datetime import datetime

# OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install with: pip install openai")

# Hugging Face integration
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers")

# Utilities
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMIntegration:
    """
    LLM integration for generating responses from retrieved context.
    Supports multiple backends and advanced prompt engineering.
    """
    
    def __init__(self,
                 backend: str = "openai",
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 max_tokens: int = 500,
                 temperature: float = 0.7,
                 system_prompt: Optional[str] = None,
                 **kwargs):
        """
        Initialize LLM integration.
        
        Args:
            backend: LLM backend ("openai", "huggingface", "local")
            model_name: Name of the model to use
            api_key: API key for external services
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            system_prompt: System prompt for the model
            **kwargs: Additional backend-specific parameters
        """
        self.backend = backend
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize components
        self.client = None
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
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
            self._initialize_openai(api_key, **kwargs)
        elif self.backend == "huggingface":
            self._initialize_huggingface(**kwargs)
        elif self.backend == "local":
            self._initialize_local_model(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _initialize_openai(self, api_key: Optional[str], **kwargs):
        """Initialize OpenAI client."""
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
    
    def _initialize_huggingface(self, **kwargs):
        """Initialize Hugging Face model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available. Install with: pip install transformers")
        
        try:
            # Default to a good open-source model if not specified
            if self.model_name == "gpt-3.5-turbo":
                self.model_name = "microsoft/DialoGPT-medium"
            
            logger.info(f"Loading Hugging Face model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=self.max_tokens + 512,  # Extra space for context
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Hugging Face model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face model: {str(e)}")
            # Fallback to a smaller model
            logger.info("Falling back to GPT-2")
            try:
                self.model_name = "gpt2"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=min(self.max_tokens + 512, 1024),
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                logger.info("Fallback to GPT-2 successful")
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                raise
    
    def _initialize_local_model(self, **kwargs):
        """Initialize local model (placeholder for future local model support)."""
        logger.warning("Local model backend not yet implemented")
        raise NotImplementedError("Local model backend not yet implemented")
    
    def prepare_context(self, 
                       retrieved_chunks: List[Dict[str, Any]], 
                       max_context_length: int = 2000) -> str:
        """
        Prepare context from retrieved chunks.
        
        Args:
            retrieved_chunks: List of retrieved document chunks
            max_context_length: Maximum length of context in characters
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant context found."
        
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(retrieved_chunks):
            # Extract text and score
            text = chunk.get('text', '').strip()
            score = chunk.get('score', 0)
            
            # Format chunk with score
            chunk_text = f"[Context {i+1}] (Relevance: {score:.3f})\n{text}\n"
            
            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_text) > max_context_length and context_parts:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n".join(context_parts)
    
    def create_prompt(self, 
                     question: str, 
                     context: str, 
                     include_instructions: bool = True) -> str:
        """
        Create a properly formatted prompt for the LLM.
        
        Args:
            question: User's question
            context: Retrieved context
            include_instructions: Whether to include detailed instructions
            
        Returns:
            Formatted prompt
        """
        instructions = """
Based on the context provided below, please answer the following question. 
Use only the information from the context. If the context doesn't contain 
enough information to answer the question, please say so clearly.

""" if include_instructions else ""
        
        prompt = f"""{instructions}Context:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def generate_response(self, 
                         question: str,
                         retrieved_chunks: List[Dict[str, Any]],
                         include_sources: bool = True,
                         **kwargs) -> Dict[str, Any]:
        """
        Generate response using retrieved context.
        
        Args:
            question: User's question
            retrieved_chunks: Retrieved document chunks
            include_sources: Whether to include source citations
            **kwargs: Additional generation parameters
            
        Returns:
            Response dictionary with answer, context, and metadata
        """
        start_time = time.time()
        
        # Prepare context
        context = self.prepare_context(retrieved_chunks)
        
        # Create prompt
        prompt = self.create_prompt(question, context)
        
        # Generate response based on backend
        try:
            if self.backend == "openai":
                response_text = self._generate_openai(prompt, **kwargs)
            elif self.backend == "huggingface":
                response_text = self._generate_huggingface(prompt, **kwargs)
            else:
                response_text = "Backend not implemented"
            
            # Clean up response
            cleaned_response = self._clean_response(response_text, question)
            
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
                result['sources'] = self._format_sources(retrieved_chunks)
            
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
    
    def _generate_openai(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": self.system_prompt},
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
    
    def _generate_huggingface(self, prompt: str, **kwargs) -> str:
        """Generate response using Hugging Face model."""
        try:
            # Add system prompt to the beginning
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            
            # Generate response
            outputs = self.pipeline(
                full_prompt,
                max_length=len(full_prompt.split()) + kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the prompt from the response
            response = generated_text[len(full_prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Hugging Face generation failed: {str(e)}")
            raise
    
    def _clean_response(self, response: str, question: str) -> str:
        """Clean and format the generated response."""
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
    
    def _format_sources(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source citations from retrieved chunks."""
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'backend': self.backend,
            'model_name': self.model_name,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'system_prompt': self.system_prompt[:100] + '...' if len(self.system_prompt) > 100 else self.system_prompt
        }


def main():
    """Test the LLM integration."""
    
    # Test with mock retrieved chunks
    mock_chunks = [
        {
            'text': 'Unlimited access to fresh, good-quality hay, such as timothy, orchard, or meadow hay, is essential for a rabbits digestive health and dental care. Hay should make up about 80% of their diet.',
            'score': 0.85,
            'metadata': {'document': 'rabbit_care_guide', 'section': 'diet'}
        },
        {
            'text': 'Offer a variety of fresh, leafy greens daily, such as romaine lettuce, kale, or cilantro, at a rate of about 1 cup per 2 pounds of body weight.',
            'score': 0.72,
            'metadata': {'document': 'rabbit_care_guide', 'section': 'vegetables'}
        }
    ]
    
    print("Testing LLM Integration")
    print("=" * 40)
    
    # Test with different backends
    backends_to_test = []
    
    # Check which backends are available
    if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
        backends_to_test.append(('openai', 'gpt-3.5-turbo'))
    
    if TRANSFORMERS_AVAILABLE:
        backends_to_test.append(('huggingface', 'gpt2'))
    
    if not backends_to_test:
        print("No LLM backends available for testing")
        print("For OpenAI: Set OPENAI_API_KEY environment variable")
        print("For Hugging Face: Install transformers library")
        return
    
    test_question = "How much hay should I feed my rabbit daily?"
    
    for backend, model in backends_to_test:
        print(f"\nTesting {backend} backend with {model}")
        print("-" * 50)
        
        try:
            # Initialize LLM
            llm = LLMIntegration(
                backend=backend,
                model_name=model,
                max_tokens=200,
                temperature=0.7
            )
            
            # Generate response
            result = llm.generate_response(test_question, mock_chunks)
            
            # Display results
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"Generation time: {result['generation_time']:.2f}s")
            print(f"Chunks used: {result['chunks_used']}")
            
            if 'sources' in result:
                print("\nSources:")
                for source in result['sources']:
                    print(f"  {source['index']}. Score: {source['score']:.3f}")
                    print(f"     {source['text_preview']}")
            
        except Exception as e:
            print(f"Failed to test {backend}: {str(e)}")
    
    print("\nLLM integration testing completed!")


if __name__ == "__main__":
    main()