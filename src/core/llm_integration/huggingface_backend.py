#llm_integration/huggingface_backend.py
#HuggingFace backend for LLM integration.
#Handles HuggingFace transformers and local model execution

import logging

# Hugging Face integration
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers")

# Configure logging
logger = logging.getLogger(__name__)


class HuggingFaceBackend:
    
    def __init__(self, model_name: str, max_tokens: int, temperature: float, **kwargs):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
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
    
    def generate_response(self, prompt: str, system_prompt: str, **kwargs) -> str:
        try:
            # Add system prompt to the beginning
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
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