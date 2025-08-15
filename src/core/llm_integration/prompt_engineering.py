#llm_integration/prompt_engineering.py
#Prompt engineering utilities for LLM integration.
#Handles context preparation and prompt formatting.

import logging
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)


class PromptEngineer:
    @staticmethod
    def prepare_context(retrieved_chunks: List[Dict[str, Any]], 
                       max_context_length: int = 2000) -> str:
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
    
    @staticmethod
    def create_prompt(question: str, 
                     context: str, 
                     include_instructions: bool = True) -> str:
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
    
    @staticmethod
    def create_system_prompt(domain: str = "general") -> str:
        """
        Create domain-specific system prompts.
        
        Args:
            domain: Domain for specialized prompts ("general", "medical", "legal", "technical")
            
        Returns:
            Domain-specific system prompt
        """
        base_prompt = """You are a helpful AI assistant that answers questions based on provided context."""
        
        domain_instructions = {
            "general": """
Instructions:
- Answer questions using ONLY the information provided in the context
- If the context doesn't contain enough information to answer the question, say so clearly
- Be accurate, concise, and helpful
- Cite specific details from the context when relevant
- If asked about something not in the context, politely explain that you don't have that information
- Format your response clearly and naturally""",
            
            "medical": """
Instructions:
- Answer questions using ONLY the information provided in the medical context
- Always emphasize that this is informational only and not medical advice
- If the context doesn't contain enough information, recommend consulting a healthcare professional
- Be precise with medical terminology when present in the context
- Do not make medical recommendations beyond what's explicitly stated in the context""",
            
            "legal": """
Instructions:
- Answer questions using ONLY the information provided in the legal context
- Always emphasize that this is informational only and not legal advice
- If the context doesn't contain enough information, recommend consulting a legal professional
- Be precise with legal terminology and cite specific sections when available
- Do not provide legal opinions beyond what's explicitly stated in the context""",
            
            "technical": """
Instructions:
- Answer questions using ONLY the information provided in the technical context
- Be precise with technical terminology and specifications
- Include relevant code examples, configurations, or procedures when present in the context
- If the context doesn't contain enough information for implementation, clearly state what's missing
- Structure technical responses with clear steps when applicable"""
        }
        
        return base_prompt + domain_instructions.get(domain, domain_instructions["general"])