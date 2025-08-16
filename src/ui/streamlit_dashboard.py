#Streamlit dashboard for the RAG Document Q&A System.
#Features document upload, real-time chat, and comprehensive system monitoring.

import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime
import json

# Add src to path for imports
current_dir = Path(__file__).parent  # src/ui/
src_dir = current_dir.parent         # src/
sys.path.append(str(src_dir))

# Core imports
try:
    from core.document_processor import DocumentProcessor
    from core.embedding_engine import EmbeddingEngine
    from core.vector_store import VectorStore
    from core.llm_integration import LLMIntegration
except ImportError as e:
    st.error(f"Import error: {e}. Make sure you're running from the project root.")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon=":3",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #f0f8ff;
        border-left-color: #1f77b4;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border-left-color: #28a745;
    }
    .source-citation {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "embedding_engine" not in st.session_state:
        st.session_state.embedding_engine = None
    
    if "llm" not in st.session_state:
        st.session_state.llm = None
    
    if "system_ready" not in st.session_state:
        st.session_state.system_ready = False

def initialize_components():
    """Initialize ML components if not already done."""
    if not st.session_state.system_ready:
        with st.spinner("Initializing AI system components..."):
            try:
                # Initialize embedding engine
                if st.session_state.embedding_engine is None:
                    st.session_state.embedding_engine = EmbeddingEngine(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        backend="sentence-transformers"
                    )
                
                # Initialize vector store
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = VectorStore(
                        backend="chromadb",
                        collection_name="streamlit_documents",
                        persist_directory="./streamlit_vector_db"
                    )
                
                # Initialize LLM
                if st.session_state.llm is None:
                    # Check if OpenAI API key is available
                    openai_key = os.getenv('OPENAI_API_KEY')
                    if openai_key:
                        try:
                            st.session_state.llm = LLMIntegration(
                                backend="openai",
                                model_name="gpt-3.5-turbo",
                                api_key=openai_key,
                                max_tokens=300
                            )
                        except Exception:
                            # Fallback to Hugging Face
                            st.session_state.llm = LLMIntegration(
                                backend="huggingface",
                                model_name="gpt2",
                                max_tokens=150
                            )
                    else:
                        # Use Hugging Face
                        st.session_state.llm = LLMIntegration(
                            backend="huggingface",
                            model_name="gpt2",
                            max_tokens=150
                        )
                
                st.session_state.system_ready = True
                st.success("AI system initialized successfully")
                
            except Exception as e:
                st.error(f"Failed to initialize components: {str(e)}")
                st.stop()

def process_uploaded_file(uploaded_file):
    """Process uploaded document and add to vector store."""
    if uploaded_file is None:
        return False
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Process document
            processor = DocumentProcessor()
            result = processor.process_document(tmp_file_path)
            
            # Generate embeddings
            chunk_texts = [chunk['text'] for chunk in result['chunks']]
            embeddings = st.session_state.embedding_engine.encode_texts(chunk_texts)
            
            # Add to vector store
            metadata = [
                {
                    "chunk_id": i,
                    "document": uploaded_file.name,
                    "upload_time": datetime.now().isoformat(),
                    "total_chunks": len(chunk_texts)
                }
                for i in range(len(chunk_texts))
            ]
            
            vector_ids = st.session_state.vector_store.add_vectors(
                embeddings, 
                chunk_texts, 
                metadata
            )
            
            # Store document info
            doc_info = {
                "name": uploaded_file.name,
                "chunks": len(result['chunks']),
                "upload_time": datetime.now().isoformat(),
                "vector_ids": vector_ids,
                "file_size": len(uploaded_file.getvalue())
            }
            
            st.session_state.documents_processed.append(doc_info)
            
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        st.success(f"Successfully processed {uploaded_file.name} ({len(result['chunks'])} chunks)")
        return True
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return False

def generate_response(question):
    """Generate response using RAG pipeline."""
    if not st.session_state.system_ready:
        return "System not ready. Please initialize components first."
    
    if not st.session_state.documents_processed:
        return "Please upload and process a document first."
    
    try:
        # Search for relevant chunks
        query_embedding = st.session_state.embedding_engine.encode_texts([question])
        search_results = st.session_state.vector_store.search(
            query_embedding[0], 
            k=3, 
            include_distances=True
        )
        
        if not search_results:
            return "No relevant information found in the uploaded documents."
        
        # Generate response using LLM
        rag_response = st.session_state.llm.generate_response(
            question, 
            search_results,
            include_sources=True
        )
        
        return rag_response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

def display_chat_message(message_type, content, sources=None):
    """Display a formatted chat message."""
    if message_type == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        if sources:
            st.markdown("**Sources:**")
            for i, source in enumerate(sources):
                st.markdown(f"""
                <div class="source-citation">
                    <strong>Source {i+1}</strong> (Relevance: {source['score']:.3f})<br>
                    {source['text_preview']}
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">RAG Document Q&A System</div>', unsafe_allow_html=True)
    st.markdown("Upload documents and ask questions to get AI-powered answers with source citations.")
    
    # Sidebar for configuration and document management
    with st.sidebar:
        st.header("System Configuration")
        
        # Initialize components button
        if not st.session_state.system_ready:
            if st.button("Initialize AI System", type="primary"):
                initialize_components()
        else:
            st.success("System Ready")
        
        st.markdown("---")
        
        # Document upload section
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="secondary"):
                if st.session_state.system_ready:
                    process_uploaded_file(uploaded_file)
                else:
                    st.warning("Please initialize the system first!")
        
        st.markdown("---")
        
        # Document status
        st.header("Document Status")
        if st.session_state.documents_processed:
            for doc in st.session_state.documents_processed:
                with st.expander(f"{doc['name']}"):
                    st.write(f"**Chunks:** {doc['chunks']}")
                    st.write(f"**Size:** {doc['file_size']:,} bytes")
                    st.write(f"**Uploaded:** {doc['upload_time'][:19]}")
        else:
            st.info("No documents processed yet")
        
        # System statistics
        if st.session_state.system_ready and st.session_state.vector_store:
            st.markdown("---")
            st.header("🔧 System Stats")
            try:
                stats = st.session_state.vector_store.get_collection_stats()
                st.metric("Total Vectors", stats.get('total_vectors', 0))
                st.metric("Documents", len(st.session_state.documents_processed))
                
                # Model info
                if st.session_state.llm:
                    model_info = st.session_state.llm.get_model_info()
                    st.write(f"**LLM Backend:** {model_info['backend']}")
                    st.write(f"**Model:** {model_info['model_name']}")
                    
            except Exception as e:
                st.error(f"Error getting stats: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chat Interface")
        
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(
                message["type"], 
                message["content"],
                message.get("sources")
            )
        
        # Chat input
        if st.session_state.system_ready and st.session_state.documents_processed:
            # Use chat input for better UX
            if prompt := st.chat_input("Ask a question about your documents..."):
                # Add user message
                st.session_state.messages.append({
                    "type": "user",
                    "content": prompt,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Generate and display response
                with st.spinner("Thinking..."):
                    response = generate_response(prompt)
                
                if isinstance(response, dict):
                    # Successful RAG response
                    st.session_state.messages.append({
                        "type": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", []),
                        "timestamp": datetime.now().isoformat(),
                        "generation_time": response.get("generation_time", 0)
                    })
                else:
                    # Error or simple string response
                    st.session_state.messages.append({
                        "type": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Rerun to display new messages
                st.rerun()
        
        elif not st.session_state.system_ready:
            st.info("Please initialize the AI system first using the sidebar.")
        elif not st.session_state.documents_processed:
            st.info("Please upload and process a document first to start asking questions.")
    
    with col2:
        st.header("Quick Actions")
        
        # Sample questions
        if st.session_state.documents_processed:
            st.subheader("Sample Questions")
            sample_questions = [
                "What is the main topic of this document?",
                "Summarize the key points",
                "What are the most important details?",
                "Are there any specific recommendations?",
                "What should I know about this topic?"
            ]
            
            for question in sample_questions:
                if st.button(question, key=f"sample_{hash(question)}", use_container_width=True):
                    # Trigger the same flow as manual input
                    st.session_state.messages.append({
                        "type": "user",
                        "content": question,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    with st.spinner("Thinking..."):
                        response = generate_response(question)
                    
                    if isinstance(response, dict):
                        st.session_state.messages.append({
                            "type": "assistant",
                            "content": response["answer"],
                            "sources": response.get("sources", []),
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        st.session_state.messages.append({
                            "type": "assistant",
                            "content": response,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    st.rerun()
        
        # Clear chat button
        if st.session_state.messages:
            st.markdown("---")
            if st.button("Clear Chat History", type="secondary", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        # System info
        st.markdown("---")
        st.subheader("~ABOUT~")
        st.write("""
        This RAG system uses:
        - **Document Processing**: Extract text from PDFs
        - **Embeddings**: Convert text to vectors
        - **Vector Store**: ChromaDB for semantic search
        - **LLM**: Generate natural language responses
        - **Citations**: Track sources used
        """)
        
        # Performance tips
        with st.expander("Performance Tips"):
            st.write("""
            - Upload smaller documents for faster processing
            - Ask specific questions for better results
            - Check sources to verify information
            - GPT-2 responses may be quirky - upgrade to OpenAI for better quality
            """)

if __name__ == "__main__":
    main()