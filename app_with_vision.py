import streamlit as st
import os
import tempfile
import time
import shutil
import json
import ollama
import fitz  # PyMuPDF
import io
from PIL import Image

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.docstore.document import Document

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage


# --- Page Config ---
st.set_page_config(
    page_title="Multimodal RAG Chatbot (Vision)",
    page_icon="👁️",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def get_ollama_models():
    """Fetches available Ollama models."""
    try:
        models_info = ollama.list()
        
        # Check if response is a dictionary (older versions)
        if isinstance(models_info, dict):
            if 'models' in models_info:
                return [m['name'] for m in models_info['models']]
        
        # Check if response is an object (newer versions)
        elif hasattr(models_info, 'models'):
            return [m.model for m in models_info.models]
            
        return []
    except Exception as e:
        # Fallback if library fails, though normally library is reliable
        print(f"Error fetching models: {e}")
        return ["ministral-3b", "ministral-8b", "gemma3:4b", "qwen3-vl:4b", "qwen3-vl:latest", "llama3"] # Default fallbacks

def clean_history(history):
    """
    Removes the appended 'Time taken' and 'Sources' from AIMessages 
    so the LLM doesn't learn to hallucinate them.
    """
    cleaned = []
    for msg in history:
        if isinstance(msg, AIMessage):
            # Split by the specific marker we use to remove appended stats
            content = msg.content.split("\n\n**Time taken:")[0]
            cleaned.append(AIMessage(content=content))
        else:
            cleaned.append(msg)
    return cleaned

def refresh_source_cache():
    """Refreshes the list of unique sources from the vectorstore."""
    if st.session_state.vectorstore:
        try:
            # Only fetch metadatas to be faster
            data = st.session_state.vectorstore.get(include=['metadatas'])
            unique_sources = set()
            if data and 'metadatas' in data:
                for meta in data['metadatas']:
                    if 'source' in meta:
                        unique_sources.add(os.path.basename(meta['source']))
            st.session_state.unique_sources = sorted(list(unique_sources))
        except Exception as e:
            st.error(f"Error refreshing source cache: {e}")
            st.session_state.unique_sources = []
    else:
        st.session_state.unique_sources = []

def load_db_config(db_path):
    """Loads configuration from the vector DB directory."""
    config_path = os.path.join(db_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def generate_image_description(image_bytes, model_name="ministral-3b"):
    """Generates a text description for an image using a VLM."""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Describe this image in detail. Include any text, labels, charts, or visual information present. Be concise but thorough.',
                    'images': [image_bytes]
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error describing image: {e}")
        return "Error generating image description."

def load_documents(uploaded_files, vlm_model="ministral-3b"):
    """Loads text and images from multiple uploaded files with description generation."""
    documents = []
    
    # Ensure static directory for images
    if not os.path.exists("static/images"):
        os.makedirs("static/images", exist_ok=True)

    for uploaded_file in uploaded_files:
        # Save to temp file
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            if file_ext == '.pdf':
                # 1. Text Extraction
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = uploaded_file.name
                documents.extend(docs)

                # 2. Image Extraction
                pdf_document = fitz.open(tmp_file_path)
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Save Image
                        image_filename = f"{os.path.splitext(uploaded_file.name)[0]}_p{page_num+1}_i{img_index+1}.{image_ext}"
                        image_save_path = os.path.join("static/images", image_filename)
                        
                        with open(image_save_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # Generate Description
                        description = generate_image_description(image_bytes, model_name=vlm_model)
                        
                        # Create Document for Image Description
                        image_doc = Document(
                            page_content=f"Image Description (Page {page_num+1}): {description}",
                            metadata={
                                "source": uploaded_file.name,
                                "page": page_num + 1,
                                "type": "image",
                                "image_path": image_save_path
                            }
                        )
                        documents.append(image_doc)
                        st.toast(f"Processed image {img_index+1} on page {page_num+1}")

            elif file_ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(tmp_file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = uploaded_file.name
                documents.extend(docs)
                
        finally:
            # Clean up temp file
            try:
                os.remove(tmp_file_path)
            except:
                pass
                
    return documents

def create_vector_db(documents, embedding_model_name, chunking_strategy="Standard", chunk_size=1000, db_path="./chroma_db"):
    """Chunks documents and creates a Chroma vector store."""
    
    # Use Ollama Embeddings
    embeddings = OllamaEmbeddings(model=embedding_model_name)

    if chunking_strategy == "Semantic":
        st.info("Using Semantic Chunking...")
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    else:
        chunk_overlap = int(chunk_size * 0.15)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        
    chunks = text_splitter.split_documents(documents)
    st.write(f"Debug: Generated {len(chunks)} chunks.")
    
    if not chunks:
        st.error("Text splitting resulted in 0 chunks.")
        return None


    # Create Persisted VectorStore
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=db_path
    )
    
    # Save Config
    try:
        with open(os.path.join(db_path, "config.json"), "w") as f:
            json.dump({"embedding_model": embedding_model_name}, f)
    except Exception as e:
        st.error(f"Failed to save DB config: {e}")
        
    return vectorstore

def get_rag_chain(vectorstore, model_name, search_kwargs=None):
    """Creates the RAG chain with history."""
    llm = ChatOllama(model=model_name, temperature=0.3)
    
    # Configure retriever with search kwargs (e.g., filter)
    if search_kwargs is None:
        search_kwargs = {}
    
    # Default k=3 if not specified
    if "k" not in search_kwargs:
        search_kwargs["k"] = 3

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    # 1. Contextualize question: reformulates the latest question based on history
    contextualize_q_system_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # 2. Answer question
    qa_system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    
    IMPORTANT: You may receive descriptions of images as context. Treat these descriptions as factual observations of the visual content.
    
    Use three sentences maximum and keep the answer concise.
    Do NOT include "Time taken" or "Sources" in your answer. These are added automatically.
    
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# --- Main App Logic ---

def main():
    st.title("👁️ Multimodal RAG Chatbot")
    
    # Initialize Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "models" not in st.session_state:
        st.session_state.models = get_ollama_models()
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = ""
    
    if "unique_sources" not in st.session_state:
        st.session_state.unique_sources = []

    if "processing_time" not in st.session_state:
        st.session_state.processing_time = 0

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("📚 Document Base")
        
        # Knowledge Base Management
        kb_name = st.text_input("Knowledge Base Name", value="vision_kb", help="Change this to switch between different document sets.")
        db_path = f"./chroma_db_{kb_name.strip()}"
        
        if "current_kb" not in st.session_state:
            st.session_state.current_kb = kb_name
        
        # Detect change in KB
        if st.session_state.current_kb != kb_name:
            st.session_state.current_kb = kb_name
            st.session_state.vectorstore = None # Reset loaded DB on switch
            st.session_state.unique_sources = [] # Reset cache
            st.rerun()

        if os.path.exists(db_path):
            if st.button("🗑️ Delete Knowledge Base", type="secondary"):
                try:
                    shutil.rmtree(db_path)
                    st.session_state.vectorstore = None
                    st.session_state.unique_sources = [] # Reset cache
                    st.success(f"Deleted KB: {kb_name}")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting: {e}")

        # Chunking Strategy Selection
        st.subheader("Chunking Strategy")
        chunking_strategy = st.selectbox(
            "Select Strategy",
            ["Standard", "Semantic"],
            index=0,
            help="Standard: Fixed size chunks. Semantic: Chunks based on meaning (slower)."
        )

        chunk_size = st.slider(
            "Chunk Size (Standard)", 
            min_value=100, 
            max_value=2000, 
            value=1000, 
            step=50, 
            help="Size of text chunks for Standard strategy. Overlap is set to 15%."
        )

        # Model Selection
        st.subheader("Model Configuration")
        
        # Chat Model Selection
        # Prefer ministral-3b as default if available
        chat_models = st.session_state.models if st.session_state.models else ["ministral-3b", "llama3"]
        default_chat_index = 0
        for i, m in enumerate(chat_models):
            if "ministral" in m.lower():
                default_chat_index = i
                break

        selected_model = st.selectbox(
            "Select Chat Model", 
            chat_models,
            index=default_chat_index,
            key="select_chat_model"
        )
        
        # Vision Model Selection (for ingestion)
        # Suggest ministral, llava, moondream, gemma, or qwen
        vision_keywords = ["ministral", "llava", "moondream", "gemma", "qwen", "llama3.2-vision"]
        vision_models = [m for m in chat_models if any(k in m.lower() for k in vision_keywords)]
        if not vision_models:
             vision_models = chat_models # Fallback
        
        selected_vision_model = st.selectbox(
            "Select Vision Model (Ingestion)", 
            vision_models,
            index=0,
            key="select_vision_model",
            help="Model used to describe images during document processing."
        )

        # Embedding Model Selection
        # Try to find qwen3-embedding or all-minilm for default
        embed_models = st.session_state.models if st.session_state.models else ["all-minilm"]
        default_embed_index = 0
        for i, m in enumerate(embed_models):
            if "qwen3-embedding" in m:
                default_embed_index = i
                break
            elif "all-minilm" in m and default_embed_index == 0: # Prefer qwen3 but fallback to all-minilm
                default_embed_index = i

        selected_embed_model = st.selectbox(
            "Select Embedding Model",
            embed_models,
            index=default_embed_index,
            key="select_embedding_model"
        )
        
        # Load Persistent DB on Start if invalid
        if st.session_state.vectorstore is None and os.path.exists(db_path):
            try:
                # Load config
                config = load_db_config(db_path)
                loaded_model = config.get("embedding_model", selected_embed_model)
                
                if loaded_model != selected_embed_model:
                     st.caption(f"ℹ️ Loaded KB using model: `{loaded_model}` (Configured)")
                
                embeddings = OllamaEmbeddings(model=loaded_model)
                st.session_state.vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
                refresh_source_cache() # Cache logic
                st.toast(f"Loaded existing KB: {kb_name}")
            except Exception as e:
                st.error(f"Failed to load existing DB: {e}")

        # List Stored Documents
        if st.session_state.vectorstore:
            with st.expander("🗃️ Stored Documents"):
                if st.session_state.unique_sources:
                    for source in st.session_state.unique_sources:
                        st.text(f"• {source}")
                else:
                    st.caption("No source metadata found.")

        # File Uploader
        uploaded_files = st.file_uploader(
            "Upload Documents (Max 5)", 
            type=['pdf', 'docx'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.error("Maximum 5 files allowed.")
            else:
                if st.button("Process Documents (with Vision)", type="primary"):
                    p_start = time.time()
                    st.toast(f"Processing started at {time.strftime('%H:%M:%S', time.localtime(p_start))}")
                    with st.spinner(f"Ingesting documents and analyzing images with {selected_vision_model}..."):
                        # 1. Load
                        raw_docs = load_documents(uploaded_files, vlm_model=selected_vision_model)
                        if not raw_docs:
                            st.error("Could not load documents.")
                        else:
                            st.success(f"Loaded {len(raw_docs)} snippets (Text + Image Descriptions).")
                            
                            # Debug: Check content length
                            total_chars = sum(len(d.page_content) for d in raw_docs)
                            st.write(f"Debug: Total characters extracted: {total_chars}")
                            
                            # 2. Vector Store
                            embed_model = selected_embed_model
                            st.session_state.embed_model = embed_model
                            
                            st.info(f"Using embedding model: `{embed_model}`")
                            
                            
                            try:
                                vs = create_vector_db(raw_docs, embed_model, chunking_strategy, chunk_size, db_path)
                                
                                if vs:
                                    st.session_state.vectorstore = vs
                                    refresh_source_cache() # Cache logic
                                    p_end = time.time()
                                    st.session_state.processing_time = p_end - p_start
                                    st.success("Vector Database Ready! You can now chat.")
                                    time.sleep(1) # Give user a moment to see success message
                                    st.rerun()
                                else:
                                    st.error("Failed to create Vector Database. No chunks generated.")
                            except Exception as e:
                                st.error(f"Error creating vector DB: {e}")
                            
        st.divider()
        st.markdown(f"**Status:** {'🟢 Vector DB Ready' if st.session_state.vectorstore else '🔴 Not Processed'}")
        if st.session_state.vectorstore:
            # Debug check
            st.caption(f"DB initialized with model: {st.session_state.embed_model}")
            if st.session_state.processing_time > 0:
                st.caption(f"Processing Time: {st.session_state.processing_time:.2f} seconds")

    # --- Chat Interface ---
    
    # Display History
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # Chat Input
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Display User Message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate Response
        if st.session_state.vectorstore is None:
            with st.chat_message("assistant"):
                st.warning("Please upload and process documents first to enable RAG.")
        else:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    try:
                        # Logic to handle @ referencing
                        clean_input = user_input
                        search_kwargs = {}
                        
                        if "@" in user_input:
                            try:
                                # Use cached sources
                                available_sources = st.session_state.unique_sources
                                            
                                words = user_input.split()
                                for word in words:
                                    if word.startswith("@") and len(word) > 1:
                                        ref = word[1:]
                                        # Find matching source
                                        matched_source = None
                                        for source in available_sources:
                                            if ref.lower() in source.lower():
                                                matched_source = source
                                                break
                                        
                                        if matched_source:
                                            search_kwargs = {'filter': {'source': matched_source}}
                                            clean_input = clean_input.replace(word, "").strip()
                                            st.caption(f"🔒 Filtering by document: `{matched_source}`")
                                            break
                            except Exception as e:
                                st.error(f"Error parsing Reference: {e}")

                        # --- RAG Chain Caching Logic ---
                        current_chain_meta = {
                            "model": selected_model,
                            "vectorstore_id": id(st.session_state.vectorstore),
                            "search_kwargs": str(search_kwargs)
                        }
                        
                        # Check if we can reuse the cached chain
                        if ("rag_chain" not in st.session_state or 
                            "rag_chain_meta" not in st.session_state or 
                            st.session_state.rag_chain_meta != current_chain_meta):
                            
                            st.caption("⚙️ Rebuilding RAG Chain...")
                            st.session_state.rag_chain = get_rag_chain(
                                st.session_state.vectorstore, 
                                selected_model, 
                                search_kwargs=search_kwargs
                            )
                            st.session_state.rag_chain_meta = current_chain_meta
                        
                        chain = st.session_state.rag_chain
                        # -------------------------------
                        
                        start_time = time.time()
                        
                        # Clean history to prevent hallucination of metadata
                        cleaned_history = clean_history(st.session_state.chat_history)
                        
                        # Invoke chain
                        response = chain.invoke({
                            "input": clean_input,
                            "chat_history": cleaned_history
                        })
                        end_time = time.time()
                        
                        answer = response['answer']
                        
                        # Calculate time taken
                        execution_time = end_time - start_time
                        minutes = int(execution_time // 60)
                        seconds = int(execution_time % 60)
                        answer += f"\n\n**Time taken:** {minutes} min {seconds} sec"
                        
                        # Extract and format sources
                        source_names = []
                        referenced_images = []
                        if 'context' in response:
                            sources = set()
                            for doc in response['context']:
                                if 'source' in doc.metadata:
                                    sources.add(os.path.basename(doc.metadata['source']))
                                # Check for image type
                                if doc.metadata.get('type') == 'image':
                                    referenced_images.append(doc.metadata.get('image_path'))

                            if sources:
                                source_list = sorted(sources)
                                source_names = source_list
                                answer += f"\n\n**Sources:** {', '.join(source_list)}"
                        
                        message_placeholder.markdown(answer)
                        
                        # Display Referenced Images
                        if referenced_images:
                            st.divider()
                            st.caption("📸 Referenced Images:")
                            cols = st.columns(len(referenced_images))
                            for idx, img_path in enumerate(referenced_images):
                                with cols[idx]:
                                    if os.path.exists(img_path):
                                        st.image(img_path, caption=f"Image Ref {idx+1}", use_column_width=True)
                        
                        # Update History
                        st.session_state.chat_history.append(HumanMessage(content=user_input))
                        st.session_state.chat_history.append(AIMessage(content=answer))
                        
                        # Save Button Logic
                        filename_base = source_names[0] if source_names else "summary"
                        filename_base = os.path.splitext(filename_base)[0]
                        safe_prompt = "".join([c for c in user_input if c.isalnum() or c in (' ','-','_')]).strip()
                        prompt_segment = safe_prompt[:7].replace(" ", "_")
                        save_filename = f"{filename_base}_{prompt_segment}.txt"
                        
                        st.download_button(
                            label="💾 Save Response",
                            data=answer,
                            file_name=save_filename,
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
