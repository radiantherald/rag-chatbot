import streamlit as st
import os
import tempfile
import time
import shutil
import json
import ollama

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage

# --- Ragas Imports ---
try:
    import nest_asyncio
    nest_asyncio.apply()
    from ragas import evaluate, RunConfig
    # Use legacy metrics (note: pre-instantiated lower-case ones for standard metrics)
    from ragas.metrics import faithfulness, answer_relevancy, LLMContextPrecisionWithoutReference
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    RAGAS_ERROR = str(e)

# --- Google Gemini Imports ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
    GENAI_ERROR = None
except ImportError as e:
    GOOGLE_GENAI_AVAILABLE = False
    GENAI_ERROR = str(e)



# --- Page Config ---
st.set_page_config(
    page_title="RAG Chatbot Profile",
    page_icon="🤖",
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
        return ["qwen2.5:14b", "llama3"] # Default fallbacks

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

def load_documents(uploaded_files):
    """Loads text from multiple uploaded files."""
    documents = []
    
    for uploaded_file in uploaded_files:
        # Save to temp file
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(tmp_file_path, extract_images=True)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = uploaded_file.name
                documents.extend(docs)
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

def evaluate_rag(query, response, contexts, eval_model_selection, embed_model_name, google_api_key=None, include_retrieval=False):
    """
    Runs Ragas evaluation on a single query-response pair.
    eval_model_selection format: "local:model_name" or "gemini:gemini-pro"
    """
    if not RAGAS_AVAILABLE:
        return None
        
    try:
        # Initialize Embeddings (always local for now to check consistency with contexts)
        # Note: Ideally evaluator embeddings should be strong too, but we reuse the RAG ones for now
        from langchain_community.embeddings import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model=embed_model_name)

        # Initialize LLM (Judge)
        llm = None
        
        if eval_model_selection.startswith("gemini:"):
            if not GOOGLE_GENAI_AVAILABLE:
                st.error("langchain-google-genai not installed.")
                return None
            if not google_api_key:
                st.error("Google API Key required for Gemini evaluation.")
                return None
            
            gemini_model_name = eval_model_selection.split(":")[1]
            
            # Configure Safety Settings to be permissive for evaluation
            # (Evaluation prompts often trigger false positives)
            safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }

            llm = ChatGoogleGenerativeAI(
                model=gemini_model_name,
                google_api_key=google_api_key,
                temperature=0,
                n=1,
                safety_settings=safety_settings
            ) 
            
        else: # local
            from langchain_community.chat_models import ChatOllama
            local_model_name = eval_model_selection.split(":", 1)[1]
            llm = ChatOllama(model=local_model_name, temperature=0)

        # Prepare dataset
        data = {
            'question': [query],
            'answer': [response],
            'contexts': [contexts], 
        }

        dataset = Dataset.from_dict(data)

        
        # Wrap LLM and Embeddings for Ragas
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

        # Define Metrics
        # Use pre-instantiated metrics for faithfulness/answer_relevancy
        metrics = [faithfulness, answer_relevancy]
        if include_retrieval:
            metrics.append(LLMContextPrecisionWithoutReference())

        # Run Evaluation
        
        # Validation: Check if contexts are empty
        if not contexts:
            st.warning("⚠️ No contexts retrieved! Metrics like 'Faithfulness' cannot be computed.")
            # We can still try to run, but expect poor results
            
        # Configure Run with long timeout
        run_config = RunConfig(timeout=600, max_retries=10)
            
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            # llm and embeddings already passed to metrics, but can be redundant here
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
            raise_exceptions=True # Let errors bubble up so we can see them!
        )

        return result
    except Exception as e:
        st.error(f"Ragas Evaluation Failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# --- Main App Logic ---

def main():
    st.title("🤖 RAG Chatbot with Document Memory")
    
    # Initialize Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "models" not in st.session_state:
        st.session_state.models = get_ollama_models()
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = ""
    if "eval_model" not in st.session_state:
        st.session_state.eval_model = "local:qwen2.5:14b"
    if "google_api_key" not in st.session_state:
        st.session_state.google_api_key = ""

    
    if "unique_sources" not in st.session_state:
        st.session_state.unique_sources = []

    if "processing_time" not in st.session_state:
        st.session_state.processing_time = 0
        
    if "last_rag_data" not in st.session_state:
        st.session_state.last_rag_data = None

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("📚 Document Base")
        
        # Knowledge Base Management
        kb_name = st.text_input("Knowledge Base Name", value="default", help="Change this to switch between different document sets.")
        db_path = f"./chroma_db_{kb_name.strip()}"
        
        if "current_kb" not in st.session_state:
            st.session_state.current_kb = kb_name
        
        # Detect change in KB
        if st.session_state.current_kb != kb_name:
            st.session_state.current_kb = kb_name
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
        selected_model = st.selectbox(
            "Select Chat Model", 
            st.session_state.models if st.session_state.models else ["qwen2.5:14b"],
            index=0,
            key="select_chat_model"
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
                if st.button("Process Documents", type="primary"):
                    p_start = time.time()
                    st.toast(f"Processing started at {time.strftime('%H:%M:%S', time.localtime(p_start))}")
                    with st.spinner("Ingesting and embedding documents..."):
                        # 1. Load
                        raw_docs = load_documents(uploaded_files)
                        if not raw_docs:
                            st.error("Could not load documents.")
                        else:
                            st.success(f"Loaded {len(raw_docs)} snippets.")
                            
                            # Debug: Check content length
                            total_chars = sum(len(d.page_content) for d in raw_docs)
                            st.write(f"Debug: Total characters extracted: {total_chars}")
                            if total_chars == 0:
                                st.error("⚠️ No text extracted! If this is a scanned PDF, IT WILL NOT WORK. Please use a PDF with selectable text.")
                            
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
                st.caption(f"Processing Text Time: {st.session_state.processing_time:.2f} seconds")

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
                # Optional: Simple chat without context
                # response = simple_chat(user_input) 
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
                            "search_kwargs": str(search_kwargs) # key needs to be hashable/comparable
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
                        
                        raw_answer = response['answer']
                        
                        # Store for evaluation
                        if 'context' in response:
                            context_text = [doc.page_content for doc in response['context']]
                            st.session_state.last_rag_data = {
                                'question': clean_input,
                                'answer': raw_answer,
                                'contexts': context_text,
                                'chat_model': selected_model,
                                'embed_model': selected_embed_model
                            }
                        
                        answer = raw_answer
                        
                        # Calculate time taken
                        execution_time = end_time - start_time
                        minutes = int(execution_time // 60)
                        seconds = int(execution_time % 60)
                        answer += f"\n\n**Time taken:** {minutes} min {seconds} sec"
                        
                        # Extract and format sources
                        source_names = []
                        if 'context' in response:
                            sources = set()
                            for doc in response['context']:
                                if 'source' in doc.metadata:
                                    sources.add(os.path.basename(doc.metadata['source']))
                            
                            if sources:
                                source_list = sorted(sources)
                                source_names = source_list
                                answer += f"\n\n**Sources:** {', '.join(source_list)}"
                        
                        message_placeholder.markdown(answer)
                        
                        # Update History
                        st.session_state.chat_history.append(HumanMessage(content=user_input))
                        st.session_state.chat_history.append(AIMessage(content=answer))
                        
                        # Save Button Logic
                        # Create filename: {original_filename}_{first_7_chars_prompt}.txt
                        # We use the first source as 'original_filename' or fallback to 'summary' if no source
                        filename_base = source_names[0] if source_names else "summary"
                        # Remove extension if present in base
                        filename_base = os.path.splitext(filename_base)[0]
                        
                        # Clean prompt for filename
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

    # --- Ragas Evaluation Section ---
    if st.session_state.last_rag_data and st.session_state.vectorstore:
        with st.expander("📊 Ragas Evaluation", expanded=False):
            st.caption("Evaluate the accuracy of the last response using Ragas metrics.")
            
            if not RAGAS_AVAILABLE:
                st.warning("Ragas library not installed. Using a placeholder.")
                st.error(f"Debug Error: {RAGAS_ERROR}")
                st.code("pip install ragas datasets pandas")
            else:
                col1, col2 = st.columns([1, 2])
                with col1:
                    # Construct list of evaluator options
                    # 1. Google Gemini Models
                    gemini_options = [
                        "gemini:gemini-2.5-flash",
                        "gemini:gemini-2.5-pro",
                        "gemini:gemini-2.0-flash", 
                        "gemini:gemini-1.5-flash",
                        "gemini:gemini-1.5-pro",
                        "gemini:gemini-flash-latest"
                    ]
                    
                    # 2. Local Ollama Models (prefixed)
                    # Deduplicate and prioritize models
                    local_models = st.session_state.models if st.session_state.models else ["qwen2.5:14b", "llama3"]
                    local_options = [f"local:{m}" for m in local_models]
                    
                    all_eval_options = gemini_options + local_options
                    
                    eval_model = st.selectbox(
                        "Select Evaluator Model (Judge)", 
                        all_eval_options,
                        index=0,
                        key="eval_model_select"
                    )
                    
                    # API Key Input if Gemini selected
                    if eval_model.startswith("gemini:"):
                        st.session_state.google_api_key = st.text_input(
                            "Google API Key", 
                            value=st.session_state.google_api_key, 
                            type="password",
                            placeholder="AIzaSy..."
                        )
                        if not GOOGLE_GENAI_AVAILABLE:
                            st.error("⚠️ Library `langchain-google-genai` missing. Checking installation...")
                    
                    if eval_model.startswith("gemini:"):
                         if st.button("🧪 Test Gemini Connection (Debug)", type="secondary"):
                             if not st.session_state.google_api_key:
                                 st.error("Enter API Key first.")
                             else:
                                 try:
                                     import google.generativeai as genai
                                     genai.configure(api_key=st.session_state.google_api_key)
                                     # Use the selected model name (strip prefix)
                                     model_name = eval_model.split(":")[1]
                                     model = genai.GenerativeModel(model_name)
                                     response = model.generate_content("Hello, are you working?")
                                     st.success(f"Connection Successful! Response: {response.text}")
                                 except Exception as e:
                                     st.error(f"Connection Failed: {e}")

                    # Retrieval Metrics Toggle
                    include_retrieval = st.checkbox(
                        "Include Embedding/Retrieval Evaluation",
                        help="Adds 'Context Relevance' to the metrics. Checks if retrieved chunks are relevant to the question (no ground truth needed)."
                    )
                    
                    if st.button("Run Evaluation"):
                        with st.spinner(f"Running evaluation using {eval_model} (this may take a moment)..."):
                            data = st.session_state.last_rag_data
                            results = evaluate_rag(
                                query=data['question'], 
                                response=data['answer'], 
                                contexts=data['contexts'], 
                                eval_model_selection=eval_model, 
                                embed_model_name=data['embed_model'],
                                google_api_key=st.session_state.google_api_key,
                                include_retrieval=include_retrieval
                            )
                            
                            if results:
                                st.write("### Evaluation Results")
                                
                                # Helper to safely extract score
                                def get_score(res, key):
                                    try:
                                        val = res[key]
                                        if isinstance(val, list): # Check if it's a list (some versions do this)
                                            return val[0]
                                        return val
                                    except KeyError:
                                        return 0.0

                                st.metric("Faithfulness", f"{get_score(results, 'faithfulness'):.2f}")
                                st.metric("Answer Relevancy", f"{get_score(results, 'answer_relevancy'):.2f}")
                                if include_retrieval:
                                    # This metric is usually keyed as 'context_precision' or similar in the result
                                    # We'll try to get it safely using the helper
                                    score = get_score(results, 'context_precision') or get_score(results, 'llm_context_precision_without_reference')
                                    st.metric("Context Precision (Ref-Free)", f"{score:.2f}")
                                st.dataframe(results.to_pandas())
                with col2:
                    st.json(st.session_state.last_rag_data)

if __name__ == "__main__":
    main()
