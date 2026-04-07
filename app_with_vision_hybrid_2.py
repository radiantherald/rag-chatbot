import streamlit as st
import os
import tempfile
import time
import shutil
import json
import base64
import pickle
import re
import ollama
import fitz  # PyMuPDF
import io
import pandas as pd
import requests
from datetime import datetime
from PIL import Image

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.docstore.document import Document

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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

# --- DeepEval Imports ---
try:
    from deepeval.metrics import (
        FaithfulnessMetric, 
        AnswerRelevancyMetric, 
        ContextualRelevancyMetric, 
        ContextualPrecisionMetric, 
        ContextualRecallMetric,
        ToxicityMetric, 
        BiasMetric, 
        SummarizationMetric
    )
    from deepeval.test_case import LLMTestCase
    from deepeval.models.base_model import DeepEvalBaseLLM
    import nest_asyncio
    nest_asyncio.apply()
    DEEPEVAL_AVAILABLE = True
except ImportError as e:
    DEEPEVAL_AVAILABLE = False
    DEEPEVAL_ERROR = str(e)

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

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

OLLAMA_HOST = "http://127.0.0.1:11434"
FIXED_GEMMA4_MODEL = "gemma4:latest"
DEFAULT_NOMIC_EMBED_MODEL = "nomic-embed-text:latest"

def get_local_http_session():
    session = requests.Session()
    session.trust_env = False
    return session


def resolve_available_model(available_models, match_terms, fallback_name):
    """Resolves a preferred installed model name, falling back to the expected tag."""
    lowered_terms = [term.lower() for term in match_terms]
    for model_name in available_models or []:
        model_name_lower = model_name.lower()
        if any(term in model_name_lower for term in lowered_terms):
            return model_name
    return fallback_name

def get_ollama_models():
    """Fetches available Ollama models."""
    try:
        session = get_local_http_session()
        response = session.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if "models" in data:
                return [m["name"] for m in data["models"]]
        return []
    except Exception as e:
        print(f"Error fetching models: {e}")
        return ["ministral-3b", "qwen3-vl:4b", "deepseek-ocr:latest", "nomic-embed-text:latest", "llama3.2:3b"]

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

def refresh_source_cache(batch_size=500):
    """Refreshes the list of unique sources from the vectorstore without pulling the full collection at once."""
    if not st.session_state.vectorstore:
        st.session_state.unique_sources = []
        return

    try:
        unique_sources = set()
        collection = st.session_state.vectorstore._collection
        total = collection.count()

        offset = 0
        while offset < total:
            data = collection.get(include=["metadatas"], limit=batch_size, offset=offset)
            metadatas = data.get("metadatas", []) if data else []
            for meta in metadatas:
                if isinstance(meta, dict) and meta.get("source"):
                    unique_sources.add(os.path.basename(meta["source"]))
            offset += batch_size

        st.session_state.unique_sources = sorted(unique_sources)
    except Exception as e:
        st.error(f"Error refreshing source cache: {e}")
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

def save_tabular_data(tabular_data, save_path):
    """Persists uploaded tabular datasets for deterministic analytics."""
    try:
        with open(save_path, "wb") as f:
            pickle.dump(tabular_data, f)
    except Exception as e:
        st.error(f"Could not save tabular data: {e}")

def load_tabular_data(save_path):
    """Loads persisted tabular datasets if they exist."""
    if not os.path.exists(save_path):
        return {}

    try:
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        st.error(f"Could not load tabular data: {e}")
        return {}

def append_kb_log(log_path, event_type, payload):
    """Appends a structured JSONL event to the knowledge-base-specific log file."""
    event = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "event_type": event_type,
        **payload,
    }

    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        print(f"Failed to append KB log: {e}")

def serialize_document_for_log(doc, max_chars=None):
    """Serializes a LangChain Document into a JSON-safe structure for logging."""
    full_content = doc.page_content if getattr(doc, "page_content", None) else ""
    truncated = False
    content = full_content
    if max_chars is not None and len(full_content) > max_chars:
        content = full_content[:max_chars]
        truncated = True

    metadata = getattr(doc, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": str(metadata)}

    return {
        "page_content": content,
        "content_length": len(full_content),
        "truncated": truncated,
        "metadata": metadata,
    }

def serialize_documents_for_log(documents, max_chars=None):
    return [serialize_document_for_log(doc, max_chars=max_chars) for doc in documents]

def summarize_documents_for_log(documents):
    """Builds source/type summaries for a list of Documents."""
    source_counts = {}
    type_counts = {}
    total_chars = 0

    for doc in documents:
        total_chars += len(doc.page_content) if getattr(doc, "page_content", None) else 0
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source", "unknown")
        doc_type = metadata.get("type", "text")
        source_counts[source] = source_counts.get(source, 0) + 1
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

    return {
        "documents_count": len(documents),
        "total_characters": total_chars,
        "source_counts": source_counts,
        "type_counts": type_counts,
    }

def serialize_scored_results_for_log(scored_results, max_chars=2500):
    """Serializes similarity-search results of the form (Document, score)."""
    serialized = []
    for doc, score in scored_results:
        serialized.append({
            "score": float(score),
            "document": serialize_document_for_log(doc, max_chars=max_chars)
        })
    return serialized

def normalize_text(value):
    """Normalizes user text, column names, and cell values for matching."""
    text = "" if value is None else str(value)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def find_matching_column(columns, aliases):
    """Finds a column whose normalized name matches any alias."""
    normalized_map = {normalize_text(col): col for col in columns}
    for alias in aliases:
        normalized_alias = normalize_text(alias)
        if normalized_alias in normalized_map:
            return normalized_map[normalized_alias]
    return None

def find_matching_value(query, series):
    """Finds the longest categorical value mentioned in the query."""
    normalized_query = normalize_text(query)
    candidates = []

    for value in series.dropna().astype(str).unique():
        normalized_value = normalize_text(value)
        if normalized_value and normalized_value in normalized_query:
            candidates.append((len(normalized_value), value))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return str(candidates[0][1])

def is_tabular_analytics_question(question, tabular_data):
    """Detects when a question should be answered via deterministic table logic."""
    if not tabular_data:
        return False

    normalized_question = normalize_text(question)
    analytics_phrases = [
        "success rate",
        "success percentage",
        "percentage of successful",
        "successful transaction",
        "failed transaction",
        "failure rate",
        "how many successful",
        "how many failed",
        "count of successful",
        "count of failed",
    ]
    return any(phrase in normalized_question for phrase in analytics_phrases)

def answer_tabular_analytics(question, tabular_data):
    """
    Computes exact success-rate style metrics over uploaded CSV/XLSX data.
    Treats response code 0 as success and any other value as failure.
    """
    normalized_question = normalize_text(question)
    category_requested = "category" in normalized_question

    category_aliases = [
        "blr category",
        "blr_category",
        "category",
        "biller category",
        "service category",
    ]
    response_code_aliases = [
        "response code",
        "response_code",
        "resp code",
        "status code",
        "responsecode",
    ]

    matched_sources = []
    inferred_category_value = None
    total_transactions = 0
    successful_transactions = 0

    for source_name, df in tabular_data.items():
        if df is None or df.empty:
            continue

        response_code_col = find_matching_column(df.columns, response_code_aliases)
        if response_code_col is None:
            continue

        category_col = find_matching_column(df.columns, category_aliases)
        working_df = df

        if category_col is not None:
            category_value = find_matching_value(question, df[category_col])
            if category_requested and category_value is None:
                continue

            if category_value is not None:
                inferred_category_value = inferred_category_value or category_value
                category_mask = (
                    df[category_col]
                    .fillna("")
                    .astype(str)
                    .map(normalize_text)
                    == normalize_text(category_value)
                )
                working_df = df[category_mask]

        if working_df.empty:
            continue

        response_codes = pd.to_numeric(working_df[response_code_col], errors="coerce")
        total_transactions += len(working_df)
        successful_transactions += int((response_codes == 0).sum())
        matched_sources.append(source_name)

    if total_transactions == 0:
        return None

    failed_transactions = total_transactions - successful_transactions
    success_rate = (successful_transactions / total_transactions) * 100

    category_phrase = ""
    if inferred_category_value:
        category_phrase = f" in `{inferred_category_value}`"

    source_phrase = ", ".join(sorted(set(matched_sources)))
    return (
        f"The success rate of transactions{category_phrase} is "
        f"**{success_rate:.2f}%**.\n\n"
        f"- Total transactions: **{total_transactions}**\n"
        f"- Successful transactions (`response code = 0`): **{successful_transactions}**\n"
        f"- Failed transactions (`response code != 0`): **{failed_transactions}**\n"
        f"- Computed from structured tabular data in: `{source_phrase}`"
    )

def create_embedding_function(embedding_model_name, local_only=False):
    """Builds the embedding function with consistent settings and clearer failures."""
    if embedding_model_name == "all-MiniLM-L6-v2":
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {"device": "cpu"}
        if local_only:
            model_kwargs["local_files_only"] = True

        try:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": False}
            )
        except Exception as embedding_error:
            if local_only:
                raise RuntimeError(
                    "Failed to initialize all-MiniLM-L6-v2 from local files. "
                    "Make sure `sentence-transformers` is installed and the model is already cached locally."
                ) from embedding_error
            raise RuntimeError(
                "Failed to initialize all-MiniLM-L6-v2. "
                "Make sure `sentence-transformers` is installed and the model can be loaded."
            ) from embedding_error

    return OllamaEmbeddings(model=embedding_model_name, base_url=OLLAMA_HOST)

def run_vision_prompt(image_bytes, model_name, prompt, timeout=(10, 1800)):
    """Runs a local Ollama multimodal request with proxy bypass and base64-encoded image bytes."""
    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(image_bytes).decode("ascii")]
            }
        ]
    }

    session = get_local_http_session()
    try:
        response = session.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except requests.exceptions.ReadTimeout as timeout_error:
        raise RuntimeError(
            f"Ollama request timed out for model `{model_name}`. "
            "Try OCR-only mode, a lower page render resolution, or a faster model."
        ) from timeout_error

def clean_extracted_text(text):
    """Normalizes extracted text while preserving basic paragraph breaks."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean_ocr_text(text):
    """Normalizes noisy OCR spacing while preserving basic paragraph breaks."""
    return clean_extracted_text(text)


def has_meaningful_native_pdf_text(text, min_chars=80):
    """Uses multiple signals to decide whether native PDF text is strong enough to trust."""
    cleaned = clean_extracted_text(text)
    if not cleaned:
        return False

    compact = re.sub(r"\s+", " ", cleaned).strip()
    if not compact:
        return False

    words = re.findall(r"\b\w+\b", compact)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    meaningful_lines = sum(1 for line in lines if len(re.findall(r"\w+", line)) >= 2)

    non_space_chars = sum(1 for ch in compact if not ch.isspace())
    alnum_chars = sum(1 for ch in compact if ch.isalnum())
    alnum_ratio = alnum_chars / max(non_space_chars, 1)

    # Strong narrative pages or dense extracted text can safely stay on native extraction.
    if len(compact) >= 250 or len(words) >= 40:
        return True

    # Moderate text with healthy structure is usually reliable enough to avoid OCR.
    if len(compact) >= 120 and len(words) >= 20:
        return True

    # Shorter pages can still be meaningful if the extracted text looks like real text
    # rather than sparse artifacts from a flattened scan.
    if len(compact) >= min_chars and len(words) >= 12 and meaningful_lines >= 3 and alnum_ratio >= 0.55:
        return True

    # Forms, brief notices, and table-heavy pages may have fewer words but several
    # structured lines of usable selectable text.
    if len(words) >= 8 and meaningful_lines >= 5 and alnum_ratio >= 0.6:
        return True

    return False

def generate_page_ocr_and_vlm(image_bytes, ocr_model_name, vlm_model_name, run_ocr=True, run_vlm=True):
    """Runs OCR, VLM, or both against a rendered page image."""
    ocr_text = None
    vlm_text = None

    if run_ocr:
        ocr_prompt = (
            "Extract only the text that is visibly present in this image. "
            "Do not summarize. Do not explain. Do not infer missing words. "
            "Do not add information that is not visible. "
            "Preserve line breaks and layout as much as possible. "
            "If a word or line is unreadable, write [UNREADABLE]."
        )
        ocr_text = run_vision_prompt(image_bytes, ocr_model_name, ocr_prompt)
        ocr_text = clean_ocr_text(ocr_text)

    if ocr_text and len(ocr_text.strip()) > 500:
        run_vlm = False

    if run_vlm:
        vlm_prompt = (
            "Describe the visual structure of this page. "
            "Focus on diagrams, flowcharts, tables, arrows, boxes, and relationships between elements. "
            "If the page is mostly text, say that briefly. "
            "Do not hallucinate text that is not visible."
        )
        vlm_text = run_vision_prompt(image_bytes, vlm_model_name, vlm_prompt)

    return ocr_text, vlm_text

def generate_image_description(image_bytes, model_name="ministral-3b"):
    """Generates a text description for an image using a VLM."""
    try:
        prompt = (
            "Extract only the text that is visibly present in this image. "
            "Do not summarize. Do not explain. Do not infer missing words. "
            "Do not add information that is not visible. "
            "Preserve line breaks and layout as much as possible. "
            "If a word or line is unreadable, write [UNREADABLE]."
            if "ocr" in model_name.lower()
            else
            "Describe this image in detail. Include any text, labels, charts, or visual information present. Be concise but thorough."
        )
        return run_vision_prompt(image_bytes, model_name, prompt)
    except Exception as e:
        print(f"Error describing image: {e}")
        return f"Error generating image description: {str(e)}"

def load_documents(
    uploaded_files,
    vlm_model="ministral-3b",
    ocr_model=None,
    scanned_pdf_mode="OCR + VLM hybrid",
    progress_callback=None,
):
    """Loads text and images from multiple uploaded files with description generation."""
    documents = []
    tabular_data = {}
    
    # Ensure static directory for images
    if not os.path.exists("static/images"):
        os.makedirs("static/images", exist_ok=True)

    total_files = max(len(uploaded_files), 1)
    for file_index, uploaded_file in enumerate(uploaded_files, start=1):
        if progress_callback:
            progress_callback(
                f"Processing file {file_index}/{total_files}: {uploaded_file.name}",
                progress=0.60 * ((file_index - 1) / total_files)
            )
        # Save to temp file
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            if file_ext == '.pdf':
                use_native_text = scanned_pdf_mode in ["Text only", "VLM only"]
                suggested_hybrid_pages = []
                suggested_ocr_pages = []

                # 1. Text Extraction
                if use_native_text:
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['source'] = uploaded_file.name
                    documents.extend(docs)

                # 2. PDF Page Processing
                pdf_document = fitz.open(tmp_file_path)
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    if progress_callback:
                        file_phase = page_num / max(len(pdf_document), 1)
                        progress_callback(
                            f"Reading PDF page {page_num+1}/{len(pdf_document)} from {uploaded_file.name}",
                            progress=0.60 * (((file_index - 1) + file_phase) / total_files)
                        )
                    native_page_text = clean_extracted_text(page.get_text("text"))
                    has_native_text = has_meaningful_native_pdf_text(native_page_text)
                    image_list = page.get_images()
                    has_embedded_images = len(image_list) > 0

                    if not use_native_text and has_native_text:
                        documents.append(Document(
                            page_content=native_page_text,
                            metadata={
                                "source": uploaded_file.name,
                                "page": page_num,
                                "type": "pdf_text"
                            }
                        ))

                    run_ocr = False
                    run_vlm = False

                    if scanned_pdf_mode == "Text only":
                        if not has_native_text:
                            suggested_ocr_pages.append(page_num + 1)
                        continue
                    elif scanned_pdf_mode == "VLM only":
                        if not has_native_text and not has_embedded_images:
                            suggested_hybrid_pages.append(page_num + 1)
                        run_vlm = has_embedded_images
                    elif scanned_pdf_mode == "OCR only":
                        run_ocr = not has_native_text
                    elif scanned_pdf_mode == "OCR + VLM hybrid":
                        run_ocr = not has_native_text
                        if has_native_text:
                            run_vlm = has_embedded_images
                        else:
                            run_vlm = True

                    if run_ocr:
                        if progress_callback:
                            progress_callback(
                                f"Running OCR on page {page_num+1}/{len(pdf_document)} of {uploaded_file.name}",
                                progress=0.60 * (((file_index - 1) + ((page_num + 0.35) / max(len(pdf_document), 1))) / total_files)
                            )
                        pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
                        image_bytes = pix.tobytes("png")
                        image_filename = f"{os.path.splitext(uploaded_file.name)[0]}_page_{page_num+1}_fullpage.png"
                        image_save_path = os.path.join("static/images", image_filename)

                        with open(image_save_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        page_ocr_text, page_vlm_text = generate_page_ocr_and_vlm(
                            image_bytes=image_bytes,
                            ocr_model_name=ocr_model or vlm_model,
                            vlm_model_name=vlm_model,
                            run_ocr=True,
                            run_vlm=run_vlm,
                        )

                        if page_ocr_text:
                            st.session_state.processed_images.append({
                                "source": uploaded_file.name,
                                "page": page_num + 1,
                                "image_path": image_save_path,
                                "description": f"[OCR]\n{page_ocr_text}"
                            })
                            documents.append(Document(
                                page_content=f"OCR Text (Page {page_num+1}): {page_ocr_text}",
                                metadata={
                                    "source": uploaded_file.name,
                                    "page": page_num + 1,
                                    "type": "ocr_page",
                                    "image_path": image_save_path
                                }
                            ))

                        if page_vlm_text:
                            st.session_state.processed_images.append({
                                "source": uploaded_file.name,
                                "page": page_num + 1,
                                "image_path": image_save_path,
                                "description": f"[VLM]\n{page_vlm_text}"
                            })
                            documents.append(Document(
                                page_content=f"Visual Description (Page {page_num+1}): {page_vlm_text}",
                                metadata={
                                    "source": uploaded_file.name,
                                    "page": page_num + 1,
                                    "type": "vlm_page",
                                    "image_path": image_save_path
                                }
                            ))

                        st.toast(f"Processed page {page_num+1} with mode: {scanned_pdf_mode}")

                    elif run_vlm:
                        if progress_callback:
                            progress_callback(
                                f"Analyzing extracted images on page {page_num+1}/{len(pdf_document)} of {uploaded_file.name}",
                                progress=0.60 * (((file_index - 1) + ((page_num + 0.35) / max(len(pdf_document), 1))) / total_files)
                            )
                        vlm_prompt = (
                            "Describe this extracted image from a PDF page. "
                            "Focus on diagrams, charts, flowcharts, tables, labels, arrows, boxes, "
                            "and relationships between visible elements. "
                            "Include any text visible inside the image. "
                            "Do not hallucinate content that is not visible."
                        )

                        for img_index, img in enumerate(image_list):
                            xref = img[0]
                            try:
                                base_image = pdf_document.extract_image(xref)
                            except Exception as e:
                                print(f"Error extracting PDF image on page {page_num+1}: {e}")
                                continue

                            image_bytes = base_image.get("image")
                            image_ext = base_image.get("ext", "png")
                            if not image_bytes:
                                continue

                            image_filename = (
                                f"{os.path.splitext(uploaded_file.name)[0]}_p{page_num+1}_i{img_index+1}.{image_ext}"
                            )
                            image_save_path = os.path.join("static/images", image_filename)

                            with open(image_save_path, "wb") as img_file:
                                img_file.write(image_bytes)

                            image_vlm_text = run_vision_prompt(image_bytes, vlm_model, vlm_prompt)

                            st.session_state.processed_images.append({
                                "source": uploaded_file.name,
                                "page": page_num + 1,
                                "image_path": image_save_path,
                                "description": f"[VLM]\n{image_vlm_text}"
                            })

                            documents.append(Document(
                                page_content=(
                                    f"Visual Description (Page {page_num+1}, Image {img_index+1}): "
                                    f"{image_vlm_text}"
                                ),
                                metadata={
                                    "source": uploaded_file.name,
                                    "page": page_num + 1,
                                    "type": "vlm_image",
                                    "image_index": img_index + 1,
                                    "image_path": image_save_path
                                }
                            ))

                        st.toast(f"Added VLM image augmentation for page {page_num+1}")

                if scanned_pdf_mode == "Text only" and suggested_ocr_pages:
                    preview = ", ".join(str(page) for page in suggested_ocr_pages[:8])
                    suffix = "..." if len(suggested_ocr_pages) > 8 else ""
                    st.warning(
                        f"{uploaded_file.name}: pages {preview}{suffix} have little or no selectable text. "
                        "If you need those pages parsed too, use `OCR only` or `OCR + VLM hybrid`."
                    )

                if scanned_pdf_mode == "VLM only" and suggested_hybrid_pages:
                    preview = ", ".join(str(page) for page in suggested_hybrid_pages[:8])
                    suffix = "..." if len(suggested_hybrid_pages) > 8 else ""
                    st.warning(
                        f"{uploaded_file.name}: pages {preview}{suffix} look like flattened scans "
                        "(little/no selectable text and no embedded images). "
                        "`VLM only` cannot inspect those pages, so `OCR + VLM hybrid` is the safer mode."
                    )

            elif file_ext in ['.docx', '.doc']:
                if progress_callback:
                    progress_callback(
                        f"Extracting text from {uploaded_file.name}",
                        progress=0.60 * ((file_index - 0.5) / total_files)
                    )
                loader = Docx2txtLoader(tmp_file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = uploaded_file.name
                documents.extend(docs)
                
                if file_ext == '.docx':
                    import zipfile
                    try:
                        with zipfile.ZipFile(tmp_file_path, 'r') as doc_zip:
                            img_index = 0
                            for info in doc_zip.infolist():
                                if info.filename.startswith("word/media/") and info.filename.endswith(('.png', '.jpeg', '.jpg', '.gif', '.bmp', '.tiff')):
                                    image_bytes = doc_zip.read(info.filename)
                                    image_ext = os.path.splitext(info.filename)[1]
                                    
                                    # Save Image
                                    image_filename = f"{os.path.splitext(uploaded_file.name)[0]}_docx_i{img_index+1}{image_ext}"
                                    image_save_path = os.path.join("static/images", image_filename)
                                    
                                    with open(image_save_path, "wb") as img_file:
                                        img_file.write(image_bytes)
                                        
                                    # Generate Description
                                    if progress_callback:
                                        progress_callback(
                                            f"Analyzing DOCX image {img_index+1} from {uploaded_file.name}",
                                            progress=0.60 * ((file_index - 0.2) / total_files)
                                        )
                                    description = generate_image_description(image_bytes, model_name=vlm_model)
                                    
                                    st.session_state.processed_images.append({
                                        "source": uploaded_file.name,
                                        "page": "N/A (DOCX)",
                                        "image_path": image_save_path,
                                        "description": description
                                    })
                                    
                                    # Create Document for Image Description
                                    image_doc = Document(
                                        page_content=f"Image Description (DOCX Image {img_index+1}): {description}",
                                        metadata={
                                            "source": uploaded_file.name,
                                            "page": "N/A",
                                            "type": "image",
                                            "image_path": image_save_path
                                        }
                                    )
                                    documents.append(image_doc)
                                    img_index += 1
                                    st.toast(f"Processed image {img_index} from {uploaded_file.name}")
                    except zipfile.BadZipFile:
                        pass # Invalid docx or actually a .doc file
                    except Exception as e:
                        print(f"Error extracting images from DOCX: {e}")
                
            elif file_ext == ".csv":
                if progress_callback:
                    progress_callback(
                        f"Loading CSV data from {uploaded_file.name}",
                        progress=0.60 * ((file_index - 0.5) / total_files)
                    )
                from langchain_community.document_loaders import CSVLoader
                df = pd.read_csv(tmp_file_path)
                tabular_data[uploaded_file.name] = df
                # CSVLoader automatically formats as "Column: Value" per row
                loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                     doc.metadata['source'] = uploaded_file.name
                documents.extend(docs)
                
            elif file_ext in [".xlsx", ".xls"]:
                if progress_callback:
                    progress_callback(
                        f"Loading Excel data from {uploaded_file.name}",
                        progress=0.60 * ((file_index - 0.5) / total_files)
                    )
                # Debug: Show we are processing Excel
                st.write(f"📂 Processing Excel: {uploaded_file.name}")
                # Read Excel
                df = pd.read_excel(tmp_file_path)
                tabular_data[uploaded_file.name] = df
                
                # Debug: Inspect data
                st.caption(f"📊 Loaded DataFrame Shape: {df.shape}")
                
                # Convert rows to "Column: Value" text with Grouping
                rows_converted = 0
                GROUP_SIZE = 10 # Group 10 rows per chunk for speed
                
                current_group_content = []
                start_row_idx = 1 # 1-based index for display
                
                for index, row in df.iterrows():
                    # Create a text block for each row
                    row_text = []
                    for col in df.columns:
                        # Skip empty values
                        if pd.notna(row[col]):
                            row_text.append(f"{col}: {row[col]}")
                    
                    row_content = ", ".join(row_text) # Use comma for compactness within a row
                    if row_content.strip():
                        # Add row number for clear reference in the chunk
                        current_group_content.append(f"[Row {index+1}] {row_content}")
                    
                    # Check if group is full
                    if len(current_group_content) >= GROUP_SIZE:
                        combined_content = "\n".join(current_group_content) # Newline between rows
                        
                        documents.append(Document(
                            page_content=combined_content,
                            metadata={
                                "source": uploaded_file.name,
                                "row_start": start_row_idx,
                                "row_end": index + 1,
                                "type": "excel_row_group"
                            }
                        ))
                        rows_converted += 1
                        current_group_content = [] # Reset
                        start_row_idx = index + 2 # Next row
                
                # Flush remaining
                if current_group_content:
                    combined_content = "\n".join(current_group_content)
                    documents.append(Document(
                        page_content=combined_content,
                        metadata={
                            "source": uploaded_file.name,
                            "row_start": start_row_idx,
                            "row_end": len(df),
                            "type": "excel_row_group"
                        }
                    ))
                    rows_converted += 1
                
                st.caption(f"✅ Generated {rows_converted} chunks (grouped by {GROUP_SIZE}) from {uploaded_file.name}")
                
            if progress_callback:
                progress_callback(
                    f"Finished processing {uploaded_file.name}",
                    progress=0.60 * (file_index / total_files)
                )

        finally:
            # Clean up temp file
            try:
                os.remove(tmp_file_path)
            except:
                pass
                
    return documents, tabular_data

def create_vector_db(
    documents,
    embedding_model_name,
    chunking_strategy="Standard",
    chunk_size=1000,
    db_path="./chroma_db",
    progress_callback=None,
):
    """Chunks documents and creates a Chroma vector store."""
    if progress_callback:
        progress_callback("Initializing embedding model...", progress=0.68)

    embeddings = create_embedding_function(embedding_model_name)
    chunk_overlap = None

    if chunking_strategy == "Semantic":
        st.info("Using Semantic Chunking...")
        if progress_callback:
            progress_callback("Running semantic chunking...", progress=0.74)
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    else:
        chunk_overlap = int(chunk_size * 0.15)
        if progress_callback:
            progress_callback(
                f"Splitting documents into standard chunks (size {chunk_size}, overlap {chunk_overlap})...",
                progress=0.74
            )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        
    chunks = text_splitter.split_documents(documents)
    st.write(f"Debug: Generated {len(chunks)} chunks.")
    if progress_callback:
        progress_callback(f"Generated {len(chunks)} chunks. Creating vector store...", progress=0.84)

    vector_db_stats = {
        "embedding_model": embedding_model_name,
        "chunking_strategy": chunking_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "source_document_count": len(documents),
        "chunk_count": len(chunks),
        "chunk_samples": serialize_documents_for_log(chunks[:5], max_chars=1200),
    }
    
    if not chunks:
        st.error("Text splitting resulted in 0 chunks.")
        return None, vector_db_stats


    # Create Persisted VectorStore
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=db_path
    )
    if progress_callback:
        progress_callback("Persisting vector store and configuration...", progress=0.94)
    
    # Save Config
    try:
        with open(os.path.join(db_path, "config.json"), "w") as f:
            json.dump({"embedding_model": embedding_model_name}, f)
    except Exception as e:
        st.error(f"Failed to save DB config: {e}")
        
    return vectorstore, vector_db_stats

class SemanticCache:
    """
    A simple semantic cache using ChromaDB.
    Stores (Question, Answer) pairs.
    """
    def __init__(self, embedding_function, persist_directory="./chroma_cache"):
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.collection_name = "semantic_cache"
        
        # Initialize Cache DB
        # Force Cosine Similarity
        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def lookup(self, question, threshold=0.80):
        """
        Checks if a similar question exists in the cache.
        Returns (answer, distance, id) tuple.
        """
        try:
            # Explicitly embed the query to ensure consistent distance calculation
            # Sometimes query_texts might not trigger the embedding function correctly on the raw collection
            query_embedding = self.embedding_function.embed_query(question)
            
            results = self.db._collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results or not results['ids'] or not results['ids'][0]:
                return None, 1.0, None # Return high distance (1.0) on miss
                
            distance = results['distances'][0][0]
            metadata = results['metadatas'][0][0]
            doc_id = results['ids'][0][0]
            
            # Threshold check
            if distance < (1 - threshold):
                return metadata.get("answer"), distance, doc_id
            
            return None, distance, None
            
        except Exception as e:
            print(f"Cache lookup error: {e}")
            return None, 1.0, None

    def delete_by_id(self, cache_id):
        """
        Deletes a specific cache entry by its ID.
        """
        try:
            self.db._collection.delete(ids=[cache_id])
            if hasattr(self.db, 'persist'):
                self.db.persist()
            return True
        except Exception as e:
            print(f"Error deleting from cache: {e}")
            return False

    def add(self, question, answer):
        """Adds a question-answer pair to the cache. Returns ID."""
        import uuid
        doc_id = str(uuid.uuid4())
        self.db.add_documents(
            [
                Document(
                    page_content=question,
                    metadata={"answer": answer, "timestamp": time.time()}
                )
            ],
            ids=[doc_id]
        )
        # Force persist
        try:
            if hasattr(self.db, "persist"):
                self.db.persist()
        except:
            pass
        return doc_id

    # --- Optional safer semantic-cache design (currently disabled) ---
    # Instead of returning the cached answer verbatim, store the retrieved
    # context chunks alongside the question/answer so a future semantically
    # similar question can be re-answered by the LLM using the cached context.
    #
    # This is safer for near-similar questions, but slower than direct answer
    # reuse because it still requires a generation step.
    #
    # def add_with_context(self, question, answer, contexts):
    #     """Adds a question-answer pair plus retrieved contexts to the cache."""
    #     import uuid
    #     doc_id = str(uuid.uuid4())
    #     self.db.add_documents(
    #         [
    #             Document(
    #                 page_content=question,
    #                 metadata={
    #                     "answer": answer,
    #                     "contexts": contexts,
    #                     "timestamp": time.time()
    #                 }
    #             )
    #         ],
    #         ids=[doc_id]
    #     )
    #     try:
    #         if hasattr(self.db, "persist"):
    #             self.db.persist()
    #     except:
    #         pass
    #     return doc_id
    #
    # def lookup_with_context(self, question, threshold=0.80):
    #     """
    #     Looks up a semantically similar cached question and returns the cached
    #     answer together with the stored contexts.
    #     """
    #     try:
    #         query_embedding = self.embedding_function.embed_query(question)
    #         results = self.db._collection.query(
    #             query_embeddings=[query_embedding],
    #             n_results=1,
    #             include=["documents", "metadatas", "distances"]
    #         )
    #
    #         if not results or not results['ids'] or not results['ids'][0]:
    #             return None, None, 1.0, None
    #
    #         distance = results['distances'][0][0]
    #         metadata = results['metadatas'][0][0]
    #         doc_id = results['ids'][0][0]
    #
    #         if distance < (1 - threshold):
    #             return metadata.get("answer"), metadata.get("contexts", []), distance, doc_id
    #
    #         return None, None, distance, None
    #     except Exception as e:
    #         print(f"Context-aware cache lookup error: {e}")
    #         return None, None, 1.0, None

def get_standalone_question(user_input, chat_history, llm_model_name):
    """
    Uses an LLM to reformulate the user input into a standalone question
    based on the chat history.
    """
    llm = ChatOllama(model=llm_model_name, temperature=0, base_url=OLLAMA_HOST)
    
    if not chat_history:
        return user_input
        
    contextualize_q_system_prompt = """You are a strict query reformulator.
    Given a chat history and the latest user question which might reference context in the chat history, 
    formulate a standalone question which can be understood without the chat history.

    STRICT RULES:
    1. NEVER answer the user's question, and NEVER analyze the chat history.
    2. NEVER output your "understanding" or thought process.
    3. Return ONLY the raw standalone question text. No preamble, no explanation.
    4. If the user question is already standalone, or unrelated to the chat history, RETURN IT EXACTLY AS IS.
    5. Ensure the output is a single clean sentence.

    EXAMPLES:
    Chat History:
    Human: What is the RTA system?
    AI: It is a Road Trip Assistant used for planning...

    User Input: explain its dependency description
    Result: What is the dependency description of the RTA system?

    User Input: how do I bake a cake?
    Result: how do I bake a cake?

    Chat History:
    Human: Describe this image
    AI: The image shows a flowchart for the Rutabara arc widgets framework.

    User Input: what is rutabara?
    Result: what is rutabara?
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    chain = contextualize_q_prompt | llm | StrOutputParser()
    
    try:
        import re
        reformulated_question = chain.invoke({
            "chat_history": chat_history,
            "input": user_input
        })
        
        # Clean up reasoning tags (e.g. from DeepSeek R1)
        reformulated_question = re.sub(r'<think>.*?</think>', '', reformulated_question, flags=re.DOTALL).strip()
        
        # Fallback if the model started generating reasoning instead of a question
        lower_q = reformulated_question.lower()
        if lower_q.startswith("thinking...") or "understanding -" in lower_q or "there is no solution" in lower_q:
            print(f"Debug: Model hallucinated reasoning. Falling back to original input.")
            return user_input
        
        # Sanity Check: If the model hallucinated an answer (long text), fallback.
        if len(reformulated_question) > 300:
            print(f"Debug: Reformulation rejected (too long, {len(reformulated_question)} chars). Falling back to input.")
            return user_input
            
        return reformulated_question
    except Exception as e:
        print(f"Error reformulating question: {e}")
        return user_input

def get_retrieval_guard_thresholds(question):
    """
    Returns grounding thresholds for the pre-retrieval relevance guard.
    Summary-style questions can tolerate broader context, while normal questions
    require stronger top-match relevance before we answer.
    """
    normalized_question = question.lower()
    is_summary_task = any(
        word in normalized_question
        for word in ["summarize", "summary", "overview", "elaborate", "explain", "detailed"]
    )

    if is_summary_task:
        return {
            "min_top_score": 0.18,
            "min_avg_score": 0.10,
            "score_sample_k": 4,
        }

    return {
        "min_top_score": 0.28,
        "min_avg_score": 0.18,
        "score_sample_k": 3,
    }

def get_response_style_instruction(question):
    """Infers the requested answer format from the user's wording."""
    normalized_question = question.lower()

    if "bullet points" in normalized_question or "bullet" in normalized_question:
        return "Return the answer as concise markdown bullet points."
    if "step by step" in normalized_question or "steps" in normalized_question:
        return "Return the answer as a numbered list."
    if "table" in normalized_question or "tabular" in normalized_question:
        return "Return the answer as a markdown table when appropriate."
    return "Return the answer in normal prose."

def check_retrieval_grounding(vectorstore, question, search_kwargs=None):
    """
    Runs a lightweight similarity-score check before invoking the full RAG chain.
    If the top matches are too weak, we refuse instead of letting the LLM answer
    from weakly related chunks.
    """
    if search_kwargs is None:
        search_kwargs = {}

    thresholds = get_retrieval_guard_thresholds(question)
    score_k = min(search_kwargs.get("k", 3), thresholds["score_sample_k"])
    similarity_kwargs = {"k": max(score_k, 1)}
    if "filter" in search_kwargs:
        similarity_kwargs["filter"] = search_kwargs["filter"]

    try:
        scored_results = vectorstore.similarity_search_with_relevance_scores(
            question,
            **similarity_kwargs
        )
    except Exception as e:
        # Fail open if score probing is unavailable for the current vectorstore.
        print(f"Retrieval guard skipped: {e}")
        return {
            "allow_answer": True,
            "reason": f"relevance scoring unavailable: {e}",
            "top_score": None,
            "avg_score": None,
            "scored_results": [],
        }

    if not scored_results:
        return {
            "allow_answer": False,
            "reason": "no relevant chunks were retrieved from this knowledge base",
            "top_score": 0.0,
            "avg_score": 0.0,
            "scored_results": [],
        }

    scores = [max(0.0, min(1.0, float(score))) for _, score in scored_results]
    top_score = scores[0]
    avg_score = sum(scores) / len(scores)

    allow_answer = (
        top_score >= thresholds["min_top_score"] and
        avg_score >= thresholds["min_avg_score"]
    )

    if not allow_answer:
        reason = (
            f"top relevance {top_score:.2f} and average relevance {avg_score:.2f} "
            f"were below the grounding thresholds for this question"
        )
    else:
        reason = "retrieved context cleared relevance guard"

    return {
        "allow_answer": allow_answer,
        "reason": reason,
        "top_score": top_score,
        "avg_score": avg_score,
        "scored_results": scored_results,
    }

def get_rag_chain_standard(vectorstore, model_name, search_kwargs=None, use_reranker=False):
    """
    Creates a standard RAG chain that takes a 'standalone_question' 
    and returns an answer.
    """
    llm = ChatOllama(model=model_name, temperature=0.3, base_url=OLLAMA_HOST)
    
    if search_kwargs is None:
        search_kwargs = {}
    if "k" not in search_kwargs:
        search_kwargs["k"] = 3

    fetch_multiplier = 5 if use_reranker else 3
    retriever_search_kwargs = dict(search_kwargs)
    retriever_search_kwargs["fetch_k"] = search_kwargs.get("k", 3) * fetch_multiplier
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=retriever_search_kwargs
    )

    reranker_status = "disabled"
    retriever = base_retriever

    if use_reranker:
        try:
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            model_path = os.path.abspath(os.path.join(".", "bge-reranker-base-local"))
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Local reranker model not found at {model_path}")

            model = HuggingFaceCrossEncoder(model_name=model_path)
            compressor = CrossEncoderReranker(model=model, top_n=search_kwargs.get("k", 3))
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            reranker_status = f"active ({os.path.basename(model_path)})"
        except Exception as e:
            reranker_status = f"fallback to standard retriever: {e}"
            st.warning(f"AI re-ranker unavailable. Using standard retriever. Reason: {e}")

    st.session_state.reranker_status = reranker_status
    
    qa_system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    
    IMPORTANT: You may receive descriptions of images as context. Treat these descriptions as factual observations of the visual content.
    
    Use as much detail as necessary to answer the question comfortably. But answer the question to the point, do not include additional information just because it is there.
    Do NOT include "Time taken" or "Sources" in your answer. These are added automatically.
    
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt + "\n\nFormatting requirement:\n{response_style}"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    # Just a simple chain: Retriever -> QA
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
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
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embeddings = create_embedding_function(embed_model_name)

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
            llm = ChatOllama(model=local_model_name, temperature=0, base_url=OLLAMA_HOST)

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

def agentic_evaluate_rag(query, response, contexts, eval_model_selection, google_api_key=None):
    """
    Evaluates RAG output using a Judge LLM prompting approach (like G-Eval).
    """
    # Initialize the LLM
    llm = None
    if eval_model_selection.startswith("gemini:"):
        if not GOOGLE_GENAI_AVAILABLE:
            st.error("langchain-google-genai not installed.")
            return None
        if not google_api_key:
            st.error("Google API Key required for Gemini evaluation.")
            return None
        
        gemini_model_name = eval_model_selection.split(":")[1]
        
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
            safety_settings=safety_settings
        ) 
        
    else: # local
        from langchain_community.chat_models import ChatOllama
        local_model_name = eval_model_selection.split(":", 1)[1]
        llm = ChatOllama(model=local_model_name, temperature=0, base_url=OLLAMA_HOST)

    eval_prompt = """
You are an expert Teacher grading a Student's answer.

Question: {question}
Provided Context: {context}

Student Answer: {student_answer}

Task:
1. Based ONLY on the Provided Context, formulate your own standard, ideal answer.
2. Compare the Student Answer to your ideal answer. 
3. Does the Student Answer contain any hallucinations? Did it miss key details?
4. Assign a score from 1 to 10 based on accuracy and completeness.

Output your response in valid JSON format exactly as follows, enclosed in backticks:
```json
{{
    "ideal_answer": "...",
    "critique": "...",
    "score": 8
}}
```
"""
    
    prompt_template = ChatPromptTemplate.from_template(eval_prompt)
    chain = prompt_template | llm
    
    context_str = "\n\n".join(contexts) if contexts else "No context provided."
    
    try:
        result = chain.invoke({
            "question": query,
            "context": context_str,
            "student_answer": response
        })
        
        # Parse JSON output
        content = result.content
        
        # Clean up formatting if model wrapped it in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
             content = content.split("```")[1].split("```")[0].strip()
             
        try:
            parsed_data = json.loads(content)
            return parsed_data
        except json.JSONDecodeError:
            st.error("Failed to parse JSON from evaluator. Raw output:")
            st.code(content)
            return None
            
    except Exception as e:
        st.error(f"Agentic Evaluation Failed: {e}")
        return None

if DEEPEVAL_AVAILABLE:
    class CustomDeepEvalLLM(DeepEvalBaseLLM):
        def __init__(self, eval_model_selection, google_api_key=None):
            self.model_name = eval_model_selection
            
            if eval_model_selection.startswith("gemini:"):
                gemini_model_name = eval_model_selection.split(":")[1]
                from langchain_google_genai import ChatGoogleGenerativeAI
                import google.generativeai as genai
                
                safety_settings = {
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
                
                self.llm = ChatGoogleGenerativeAI(
                    model=gemini_model_name,
                    google_api_key=google_api_key,
                    temperature=0,
                    safety_settings=safety_settings
                )
            else:
                local_model_name = eval_model_selection.split(":", 1)[1]
                from langchain_community.chat_models import ChatOllama
                self.llm = ChatOllama(model=local_model_name, temperature=0, base_url=OLLAMA_HOST)

        def load_model(self):
            return self.llm

        def generate(self, prompt: str) -> str:
            response = self.llm.invoke(prompt)
            return response.content

        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)

        def get_model_name(self):
            return self.model_name

def deepeval_evaluate_rag(
    query, 
    response, 
    contexts, 
    eval_model_selection, 
    google_api_key=None,
    run_faithfulness=True,
    run_answer_relevancy=True,
    run_contextual_precision=False,
    run_contextual_recall=False,
    run_contextual_relevancy=False
):
    if not DEEPEVAL_AVAILABLE:
        st.error(f"DeepEval not available. Error: {DEEPEVAL_ERROR}")
        return None
        
    try:
        custom_llm = CustomDeepEvalLLM(eval_model_selection, google_api_key)
        
        # Format the test case
        # Note: Summarization requires input (source docs) and actual_output (summary)
        # DeepEval Contextual metrics require retrieval_context AND expected_output ideally, 
        # but can work ref-free depending on the metric. Here we use the LLM's response as the reference
        # for expected_output to allow ContextualPrecision and ContextualRecall to run.
        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=response,
            retrieval_context=contexts
        )
        
        metrics_results = {}
        
        # Determine if we should run the summarization metric
        is_summary_task = any(word in query.lower() for word in ["summarize", "summary", "elaborate", "explain"])
        
        # Define the metrics we want to run
        # Note: async_mode=False is crucial here because Streamlit runs its own asyncio event loop.
        # DeepEval's default async execution will conflict with Streamlit's loop and cause the app to hang indefinitely.
        # We also set strict_mode=False where applicable to suppress excess stdout glitches (rotating text) in Streamlit's terminal capture.
        metrics_to_run = []
        if run_faithfulness:
            metrics_to_run.append(("faithfulness", FaithfulnessMetric(threshold=0.5, model=custom_llm, include_reason=True, async_mode=False, strict_mode=False)))
        if run_answer_relevancy:
            metrics_to_run.append(("answer_relevancy", AnswerRelevancyMetric(threshold=0.5, model=custom_llm, include_reason=True, async_mode=False, strict_mode=False)))
        if run_contextual_precision:
            metrics_to_run.append(("contextual_precision", ContextualPrecisionMetric(threshold=0.5, model=custom_llm, include_reason=True, async_mode=False, strict_mode=False)))
        if run_contextual_recall:
            metrics_to_run.append(("contextual_recall", ContextualRecallMetric(threshold=0.5, model=custom_llm, include_reason=True, async_mode=False, strict_mode=False)))
        if run_contextual_relevancy:
            metrics_to_run.append(("contextual_relevancy", ContextualRelevancyMetric(threshold=0.5, model=custom_llm, include_reason=True, async_mode=False, strict_mode=False)))
        
        if is_summary_task:
            # Reconstruct test case specifically for Summarization 
            # (where 'input' needs to be the text being summarized, i.e., the contexts)
            summary_context_str = "\n".join(contexts) if contexts else "No context available."
            summary_test_case = LLMTestCase(
                input=summary_context_str,
                actual_output=response
            )
            metrics_to_run.append(
                ("summarization", SummarizationMetric(threshold=0.5, model=custom_llm, assessment_questions=["Does the summary accurately reflect the source material?"], async_mode=False, strict_mode=False))
            )

        if not metrics_to_run and not is_summary_task:
            st.warning("No metrics selected to run.")
            return {}

        for name, metric in metrics_to_run:
            try:
                if name == "summarization":
                    metric.measure(summary_test_case)
                else:
                    metric.measure(test_case)
                
                metrics_results[name] = {
                    "score": metric.score,
                    "reason": metric.reason,
                    "is_successful": metric.is_successful
                }
            except Exception as e:
                metrics_results[name] = {"score": 0.0, "reason": f"Metric failed to compute: {str(e)}", "is_successful": False}
        
        return metrics_results
    except Exception as e:
        st.error(f"DeepEval failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# --- Authentication & User Management ---
import auth

def login_page():
    st.title("🔒 Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if auth.login_user(username, password):
                    st.session_state.auth_status = True
                    st.session_state.username = username
                    st.session_state.user_id = username 
                    st.success("Login successful!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            register_submitted = st.form_submit_button("Register")
            
            if register_submitted:
                if new_pass != confirm_pass:
                    st.error("Passwords do not match")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success, msg = auth.create_user(new_user, new_pass)
                    if success:
                        st.success(msg + " Please login.")
                    else:
                        st.error(msg)

def logout():
    st.session_state.auth_status = False
    st.session_state.username = None
    st.session_state.user_id = None
    st.session_state.vectorstore = None # Clear loaded DB
    st.session_state.semantic_cache = None
    st.session_state.chat_history = [] # Clear history
    st.session_state.unique_sources = [] # Clear source metadata
    st.session_state.tabular_data = {}
    st.rerun()

# --- Main App Logic ---

def main():
    st.set_page_config(page_title="Multimodal RAG", page_icon="👁️", layout="wide")
    
    # Initialize Auth State
    if "auth_status" not in st.session_state:
        st.session_state.auth_status = False
    if "username" not in st.session_state:
        st.session_state.username = None

    # Show Login if not authenticated
    if not st.session_state.auth_status:
        login_page()
        return

    # --- Authenticated App ---
    
    st.title(f"👁️ Multimodal RAG Chatbot")
    
    # Initialize Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "models" not in st.session_state:
        st.session_state.models = get_ollama_models()
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = ""
        
    if "semantic_cache" not in st.session_state:
        st.session_state.semantic_cache = None
    
    if "unique_sources" not in st.session_state:
        st.session_state.unique_sources = []

    if "processing_time" not in st.session_state:
        st.session_state.processing_time = 0
    if "eval_model" not in st.session_state:
        st.session_state.eval_model = "local:ministral-3b"
    if "google_api_key" not in st.session_state:
        st.session_state.google_api_key = ""
    if "last_rag_data" not in st.session_state:
        st.session_state.last_rag_data = None
    if "processed_images" not in st.session_state:
        st.session_state.processed_images = []
    if "tabular_data" not in st.session_state:
        st.session_state.tabular_data = {}
    if "use_reranker" not in st.session_state:
        st.session_state.use_reranker = False
    if "reranker_status" not in st.session_state:
        st.session_state.reranker_status = "disabled"

    # User Data Directory
    user_data_dir = os.path.join("user_data", st.session_state.username)
    os.makedirs(user_data_dir, exist_ok=True)

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.write(f"👤 **Logged in as:** {st.session_state.username}")
        if st.button("Logout", type="secondary"):
            logout()
        st.divider()
        
        st.header("📚 Document Base")
        
        # Knowledge Base Management
        # DEFAULT KB Path is now USER SPECIFIC
        kb_name = st.text_input("Knowledge Base Name", value="vision_kb", help="Change this to switch between different document sets.")
        
        # KEY CHANGE: Path is now isolated per user
        db_path = os.path.join(user_data_dir, f"chroma_db_{kb_name.strip()}")
        cache_path = os.path.join(user_data_dir, f"chroma_cache_{kb_name.strip()}")
        debug_images_path = os.path.join(user_data_dir, f"debug_images_{kb_name.strip()}.json")
        tabular_data_path = os.path.join(user_data_dir, f"tabular_data_{kb_name.strip()}.pkl")
        kb_log_path = os.path.join(user_data_dir, f"kb_log_{kb_name.strip()}.jsonl")
        
        if "current_kb" not in st.session_state:
            st.session_state.current_kb = kb_name
        
        # Detect change in KB
        if st.session_state.current_kb != kb_name:
            st.session_state.current_kb = kb_name
            st.session_state.vectorstore = None # Reset loaded DB on switch
            st.session_state.unique_sources = [] # Reset cache
            st.session_state.tabular_data = {}
            st.rerun()

        if os.path.exists(db_path):
            if st.button("🗑️ Delete Knowledge Base", type="secondary"):
                try:
                    # Release references before deletion to avoid WinError 32 file lock
                    if st.session_state.vectorstore:
                        try:
                            # Langchain wrapped chroma
                            st.session_state.vectorstore.delete_collection()
                        except:
                            pass
                        try:
                            # Give chroma a hard instruction to release sqlite handles
                            st.session_state.vectorstore._client.clear_system_cache()
                        except:
                            pass
                        
                    st.session_state.vectorstore = None
                    st.session_state.semantic_cache = None
                    import gc
                    gc.collect()
                    time.sleep(0.5) # Allow OS to release locks
                    
                    if os.path.exists(db_path):
                        shutil.rmtree(db_path)
                    if os.path.exists(cache_path):
                        shutil.rmtree(cache_path)
                    if os.path.exists(debug_images_path):
                        os.remove(debug_images_path)
                    if os.path.exists(tabular_data_path):
                        os.remove(tabular_data_path)
                    if os.path.exists(kb_log_path):
                        os.remove(kb_log_path)
                        
                    st.session_state.unique_sources = [] # Reset cache
                    st.session_state.tabular_data = {}
                    st.session_state.processed_images = [] # Clear debugging images
                    st.success(f"Deleted KB: {kb_name}")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting: {e}")
        
        st.divider()
        
        # New Chat Button
        if st.button("🆕 New Chat", type="primary", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.unique_sources = []
            st.session_state.processing_time = 0
            st.session_state.processed_images = [] # Clear debugging images
            
            # Clear Temporary Session Image
            st.session_state.temp_image_desc = None
            if "current_image_id" in st.session_state:
                del st.session_state.current_image_id
                
            st.rerun()

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
        ollama_models = st.session_state.models if st.session_state.models else []
        fixed_gemma4_model = resolve_available_model(
            ollama_models,
            ["gemma4"],
            FIXED_GEMMA4_MODEL
        )
        selected_model = fixed_gemma4_model
        selected_ocr_model = fixed_gemma4_model
        selected_vlm_model = fixed_gemma4_model

        st.info(
            f"This app version uses a fixed Gemma4 pipeline for chat, OCR, and VLM: `{fixed_gemma4_model}`"
        )
        if not any("gemma4" in model.lower() for model in ollama_models):
            st.warning(
                f"`{FIXED_GEMMA4_MODEL}` was not found in the currently discovered Ollama models. "
                "The app will still target that model tag, so make sure it is installed locally."
            )
        
        processing_mode = st.selectbox(
            "Scanned PDF Processing Mode",
            ["Text only", "VLM only", "OCR only", "OCR + VLM hybrid"],
            index=0,
            help="Choose how PDFs should be processed: native text only, text plus visual augmentation, OCR, or OCR plus VLM."
        )

        # Embedding Model Selection
        resolved_nomic_model = resolve_available_model(
            ollama_models,
            ["nomic-embed-text"],
            DEFAULT_NOMIC_EMBED_MODEL
        )
        embed_models = ["all-MiniLM-L6-v2", resolved_nomic_model]

        selected_embed_model = st.selectbox(
            "Select Embedding Model",
            embed_models,
            index=0,
            key="select_embedding_model"
        )
        
        # Retrieval Configuration
        st.subheader("Retrieval Configuration")
        target_k = st.slider(
            "Retrieval Chunks (Top-K)",
            min_value=2,
            max_value=25,
            value=5,
            help="Increase this for comprehensive summaries. More chunks give the LLM more context but cost more tokens."
        )
        use_reranker = st.checkbox(
            "Use AI Re-Ranker",
            value=st.session_state.use_reranker,
            help="Uses the local cross-encoder reranker for better chunk ordering. Slower, and requires sentence-transformers."
        )
        st.session_state.use_reranker = use_reranker
        st.caption(f"Re-ranker status: `{st.session_state.reranker_status}`")
        
        # Load Persistent DB on Start if invalid
        if st.session_state.vectorstore is None and os.path.exists(db_path):
            try:
                # Load config
                config = load_db_config(db_path)
                loaded_model = config.get("embedding_model", selected_embed_model)
                
                if loaded_model != selected_embed_model:
                     st.caption(f"ℹ️ Loaded KB using model: `{loaded_model}` (Configured)")
                
                embeddings = create_embedding_function(loaded_model, local_only=True)
                st.session_state.vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
                st.session_state.embed_model = loaded_model
                st.session_state.tabular_data = load_tabular_data(tabular_data_path)
                
                # Load/Init Cache
                st.session_state.semantic_cache = SemanticCache(
                    embedding_function=embeddings,
                    persist_directory=cache_path # User specific
                )
                
                # Load persistent processed images if available
                if os.path.exists(debug_images_path):
                    try:
                        with open(debug_images_path, "r") as f:
                            st.session_state.processed_images = json.load(f)
                    except Exception as e:
                        print(f"Failed to load debug images: {e}")
                        st.session_state.processed_images = []
                
                refresh_source_cache() # Cache logic
                st.toast(f"Loaded existing KB: {kb_name}")
            except Exception as e:
                st.session_state.vectorstore = None
                st.session_state.semantic_cache = None
                if "407" in str(e) or "authenticationrequired" in str(e).lower():
                    st.error(
                        "Failed to load existing DB because the embedding model tried to use a network path that requires authentication. "
                        "For `all-MiniLM-L6-v2`, ensure the model is cached locally before loading the KB."
                    )
                else:
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
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.error("Maximum 5 files allowed.")
            else:
                if st.button("Process Documents (with Vision)", type="primary"):
                    p_start = time.time()
                    ingestion_trace = [
                        {"step": "process_documents_clicked", "at": datetime.now().astimezone().isoformat()}
                    ]
                    st.toast(f"Processing started at {time.strftime('%H:%M:%S', time.localtime(p_start))}")
                    status_box = st.status(
                        f"Ingesting documents using OCR model '{selected_ocr_model}' and VLM model '{selected_vlm_model}'...",
                        expanded=True
                    )
                    progress_bar = st.progress(0)
                    progress_caption = st.empty()

                    def update_ingestion_ui(message, progress=None):
                        status_box.update(label=message, state="running", expanded=True)
                        status_box.write(message)
                        progress_caption.caption(message)
                        if progress is not None:
                            bounded = max(0, min(100, int(progress * 100)))
                            progress_bar.progress(bounded)

                    # 1. Load
                    st.session_state.processed_images = []
                    ingestion_trace.append({
                        "step": "load_documents_started",
                        "uploaded_files": [uploaded_file.name for uploaded_file in uploaded_files],
                        "processing_mode": processing_mode,
                        "ocr_model": selected_ocr_model,
                        "vlm_model": selected_vlm_model,
                        "embedding_model": selected_embed_model,
                        "chunking_strategy": chunking_strategy,
                        "chunk_size": chunk_size,
                    })
                    update_ingestion_ui("Preparing files for ingestion...", progress=0.02)
                    raw_docs, loaded_tabular_data = load_documents(
                        uploaded_files,
                        vlm_model=selected_vlm_model,
                        ocr_model=selected_ocr_model,
                        scanned_pdf_mode=processing_mode,
                        progress_callback=update_ingestion_ui
                    )
                    if not raw_docs:
                        status_box.update(label="Document loading failed.", state="error", expanded=True)
                        progress_bar.progress(100)
                        append_kb_log(
                            kb_log_path,
                            "ingestion_error",
                            {
                                "kb_name": kb_name,
                                "user": st.session_state.username,
                                "message": "Could not load documents.",
                                "uploaded_files": [uploaded_file.name for uploaded_file in uploaded_files],
                                "processing_mode": processing_mode,
                                "observable_reasoning_trace": ingestion_trace,
                                "processed_images_count": len(st.session_state.processed_images),
                            }
                        )
                        st.error("Could not load documents.")
                    else:
                        update_ingestion_ui(f"Loaded {len(raw_docs)} snippets. Preparing vector store...", progress=0.64)
                        st.success(f"Loaded {len(raw_docs)} snippets (Text + Image Descriptions).")
                        raw_doc_summary = summarize_documents_for_log(raw_docs)
                        ingestion_trace.append({
                            "step": "load_documents_completed",
                            "raw_document_summary": raw_doc_summary,
                            "tabular_sources": sorted(list(loaded_tabular_data.keys())),
                            "processed_images_count": len(st.session_state.processed_images),
                        })
                        st.session_state.unique_sources = sorted({
                            os.path.basename(doc.metadata["source"])
                            for doc in raw_docs
                            if isinstance(doc.metadata, dict) and doc.metadata.get("source")
                        })
                        
                        # Debug: Check content length
                        total_chars = sum(len(d.page_content) for d in raw_docs)
                        st.write(f"Debug: Total characters extracted: {total_chars}")
                        
                        # 2. Vector Store
                        embed_model = selected_embed_model
                        st.session_state.embed_model = embed_model
                        
                        st.info(f"Using embedding model: `{embed_model}`")
                        
                        
                        try:
                            vs, vector_db_stats = create_vector_db(
                                raw_docs,
                                embed_model,
                                chunking_strategy,
                                chunk_size,
                                db_path,
                                progress_callback=update_ingestion_ui
                            )
                            ingestion_trace.append({
                                "step": "vector_db_built",
                                "vector_db_stats": vector_db_stats,
                            })
                            
                            if vs:
                                st.session_state.vectorstore = vs
                                st.session_state.tabular_data = loaded_tabular_data
                                save_tabular_data(loaded_tabular_data, tabular_data_path)
                                refresh_source_cache() # Cache logic
                                
                                # Initialize Cache with same embeddings
                                st.session_state.semantic_cache = SemanticCache(
                                    embedding_function=vs.embeddings, # Use same embeddings as main DB
                                    persist_directory=cache_path # User specific
                                )
                                
                                # Persist Debug Images
                                try:
                                    with open(debug_images_path, "w") as f:
                                        json.dump(st.session_state.processed_images, f)
                                except Exception as e:
                                    st.error(f"Could not persist debug images: {e}")
                                
                                # Force fresh clients to be created on the next rerun.
                                # This avoids stale embedding/vectorstore handles, especially for local HF models.
                                st.session_state.vectorstore = None
                                st.session_state.semantic_cache = None
                                p_end = time.time()
                                st.session_state.processing_time = p_end - p_start
                                append_kb_log(
                                    kb_log_path,
                                    "ingestion",
                                    {
                                        "kb_name": kb_name,
                                        "user": st.session_state.username,
                                        "processing_time_seconds": st.session_state.processing_time,
                                        "uploaded_files": [uploaded_file.name for uploaded_file in uploaded_files],
                                        "processing_mode": processing_mode,
                                        "ocr_model": selected_ocr_model,
                                        "vlm_model": selected_vlm_model,
                                        "embedding_model": selected_embed_model,
                                        "chunking_strategy": chunking_strategy,
                                        "chunk_size": chunk_size,
                                        "raw_document_summary": raw_doc_summary,
                                        "vector_db_stats": vector_db_stats,
                                        "tabular_sources": sorted(list(loaded_tabular_data.keys())),
                                        "processed_images_count": len(st.session_state.processed_images),
                                        "processed_images_preview": st.session_state.processed_images[:5],
                                        "llm_hidden_chain_of_thought_captured": False,
                                        "observable_reasoning_trace": ingestion_trace,
                                    }
                                )
                                progress_bar.progress(100)
                                progress_caption.caption("Ingestion complete.")
                                status_box.update(label="Knowledge base is ready.", state="complete", expanded=True)
                                st.success("Vector Database and Cache Ready! You can now chat.")
                                time.sleep(1) # Give user a moment to see success message
                                st.rerun()
                            else:
                                status_box.update(label="Vector database creation failed.", state="error", expanded=True)
                                progress_bar.progress(100)
                                append_kb_log(
                                    kb_log_path,
                                    "ingestion_error",
                                    {
                                        "kb_name": kb_name,
                                        "user": st.session_state.username,
                                        "message": "Failed to create Vector Database. No chunks generated.",
                                        "uploaded_files": [uploaded_file.name for uploaded_file in uploaded_files],
                                        "processing_mode": processing_mode,
                                        "vector_db_stats": vector_db_stats,
                                        "observable_reasoning_trace": ingestion_trace,
                                    }
                                )
                                st.error("Failed to create Vector Database. No chunks generated.")
                        except Exception as e:
                            status_box.update(label="Vector database creation raised an error.", state="error", expanded=True)
                            progress_bar.progress(100)
                            append_kb_log(
                                kb_log_path,
                                "ingestion_error",
                                {
                                    "kb_name": kb_name,
                                    "user": st.session_state.username,
                                    "message": str(e),
                                    "uploaded_files": [uploaded_file.name for uploaded_file in uploaded_files],
                                    "processing_mode": processing_mode,
                                    "observable_reasoning_trace": ingestion_trace,
                                }
                            )
                            st.error(f"Error creating vector DB: {e}")

        st.sidebar.divider()
        st.sidebar.subheader("🖼️ Temporary Session Image")
        st.sidebar.caption("Upload an image for this chat only. Clears on 'New Chat'.")
        
        uploaded_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="session_img")
        
        if uploaded_image:
            # Check if this is a new image or already processed
            # We use file validation by name/size
            img_id = f"{uploaded_image.name}_{uploaded_image.size}"
            
            if "current_image_id" not in st.session_state or st.session_state.current_image_id != img_id:
                # Process new image
                st.session_state.current_image_id = img_id
                st.session_state.temp_image_desc = None # Reset
                
                # Display
                st.sidebar.image(uploaded_image, caption="Session Image", use_column_width=True)
                
                with st.sidebar.status(f"Analyzing image with {selected_vlm_model}...", expanded=True):
                    img_bytes = uploaded_image.getvalue()
                    desc = generate_image_description(img_bytes, model_name=selected_vlm_model)
                    st.session_state.temp_image_desc = desc
                    st.success("Image analyzed!")
            else:
                 # Already processed, just display
                 st.sidebar.image(uploaded_image, caption="Session Image", use_column_width=True)
                 if "temp_image_desc" in st.session_state and st.session_state.temp_image_desc:
                     if st.session_state.temp_image_desc.startswith("Error"):
                         st.sidebar.error(st.session_state.temp_image_desc)
                     else:
                         with st.sidebar.expander("See Analysis"):
                             st.write(st.session_state.temp_image_desc)

        # --- Debugging UI for Image Extraction ---
        st.sidebar.divider()
        with st.sidebar.expander("OCR Debug: Image Extraction & Descriptions", expanded=True):
            processed = st.session_state.get("processed_images", [])
            st.caption(f"Processed image entries: {len(processed)}")
            if not processed:
                st.warning("No processed images were captured in this session.")
            else:
                for idx, img_data in enumerate(processed):
                    st.markdown(f"**Image {idx+1} from {img_data['source']} (Page {img_data['page']})**")
                    col_img, col_desc = st.columns([1, 2])

                    with col_img:
                        if os.path.exists(img_data['image_path']):
                            st.image(img_data['image_path'], use_column_width=True)
                        else:
                            st.error(f"Image file not found: {img_data['image_path']}")

                    with col_desc:
                        st.markdown("**Generated Description / OCR Output:**")
                        st.text_area(
                            f"OCR Output {idx+1}",
                            value=img_data['description'],
                            height=250,
                            key=f"ocr_text_{idx}"
                        )

                        if st.button("Show Embedding Array", key=f"ocr_btn_emb_{idx}"):
                            if st.session_state.vectorstore:
                                with st.spinner("Generating text embeddings..."):
                                    try:
                                        emb = st.session_state.vectorstore.embeddings.embed_query(img_data['description'])
                                        st.caption(f"**Dimensions:** `{len(emb)}`")
                                        formatted_emb = ", ".join(f"{x:.4f}" for x in emb[:10])
                                        st.code(f"[{formatted_emb}, ...]", language="json")
                                    except Exception as e:
                                        st.error(f"Error getting embedding: {e}")
                            else:
                                st.warning("Please verify your knowledge base is loaded!")
                    st.divider()
        if False:
            st.sidebar.divider()
            with st.sidebar.expander("🐛 Debug: Image Extraction & Descriptions", expanded=False):
                st.caption(f"Showing {len(st.session_state.processed_images)} extracted images and their generated text descriptions.")
                for idx, img_data in enumerate(st.session_state.processed_images):
                    st.markdown(f"**Image {idx+1} from {img_data['source']} (Page {img_data['page']})**")
                    col_img, col_desc = st.columns([1, 2])
                    
                    with col_img:
                        if os.path.exists(img_data['image_path']):
                            st.image(img_data['image_path'], use_column_width=True)
                        else:
                            st.error("Image file not found.")
                            
                    with col_desc:
                        st.markdown("**Generated Description:**")
                        st.info(img_data['description'])
                        
                        # Show Embedding toggle button
                        if st.button("👁️ Show Embedding Array", key=f"btn_emb_{idx}"):
                            if st.session_state.vectorstore:
                                with st.spinner("Generating text embeddings..."):
                                    try:
                                        # Use the current vector DB's embedding function
                                        emb = st.session_state.vectorstore.embeddings.embed_query(img_data['description'])
                                        st.caption(f"**Dimensions:** `{len(emb)}`")
                                        # Show first 10 numbers instead of the whole thing
                                        formatted_emb = ", ".join(f"{x:.4f}" for x in emb[:10])
                                        st.code(f"[{formatted_emb}, ...]", language="json")
                                    except Exception as e:
                                        st.error(f"Error getting embedding: {e}")
                            else:
                                st.warning("Please verify your knowledge base is loaded!")
                    st.divider()

        
        st.sidebar.divider()
        st.sidebar.markdown(f"**Status:** {'🟢 Vector DB Ready' if st.session_state.vectorstore else '🔴 Not Processed'}")
        if st.session_state.vectorstore:
            # Debug check
            st.sidebar.caption(f"DB initialized with model: {st.session_state.embed_model}")
            if st.session_state.processing_time > 0:
                st.sidebar.caption(f"Processing Time: {st.session_state.processing_time:.2f} seconds")

    # --- Chat Interface ---
    
    # Display History
    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
                
                # Button Column
                col_actions, _ = st.columns([2, 5])
                with col_actions:
                    # Save Button (Download)
                    save_filename = f"response_{i}.txt"
                    # Try to get prompt context for better filename
                    if i > 0 and isinstance(st.session_state.chat_history[i-1], HumanMessage):
                        prompt_preview = st.session_state.chat_history[i-1].content
                        safe_prompt = "".join([c for c in prompt_preview if c.isalnum() or c in (' ','-','_')]).strip()
                        if safe_prompt:
                            save_filename = f"response_{safe_prompt[:15].replace(' ', '_')}.txt"
                    
                    st.download_button(
                        label="💾 Save", 
                        data=message.content,
                        file_name=save_filename,
                        mime="text/plain",
                        key=f"save_hist_{i}",
                        help="Download this response"
                    )
                    
                    # Clear Cache Button
                    cache_id = message.additional_kwargs.get("cache_id")
                    if cache_id:
                        if st.button("🧹 Clear Cache", key=f"clear_{i}", help="Remove this specific response from cache."):
                            if st.session_state.semantic_cache:
                                deleted = st.session_state.semantic_cache.delete_by_id(cache_id)
                                if deleted:
                                    st.success("Cache cleared!")
                                    # Remove ID so button disappears
                                    message.additional_kwargs["cache_id"] = None 
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.warning("Already deleted or not found.")

    # Chat Input
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Display User Message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate Response
        if st.session_state.vectorstore is None and not st.session_state.get("temp_image_desc"):
            with st.chat_message("assistant"):
                st.warning("Please upload and process documents first to enable RAG, or upload a temporary image.")
        else:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    try:
                        # Logic to handle @ referencing
                        clean_input = user_input
                        search_kwargs = {}
                        query_trace = [
                            {
                                "step": "query_received",
                                "user_input": user_input,
                                "at": datetime.now().astimezone().isoformat(),
                            }
                        ]
                        matched_source = None
                        grounding_check = None
                        
                        # Inject Temporary Image Context
                        if "temp_image_desc" in st.session_state and st.session_state.temp_image_desc:
                            # We prepend the image description so the LLM knows about it
                            img_context = f"\n[CONTEXT: The user has uploaded an image. Here is its description: {st.session_state.temp_image_desc}]\n"
                            # We don't modify clean_input yet because that's for cache lookup.
                            # But we should modify user_input or the subsequent 'standalone_question' logic.
                            # Actually, let's append it to the history processing or standalone question prompt?
                            # Simplest: Append to input for reformulation, but maybe keep clean_input clean for cache?
                            # Strategy: Append to 'clean_input' so it becomes part of the question for RAG and Cache.
                            clean_input = f"{clean_input}\n{img_context}"
                            st.caption("🖼️ Using temporary image context.")
                            query_trace.append({
                                "step": "temporary_image_context_applied",
                                "description_length": len(st.session_state.temp_image_desc),
                            })
                        
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
                                            query_trace.append({
                                                "step": "source_filter_applied",
                                                "matched_source": matched_source,
                                                "raw_reference": word,
                                            })
                                            break
                            except Exception as e:
                                st.error(f"Error parsing Reference: {e}")
                                query_trace.append({
                                    "step": "source_filter_error",
                                    "error": str(e),
                                })

                        # --- Semantic Cache & RAG Logic ---
                        
                        start_time = time.time()
                        query_trace.append({
                            "step": "query_pipeline_started",
                            "clean_input": clean_input,
                        })
                        
                        # Intent Detection for dynamic K scaling
                        lower_input = clean_input.lower()
                        is_summary_task = any(word in lower_input for word in ["summarize", "summary", "elaborate", "explain", "detailed", "all"])
                        
                        # Apply slider K or boosted K
                        dynamic_k = target_k
                        if is_summary_task:
                            dynamic_k = max(target_k, 15)
                            st.caption(f"🚀 **Summary Intent Detected!** Temporarily boosting retrieval chunks to `{dynamic_k}`.")
                            query_trace.append({
                                "step": "summary_intent_detected",
                                "target_k": target_k,
                                "dynamic_k": dynamic_k,
                            })
                            
                        search_kwargs["k"] = dynamic_k
                        
                        cached_answer = None
                        cache_distance = 0.0
                        standalone_question = clean_input # Default
                        cache_id = None
                        is_tabular_query = is_tabular_analytics_question(clean_input, st.session_state.tabular_data)
                        query_trace.append({
                            "step": "query_type_classified",
                            "is_tabular_query": is_tabular_query,
                            "search_kwargs": dict(search_kwargs),
                        })
                        
                        # 0. Optimistic Lookup Check (Bypass reformulation drift for explicit Qs)
                        optimistic_hit = False
                        if is_tabular_query:
                            st.caption("Using deterministic tabular analysis for this question.")
                            query_trace.append({
                                "step": "tabular_analytics_selected",
                            })
                        elif st.session_state.semantic_cache and len(clean_input) > 15:
                            # 15 chars is roughly 3-4 words. e.g. "What is RTA?"
                            # Use relaxed threshold (same as standard) because even reformulated questions
                            # in cache should be semantically close to raw input.
                            cached_answer, cache_distance, cache_id = st.session_state.semantic_cache.lookup(clean_input, threshold=0.80)
                            query_trace.append({
                                "step": "semantic_cache_optimistic_lookup",
                                "threshold": 0.80,
                                "distance": cache_distance,
                                "hit": bool(cached_answer),
                            })
                            if cached_answer:
                                optimistic_hit = True
                        
                        if not cached_answer:
                        
                            # 1. Reformulate Question (Contextualize)
                            # We do this FIRST to check cache against the "real" question.
                            cleaned_history = clean_history(st.session_state.chat_history)
                            standalone_question = get_standalone_question(clean_input, cleaned_history, selected_model)
                            
                            st.caption(f"🧠 Understanding: `{standalone_question}`")
                            query_trace.append({
                                "step": "standalone_question_created",
                                "standalone_question": standalone_question,
                            })
                            
                            # 2. Check Cache (Normal)
                            if is_tabular_query:
                                cached_answer = answer_tabular_analytics(standalone_question, st.session_state.tabular_data)
                                query_trace.append({
                                    "step": "tabular_analytics_answered",
                                    "hit": bool(cached_answer),
                                })
                            elif st.session_state.semantic_cache:
                                cached_answer, cache_distance, cache_id = st.session_state.semantic_cache.lookup(standalone_question, threshold=0.80)
                                query_trace.append({
                                    "step": "semantic_cache_lookup",
                                    "threshold": 0.80,
                                    "distance": cache_distance,
                                    "hit": bool(cached_answer),
                                })
                        
                        # Debug Info
                        with st.expander("🧠 Cache Debug Info"):
                            if optimistic_hit:
                                st.write("**Optimistic Hit!** (Bypassed Reformulation)")
                                st.write(f"**Input:** `{clean_input}`")
                            else:
                                st.write(f"**Standalone Question:** `{standalone_question}`")
                            
                            st.write(f"**Cache Distance:** {cache_distance:.4f} (Threshold: {'< 0.05' if optimistic_hit else '< 0.20'})")
                            
                            if cached_answer:
                                st.success("Hit! ⚡")
                            else:
                                st.warning("Miss")
                            
                        if cached_answer:
                            # --- Optional safer semantic-cache replay path (currently disabled) ---
                            # If you enable `lookup_with_context(...)` above, you can avoid returning
                            # the cached answer verbatim and instead re-answer the current question
                            # from the cached retrieval chunks.
                            #
                            # Example flow:
                            # cached_answer, cached_contexts, cache_distance, cache_id = (
                            #     st.session_state.semantic_cache.lookup_with_context(
                            #         standalone_question,
                            #         threshold=0.80
                            #     )
                            # )
                            #
                            # if cached_answer and cached_contexts:
                            #     llm = ChatOllama(model=selected_model, temperature=0.3, base_url=OLLAMA_HOST)
                            #     qa_system_prompt = (
                            #         "You are an assistant for question-answering tasks. "
                            #         "Use the provided cached context to answer the current question. "
                            #         "If the cached context is insufficient, say so clearly."
                            #     )
                            #     qa_prompt = ChatPromptTemplate.from_messages([
                            #         ("system", qa_system_prompt),
                            #         ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
                            #     ])
                            #     chain = qa_prompt | llm | StrOutputParser()
                            #     regenerated_answer = chain.invoke({
                            #         "context": "\n\n".join(cached_contexts),
                            #         "question": standalone_question
                            #     })
                            #     response_text = regenerated_answer
                            # else:
                            #     response_text = cached_answer

                            response_text = cached_answer
                            end_time = time.time()
                            
                            # Calculate time taken
                            execution_time = end_time - start_time
                            minutes = int(execution_time // 60)
                            seconds = int(execution_time % 60)
                            timing_str = f"{minutes} min {seconds} sec"
                            if is_tabular_query:
                                st.session_state.last_rag_data = None
                            
                            response_text += f"\n\n⚡ **Cached Response** | **Time taken:** {timing_str}"
                            query_trace.append({
                                "step": "cached_answer_returned",
                                "cache_id": cache_id,
                                "cache_distance": cache_distance,
                                "execution_time_seconds": execution_time,
                                "used_tabular_logic": is_tabular_query,
                            })
                            append_kb_log(
                                kb_log_path,
                                "query",
                                {
                                    "kb_name": kb_name,
                                    "user": st.session_state.username,
                                    "query_time_seconds": execution_time,
                                    "question": user_input,
                                    "clean_input": clean_input,
                                    "standalone_question": standalone_question,
                                    "answer": response_text,
                                    "semantic_cache_hit": True,
                                    "semantic_cache_id": cache_id,
                                    "semantic_cache_distance": cache_distance,
                                    "used_tabular_logic": is_tabular_query,
                                    "used_rag": False,
                                    "used_direct_image_mode": False,
                                    "source_filter": matched_source,
                                    "search_kwargs": search_kwargs,
                                    "retrieved_chunks": [],
                                    "retrieved_chunk_count": 0,
                                    "grounding_check": None,
                                    "response_style_instruction": None,
                                    "observable_reasoning_trace": query_trace,
                                    "llm_hidden_chain_of_thought_captured": False,
                                    "validation_note": "This log captures observable reasoning artifacts and retrieved evidence, not hidden model chain-of-thought.",
                                }
                            )
                            
                            message_placeholder.markdown(response_text)

                            # Update History
                            st.session_state.chat_history.append(HumanMessage(content=user_input))
                            st.session_state.chat_history.append(AIMessage(
                                content=response_text,
                                additional_kwargs={
                                    "is_cached": True,
                                    "cache_id": cache_id
                                }
                            ))
                            # Save Button Logic (for cached response)
                            filename_base = "cached_response"
                            safe_prompt = "".join([c for c in user_input if c.isalnum() or c in (' ','-','_')]).strip()
                            prompt_segment = safe_prompt[:7].replace(" ", "_")
                            save_filename = f"{filename_base}_{prompt_segment}.txt"
                            
                            st.download_button(
                                label="💾 Save Response",
                                data=response_text,
                                file_name=save_filename,
                                mime="text/plain"
                            )
                            st.rerun() # Force UI refresh to show Clear Cache button immediately

                        else:
                            # 3. Cache Miss - Run RAG or Direct Chat
                            is_cached = False
                            should_cache_answer = True
                            response_style = get_response_style_instruction(user_input)
                            query_trace.append({
                                "step": "response_style_selected",
                                "response_style_instruction": response_style,
                            })
                            
                            # --- A. Standard RAG (documents loaded) ---
                            if st.session_state.vectorstore:
                                grounding_check = check_retrieval_grounding(
                                    st.session_state.vectorstore,
                                    standalone_question,
                                    search_kwargs=search_kwargs
                                )
                                query_trace.append({
                                    "step": "grounding_check_completed",
                                    "allow_answer": grounding_check["allow_answer"],
                                    "top_score": grounding_check["top_score"],
                                    "avg_score": grounding_check["avg_score"],
                                    "reason": grounding_check["reason"],
                                })

                                with st.expander("Grounding Check", expanded=False):
                                    st.write(f"**Top relevance score:** `{grounding_check['top_score']}`")
                                    st.write(f"**Average relevance score:** `{grounding_check['avg_score']}`")
                                    st.write(f"**Decision:** `{'allow' if grounding_check['allow_answer'] else 'refuse'}`")
                                    st.write(f"**Reason:** {grounding_check['reason']}")

                                if not grounding_check["allow_answer"]:
                                    full_answer = (
                                        "I could not find enough relevant information in this knowledge base "
                                        "to answer that confidently."
                                    )
                                    context = []
                                    should_cache_answer = False
                                    message_placeholder.markdown(full_answer)
                                    response = {
                                        "answer": full_answer,
                                        "context": context
                                    }
                                else:
                                # Check if we can reuse the cached chain (Standard Chain now)
                                    current_chain_meta = {
                                        "model": selected_model,
                                        "vectorstore_id": id(st.session_state.vectorstore),
                                        "search_kwargs": str(search_kwargs),
                                        "use_reranker": st.session_state.get("use_reranker", False)
                                    }
                                    
                                    if ("rag_chain" not in st.session_state or 
                                        "rag_chain_meta" not in st.session_state or 
                                        st.session_state.rag_chain_meta != current_chain_meta):
                                        
                                        st.caption("⚙️ Rebuilding RAG Chain...")
                                        st.session_state.rag_chain = get_rag_chain_standard(
                                            st.session_state.vectorstore, 
                                            selected_model, 
                                            search_kwargs=search_kwargs,
                                            use_reranker=st.session_state.get("use_reranker", False)
                                        )
                                        st.session_state.rag_chain_meta = current_chain_meta
                                    
                                    chain = st.session_state.rag_chain
                                    
                                    # Invoke chain using the STANDALONE question with Streaming
                                    # We don't pass history here because we already resolved it!
                                    
                                    full_answer = ""
                                    context = []
                                    
                                    # Stream the response
                                    for chunk in chain.stream({
                                        "input": standalone_question,
                                        "response_style": response_style
                                    }):
                                        if "answer" in chunk:
                                            full_answer += chunk["answer"]
                                            message_placeholder.markdown(full_answer + "▌")
                                        
                                        if "context" in chunk:
                                            context = chunk["context"]
        
                                    # Final update to remove cursor
                                    message_placeholder.markdown(full_answer)
                                    
                                    # Reconstruct response object for downstream logic
                                    response = {
                                        "answer": full_answer,
                                        "context": context
                                    }
                            
                            # --- B. Direct Chat (Image Only, no documents) ---
                            else:
                                st.caption("📸 Using Temporary Image Context (Direct Mode)")
                                # Use a simple chain
                                llm = ChatOllama(model=selected_model, temperature=0.7, base_url=OLLAMA_HOST)
                                # Simple prompt that includes history implicitly via standalone_question
                                prompt = ChatPromptTemplate.from_messages([
                                    ("system", "You are a helpful assistant. Answer the user's question in detail.\n\nFormatting requirement:\n{response_style}"),
                                    ("human", "{input}"),
                                ])
                                chain = prompt | llm | StrOutputParser()
                                
                                full_answer = ""
                                for chunk in chain.stream({
                                    "input": standalone_question,
                                    "response_style": response_style
                                }):
                                    full_answer += chunk
                                    message_placeholder.markdown(full_answer + "▌")
                                
                                message_placeholder.markdown(full_answer)
                                
                                response = {
                                    "answer": full_answer,
                                    "context": [] # No retrieved docs
                                }


                            
                            # 4. Update Cache
                            new_cache_id = None
                            if st.session_state.semantic_cache and should_cache_answer:
                                # Current live behavior stores only question -> answer.
                                new_cache_id = st.session_state.semantic_cache.add(standalone_question, full_answer)
                                #
                                # Optional safer behavior (currently disabled):
                                # store the retrieved chunks too, so a future semantically similar
                                # question can be re-answered from cached context instead of
                                # returning the old answer verbatim.
                                #
                                # if (
                                #     st.session_state.semantic_cache and
                                #     should_cache_answer and
                                #     'context' in response and response['context']
                                # ):
                                #     cached_contexts = [doc.page_content for doc in response['context']]
                                #     new_cache_id = st.session_state.semantic_cache.add_with_context(
                                #         standalone_question,
                                #         full_answer,
                                #         cached_contexts
                                #     )

                            # Store for evaluation
                            if 'context' in response and response['context']:
                                context_text = [doc.page_content for doc in response['context']]
                                st.session_state.last_rag_data = {
                                    'question': standalone_question,
                                    'answer': response['answer'],
                                    'contexts': context_text,
                                    'chat_model': selected_model,
                                    'embed_model': selected_embed_model
                                }
                            else:
                                st.session_state.last_rag_data = None

                            end_time = time.time()
                            
                            answer = response['answer']
                            
                            # Calculate time taken
                            execution_time = end_time - start_time
                            minutes = int(execution_time // 60)
                            seconds = int(execution_time % 60)
                            timing_str = f"{minutes} min {seconds} sec"
                            
                            if is_cached:
                                answer += f"\n\n⚡ **Cached Response** | **Time taken:** {timing_str}"
                            else:
                                answer += f"\n\n**Time taken:** {timing_str}"
                            
                            # Extract and format sources
                            source_names = []
                            referenced_images = []
                            if 'context' in response:
                                source_map = {} # {filename: {pages}}
                                for doc in response['context']:
                                    src = doc.metadata.get('source', 'Unknown')
                                    src = os.path.basename(src)
                                    
                                    # Handle Page Numbers
                                    # PyPDFLoader uses 'page' (0-indexed)
                                    # Our Image Loader uses 'page' (1-indexed)
                                    page = doc.metadata.get('page')
                                    
                                    if src not in source_map:
                                        source_map[src] = set()
                                        
                                    if page is not None:
                                        # Try to normalize to 1-indexed for display
                                        # If it's a standard int (likely PyPDFLoader 0-indexed), add 1
                                        # If it's our custom image loader, it might already be 1-indexed?
                                        # Let's check the type or source. 
                                        # Custom image loader sets type='image'.
                                        if doc.metadata.get('type') == 'image':
                                            # Our custom code sat page=page_num+1, so it's already 1-indexed
                                            source_map[src].add(page)
                                        else:
                                            # Likely PyPDFLoader (0-indexed)
                                            try:
                                                source_map[src].add(int(page) + 1)
                                            except:
                                                # If page is not an int, just add raw
                                                source_map[src].add(page)
                                    
                                    # Check for image type for referenced_images list
                                    if doc.metadata.get('type') == 'image':
                                        referenced_images.append(doc.metadata.get('image_path'))
                                
                                formatted_sources = []
                                for src in sorted(source_map.keys()):
                                    pages = source_map[src]
                                    if pages:
                                        # Sort pages
                                        try:
                                            sorted_pages = sorted(pages, key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else str(x))
                                            page_str = ", ".join(str(p) for p in sorted_pages)
                                            formatted_sources.append(f"{src} (Pages: {page_str})")
                                        except:
                                            formatted_sources.append(src)
                                    else:
                                        formatted_sources.append(src)
                                        
                                if formatted_sources:
                                    source_names = formatted_sources
                                    answer += f"\n\n**Sources:** {'; '.join(formatted_sources)}"

                            retrieved_chunks_log = []
                            if 'context' in response and response['context']:
                                retrieved_chunks_log = serialize_documents_for_log(response['context'])
                            grounding_log = None
                            if grounding_check:
                                grounding_log = {
                                    "allow_answer": grounding_check["allow_answer"],
                                    "reason": grounding_check["reason"],
                                    "top_score": grounding_check["top_score"],
                                    "avg_score": grounding_check["avg_score"],
                                    "scored_candidates": serialize_scored_results_for_log(
                                        grounding_check.get("scored_results", []),
                                        max_chars=1800
                                    ),
                                }

                            query_trace.append({
                                "step": "answer_completed",
                                "execution_time_seconds": execution_time,
                                "retrieved_chunk_count": len(retrieved_chunks_log),
                                "sources": source_names,
                                "used_direct_image_mode": st.session_state.vectorstore is None,
                            })
                            append_kb_log(
                                kb_log_path,
                                "query",
                                {
                                    "kb_name": kb_name,
                                    "user": st.session_state.username,
                                    "query_time_seconds": execution_time,
                                    "question": user_input,
                                    "clean_input": clean_input,
                                    "standalone_question": standalone_question,
                                    "answer": answer,
                                    "chat_model": selected_model,
                                    "embedding_model": st.session_state.embed_model,
                                    "semantic_cache_hit": False,
                                    "semantic_cache_id": new_cache_id,
                                    "semantic_cache_distance": cache_distance,
                                    "used_tabular_logic": False,
                                    "used_rag": bool(st.session_state.vectorstore),
                                    "used_direct_image_mode": st.session_state.vectorstore is None,
                                    "source_filter": matched_source,
                                    "search_kwargs": search_kwargs,
                                    "response_style_instruction": response_style,
                                    "retrieved_chunks": retrieved_chunks_log,
                                    "retrieved_chunk_count": len(retrieved_chunks_log),
                                    "source_names": source_names,
                                    "grounding_check": grounding_log,
                                    "last_rag_contexts": context_text if 'context_text' in locals() else [],
                                    "observable_reasoning_trace": query_trace,
                                    "llm_hidden_chain_of_thought_captured": False,
                                    "validation_note": "This log captures observable reasoning artifacts and retrieved evidence, not hidden model chain-of-thought.",
                                }
                            )
                            
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
                            st.session_state.chat_history.append(AIMessage(
                                content=answer,
                                additional_kwargs={"cache_id": new_cache_id}
                            ))
                            
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
                            st.rerun() # Force UI refresh to show Clear Cache button immediately
                        
                    except Exception as e:
                        append_kb_log(
                            kb_log_path,
                            "query_error",
                            {
                                "kb_name": kb_name,
                                "user": st.session_state.username,
                                "question": user_input,
                                "message": str(e),
                                "observable_reasoning_trace": query_trace if 'query_trace' in locals() else [],
                                "llm_hidden_chain_of_thought_captured": False,
                            }
                        )
                        if "client has been closed" in str(e).lower():
                            st.session_state.vectorstore = None
                            st.session_state.semantic_cache = None
                            st.warning("The vector DB client became stale. Reloading the knowledge base on the next run may fix it.")
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
                            start_time = time.time()
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
                            end_time = time.time()
                            st.info(f"⏱️ Evaluation completed in {end_time - start_time:.2f} seconds.")
                            
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

        with st.expander("🤖 Custom Agentic Evaluation (LLM-as-a-Judge)", expanded=False):
            st.caption("Evaluate the accuracy of the last response using a Judge LLM.")
            
            # Construct evaluators
            gemini_options_agentic = [
                "gemini:gemini-2.5-flash",
                "gemini:gemini-2.5-pro",
                "gemini:gemini-2.0-flash", 
                "gemini:gemini-1.5-flash",
                "gemini:gemini-1.5-pro",
            ]
            
            local_models_agentic = st.session_state.models if st.session_state.models else ["qwen2.5:14b", "llama3"]
            local_options_agentic = [f"local:{m}" for m in local_models_agentic]
            
            all_eval_options_agentic = gemini_options_agentic + local_options_agentic
            
            eval_model_agentic = st.selectbox(
                "Select Judge Model", 
                all_eval_options_agentic,
                index=0,
                key="eval_model_agentic_select"
            )
            
            if eval_model_agentic.startswith("gemini:"):
                # Use same API key from session state, update if changed
                key_input = st.text_input(
                    "Google API Key (Shared with Ragas)", 
                    value=st.session_state.google_api_key, 
                    type="password",
                    key="google_api_key_agentic"
                )
                if key_input != st.session_state.google_api_key:
                    st.session_state.google_api_key = key_input
                    st.rerun()
                    
            if st.button("Run Agentic Evaluation"):
                with st.spinner(f"Grading with {eval_model_agentic}..."):
                    start_time = time.time()
                    data = st.session_state.last_rag_data
                    results = agentic_evaluate_rag(
                        query=data['question'], 
                        response=data['answer'], 
                        contexts=data['contexts'], 
                        eval_model_selection=eval_model_agentic, 
                        google_api_key=st.session_state.google_api_key
                    )
                    end_time = time.time()
                    st.info(f"⏱️ Agentic Evaluation completed in {end_time - start_time:.2f} seconds.")
                    
                    if results:
                        st.subheader(f"Score: {results.get('score', 'N/A')}/10")
                        st.write("### Critique")
                        st.info(results.get('critique', 'No critique provided.'))
                        st.write("### Ideal Answer")
                        st.success(results.get('ideal_answer', 'No ideal answer provided.'))
                        
                        # Fix for Unindent/Unexpected indentation:
                        # st.write("### 📝 Summarization Metric (Triggered by Prompt)")
                        # st.info("Because you asked for a 'summary' or 'elaboration', we ran a dedicated Summarization evaluation on the output vs the retrieved context.")
                        # display_metric(st.container(), "Summarization Effectiveness", results.get("summarization", {}))

if __name__ == "__main__":
    main()

