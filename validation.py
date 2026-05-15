import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

import ast

ChatOpenAI = None
OpenAIEmbeddings = None
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError as e:
    pass

Chroma = None
try:
    from langchain.vectorstores import Chroma
except ImportError:
    pass

OllamaEmbeddings = None
try:
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    pass

HuggingFaceEmbeddings = None
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    pass

GoogleGenerativeAIEmbeddings = None
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ImportError:
    pass

if Chroma is not None and OllamaEmbeddings is not None and HuggingFaceEmbeddings is not None:
    LANGCHAIN_REPLAY_AVAILABLE = True
    LANGCHAIN_REPLAY_ERROR = None
else:
    LANGCHAIN_REPLAY_AVAILABLE = False
    LANGCHAIN_REPLAY_ERROR = "One or more core LangChain packages are missing."

try:
    import streamlit as st
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        get_script_run_ctx = None
    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None
    get_script_run_ctx = None
    STREAMLIT_AVAILABLE = False


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
JSON_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def is_running_in_streamlit() -> bool:
    if not STREAMLIT_AVAILABLE or get_script_run_ctx is None:
        return False
    return get_script_run_ctx() is not None


def discover_kb_logs(root_dir: str) -> List[str]:
    matches: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("kb_log_") and filename.endswith(".jsonl"):
                matches.append(os.path.join(dirpath, filename))
    return sorted(matches)


def fetch_ollama_models(host: str = DEFAULT_OLLAMA_HOST) -> List[str]:
    session = requests.Session()
    session.trust_env = False
    response = session.get(f"{host}/api/tags", timeout=(3, 10))
    response.raise_for_status()
    data = response.json()
    return sorted([item["name"] for item in data.get("models", []) if item.get("name")])


def fetch_openai_models(base_url: str, api_key: str) -> List[str]:
    session = requests.Session()
    session.trust_env = False
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{base_url.rstrip('/')}/models"
    try:
        response = session.get(url, headers=headers, timeout=(3, 10))
        response.raise_for_status()
        data = response.json()
        return sorted([item["id"] for item in data.get("data", []) if item.get("id")])
    except Exception as e:
        print(f"Error fetching OpenAI models: {e}")
        return ["glm-4.5v", "kimi-k2.5", "qwen3-coder-30b-a3b-instruct"]


def load_db_config(db_path: str) -> Dict[str, Any]:
    config_path = os.path.join(db_path, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


_EMBEDDING_CACHE: Dict[str, Any] = {}


class CustomOllamaAPIEmbeddings:
    """
    Bypasses LangChain's native Ollama wrapper to safely communicate with 
    Ollama's /api/embed endpoint. Prevents HTTP crashes, retry loops, and 
    extreme slowness with models like qwen or mxbai.
    """
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model_name,
            "input": texts,
            "truncate": True,
            "options": {"num_ctx": 8192}
        }
        session = requests.Session()
        session.trust_env = False
        try:
            resp = session.post(f"{self.base_url}/api/embed", json=payload)
            resp.raise_for_status()
            return resp.json().get("embeddings", [])
        except requests.exceptions.HTTPError as ext:
            if "404" in str(ext) or "400" in str(ext):
                embeddings = []
                for text in texts:
                    safe_text = text[:1000]
                    legacy_payload = {
                        "model": self.model_name,
                        "prompt": safe_text,
                        "options": {"num_ctx": 8192}
                    }
                    r = session.post(f"{self.base_url}/api/embeddings", json=legacy_payload)
                    r.raise_for_status()
                    embeddings.append(r.json().get("embedding", []))
                return embeddings
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), 15):
            batch = texts[i:i+15]
            all_embeddings.extend(self._embed_batch(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]


class CustomGeminiAPIEmbeddings:
    """
    Bypasses LangChain's native Gemini wrapper to safely communicate with
    Google's REST API. Forces sequential processing and explicit timeouts 
    to permanently prevent '504 Deadline Exceeded' errors during validation.
    """
    def __init__(self, model_name: str, api_key: str):
        # API expects raw model name in URL, but prefixed in payload
        self.raw_model = model_name.replace("models/", "")
        
        # Map legacy or Vertex names to the actual 2026 Gemini REST endpoints
        if self.raw_model in ["text-embedding-004", "embedding-004"]:
            self.raw_model = "gemini-embedding-2"
        elif self.raw_model == "embedding-001":
            self.raw_model = "gemini-embedding-001"
            
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def _embed_single(self, text: str) -> List[float]:
        import time
        url = f"{self.base_url}/{self.raw_model}:embedContent?key={self.api_key}"
        # Ensure text is not empty and truncated to a safe length for embeddings
        safe_text = text.strip()[:10000]
        if not safe_text:
            safe_text = " "
            
        payload = {
            "model": f"models/{self.raw_model}",
            "content": {"parts": [{"text": safe_text}]}
        }
        
        session = requests.Session()
        session.trust_env = False
        last_err = None
        
        for attempt in range(4):
            try:
                resp = session.post(url, json=payload, timeout=20.0)
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(2 * (attempt + 1))
                    resp.raise_for_status()
                resp.raise_for_status()
                data = resp.json()
                return data.get("embedding", {}).get("values", [])
            except Exception as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
                
        raise RuntimeError(f"Gemini API Error after retries: {last_err}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Process sequentially. Validation sets are small (~5 chunks), 
        # so sequential is fast and 100% immune to batch timeouts.
        return [self._embed_single(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)


def create_embedding_function(
    embedding_model_name: str,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    local_only: bool = True,
    gemini_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
):
    if not LANGCHAIN_REPLAY_AVAILABLE:
        raise RuntimeError(
            "Replay validation requires LangChain/Chroma embedding dependencies. "
            f"Import error: {LANGCHAIN_REPLAY_ERROR}"
        )

    cache_key = f"{embedding_model_name}::{ollama_host}::{local_only}"
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    if embedding_model_name == "all-MiniLM-L6-v2":
        model_kwargs = {"device": "cpu"}
        if local_only:
            model_kwargs["local_files_only"] = True
        fn = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
        )
        _EMBEDDING_CACHE[cache_key] = fn
        return fn

    # Support Gemini Embeddings using custom robust REST client
    if ("embedding" in embedding_model_name.lower()) and gemini_api_key:
        gemini_models = {"text-embedding-004", "models/embedding-001", "gemini-embedding-2", "gemini-embedding-001"}
        if embedding_model_name in gemini_models or "text-embedding" in embedding_model_name or "gemini-embedding" in embedding_model_name:
            fn = CustomGeminiAPIEmbeddings(
                model_name=embedding_model_name, 
                api_key=gemini_api_key
            )
            _EMBEDDING_CACHE[cache_key] = fn
            return fn

    # Check for openai prefix
    is_openai = embedding_model_name.startswith("[OpenAI]")
    clean_model_name = embedding_model_name.replace("[OpenAI] ", "").replace("[Ollama] ", "")
    
    if is_openai and OpenAIEmbeddings:
        fn = OpenAIEmbeddings(model=clean_model_name, openai_api_base=openai_base_url, api_key=openai_api_key or "sk-dummy", openai_api_key=openai_api_key or "sk-dummy")
        _EMBEDDING_CACHE[cache_key] = fn
        return fn

    fn = CustomOllamaAPIEmbeddings(model_name=clean_model_name, base_url=ollama_host)
    _EMBEDDING_CACHE[cache_key] = fn
    return fn


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python, no numpy)."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _spearman_rank_correlation(scores_a: List[float], scores_b: List[float]) -> Optional[float]:
    """Compute Spearman rank correlation between two score lists (pure Python).
    
    Returns a value between -1.0 and 1.0:
      1.0 = perfect agreement on ranking
      0.0 = no correlation
     -1.0 = completely opposite rankings
    Returns None if fewer than 3 items (not meaningful).
    """
    n = min(len(scores_a), len(scores_b))
    if n < 3:
        return None

    def _rank(scores: List[float]) -> List[float]:
        indexed = sorted(range(n), key=lambda i: scores[i], reverse=True)
        ranks = [0.0] * n
        for rank_pos, idx in enumerate(indexed):
            ranks[idx] = rank_pos + 1.0
        return ranks

    ranks_a = _rank(scores_a[:n])
    ranks_b = _rank(scores_b[:n])

    d_squared = sum((ra - rb) ** 2 for ra, rb in zip(ranks_a, ranks_b))
    return round(1 - (6 * d_squared) / (n * (n ** 2 - 1)), 4)


def compute_embedding_comparison(
    event: Dict[str, Any],
    reference_embedding_model: str,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    gemini_api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Apples-to-apples embedding comparison for a query event.

    Re-embeds the query and retrieved chunk texts with BOTH the production
    embedding model and the reference model, computes cosine similarities
    for each, and compares them on the same scale.
    """
    if event.get("event_type") != "query":
        return None

    query = (
        event.get("standalone_question")
        or event.get("question")
        or event.get("clean_input")
    )
    if not query:
        return None

    retrieved_chunks = event.get("retrieved_chunks", [])
    if not retrieved_chunks:
        return None

    # Get the production embedding model name from the log
    production_model_name = event.get("embedding_model")
    if not production_model_name:
        return {"error": "No embedding_model found in event — cannot compute production cosine similarities."}

    # Use grounding check to determine how many chunks were actually used
    grounding = event.get("grounding_check") or {}
    scored_candidates = grounding.get("scored_candidates", [])
    if scored_candidates and len(scored_candidates) < len(retrieved_chunks):
        retrieved_chunks = retrieved_chunks[:len(scored_candidates)]

    try:
        # Create embedding functions for both models
        ref_fn = create_embedding_function(
            reference_embedding_model,
            ollama_host=ollama_host,
            local_only=True,
            gemini_api_key=gemini_api_key,
        )
        prod_fn = create_embedding_function(
            production_model_name,
            ollama_host=ollama_host,
            local_only=True,
            gemini_api_key=gemini_api_key,
        )

        chunk_texts: List[str] = []
        for chunk in retrieved_chunks:
            text = chunk.get("page_content", "") if isinstance(chunk, dict) else str(chunk)
            chunk_texts.append(text if text.strip() else " ")

        # Compute cosine similarities for BOTH models (same metric, same scale)
        ref_query_vec = ref_fn.embed_query(query)
        ref_chunk_vecs = ref_fn.embed_documents(chunk_texts)
        ref_scores = [round(_cosine_similarity(ref_query_vec, vec), 6) for vec in ref_chunk_vecs]

        prod_query_vec = prod_fn.embed_query(query)
        prod_chunk_vecs = prod_fn.embed_documents(chunk_texts)
        prod_scores = [round(_cosine_similarity(prod_query_vec, vec), 6) for vec in prod_chunk_vecs]

        # Per-chunk comparison
        comparison: List[Dict[str, Any]] = []
        for i, chunk in enumerate(retrieved_chunks):
            source = ""
            if isinstance(chunk, dict):
                source = chunk.get("metadata", {}).get("source", "")
            r = ref_scores[i] if i < len(ref_scores) else None
            p = prod_scores[i] if i < len(prod_scores) else None
            entry: Dict[str, Any] = {
                "chunk_index": i,
                "source": source,
                "reference_cosine": r,
                "production_cosine": p,
            }
            if r is not None and p is not None:
                entry["delta"] = round(p - r, 6)  # positive = production outperforms
            comparison.append(entry)

        # Aggregate metrics
        ref_avg = sum(ref_scores) / len(ref_scores) if ref_scores else 0
        prod_avg = sum(prod_scores) / len(prod_scores) if prod_scores else 0

        ref_top1 = max(ref_scores) if ref_scores else 0
        prod_top1 = max(prod_scores) if prod_scores else 0

        ref_sorted = sorted(ref_scores, reverse=True)
        prod_sorted = sorted(prod_scores, reverse=True)
        ref_top3_avg = sum(ref_sorted[:3]) / min(3, len(ref_sorted)) if ref_sorted else 0
        prod_top3_avg = sum(prod_sorted[:3]) / min(3, len(prod_sorted)) if prod_sorted else 0

        # Do both models agree on the top chunk?
        top_agrees = None
        if len(ref_scores) > 1 and len(prod_scores) > 1:
            ref_best = max(range(len(ref_scores)), key=lambda j: ref_scores[j])
            prod_best = max(range(len(prod_scores)), key=lambda j: prod_scores[j])
            top_agrees = ref_best == prod_best

        # Spearman rank correlation: do both models rank chunks the same way?
        rank_corr = _spearman_rank_correlation(ref_scores, prod_scores)

        return {
            "scoring_method": "cosine_similarity",
            "note": "positive delta = production outperforms, negative = underperforms",
            "reference_model": reference_embedding_model,
            "production_model": production_model_name,
            "query_used": query,
            "chunks_compared": len(retrieved_chunks),
            "reference_top1": round(ref_top1, 4),
            "production_top1": round(prod_top1, 4),
            "delta_top1": round(prod_top1 - ref_top1, 4),
            "reference_top3_avg": round(ref_top3_avg, 4),
            "production_top3_avg": round(prod_top3_avg, 4),
            "delta_top3_avg": round(prod_top3_avg - ref_top3_avg, 4),
            "reference_avg": round(ref_avg, 4),
            "production_avg": round(prod_avg, 4),
            "delta_avg": round(prod_avg - ref_avg, 4),
            "top_chunk_agreement": top_agrees,
            "rank_correlation": rank_corr,
            "per_chunk_comparison": comparison,
        }
    except Exception as exc:
        return {
            "error": f"embedding comparison failed: {exc}",
            "reference_model": reference_embedding_model,
        }



def parse_maybe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            try:
                parsed = ast.literal_eval(value)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return {}


def derive_kb_db_path(log_file: str, event: Dict[str, Any], kb_db_path: Optional[str] = None) -> str:
    if kb_db_path:
        return kb_db_path
    kb_name = event.get("kb_name")
    if not kb_name:
        raise RuntimeError("Replay validation could not determine kb_name from the log event.")
    return os.path.join(os.path.dirname(os.path.abspath(log_file)), f"chroma_db_{kb_name}")


def build_answer_similarity_score(answer_a: str, answer_b: str) -> float:
    words_a = set(re.findall(r"\w+", (answer_a or "").lower()))
    words_b = set(re.findall(r"\w+", (answer_b or "").lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    if union == 0:
        return 0.0
    return round(intersection / union, 4)


def serialize_replay_documents(documents: List[Any], max_chars: int = 2200) -> List[Dict[str, Any]]:
    return [truncate_text({"page_content": getattr(doc, "page_content", ""), "metadata": getattr(doc, "metadata", {})}, max_chars=max_chars) for doc in documents]


def generate_reference_answer(
    provider: str,
    model: str,
    question: str,
    contexts: List[Any],
    response_style_instruction: Optional[str],
    ollama_host: str,
    gemini_api_key: Optional[str],
    **kwargs
) -> str:
    context_blocks = []
    for idx, doc in enumerate(contexts, start=1):
        context_blocks.append(f"[Context {idx}]\n{getattr(doc, 'page_content', '')}")
    context_text = "\n\n".join(context_blocks) if context_blocks else "No retrieved context available."
    style = response_style_instruction or "Return the answer in normal prose."

    prompt = f"""
You are a careful reference-answer generator for replay-based RAG validation.

Answer the question strictly from the supplied retrieved context.
If the context is insufficient, say so explicitly.

Formatting requirement:
{style}

Question:
{question}

Retrieved Context:
{context_text}

Return only the final answer text.
""".strip()

    clean_model = model.replace("[OpenAI] ", "").replace("[Ollama] ", "")
    if provider == "ollama" or model.startswith("[Ollama]"):
        return call_ollama(model=clean_model, prompt=prompt, host=ollama_host, json_mode=False)
    if provider == "openai" or model.startswith("[OpenAI]"):
        if ChatOpenAI:
            chat = ChatOpenAI(model=clean_model, openai_api_base=kwargs.get("openai_base_url"), api_key=kwargs.get("openai_api_key") or "sk-dummy", openai_api_key=kwargs.get("openai_api_key") or "sk-dummy", temperature=0)
            return chat.invoke(prompt).content
        else:
            raise RuntimeError("langchain_openai not installed")
    if provider == "gemini":
        if not gemini_api_key:
            raise RuntimeError("Gemini replay validation requires an API key.")
        return call_gemini(model=clean_model, prompt=prompt, api_key=gemini_api_key, json_mode=False)
    raise ValueError(f"Unsupported provider: {provider}")


def build_replay_validation_context(
    event: Dict[str, Any],
    log_file: str,
    ollama_host: str,
    provider: str,
    model: str,
    gemini_api_key: Optional[str],
    openai_base_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    kb_db_path: Optional[str] = None,
    reference_embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    if event.get("event_type") != "query":
        return {
            "applies": False,
            "reason": "replay_from_kb applies only to query events",
        }

    if not LANGCHAIN_REPLAY_AVAILABLE:
        return {
            "applies": False,
            "reason": f"replay validation dependencies unavailable: {LANGCHAIN_REPLAY_ERROR}",
        }

    db_path = derive_kb_db_path(log_file, event, kb_db_path=kb_db_path)
    if not os.path.exists(db_path):
        return {
            "applies": False,
            "reason": f"knowledge base path not found: {db_path}",
        }

    search_kwargs = parse_maybe_dict(event.get("search_kwargs"))
    k = int(search_kwargs.get("k", 4) or 4)
    replay_filter = search_kwargs.get("filter")

    db_config = load_db_config(db_path)
    production_embedding_model = db_config.get("embedding_model") or event.get("embedding_model")
    replay_embedding_model = reference_embedding_model or production_embedding_model
    if not replay_embedding_model:
        return {
            "applies": False,
            "reason": "could not determine embedding model for replay",
        }

    embedding_function = create_embedding_function(
        replay_embedding_model,
        ollama_host=ollama_host,
        local_only=True,
    )
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_function,
    )

    query = event.get("standalone_question") or event.get("question") or event.get("clean_input")
    if not query:
        return {
            "applies": False,
            "reason": "query event did not contain a usable question for replay",
        }

    retrieval_kwargs: Dict[str, Any] = {"k": k}
    if replay_filter:
        retrieval_kwargs["filter"] = replay_filter
    replay_docs = vectorstore.similarity_search(query, **retrieval_kwargs)

    reference_answer = generate_reference_answer(
        provider=provider,
        model=model,
        question=query,
        contexts=replay_docs,
        response_style_instruction=event.get("response_style_instruction"),
        ollama_host=ollama_host,
        gemini_api_key=gemini_api_key,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
    )

    original_answer = event.get("answer") or ""
    answer_similarity = build_answer_similarity_score(original_answer, reference_answer)

    return {
        "applies": True,
        "db_path": db_path,
        "query_used_for_replay": query,
        "production_embedding_model": production_embedding_model,
        "replay_embedding_model": replay_embedding_model,
        "retrieval_kwargs": retrieval_kwargs,
        "replayed_retrieved_chunk_count": len(replay_docs),
        "replayed_retrieved_chunks": serialize_replay_documents(replay_docs),
        "reference_answer": truncate_text(reference_answer, max_chars=5000),
        "reference_answer_similarity": answer_similarity,
    }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                payload["_line_no"] = line_no
                events.append(payload)
            except json.JSONDecodeError as exc:
                events.append(
                    {
                        "_line_no": line_no,
                        "event_type": "log_parse_error",
                        "raw_line": raw_line,
                        "parse_error": str(exc),
                    }
                )
    return events


def filter_events(
    events: List[Dict[str, Any]],
    event_types: Optional[List[str]] = None,
    max_events: Optional[int] = None,
    target_line_numbers: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    filtered = events
    
    if target_line_numbers is not None:
        target_set = set(target_line_numbers)
        filtered = [evt for evt in filtered if evt.get("_line_no") in target_set]
        
    if event_types:
        wanted = {item.strip() for item in event_types if item.strip()}
        filtered = [event for event in filtered if event.get("event_type") in wanted]
    if max_events is not None and max_events >= 0:
        filtered = filtered[-max_events:]
    return filtered


def truncate_text(value: Any, max_chars: int = 4000) -> Any:
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + f"\n...[truncated, original length={len(value)}]"
    if isinstance(value, list):
        return [truncate_text(item, max_chars=max_chars) for item in value]
    if isinstance(value, dict):
        return {key: truncate_text(item, max_chars=max_chars) for key, item in value.items()}
    return value


def prune_event_for_prompt(event: Dict[str, Any]) -> Dict[str, Any]:
    event_type = event.get("event_type")
    base = {
        "timestamp": event.get("timestamp"),
        "event_type": event_type,
        "kb_name": event.get("kb_name"),
        "user": event.get("user"),
        "_line_no": event.get("_line_no"),
    }

    if event_type == "ingestion":
        base.update(
            {
                "processing_time_seconds": event.get("processing_time_seconds"),
                "uploaded_files": event.get("uploaded_files"),
                "processing_mode": event.get("processing_mode"),
                "ocr_model": event.get("ocr_model"),
                "vlm_model": event.get("vlm_model"),
                "embedding_model": event.get("embedding_model"),
                "chunking_strategy": event.get("chunking_strategy"),
                "chunk_size": event.get("chunk_size"),
                "raw_document_summary": event.get("raw_document_summary"),
                "vector_db_stats": truncate_text(event.get("vector_db_stats"), max_chars=2500),
                "tabular_sources": event.get("tabular_sources"),
                "processed_images_count": event.get("processed_images_count"),
                "processed_images_preview": truncate_text(event.get("processed_images_preview", []), max_chars=1500),
                "observable_reasoning_trace": truncate_text(event.get("observable_reasoning_trace", []), max_chars=2000),
                "validation_note": event.get("validation_note"),
            }
        )
        return base

    if event_type == "query":
        retrieved_chunks = event.get("retrieved_chunks", [])
        raw_answer = event.get("answer", "")
        
        # Dynamically separate the raw query into its internal reasoning trace and its final answer
        import re
        think_text = ""
        if isinstance(raw_answer, str):
            if "\n\n**Time taken:**" in raw_answer:
                parts = raw_answer.split("\n\n**Time taken:**", 1)
                raw_answer = parts[0]
                system_footer = "\n\n**Time taken:**" + parts[1]
                
            match = re.search(r'(?:<think>|<\|channel>thought\n?)(.*?)(?:</?[a-zA-Z_:-][^>]*>|$)', raw_answer, flags=re.DOTALL)
            if match:
                think_text = match.group(1).strip()
                raw_answer = raw_answer[match.end():].strip()
                
            if "system_footer" in locals() and system_footer:
                raw_answer += "\n" + system_footer
                
        base.update(
            {
                "query_time_seconds": event.get("query_time_seconds"),
                "question": event.get("question"),
                "clean_input": event.get("clean_input"),
                "standalone_question": event.get("standalone_question"),
                "llm_reasoning_trace": truncate_text(think_text, max_chars=4000) if think_text else None,
                "answer": truncate_text(raw_answer, max_chars=5000),
                "chat_model": event.get("chat_model"),
                "embedding_model": event.get("embedding_model"),
                "semantic_cache_hit": event.get("semantic_cache_hit"),
                "semantic_cache_id": event.get("semantic_cache_id"),
                "semantic_cache_distance": event.get("semantic_cache_distance"),
                "used_tabular_logic": event.get("used_tabular_logic"),
                "used_rag": event.get("used_rag"),
                "used_direct_image_mode": event.get("used_direct_image_mode"),
                "source_filter": event.get("source_filter"),
                "search_kwargs": event.get("search_kwargs"),
                "response_style_instruction": event.get("response_style_instruction"),
                "retrieved_chunk_count": event.get("retrieved_chunk_count"),
                "retrieved_chunks": truncate_text(retrieved_chunks[:6], max_chars=2200),
                "source_names": event.get("source_names"),
                "grounding_check": truncate_text(event.get("grounding_check"), max_chars=2000),
                "embedding_comparison": event.get("embedding_comparison"),
                "replay_validation_context": truncate_text(event.get("replay_validation_context"), max_chars=3500),
                "observable_reasoning_trace": truncate_text(event.get("observable_reasoning_trace", []), max_chars=2500),
                "validation_note": event.get("validation_note"),
            }
        )
        return base

    if event_type in {"ingestion_error", "query_error", "log_parse_error"}:
        base.update(truncate_text(event, max_chars=3000))
        return base

    base.update(truncate_text(event, max_chars=3000))
    return base


def build_validation_prompt(event: Dict[str, Any]) -> str:
    return build_validation_prompt_with_options(event)


def build_validation_prompt_with_options(
    event: Dict[str, Any],
    reference_embedding_model: Optional[str] = None,
    reference_embedding_notes: Optional[str] = None,
    validation_mode: str = "log_only",
) -> str:
    event_type = event.get("event_type", "unknown")
    event_json = json.dumps(prune_event_for_prompt(event), ensure_ascii=False, indent=2)
    embedding_validation_enabled = bool(reference_embedding_model)
    embedding_context = ""
    if embedding_validation_enabled:
        embedding_context = (
            f"\nEmbedding comparison data is available in the `embedding_comparison` field of the event.\n"
            f"- Reference embedding model: `{reference_embedding_model}`\n"
            f"- Benchmark notes: `{reference_embedding_notes or 'none provided'}`\n"
            "- The comparison re-embeds the query and retrieved chunks with the reference model and computes cosine similarities.\n"
            "- Compare `reference_score` vs `production_score` per chunk. A positive delta means the reference model finds that chunk more relevant.\n"
            "- Use `reference_avg_score` vs `production_avg_score` and `top_chunk_agreement` to assess overall embedding quality.\n"
            "- If the field contains an `error` key, the comparison failed — note this in your assessment.\n"
        )

    replay_context = ""
    if validation_mode == "replay_from_kb" and event.get("replay_validation_context"):
        replay_context = (
            "\nReplay validation context is present.\n"
            "- Compare the original logged retrieval/answer against the replayed retrieval and the regenerated reference answer.\n"
            "- Use replay context as a stronger audit signal for retrieval quality, reasoning quality, and answer quality.\n"
            "- If replay and original disagree materially, call that out explicitly.\n"
        )

    if validation_mode == "log_only":
        event_specific_instructions = ""
        if event_type == "ingestion":
            event_specific_instructions = (
                "- IMPORTANT: This is an 'ingestion' event. You MUST ONLY evaluate the `ingestion_validation` section.\n"
                "- Set `applies`: true and provide a meaningful score (0-100) based on extraction and chunking quality.\n"
                "- For `retrieval_validation`, `chain_of_thought_validation`, and `answer_quality_validation`, set `score`: 0, set `applies`: false (if present), and state 'Not applicable for ingestion events'."
            )
        else:
            event_specific_instructions = (
                "- IMPORTANT: This is a 'query' event. You MUST NOT evaluate the `ingestion_validation` section.\n"
                "- For `ingestion_validation`, set `applies`: false, `score`: 0, and `assessment` to 'Not applicable for query events'.\n"
                "- You MUST fully evaluate `retrieval_validation`, `chain_of_thought_validation`, and `answer_quality_validation`."
            )

        # ── FULL VALIDATION PROMPT (log_only mode) ──
        return f"""
You are a meticulous multimodal RAG validation auditor.

Your job is to validate one structured log event from a knowledge-base-specific log file.

Important rule about chain-of-thought:
- The `observable_reasoning_trace` parameter captures the application system's execution path (retrieval constraints, cache logic, chunk scoring).
- The `llm_reasoning_trace` parameter captures the generator model's explicit <think> block internal logic. If it is null, the generator model's internal reasoning is hidden.
- When validating chain-of-thought, evaluate both the system trace AND the LLM trace (if present). You must ensure the generator's internal reasoning correctly interprets the retrieved evidence without hallucinations.
- Do NOT fault the `answer` if it does not contain reasoning; the reasoning is isolated in `llm_reasoning_trace`.

Validate this event with emphasis on:
- factual grounding against retrieved chunks
- whether the answer was appropriate for the retrieved evidence
- whether the observable reasoning trace is coherent and complete
- whether cache behavior was risky
- whether chunk retrieval quality appears sufficient

{event_specific_instructions}

Scoring Instructions:
- Each validation section below (`ingestion_validation`, `retrieval_validation`, `chain_of_thought_validation`, `answer_quality_validation`) MUST include its own `score` (0-100), `assessment`, `evidence`, and `recommendations`.
- `retrieval_validation` must assess cache behavior.
- `overall_score` (0-100): Your general holistic score for the event after considering all scored categories.
- Return ONLY the fields shown in the schema below. Do NOT add any extra fields.

Event type: {event_type}

Structured log event:
{event_json}

Return valid JSON only. Your response MUST contain exactly these fields in this order:
{{
  "event_type": "{event_type}",
  "summary": "short overall assessment",
  "overall_score": 0,
  "risk_level": "low|medium|high|critical",
  "ingestion_validation": {{
    "applies": true,
    "score": 0,
    "assessment": "...",
    "evidence": "...",
    "recommendations": "..."
  }},
  "retrieval_validation": {{
    "retrieval_quality": "poor|fair|good|excellent",
    "chunk_relevance_assessment": "...",
    "cache_behavior_risk": "low|medium|high|critical",
    "score": 0,
    "assessment": "...",
    "evidence": "...",
    "recommendations": "..."
  }},
  "chain_of_thought_validation": {{
    "scope": "observable_app_trace_only|system_and_generator_llm_reasoning",
    "trace_coherence_score": 0,
    "trace_supported_by_evidence": true,
    "missing_steps": ["..."],
    "suspicious_steps": ["..."],
    "hallucination_risk": "low|medium|high|critical",
    "score": 0,
    "assessment": "...",
    "evidence": "...",
    "recommendations": "..."
  }},
  "answer_quality_validation": {{
    "score": 0,
    "factual_accuracy": "low|medium|high",
    "completeness": "incomplete|partial|complete",
    "assessment": "...",
    "evidence": "...",
    "recommendations": "..."
  }},
  "issues": [
    {{
      "severity": "low|medium|high|critical",
      "title": "...",
      "details": "..."
    }}
  ],
  "recommendations": ["..."],
  "confidence": 0.0
}}
""".strip()

    else:
        # ── EMBEDDING-FOCUSED PROMPT (replay_from_kb mode) ──
        return f"""
You are a meticulous multimodal RAG validation auditor.

Your job is to validate one structured log event from a knowledge-base-specific log file.
{embedding_context}
{replay_context}

Scoring Instructions:
- Focus ONLY on the `embedding_validation` section. Provide `score` (0-100), `assessment`, `evidence`, and `recommendations`.
- `overall_score` (0-100): Your general holistic score for the embedding comparison.
- Do NOT add any other validation sections (no ingestion_validation, no retrieval_validation, no chain_of_thought_validation, no replay_validation). Only return exactly the fields shown in the schema below.

Event type: {event_type}

Structured log event:
{event_json}

Return valid JSON only. Your response MUST contain exactly these fields and no others:
{{
  "event_type": "{event_type}",
  "summary": "short overall assessment",
  "overall_score": 0,
  "risk_level": "low|medium|high|critical",
{'''  "embedding_validation": {{
    "applies": true,
    "reference_embedding_model": "...",
    "comparative_judgement": "production_weaker|comparable|production_stronger|insufficient_evidence|not_requested",
    "score": 0,
    "assessment": "...",
    "evidence": "...",
    "recommendations": "..."
  }},
''' if embedding_validation_enabled else ''}  "issues": [
    {{
      "severity": "low|medium|high|critical",
      "title": "...",
      "details": "..."
    }}
  ],
  "recommendations": ["..."],
  "confidence": 0.0
}}
""".strip()



def call_ollama(
    model: str,
    prompt: str,
    host: str = DEFAULT_OLLAMA_HOST,
    json_mode: bool = True,
) -> str:
    session = requests.Session()
    session.trust_env = False
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "options": {
            "temperature": 0,
            "num_ctx": 32768
        },
    }
    if json_mode:
        payload["format"] = "json"
    response = session.post(f"{host}/api/chat", json=payload, timeout=(10, 1800))
    if not response.ok:
        raise RuntimeError(f"Ollama API error {response.status_code}: {response.text}")
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def call_gemini(model: str, prompt: str, api_key: str, json_mode: bool = True) -> str:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is not installed. Install it before using Gemini validation."
        ) from exc

    genai.configure(api_key=api_key)
    generation_model = genai.GenerativeModel(model)
    generation_config = {"temperature": 0}
    if json_mode:
        generation_config["response_mime_type"] = "application/json"
    response = generation_model.generate_content(prompt, generation_config=generation_config)
    try:
        text = response.text
        if text:
            return text
    except ValueError:
        pass

    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise RuntimeError(f"Gemini returned no text output. It may have been blocked. Response: {response}")

    parts: List[str] = []
    first_candidate = candidates[0]
    content = getattr(first_candidate, "content", None)
    if content:
        for part in getattr(content, "parts", []):
            if hasattr(part, "text") and part.text:
                parts.append(part.text)

    if not parts:
        raise RuntimeError("Gemini returned no usable text parts.")
    return "".join(parts)


def extract_json_block(text: str) -> str:
    fenced = JSON_FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1].strip()
    return text.strip()


def parse_model_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    candidate = extract_json_block(text)
    # Quick fix for LLMs generating trailing commas
    candidate = re.sub(r',\s*\}', '}', candidate)
    candidate = re.sub(r',\s*\]', ']', candidate)
    
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        err_msg = str(exc)
        if "Extra data" in err_msg or "Multiple JSON" in err_msg or "Expecting value" in err_msg:
            # Fix Gemini chunk fragmentation: { "a": 1 } { "b": 2 } -> [ { "a": 1 }, { "b": 2 } ]
            fixed = re.sub(r'\}\s*\{', '},{', candidate)
            fixed_array = f"[{fixed}]"
            try:
                array_json = json.loads(fixed_array)
                if isinstance(array_json, list):
                    merged = {}
                    for chunk in array_json:
                        if isinstance(chunk, dict):
                            merged.update(chunk)
                    return merged, None
            except Exception:
                pass
        return None, f"Parse Error: {err_msg}. Output snippet: {candidate[:1000]}"


def validate_event(
    event: Dict[str, Any],
    provider: str,
    model: str,
    ollama_host: str,
    gemini_api_key: Optional[str],
    openai_base_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    reference_embedding_model: Optional[str] = None,
    reference_embedding_notes: Optional[str] = None,
    validation_mode: str = "log_only",
) -> Dict[str, Any]:
    prompt = build_validation_prompt_with_options(
        event,
        reference_embedding_model=reference_embedding_model,
        reference_embedding_notes=reference_embedding_notes,
        validation_mode=validation_mode,
    )

    clean_model = model.replace("[OpenAI] ", "").replace("[Ollama] ", "")

    if provider == "ollama" or model.startswith("[Ollama]"):
        raw_response = call_ollama(model=clean_model, prompt=prompt, host=ollama_host, json_mode=True)
    elif provider == "gemini":
        if not gemini_api_key:
            raise RuntimeError("Gemini validation requires an API key.")
        raw_response = call_gemini(model=clean_model, prompt=prompt, api_key=gemini_api_key, json_mode=True)
    elif provider == "openai" or provider == "mixed" or model.startswith("[OpenAI]"):
        if ChatOpenAI:
            chat = ChatOpenAI(
                model=clean_model, 
                openai_api_base=openai_base_url, 
                api_key=openai_api_key or "sk-dummy",
                openai_api_key=openai_api_key or "sk-dummy", 
                temperature=0,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            raw_response = chat.invoke(prompt).content
        else:
            raise RuntimeError("langchain_openai not installed")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    parsed, parse_error = parse_model_json(raw_response)
    return {
        "source_event_line": event.get("_line_no"),
        "source_event_type": event.get("event_type"),
        "source_question": event.get("question"),
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "raw_model_response": raw_response,
        "parsed_validation": parsed,
        "parse_error": parse_error,
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    parsed_count = sum(1 for item in results if item.get("parsed_validation"))
    parse_failures = total - parsed_count
    risk_counts: Dict[str, int] = {}
    high_risk_events: List[Dict[str, Any]] = []
    score_total = 0.0
    score_count = 0
    ingestion_total = 0.0
    ingestion_count = 0
    retrieval_total = 0.0
    retrieval_count = 0
    reasoning_total = 0.0
    reasoning_count = 0
    answer_quality_total = 0.0
    answer_quality_count = 0
    embedding_total = 0.0
    embedding_count = 0

    for result in results:
        parsed = result.get("parsed_validation") or {}
        risk = parsed.get("risk_level")
        if risk:
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        if risk in {"high", "critical"}:
            high_risk_events.append(
                {
                    "source_event_line": result.get("source_event_line"),
                    "event_type": result.get("source_event_type"),
                    "summary": parsed.get("summary"),
                    "risk_level": risk,
                }
            )
        score = parsed.get("overall_score")
        if isinstance(score, (int, float)):
            score_total += float(score)
            score_count += 1

        ingestion_score = parsed.get("ingestion_validation", {}).get("score")
        if isinstance(ingestion_score, (int, float)):
            ingestion_total += float(ingestion_score)
            ingestion_count += 1

        retrieval_score = parsed.get("retrieval_validation", {}).get("score")
        if isinstance(retrieval_score, (int, float)):
            retrieval_total += float(retrieval_score)
            retrieval_count += 1

        reasoning_score = parsed.get("chain_of_thought_validation", {}).get("score")
        if isinstance(reasoning_score, (int, float)):
            reasoning_total += float(reasoning_score)
            reasoning_count += 1

        answer_quality_score = parsed.get("answer_quality_validation", {}).get("score")
        if isinstance(answer_quality_score, (int, float)):
            answer_quality_total += float(answer_quality_score)
            answer_quality_count += 1

        embedding_score = parsed.get("embedding_validation", {}).get("score")
        if isinstance(embedding_score, (int, float)):
            embedding_total += float(embedding_score)
            embedding_count += 1

    average_score = round(score_total / score_count, 2) if score_count else None
    average_ingestion = round(ingestion_total / ingestion_count, 2) if ingestion_count else None
    average_retrieval = round(retrieval_total / retrieval_count, 2) if retrieval_count else None
    average_reasoning = round(reasoning_total / reasoning_count, 2) if reasoning_count else None
    average_answer_quality = round(answer_quality_total / answer_quality_count, 2) if answer_quality_count else None
    average_embedding = round(embedding_total / embedding_count, 2) if embedding_count else None
    return {
        "validated_events": total,
        "parsed_events": parsed_count,
        "parse_failures": parse_failures,
        "risk_counts": risk_counts,
        "average_overall_score": average_score,
        "average_ingestion_score": average_ingestion,
        "average_retrieval_score": average_retrieval,
        "average_reasoning_score": average_reasoning,
        "average_answer_quality_score": average_answer_quality,
        "average_embedding_score": average_embedding,
        "high_risk_events": high_risk_events,
    }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_markdown_report(
    log_path: str,
    provider: str,
    model: str,
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
    reference_embedding_model: Optional[str] = None,
    reference_embedding_notes: Optional[str] = None,
    validation_mode: str = "log_only",
) -> str:
    lines: List[str] = []
    lines.append(f"# Validation Report for `{os.path.basename(log_path)}`")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().astimezone().isoformat()}`")
    lines.append(f"- Provider: `{provider}`")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Validation mode: `{validation_mode}`")
    lines.append(f"- Reference embedding model: `{reference_embedding_model or 'not used'}`")
    if reference_embedding_notes:
        lines.append(f"- Reference embedding notes: `{reference_embedding_notes}`")
    lines.append(f"- Events validated: `{summary.get('validated_events')}`")
    lines.append(f"- Parsed results: `{summary.get('parsed_events')}`")
    lines.append(f"- Parse failures: `{summary.get('parse_failures')}`")
    lines.append(f"- Average overall score: `{summary.get('average_overall_score')}`")
    lines.append(f"- Average ingestion score: `{summary.get('average_ingestion_score')}`")
    lines.append(f"- Average embedding score: `{summary.get('average_embedding_score')}`")
    lines.append(f"- Average retrieval score: `{summary.get('average_retrieval_score')}`")
    lines.append(f"- Average reasoning score: `{summary.get('average_reasoning_score')}`")
    lines.append(f"- Average answer quality score: `{summary.get('average_answer_quality_score')}`")
    lines.append("")
    lines.append("## Risk Counts")
    lines.append("")
    risk_counts = summary.get("risk_counts", {})
    if risk_counts:
        for key in sorted(risk_counts.keys()):
            lines.append(f"- `{key}`: {risk_counts[key]}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Event Results")
    lines.append("")
    for result in results:
        parsed = result.get("parsed_validation") or {}
        event_header = f"### Line {result.get('source_event_line')} - `{result.get('source_event_type')}`"
        if result.get("source_question"):
            event_header += f"\n**Question:** `{result.get('source_question')}`"
            
        lines.append(event_header)
        lines.append("")
        if result.get("parse_error"):
            lines.append(f"- Parse error: `{result['parse_error']}`")
            lines.append("")
            continue

        lines.append(f"- Summary: {parsed.get('summary')}")
        lines.append(f"- Overall score: `{parsed.get('overall_score')}`")
        lines.append(f"- Risk level: `{parsed.get('risk_level')}`")
        ingestion_val = parsed.get("ingestion_validation", {})
        lines.append(f"- Ingestion (Score: {ingestion_val.get('score')}): {ingestion_val.get('assessment')}")
        embedding_val = parsed.get("embedding_validation", {})
        lines.append(f"- Embedding (Score: {embedding_val.get('score')}): {embedding_val.get('assessment')}")
        retrieval = parsed.get("retrieval_validation", {})
        lines.append(f"- Retrieval (Score: {retrieval.get('score')}): {retrieval.get('assessment', retrieval.get('chunk_relevance_assessment'))}")
        cot = parsed.get("chain_of_thought_validation", {})
        lines.append(f"- Chain-of-Thought (Score: {cot.get('score')}): {cot.get('assessment')}")
        aq = parsed.get("answer_quality_validation", {})
        lines.append(f"- Answer Quality (Score: {aq.get('score')}): {aq.get('assessment')}")
        replay_validation = parsed.get("replay_validation", {})
        lines.append(f"- Replay validation: {replay_validation.get('assessment')}")

        issues = parsed.get("issues", [])
        if issues:
            lines.append("- Issues:")
            for issue in issues:
                lines.append(
                    f"  - [{issue.get('severity')}] {issue.get('title')}: {issue.get('details')}"
                )

        recommendations = parsed.get("recommendations", [])
        if recommendations:
            lines.append("- Recommendations:")
            for recommendation in recommendations:
                lines.append(f"  - {recommendation}")

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_markdown(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def derive_output_paths(log_path: str, output_base: Optional[str]) -> Tuple[str, str]:
    if output_base:
        json_path = output_base if output_base.endswith(".json") else output_base + ".json"
        md_base = output_base[:-5] if json_path.endswith(".json") else output_base
        md_path = md_base + ".md"
        return json_path, md_path

    base = os.path.splitext(log_path)[0] + "_validation_report"
    return base + ".json", base + ".md"


def run_validation(
    log_file: str,
    provider: str,
    model: str,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    gemini_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    reference_embedding_model: Optional[str] = None,
    reference_embedding_notes: Optional[str] = None,
    validation_mode: str = "log_only",
    kb_db_path: Optional[str] = None,
    event_types: Optional[List[str]] = None,
    max_events: Optional[int] = None,
    target_line_numbers: Optional[List[int]] = None,
    output_base: Optional[str] = None,
    progress_callback=None,
) -> Tuple[Dict[str, Any], str, str]:
    """Run validation over filtered log events.

    Args:
        progress_callback: Optional callable invoked after each event with
            (current: int, total: int, event: dict, result: dict).
            Use this to drive progress bars or live status output.
    """
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")

    events = load_jsonl(log_file)
    # Removed the hardcoded limitation that forced replay_from_kb mode to only run 1 event.
    events = filter_events(
        events, 
        event_types=event_types or None, 
        max_events=max_events, 
        target_line_numbers=target_line_numbers
    )
    if not events:
        raise RuntimeError("No matching events found in the log file.")

    total = len(events)
    results: List[Dict[str, Any]] = []
    import time
    for idx, event in enumerate(events, start=1):
        start_time = time.time()
        try:
            event_for_validation = dict(event)
            # Lightweight embedding comparison (works in any mode)
            if reference_embedding_model and event.get("event_type") == "query":
                try:
                    embed_cmp = compute_embedding_comparison(
                        event,
                        reference_embedding_model,
                        ollama_host=ollama_host,
                        gemini_api_key=gemini_api_key,
                    )
                    if embed_cmp:
                        event_for_validation["embedding_comparison"] = embed_cmp
                except Exception:
                    event_for_validation["embedding_comparison"] = {
                        "error": "embedding comparison could not be computed",
                        "reference_model": reference_embedding_model,
                    }
            if validation_mode == "replay_from_kb":
                try:
                    replay_context = build_replay_validation_context(
                        event_for_validation,
                        log_file=log_file,
                        ollama_host=ollama_host,
                        provider=provider,
                        model=model,
                        gemini_api_key=gemini_api_key,
                        openai_base_url=openai_base_url,
                        openai_api_key=openai_api_key,
                        kb_db_path=kb_db_path,
                        reference_embedding_model=reference_embedding_model,
                    )
                    event_for_validation["replay_validation_context"] = replay_context
                except Exception as replay_exc:
                    event_for_validation["replay_validation_context"] = {
                        "applies": False,
                        "reason": f"replay setup failed: {replay_exc}",
                    }
            result = validate_event(
                event=event_for_validation,
                provider=provider,
                model=model,
                ollama_host=ollama_host,
                gemini_api_key=gemini_api_key,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
                reference_embedding_model=reference_embedding_model,
                reference_embedding_notes=reference_embedding_notes,
                validation_mode=validation_mode,
            )
        except Exception as exc:
            result = {
                "source_event_line": event.get("_line_no"),
                "source_event_type": event.get("event_type"),
                "source_question": event.get("question"),
                "provider": provider,
                "model": model,
                "raw_model_response": None,
                "parsed_validation": None,
                "parse_error": f"validation_call_failed: {exc}",
            }
        result["time_taken"] = time.time() - start_time
        results.append(result)
        if progress_callback is not None:
            progress_callback(idx, total, event, result)

    summary = summarize_results(results)
    report = {
        "log_file": os.path.abspath(log_file),
        "provider": provider,
        "model": model,
        "validation_mode": validation_mode,
        "kb_db_path": kb_db_path,
        "reference_embedding_model": reference_embedding_model,
        "reference_embedding_notes": reference_embedding_notes,
        "generated_at": datetime.now().astimezone().isoformat(),
        "summary": summary,
        "results": results,
    }

    json_path, md_path = derive_output_paths(log_file, output_base)
    write_json(json_path, report)
    write_markdown(
        md_path,
        build_markdown_report(
            log_file,
            provider,
            model,
            summary,
            results,
            reference_embedding_model=reference_embedding_model,
            reference_embedding_notes=reference_embedding_notes,
            validation_mode=validation_mode,
        ),
    )
    return report, json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate KB-specific JSONL logs with Gemini or local Ollama."
    )
    parser.add_argument("--log-file", required=True, help="Path to the KB JSONL log file.")
    parser.add_argument(
        "--provider",
        required=True,
        choices=["ollama", "gemini"],
        help="Validation model provider.",
    )
    parser.add_argument("--model", required=True, help="Model name to use for validation.")
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help="Local Ollama host, default http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Gemini API key. Defaults to GOOGLE_API_KEY env var.",
    )
    parser.add_argument(
        "--event-types",
        default="",
        help="Comma-separated event types to validate, e.g. query,ingestion",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap on number of events to validate from the log.",
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help="Optional output base path. Produces both .json and .md reports.",
    )
    parser.add_argument(
        "--reference-embedding-model",
        default=None,
        help="Optional stronger/reference embedding model name for comparative embedding validation.",
    )
    parser.add_argument(
        "--reference-embedding-notes",
        default=None,
        help="Optional notes describing the embedding benchmark context.",
    )
    parser.add_argument(
        "--validation-mode",
        default="log_only",
        choices=["log_only", "replay_from_kb"],
        help="Whether to validate strictly from logs or replay retrieval from the KB.",
    )
    parser.add_argument(
        "--kb-db-path",
        default=None,
        help="Optional explicit path to the Chroma DB for replay_from_kb mode.",
    )
    return parser.parse_args()


def cli_main() -> int:
    args = parse_args()
    event_types = [item.strip() for item in args.event_types.split(",") if item.strip()]

    RISK_ICON = {"low": "OK ", "medium": "/!\\ ", "high": "[!]", "critical": "[X]"}

    def _cli_progress(current: int, total: int, event: Dict[str, Any], result: Dict[str, Any]) -> None:
        event_type = event.get("event_type", "unknown")
        line_no = event.get("_line_no", "?")
        parsed = result.get("parsed_validation") or {}
        risk = parsed.get("risk_level", "")
        score = parsed.get("overall_score", "")
        icon = RISK_ICON.get(risk, "   ")
        err = result.get("parse_error")
        bar_filled = int((current / total) * 20)
        bar = "#" * bar_filled + "-" * (20 - bar_filled)
        if err:
            status = f"parse_error: {err[:60]}"
        else:
            ingestion = parsed.get("ingestion_score", "")
            retrieval = parsed.get("retrieval_score", "")
            reasoning = parsed.get("reasoning_score", "")
            answer_quality = parsed.get("answer_quality_score", "")
            status = (
                f"score={score} ing={ingestion} ret={retrieval} reas={reasoning} ans={answer_quality} "
                f"risk={risk or 'n/a'}"
            )
        time_taken = result.get("time_taken", 0.0)
        print(
            f"[{bar}] {current:>3}/{total}  line {str(line_no):<5} {event_type:<18} {icon} [{time_taken:.1f}s] {status}",
            flush=True,
        )

    try:
        _, json_path, md_path = run_validation(
            log_file=args.log_file,
            provider=args.provider,
            model=args.model,
            ollama_host=args.ollama_host,
            gemini_api_key=args.gemini_api_key,
            openai_base_url=args.openai_base_url if hasattr(args, 'openai_base_url') else None,
            openai_api_key=args.openai_api_key if hasattr(args, 'openai_api_key') else None,
            reference_embedding_model=args.reference_embedding_model,
            reference_embedding_notes=args.reference_embedding_notes,
            validation_mode=args.validation_mode,
            kb_db_path=args.kb_db_path,
            event_types=event_types or None,
            max_events=args.max_events,
            output_base=args.output_base,
            progress_callback=_cli_progress,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 0


def render_streamlit_report(report: Dict[str, Any], json_path: str, md_path: str) -> None:
    summary = report.get("summary", {})
    st.success("Validation completed.")
    st.write(f"JSON report: `{json_path}`")
    st.write(f"Markdown report: `{md_path}`")
    st.write(f"Validation mode: `{report.get('validation_mode', 'log_only')}`")
    if report.get("kb_db_path"):
        st.write(f"Replay KB path: `{report.get('kb_db_path')}`")
    st.write(
        f"Reference embedding model: `{report.get('reference_embedding_model') or 'not used'}`"
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Events Validated", summary.get("validated_events", 0))
    col2.metric("Parsed Results", summary.get("parsed_events", 0))
    col3.metric("Parse Failures", summary.get("parse_failures", 0))
    col4.metric("Average Score", summary.get("average_overall_score", "n/a"))
    col5.metric("Avg Ingestion", summary.get("average_ingestion_score", "n/a"))

    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
    score_col1.metric("Avg Embedding", summary.get("average_embedding_score", "n/a"))
    score_col2.metric("Avg Retrieval", summary.get("average_retrieval_score", "n/a"))
    score_col3.metric("Avg Reasoning", summary.get("average_reasoning_score", "n/a"))
    score_col4.metric("Avg Answer Quality", summary.get("average_answer_quality_score", "n/a"))

    risk_counts = summary.get("risk_counts", {})
    with st.expander("Risk Counts", expanded=True):
        if risk_counts:
            st.json(risk_counts)
        else:
            st.write("No risk counts available.")

    high_risk_events = summary.get("high_risk_events", [])
    with st.expander("High Risk Events", expanded=bool(high_risk_events)):
        if high_risk_events:
            st.dataframe(high_risk_events, use_container_width=True)
        else:
            st.write("No high-risk events found.")

    results = report.get("results", [])
    with st.expander("Per-Event Results", expanded=False):
        for result in results:
            time_taken = result.get("time_taken")
            time_str = f" ({time_taken:.1f}s)" if time_taken is not None else ""
            label = f"Line {result.get('source_event_line')} - {result.get('source_event_type')}{time_str}"
            with st.container():
                st.markdown(f"### {label}")
                if result.get("source_question"):
                    st.caption(f"**Question:** {result.get('source_question')}")
                if result.get("parse_error"):
                    st.error(result["parse_error"])
                parsed = result.get("parsed_validation")
                if parsed:
                    # ── Score cards row ──
                    risk = parsed.get("risk_level", "n/a")
                    risk_color = {"low": "green", "medium": "orange", "high": "red", "critical": "red"}.get(risk, "gray")
                    st.markdown(
                        f"**Overall Score:** <span style='font-size:1.6em; font-weight:bold;'>{parsed.get('overall_score', 'n/a')}</span>"
                        f" &nbsp;&nbsp; **Risk:** <span style='font-size:1.4em; font-weight:bold; color:{risk_color};'>{risk.upper()}</span>",
                        unsafe_allow_html=True,
                    )

                    # Collect all validation section scores
                    section_scores = []
                    for key, display_name in [
                        ("ingestion_validation", "Ingestion"),
                        ("embedding_validation", "Embedding"),
                        ("retrieval_validation", "Retrieval"),
                        ("chain_of_thought_validation", "Reasoning"),
                        ("answer_quality_validation", "Answer Quality"),
                    ]:
                        section = parsed.get(key)
                        if section and isinstance(section, dict) and "score" in section:
                            section_scores.append((display_name, section["score"]))

                    if section_scores:
                        cols = st.columns(len(section_scores))
                        for col, (name, s) in zip(cols, section_scores):
                            col.metric(name, s)

                    # Show detailed JSON below
                    st.json(parsed)
                st.divider()


@st.cache_data(show_spinner=False)
def load_jsonl_cached(log_file: str) -> List[Dict[str, Any]]:
    if not os.path.exists(log_file):
        return []
    return load_jsonl(log_file)


def run_streamlit_app() -> None:
    st.set_page_config(page_title="KB Log Validator", page_icon="🧪", layout="wide")
    st.title("KB Log Validator")
    st.caption(
        "Validate KB-specific JSONL logs using either a local Ollama model or Gemini."
    )

    workspace_root = os.getcwd()
    discovered_logs = discover_kb_logs(workspace_root)
    default_log = discovered_logs[0] if discovered_logs else ""

    with st.sidebar:
        st.header("Validation Settings")
        if discovered_logs:
            selected_log = st.selectbox(
                "Discovered KB Logs",
                options=discovered_logs,
                index=0,
                help="Choose a KB-specific log file discovered under the current workspace.",
            )
        else:
            selected_log = ""
            st.info("No `kb_log_*.jsonl` files were found under the current workspace.")

        log_file = st.text_input(
            "Log File Path",
            value=selected_log or default_log,
            help="You can paste a full JSONL log path here if it is not listed above.",
        ).strip()

        provider = st.selectbox("Provider", ["ollama", "gemini", "openai", "mixed"], index=0)
        ollama_host = st.text_input("Ollama Host", value=DEFAULT_OLLAMA_HOST).strip()
        openai_base_url = st.text_input("OpenAI Base URL", value=os.environ.get("OPENAI_BASE_URL", "")).strip()
        openai_api_key = st.text_input("OpenAI API Key", value=os.environ.get("OPENAI_API_KEY", ""), type="password").strip()

        gemini_api_key = st.text_input(
            "Gemini API Key",
            value=os.environ.get("GOOGLE_API_KEY", ""),
            type="password",
            help="Required if using Gemini for validation or as a reference embedding model.",
        ).strip() or None

        if provider == "ollama":
            models: List[str] = []
            model_fetch_error: Optional[str] = None
            if ollama_host:
                try:
                    models = fetch_ollama_models(ollama_host)
                except Exception as exc:
                    model_fetch_error = str(exc)

            if model_fetch_error:
                st.warning(f"Could not load Ollama models: {model_fetch_error}")

            if models:
                model = st.selectbox("Ollama Model", options=models, index=0)
            else:
                model = st.text_input("Ollama Model", value="qwen2.5:14b").strip()
        elif provider == "openai":
            models: List[str] = []
            if openai_base_url and openai_api_key:
                models = fetch_openai_models(openai_base_url, openai_api_key)
            if models:
                model = st.selectbox("OpenAI Model", options=models, index=0)
            else:
                model = st.text_input("OpenAI Model", value="glm-4.5v").strip()
        elif provider == "mixed":
            models: List[str] = []
            if ollama_host:
                try:
                    models.extend([f"[Ollama] {m}" for m in fetch_ollama_models(ollama_host)])
                except:
                    pass
            if openai_base_url and openai_api_key:
                try:
                    models.extend([f"[OpenAI] {m}" for m in fetch_openai_models(openai_base_url, openai_api_key)])
                except:
                    pass
            if not models:
                models = ["[OpenAI] glm-4.5v", "[OpenAI] kimi-k2.5", "[Ollama] qwen2.5:14b"]
            model = st.selectbox("Mixed Model", options=models, index=0)
        else:
            model = st.text_input("Gemini Model", value="gemini-2.5-pro").strip()

        event_types = st.multiselect(
            "Event Types",
            options=["query", "ingestion", "query_error", "ingestion_error", "log_parse_error"],
            default=["query", "ingestion"],
            help="Restrict validation to specific event types.",
        )
        validation_mode = st.selectbox(
            "Validation Mode",
            options=["log_only", "replay_from_kb"],
            index=0,
            help="`replay_from_kb` re-opens the Chroma DB, reruns retrieval for query events, and generates a reference answer.",
        )
        if validation_mode == "replay_from_kb" and not LANGCHAIN_REPLAY_AVAILABLE:
            st.warning(
                "Replay mode needs LangChain/Chroma embedding dependencies in this environment. "
                f"Current import error: {LANGCHAIN_REPLAY_ERROR}"
            )
        kb_db_path = st.text_input(
            "KB DB Path (optional)",
            value="",
            help="Optional explicit path to the Chroma DB. Leave blank to auto-derive it from the log location and kb_name.",
        ).strip() or None
        
        select_specific_logs = st.checkbox("Select specific logs to evaluate", value=False)
        target_line_numbers = None
        max_events_raw = 10
        
        if select_specific_logs:
            if log_file and os.path.exists(log_file):
                all_events = load_jsonl_cached(log_file)
                if event_types:
                    wanted = {item.strip() for item in event_types if item.strip()}
                    all_events = [e for e in all_events if e.get("event_type") in wanted]
                
                display_events = all_events[-500:]
                
                options = {}
                for e in display_events:
                    line_no = e.get("_line_no", "?")
                    etype = e.get("event_type", "unknown")
                    summary = ""
                    if etype == "query":
                        summary = str(e.get("question", ""))[:60]
                    elif etype == "ingestion":
                        summary = str(e.get("document_id", ""))[:60]
                    else:
                        summary = str(e.get("error", ""))[:60]
                        
                    label = f"Line {line_no} [{etype}] - {summary}"
                    options[label] = line_no
                
                selected_labels = st.multiselect("Select Logs (Max 500 recent)", options=list(options.keys()))
                if selected_labels:
                    target_line_numbers = [options[lbl] for lbl in selected_labels]
            else:
                st.warning("Please provide a valid Log File Path first.")
        else:
            max_events_raw = st.number_input(
                "Max Events (Latest N)",
                min_value=1,
                value=10,
                step=1,
                help="Limit how many matching events are validated in one run. Picks the latest events.",
            )

        output_base = st.text_input(
            "Output Base Path (optional)",
            value="",
            help="If provided, the validator writes `<base>.json` and `<base>.md`.",
        ).strip() or None
        
        if validation_mode == "replay_from_kb":
            enable_embedding_validation = st.checkbox(
                "Enable optional embedding comparison",
                value=False,
                help="Re-embeds the query and retrieved chunks with a reference model to compute real cosine-similarity scores for comparison against the production model.",
            )
            if enable_embedding_validation:
                embed_options = ["all-MiniLM-L6-v2"]
                
                # Always show Gemini models if the library is available
                if GoogleGenerativeAIEmbeddings:
                    embed_options.extend(["gemini-embedding-2", "text-embedding-004"])
                
                try:
                    if ollama_host:
                        # Fetch Ollama models regardless of main validator
                        embed_options.extend(fetch_ollama_models(ollama_host))
                except Exception:
                    pass
                
                # Deduplicate and sort
                embed_options = sorted(list(set(embed_options)))
                    
                reference_embedding_model = st.selectbox(
                    "Reference Embedding Model",
                    options=embed_options,
                    index=0,
                    help="Name of the embedding model to compare against.",
                )
                reference_embedding_notes = st.text_input(
                    "Reference Embedding Notes (optional)",
                    value="Use as a stronger semantic retrieval baseline.",
                    help="Any notes about the comparison setup or why this model is the benchmark.",
                ).strip() or None
            else:
                reference_embedding_model = None
                reference_embedding_notes = None
        else:
            reference_embedding_model = None
            reference_embedding_notes = None

    st.write("Selected log file:", f"`{log_file}`" if log_file else "_none_")

    if st.button("Run Validation", type="primary"):
        if not log_file:
            st.error("Please select or enter a log file path.")
            return
        if provider == "gemini" and not gemini_api_key:
            st.error("Gemini validation requires an API key.")
            return
        if not model:
            st.error("Please provide a model name.")
            return

        # ── Live progress UI ──────────────────────────────────────
        progress_bar = st.progress(0, text="Starting validation...")
        status_box = st.empty()
        results_log = st.empty()
        _live_rows: List[str] = []

        RISK_ICON = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}

        def _score_badge(label: str, value) -> str:
            """Format a score as a bold colored badge."""
            if value == "" or value is None:
                return ""
            try:
                v = int(value)
                if v >= 80:
                    return f"🟢 **{label}: {v}**"
                elif v >= 60:
                    return f"🟡 **{label}: {v}**"
                elif v >= 40:
                    return f"🟠 **{label}: {v}**"
                else:
                    return f"🔴 **{label}: {v}**"
            except (ValueError, TypeError):
                return f"⚪ **{label}: {value}**"

        def _on_progress(current: int, total: int, event: Dict[str, Any], result: Dict[str, Any]) -> None:
            fraction = current / total
            pct = int(fraction * 100)
            event_type = event.get("event_type", "unknown")
            line_no = event.get("_line_no", "?")

            # Update the progress bar text
            progress_bar.progress(
                fraction,
                text=f"Validating event {current} of {total} — line {line_no} ({event_type})",
            )

            # Build a status line for this result
            parsed = result.get("parsed_validation") or {}
            risk = parsed.get("risk_level", "")
            score = parsed.get("overall_score", "")
            ingestion = parsed.get("ingestion_validation", {}).get("score", "")
            embedding = parsed.get("embedding_validation", {}).get("score", "")
            retrieval = parsed.get("retrieval_validation", {}).get("score", "")
            reasoning = parsed.get("chain_of_thought_validation", {}).get("score", "")
            answer_quality = parsed.get("answer_quality_validation", {}).get("score", "")
            icon = RISK_ICON.get(risk, "🔹")
            err = result.get("parse_error")
            time_taken = result.get("time_taken", 0.0)
            if err:
                row = f"❌ **Line {line_no}** `{event_type}` ({time_taken:.1f}s) — parse error: `{err[:80]}`"
            else:
                badges = " ".join(filter(None, [
                    _score_badge("Overall", score),
                    _score_badge("Ingestion", ingestion),
                    _score_badge("Embedding", embedding),
                    _score_badge("Retrieval", retrieval),
                    _score_badge("Reasoning", reasoning),
                    _score_badge("Answer", answer_quality),
                ]))
                row = f"{icon} **Line {line_no}** `{event_type}` ({time_taken:.1f}s) 🔹 Risk: **{risk or 'n/a'}**\n\n{badges}"
            _live_rows.append(row)

            # Show last 8 rows so the box doesn't grow forever
            results_log.markdown("\n\n---\n\n".join(_live_rows[-8:]))
            status_box.caption(f"Progress: {pct}% ({current}/{total} events processed)")

        try:
            report, json_path, md_path = run_validation(
                log_file=log_file,
                provider=provider,
                model=model,
                ollama_host=ollama_host,
                gemini_api_key=gemini_api_key,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
                reference_embedding_model=reference_embedding_model,
                reference_embedding_notes=reference_embedding_notes,
                validation_mode=validation_mode,
                kb_db_path=kb_db_path,
                event_types=event_types or None,
                max_events=int(max_events_raw) if not select_specific_logs else None,
                target_line_numbers=target_line_numbers,
                output_base=output_base,
                progress_callback=_on_progress,
            )
        except Exception as exc:
            progress_bar.empty()
            status_box.empty()
            results_log.empty()
            st.exception(exc)
            return

        # Clear live widgets and hand off to the final report renderer
        progress_bar.progress(1.0, text="Validation complete ✅")
        status_box.empty()
        results_log.empty()

        render_streamlit_report(report, json_path, md_path)


if __name__ == "__main__":
    if is_running_in_streamlit():
        run_streamlit_app()
    else:
        raise SystemExit(cli_main())

