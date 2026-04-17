import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

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
) -> List[Dict[str, Any]]:
    filtered = events
    if event_types:
        wanted = {item.strip() for item in event_types if item.strip()}
        filtered = [event for event in filtered if event.get("event_type") in wanted]
    if max_events is not None and max_events >= 0:
        filtered = filtered[:max_events]
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
        base.update(
            {
                "query_time_seconds": event.get("query_time_seconds"),
                "question": event.get("question"),
                "clean_input": event.get("clean_input"),
                "standalone_question": event.get("standalone_question"),
                "answer": truncate_text(event.get("answer"), max_chars=5000),
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
    event_type = event.get("event_type", "unknown")
    event_json = json.dumps(prune_event_for_prompt(event), ensure_ascii=False, indent=2)

    accuracy_scoring_instruction = (
        "- `accuracy_score` (0-100): Must represent a blended evaluation of (1) the accuracy and relevance of retrieved chunks, (2) how accurately the chain of thought was followed, and (3) the factual accuracy of the final answer.\n"
        if event_type == "query" else ""
    )
    accuracy_schema = (
        '  "accuracy_score": 0,\n'
        if event_type == "query" else ""
    )

    return f"""
You are a meticulous multimodal RAG validation auditor.

Your job is to validate one structured log event from a knowledge-base-specific log file.

Important rule about chain-of-thought:
- You do NOT have access to the model's hidden private reasoning.
- When asked to perform chain-of-thought validation, validate ONLY the observable reasoning trace captured in the log:
  - standalone question creation
  - cache decisions
  - retrieval settings
  - grounding checks
  - retrieved chunks
  - final answer
- Never claim access to hidden internal reasoning.

Validate this event with emphasis on:
- factual grounding against retrieved chunks
- whether the answer was appropriate for the retrieved evidence
- whether the observable reasoning trace is coherent and complete
- whether cache behavior was risky
- whether chunk retrieval quality appears sufficient
- whether the event is suitable for downstream stronger-model validation

Scoring Instructions:
{accuracy_scoring_instruction}- `overall_score` (0-100): Your general holistic score for the event.

Event type: {event_type}

Structured log event:
{event_json}

Return valid JSON only with this schema:
{{
  "event_type": "{event_type}",
  "summary": "short overall assessment",
  "overall_score": 0,
{accuracy_schema}  "risk_level": "low|medium|high|critical",
  "chain_of_thought_validation": {{
    "scope": "observable_trace_only",
    "trace_coherence_score": 0,
    "trace_supported_by_evidence": true,
    "missing_steps": ["..."],
    "suspicious_steps": ["..."],
    "assessment": "..."
  }},
  "grounding_validation": {{
    "supported_by_retrieved_chunks": true,
    "hallucination_risk": "low|medium|high|critical",
    "assessment": "..."
  }},
  "retrieval_validation": {{
    "retrieval_quality": "poor|fair|good|excellent",
    "chunk_relevance_assessment": "..."
  }},
  "cache_validation": {{
    "cache_behavior_risk": "low|medium|high|critical",
    "assessment": "..."
  }},
  "ingestion_validation": {{
    "applies": true,
    "assessment": "..."
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


def call_ollama(model: str, prompt: str, host: str = DEFAULT_OLLAMA_HOST) -> str:
    session = requests.Session()
    session.trust_env = False
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "options": {"temperature": 0},
    }
    response = session.post(f"{host}/api/chat", json=payload, timeout=(10, 1800))
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def call_gemini(model: str, prompt: str, api_key: str) -> str:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is not installed. Install it before using Gemini validation."
        ) from exc

    genai.configure(api_key=api_key)
    generation_model = genai.GenerativeModel(model)
    response = generation_model.generate_content(
        prompt,
        generation_config={"temperature": 0, "response_mime_type": "application/json"},
    )
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
) -> Dict[str, Any]:
    prompt = build_validation_prompt(event)

    if provider == "ollama":
        raw_response = call_ollama(model=model, prompt=prompt, host=ollama_host)
    elif provider == "gemini":
        if not gemini_api_key:
            raise RuntimeError("Gemini validation requires an API key.")
        raw_response = call_gemini(model=model, prompt=prompt, api_key=gemini_api_key)
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
    accuracy_total = 0.0
    accuracy_count = 0

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

        accuracy = parsed.get("accuracy_score")
        if isinstance(accuracy, (int, float)):
            accuracy_total += float(accuracy)
            accuracy_count += 1

    average_score = round(score_total / score_count, 2) if score_count else None
    average_accuracy = round(accuracy_total / accuracy_count, 2) if accuracy_count else None
    return {
        "validated_events": total,
        "parsed_events": parsed_count,
        "parse_failures": parse_failures,
        "risk_counts": risk_counts,
        "average_overall_score": average_score,
        "average_accuracy_score": average_accuracy,
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
) -> str:
    lines: List[str] = []
    lines.append(f"# Validation Report for `{os.path.basename(log_path)}`")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().astimezone().isoformat()}`")
    lines.append(f"- Provider: `{provider}`")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Events validated: `{summary.get('validated_events')}`")
    lines.append(f"- Parsed results: `{summary.get('parsed_events')}`")
    lines.append(f"- Parse failures: `{summary.get('parse_failures')}`")
    lines.append(f"- Average overall score: `{summary.get('average_overall_score')}`")
    lines.append(f"- Average accuracy score: `{summary.get('average_accuracy_score')}`")
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
        lines.append(f"- Accuracy score: `{parsed.get('accuracy_score')}`")
        lines.append(f"- Risk level: `{parsed.get('risk_level')}`")
        cot = parsed.get("chain_of_thought_validation", {})
        lines.append(
            f"- Observable chain-of-thought validation: {cot.get('assessment')}"
        )
        grounding = parsed.get("grounding_validation", {})
        lines.append(f"- Grounding: {grounding.get('assessment')}")
        retrieval = parsed.get("retrieval_validation", {})
        lines.append(f"- Retrieval: {retrieval.get('chunk_relevance_assessment')}")
        cache = parsed.get("cache_validation", {})
        lines.append(f"- Cache: {cache.get('assessment')}")

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
    event_types: Optional[List[str]] = None,
    max_events: Optional[int] = None,
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
    events = filter_events(events, event_types=event_types or None, max_events=max_events)
    if not events:
        raise RuntimeError("No matching events found in the log file.")

    total = len(events)
    results: List[Dict[str, Any]] = []
    for idx, event in enumerate(events, start=1):
        try:
            result = validate_event(
                event=event,
                provider=provider,
                model=model,
                ollama_host=ollama_host,
                gemini_api_key=gemini_api_key,
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
        results.append(result)
        if progress_callback is not None:
            progress_callback(idx, total, event, result)

    summary = summarize_results(results)
    report = {
        "log_file": os.path.abspath(log_file),
        "provider": provider,
        "model": model,
        "generated_at": datetime.now().astimezone().isoformat(),
        "summary": summary,
        "results": results,
    }

    json_path, md_path = derive_output_paths(log_file, output_base)
    write_json(json_path, report)
    write_markdown(md_path, build_markdown_report(log_file, provider, model, summary, results))
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
        accuracy = parsed.get("accuracy_score", "")
        icon = RISK_ICON.get(risk, "   ")
        err = result.get("parse_error")
        bar_filled = int((current / total) * 20)
        bar = "#" * bar_filled + "-" * (20 - bar_filled)
        if err:
            status = f"parse_error: {err[:60]}"
        else:
            status = f"score={score} acc={accuracy} risk={risk or 'n/a'}"
        print(
            f"[{bar}] {current:>3}/{total}  line {str(line_no):<5} {event_type:<18} {icon} {status}",
            flush=True,
        )

    try:
        _, json_path, md_path = run_validation(
            log_file=args.log_file,
            provider=args.provider,
            model=args.model,
            ollama_host=args.ollama_host,
            gemini_api_key=args.gemini_api_key,
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

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Events Validated", summary.get("validated_events", 0))
    col2.metric("Parsed Results", summary.get("parsed_events", 0))
    col3.metric("Parse Failures", summary.get("parse_failures", 0))
    col4.metric("Average Score", summary.get("average_overall_score", "n/a"))
    col5.metric("Average Accuracy", summary.get("average_accuracy_score", "n/a"))

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
            label = f"Line {result.get('source_event_line')} - {result.get('source_event_type')}"
            with st.container():
                st.markdown(f"**{label}**")
                if result.get("source_question"):
                    st.caption(f"**Question:** {result.get('source_question')}")
                if result.get("parse_error"):
                    st.error(result["parse_error"])
                parsed = result.get("parsed_validation")
                if parsed:
                    st.json(parsed)
                st.divider()


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

        provider = st.selectbox("Provider", ["ollama", "gemini"], index=0)
        ollama_host = st.text_input("Ollama Host", value=DEFAULT_OLLAMA_HOST).strip()

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

            gemini_api_key = None
        else:
            model = st.text_input("Gemini Model", value="gemini-2.5-pro").strip()
            gemini_api_key = st.text_input(
                "Gemini API Key",
                value=os.environ.get("GOOGLE_API_KEY", ""),
                type="password",
            ).strip() or None

        event_types = st.multiselect(
            "Event Types",
            options=["query", "ingestion", "query_error", "ingestion_error", "log_parse_error"],
            default=["query", "ingestion"],
            help="Restrict validation to specific event types.",
        )
        max_events_raw = st.number_input(
            "Max Events",
            min_value=1,
            value=10,
            step=1,
            help="Limit how many matching events are validated in one run.",
        )
        output_base = st.text_input(
            "Output Base Path (optional)",
            value="",
            help="If provided, the validator writes `<base>.json` and `<base>.md`.",
        ).strip() or None

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

        # ── Live progress UI ──────────────────────────────────────────────
        progress_bar = st.progress(0, text="Starting validation…")
        status_box = st.empty()
        results_log = st.empty()
        _live_rows: List[str] = []

        RISK_ICON = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}

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
            accuracy = parsed.get("accuracy_score", "")
            icon = RISK_ICON.get(risk, "⚪")
            err = result.get("parse_error")
            if err:
                row = f"❌ **Line {line_no}** `{event_type}` — parse error: `{err[:80]}`"
            else:
                row = (
                    f"{icon} **Line {line_no}** `{event_type}` "
                    f"— score `{score}` · accuracy `{accuracy}` · risk `{risk or 'n/a'}`"
                )
            _live_rows.append(row)

            # Show last 8 rows so the box doesn't grow forever
            results_log.markdown("\n\n".join(_live_rows[-8:]))
            status_box.caption(f"Progress: {pct}% ({current}/{total} events processed)")

        try:
            report, json_path, md_path = run_validation(
                log_file=log_file,
                provider=provider,
                model=model,
                ollama_host=ollama_host,
                gemini_api_key=gemini_api_key,
                event_types=event_types or None,
                max_events=int(max_events_raw),
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
