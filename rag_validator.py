import os
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
import json
import logging
import csv
import asyncio
from typing import List, Dict, Optional

# NEW: Fix for Python async event loops to prevent DeepEval deadlocks!
import nest_asyncio
nest_asyncio.apply()

# LangChain document loaders for the Generator
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
# Attempt to import DeepEval components
try:
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        GEval
    )
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.models.base_model import DeepEvalBaseLLM
    DEEPEVAL_AVAILABLE = True
except Exception as e:
    DEEPEVAL_AVAILABLE = False
    print(f"🚨 ACTUAL IMPORT ERROR: {e}")

# --- Helper Function: Read Raw Files for Question Generation ---
def load_raw_document(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw file not found: {file_path}")
    
    ext = file_path.lower().split('.')[-1]
    print(f"📄 [DEBUG] Loading raw document: {file_path} (Type: {ext})")
    
    try:
        if ext == 'pdf':
            loader = PyPDFLoader(file_path)
        elif ext == 'md':
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding='utf-8')
            
        docs = loader.load()
        full_text = "\n\n".join([doc.page_content for doc in docs])
        print(f"✅ [DEBUG] Successfully extracted {len(full_text)} characters from document.")
        return full_text
    except Exception as e:
        print(f"❌ [DEBUG] Failed to read file {file_path}. Error: {e}")
        return ""

# --- Custom LLM Judge Wrapper ---
class UniversalJudgeLLM(DeepEvalBaseLLM if DEEPEVAL_AVAILABLE else object):
    def __init__(self, provider: str, model_name: str, api_key: str = None, ollama_url: str = "http://127.0.0.1:11434"):
        self.model_name = model_name
        self.provider = provider.lower()
        
        if self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            import google.generativeai as genai
            
            safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
            
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=0,  
                safety_settings=safety_settings
            )
        elif self.provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            self.llm = ChatOllama(
                model=self.model_name, 
                temperature=0, 
                base_url=ollama_url
            )
        else:
            raise ValueError("Provider must be 'gemini' or 'ollama'")

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content

    # CRITICAL FIX: This MUST be a true async function using ainvoke!
    async def a_generate(self, prompt: str) -> str:
        response = await self.llm.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return f"{self.provider}:{self.model_name}"

# --- STEP 1: Synthetic Question Generator ---
class SyntheticDataGenerator:
    def __init__(self, judge_llm: UniversalJudgeLLM):
         self.judge = judge_llm
        
    def generate_gold_dataset(self, raw_text: str, num_questions: int = 3, output_json_path: str = "synthetic_gold_dataset.json"):
        import random
        chunk_size = 10000
        
 
        if len(raw_text) > (chunk_size + 6000):
            start_idx = random.randint(6000, len(raw_text) - chunk_size)
            sample_context = raw_text[start_idx : start_idx + chunk_size]
        else:
            sample_context = raw_text
            
        print(f"🧠 [DEBUG] Sending {len(sample_context):,} characters to Judge to generate questions...")
        prompt = f"""
        You are a curious employee using an AI Assistant to understand a complex company document.
        Read the following source text, and generate EXACTLY {num_questions} Question, Answer, and Context mappings.
        
        STRICT RULES:
        1. The questions MUST be conversational and analytical (e.g., "How does the system handle...", "What should I do if...", "Can you explain why...", "give a summary of ...").
        2. DO NOT ask robotic trivia questions (Never ask for section numbers, document version numbers, or copyright owners).
        3. Formulate the question exactly how a human would type it into a chat window.
        Source Text:
        {sample_context}

        Output STRICT valid JSON in the following schema:
        [
            {{
                "question": "...",
                "expected_answer": "...",
                "context_used": ["exact quote from text covering the answer"]
            }}
        ]
        """
        response = self.judge.generate(prompt)
        
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
                
            dataset = json.loads(response)
            with open(output_json_path, "w", encoding="utf-8") as out_f:
                 json.dump(dataset, out_f, indent=4)
            return dataset
        except Exception as e:
             print(f"❌ [DEBUG] Judge failed to format JSON properly. Error: {e}")
             return None

# --- STEP 2: Telemetry Log Parser ---
class TelemetryLogParser:
    @staticmethod
    def parse_logs_to_test_cases(log_filepath: str, gold_dataset_path: str = "synthetic_gold_dataset.json") -> List[Dict]:
        print(f"\n🔍 [DEBUG] Starting Telemetry Parser...")
        if not os.path.exists(log_filepath):
            print(f"❌ [DEBUG] Log file not found at {log_filepath}")
            return []

        gold_map = {}
        if os.path.exists(gold_dataset_path):
            try:
                with open(gold_dataset_path, "r", encoding="utf-8") as gf:
                    gold_data = json.load(gf)
                    for item in gold_data:
                        clean_q = item.get("question", "").strip().lower()
                        gold_map[clean_q] = item.get("expected_answer", "")
                print(f"🏆 [DEBUG] Successfully loaded {len(gold_map)} 'Expected Answers' from Golden Dataset.")
            except Exception as e:
                print(f"❌ [DEBUG] Failed to read Golden Dataset: {e}")

        test_cases = []
        parsed_count = 0
        skipped_count = 0
        print(f"📖 [DEBUG] Opening log file: {log_filepath}")
        
        with open(log_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip(): continue
                try:
                    log_entry = json.loads(line)
                    if log_entry.get("event_type") != "query":
                        continue
                        
                    parsed_count += 1
                    question = log_entry.get("question", "")
                    answer = log_entry.get("answer", "")
                    
                    raw_chunks = log_entry.get("retrieved_chunks", [])
                    context_blocks = []
                    for chunk in raw_chunks:
                        if "page_content" in chunk:
                            context_blocks.append(chunk["page_content"])
                    
                    clean_q_key = question.strip().lower()
                    if clean_q_key in gold_map:
                        expected_output = gold_map[clean_q_key]
                        print(f"  ✅ [DEBUG] Matched Query to Golden Dataset: '{question}'")
                        parsed_count += 1
                        
                        raw_chunks = log_entry.get("retrieved_chunks", [])
                        context_blocks = [chunk["page_content"] for chunk in raw_chunks if "page_content" in chunk]

                        test_cases.append({
                            "question": question,
                            "answer": answer,
                            "context": context_blocks,
                            "expected_output": expected_output,
                            "execution_time": round(log_entry.get("query_time_seconds", 0), 2),
                            "cache_hit": log_entry.get("semantic_cache_hit", False),
                            "is_tabular": log_entry.get("used_tabular_logic", False)
                        })
                    else:
                        skipped_count += 1
                except json.JSONDecodeError:
                    pass
                    
        print(f"✅ [DEBUG] Parser finished! Extracted {parsed_count} full QA pairs from the logs.")
        return test_cases

# --- STEP 3: DeepEval Evaluator ---
if DEEPEVAL_AVAILABLE:
    class RAGEvaluator:
        def __init__(self, judge_llm: UniversalJudgeLLM, strict_mode: bool = False):
            self.judge = judge_llm
            self.results = []
            
            self.metrics = {
                "faithfulness": FaithfulnessMetric(threshold=0.5, model=self.judge, include_reason=True, async_mode=False, strict_mode=strict_mode),
                "answer_relevancy": AnswerRelevancyMetric(threshold=0.5, model=self.judge, include_reason=True, async_mode=False, strict_mode=strict_mode),
                "contextual_precision": ContextualPrecisionMetric(threshold=0.5, model=self.judge, include_reason=True, async_mode=False, strict_mode=strict_mode)
            }

        def evaluate_response(self, tc: Dict) -> Dict:
            question = tc.get("question", "")
            generated_answer = tc.get("answer", "")
            retrieved_context = tc.get("context", [])
            expected_output = tc.get("expected_output", generated_answer)
            cache_hit = tc.get("cache_hit", False)

            print(f"\n==================================================")
            print(f"⚖️  [JUDGE] Now Evaluating Question: '{question}'")
            print(f"==================================================")
            
            report = {
                "query": question,
                "execution_time": tc.get("execution_time", 0),
                "cache_hit": cache_hit,
                "metrics": {},
                "passed_all": True
            }

            if tc.get("is_tabular", False):
                print("  ⏭️  [DEBUG] Tabular query detected. Bypassing LLM Judge metrics.")
                report["metrics"]["tabular_match"] = {"score": 1.0, "passed": True, "reasoning": "Tabular bypass."}
                self.results.append(report)
                return report

            test_case = LLMTestCase(
                input=question,
                actual_output=generated_answer,
                retrieval_context=retrieved_context,
                expected_output=expected_output
            )

            for name, metric_obj in self.metrics.items():
                if cache_hit and name == "contextual_precision":
                    print(f"  ⏭️  [DEBUG] Skipping {name} (Cache Hit - No DB search occurred)")
                    continue
                    
                try:
                    print(f"  ⏳ [JUDGE] Measuring {name.upper()}... (Talking to Ollama)")
                    metric_obj.measure(test_case)
                    print(f"  ✅ [JUDGE] {name.upper()} Score: {metric_obj.score} | Passed: {metric_obj.is_successful}")
                    print(f"      Reason: {metric_obj.reason[:100]}...")
                    
                    report["metrics"][name] = {
                        "score": round(metric_obj.score, 4),    
                        "passed": metric_obj.is_successful,
                        "reasoning": metric_obj.reason          
                    }
                    if not metric_obj.is_successful: report["passed_all"] = False
                except Exception as e:
                    print(f"  ❌ [JUDGE] ERROR calculating {name}: {str(e)}")
                    report["metrics"][name] = {"score": 0.0, "passed": False, "reasoning": f"ERROR: {str(e)}"}

            self.results.append(report)
            return report

        def batch_evaluate(self, test_cases: List[Dict]) -> List[Dict]:
            for idx, tc in enumerate(test_cases):
                self.evaluate_response(tc)
            return self.results

        def export_report(self, filepath: str = "validation_report.csv"):
            if not self.results: return
            print(f"\n💾 [DEBUG] Exporting final CSV Report to '{filepath}'")
            headers = ["Query", "Status", "Time (sec)", "Cache Hit", "Faithfulness", "Relevancy", "Precision", "Error Snippet"]
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for r in self.results:
                    m = r["metrics"]
                    time_sec = r.get("execution_time", "N/A")
                    cache = "Yes" if r.get("cache_hit") else "No"
                    f_score = m.get("faithfulness", {}).get("score", "N/A")
                    r_score = m.get("answer_relevancy", {}).get("score", "N/A")
                    p_score = m.get("contextual_precision", {}).get("score", "N/A")
                    
                    reasoning = "Clean pass."
                    if not r["passed_all"]:
                        failed_reasons = []
                        for metric_name, metric_data in m.items():
                            if not metric_data.get("passed", True):
                                full_reason = metric_data.get("reasoning", "No reason provided.")
                                clean_reason = full_reason.replace("\n", " ") 
                                failed_reasons.append(f"[{metric_name.upper()}]: {clean_reason}")
                        reasoning = " | ".join(failed_reasons)
                    writer.writerow([r["query"], "PASS" if r["passed_all"] else "FAIL", time_sec, cache, f_score, r_score, p_score, reasoning])
        def print_diagnostic_summary(self):
            if not self.results: return
            print("\n" + "="*50 + "\n🧠 AI DIAGNOSTIC ENGINE INITIALIZING...\n" + "="*50)

            failures = []
            for r in self.results:
                if not r["passed_all"]:
                    issue_desc = f"Question: {r['query']}\n"
                    for metric_name, metric_data in r.get("metrics", {}).items():
                        if not metric_data.get("passed", True):
                            issue_desc += f" - Failed {metric_name.upper()} (Score: {metric_data.get('score')}): {metric_data.get('reasoning')}\n"
                    failures.append(issue_desc)

            if not failures:
                print("🟢 SYSTEM HEALTHY: The Judge found no errors. Your RAG pipeline is perfect!\n" + "="*50)
                return

            print(f"⏳ Analyzing {len(failures)} failed queries to determine root causes...\n")
            
            diagnostic_prompt = f"""
            You are an elite AI Systems Engineer diagnosing a Python Streamlit RAG application.
            Architecture: Chroma DB (Vector), Llama 3.2 (Generator LLM), optional Cross-Encoder Re-ranker.

            Here are the specific metric failures and reasoning from the latest DeepEval test run:
            --- ERROR LOGS ---
            {"\n".join(failures)}
            ------------------

            TASK:
            Based on the errors above, provide a strict, bulleted ACTION PLAN on exactly what parameters the developer must change in their bot's code.
            Structure your response strictly into these categories:
            1. 🔍 RETRIEVAL FIXES (e.g., Increase Top-K `k` value, Chunk Size, enable Re-ranker, change chunk overlap).
            2. ✍️ GENERATION FIXES (e.g., Lower LLM Temperature, inject specific anti-hallucination rules into the system prompt).
            3. 🎯 INTENT FIXES (e.g., Improve Standalone Question re-writer prompt).

            Do not write generic apologies or summaries. Be highly technical, direct, and specify the exact parameters to tune based on the exact errors provided.
            """
            try:
                action_plan = self.judge.generate(diagnostic_prompt)
                print("🩺 --- AI JUDGE ARCHITECTURAL RECOMMENDATIONS ---")
                print(action_plan.strip())
            except Exception as e:
                print(f"❌ Failed to generate AI diagnostic: {e}")
            print("\n" + "="*50)

else:
    class RAGEvaluator: pass
# --- STEP 4: Data Loss Audit ---
def run_data_loss_audit(db_path: str, embed_model_name: str, raw_file_path: str):
    print("\n" + "="*50)
    print("🕵️  CHROMA DATABASE HEALTH AUDIT  🕵️")
    print("="*50)

    if not os.path.exists(db_path):
        print(f"❌ ERROR: Database folder not found at {db_path}")
        return

    raw_characters = 0
    if os.path.exists(raw_file_path):
        raw_text = load_raw_document(raw_file_path)
        raw_characters = len(raw_text)
    else:
        print(f"⚠️ Raw file not found at {raw_file_path}. Skipping comparison.")

    try:
        if embed_model_name == "all-MiniLM-L6-v2":
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        else:
            embeddings = OllamaEmbeddings(model=embed_model_name, base_url="http://127.0.0.1:11434")
            
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        db_data = vectorstore.get()
        
        target_filename = os.path.basename(raw_file_path) if raw_file_path else ""
        total_chunks_in_db = len(db_data.get("ids", []))
        unique_sources = {os.path.basename(meta["source"]) for meta in db_data.get("metadatas", []) if meta and "source" in meta}

        # Filter the database to ONLY count chunks from the specific file
        file_specific_characters = 0
        file_specific_chunks = 0
        for i, meta in enumerate(db_data.get("metadatas", [])):
            if meta and "source" in meta and os.path.basename(meta["source"]) == target_filename:
                file_specific_characters += len(db_data["documents"][i])
                file_specific_chunks += 1

        print("-" * 50)
        print("📊 INTEGRITY METRICS:")
        print("-" * 50)
        if raw_characters > 0:
            print(f"🎯 Target File:               {target_filename}")
            print(f"📄 Raw File Characters:       {raw_characters:,}")
            print(f"💾 DB Characters (This file): {file_specific_characters:,}")
            print(f"🧩 DB Chunks (This file):     {file_specific_chunks:,}")
            
            # Use a 95% threshold to account for LangChain stripping invisible formatting/spaces
            if file_specific_characters >= (raw_characters * 0.95):
                print("✅ DATA LOSS CHECK:          PASS (Data is fully embedded)")
            else:
                print("❌ DATA LOSS CHECK:          FAIL (Database is missing data for this file!)")
            print("-" * 50)

        print(f"✅ Total Chunks Stored:      {total_chunks_in_db:,}")
        print(f"✅ Unique Files Found:       {len(unique_sources)}")
        for source in unique_sources: 
            print(f"   📄 -> {source}")

    except Exception as e:
        print(f"\n❌ AUDIT FAILED: {e}")
# --- MAIN WORKFLOW ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Setup
    judge_model = UniversalJudgeLLM(provider="ollama", model_name="qwen2.5:14b")
    evaluator = RAGEvaluator(judge_llm=judge_model)

    print("\n" + "="*50)
    print("🤖 ENTERPRISE RAG PIPELINE VALIDATOR")
    print("="*50)
    raw_file = input("Enter the filename (e.g., my_document.pdf): ")
    log_path = input("Enter the path to your log file (e.g., user_data/admin1/kb_log_vision_kb2.jsonl): ")
    print("Choose an action:")
    print("1: Generate Test Questions from a PDF")
    print("2: Grade Chatbot Logs")
    print("3: Check data loss")
    choice = input("\nEnter 1 or 2 or 3: ")

    if choice == "1":
        # ---------------------------------------------------------
        # MODE 1: GENERATE QUESTIONS
        # ---------------------------------------------------------
        # raw_file = input("Enter the filename (e.g., my_document.pdf): ")
        raw_text = load_raw_document(raw_file)
        if raw_text:
            generator = SyntheticDataGenerator(judge_model)
            print("Generating 3 test questions... Please wait.")
            gold_data = generator.generate_gold_dataset(raw_text, num_questions=3)
            if gold_data:
                print("\n✅ Success! Here are your test questions:")
                for item in gold_data:
                    print(f"- {item['question']}")
                print("\n👉 ACTION REQUIRED: Open your Chatbot app, ask it these 3 questions, then come back and run Option 2 to grade the logs!")

    elif choice == "2":
        # ---------------------------------------------------------
        # MODE 2: GRADE THE LOGS
        # ---------------------------------------------------------
        # log_path = input("Enter the path to your log file (e.g., user_data/admin1/kb_log_vision_kb2.jsonl): ")
        
        test_cases = TelemetryLogParser.parse_logs_to_test_cases(log_path)
        
        if not test_cases:
            print("❌ No valid queries found in the log file.")
        else:
            print(f"\nFound {len(test_cases)} queries in the log file. Grading now...\n")
            evaluator.batch_evaluate(test_cases)
            evaluator.export_report("validation_report.csv")
            evaluator.print_diagnostic_summary()
            print("✅ Grading Complete! Check validation_report.csv")

    elif choice == "3":  # <--- NEW LOGIC FOR OPTION 3
        print("\n--- Running Data Loss Audit ---")
        db_path = input("Enter the path to your Chroma DB folder (e.g., ./user_data/admin1/chroma_db_vision_kb): ")
        embed_model = input("Enter the exact embedding model name (e.g., nomic-embed-text:latest): ")
        # raw_file = input("Enter the raw PDF filename to compare against: ")
        
        run_data_loss_audit(db_path, embed_model, raw_file)
    else:
        print("Invalid choice.")
