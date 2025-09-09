# config.py
from pathlib import Path
import os

# --- Core Directories ---
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
RAW_TEXT_DIR = OUTPUT_DIR / "raw_text"
SUMMARY_DIR = OUTPUT_DIR / "summaries"
PSEUDOCODE_DIR = OUTPUT_DIR / "pseudocode"
CODE_DIR = OUTPUT_DIR / "code"
REPORT_DIR = OUTPUT_DIR / "reports"
# DIAGRAM_DIR = OUTPUT_DIR / "diagrams" # Diagrams are no longer a primary output
RESEARCH_PAPERS_DIR = BASE_DIR / "research_papers"

# --- Create Directories ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
PSEUDOCODE_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
# DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# --- Ollama Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# General context window for Ollama models. Agents can have specific env vars to override.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", 8192)) # Default, e.g., for /7b

# --- Model Selection ---
# Ollama models
_DEFAULT_OLLAMA_MODEL = "deepseek-r1:1.5b" # A common, smaller Gemma model for Ollama
LLM_MODEL_GENERAL = os.getenv("LLM_MODEL_GENERAL", _DEFAULT_OLLAMA_MODEL) # For summary, pseudocode section extraction (Ollama)
LLM_MODEL_PSEUDOCODE_GEN = os.getenv("LLM_MODEL_PSEUDOCODE_GEN", LLM_MODEL_GENERAL) # Specific for pseudocode generation if different (Ollama)
LLM_MODEL_QA = os.getenv("LLM_MODEL_QA", _DEFAULT_OLLAMA_MODEL) # For Q&A RAG (Ollama)
LLM_MODEL_EMBEDDING = os.getenv("LLM_MODEL_EMBEDDING", "nomic-embed-text") # Defaulting to a common Ollama embedding model

# OpenRouter models (used by CodeGenerationAgent)
OPENROUTER_MODEL_CODE = os.getenv("OPENROUTER_MODEL_CODE", "openai/gpt-4o-mini") # For code generation via OpenRouter

# --- LLM Settings ---
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS_COMPLETION = 2048
TEMPERATURE_SYNTAX = 0.0 # For tasks requiring precise, deterministic output

# --- QnA Chatbot Specific ---
DEFAULT_CHUNK_SIZE = 5000
DEFAULT_CHUNK_OVERLAP = 700
DEFAULT_RETRIEVER_K = 5
#which cnn model got the best results for mammogram classification as normal and abnormal
# --- Report Generation ---
REPORT_FONT = "Arial"
REPORT_TITLE_FONT_SIZE = 16
REPORT_HEADING_FONT_SIZE = 12
REPORT_BODY_FONT_SIZE = 10
REPORT_CODE_FONT_SIZE = 9

# --- Logging ---
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
AGENT_LOG_FILE = LOG_DIR / "orchestrator_main.log" # Main orchestrator log
PDF_PARSER_LOG_FILE = LOG_DIR / "pdf_parser_advanced.log"
SUMMARIZER_AGENT_LOG_FILE = LOG_DIR / "summarizer.log"
PSEUDOCODE_AGENT_LOG_FILE = LOG_DIR / "pseudocode_agent.log"
CODE_GENERATOR_AGENT_LOG_FILE = LOG_DIR / "code_generator.log"
REPORT_GENERATOR_AGENT_LOG_FILE = LOG_DIR / "report_generator.log"
# Add other agent logs if needed

# --- Print Configuration on Load ---
print(f"--- Configuration Loaded (config.py) ---")
print(f"  Ollama Base URL: {OLLAMA_BASE_URL}")
print(f"  Ollama General Context Window (num_ctx): {OLLAMA_NUM_CTX}")
print(f"  --- Ollama Models ---")
print(f"    General LLM Model (Summary, Section Extract): {LLM_MODEL_GENERAL}")
print(f"    Pseudocode Generation LLM Model: {LLM_MODEL_PSEUDOCODE_GEN}")
print(f"    QA LLM Model: {LLM_MODEL_QA}")
print(f"    Embedding Model: {LLM_MODEL_EMBEDDING}")
print(f"  --- OpenRouter Models ---")
print(f"    Code Generation Model (OpenRouter): {OPENROUTER_MODEL_CODE}")
print(f"  --- LLM Settings ---")
print(f"    Default Temperature: {DEFAULT_TEMPERATURE}")
print(f"    Syntax Temperature: {TEMPERATURE_SYNTAX}")
print(f"  --- RAG Settings ---")
print(f"    RAG Chunk Size (Default): {DEFAULT_CHUNK_SIZE}")
print(f"  --- Output & Logging ---")
print(f"    Output Base Directory: {OUTPUT_DIR}")
print(f"    Main Orchestrator Log: {AGENT_LOG_FILE}")
print(f"--------------------------------------")