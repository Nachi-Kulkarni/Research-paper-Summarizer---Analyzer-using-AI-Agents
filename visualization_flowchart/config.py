# config.py
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress urllib3 warnings
import urllib3
try:
    urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
except AttributeError:
    # NotOpenSSLWarning was removed in urllib3 2.0.0.
    # If it doesn't exist, we don't need to disable it.
    pass

# --- Core Directories ---
BASE_DIR = Path(__file__).resolve().parent # Use resolve() for robustness
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
RAW_TEXT_DIR = OUTPUT_DIR / "raw_text"  # For extracted raw text files
PSEUDOCODE_DIR = OUTPUT_DIR / "pseudocode" # Intermediate storage
DIAGRAM_DIR = OUTPUT_DIR / "diagrams"      # Final diagrams
FINAL_OUTPUT_DIR = OUTPUT_DIR / "final_results" # Combined markdown outputs
RESEARCH_PAPERS_DIR = BASE_DIR / "research_papers"
TEMP_DIR = BASE_DIR / "temp_files" # Directory for temporary files

# --- Create Directories ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
PSEUDOCODE_DIR.mkdir(parents=True, exist_ok=True)
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_PAPERS_DIR.mkdir(parents=True, exist_ok=True) # Ensure input dir exists
TEMP_DIR.mkdir(parents=True, exist_ok=True) # Ensure temp dir exists

# --- OpenRouter Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY must be set in environment variables")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free")  

# --- LLM Settings ---
# Moderate temperature for creative tasks like summary/pseudo/code
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9  # Nucleus sampling parameter (0-1)
# Stricter temperature for syntax generation (Mermaid)
TEMPERATURE_SYNTAX = 0.1
MAX_CONTENT_LENGTH_FOR_DIAGRAM_PROMPT = 40000 # Max characters of paper content for diagram prompt

# --- Mermaid / MMDC Settings ---
MMDC_CHECK_TIMEOUT = 15
MMDC_RENDER_TIMEOUT = 60
PNG_BACKGROUND_COLOR = "white" # Simple background
# CSS_FILENAME = "diagram_style.css" # Optional: Add back if you have a CSS file
DEFAULT_DIAGRAM_STYLE_CSS = BASE_DIR / "diagram_style.css"
MMDC_DEFAULT_THEME = "default" # Options: "default", "dark", "forest", "neutral"
MMDC_DEFAULT_BG_COLOR = "white" # Background color for mmdc

# --- Logging ---
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
# Define specific log file paths
MAIN_PROCESS_LOG_FILE = LOG_DIR / "process_text_ollama.log"
DIAGRAM_GEN_LOG_FILE = LOG_DIR / "generate_diagram.log"
PDF_PARSER_LOG_FILE = LOG_DIR / "pdf_parser_advanced.log"
