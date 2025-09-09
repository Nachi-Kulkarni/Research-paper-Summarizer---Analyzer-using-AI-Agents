# generate_diagram.py

import argparse
import logging
from pathlib import Path
import sys
import os
import re
import subprocess
from typing import Optional, Dict, List, Set, Tuple, Literal
import time

# --- Third-party Imports ---
import requests
import json

# --- Local Imports ---
import config # Shared configuration

# --- Logger Setup ---
log_file = config.LOG_DIR / "generate_diagram.log"
logger = logging.getLogger("diagram_generator")
if not logger.hasHandlers():
    # File Handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO) # Default console level
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG) # Default logger level, handlers will filter
    logger.propagate = False

# --- Constants ---
SUPPORTED_DIAGRAM_TYPES = Literal["flowchart", "sequence", "class", "state", "er", "gantt", "pie", "mindmap", "timeline"]
DEFAULT_DIAGRAM_TYPE: SUPPORTED_DIAGRAM_TYPES = "flowchart"
DEFAULT_MAX_ELEMENTS = 10 # Default for the argument

# --- Helper Functions ---

def invoke_openrouter(
    system_prompt: str,
    user_prompt: str,
    description: str,
    temperature: float = config.TEMPERATURE_SYNTAX
) -> str:
    """Invoke OpenRouter API with proper error handling and logging."""
    try:
        logger.info(f"Generating {description} via OpenRouter (model: {config.OPENROUTER_MODEL})...")
        logger.debug(f"System Prompt: {system_prompt}")
        logger.debug(f"User Prompt: {user_prompt}")
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo/your-project", 
            "X-Title": "AI Diagram Generator" 
        }
        payload = {
            "model": config.OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature
        }

        response = requests.post(
            f"{config.OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=90 
        )
        response.raise_for_status()

        response_data = response.json()
        result = response_data["choices"][0]["message"]["content"].strip()
        logger.debug(f"OpenRouter raw response for {description}: {result}")

        if not result:
            raise ValueError(f"Empty response for {description}")
        try:
            json_result = json.loads(result)
            if "error" in json_result:
                raise ValueError(f"LLM returned an error: {json_result['error']}")
        except json.JSONDecodeError:
            pass 

        if "[Error" in result or "[LLM Error]" in result.upper():
             raise ValueError(f"LLM indicated an error in its response for {description}: {result[:200]}")

        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for {description}: {e}")
        return f"[Error: API request failed - {e}]"
    except (ValueError, KeyError, IndexError) as e:
        logger.error(f"Failed to process LLM response for {description}: {e}")
        return f"[Error: Processing LLM response - {e}]"
    except Exception as e:
        logger.error(f"Unexpected error generating {description}: {e}")
        return f"[Error: Unexpected - {e}]"

def _clean_mermaid_syntax_basic(raw_syntax: str) -> str:
    """Enhanced cleaning: removes markdown fences, explanatory text, and fixes common issues."""
    logger.debug("Applying enhanced Mermaid syntax cleaning...")
    clean_syntax = raw_syntax.strip()
    
    # Remove markdown code fences
    match = re.match(r"^```(?:mermaid)?\s*(.*?)\s*```$", clean_syntax, re.DOTALL | re.IGNORECASE)
    if match:
        clean_syntax = match.group(1).strip()
    
    # Split into lines and filter out explanatory text
    lines = clean_syntax.split('\n')
    diagram_lines = []
    found_diagram_start = False
    
    for line in lines:
        stripped_line = line.strip()
        
        # Skip empty lines and comments
        if not stripped_line or stripped_line.startswith('%%'):
            if found_diagram_start:
                diagram_lines.append(line)
            continue
            
        # Check if this is a diagram declaration
        if re.match(r'^\s*(graph\s+|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|mindmap|timeline)', stripped_line, re.IGNORECASE):
            found_diagram_start = True
            diagram_lines.append(line)
            continue
            
        # If we've started the diagram, include lines that look like diagram syntax
        if found_diagram_start:
            # Include lines that contain arrows, node definitions, or other diagram elements
            if ('-->' in stripped_line or 
                '--' in stripped_line or 
                re.match(r'^\s*[A-Za-z0-9_]+\s*[\[\(]', stripped_line) or
                '->' in stripped_line or
                ': ' in stripped_line):
                diagram_lines.append(line)
            # Stop if we hit explanatory text (lines starting with words like "Key", "The", etc.)
            elif re.match(r'^\s*(Key|The|This|Note|Explanation|Changes|made|Fixed)', stripped_line, re.IGNORECASE):
                break
        else:
            # Before diagram start, skip obvious explanatory text
            if not re.match(r'^\s*(Key|The|This|Note|Here|Below|Above)', stripped_line, re.IGNORECASE):
                diagram_lines.append(line)
    
    clean_syntax = '\n'.join(diagram_lines).strip()
    
    # Fix common syntax issues
    # Replace literal \n with space
    clean_syntax = re.sub(r'\\n', ' ', clean_syntax)
    
    # Remove problematic characters from node labels
    # Fix node labels with parentheses - replace (text) with just text
    clean_syntax = re.sub(r'\[([^[\]]*)\(([^()]*)\)([^[\]]*)\]', r'[\1\2\3]', clean_syntax)
    
    return clean_syntax

def _get_diagram_declaration(syntax: str) -> Optional[str]:
    """Extracts the diagram declaration (e.g., 'graph TD', 'sequenceDiagram'), ignoring leading comments."""
    declaration_found = None
    lines = syntax.splitlines()
    
    patterns = [
        r"^\s*(graph\s+(?:LR|TD|RL|BT);?)", 
        r"^\s*(sequenceDiagram)",
        r"^\s*(classDiagram)",
        r"^\s*(stateDiagram(?:-v2)?)",
        r"^\s*(erDiagram)",
        r"^\s*(gantt)",
        r"^\s*(pie)",
        r"^\s*(mindmap)",
        r"^\s*(timeline)"
    ]

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("%%"): 
            continue
        
        if stripped_line.lower().startswith("title "):
            logger.debug(f"Found 'title' directive in syntax: '{stripped_line[:50]}...'. This should not be generated by the LLM in the Mermaid code itself.")
            continue 
            
        for pattern in patterns:
            match = re.match(pattern, stripped_line, re.IGNORECASE)
            if match:
                declaration_found = match.group(1)
                return declaration_found 
        
        if declaration_found is None: 
            pass
            
    return declaration_found

def _ensure_declaration(syntax: str, expected_type: SUPPORTED_DIAGRAM_TYPES = "flowchart", orientation: str = "LR") -> str:
    """Ensures a Mermaid diagram declaration is present, adding a default if missing."""
    current_decl_str = _get_diagram_declaration(syntax) 

    if current_decl_str:
        if current_decl_str.lower().startswith("graph") and not current_decl_str.rstrip().endswith(';'):
            pattern_to_fix = r"^(\s*" + re.escape(current_decl_str) + r")(\s*(?!\s*;))"
            if re.match(pattern_to_fix, syntax, re.IGNORECASE):
                 syntax = re.sub(pattern_to_fix, r"\1;", syntax, count=1, flags=re.IGNORECASE)
                 logger.debug(f"Added semicolon to flowchart declaration: '{current_decl_str};'")
        return syntax.strip()

    logger.warning(f"Mermaid syntax declaration missing. Attempting to add for type: {expected_type}")
    default_syntax_map = {
        "flowchart": f"graph {orientation.upper()};\n",
        "sequence": "sequenceDiagram\n",
        "class": "classDiagram\n",
        "state": "stateDiagram-v2\n",
        "er": "erDiagram\n",
        "gantt": "gantt\n    dateFormat YYYY-MM-DD\n    %% title Default Gantt Chart (auto-added)\n", # Comment out auto-added titles
        "pie": "pie\n    %% title Default Pie Chart (auto-added)\n", # Comment out auto-added titles
        "mindmap": "mindmap\n",
        "timeline": "timeline\n    %% title Default Timeline (auto-added)\n" # Comment out auto-added titles
    }
    
    prefix_syntax = ""
    if expected_type in default_syntax_map:
        prefix_syntax = default_syntax_map[expected_type]
        # For gantt, pie, timeline, if the incoming syntax is very short (likely just content)
        # use the full default prefix. Otherwise, just use the first line (declaration).
        if expected_type not in ["gantt", "pie", "timeline"] or len(syntax.strip().splitlines()) >= 2:
            # Ensure only the declaration part is used if it's multi-line
            prefix_syntax = prefix_syntax.splitlines()[0] + "\n" 
        return prefix_syntax + syntax.strip()

    logger.error(f"Unsupported diagram type '{expected_type}' for adding default declaration. Returning original.")
    return syntax.strip()


def _is_valid_mermaid_file(filepath: Path) -> bool:
    """Checks if a file seems to be a valid, non-empty Mermaid file for mmdc."""
    if not filepath.exists() or filepath.stat().st_size < 5:
        return False
    content = filepath.read_text(encoding='utf-8').strip()
    if not content: return False
    return _get_diagram_declaration(content) is not None


def validate_mermaid_syntax_with_mmdc(mermaid_syntax: str, temp_dir: Path) -> Tuple[bool, str]:
    """Validates Mermaid syntax by attempting a dry-run render with mmdc."""
    logger.debug("Validating Mermaid syntax with mmdc...")
    if not mermaid_syntax.strip():
        return False, "Syntax is empty."

    temp_mmd_file = temp_dir / "validation_temp.mmd"
    temp_validation_output_image = temp_dir / "validation_temp_image.png"

    try:
        temp_mmd_file.write_text(mermaid_syntax, encoding='utf-8')
        if not _is_valid_mermaid_file(temp_mmd_file):
             return False, "Generated syntax file is invalid or empty before mmdc check (missing diagram declaration)."

        try:
            subprocess.run(["mmdc", "--version"], check=True, capture_output=True, timeout=10)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error(f"mmdc not available or version check failed: {e}")
            return False, "mmdc tool not found or not working. Cannot validate or render."

        css_file = config.DEFAULT_DIAGRAM_STYLE_CSS if config.DEFAULT_DIAGRAM_STYLE_CSS.exists() else None
        cmd = ["mmdc", "-i", str(temp_mmd_file), "-o", str(temp_validation_output_image)]
        if css_file: # Use the global config default if no specific one is passed to render
            cmd.extend(["-C", str(css_file)])
        
        # Add background color from config for validation as well, to mimic render
        cmd.extend(["-b", config.MMDC_DEFAULT_BG_COLOR])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20) 

        if result.returncode == 0 and temp_validation_output_image.exists() and temp_validation_output_image.stat().st_size > 100:
            logger.debug("mmdc validation successful.")
            return True, ""
        else:
            error_msg = f"mmdc validation failed. Return code: {result.returncode}.\nCommand: {' '.join(cmd)}\nStderr: {result.stderr}\nStdout: {result.stdout}"
            logger.warning(error_msg)
            detailed_error = result.stderr if result.stderr else result.stdout
            if not detailed_error: detailed_error = "mmdc failed without specific error output."
            return False, detailed_error

    except Exception as e:
        logger.error(f"Exception during mmdc validation: {e}")
        return False, f"Exception during validation: {str(e)}"
    finally:
        if temp_mmd_file.exists(): temp_mmd_file.unlink(missing_ok=True)
        if temp_validation_output_image.exists(): temp_validation_output_image.unlink(missing_ok=True)


def render_mermaid_diagram(
    mermaid_syntax: str,
    output_path_base: Path,
    output_format: Literal["png", "svg", "mmd"] = "png",
    css_file_path: Optional[Path] = None # This is the one passed from main()
) -> bool:
    """Renders Mermaid syntax to PNG or SVG using mmdc, and saves .mmd."""
    output_path = output_path_base.with_suffix(f".{output_format}")
    syntax_output_path = output_path_base.with_suffix('.mmd')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        syntax_output_path.write_text(mermaid_syntax, encoding='utf-8')
        logger.info(f"Saved final Mermaid syntax to: {syntax_output_path}")
    except IOError as e:
        logger.error(f"Failed to save .mmd file {syntax_output_path}: {e}")
        return False 

    if output_format == "mmd": 
        logger.info(f"Only .mmd output requested. Diagram rendering skipped.")
        return True

    try:
        if not _is_valid_mermaid_file(syntax_output_path):
            logger.error(f"Mermaid syntax file {syntax_output_path} is invalid or empty (missing diagram declaration). Cannot render.")
            return False

        try:
            subprocess.run(["mmdc", "--version"], check=True, capture_output=True, timeout=10)
        except Exception as e:
            logger.error(f"mmdc not available: {e}")
            return False

        # Use the css_file_path passed to this function (from CLI or config fallback in main)
        resolved_css_path = css_file_path 
        if resolved_css_path and not resolved_css_path.exists():
            logger.warning(f"CSS file {resolved_css_path} not found. Rendering with mmdc defaults (theme {config.MMDC_DEFAULT_THEME}).")
            resolved_css_path = None 
        elif not resolved_css_path:
             logger.info(f"No specific CSS file provided. Rendering with mmdc defaults (theme {config.MMDC_DEFAULT_THEME}).")


        cmd = [
            "mmdc",
            "-i", str(syntax_output_path),
            "-o", str(output_path),
            "-t", config.MMDC_DEFAULT_THEME, 
            "-b", config.MMDC_DEFAULT_BG_COLOR # Background from config
        ]
        if resolved_css_path: # Only add -C if a valid CSS path is resolved
            cmd.extend(["-C", str(resolved_css_path)])
        
        logger.info(f"Rendering with mmdc: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            logger.error(f"mmdc failed with code {result.returncode}")
            logger.error(f"mmdc command: {' '.join(cmd)}")
            logger.error(f"mmdc stderr: {result.stderr}")
            logger.error(f"mmdc stdout: {result.stdout}")
            return False

        if not output_path.exists() or output_path.stat().st_size < 100: 
            logger.error(f"Rendered empty or invalid image/SVG: {output_path}. mmdc stdout: {result.stdout}, stderr: {result.stderr}")
            return False

        logger.info(f"Successfully rendered diagram to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Diagram rendering failed: {str(e)}")
        if output_path.exists():
            try: output_path.unlink(missing_ok=True)
            except OSError: pass
        return False

def _check_openrouter() -> bool:
    if not hasattr(config, 'OPENROUTER_API_KEY') or not config.OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not configured in config.py")
        return False
    return True

def generate_llm_prompt(
    content: str,
    diagram_type: SUPPORTED_DIAGRAM_TYPES,
    orientation: Optional[str] = "LR",
    max_elements: Optional[int] = 10, # Default defined at constants
    previous_error: Optional[str] = None,
    previous_syntax: Optional[str] = None
) -> Tuple[str, str]:
    """Generates the system and user prompts for the LLM."""

    system_prompt = f"""You are an expert in generating Mermaid.js syntax for various diagram types.
Your primary goal is to create accurate and concise Mermaid syntax based on the provided text content.

You MUST provide a concise, descriptive title for the diagram. This title should be provided separately, on its own line, before the '--- TITLE_END ---' delimiter.
DO NOT embed the title within the Mermaid syntax itself using the 'title' keyword.

Output Format:
Return the standalone title AND the Mermaid syntax separated by '--- TITLE_END ---'.

Example of the COMPLETE expected output:
My Diagram's Awesome Title
--- TITLE_END ---
graph TD;
  A[Start] --> B[Process Step];
  B --> C[End];

Diagram Type: {diagram_type}
Orientation: {orientation if diagram_type == "flowchart" else 'N/A for this diagram type'}
Maximum Elements: **IMPORTANT: The entire diagram MUST NOT exceed {max_elements} visual boxes/nodes.** Focus on the main steps/entities to stay within this limit. Be concise.

**CRITICAL SYNTAX RULES:**
- **NEVER use literal \\n characters in node labels.** If you need multi-line text, use simple short labels instead.
- **AVOID parentheses in node labels** as they cause parser errors. Use descriptive text without special characters.
- **DO NOT include any explanatory text after the Mermaid diagram.** Only provide the diagram syntax after the title.

Key considerations:
- Focus on the most critical entities and relationships to adhere to the {max_elements} box limit.
- Use clear and concise labels for nodes and edges without special characters.
- Adhere strictly to Mermaid.js syntax.
- For flowcharts, the diagram declaration (e.g., `graph TD`) MUST end with a semicolon (e.g., `graph TD;`).
- Each node definition (e.g., `A[Node A]`) and each link (e.g., `A --> B`) should ideally be on its own separate line. This improves readability and compatibility.
- If it's a flowchart, ensure it has a clear start and end.
- Ensure all nodes referenced are also defined.
- Make sure node IDs are unique and valid.
- Double-check arrow directions and types.
- For flowcharts, the default orientation is {orientation}. The declaration should be like `graph {orientation.upper()};`.
- Keep node labels short and simple - avoid complex descriptions that might contain problematic characters.
"""

    if previous_error and previous_syntax:
        system_prompt += f"""

Previous Attempt Review:
The following Mermaid syntax you previously generated resulted in an error:
--- PREVIOUS SYNTAX ---
{previous_syntax}
--- ERROR MESSAGE ---
{previous_error}
---
Please carefully review the error and the syntax. Identify the cause of the error and provide a corrected Mermaid syntax block.
The error `Expecting 'SQE', ..., got 'PS'` on a line like `K --> L[Node Text]; L --> M` often means there's an issue with how statements are separated or formatted.
Ensure:
1. The flowchart declaration (e.g., `graph LR;`) is correct and ends with a semicolon.
2. Each node definition (e.g., `N[Node Label]`) is well-formed.
3. Each link (e.g., `N1 --> N2`) is on its own line or clearly separated if multiple links are on one line (though separate lines are preferred for clarity).
4. There are no unexpected characters or formatting issues between the end of one statement (like a node definition with a semicolon) and the beginning of the next link.
DO NOT include a `title "..."` directive inside the Mermaid code block. Provide the title separately before '--- TITLE_END ---' only.
Crucially, re-evaluate the diagram's complexity to ensure it does not exceed the {max_elements} box/node limit. Simplify if necessary.
"""
    else:
        system_prompt += f"\nGenerate the Mermaid syntax based on the user's content below. Remember to provide the title separately and DO NOT embed it in the Mermaid code. Ensure each node definition and link is clearly on its own line where possible. Strictly adhere to the {max_elements} box/node limit."

    user_prompt = f"""Content to analyze for diagram generation:
{content[:config.MAX_CONTENT_LENGTH_FOR_DIAGRAM_PROMPT]}
"""
    if len(content) > config.MAX_CONTENT_LENGTH_FOR_DIAGRAM_PROMPT:
        user_prompt += "\n[Content truncated due to length]"

    return system_prompt, user_prompt


def process_input_file(file_path: Path) -> Optional[str]:
    """Reads content from .md, .txt, or parses .pdf file."""
    logger.info(f"Processing input file: {file_path}")
    if not file_path.is_file():
        logger.error(f"Input file not found: {file_path}")
        return None
    try:
        if file_path.suffix.lower() == '.pdf':
            try:
                from pdf_parser_advanced import process_single_pdf_advanced
                
                pdf_text_output_dir = Path(config.TEMP_DIR) / "pdf_extracted_texts"
                pdf_text_output_dir.mkdir(parents=True, exist_ok=True)

                pdf_processing_result = process_single_pdf_advanced(file_path, pdf_text_output_dir)
                
                actual_raw_text_path: Optional[Path] = None
                if isinstance(pdf_processing_result, dict):
                    path_val = pdf_processing_result.get('text_path')
                    if path_val and isinstance(path_val, str):
                        actual_raw_text_path = Path(path_val)
                    elif isinstance(path_val, Path):
                        actual_raw_text_path = path_val
                elif isinstance(pdf_processing_result, Path):
                    actual_raw_text_path = pdf_processing_result
                elif isinstance(pdf_processing_result, str):
                     actual_raw_text_path = Path(pdf_processing_result)
                else:
                    logger.error(f"PDF parser returned an unexpected type: {type(pdf_processing_result)}. Expected dict, Path, or str path.")
                    return None

                if not (actual_raw_text_path and actual_raw_text_path.exists()):
                    logger.error(f"PDF processing failed to produce a valid text_path for {file_path}. Path: {actual_raw_text_path}, Parser result: {pdf_processing_result}")
                    return None
                
                content = actual_raw_text_path.read_text(encoding='utf-8').strip()
            except ImportError:
                logger.error("pdf_parser_advanced module not found. Cannot process PDF files.")
                logger.info("To process PDFs, ensure 'pdf_parser_advanced.py' is available and dependencies (pdfminer.six) are installed.")
                return None
            except Exception as e:
                logger.error(f"Error processing PDF {file_path} with pdf_parser_advanced: {e}")
                return None
        elif file_path.suffix.lower() in ['.md', '.txt', '.mmd']:
            content = file_path.read_text(encoding='utf-8').strip()
        else:
            logger.error(f"Unsupported file type: {file_path.suffix}. Please use .md, .txt, or .pdf.")
            return None

        if not content:
            logger.warning(f"Input file is empty: {file_path}")
            return None
        logger.info(f"Read {len(content)} characters from {file_path.name}.")
        return content
    except Exception as e:
        logger.error(f"Failed to read or process input file {file_path}: {e}")
        return None

def _fix_common_mermaid_issues(syntax: str) -> str:
    """
    Fix common Mermaid syntax issues that cause parsing errors.
    This function addresses the most frequent problems found in LLM-generated Mermaid code.
    """
    logger.debug("Applying common Mermaid syntax fixes...")
    
    # Split into lines for processing
    lines = syntax.split('\n')
    fixed_lines = []
    
    for line in lines:
        original_line = line
        
        # Skip empty lines and comments
        if not line.strip() or line.strip().startswith('%%'):
            fixed_lines.append(line)
            continue
            
        # Fix literal \n in node labels (replace with space)
        line = re.sub(r'\\n', ' ', line)
        
        # Fix node labels with parentheses - remove or simplify
        # Pattern: [text (more text)] becomes [text more text]
        line = re.sub(r'\[([^[\]]*)\(([^()]*)\)([^[\]]*)\]', r'[\1 \2\3]', line)
        
        # Fix multiple spaces in node labels
        line = re.sub(r'\[([^[\]]*)\s+([^[\]]*)\]', r'[\1 \2]', line)
        
        # Ensure proper spacing around arrows
        line = re.sub(r'(\w)\s*-->\s*(\w)', r'\1 --> \2', line)
        
        # Remove extra characters that might cause issues
        line = re.sub(r'[^\w\s\[\](){}.,;:->|`"\'=+-]', '', line)
        
        if line != original_line:
            logger.debug(f"Fixed line: '{original_line}' -> '{line}'")
            
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

# --- Main Generation Logic ---
def generate_diagram_flow(
    input_content: str,
    diagram_type: SUPPORTED_DIAGRAM_TYPES,
    orientation: Optional[str],
    max_elements: Optional[int],
    output_path_base: Path,
    output_format: Literal["png", "svg", "mmd"],
    css_file: Optional[Path], # CSS file specifically for this generation flow
    total_llm_attempts: int 
) -> bool:
    """Core logic for generating a single diagram, including refinement attempts and title extraction."""
    mermaid_syntax = ""
    diagram_title = "Diagram (Title not generated by LLM)" 
    generated_successfully = False
    temp_dir = Path(config.TEMP_DIR) / f"diagram_gen_{os.urandom(4).hex()}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    error_message_for_prompt: Optional[str] = None
    previous_syntax_for_prompt: Optional[str] = None

    TITLE_SYNTAX_SEPARATOR = "--- TITLE_END ---"

    for attempt in range(1, total_llm_attempts + 1):
        logger.info(f"--- Diagram Generation Attempt {attempt} of {total_llm_attempts} ---")
        system_prompt, user_prompt = generate_llm_prompt(
            input_content,
            diagram_type,
            orientation,
            max_elements, # Pass the max_elements constraint
            error_message_for_prompt,
            previous_syntax_for_prompt
        )

        raw_llm_response = invoke_openrouter(system_prompt, user_prompt, f"{diagram_type} syntax (attempt {attempt})")

        if raw_llm_response.startswith("[Error:"):
            logger.error(f"LLM invocation failed on attempt {attempt}: {raw_llm_response}")
            error_message_for_prompt = raw_llm_response 
            previous_syntax_for_prompt = mermaid_syntax if mermaid_syntax else "[No previous syntax generated]"
            if attempt == total_llm_attempts: 
                logger.error("Max LLM attempts reached. LLM failed to generate usable response.")
                _cleanup_temp_dir(temp_dir)
                return False
            logger.info("Retrying with error context...")
            time.sleep(config.RETRY_DELAY_SECONDS) 
            continue
        
        if TITLE_SYNTAX_SEPARATOR in raw_llm_response:
            parts = raw_llm_response.split(TITLE_SYNTAX_SEPARATOR, 1)
            diagram_title = parts[0].strip() 
            mermaid_syntax = parts[1].strip() if len(parts) > 1 else ""
            logger.info(f"Extracted Standalone Diagram Title: {diagram_title}")
        else:
            logger.warning(f"LLM response did not contain the expected separator '{TITLE_SYNTAX_SEPARATOR}'. Treating entire response as Mermaid syntax and using default title.")
            mermaid_syntax = raw_llm_response 
            diagram_title = "Diagram (Title not extracted due to format error)" 

        if not mermaid_syntax.strip():
            logger.warning(f"Empty Mermaid syntax after parsing on attempt {attempt}.")
            error_message_for_prompt = "LLM returned empty Mermaid syntax after parsing title and syntax."
            previous_syntax_for_prompt = raw_llm_response 
            if attempt == total_llm_attempts:
                logger.error("Max LLM attempts reached. Failed to get non-empty Mermaid syntax.")
                _cleanup_temp_dir(temp_dir)
                return False
            logger.info("Retrying due to empty syntax...")
            time.sleep(config.RETRY_DELAY_SECONDS) 
            continue

        # Apply enhanced cleaning and validation
        mermaid_syntax = _clean_mermaid_syntax_basic(mermaid_syntax)
        mermaid_syntax = _ensure_declaration(mermaid_syntax, diagram_type, orientation or "LR")
        
        logger.debug(f"Cleaned Mermaid syntax (attempt {attempt}):\n{mermaid_syntax}")

        # Apply common syntax fixes
        mermaid_syntax = _fix_common_mermaid_issues(mermaid_syntax)

        is_valid, validation_output = validate_mermaid_syntax_with_mmdc(mermaid_syntax, temp_dir)

        if is_valid:
            logger.info(f"Syntax validated successfully on attempt {attempt}!")
            generated_successfully = True
            break
        else:
            logger.warning(f"Syntax validation failed on attempt {attempt}. MMDC Output: {validation_output}")
            error_message_for_prompt = validation_output
            previous_syntax_for_prompt = mermaid_syntax
            if attempt == total_llm_attempts:
                logger.error("Max LLM attempts reached. Syntax validation failed.")
                break 
            logger.info("Retrying with validation error context...")
            time.sleep(config.RETRY_DELAY_SECONDS) 

    _cleanup_temp_dir(temp_dir)

    if generated_successfully and mermaid_syntax.strip():
        commented_title_for_mmd = f"%% Diagram Title (from LLM): {diagram_title}\n"
        final_mermaid_syntax_for_file = commented_title_for_mmd + mermaid_syntax
        
        # Pass the css_file received by this function to render_mermaid_diagram
        if render_mermaid_diagram(final_mermaid_syntax_for_file, output_path_base, output_format, css_file):
            print(f"\n--- Diagram Generation Successful ---")
            print(f"Diagram Title (for reference): {diagram_title}") 
            if output_format != "mmd":
                print(f"Diagram saved to: {output_path_base.with_suffix(f'.{output_format}').resolve()}")
            print(f"Mermaid syntax (with title comment) saved to: {output_path_base.with_suffix('.mmd').resolve()}")
            return True
        else:
            logger.error("Failed to render the validated diagram.")
            return False
    else: 
        logger.error("Failed to generate and validate Mermaid syntax after all attempts.")
        if mermaid_syntax and mermaid_syntax.strip(): 
            syntax_output_path = output_path_base.with_suffix('.mmd')
            try:
                comment_for_failed_syntax = f"%% Diagram Title (from LLM): {diagram_title} (LAST ATTEMPT - VALIDATION FAILED OR OTHER ERROR)\n"
                final_problematic_syntax = comment_for_failed_syntax + mermaid_syntax
                syntax_output_path.write_text(final_problematic_syntax, encoding='utf-8')
                logger.info(f"Saved last problematic Mermaid syntax (with title comment) to: {syntax_output_path}")
            except IOError as e:
                logger.error(f"Failed to save problematic .mmd file {syntax_output_path}: {e}")
        
        print("\n--- Diagram Generation Failed ---")
        print("Could not generate a valid diagram after multiple attempts.")
        return False

def _cleanup_temp_dir(temp_dir_path: Path):
    """Cleans up the temporary validation directory."""
    if temp_dir_path.exists():
        for item in temp_dir_path.iterdir():
            try:
                item.unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"Could not remove temp file {item} in {temp_dir_path}: {e}")
        try:
            temp_dir_path.rmdir()
        except OSError as e:
            logger.warning(f"Could not remove temp directory {temp_dir_path}: {e}")


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Generate Mermaid diagrams from text content using an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file", help="Path to the input file (.md, .txt, .pdf, or .mmd for re-rendering).")
    parser.add_argument(
        "-o", "--output_dir",
        help="Directory to save diagram and .mmd file.",
        default=str(config.DIAGRAM_DIR)
    )
    parser.add_argument(
        "-t", "--type",
        choices=[t for t in SUPPORTED_DIAGRAM_TYPES.__args__], 
        default=DEFAULT_DIAGRAM_TYPE,
        help="Type of Mermaid diagram to generate."
    )
    parser.add_argument(
        "--orientation",
        choices=["LR", "TD", "RL", "BT"],
        default="LR",
        help="Orientation for flowcharts (Left-to-Right, Top-to-Bottom, etc.)."
    )
    parser.add_argument(
        "--max_elements",
        type=int,
        default=DEFAULT_MAX_ELEMENTS,
        help="Strict maximum number of visual boxes/nodes in the diagram."
    )
    parser.add_argument(
        "-f", "--format",
        choices=["png", "svg", "mmd"],
        default="png",
        help="Output format for the diagram. 'mmd' will only save the syntax file."
    )
    parser.add_argument(
        "--css_file",
        type=Path,
        default=None, 
        help=f"Path to a custom CSS file for mmdc styling. If not provided, config.DEFAULT_DIAGRAM_STYLE_CSS will be used if set and exists."
    )
    parser.add_argument(
        "--refine_attempts",
        type=int,
        default=1, 
        help="Number of *additional* attempts to refine the Mermaid syntax with the LLM if validation fails. "
             "For example, 1 means 1 initial attempt + 1 refinement attempt (total 2 LLM calls)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level for console)."
    )
    args = parser.parse_args()

    if args.verbose:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler): 
                handler.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled for console.")

    input_path = Path(args.input_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    Path(config.TEMP_DIR).mkdir(parents=True, exist_ok=True)


    base_filename = input_path.stem
    if base_filename.endswith("_pseudocode"): 
        base_filename = base_filename.replace("_pseudocode", "")
    output_base_path = output_dir / f"{base_filename}_{args.type}_diagram"

    logger.info(f"--- Starting Diagram Generation for: {input_path.name} ---")
    logger.info(f"Type: {args.type}, Output: {output_base_path.name}.{args.format}, Max Elements: {args.max_elements}")


    if not _check_openrouter():
        logger.critical("OpenRouter API not configured. Cannot proceed.")
        sys.exit(1)

    input_content = process_input_file(input_path)
    if not input_content:
        logger.critical(f"Could not get content from input file: {input_path}")
        sys.exit(1)

    total_llm_attempts = 1 + args.refine_attempts

    # Determine which CSS file to use
    css_to_use: Optional[Path] = None
    if args.css_file: # CLI argument takes precedence
        if args.css_file.exists():
            css_to_use = args.css_file
            logger.info(f"Using CSS file from CLI: {css_to_use}")
        else:
            logger.warning(f"CSS file specified via CLI not found: {args.css_file}. Will use mmdc defaults.")
    elif hasattr(config, 'DEFAULT_DIAGRAM_STYLE_CSS') and config.DEFAULT_DIAGRAM_STYLE_CSS and config.DEFAULT_DIAGRAM_STYLE_CSS.exists():
        css_to_use = config.DEFAULT_DIAGRAM_STYLE_CSS
        logger.info(f"Using default CSS file from config: {css_to_use}")
    else:
        logger.info("No CSS file provided or default in config not found. Using mmdc default styling.")


    if input_path.suffix.lower() == '.mmd' and args.format != 'mmd':
        logger.info(f"Input is a .mmd file. Attempting direct render to {args.format}.")
        
        processed_mmd_content = _clean_mermaid_syntax_basic(input_content)
        processed_mmd_content = _ensure_declaration(processed_mmd_content, args.type, args.orientation)
        logger.debug(f"Syntax for direct .mmd render after cleaning/fixing:\n{processed_mmd_content}")

        temp_mmd_for_direct_render = Path(config.TEMP_DIR) / f"direct_render_temp_{input_path.stem}.mmd"
        temp_mmd_for_direct_render.write_text(processed_mmd_content, encoding='utf-8')

        if not _is_valid_mermaid_file(temp_mmd_for_direct_render):
            logger.error(f"The provided or processed .mmd file {input_path} seems invalid or empty. Cannot render directly.")
            if temp_mmd_for_direct_render.exists(): temp_mmd_for_direct_render.unlink(missing_ok=True)
            sys.exit(1)
        
        success = render_mermaid_diagram(processed_mmd_content, output_base_path, args.format, css_to_use)
        if temp_mmd_for_direct_render.exists(): temp_mmd_for_direct_render.unlink(missing_ok=True)

    else: 
        success = generate_diagram_flow(
            input_content,
            args.type,
            args.orientation,
            args.max_elements, # Pass the constraint
            output_base_path,
            args.format,
            css_to_use, # Pass the resolved CSS file
            total_llm_attempts 
        )

    if not success: 
        print(f"Check logs at '{log_file.resolve()}' for details.")
        if (output_base_path.with_suffix('.mmd')).exists():
             print(f"Last generated Mermaid syntax (potentially problematic) saved to: {output_base_path.with_suffix('.mmd').resolve()}")

    print("--- Finished ---")

if __name__ == "__main__":
    # --- Dummy pdf_parser_advanced.py setup ---
    dummy_pdf_parser_path = Path(__file__).parent / "pdf_parser_advanced.py"
    if not dummy_pdf_parser_path.exists():
        with open(dummy_pdf_parser_path, "w", encoding='utf-8') as f:
            f.write("""
# Dummy pdf_parser_advanced.py
from pathlib import Path
import logging
logger = logging.getLogger(__name__) 

def process_single_pdf_advanced(pdf_path: Path, output_dir: Path) -> Path: 
    logger.warning(f"Using DUMMY PDF parser for {pdf_path.name}. No actual PDF processing will occur.")
    dummy_text_content = f"This is DUMMY text extracted from PDF: {pdf_path.name}. PDF parsing is not truly implemented in this dummy file."
    output_dir.mkdir(parents=True, exist_ok=True) 
    txt_output_path = output_dir / f"{pdf_path.stem}_dummy_extracted.txt"
    try:
        txt_output_path.write_text(dummy_text_content, encoding='utf-8')
        logger.info(f"Dummy PDF parser created: {txt_output_path}")
    except Exception as e:
        logger.error(f"Dummy PDF parser failed to write file: {e}")
        return Path(output_dir / "dummy_extraction_failed.txt") 
    return txt_output_path
""")
        print(f"Created dummy pdf_parser_advanced.py at {dummy_pdf_parser_path.resolve()} for basic script operation without real PDF parsing.")
        print("If you have a real 'pdf_parser_advanced.py', ensure it's in the same directory or your PYTHONPATH.")

    # --- Ensure config attributes exist (for resilience) ---
    if not hasattr(config, 'RETRY_DELAY_SECONDS'):
        setattr(config, 'RETRY_DELAY_SECONDS', 2) 
    
    # For the CSS styling part:
    # If you want the script to automatically use your dark_yellow_diagram.css when no --css_file is given,
    # you'd set it in your actual config.py:
    # config.DEFAULT_DIAGRAM_STYLE_CSS = Path("path/to/your/dark_yellow_diagram.css")
    # config.MMDC_DEFAULT_BG_COLOR = "#000000"
    # For now, the script relies on passing --css_file or having these in config.py already.
    # We'll ensure MMDC_DEFAULT_BG_COLOR is black if using your suggested CSS.
    if not hasattr(config, 'MMDC_DEFAULT_BG_COLOR'):
        setattr(config, 'MMDC_DEFAULT_BG_COLOR', '#333') # A generic dark default if not black
    
    # If a specific CSS (like your yellow box one) is intended to be the default,
    # it's best set in config.py. For this run, we'll assume CLI or existing config handles it.
    # If you passed --css_file path/to/dark_yellow_diagram.css, and 
    # config.MMDC_DEFAULT_BG_COLOR = "#000000" (either set in file or dynamically for this test), it should work.
    
    # Example: For testing your yellow CSS, you might temporarily set this in config.py
    # Or, ensure your config.py has:
    # DEFAULT_DIAGRAM_STYLE_CSS = Path("dark_yellow_diagram.css") # if it's in the same dir as script
    # MMDC_DEFAULT_BG_COLOR = "#000000"

    main()