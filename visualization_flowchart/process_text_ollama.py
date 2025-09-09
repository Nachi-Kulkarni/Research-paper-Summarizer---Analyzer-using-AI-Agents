# process_text_ollama.py

import argparse
import logging
from pathlib import Path
import sys
import os
import re
import time

# --- Third-party Imports ---
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    sys.exit("Please install langchain-ollama: pip install -U langchain-ollama")

# --- Local Imports ---
import config # Shared configuration
try:
    # Import only the text extraction function
    from pdf_parser_advanced import extract_structured_text
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Failed to import 'extract_structured_text' from 'pdf_parser_advanced.py': {e}")
    sys.exit(1)

# --- Logger Setup ---
log_file = config.LOG_DIR / "process_text_ollama.log"
logger = logging.getLogger("ollama_processor")
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
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# --- Helper Function ---
def invoke_ollama(llm_instance: OllamaLLM, prompt: str, task_name: str) -> str:
    """Helper to invoke Ollama and handle basic errors."""
    logger.info(f"Invoking Ollama for: {task_name}")
    logger.debug(f"Prompt (first 500 chars): {prompt[:500]}...")
    try:
        start_time = time.time()
        response = llm_instance.invoke(prompt)
        elapsed = time.time() - start_time
        logger.info(f"Ollama call for {task_name} completed in {elapsed:.2f}s.")
        if not response or not response.strip():
            logger.warning(f"Ollama returned an empty response for {task_name}.")
            return f"<{task_name.lower()}>[LLM returned empty response]</{task_name.lower()}>"
        # Basic refusal check
        if "sorry" in response.lower() or "cannot fulfill" in response.lower() or "i need the text" in response.lower():
             logger.warning(f"Ollama likely refused task '{task_name}': {response[:100]}...")
             return f"<{task_name.lower()}>[LLM refused or could not process: {response[:100]}]</{task_name.lower()}>"
        return response.strip()
    except Exception as e:
        logger.error(f"Error during Ollama call for {task_name}: {e}", exc_info=True)
        if "llama runner process has terminated" in str(e):
             logger.error("Ollama runner crashed. Check resources/Ollama logs.")
        return f"<{task_name.lower()}>[Error during LLM call: {e}]</{task_name.lower()}>"

def extract_code(raw_output: str) -> str:
    """Extracts Python code strictly from a ```python ... ``` block."""
    match = re.search(r'```python\s*(.*?)\s*```', raw_output, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1).strip()
        return code if code else "[Extracted code block was empty]"
    else:
        logger.warning("Could not find ```python block in code generation output.")
        # Return the raw output but indicate it wasn't formatted correctly
        return f"[Code block not found in LLM output]:\n{raw_output}"

def clean_pseudocode(raw_output: str) -> str:
    """Basic cleaning for pseudocode output."""
    code = raw_output.strip()
     # Remove markdown fences if present
    if code.startswith("```pseudocode"): code = code[len("```pseudocode"):].strip()
    if code.startswith("```"): code = code[len("```"):].strip()
    if code.endswith("```"): code = code[:-len("```")].strip()
    # Remove common preambles/refusals
    preambles_lower = ["here is the pseudocode:", "pseudocode:", "okay, here's the pseudocode:"]
    code_lower = code.lower()
    for preamble in preambles_lower:
        if code_lower.startswith(preamble):
            code = code[len(preamble):].lstrip()
            break
    return code if code else "[Pseudocode generation resulted in empty output after cleaning]"


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Parse PDF and call Ollama for Summary, Pseudocode, and Code.")
    parser.add_argument("pdf_file", help="Path to the input PDF file.")
    parser.add_argument("-o", "--output_dir", help="Directory to save pseudocode and final output.", default=None)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers: handler.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled.")

    pdf_path = Path(args.pdf_file)
    if not pdf_path.is_file():
        logger.critical(f"Input PDF not found: {pdf_path}")
        sys.exit(1)

    base_filename = pdf_path.stem
    logger.info(f"--- Starting Processing for: {base_filename} ---")

    # --- 1. Parse PDF ---
    logger.info("Step 1: Parsing PDF...")
    parsed_text, _ = extract_structured_text(pdf_path) # Ignore metadata
    if not parsed_text:
        logger.critical(f"PDF parsing failed for {pdf_path.name}. Cannot continue.")
        sys.exit(1)
    logger.info(f"PDF Parsed. Extracted {len(parsed_text)} characters.")

    # --- 2. Initialize Ollama ---
    logger.info("Step 2: Initializing Ollama...")
    try:
        # Use general model for all tasks for simplicity, can be changed in config
        llm = OllamaLLM(
            model=config.LLM_MODEL_GENERAL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.DEFAULT_TEMPERATURE, # Use moderate temp
            top_p=config.DEFAULT_TOP_P,
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", 8192)) # Use reasonable context
        )
        logger.info(f"OllamaLLM initialized (Model: {config.LLM_MODEL_GENERAL}, Ctx: {llm.num_ctx}).")
    except Exception as e:
        logger.critical(f"Failed to initialize OllamaLLM: {e}", exc_info=True)
        sys.exit(1)

    # --- 3. Define Prompts ---
    summary_prompt = (
        "You are an expert academic text summarizer. "
        "Generate **around 5 to 7 key bullet points** summarizing the main aspects (objectives, methods, findings, conclusions) of the following text. "
        "Format the output **only as a markdown list** using '-' or '*' for bullets. "
        "Output **only the bullet points**. Do NOT include any introductory text or explanations.\n\n"
        "--- START OF TEXT ---\n{text}\n--- END OF TEXT ---\n\n"
        "Summary (MUST be ONLY the bullet points):"
    ).format(text=parsed_text)

    pseudocode_prompt = (
        "You are an expert programmer. Convert the following text section into clear, step-by-step pseudocode. "
        "Use standard conventions (e.g., INPUT, OUTPUT, SET, IF/THEN/ELSE, WHILE, FOR, CALL) and indentation. Focus on the core logic described. "
        "Output ONLY the pseudocode. NO explanations.\n\n"
        "--- START OF TEXT ---\n{text}\n--- END OF TEXT ---\n\n"
        "Pseudocode:"
    ).format(text=parsed_text) # Use full parsed text for pseudocode context

    # Code prompt will be constructed later using pseudocode output

    # --- 4. LLM Calls ---
    logger.info("Step 3: Performing LLM calls...")

    # Call for Summary
    summary_output_raw = invoke_ollama(llm, summary_prompt, "Summary")
    summary_output_tagged = f"<summary>\n{summary_output_raw}\n</summary>"

    # Call for Pseudocode
    pseudocode_output_raw = invoke_ollama(llm, pseudocode_prompt, "Pseudocode")
    pseudocode_cleaned = clean_pseudocode(pseudocode_output_raw) # Clean for saving/next step
    pseudocode_output_tagged = f"<pseudocode>\n{pseudocode_cleaned}\n</pseudocode>" # Tag cleaned version

    # Call for Code Generation (using the cleaned pseudocode)
    code_output_tagged = "<code>\n[Pseudocode generation failed, skipping code generation]\n</code>" # Default if pseudocode failed
    if "[Error" not in pseudocode_cleaned and "[LLM" not in pseudocode_cleaned and pseudocode_cleaned.strip():
        code_prompt = (
            "You are an expert Python programmer. Translate the provided pseudocode into functional Python code. "
            "Generate clean, commented Python code. Include likely imports if needed. Define functions or classes appropriately. "
            "CRITICAL: Output ONLY the raw Python code, enclosed within a single markdown code block starting with ```python and ending with ```. "
            "DO NOT include ANY explanations or text outside this code block.\n\n"
            "--- START OF PSEUDOCODE ---\n{pseudocode}\n--- END OF PSEUDOCODE ---\n\n"
            "Python Code Output (MUST be ONLY the ```python block):"
        ).format(pseudocode=pseudocode_cleaned)

        code_output_raw = invoke_ollama(llm, code_prompt, "Code Generation")
        code_extracted = extract_code(code_output_raw) # Extract the block
        code_output_tagged = f"<code>\n{code_extracted}\n</code>"
    else:
        logger.warning("Skipping code generation as pseudocode generation failed or was empty.")


    # --- 5. Output Results ---
    logger.info("Step 4: Compiling and Saving Results...")

    final_output_content = (
        f"# Ollama Processing Results for: {base_filename}\n\n"
        f"{summary_output_tagged}\n\n"
        f"{pseudocode_output_tagged}\n\n"
        f"{code_output_tagged}\n"
    )

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default to outputs/ derived from config or relative path
        output_dir = Path(config.OUTPUT_DIR) if hasattr(config, 'OUTPUT_DIR') else Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main output
    output_file_path = output_dir / f"{base_filename}_ollama_output.md" # Use .md for better viewing
    try:
        output_file_path.write_text(final_output_content, encoding='utf-8')
        logger.info(f"Combined output saved to: {output_file_path}")
    except IOError as e:
        logger.error(f"Failed to save combined output to {output_file_path}: {e}")

    # Save pseudocode separately for diagram generator
    pseudocode_dir = Path(config.PSEUDOCODE_DIR) if hasattr(config, 'PSEUDOCODE_DIR') else output_dir / "pseudocode"
    pseudocode_dir.mkdir(parents=True, exist_ok=True)
    pseudocode_file_path = pseudocode_dir / f"{base_filename}_pseudocode.md"
    # Only save if pseudocode generation didn't fail critically
    if "[Error" not in pseudocode_cleaned and "[LLM" not in pseudocode_cleaned:
        try:
            pseudocode_file_path.write_text(pseudocode_cleaned, encoding='utf-8')
            logger.info(f"Pseudocode saved separately to: {pseudocode_file_path} (for diagram generation)")
        except IOError as e:
            logger.error(f"Failed to save separate pseudocode to {pseudocode_file_path}: {e}")
    else:
         logger.warning(f"Separate pseudocode file not saved due to generation errors.")


    print("\n--- Processing Summary ---")
    print(f"Results saved to {output_file_path}")
    if pseudocode_file_path.exists():
        print(f"Pseudocode for diagram generation saved to {pseudocode_file_path}")
    print("--- Finished ---")

if __name__ == "__main__":
    main()