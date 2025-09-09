import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Union, List, Optional
import os # <--- ADD THIS LINE

# --- Load environment variables ---
import dotenv
dotenv.load_dotenv()

# --- Local Imports ---
import config # Shared configuration

# Import agent classes
from agents.summarizer import SummarizationAgent
from agents.pseudocode_agent import PseudocodeAgent
from agents.code_generator import CodeGenerationAgent
from agents.report_generator import ReportGenerationAgent

# --- Import the ADVANCED PDF parsing function ---
# This function is expected to return the Path to the generated text file or an empty string/None on failure.
try:
    # Import the function that processes a single PDF and saves the text
    from pdf_parser_advanced import process_single_pdf_advanced
    logging.info("Using ADVANCED PDF parser function (process_single_pdf_advanced).")
except ImportError as e:
    # Critical error if the required parser isn't found.
    logging.critical(f"CRITICAL ERROR: Failed to import 'process_single_pdf_advanced' from 'pdf_parser_advanced.py': {e}")
    logging.critical("Ensure 'pdf_parser_advanced.py' is in the project root or accessible in the Python path.")
    sys.exit(1) # Exit if the required parser cannot be imported
# --- End PDF Parser Import ---


# --- Logger Setup for Main Orchestrator ---
# Use a distinct name for the main logger
main_orchestrator_logger = logging.getLogger("main_orchestrator")
# Use the central AGENT_LOG_FILE defined in config for combined logging
log_file = config.AGENT_LOG_FILE
if not main_orchestrator_logger.hasHandlers():
    # File Handler (appends to the central agent log)
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a') # Append mode
    file_formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    main_orchestrator_logger.addHandler(file_handler)

    # Console Handler (for immediate feedback)
    console_handler = logging.StreamHandler(sys.stdout)
    # Simpler format for console during orchestration
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [MainWorkflow] - %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO) # Show INFO level on console
    main_orchestrator_logger.addHandler(console_handler)

    main_orchestrator_logger.setLevel(logging.INFO) # Default level for file
    main_orchestrator_logger.propagate = False # Prevent double logging if root is configured


def process_single_pdf_workflow(pdf_path: Path, skip_steps: List[str] = []) -> bool:
    """
    Runs the complete agent workflow for a single PDF file using the advanced parser.

    Args:
        pdf_path: Path to the input PDF file.
        skip_steps: A list of step names (lowercase) to skip (e.g., ['code', 'report']).
                    Valid steps: 'summary', 'pseudo', 'code', 'report'. PDF parsing cannot be skipped.

    Returns:
        True if the workflow completed without critical PDF parsing failure,
        False if PDF parsing failed critically. Downstream agent failures issue warnings but return True.
    """
    start_time = time.time()
    logger = main_orchestrator_logger # Use the dedicated logger
    logger.info(f"--- Starting Workflow for: {pdf_path.name} ---")
    base_filename: str = "" # Will be derived after successful PDF parsing

    # Normalize skip_steps to lowercase
    skip_steps = [step.lower() for step in skip_steps]
    logger.info(f"Skipping steps: {skip_steps}" if skip_steps else "Running all steps.")

    # --- Step 1: Parse PDF (Advanced Parser) ---
    # This step is mandatory for the workflow to proceed.
    logger.info(f"[Step 1/5] Parsing PDF: {pdf_path.name}")
    raw_text_path: Optional[Path] = None
    try:
        # Call the imported advanced parser function
        parser_output = process_single_pdf_advanced(pdf_path)

        # Check the output type and validity
        if isinstance(parser_output, Path) and parser_output.is_file():
            raw_text_path = parser_output
            # --- MODIFIED: Derive base_filename from the raw text file stem, STRIPPING _raw ---
            raw_stem = raw_text_path.stem # e.g., MyPaper_raw
            if raw_stem.endswith("_raw"):
                base_filename = raw_stem[:-len("_raw")] # e.g., MyPaper
            else:
                base_filename = raw_stem # Fallback if no _raw suffix
                logger.warning(f"Raw text filename stem '{raw_stem}' does not end with '_raw'. Using full stem as base_filename.")
            # --- END MODIFICATION ---
            logger.info(f"PDF parsed successfully. Raw text saved to: {raw_text_path}")
            logger.info(f"Derived base filename (for outputs): {base_filename}")
            # Log a snippet for verification
            try:
                 with open(raw_text_path, 'r', encoding='utf-8') as f:
                      preview = "".join(f.readline() for _ in range(3)) # Read first 3 lines
                 logger.debug(f"Raw text preview (first 3 lines):\n{preview.strip()}")
            except Exception as read_err:
                 logger.warning(f"Could not read preview from {raw_text_path}: {read_err}")
        else:
            # Handle failure cases where parser returns empty string or invalid path
            logger.error(f"Critical Failure: PDF parsing failed or did not produce a valid text file for {pdf_path.name}.")
            parser_log_file = config.PDF_PARSER_LOG_FILE # From config
            logger.error(f"Check PDF parser logs ('{parser_log_file}') for details.")
            return False # Indicate critical failure - workflow cannot continue

    except Exception as parse_err:
        logger.error(f"Critical Failure: Unexpected error during PDF parsing of {pdf_path.name}: {parse_err}", exc_info=True)
        return False # Indicate critical failure

    # Ensure base_filename was set if parsing succeeded
    if not base_filename:
        logger.error(f"Critical internal error: base_filename not set after PDF parsing for {pdf_path.name}. Aborting workflow.")
        return False

    # --- Conditional Step Execution ---
    summary_generated = False
    pseudocode_files_generated = False
    code_files_generated = False

    # --- Step 2: Summarization ---
    if "summary" not in skip_steps:
        logger.info(f"[Step 2/5] Generating Summary for: {base_filename}")
        try:
            summarizer = SummarizationAgent() # Uses config defaults
            # The SummarizationAgent derives its output filename based on the input raw_text_path,
            # typically stripping "_raw" itself. We pass raw_text_path.
            summary_path = summarizer.process_text_file(raw_text_path)
            if summary_path:
                logger.info(f"Summary generated successfully: {summary_path}")
                summary_generated = True
            else:
                # Summarizer logs failure details internally
                logger.warning(f"Summary generation failed or produced no file for {base_filename}. Check summarizer logs.")
        except Exception as e:
            logger.error(f"Error during Summarization Agent execution for {base_filename}: {e}", exc_info=True)
            logger.warning("Continuing workflow despite summarization error.")
    else: logger.info("[Step 2/5] SKIPPED - Summary Generation")


    # --- Step 3: Pseudocode & Diagram Generation ---
    if "pseudo" not in skip_steps:
        logger.info(f"[Step 3/5] Generating Pseudocode for: {base_filename}") # Diagrams removed
        pseudocode_output_paths: List[Path] = [] # Expected to be a list with one path
        try:
            # Read the text content again for this agent
            try:
                full_text_content = raw_text_path.read_text(encoding='utf-8')
                if not full_text_content.strip(): raise ValueError("Raw text file content is empty.")
            except Exception as e:
                 logger.error(f"Failed to read raw text file {raw_text_path} for pseudocode generation: {e}. Skipping Pseudocode step.")
                 full_text_content = None

            if full_text_content:
                pseudo_agent = PseudocodeAgent() # Uses config defaults
                # Pass the derived (stripped) base_filename for consistent output naming
                pseudocode_output_paths = pseudo_agent.generate_pseudocode_for_paper_sections(
                    full_text=full_text_content,
                    base_filename=base_filename, # Use the derived (stripped) base_filename
                    output_dir_pseudocode=config.PSEUDOCODE_DIR
                )

                # Check if the consolidated pseudocode file was generated
                # The filename should be base_filename + "_pseudocode.md"
                expected_pseudocode_file = config.PSEUDOCODE_DIR / f"{base_filename}_pseudocode.md"
                if pseudocode_output_paths and expected_pseudocode_file.exists():
                    logger.info(f"Consolidated pseudocode generated successfully: {expected_pseudocode_file}")
                    pseudocode_files_generated = True
                else:
                    logger.warning(f"Consolidated pseudocode generation failed or produced no file for {base_filename}. Expected: {expected_pseudocode_file}")
                    if pseudocode_output_paths:
                        logger.debug(f"Pseudocode agent returned paths: {pseudocode_output_paths}, but expected file not found or list empty.")
                    pseudocode_output_paths = [] # Ensure it's an empty list on failure

            else: # Case where reading the raw_text file failed
                logger.warning("Skipping Pseudocode generation because reading the raw text file failed.")

        except Exception as e:
            logger.error(f"Error during Pseudocode Agent execution for {base_filename}: {e}", exc_info=True)
            logger.warning("Continuing workflow despite Pseudocode error.")
    else: logger.info("[Step 3/5] SKIPPED - Pseudocode Generation")


    # --- Step 4: Code Generation ---
    # Depends on pseudocode files existing
    if "code" not in skip_steps:
        if pseudocode_files_generated:
            logger.info(f"[Step 4/5] Generating Code from Pseudocode for: {base_filename}")
            try:
                code_agent = CodeGenerationAgent() # Uses config defaults (OpenRouter)

                # Path to the single consolidated pseudocode file
                consolidated_pseudocode_path = config.PSEUDOCODE_DIR / f"{base_filename}_pseudocode.md"

                if consolidated_pseudocode_path.is_file():
                    logger.info(f"Attempting code generation from consolidated pseudocode: {consolidated_pseudocode_path.name}")
                    # Pass base_filename and output_dir explicitly to code_agent
                    code_path = code_agent.process_pseudocode_file(
                        input_pseudocode_path=consolidated_pseudocode_path,
                        output_dir=config.CODE_DIR,
                        base_filename=base_filename # Pass the stripped base_filename
                    )
                    if code_path:
                        logger.info(f"Consolidated code generated successfully: {code_path}")
                        code_files_generated = True
                    else:
                        logger.warning(f"Consolidated code generation failed for pseudocode: {consolidated_pseudocode_path.name}. Check code generator logs.")
                else:
                    logger.warning(f"Skipping code generation: Consolidated pseudocode file not found at {consolidated_pseudocode_path}")

            except Exception as e:
                logger.error(f"Error during Code Generation Agent execution for {base_filename}: {e}", exc_info=True)
                logger.warning("Continuing workflow despite code generation error.")
        else:
             logger.warning(f"Skipping Code Generation for {base_filename} as no pseudocode files were confirmed generated in the previous step.")
    else: logger.info("[Step 4/5] SKIPPED - Code Generation")


    # --- Step 5: Report Generation ---
    if "report" not in skip_steps:
        logger.info(f"[Step 5/5] Generating PDF Report for: {base_filename}")
        try:
            # Check if there's anything to report on (summary, consolidated pseudocode, or consolidated code)
            if summary_generated or pseudocode_files_generated or code_files_generated:
                report_agent = ReportGenerationAgent() # Uses config defaults
                report_path = report_agent.generate_report(base_filename) # Pass the derived (stripped) base_filename
                if report_path:
                    logger.info(f"Report generated successfully: {report_path}")
                else:
                    # Report agent logs failure details internally
                    logger.error(f"Report generation failed for {base_filename}. Check report generator logs.")
            else:
                 logger.warning(f"Skipping report generation for {base_filename} as no core text components (summary, pseudocode, code) were successfully generated.")

        except Exception as e:
            logger.error(f"Error during Report Generation Agent execution for {base_filename}: {e}", exc_info=True)
    else: logger.info("[Step 5/5] SKIPPED - Report Generation")


    # --- Workflow Completion ---
    elapsed_time = time.time() - start_time
    logger.info(f"--- Finished Workflow for: {pdf_path.name} in {elapsed_time:.2f} seconds ---")
    print("-" * 60) # Separator in console output
    return True


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent PDF Processing Workflow (using Advanced Parser).")
    parser.add_argument(
        "pdf_files",
        metavar="PDF_FILE",
        type=str,
        nargs='+', # Accept one or more PDF files
        help="Path(s) to the PDF file(s) to process."
    )
    parser.add_argument(
        "--skip",
        metavar="STEP",
        type=str,
        nargs='*', # Allow zero or more steps to skip
        default=[],
        choices=['summary', 'pseudo', 'code', 'report'], # Allowed steps to skip
        help="Specify steps to skip (e.g., --skip code report). Allowed: summary, pseudo, code, report."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level) to console and file."
    )

    args = parser.parse_args()
    logger = main_orchestrator_logger # Use the dedicated logger

    # --- Setup Logging Level ---
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        # Also set console handler level lower for verbosity
        for handler in logger.handlers:
             if isinstance(handler, logging.StreamHandler): # Check if it's the console handler
                  handler.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled (DEBUG level).")
    else:
         logger.setLevel(logging.INFO) # Ensure default is INFO


    logger.info("========================================================")
    logger.info("=== Starting Multi-Agent PDF Processing Workflow ===")
    logger.info(f"=== Using Advanced PDF Parser                ===")
    logger.info(f"=== Configured Ollama Base URL: {config.OLLAMA_BASE_URL} ===")
    logger.info(f"=== Configured OpenRouter API Key: {'SET' if os.getenv('OPENROUTER_API_KEY') else 'NOT SET - Code Generation will fail!'} ===")
    logger.info("========================================================")

    pdf_paths = [Path(f) for f in args.pdf_files]
    success_count = 0
    failure_count = 0
    skipped_count = 0

    input_dir_config = config.RESEARCH_PAPERS_DIR
    logger.info(f"Processing PDF files provided via command line.")
    logger.debug(f"(Default configured input dir: {input_dir_config})")
    if not input_dir_config.exists():
         logger.warning(f"Configured default input directory '{input_dir_config}' does not exist. Ensure provided paths are correct.")


    # --- Process each PDF file ---
    for pdf_path in pdf_paths:
        if not pdf_path.is_file():
            logger.error(f"Input Error: PDF file not found at '{pdf_path}'. Skipping.")
            skipped_count += 1
            continue

        try:
            workflow_ok = process_single_pdf_workflow(pdf_path, skip_steps=args.skip)
            if workflow_ok:
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            logger.critical(f"CRITICAL UNHANDLED EXCEPTION during workflow for {pdf_path.name}: {e}", exc_info=True)
            failure_count += 1
            logger.critical("Workflow aborted for this file due to unexpected error.")


    # --- Final Summary ---
    logger.info("=============================================")
    logger.info("=== Multi-Agent PDF Processing Finished ===")
    logger.info(f"Total Files Provided: {len(pdf_paths)}")
    logger.info(f"Files Skipped (Not Found): {skipped_count}")
    logger.info(f"Workflows Attempted: {len(pdf_paths) - skipped_count}")
    logger.info(f"  - Completed (Past Parsing): {success_count}")
    logger.info(f"  - Failed (Parsing or Critical Error): {failure_count}")
    logger.info("=============================================")
    logger.info(f"Consolidated logs available in: {config.AGENT_LOG_FILE}")
    logger.info(f"(Parser-specific logs in: {config.PDF_PARSER_LOG_FILE})")
    logger.info(f"(Individual agent logs in: {config.LOG_DIR}/<agent_name>.log)")