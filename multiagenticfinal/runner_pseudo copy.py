# multiagenticfinal/runner_pseudo.py

import logging
from pathlib import Path
import sys

# Ensure the project root is in the Python path if running this script directly
# This allows for imports like 'config' and 'agents.pseudocode_agent'
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config  # Import configuration
from agents.pseudocode_agent import PseudocodeAgent # Import the agent

if __name__ == "__main__":
    # Basic logging setup for the runner script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # More detailed format
        datefmt=config.LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout), # Log to console
            logging.FileHandler(config.LOG_DIR / "runner_pseudo.log", mode='a', encoding='utf-8') # Log to a specific file
        ]
    )
    runner_logger = logging.getLogger(__name__)
    runner_logger.info("--- Pseudocode Agent Runner Script Started ---")

    try:
        runner_logger.info("--- Initializing PseudocodeAgent (for pseudocode generation only) ---")
        # Initialize agent (it will set up its own specific logger as per its _setup_logger method)
        agent = PseudocodeAgent()
        runner_logger.info("--- PseudocodeAgent Initialized by Runner ---")

        # --- Configuration for the example run ---
        # You can modify this or use argparse to make it more flexible
        example_paper_name = "template" # Example: "attention", "template", "your_paper_name"
        # Construct path to the raw text file
        # Assuming RAW_TEXT_DIR is correctly configured in config.py and points to 'multiagenticfinal/outputs/raw_text/'
        text_file_path = config.RAW_TEXT_DIR / f"{example_paper_name}_raw.txt"
        runner_logger.info(f"Target raw text file: {text_file_path}")


        runner_logger.info(f"\n--- Processing Sections for Pseudocode from Text File: {text_file_path} ---")

        if not text_file_path.is_file():
            runner_logger.error(f"Required text file not found: {text_file_path}.")
            runner_logger.error(f"Please ensure the PDF parser has run successfully for '{example_paper_name}.pdf' "
                                f"or place the correct text file in the '{config.RAW_TEXT_DIR}' directory.")
            exit(1)

        runner_logger.info(f"Reading full text content from: {text_file_path}")
        try:
            full_paper_text = text_file_path.read_text(encoding='utf-8')
            if not full_paper_text.strip():
                runner_logger.error(f"Text file {text_file_path} is empty.")
                exit(1)
            runner_logger.info(f"Read {len(full_paper_text)} characters from {text_file_path}.")
        except Exception as e:
            runner_logger.error(f"Failed to read text file {text_file_path}: {e}", exc_info=True)
            exit(1)

        # Call the agent's main processing method
        generated_pseudocode_files = agent.generate_pseudocode_for_paper_sections(
            full_text=full_paper_text,
            base_filename=example_paper_name,
            output_dir_pseudocode=config.PSEUDOCODE_DIR, # Uses config for output
            raw_text_dir=config.RAW_TEXT_DIR # Passes configured raw text dir
        )

        if generated_pseudocode_files:
             runner_logger.info("\n--- Successfully Generated Pseudocode Files by Runner: ---")
             for file_path in generated_pseudocode_files:
                 runner_logger.info(f"  - {file_path}")
        else:
             runner_logger.error("\n--- Pseudocode Generation Failed or No Sections Processed by Runner ---")
             # Agent's internal logger should have more details.
             # This runner log confirms the agent's overall outcome for this run.
             runner_logger.error(f"Check agent logs ('{config.PSEUDOCODE_AGENT_LOG_FILE}') and runner logs for details.")

    except FileNotFoundError as fnf:
        runner_logger.error(f"File not found error during runner execution: {fnf}", exc_info=True)
    except ValueError as ve:
        runner_logger.error(f"Value error during runner execution: {ve}", exc_info=True)
    except Exception as e:
        runner_logger.error(f"Unexpected error during runner execution: {str(e)}", exc_info=True)

    runner_logger.info("--- Pseudocode Agent Runner Script Finished ---")