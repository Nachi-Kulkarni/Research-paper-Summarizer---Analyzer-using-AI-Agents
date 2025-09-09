import logging
import sys
import re
from pathlib import Path
from typing import Optional, List
import os

# --- Third-party Imports ---
from langchain_ollama import OllamaLLM 
from langchain_core.output_parsers import StrOutputParser

# --- Local Imports ---
import config

# --- Logger Setup ---
log_file = config.LOG_DIR / "summarizer.log"
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class SummarizationAgent:
    """
    Generates a bullet-point summary using a local Ollama LLM.
    Uses simplified, direct LLM invocation.
    """

    COMMON_PREAMBLES = [
        "here are the key points:", "here is a summary:", "here are the main points:",
        "here are 10-12 key points:", "here are some key points:", "summary:",
        "key points:", "okay, here are the points:", "sure, here is the summary:",
        "based on the text:", "please provide the text", "i need the text", 
    ]

    def __init__(self, model_name: str = "gemma3:4b"): # Changed model for broader Ollama compatibility
        self.logger = logging.getLogger(__name__)
        # A flag to indicate if the chosen model is vision-capable (relevant for process_text_file)
        # This is a simplified check; real vision capability depends on the model itself.
        self.model_name_is_vision_like = "llava" in model_name.lower() or "bakllava" in model_name.lower()
        self.logger.info(f"Initializing SummarizationAgent with model: {model_name} (Vision-like: {self.model_name_is_vision_like})")
        
        self.model_name = model_name
        self.base_url = config.OLLAMA_BASE_URL
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", 4096)) # Adjusted default for smaller models
        self.logger.info(f"Using context window (num_ctx): {self.num_ctx}")

        self.system_instruction = (
            "You are an expert document summarizer. Your primary goal is to explain the provided content in a way that is **extremely easy to understand, in 10 points**, even for someone not deeply familiar with the specific topic. "
            "Your task is to generate a comprehensive yet clear summary covering the main aspects (objectives, methods, findings, conclusions) of the provided content. "
            "Prioritize **clarity and ease of understanding** above extreme brevity. Feel free to use a full page if necessary to achieve this. "
            "Structure the summary for readability. You can use well explained bullet points, or a combination. "
            "use bullet points, ensure each point is explained sufficiently. "
            "Explain any jargon or complex terms simply. "
            "CRITICAL: Output **only the summary content**. Do NOT include any introductory text like 'Here is the summary:', concluding sentences, apologies, or explanations *about the summary itself* before or after the main summary content."
        )
        self.user_instruction_template = ( 
             "\n\nPlease provide a comprehensive and very easy-to-understand summary of the text below. Explain all key aspects clearly.in 10 points\n\n"
             "--- START OF TEXT ---\n{input_text}\n--- END OF TEXT ---\n\n"
             "Easy-to-understand Summary (MUST be ONLY the summary content itself):"
        )

        try:
            self.llm = OllamaLLM(
                model=self.model_name,
                base_url=self.base_url,
                temperature=config.DEFAULT_TEMPERATURE,
                top_p=config.DEFAULT_TOP_P,
                num_ctx=self.num_ctx
            )
            self.logger.info(f"Ollama LLM (OllamaLLM) initialized successfully (model={self.model_name}, num_ctx={self.num_ctx}).")
        except Exception as e:
            self.logger.error(f"Failed to initialize OllamaLLM ({self.model_name} at {self.base_url}, num_ctx={self.num_ctx}): {e}", exc_info=True)
            raise ValueError(f"Could not initialize OllamaLLM for Summarizer.") from e


    def _strip_preamble(self, text: str) -> str:
        """Attempts to remove common introductory/refusal phrases."""
        stripped_text = text.strip()
        text_lower = stripped_text.lower()
        for _ in range(3): 
            found_preamble = False
            original_length = len(stripped_text)
            for preamble in self.COMMON_PREAMBLES:
                if text_lower.startswith(preamble):
                    preamble_len = len(preamble)
                    stripped_text = stripped_text[preamble_len:].lstrip("*: .\n") # Strip more aggressively
                    text_lower = stripped_text.lower() 
                    self.logger.debug(f"Removed preamble/refusal '{preamble}'.")
                    found_preamble = True
                    break 
            if not found_preamble or not stripped_text or len(stripped_text) == original_length:
                break
        return stripped_text

    def _invoke_llm_direct(self, text: str) -> Optional[str]:
        """
        Invokes the LLM directly with a constructed prompt string and validates output.
        """
        if not text or not text.strip():
            self.logger.error("Input text for summarization is empty.")
            return None

        user_part = self.user_instruction_template.format(input_text=text)
        full_prompt = f"{self.system_instruction}\n{user_part}"
        
        self.logger.debug(f"Invoking summarization LLM directly (prompt length approx: {len(full_prompt)} chars)...")
        try:
            response = self.llm.invoke(full_prompt)
            self.logger.debug(f"Raw summary response received (length {len(response)}).")

            if not response or not response.strip():
                 self.logger.error("LLM invocation returned an empty response.")
                 return None

            processed_response = self._strip_preamble(response)
            self.logger.debug(f"Response after potential preamble stripping (first 100): '{processed_response[:100]}...'")

            cleaned_response = processed_response.strip()
            if cleaned_response.startswith("```markdown"): cleaned_response = cleaned_response[len("```markdown"):].strip()
            elif cleaned_response.startswith("```"): cleaned_response = cleaned_response[len("```"):].strip()
            if cleaned_response.endswith("```"): cleaned_response = cleaned_response[:-len("```")].strip()

            if not cleaned_response.strip():
                 self.logger.error("Response became empty after cleaning/stripping.")
                 return None

            return cleaned_response

        except Exception as e:
            if "llama runner process has terminated" in str(e):
                 self.logger.error(f"Ollama runner crashed during summarization: {e}", exc_info=True)
                 self.logger.error("Check RAM/resources.")
            else:
                 self.logger.error(f"LLM invocation failed during summarization: {e}", exc_info=True)
            self.logger.error(f"Model: {self.model_name}, Context: {self.num_ctx}")
            return None

    def generate_summary(self, text_content: str, output_dir: Path, base_filename: str) -> Optional[Path]:
        """Generates summary using direct invocation and saves it."""
        self.logger.info(f"Starting summary generation for: {base_filename}")
        if not text_content or not text_content.strip():
            self.logger.error("Input text content is empty for summary generation.")
            return None

        summary = self._invoke_llm_direct(text_content)

        if not summary:
            self.logger.error(f"Failed to generate a valid summary for {base_filename}. No file saved.")
            return None

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.error(f"Failed to create output directory {output_dir}: {e}")
            return None

        summary_filepath = output_dir / f"{base_filename}_summary.md"
        try:
            summary_filepath.write_text(summary, encoding='utf-8')
            self.logger.info(f"Summary saved successfully to: {summary_filepath}")
            return summary_filepath
        except IOError as e:
            self.logger.error(f"Failed to write summary to {summary_filepath}: {e}")
            return None

    def process_text_file(self, input_path: Path) -> Optional[Path]:
        """
        Reads a text file (expected from main.py), generates a summary, and saves it.
        The base_filename for the output summary is derived by stripping "_raw" if present.
        """
        self.logger.info(f"Processing file for summarization: {input_path}")
        if not input_path.is_file():
            self.logger.error(f"Input file not found: {input_path}")
            return None

        text_content: Optional[str] = None
        base_filename_for_output: str

        try:
            # This agent is primarily called with .txt files from main.py's PDF parsing step.
            if input_path.suffix.lower() == '.txt':
                text_content = input_path.read_text(encoding='utf-8').strip()
                if not text_content:
                    self.logger.error(f"Input text file is empty: {input_path}")
                    return None
                
                # Derive base filename from text file (e.g., "paper_raw.txt" -> "paper")
                stem_name = input_path.stem # e.g., "MyPaper_raw"
                if stem_name.endswith("_raw"):
                    base_filename_for_output = stem_name[:-len("_raw")] # e.g., "MyPaper"
                else:
                    self.logger.warning(f"Input filename stem '{stem_name}' (from '{input_path.name}') doesn't end with '_raw'. Using full stem as base_filename.")
                    base_filename_for_output = stem_name
                self.logger.debug(f"Text input; derived base filename for summary: {base_filename_for_output}")
            
            elif self.model_name_is_vision_like and input_path.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg']:
                # This is an alternative path if a vision model is used and main.py directly sends a PDF/image.
                # Current main.py workflow does not do this for summarization.
                self.logger.warning(f"Attempting to process image/PDF {input_path} with vision model {self.model_name}. This path is not standard in main.py.")
                # Vision models with OllamaLLM usually expect image bytes in the prompt or specific handling.
                # _invoke_llm_direct needs to be adapted for `images` kwarg if this path is to be fully supported.
                # For now, creating a placeholder text.
                text_content = f"Summarize the visual content of the file: {input_path.name}."
                base_filename_for_output = input_path.stem # e.g., "MyImage" from "MyImage.png"
                self.logger.debug(f"Vision input; derived base filename for summary: {base_filename_for_output}")
            else:
                self.logger.error(f"Unsupported file type for summarization or model not vision-capable: {input_path.suffix}. Expected .txt or vision file with capable model.")
                return None

        except Exception as e:
            self.logger.error(f"Failed to read or process input file {input_path}: {e}", exc_info=True)
            return None

        if text_content is None:
            self.logger.error(f"Text content for summarization is None after processing {input_path}. This is unexpected.")
            return None

        output_summary_dir = config.SUMMARY_DIR
        return self.generate_summary(text_content, output_summary_dir, base_filename_for_output)


# --- Example Usage (__main__ block) ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    main_logger = logging.getLogger(__name__)

    if len(sys.argv) < 2:
        main_logger.error("Usage: python agents/summarizer.py <path_to_raw_text_file.txt>")
        example_paper_name = "attention" # Assumes a file like "attention_raw.txt"
        # Construct a plausible default path, preferring a test file if it exists.
        test_file_path = config.RAW_TEXT_DIR / "test_example_raw.txt" # Example test file
        if test_file_path.exists():
            default_input_path = test_file_path
        else:
            default_input_path = config.RAW_TEXT_DIR / f"{example_paper_name}_raw.txt"
        
        # Create a dummy file if none exists for testing
        if not default_input_path.exists():
            main_logger.warning(f"Default input file {default_input_path} not found. Creating a dummy file for testing.")
            try:
                default_input_path.parent.mkdir(parents=True, exist_ok=True)
                default_input_path.write_text("This is a short test document about attention mechanisms in neural networks. Attention is all you need.", encoding='utf-8')
                main_logger.info(f"Created dummy file: {default_input_path}")
            except Exception as e:
                main_logger.error(f"Could not create dummy file: {e}")
                sys.exit(1)
        
        main_logger.warning(f"No input file provided. Trying default: {default_input_path}")
        input_path_arg = default_input_path
    else:
        input_path_arg = Path(sys.argv[1])

    if not input_path_arg.is_file():
         main_logger.error(f"CRITICAL ERROR: Input text file not found: {input_path_arg}")
         sys.exit(1)

    try:
        main_logger.info("--- Initializing SummarizationAgent ---")
        agent = SummarizationAgent() # Uses default model
        main_logger.info(f"--- SummarizationAgent Initialized (Model: {agent.model_name}) ---")

        main_logger.info(f"--- Processing file: {input_path_arg} ---")
        summary_file_path = agent.process_text_file(input_path_arg)

        if summary_file_path:
            main_logger.info(f"\n--- Summary Generation Successful ---")
            main_logger.info(f"Summary saved to: {summary_file_path}")
            try:
                main_logger.info("\n--- Generated Summary ---")
                print(summary_file_path.read_text(encoding='utf-8'))
                main_logger.info("--- End of Summary ---")
            except Exception as e:
                main_logger.warning(f"Could not read back summary file for display: {e}")
        else:
            main_logger.error("\n--- Summary Generation Failed ---")
            main_logger.error(f"Check agent logs ('{log_file}') for details.")

    except ValueError as ve:
        main_logger.critical(f"Failed to initialize agent: {ve}")
    except Exception as e:
        main_logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)

    main_logger.info("\n--- Summarizer Script Finished ---")