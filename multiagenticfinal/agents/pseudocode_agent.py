# --- Imports ---
# --- Standard Imports ---
import logging
import os
# import subprocess # No longer needed as mmdc is removed
# import tempfile # Not used
import re
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union # Removed Set as it was mainly for Mermaid node IDs
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
# import requests # Redundant
# import json # Redundant
import config # Import configuration

class LLMConfig:
    DEFAULT_MODEL = "gemma3:4b" # Changed default model for broader Ollama compatibility
    DEFAULT_TEMPERATURE_EXTRACT = 0.1
    DEFAULT_TEMPERATURE_SYNTAX = config.TEMPERATURE_SYNTAX # For pseudocode generation
    DEFAULT_TOP_P = config.DEFAULT_TOP_P
    DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX_PSEUDO", 8192)) # Env var specific to pseudocode agent context

    @classmethod
    def get_llm(cls,
                temperature: Optional[float] = None,
                top_p: float = DEFAULT_TOP_P,
                model_name: str = DEFAULT_MODEL,
                base_url: str = config.OLLAMA_BASE_URL,
                num_ctx: int = DEFAULT_NUM_CTX
                ):
        logger = logging.getLogger(__name__) # Get logger inside method
        effective_model = model_name if model_name else cls.DEFAULT_MODEL
        effective_temp = temperature if temperature is not None else cls.DEFAULT_TEMPERATURE_SYNTAX
        effective_top_p = top_p if top_p is not None else cls.DEFAULT_TOP_P

        try:
            effective_temp = float(effective_temp)
        except (ValueError, TypeError):
            logger.warning(f"Invalid temperature value '{effective_temp}', defaulting to 0.0.")
            effective_temp = 0.0

        try:
            effective_top_p = float(effective_top_p)
        except (ValueError, TypeError):
            logger.warning(f"Invalid top_p value '{effective_top_p}', defaulting to {cls.DEFAULT_TOP_P}.")
            effective_top_p = cls.DEFAULT_TOP_P

        llm_params = {
            "model": effective_model,
            "temperature": effective_temp,
            "top_p": effective_top_p,
            "base_url": base_url,
            "num_ctx": num_ctx, 
        }
        logger.info(f"Initializing Ollama LLM: model={effective_model}, temp={effective_temp:.2f}, top_p={effective_top_p:.2f}, base_url={base_url}, num_ctx={num_ctx}")

        try:
            llm = OllamaLLM(
                model=llm_params.get("model"),
                base_url=llm_params.get("base_url"),
                temperature=llm_params.get("temperature"),
                top_p=llm_params.get("top_p"),
                num_ctx=llm_params.get("num_ctx")
            )
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}", exc_info=True)
            raise ValueError(f"Could not initialize Ollama LLM. Ensure Ollama is running at {base_url} and model '{effective_model}' is available.") from e

# --- End LLM Configuration ---


class PseudocodeAgent:
    """
    Extracts sections from documents and generates simple pseudocode for them.
    Saves the generated pseudocode to files.
    """

    def __init__(self,
                 model_name: str = config.LLM_MODEL_GENERAL): # Uses general model from config

        self.logger = self._setup_logger()
        self.logger.info("PseudocodeAgent initializing (Mermaid/diagram functionality REMOVED)...")
        self.model_name = model_name 

        # --- Prompts ---
        self.section_extract_system_prompt = SystemMessage(content=(
    "You are an expert document analyzer. Your task is to analyze the provided document content and extract specific key sections, primarily the 'abstract' and 'methodology'.\n"
    "Output Rules:\n"
    "1. Your entire response MUST be a single, valid JSON object.\n"
    "2. The JSON object should contain two main keys: \"abstract\" and \"methodology\".\n"
    "3. For the \"abstract\" key, provide the verbatim text of the abstract section if found. If not explicitly labeled, try to identify a concise summary at the beginning of the document that serves as an abstract.\n"
    "4. For the \"methodology\" key, extract the content describing the methods, techniques, algorithms, or experimental setup used in the work. This might be under headings like 'Methods', 'Methodology', 'Experimental Setup', 'Approach', etc. If the methodology is described across multiple subsections, try to consolidate them into a coherent block of text.\n"
    "5. If either the abstract or methodology section cannot be clearly identified or is missing from the provided text, the value for that key in the JSON object should be an empty string (\"\").\n"
    "6. Do NOT include any introductory phrases, explanations, or any text outside of the JSON object itself. Your response should start with `{` and end with `}`.\n\n"
))


        self.section_extract_prompt_template = ChatPromptTemplate.from_messages([
            self.section_extract_system_prompt,
            # MessagesPlaceholder("images", optional=True), # Kept for potential future use with vision models
            ("user", "Input Content:\n\n{full_text}\n\nGenerate the JSON object based on the input content above, strictly following all rules.")
        ])

        self.pseudocode_system_prompt = SystemMessage(content=(
           "You are an expert programmer tasked with converting scientific or technical text into step-by-step pseudocode.\n"
           "Your primary goal is to create a CONCISE yet comprehensive summary of the core logic, algorithms, or processes described in the input text, presented in a clear pseudocode format.\n"
           "Pseudocode Generation Rules:\n"
           "1.  **Clarity and Conciseness**: Write very simple, clear, and CONCISE pseudocode. Focus on high-level steps and the essential logic. Avoid overly detailed or line-by-line conversion of the source text. Summarize where appropriate.\n"
           "2.  **Core Logic Focus**: Identify and represent the critical sequence of operations, decisions, and data flow. Do not include minor details, background information, or results unless they are integral to understanding the process itself.\n"
           "3.  **Standard Conventions**: Use standard pseudocode conventions. Keywords like `FUNCTION`, `PROCEDURE`, `IF-THEN-ELSE`, `ELSEIF`, `WHILE-DO`, `FOR-EACH`, `LOOP`, `REPEAT-UNTIL`, `RETURN`, `BEGIN`, `END`, `INPUT`, `OUTPUT`, `DISPLAY`, `SET`, `CALL` should be capitalized for clarity.\n"
           "4.  **Structure**: Use indentation to show control flow and block structure. Number major steps if it enhances readability. Sub-steps can be indicated with nested numbering or bullet points (e.g., `*`, `-`).\n"
           "5.  **Variable and Function Naming**: Use descriptive names for variables and functions/procedures (e.g., `calculated_value`, `ProcessDataItem(item)`).\n"
           "6.  **Mathematical Operations**: Represent mathematical formulas or complex calculations with clear descriptions or standard mathematical notation if simple (e.g., `SET y = m*x + c`, `CALCULATE accuracy_score USING true_labels, predicted_labels`).\n"
           "7.  **Data Structures**: Mention key data structures if central to the logic (e.g., `Initialize empty LIST results_list`, `FOR EACH item IN input_array`).\n"
           "8.  **Completeness for Core Logic**: Ensure the pseudocode covers the entire core process described. If the text describes multiple distinct algorithms, try to represent each, perhaps as separate functions or clearly demarcated blocks.\n"
           "9.  **Output Format**: Output ONLY the pseudocode. Do NOT include any explanations, introductory remarks (like \"Here is the pseudocode:\"), or concluding sentences outside the pseudocode itself. The entire response should be pseudocode.\n"
           "10. **Termination**: STOP generating pseudocode once the core algorithm(s) or process(es) described in the input text have been adequately summarized. Do not add filler, repeat steps unnecessarily, or try to invent logic not present in the text."
        ))

        self.pseudocode_prompt_template = ChatPromptTemplate.from_messages([
            self.pseudocode_system_prompt, ("user", "Generate pseudocode for the following text section:\n\nInput Text:\n```\n{input}\n```\n\nPseudocode:")
        ])
        self.logger.debug("Prompts defined (Section Extraction & Pseudocode Generation).")
        self.logger.info("PseudocodeAgent initialization complete (no diagram rendering).")

    @staticmethod
    def _setup_logger():
        logger = logging.getLogger(__name__)
        if not logger.hasHandlers():
            log_file = config.PSEUDOCODE_AGENT_LOG_FILE
            handler = logging.FileHandler(log_file, encoding='utf-8', mode='a') # Append mode
            formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        return logger

    def _invoke_llm(self, prompt_template: ChatPromptTemplate, temperature: float, model_name: Optional[str] = None, **kwargs) -> str:
        self.logger.debug(f"Invoking LLM (model: {model_name or self.model_name}, temp={temperature:.2f})...")
        self.logger.debug(f"LLM kwargs: {kwargs.keys()}") # Log only keys for brevity
        effective_model = model_name if model_name else self.model_name
        try:
            llm = LLMConfig.get_llm(
                temperature=temperature,
                model_name=effective_model,
                num_ctx=LLMConfig.DEFAULT_NUM_CTX # Use agent specific context window
            )
            chain = prompt_template | llm # For OllamaLLM, output is string
            response_str = chain.invoke(kwargs)
            content = response_str.strip()

            self.logger.debug(f"LLM response received (len: {len(content)}).")
            if not content or "sorry" in content.lower() or "cannot fulfill" in content.lower() or "unable to" in content.lower():
                 self.logger.warning(f"LLM response might be a refusal or empty: '{content[:100]}...'")
            return content
        except Exception as e:
            self.logger.error(f"Ollama invocation failed: {e}", exc_info=True)
            self.logger.error(f"Model used: {effective_model}, Temp: {temperature}")
            self.logger.error(f"Ollama Base URL from config: {config.OLLAMA_BASE_URL}")
            raise

    def _extract_sections(self, full_text: str) -> Dict[str, str]: # Expects string
        self.logger.info("Attempting to extract Abstract & Methods sections...")
        extracted = {"abstract": "", "methodology": ""}
        
        if not full_text or not full_text.strip():
            self.logger.warning("Input content for section extraction is empty.")
            return extracted
            
        self.logger.debug(f"Content for section extraction (first 200 chars): {full_text[:200]}")
        try:
            # Using the general model specified for the agent, or default from LLMConfig if section model is different
            section_extraction_model = LLMConfig.DEFAULT_MODEL # Could be different if needed
            resp = self._invoke_llm(
                self.section_extract_prompt_template,
                temperature=LLMConfig.DEFAULT_TEMPERATURE_EXTRACT,
                model_name=section_extraction_model,
                full_text=full_text 
            )
            self.logger.debug(f"Raw section extract response (first 200 chars):\n{resp[:200]}")
            
            # Attempt to parse JSON from the response
            json_str = resp
            # Remove markdown fences if present
            match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', resp, re.DOTALL)
            if match:
                json_str = match.group(1)
            else: # If no fences, try to find JSON directly
                json_start = json_str.find('{')
                json_end = json_str.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_str = json_str[json_start : json_end+1]
                else:
                    self.logger.error(f"Could not find valid JSON structure in response: {resp}")
                    return extracted # Return empty if no JSON found

            self.logger.debug(f"Attempting to parse JSON: {json_str}")
            data = json.loads(json_str)

            if isinstance(data, dict):
                extracted["abstract"] = str(data.get("abstract", "")).strip()
                extracted["methodology"] = str(data.get("methodology", "")).strip()
                self.logger.info(f"Parsed sections. Abstract: {len(extracted['abstract'])} chars, Methods: {len(extracted['methodology'])} chars.")
                self.logger.debug(f"Extracted Abstract (first 100 chars): {extracted['abstract'][:100]}")
                self.logger.debug(f"Extracted Methodology (first 100 chars): {extracted['methodology'][:100]}")
            else:
                self.logger.error(f"Section extraction response was JSON but not a dictionary: {type(data)}")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed during section extraction: {e}. Response was: {resp}")
        except Exception as e:
            self.logger.error(f"LLM call or processing for section extraction failed: {e}", exc_info=True)

        if not extracted["abstract"]: self.logger.warning("Abstract section extraction appears to have failed or section was empty.")
        if not extracted["methodology"]: self.logger.warning("Methodology section extraction appears to have failed or section was empty.")
        return extracted

    def generate_pseudocode(self, text: str, output_dir: Path, base_filename: str) -> Tuple[Optional[str], Optional[Path]]:
        if not text or not text.strip():
            self.logger.error(f"Input text for pseudocode generation for '{base_filename}' is empty.")
            return None, None

        self.logger.info(f"Generating pseudocode for '{base_filename}'...")
        self.logger.debug(f"Input text length: {len(text)} chars. Text (first 100): {text[:100]}")
        output_dir.mkdir(parents=True, exist_ok=True)
        pseudocode_filepath = output_dir / f"{base_filename}_pseudocode.md"
        self.logger.info(f"Pseudocode will be saved to: {pseudocode_filepath}")

        try:
            pseudocode = self._invoke_llm(
                self.pseudocode_prompt_template,
                temperature=LLMConfig.DEFAULT_TEMPERATURE_SYNTAX,
                model_name=self.model_name, # Use agent's general model
                input=text
            )

            # Clean up potential markdown fences around pseudocode
            if pseudocode.lower().startswith("```pseudocode"):
                pseudocode = pseudocode[len("```pseudocode"):].strip()
            elif pseudocode.startswith("```"): 
                 pseudocode = pseudocode[len("```"):].strip()
            if pseudocode.endswith("```"):
                pseudocode = pseudocode[:-len("```")].strip()

            if not pseudocode or len(pseudocode) < 20 or "generate pseudocode" in pseudocode.lower() or "unable to provide" in pseudocode.lower():
                 self.logger.warning(f"Generated pseudocode for '{base_filename}' seems unusually short, generic, or a refusal: '{pseudocode[:150]}...'")
                 if not pseudocode: return None, None # Explicitly return None if empty after cleaning

            self.logger.info(f"Pseudocode generation successful for '{base_filename}'.")
            self.logger.debug(f"Generated Pseudocode (first 200 chars):\n---\n{pseudocode[:200]}\n---")

            try:
                pseudocode_filepath.write_text(pseudocode, encoding='utf-8')
                self.logger.info(f"Pseudocode saved to: {pseudocode_filepath}")
                return pseudocode, pseudocode_filepath
            except IOError as e:
                self.logger.error(f"Failed to write pseudocode to {pseudocode_filepath}: {e}", exc_info=True)
                return pseudocode, None # Return pseudocode string even if save fails, path is None

        except Exception as e:
            self.logger.error(f"Pseudocode generation failed for '{base_filename}': {e}", exc_info=True)
            return None, None

    def generate_pseudocode_for_paper_sections(
        self,
        full_text: Optional[str] = None, # Expect string content directly
        base_filename: str = "paper",
        output_dir_pseudocode: Path = config.PSEUDOCODE_DIR,
        raw_text_dir: Optional[Path] = None # Kept for compatibility but full_text preferred
        ) -> List[Path]:
        self.logger.info(f"Starting 'generate_pseudocode_for_paper_sections' for: {base_filename}")
        
        output_dir_pseudocode.mkdir(parents=True, exist_ok=True)

        effective_text = full_text
        if not effective_text and raw_text_dir: # Fallback to reading from file if full_text not given
            raw_text_path = raw_text_dir / f"{base_filename}_raw.txt" # This assumes base_filename needs _raw appended
            self.logger.warning(f"full_text not provided, attempting to read from {raw_text_path}. This fallback might use an inconsistent base_filename if the caller's base_filename already stripped '_raw'.")
            try:
                if raw_text_path.exists():
                    effective_text = raw_text_path.read_text(encoding='utf-8')
                    self.logger.info(f"Successfully read raw text from: {raw_text_path}")
                else:
                    self.logger.error(f"Raw text file not found at fallback path: {raw_text_path}")
                    return []
            except Exception as e:
                self.logger.error(f"Failed to read raw text file from fallback path: {e}")
                return []
        
        if not effective_text or not effective_text.strip():
            self.logger.error(f"No valid text content available for processing for {base_filename}.")
            return []

        self.logger.info(f"Calling _extract_sections for {base_filename}...")
        sections = self._extract_sections(effective_text) # Pass string
        self.logger.info(f"Extracted sections for {base_filename}: {list(s for s, c in sections.items() if c)}") # Log only non-empty extracted sections
        
        if not any(sections.values()): # Check if all section values are empty strings
            self.logger.error(f"No sections (Abstract/Methodology) could be extracted or all were empty for {base_filename}. Aborting pseudocode generation.")
            return []

        consolidated_text_parts = []
        if sections.get("abstract"):
            consolidated_text_parts.append(f"--- ABSTRACT ---\n{sections['abstract']}")
        if sections.get("methodology"):
            consolidated_text_parts.append(f"--- METHODOLOGY ---\n{sections['methodology']}")
        
        if not consolidated_text_parts:
            self.logger.error(f"No content found in extracted Abstract/Methodology sections for {base_filename} to consolidate.")
            return []
            
        consolidated_text = "\n\n".join(consolidated_text_parts).strip()
        if not consolidated_text:
            self.logger.error(f"Consolidated text for {base_filename} is empty after joining sections.")
            return []

        self.logger.info(f"Consolidated text from sections for {base_filename} (length: {len(consolidated_text)} chars).")
        
        # Use the 'base_filename' passed by main.py (which should be stripped of _raw)
        # for the output pseudocode file.
        self.logger.info(f"--- Generating consolidated pseudocode for: {base_filename} ---")
        try:
            pseudocode_str, pseudocode_path = self.generate_pseudocode(
                consolidated_text,
                output_dir=output_dir_pseudocode,
                base_filename=base_filename 
            )

            if pseudocode_path: # Check if path is not None (i.e., file saved)
                self.logger.info(f"--- Consolidated pseudocode generated and saved to {pseudocode_path} ---")
                return [pseudocode_path] 
            elif pseudocode_str: # Generated but not saved
                self.logger.warning(f"--- Consolidated pseudocode generated for {base_filename}, but FAILED TO SAVE to file. ---")
                return []
            else: # Not generated
                self.logger.error(f"Consolidated pseudocode generation failed for {base_filename}.")
                return []
        except Exception as e:
            self.logger.error(f"Unexpected error generating consolidated pseudocode for {base_filename}: {e}", exc_info=True)
            return []

    # --- Deprecated/Alternative Methods (Updated for pseudocode only) ---
    def generate_pseudocode_from_text(
        self,
        text: str,
        output_dir: Path = config.PSEUDOCODE_DIR,
        output_filename: Optional[str] = None # This is base_filename
    ) -> Optional[Path]:
        self.logger.warning("[DEPRECATED API STYLE] Calling generate_pseudocode_from_text. Consider using generate_pseudocode_for_paper_sections for standard workflow.")
        base_fname = output_filename or "standalone_text" # Ensure a default
        
        try:
            self.logger.info(f"Generating pseudocode for single text block, output to: {output_dir / (base_fname + '_pseudocode.md')}")
            _pseudo_str, pseudo_path = self.generate_pseudocode(text, output_dir, base_fname)
            
            if pseudo_path:
                self.logger.info(f"Pseudocode for single text block saved to {pseudo_path}")
            elif _pseudo_str:
                 self.logger.warning(f"Pseudocode generated for '{base_fname}' but failed to save.")
            else:
                 self.logger.error(f"Pseudocode generation failed for '{base_fname}'.")
            return pseudo_path # Return path (None if save failed or generation failed)
        except Exception as e:
            self.logger.error(f"Unexpected error generating pseudocode from text for '{base_fname}': {e}", exc_info=True)
            return None

    def generate_pseudocode_from_file(
        self,
        input_path: Path,
        output_dir: Path = config.PSEUDOCODE_DIR,
        output_filename: Optional[str] = None # This is base_filename
    ) -> Optional[Path]:
        self.logger.warning("[DEPRECATED API STYLE] Calling generate_pseudocode_from_file. Consider using main workflow with generate_pseudocode_for_paper_sections.")
        if not input_path.is_file():
            self.logger.error(f"Input path is not a file: {input_path}")
            return None
        
        self.logger.info(f"Processing single file for pseudocode generation: '{input_path.name}'. Output dir: {output_dir}")
        try:
            content = input_path.read_text(encoding='utf-8').strip()
            if not content:
                self.logger.error(f"Input file {input_path} is empty.")
                return None
            
            effective_output_filename = output_filename or input_path.stem
            return self.generate_pseudocode_from_text(
                text=content,
                output_dir=output_dir,
                output_filename=effective_output_filename
            )
        except Exception as e:
            self.logger.error(f"Error processing file {input_path} for pseudocode: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, # Set to DEBUG for more verbose output from agent
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)] # Log to console for standalone run
    )
    main_logger = logging.getLogger(__name__) # This will use the basicConfig

    try:
        main_logger.info("--- Initializing PseudocodeAgent (for pseudocode generation only) ---")
        agent = PseudocodeAgent()
        main_logger.info("--- PseudocodeAgent Initialized ---")

        # Example: Use a default text file if no argument is provided
        if len(sys.argv) > 1:
            text_file_path_arg = Path(sys.argv[1])
            if not text_file_path_arg.is_file():
                main_logger.error(f"Provided text file not found: {text_file_path_arg}")
                sys.exit(1)
            example_paper_name_arg = text_file_path_arg.stem.replace("_raw", "") # Derive base name
        else:
            example_paper_name_arg = "attention" # Default example
            text_file_path_arg = config.RAW_TEXT_DIR / f"{example_paper_name_arg}_raw.txt"
            main_logger.warning(f"No input file provided. Using default: {text_file_path_arg}")
            if not text_file_path_arg.exists():
                main_logger.info(f"Default file {text_file_path_arg} not found. Creating a dummy file.")
                try:
                    text_file_path_arg.parent.mkdir(parents=True, exist_ok=True)
                    dummy_content = (
                        "Abstract: This paper is about testing. It has an abstract and a methodology.\n\n"
                        "Methodology: First, we define the problem. Second, we collect data. Third, we analyze results. This involves several steps like data cleaning and statistical modeling. Finally, we conclude."
                    )
                    text_file_path_arg.write_text(dummy_content, encoding='utf-8')
                    main_logger.info(f"Created dummy file: {text_file_path_arg}")
                except Exception as e_create:
                    main_logger.error(f"Could not create dummy file {text_file_path_arg}: {e_create}")
                    sys.exit(1)


        main_logger.info(f"\n--- Processing Sections for Pseudocode from Text File: {text_file_path_arg} (Base name: {example_paper_name_arg}) ---")

        try:
            full_paper_text_content = text_file_path_arg.read_text(encoding='utf-8')
            if not full_paper_text_content.strip():
                main_logger.error(f"Text file {text_file_path_arg} is empty.")
                sys.exit(1)
            main_logger.info(f"Read {len(full_paper_text_content)} characters.")
        except Exception as e_read:
            main_logger.error(f"Failed to read text file {text_file_path_arg}: {e_read}", exc_info=True)
            sys.exit(1)

        generated_pseudocode_files_list = agent.generate_pseudocode_for_paper_sections(
            full_text=full_paper_text_content,
            base_filename=example_paper_name_arg, # Pass the stripped base name
            output_dir_pseudocode=config.PSEUDOCODE_DIR
        )

        if generated_pseudocode_files_list:
             main_logger.info("\n--- Successfully Generated Pseudocode Files: ---")
             for file_path_item in generated_pseudocode_files_list:
                 main_logger.info(f"  - {file_path_item}")
                 try:
                     print("\n--- Pseudocode Content ---")
                     print(file_path_item.read_text(encoding='utf-8'))
                     print("--- End of Pseudocode Content ---\n")
                 except Exception as e_display:
                     main_logger.warning(f"Could not display content of {file_path_item}: {e_display}")
        else:
             main_logger.error("\n--- Pseudocode Generation Failed or No Sections Processed ---")
             main_logger.error(f"Check agent logs ('{config.PSEUDOCODE_AGENT_LOG_FILE}') for details.")

    except FileNotFoundError as fnf_err: main_logger.error(f"File not found error during setup: {fnf_err}")
    except ValueError as ve_err: main_logger.error(f"Value error during setup or agent call: {ve_err}", exc_info=True)
    except Exception as unexp_err: main_logger.error(f"Unexpected error during main execution: {str(unexp_err)}", exc_info=True)

    main_logger.info("--- Pseudocode Agent Script Execution Finished ---")