import logging
import sys
import re
from pathlib import Path
from typing import Optional
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# --- Configuration Start ---
# Replacing 'import config' for self-contained execution
LOG_DIR = Path("logs")
CODE_DIR = Path("outputs/code") # Directory where generated Python files will be saved
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Ensure base directories exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR.mkdir(parents=True, exist_ok=True)
# --- Configuration End ---

log_file = LOG_DIR / "code_generator.log"
logger = logging.getLogger(__name__) # Module-level logger
if not logger.hasHandlers():
    handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class CodeGenerationAgent:
    def __init__(self, model_name: str = "deepseek/deepseek-chat-v3-0324:free"): 
        self.logger = logging.getLogger(__name__)
        
        openrouter_key_env = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key_env or not openrouter_key_env.strip():
            self.logger.critical("OPENROUTER_API_KEY environment variable not set or empty.")
            raise ValueError("OPENROUTER_API_KEY environment variable must be set to use OpenRouter for code generation.")
        self.openrouter_api_key = openrouter_key_env.strip()
        
        self.actual_llm_model_name = model_name 

        self.system_prompt = SystemMessage(content=(
            "You are an **Advanced Pseudocode-to-Python Code Implementer**. "
            "Your primary task is to convert detailed pseudocode into fully functional and runnable Python code. "
            "The Python code's logic, classes, functions, and control flow should accurately reflect the pseudocode. "
            "Strive to implement all described functionality, not just create skeletons. "
            "Key Directives:\n"
            "1.  **Direct Translation for Simple Steps:** Convert simple pseudocode lines (e.g., assignments, basic arithmetic, PRINT statements) into their direct Python equivalents.\n"
            "2.  **Full Implementation for Complex Operations:** For any pseudocode step describing a complex algorithm, model interaction, data processing, or significant I/O (e.g., 'Train EDCNN_model', 'Apply U-Net Segmentation', 'Load_Dataset_X', 'Perform_Statistical_Analysis'), you MUST attempt to generate a complete Python implementation. This includes:\n"
            "    a. Defining necessary functions with descriptive, snake_case names that reflect the pseudocode.\n"
            "    b. Ensuring functions accept parameters as implied or stated in the pseudocode.\n"
            "    c. Writing the functional code to perform the described operation. If specific libraries (e.g., machine learning, image processing, data analysis) are implied or would typically be used for such a task, make a reasonable attempt to use common and appropriate Python libraries (e.g., scikit-learn, TensorFlow, PyTorch, OpenCV, Pandas, NumPy, SciPy). Clearly state any major assumptions made if the pseudocode is ambiguous about the specifics of an algorithm or library by adding a comment in the code.\n"
            "3.  **Direct Variable Initialization and Computation:** If a pseudocode step results in a value, implement the logic to compute and assign that value directly. Avoid using placeholders like `variable_name = None # TODO...` unless the computation is truly impossible to infer from the pseudocode. \n"
            "4.  **Comprehensive Imports:** Include all necessary Python import statements for the libraries and modules used in your implementation (e.g., `import os`, `import numpy as np`, `from sklearn.model_selection import train_test_split`). Ensure all imports are at the beginning of the script where conventional.\n"
            "5.  **Informative Comments:** Replicate comments from the pseudocode. You may add brief, essential comments to clarify complex parts of your generated Python code if it aids understanding. Avoid excessive or obvious comments.\n"
            "6.  **Main Execution Block:** If the pseudocode describes or implies a main execution sequence or script entry point, structure this within an `if __name__ == '__main__':` block in the Python code.\n"
            "7.  **Output Format:** Respond ONLY with the Python code, enclosed in a single markdown code block: ```python\n[YOUR PYTHON CODE HERE]\n```. No introductory text, explanations, or summaries outside the code block. If you need to make an assumption due to ambiguity in the pseudocode, briefly note it as a comment within the generated Python code near the relevant section.\n"
            "8.  **Error Handling (Basic):** Where appropriate and simple to infer, include basic error handling (e.g., try-except blocks for file operations)."
        ))

        self.human_template = (
            "Translate the provided pseudocode into a complete and functional Python script. "
            "Adhere strictly to the system directives. "
            "The Python code MUST be a functional implementation, mirroring the pseudocode's logic, classes, functions, and control flow. "
            "Implement all complex operations and external model calls to the best of your ability, following the system directives. "
            "If the pseudocode implies a main execution block, create an `if __name__ == '__main__':` block. "
            "Produce ONLY the Python code within a single ```python ... ``` block. No explanations.\n\n"
            "Pseudocode:\n"
            "```pseudocode\n{pseudocode}\n```\n\n"
            "Python Code (ensure it's a runnable and functional implementation, strictly following all instructions):"
        )

        self.human_message_prompt = HumanMessagePromptTemplate.from_template(self.human_template)

        self.prompt_template = ChatPromptTemplate.from_messages([
            self.system_prompt,
            self.human_message_prompt
        ])

        self.llm = ChatOpenAI(
            model=self.actual_llm_model_name,
            temperature=0.1,
            openai_api_key=self.openrouter_api_key, 
            openai_api_base="https://openrouter.ai/api/v1", # Specify OpenRouter API base
             # Optional: add headers if OpenRouter suggests them for your app
            # default_headers={ 
            #     "HTTP-Referer": os.getenv("YOUR_APP_URL", "http://localhost"), # Replace with your app's URL
            #     "X-Title": os.getenv("YOUR_APP_NAME", "CodeGenerationAgent") # Replace with your app's name
            # }
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        self.logger.info(f"CodeGenerationAgent initialized with model '{self.actual_llm_model_name}' via OpenRouter.")

    def _extract_python_code(self, raw_output: str) -> Optional[str]:
        self.logger.debug("Attempting to extract Python code block...")
        match = re.search(r'```(?:python)?\s*(.*?)\s*```', raw_output, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            if code:
                self.logger.debug(f"Extracted code block (length: {len(code)}).")
                return code
            else:
                self.logger.warning("Found python code block but it was empty.")
                return None
        else:
            self.logger.warning(f"Could not find ```python block in the output. Raw output (first 300 chars): {raw_output[:300]}")
            stripped_output = raw_output.strip()
            if stripped_output and not any(phrase in stripped_output.lower() for phrase in ["sorry", "i cannot", "as an ai", "here is the python", "certainly", "the python code for the given pseudocode is"]):
                if any(keyword in stripped_output for keyword in ["def ", "import ", "class ", "return ", "for ", "while ", "if ", "="]):
                    self.logger.warning("No ```python block. Applying fallback: assuming raw output is code. This is a risky fallback.")
                    return stripped_output
            return None

    def _invoke_llm_chain(self, pseudocode: str) -> Optional[str]:
        if not pseudocode or not pseudocode.strip():
            self.logger.error("Input pseudocode is empty.")
            return None

        self.logger.debug(f"Invoking code generation chain (pseudocode length: {len(pseudocode)})...")
        invoke_params = {"pseudocode": pseudocode}

        try:
            response = self.chain.invoke(invoke_params)
            self.logger.debug(f"Raw code response received (length {len(response)}).")

            response_lower = response.lower()
            refusal_phrases = ["sorry", "cannot fulfill", "unable to", "i cannot", "as an ai", "i am not able", "i'm not able to provide"]
            if any(phrase in response_lower for phrase in refusal_phrases):
                 self.logger.error(f"LLM Refusal or explanation detected: '{response[:150]}...'")
                 return None

            extracted_code = self._extract_python_code(response)

            if not extracted_code:
                 self.logger.error("Code extraction failed - no valid ```python block and fallback failed/not applied.")
                 return None
            
            if len(extracted_code) < 15 and ("pseudocode" in extracted_code.lower() or "python" in extracted_code.lower() or "translate" in extracted_code.lower() or "provide" in extracted_code.lower()):
                self.logger.warning(f"Extracted code is very short and seems like an LLM comment on the task, not actual code: '{extracted_code}'")
                return None

            return extracted_code

        except Exception as e:
            if "llama runner process has terminated" in str(e): 
                 self.logger.error(f"LLM runner process crashed during code generation: {e}", exc_info=True)
            else:
                 self.logger.error(f"LLM chain invocation failed during code generation: {e}", exc_info=True)
            return None

    def generate_code(self, pseudocode_content: str, output_dir: Path, base_filename: str) -> Optional[Path]:
        self.logger.info(f"Starting code generation for: {base_filename}")
        if not pseudocode_content or not pseudocode_content.strip():
            self.logger.error("Input pseudocode content is empty.")
            return None

        generated_code = self._invoke_llm_chain(pseudocode_content)

        if not generated_code:
            self.logger.error(f"Failed to generate/extract code for {base_filename}. No file saved.")
            return None

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
             self.logger.error(f"Failed to create output directory {output_dir}: {e}")
             return None

        code_filepath = output_dir / f"{base_filename}_code.py"
        try:
            code_filepath.write_text(generated_code, encoding='utf-8')
            self.logger.info(f"Python code saved successfully to: {code_filepath}")
            return code_filepath
        except IOError as e:
            self.logger.error(f"Failed to write code to {code_filepath}: {e}")
            return None

    def process_pseudocode_file(self, input_pseudocode_path: Path, output_dir: Path, base_filename: str) -> Optional[Path]:
        """
        Processes a pseudocode file to generate Python code.

        Args:
            input_pseudocode_path: Path to the pseudocode file (.md).
            output_dir: Directory to save the generated Python file.
            base_filename: Base filename for the output Python file (e.g., "mypaper" -> "mypaper_code.py").

        Returns:
            Path to the generated Python file, or None on failure.
        """
        self.logger.info(f"Processing pseudocode file: {input_pseudocode_path} for base_filename: {base_filename}")
        if not input_pseudocode_path.is_file():
            self.logger.error(f"Input pseudocode file not found: {input_pseudocode_path}")
            return None

        self.logger.info(f"Using provided base filename for code output: {base_filename}")
        self.logger.info(f"Code will be saved in directory: {output_dir}")

        try:
            pseudocode_content = input_pseudocode_path.read_text(encoding='utf-8').strip()
            if not pseudocode_content:
                self.logger.error(f"Input pseudocode file is empty: {input_pseudocode_path}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to read pseudocode file {input_pseudocode_path}: {e}")
            return None

        return self.generate_code(pseudocode_content, output_dir, base_filename)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)] 
    )
    main_logger = logging.getLogger(__name__)

    if len(sys.argv) < 2:
        main_logger.error("Usage: python agents/code_generator.py <path_to_pseudocode_file.md>")
        # ... (rest of the example usage help text remains the same) ...
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.is_file():
         main_logger.error(f"CRITICAL ERROR: Input pseudocode file not found: {input_path}")
         sys.exit(1)
    
    # Derive base_filename for the standalone test
    # This mirrors how main.py might derive it or how user expects it
    test_base_filename: str
    if input_path.name.endswith("_pseudocode.md"):
        test_base_filename = input_path.name[:-len("_pseudocode.md")]
    elif input_path.name.endswith(".md"): # general markdown
        test_base_filename = input_path.stem # my_project.md -> my_project
    else: # any other file, use stem
        test_base_filename = input_path.stem
    main_logger.info(f"Derived base filename for standalone test: {test_base_filename}")


    try:
        main_logger.info("--- Initializing CodeGenerationAgent ---") 
        agent = CodeGenerationAgent() 
        main_logger.info(f"--- CodeGenerationAgent Initialized (Model: {agent.actual_llm_model_name}) ---")

        main_logger.info(f"--- Processing pseudocode file: {input_path} ---")
        # Call with the new signature, using global CODE_DIR for standalone test
        code_file_path = agent.process_pseudocode_file(
            input_pseudocode_path=input_path,
            output_dir=CODE_DIR, 
            base_filename=test_base_filename
        )

        if code_file_path:
            main_logger.info(f"\n--- Code Generation Successful ---")
            main_logger.info(f"Python code saved to: {code_file_path}")
            try:
                 main_logger.info("\n--- Generated Code ---")
                 print(code_file_path.read_text(encoding='utf-8'))
                 main_logger.info("--- End of Code ---")
            except Exception as e:
                 main_logger.warning(f"Could not read back generated code file for display: {e}")
        else:
            main_logger.error("\n--- Code Generation Failed ---")
            main_logger.error(f"Check agent logs ('{log_file}') for details.")

    except ValueError as ve: 
        main_logger.critical(f"Configuration or Initialization Error: {ve}", exc_info=False)
        main_logger.info(f"Ensure OPENROUTER_API_KEY is set correctly in your environment.")
    except Exception as e:
        main_logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)

    main_logger.info("\n--- Code Generation Script Finished ---")