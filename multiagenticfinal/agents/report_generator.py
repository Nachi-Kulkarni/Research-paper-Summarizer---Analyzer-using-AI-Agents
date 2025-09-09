import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import os
import re

try:
    from fpdf import FPDF
    from fpdf.enums import Align
except ImportError:
    sys.exit("CRITICAL ERROR: fpdf2 library not found. Please install it: pip install fpdf2")

PDF2IMAGE_AVAILABLE = False
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    print("INFO: pdf2image library not found. PDF embedding as image will not be available. To enable, run: pip install pdf2image")


try:
    import config
except ImportError:
    print("WARNING: config.py not found. Using default configurations.", file=sys.stderr)
    class DefaultConfig:
        LOG_DIR = Path(".") / "logs"
        SUMMARY_DIR = Path(".") / "outputs/summaries" # Adjusted default
        PSEUDOCODE_DIR = Path(".") / "outputs/pseudocode" # Adjusted default
        CODE_DIR = Path(".") / "outputs/code" # Adjusted default
        REPORT_DIR = Path(".") / "outputs/reports" # Adjusted default
        RAW_TEXT_DIR = Path(".") / "outputs/raw_text" # Adjusted default
        RESEARCH_PAPERS_DIR = Path(".") / "research_papers" # Adjusted default
        LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s"
        LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
        REPORT_TITLE_FONT_SIZE = 18
        REPORT_HEADING_FONT_SIZE = 14
        REPORT_BODY_FONT_SIZE = 10
        REPORT_CODE_FONT_SIZE = 9
        
    config = DefaultConfig()
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    config.PSEUDOCODE_DIR.mkdir(parents=True, exist_ok=True)
    config.CODE_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)


module_name = Path(__file__).stem 
log_file = config.LOG_DIR / f"{module_name}.log"

try:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
except OSError as e:
    print(f"ERROR: Could not create log directory {config.LOG_DIR}. Logging to console only for this module. Error: {e}", file=sys.stderr)
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers(): 
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        handler = logging.FileHandler(log_file, encoding='utf-8', mode='a') 
        formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False 
        logger.info(f"Initialized logger. Agent logs in: {log_file}")


class ReportGenerationAgent:
    SECTION_ORDER = ["overall_summary", "pseudocode", "code"] # Simplified, methodology often part of pseudocode
    SECTION_TITLES = {
        "overall_summary": "Overall Summary",
        "pseudocode": "Pseudocode",
        "code": "Python Code",
        # "diagrams": "Diagrams and Visuals" # Can be re-added if used
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ReportGenerationAgent (Core PDF Fonts Mode)...")

        try:
            self.report_font: str = "Arial" 
            self.code_font: str = "Courier" 
            self.title_font_size: int = getattr(config, 'REPORT_TITLE_FONT_SIZE', 18)
            self.heading_font_size: int = getattr(config, 'REPORT_HEADING_FONT_SIZE', 14)
            self.body_font_size: int = getattr(config, 'REPORT_BODY_FONT_SIZE', 10)
            self.code_font_size: int = getattr(config, 'REPORT_CODE_FONT_SIZE', 9)
            self.logger.debug(
                f"Using Core PDF Fonts: Report='{self.report_font}', Code='{self.code_font}'. "
                f"Sizes: Title={self.title_font_size}, Heading={self.heading_font_size}, "
                f"Body={self.body_font_size}, Code={self.code_font_size}"
            )
        except AttributeError as e:
            self.logger.error(f"Configuration error during font setup: {e}. Using hardcoded defaults.", exc_info=True)
            self.report_font = "Arial"
            self.code_font = "Courier"
            self.title_font_size = 18
            self.heading_font_size = 14
            self.body_font_size = 10
            self.code_font_size = 9
            self.logger.warning("Using default report font settings due to unexpected config issue.")

    def _sanitize_text_for_fpdf(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text) 

        replacements = {
            '\u2018': "'", '\u2019': "'", '\u201C': '"', '\u201D': '"',
            '\u2013': '-', '\u2014': '--', '\u2022': '*', '\u2026': '...',
            '\u00A0': ' ', '\u20AC': 'EUR',
        }
        for unicode_char, replacement_char in replacements.items():
            text = text.replace(unicode_char, replacement_char)

        try:
            text.encode('latin-1')
            return text
        except UnicodeEncodeError:
            sanitized_text = text.encode('latin-1', 'replace').decode('latin-1')
            original_sample = text[:80].replace('\n', ' ')
            sanitized_sample = sanitized_text[:80].replace('\n', ' ')
            if text != sanitized_text: # Log only if changes were made by 'replace'
                self.logger.warning(
                    f"Text contained non-Latin-1 characters after specific replacements. "
                    f"Used 'replace' strategy. Sample (original -> sanitized): '{original_sample}' -> '{sanitized_sample}'"
                )
            return sanitized_text

    def _set_font_safe(self, pdf: FPDF, font_family: str, style: str = '', size: Optional[int] = None) -> None:
        try:
            current_size = size if size is not None else pdf.font_size_pt
            pdf.set_font(font_family, style, current_size)
        except RuntimeError as e: 
            self.logger.error(f"FPDF Error setting font '{font_family}' style '{style}' size {size}: {e}. Falling back to Arial.")
            pdf.set_font("Arial", style, size if size is not None else self.body_font_size)


    def _add_heading(self, pdf: FPDF, text: str, level: int = 1):
        if level == 0: 
            font_size = self.title_font_size
            style = 'B'
            ln_after = 8 
            align = Align.C
        elif level == 1: 
            font_size = self.heading_font_size
            style = 'B'
            pdf.ln(6) # Space before section heading
            ln_after = 4
            align = Align.L
        else: 
            font_size = max(self.heading_font_size - 2, self.body_font_size)
            style = 'B'
            pdf.ln(4)
            ln_after = 2
            align = Align.L

        sanitized_text = self._sanitize_text_for_fpdf(text)
        self._set_font_safe(pdf, self.report_font, style, font_size)
        
        pdf.multi_cell(0, 10, sanitized_text, new_x="LMARGIN", new_y="NEXT", align=align, markdown=False)
        
        if ln_after > 0:
            pdf.ln(ln_after)
        self._set_font_safe(pdf, self.report_font, '', self.body_font_size)


    def _add_paragraph(self, pdf: FPDF, text: str):
        sanitized_text = self._sanitize_text_for_fpdf(text)
        self._set_font_safe(pdf, self.report_font, '', self.body_font_size)

        processed_lines = []
        for line in sanitized_text.split('\n'):
            stripped_line = line.lstrip()
            if stripped_line.startswith("* ") or stripped_line.startswith("- "):
                indent = "  " * (len(line) - len(stripped_line)) 
                processed_lines.append(f"{indent}* {stripped_line[2:]}")
            else:
                processed_lines.append(line)
        final_text_to_render = "\n".join(processed_lines)

        pdf.multi_cell(0, 5, final_text_to_render, markdown=False, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3) 

    def _add_code_block(self, pdf: FPDF, code: str):
        sanitized_code = self._sanitize_text_for_fpdf(code)
        self._set_font_safe(pdf, self.code_font, '', self.code_font_size)
        pdf.set_fill_color(240, 240, 240) 

        code_to_write = sanitized_code.replace('\t', '    ')

        pdf.multi_cell(0, 5, code_to_write, border=0, fill=True, markdown=False, new_x="LMARGIN", new_y="NEXT")
        self._set_font_safe(pdf, self.report_font, '', self.body_font_size) 
        pdf.ln(3) 

    def _add_image_or_pdf_page(self, pdf: FPDF, item_path: Path):
        # This method remains complex and depends on external libraries like pdf2image.
        # For brevity and focus on core text report, significant changes are omitted here.
        # The existing logic for adding images if PDF2IMAGE_AVAILABLE is assumed.
        if not item_path or not item_path.is_file():
            self.logger.warning(f"Visual content file not found or invalid: {item_path}")
            self._add_paragraph(pdf, self._sanitize_text_for_fpdf(f"[Visual content missing: {item_path.name}]"))
            return
        # ... (rest of the image/pdf embedding logic) ...
        self.logger.info(f"Placeholder: Image/PDF page embedding for {item_path} would happen here.")


    def _find_content_files(self, base_filename: str) -> Dict[str, Path]:
        """
        Finds content files based on the (stripped) base_filename.
        Example: base_filename="MyPaper"
        - Summary: MyPaper_summary.md
        - Pseudocode: MyPaper_pseudocode.md
        - Code: MyPaper_code.py
        """
        self.logger.info(f"Searching for content files with base name: {base_filename}")
        found_files: Dict[str, Path] = {}

        summary_path = config.SUMMARY_DIR / f"{base_filename}_summary.md"
        if summary_path.is_file():
            found_files["overall_summary"] = summary_path
            self.logger.info(f"Found summary: {summary_path}")
        else:
            self.logger.warning(f"No summary file found: {summary_path}")
            # Fallback for abstract summary if convention exists
            abstract_summary_path = config.SUMMARY_DIR / f"{base_filename}_abstract_summary.md"
            if abstract_summary_path.is_file():
                found_files["overall_summary"] = abstract_summary_path
                self.logger.info(f"Found abstract summary (fallback): {abstract_summary_path}")


        pseudocode_path = config.PSEUDOCODE_DIR / f"{base_filename}_pseudocode.md"
        if pseudocode_path.is_file():
            found_files["pseudocode"] = pseudocode_path
            self.logger.info(f"Found pseudocode: {pseudocode_path}")
        else:
            self.logger.warning(f"No pseudocode file found: {pseudocode_path}")


        code_path = config.CODE_DIR / f"{base_filename}_code.py"
        if code_path.is_file():
            found_files["code"] = code_path
            self.logger.info(f"Found code file: {code_path}")
        else:
             # Check for older naming convention if base_filename itself was "project_code"
            alt_code_path = config.CODE_DIR / f"{base_filename}.py"
            if alt_code_path.is_file() and base_filename.endswith("_code"):
                 found_files["code"] = alt_code_path
                 self.logger.info(f"Found code file (alternative naming): {alt_code_path}")
            else:
                self.logger.warning(f"No Python code file found (checked {code_path} and potentially {alt_code_path})")
        
        self.logger.debug(f"Found content files: {found_files}")
        return found_files

    def generate_report(self, base_filename: str, output_filename: Optional[str] = None) -> Optional[Path]:
        self.logger.info(f"Starting report generation for '{base_filename}' using core PDF fonts.")

        report_name_stem = output_filename if output_filename else f"{base_filename}_report"
        report_filepath = config.REPORT_DIR / f"{report_name_stem}.pdf"

        try:
            config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.error(f"Failed to create report directory {config.REPORT_DIR}: {e}", exc_info=True)
            return None

        component_files = self._find_content_files(base_filename)
        if not any(component_files.values()): 
            self.logger.error(
                f"No content files (summary, pseudocode, code) found for base '{base_filename}'. "
                "Cannot generate an empty report."
            )
            return None

        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=True, margin=15) 
        pdf.set_margins(left=15, top=15, right=15)
        pdf.add_page()
        self._set_font_safe(pdf, self.report_font, '', self.body_font_size) 

        title_text = f"Analysis Report: {base_filename.replace('_', ' ').replace('-', ' ').title()}"
        self._add_heading(pdf, title_text, level=0)

        for section_key in self.SECTION_ORDER:
            file_path = component_files.get(section_key)
            if file_path and file_path.is_file():
                section_title = self.SECTION_TITLES.get(section_key, section_key.replace('_', ' ').title())
                self._add_heading(pdf, section_title, level=1)
                try:
                    content = file_path.read_text(encoding='utf-8').strip()
                    if not content:
                        self.logger.warning(f"Content file {file_path} is empty. Skipping section '{section_title}'.")
                        self._add_paragraph(pdf, "[Section content is empty]")
                        continue

                    if section_key in ["pseudocode", "code"]:
                        self._add_code_block(pdf, content)
                    else: # Summary
                        self._add_paragraph(pdf, content)
                except Exception as e:
                    self.logger.error(f"Failed to read/add content from {file_path} for section '{section_title}': {e}", exc_info=True)
                    self._add_paragraph(pdf, self._sanitize_text_for_fpdf(f"[Error reading content for {section_title}]"))
            elif section_key in self.SECTION_TITLES: # Only log if it's an expected section
                self.logger.info(f"No file found for section: '{self.SECTION_TITLES[section_key]}'. Skipping.")


        try:
            pdf.output(str(report_filepath))
            self.logger.info(f"Report successfully generated: {report_filepath}")
            return report_filepath
        except Exception as e:
            self.logger.error(f"Failed to save PDF report to {report_filepath}: {e}", exc_info=True)
            return None

# --- Example Usage (__main__ block) ---
if __name__ == "__main__":
    if not logging.getLogger(__name__).hasHandlers() and not logger.hasHandlers(): 
        main_exec_logger = logging.getLogger(__name__ + "_main")
        main_exec_logger.setLevel(logging.INFO) 
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        main_exec_logger.addHandler(console_handler)
        main_exec_logger.propagate = False
    else:
        main_exec_logger = logger 

    main_exec_logger.info("--- Report Generation Script Started ---")

    if len(sys.argv) < 2:
        main_exec_logger.error("Usage: python agents/report_generator.py <base_filename> [output_pdf_name]")
        main_exec_logger.warning("No base_filename provided. Defaulting to 'example_paper' for demonstration.")
        base_filename_arg = "example_paper" # Default for demonstration
        # Create dummy files for the default example
        main_exec_logger.info(f"Creating dummy files for base_filename: '{base_filename_arg}'")
        try:
            (config.SUMMARY_DIR).mkdir(parents=True, exist_ok=True)
            (config.PSEUDOCODE_DIR).mkdir(parents=True, exist_ok=True)
            (config.CODE_DIR).mkdir(parents=True, exist_ok=True)

            (config.SUMMARY_DIR / f"{base_filename_arg}_summary.md").write_text("This is a dummy summary for the example paper.", encoding='utf-8')
            (config.PSEUDOCODE_DIR / f"{base_filename_arg}_pseudocode.md").write_text("BEGIN\n  PRINT 'Hello from dummy pseudocode'\nEND", encoding='utf-8')
            (config.CODE_DIR / f"{base_filename_arg}_code.py").write_text("print('Hello from dummy Python code')", encoding='utf-8')
            main_exec_logger.info("Dummy files created successfully.")
        except Exception as e:
            main_exec_logger.error(f"Could not create dummy files: {e}")
            sys.exit(1) # Exit if dummy files cannot be created for demo
    else:
        base_filename_arg = sys.argv[1]

    output_pdf_name_arg = sys.argv[2] if len(sys.argv) > 2 else None

    main_exec_logger.info(f"Attempting to generate report for base_filename: '{base_filename_arg}'.")
    main_exec_logger.info(f"Expected files (example paths, ensure they exist for your base_filename):")
    main_exec_logger.info(f"  - Summary: {config.SUMMARY_DIR / f'{base_filename_arg}_summary.md'}")
    main_exec_logger.info(f"  - Pseudocode: {config.PSEUDOCODE_DIR / f'{base_filename_arg}_pseudocode.md'}")
    main_exec_logger.info(f"  - Code: {config.CODE_DIR / f'{base_filename_arg}_code.py'}")

    try:
        main_exec_logger.info("--- Initializing ReportGenerationAgent ---")
        report_agent = ReportGenerationAgent()
        main_exec_logger.info("--- ReportGenerationAgent Initialized ---")

        report_file_path = report_agent.generate_report(base_filename_arg, output_filename=output_pdf_name_arg)

        if report_file_path and report_file_path.exists():
            main_exec_logger.info("--- Report Generation Successful ---")
            main_exec_logger.info(f"Report saved to: {report_file_path.resolve()}")
        else:
            main_exec_logger.error("--- Report Generation Failed ---")
            main_exec_logger.error(f"Please check agent logs (e.g., '{log_file.resolve()}') and verify input files for '{base_filename_arg}'.")

    except Exception as e:
        main_exec_logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)

    main_exec_logger.info("--- Report Generation Script Finished ---")