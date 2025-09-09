# pdf_parser_advanced.py

import sys
import logging
from typing import Optional, List, Union, Tuple, Dict, Any
from pathlib import Path
from multiprocessing import Pool, cpu_count
import statistics
import re
import fitz  # PyMuPDF (ensure it's installed: pip install pymupdf)

# --- Local Imports / Configuration ---
# Try to import config, but provide defaults if it fails (e.g., when run standalone)
try:
    import config
    LOG_DIR = config.LOG_DIR
    RAW_TEXT_DIR = config.RAW_TEXT_DIR
    LOG_FORMAT = config.LOG_FORMAT
    LOG_DATE_FORMAT = config.LOG_DATE_FORMAT
    # Use a distinct log file name for this parser
    PDF_PARSER_LOG_FILE = config.LOG_DIR / "pdf_parser_advanced.log"
    PDF_PARSER_LOG_LEVEL = logging.INFO # Or DEBUG for more details
except ImportError:
    print("Warning: config.py not found. Using default paths and logging format for pdf_parser_advanced.", file=sys.stderr)
    BASE_DIR = Path(__file__).resolve().parent.parent # Assumes script is in 'agents' or similar subdir
    OUTPUT_DIR = BASE_DIR / "outputs"
    LOG_DIR = OUTPUT_DIR / "logs"
    RAW_TEXT_DIR = OUTPUT_DIR / "raw_text"
    LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    PDF_PARSER_LOG_FILE = LOG_DIR / "pdf_parser_advanced.log"
    PDF_PARSER_LOG_LEVEL = logging.INFO
    # Ensure directories exist if running standalone
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)

# --- Logger Setup ---
parser_logger = logging.getLogger("pdf_parser_advanced")
if not parser_logger.hasHandlers():
    parser_logger.setLevel(PDF_PARSER_LOG_LEVEL)
    # File Handler
    fh = logging.FileHandler(PDF_PARSER_LOG_FILE, encoding='utf-8', mode='a') # Append mode
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    fh.setFormatter(formatter)
    parser_logger.addHandler(fh)
    # Prevent propagation if used within the main workflow logger
    parser_logger.propagate = False

# --- Constants for Layout Analysis ---
HEADER_MARGIN_RATIO = 0.09  # Ignore top 9% of page height
FOOTER_MARGIN_RATIO = 0.09  # Ignore bottom 9% of page height
COLUMN_THRESHOLD_RATIO = 0.45 # Min horizontal distance (as fraction of page width) to assume different columns
PARA_VERTICAL_TOLERANCE_RATIO = 0.5 # Max vertical distance (relative to font size) between lines in a para
HEADING_SIZE_THRESHOLD_RATIO = 1.1 # Font size must be > X * common_size
HEADING_BOLD_WEIGHT = 600 # Threshold for considering font weight as bold (adjust based on font)

# --- Helper Functions ---

def get_common_font_info(page: fitz.Page) -> Tuple[float, int]:
    """Analyzes fonts on the page to find the most common size and estimated weight."""
    sizes = []
    weights = [] # Approximate weights (e.g., 400 normal, 700 bold)
    try:
        # Extract text spans with font info
        text_instances = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES)["blocks"]
        for block in text_instances:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(round(span.get("size", 0), 1)) # Round size slightly
                    font_flags = span.get("flags", 0)
                    # Check PyMuPDF bold flag (more reliable than font name)
                    is_bold = (font_flags & (1 << 4)) > 0
                    # Check font name as fallback
                    if not is_bold and "bold" in span.get("font", "").lower():
                        is_bold = True
                    weights.append(700 if is_bold else 400) # Simple weight approximation

        if not sizes:
            parser_logger.debug(f"Page {page.number}: No text spans found to determine common font.")
            return 10.0, 400 # Default if no text found
        # Use mode (most frequent value)
        common_size = statistics.mode(sizes) if sizes else 10.0
        common_weight = statistics.mode(weights) if weights else 400
        parser_logger.debug(f"Page {page.number}: Common Font Size={common_size:.1f}, Approx. Weight={common_weight}")
        return float(common_size), int(common_weight)
    except Exception as e:
        parser_logger.warning(f"Page {page.number}: Could not determine common font info: {e}")
        return 10.0, 400 # Default fallback

def blocks_are_in_same_column(block1: Dict, block2: Dict, page_width: float) -> bool:
    """Checks if two blocks are likely in the same text column."""
    b1_x0, b1_x1 = block1['bbox'][0], block1['bbox'][2]
    b2_x0, b2_x1 = block2['bbox'][0], block2['bbox'][2]
    column_threshold_abs = page_width * COLUMN_THRESHOLD_RATIO

    # Calculate the horizontal distance between the midpoints of the blocks
    center1 = (b1_x0 + b1_x1) / 2
    center2 = (b2_x0 + b2_x1) / 2
    horizontal_distance = abs(center1 - center2)

    # If the distance is less than the threshold, assume same column
    return horizontal_distance < column_threshold_abs

def clean_text(text: str) -> str:
    """Basic text cleaning: removes null bytes, trims, normalizes whitespace, fixes ligatures."""
    if not isinstance(text, str): return "" # Handle non-string input gracefully
    text = text.replace('\x00', '').strip()
    # Replace common ligatures (PyMuPDF flags handle many, but some might slip through)
    ligatures = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"}
    for lig, repl in ligatures.items():
        text = text.replace(lig, repl)
    # Normalize whitespace (replace multiple spaces/newlines with single space)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove hyphenation at line breaks if text is combined later (tricky, basic removal here)
    # text = text.replace('- ', '') # Be careful with this, might join words incorrectly
    return text

# --- Core Extraction Logic ---

def extract_structured_text(pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Extracts structured text content and metadata from a PDF using layout analysis.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A tuple containing:
          - The extracted text formatted as a Markdown-like string.
          - A dictionary containing document metadata.
        Returns ("", {}) if extraction fails.
    """
    parser_logger.info(f"Starting structured text extraction for: {pdf_path.name}")
    if not pdf_path.exists():
        parser_logger.error(f"File not found: {pdf_path}")
        return "", {}

    structured_content: List[str] = []
    doc_metadata: Dict[str, Any] = {}

    try:
        doc = fitz.open(pdf_path)
        doc_metadata = doc.metadata or {}
        parser_logger.info(f"Processing PDF: '{pdf_path.name}', Pages: {len(doc)}")
        if doc_metadata:
            parser_logger.debug(f"Raw Metadata: {doc_metadata}")
            # Add cleaned metadata to output
            structured_content.append("## Document Metadata") # H2 for metadata
            for key, value in doc_metadata.items():
                 # Clean key and value, skip if value is empty/none
                 clean_key = key.replace(":", "").strip().capitalize()
                 if isinstance(value, str): clean_value = value.strip()
                 else: clean_value = value
                 if clean_key and clean_value:
                      structured_content.append(f"- **{clean_key}**: {clean_value}")
            structured_content.append("\n---\n") # Separator after metadata


        for page_num, page in enumerate(doc, start=1):
            structured_content.append(f"\n--- Page {page_num} ---\n")
            page_rect = page.rect
            page_height = page_rect.height
            page_width = page_rect.width
            header_y_limit = page_height * HEADER_MARGIN_RATIO
            footer_y_limit = page_height * (1 - FOOTER_MARGIN_RATIO)
            common_font_size, common_font_weight = get_common_font_info(page)
            # Calculate vertical tolerance based on common font size
            para_vertical_tolerance_abs = common_font_size * PARA_VERTICAL_TOLERANCE_RATIO

            parser_logger.debug(f"Page {page_num}: Dimensions={page_width:.1f}x{page_height:.1f}, HeaderLimit={header_y_limit:.1f}, FooterLimit={footer_y_limit:.1f}, ParaTol={para_vertical_tolerance_abs:.1f}")

            try:
                # Get text blocks sorted primarily by vertical position, then horizontal
                # Flags aim to preserve spaces and handle ligatures where possible
                blocks = page.get_text("blocks", sort=True, flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES)
                if not blocks:
                     parser_logger.warning(f"Page {page_num}: No text blocks found by get_text('blocks').")
                     structured_content.append("[No text content detected on this page]\n")
                     continue

                # 1. Filter headers/footers based on calculated limits
                content_blocks = []
                for block in blocks:
                    # block format: (x0, y0, x1, y1, "text lines...", block_no, block_type)
                    bbox = block[:4] # (x0, y0, x1, y1)
                    block_text = block[4]
                    if not block_text or not block_text.strip(): continue # Skip blocks with no text

                    # Filter based on vertical position (y0 of block start, y1 of block end)
                    if bbox[1] < header_y_limit or bbox[3] > footer_y_limit:
                        # parser_logger.debug(f"Page {page_num}: Skipping potential header/footer block at {bbox}")
                        continue
                    # Add block details as dict for easier access
                    content_blocks.append({
                        "bbox": bbox,
                        "text": block_text,
                        "lines": block_text.strip().split('\n'), # Basic line split
                        "block_no": block[5],
                         # Note: Need more detailed span info for font analysis within block
                    })

                if not content_blocks:
                     parser_logger.warning(f"Page {page_num}: No content blocks remaining after filtering headers/footers.")
                     structured_content.append("[No main content found on this page after filtering]\n")
                     continue

                # 2. Process sorted content blocks to infer structure (paragraphs, headings)
                current_paragraph_lines: List[str] = []
                for i, block in enumerate(content_blocks):
                    block_text = block['text'].strip()
                    block_lines = block['lines']
                    block_bbox = block['bbox']

                    # --- Font Analysis (Requires 'dict' or 'rawdict' extraction) ---
                    # Re-extract with dict for font info of this block (less efficient but needed here)
                    block_dict = page.get_text("dict", clip=block_bbox)["blocks"]
                    avg_font_size = common_font_size
                    avg_font_weight = common_font_weight
                    is_bold_block = False
                    span_sizes = []
                    span_weights = []
                    if block_dict:
                        for line_dict in block_dict[0].get("lines",[]):
                            for span_dict in line_dict.get("spans",[]):
                                span_sizes.append(round(span_dict.get("size", common_font_size),1))
                                font_flags = span_dict.get("flags", 0)
                                is_bold = (font_flags & (1 << 4)) > 0 or "bold" in span_dict.get("font","").lower()
                                span_weights.append(700 if is_bold else 400)

                        if span_sizes: avg_font_size = statistics.mean(span_sizes)
                        if span_weights: avg_font_weight = statistics.mean(span_weights)
                        is_bold_block = avg_font_weight >= HEADING_BOLD_WEIGHT
                    # --- End Font Analysis ---

                    # Simple Heading Detection
                    is_heading = (avg_font_size > common_font_size * HEADING_SIZE_THRESHOLD_RATIO) or \
                                 (is_bold_block and avg_font_weight > common_font_weight + 100) # Bold and heavier than common

                    # List item detection (simple prefix check)
                    first_line_clean = clean_text(block_lines[0]) if block_lines else ""
                    is_list_item = re.match(r'^(\*|\-|\•|\d{1,2}[\.\)])\s+', first_line_clean) is not None

                    # --- Determine if starting a new element ---
                    start_new_element = False
                    if i == 0: # First block on page is always new
                        start_new_element = True
                    else:
                        prev_block = content_blocks[i-1]
                        prev_block_bbox = prev_block['bbox']
                        vertical_gap = block_bbox[1] - prev_block_bbox[3] # Gap between bottom of prev and top of current
                        in_same_col = blocks_are_in_same_column(prev_block, block, page_width)

                        # Conditions to start new element:
                        # 1. Different column
                        # 2. Large vertical gap (more than typical line spacing)
                        # 3. Current block is detected as a heading
                        # 4. Current block looks like a list item (usually starts a new logical block)
                        # 5. Previous block ended a paragraph abruptly (e.g., ended with '.') - Harder rule
                        if not in_same_col or \
                           vertical_gap > para_vertical_tolerance_abs * 1.5 or \
                           is_heading or \
                           is_list_item:
                            start_new_element = True

                    # --- Output previous paragraph if starting new element ---
                    if start_new_element and current_paragraph_lines:
                        para_text = " ".join(clean_text(line) for line in current_paragraph_lines)
                        structured_content.append(para_text + "\n") # Add newline after paragraph
                        current_paragraph_lines = [] # Reset for next paragraph

                    # --- Process current block ---
                    cleaned_block_lines = [clean_text(line) for line in block_lines if clean_text(line)]
                    if not cleaned_block_lines: continue # Skip if block became empty after cleaning

                    if is_heading:
                        # Output the completed paragraph before the heading (if any)
                        if current_paragraph_lines:
                             para_text = " ".join(clean_text(line) for line in current_paragraph_lines)
                             structured_content.append(para_text + "\n")
                             current_paragraph_lines = []
                        # Add heading with markdown
                        level = 1 if avg_font_size > common_font_size * 1.3 else 2 # Basic level heuristic
                        heading_text = " ".join(cleaned_block_lines)
                        structured_content.append(f"{'#' * level} {heading_text}\n")
                    elif is_list_item:
                         # Output the completed paragraph before the list item (if any)
                        if current_paragraph_lines:
                             para_text = " ".join(clean_text(line) for line in current_paragraph_lines)
                             structured_content.append(para_text + "\n")
                             current_paragraph_lines = []
                        # Add list item (basic markdown, assumes block is one list item)
                        list_item_text = " ".join(cleaned_block_lines)
                        # Ensure it starts with a standard list marker for consistency downstream
                        list_item_text = re.sub(r'^(\*|\-|\•|\d{1,2}[\.\)])\s+', '', list_item_text) # Remove original marker
                        structured_content.append(f"- {list_item_text}\n") # Use standard '-' marker
                    else: # Part of a paragraph
                        current_paragraph_lines.extend(cleaned_block_lines)

                # --- Append any remaining paragraph at the end of the page ---
                if current_paragraph_lines:
                    para_text = " ".join(clean_text(line) for line in current_paragraph_lines)
                    structured_content.append(para_text + "\n")

            except Exception as page_e:
                parser_logger.error(f"Failed to process page {page_num} of '{pdf_path.name}': {page_e}", exc_info=True)
                structured_content.append(f"\n[Error processing page {page_num}]\n")

        # --- Final Assembly and Cleanup ---
        doc.close() # Close the PDF document
        final_text = "".join(structured_content).strip()
        # Further cleanup: Replace excessive newlines/spaces
        final_text = re.sub(r'\n{3,}', '\n\n', final_text) # Max 2 consecutive newlines
        final_text = re.sub(r' {2,}', ' ', final_text) # Max 1 consecutive space
        parser_logger.info(f"Successfully extracted structured text ({len(final_text)} chars) from '{pdf_path.name}'")
        return final_text, doc_metadata

    except fitz.fitz.FileDataError as fe:
        parser_logger.error(f"Failed to open or process - Not a PDF or corrupted file: '{pdf_path.name}': {fe}")
        return "", {}
    except Exception as e:
        parser_logger.error(f"General failure extracting structured text from '{pdf_path.name}': {e}", exc_info=True)
        if 'doc' in locals() and doc: doc.close() # Ensure doc is closed on error
        return "", {}


def save_structured_text(text: str, pdf_path: Path, output_dir: Path) -> Path:
    """Saves the extracted structured text to a file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use "_raw.txt" extension for compatibility with the main workflow expecting this file
    output_filename = f"{pdf_path.stem}_raw.txt"
    output_path = output_dir / output_filename
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        parser_logger.info(f"Saved structured text to: {output_path}")
        return output_path
    except IOError as e:
        parser_logger.error(f"Failed to save structured text for '{pdf_path.name}' to '{output_path}': {e}", exc_info=True)
        raise # Re-raise the exception to be caught by the caller


# --- Main Processing Functions ---

def process_single_pdf_advanced(pdf_path: Union[str, Path]) -> Optional[Path]:
    """
    Processes a single PDF using the advanced structured extraction and saves the result.
    Designed to be called by the main workflow.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Path to the saved text file if successful, None otherwise.
    """
    pdf_path = Path(pdf_path)
    parser_logger.info(f"Processing (Advanced Mode): {pdf_path.name}")

    structured_text, _ = extract_structured_text(pdf_path) # Ignore metadata return for now

    if not structured_text:
        parser_logger.warning(f"Structured text extraction failed or yielded empty content for {pdf_path.name}. No file saved.")
        return None # Indicate failure

    try:
        # Use the configured RAW_TEXT_DIR for saving
        saved_path = save_structured_text(structured_text, pdf_path, output_dir=RAW_TEXT_DIR)
        return saved_path
    except IOError:
        # Error already logged by save_structured_text
        return None # Indicate failure


def process_pdfs_advanced_parallel(pdf_paths: List[Union[str, Path]]) -> List[Path]:
    """Processes multiple PDFs in parallel using the advanced method."""
    num_files = len(pdf_paths)
    if num_files == 0: return []
    parser_logger.info(f"Starting ADVANCED BATCH processing of {num_files} PDFs using multiprocessing.")
    # Limit processes to avoid overwhelming system resources, max out at cpu_count or a reasonable number like 4-8
    num_processes = min(max(1, cpu_count() // 2), 8, num_files)
    parser_logger.info(f"Using {num_processes} parallel process(es).")

    successful_paths = []
    # Use try-finally to ensure pool closure
    pool = None
    try:
        pool = Pool(processes=num_processes)
        # Map the processing function over the list of paths
        # process_single_pdf_advanced returns Path or None
        results = pool.map(process_single_pdf_advanced, pdf_paths)
        successful_paths = [result for result in results if isinstance(result, Path)]
    except Exception as pool_e:
         parser_logger.error(f"Error during parallel processing: {pool_e}", exc_info=True)
    finally:
        if pool:
            pool.close()
            pool.join()

    failure_count = num_files - len(successful_paths)
    parser_logger.info(f"Advanced batch processing finished. Success: {len(successful_paths)}, Failures/Empty: {failure_count}")
    return successful_paths


# --- Standalone Execution Block ---
def main_standalone():
    """Main execution block for running the advanced parser directly from the command line."""
    # Add console handler when run standalone for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s') # Simple format
    console_handler.setFormatter(console_formatter)
    # Add handler only if not already present (e.g., from workflow import)
    if not any(isinstance(h, logging.StreamHandler) for h in parser_logger.handlers):
        parser_logger.addHandler(console_handler)
    parser_logger.setLevel(logging.INFO) # Ensure INFO level for console

    if len(sys.argv) < 2:
        print(f"\nUsage: python {Path(__file__).name} path/to/file1.pdf [path/to/file2.pdf...]", file=sys.stderr)
        print("\nProcesses one or more PDF files using advanced layout analysis.", file=sys.stderr)
        print(f"Outputs structured text files to: {RAW_TEXT_DIR}", file=sys.stderr)
        print(f"Logs details to: {PDF_PARSER_LOG_FILE}", file=sys.stderr)
        sys.exit(1)

    pdf_paths_str = sys.argv[1:]
    pdf_paths = [Path(p) for p in pdf_paths_str]
    valid_paths = []
    for p in pdf_paths:
        if not p.is_file():
            parser_logger.error(f"Input Error: File not found, skipping: {p}")
            print(f"Error: Input file not found, skipping: {p}", file=sys.stderr)
        else:
            valid_paths.append(p)

    if not valid_paths:
        print("Error: No valid input PDF files provided.", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing {len(valid_paths)} valid PDF file(s) using ADVANCED parser...")
    # Use the parallel processing function for multiple files
    output_files = process_pdfs_advanced_parallel(valid_paths)

    if not output_files:
        print(f"\nProcessing finished, but no text files were successfully created in '{RAW_TEXT_DIR}'.", file=sys.stderr)
        print(f"Check log file '{PDF_PARSER_LOG_FILE}' for detailed errors.", file=sys.stderr)
        sys.exit(1) # Indicate failure if no files were output

    print(f"\nSuccessfully processed {len(output_files)} files. Output saved to '{RAW_TEXT_DIR}':")
    for path in output_files:
        print(f"  - {path.name}")
    print(f"\nDetailed logs available in: {PDF_PARSER_LOG_FILE}")


if __name__ == "__main__":
    main_standalone()