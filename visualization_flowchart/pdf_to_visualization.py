"""
Main script to process PDF research papers and generate visualizations
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

# Local imports
from pdf_parser_advanced import process_single_pdf_advanced
from visualization_agent import generate_visualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_to_visualization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def ensure_diagrams_dir() -> Path:
    """Ensure diagrams directory exists and return its path"""
    diagrams_dir = Path("diagrams")
    diagrams_dir.mkdir(exist_ok=True)
    return diagrams_dir

def save_visualizations(images: List[str], pdf_path: Path, output_dir: Path) -> List[Path]:
    """Save base64 encoded images to PNG files"""
    saved_paths = []
    for i, img_data in enumerate(images):
        try:
            import base64
            img_bytes = base64.b64decode(img_data)
            output_path = output_dir / f"{pdf_path.stem}_viz_{i+1}.png"
            with open(output_path, 'wb') as f:
                f.write(img_bytes)
            saved_paths.append(output_path)
            logger.info(f"Saved visualization to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization {i+1}: {str(e)}")
    return saved_paths

def process_pdf_to_visualizations(pdf_path: Path) -> Optional[List[Path]]:
    """
    Process a PDF file and generate visualizations
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of paths to generated visualizations or None if failed
    """
    try:
        # Step 1: Process PDF to extract text and images
        logger.info(f"Processing PDF: {pdf_path.name}")
        result = process_single_pdf_advanced(pdf_path)
        if not result or not result.get('text'):
            logger.error("Failed to extract text from PDF")
            return None
            
        # Step 2: Generate visualizations
        logger.info("Generating visualizations...")
        images = generate_visualization(
            paper_text=result['text'],
            paper_images=result.get('images', [])
        )
        
        if not images:
            logger.error("No visualizations generated")
            return None
            
        # Step 3: Save visualizations
        diagrams_dir = ensure_diagrams_dir()
        saved_paths = save_visualizations(images, pdf_path, diagrams_dir)
        
        return saved_paths
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {str(e)}", exc_info=True)
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations from research paper PDFs')
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)

    result_paths = process_pdf_to_visualizations(pdf_path)
    
    if result_paths:
        print("\nSuccessfully generated visualizations:")
        for path in result_paths:
            print(f"- {path}")
    else:
        print("\nFailed to generate visualizations")
        sys.exit(1)

if __name__ == "__main__":
    main()