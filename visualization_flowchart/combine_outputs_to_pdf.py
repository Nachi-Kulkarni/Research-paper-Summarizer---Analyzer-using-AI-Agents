import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from fpdf import FPDF
from PIL import Image # For getting image dimensions

# Assuming config.py is in the same directory or accessible
try:
    import config
    DIAGRAM_DIR = config.DIAGRAM_DIR
    OUTPUT_DIR_GEN_DIAGRAM = config.OUTPUT_DIR / "diagrams" # As used in generate_diagram.py
except ImportError:
    print("Error: config.py not found or DIAGRAM_DIR/OUTPUT_DIR not defined. Using default paths.")
    # Define fallbacks if config is not available (adjust as needed)
    BASE_DIR_FALLBACK = Path(__file__).resolve().parent
    DIAGRAM_DIR = BASE_DIR_FALLBACK / "diagrams"
    OUTPUT_DIR_GEN_DIAGRAM = BASE_DIR_FALLBACK / "outputs" / "diagrams"


def run_script(script_name: str, pdf_input_path: Path) -> Optional[List[Path]]:
    """Runs a given script with the PDF input and returns output image paths."""
    script_path = Path(__file__).resolve().parent / script_name
    if not script_path.exists():
        print(f"Error: Script {script_name} not found at {script_path}")
        return None

    print(f"Running {script_name} for {pdf_input_path.name}...")
    try:
        process = subprocess.run(
            [sys.executable, str(script_path), str(pdf_input_path)],
            capture_output=True,
            text=True,
            check=False  # Don't raise exception for non-zero exit, handle manually
        )
        print(f"--- Output from {script_name} ---")
        print(process.stdout)
        if process.stderr:
            print(f"--- Errors from {script_name} ---")
            print(process.stderr)
        
        if process.returncode != 0:
            print(f"Error: {script_name} failed with exit code {process.returncode}")
            return None

        # Parse output to find image paths
        image_paths = []
        if script_name == "pdf_to_visualization.py":
            # pdf_to_visualization.py prints paths like "- diagrams/attention_viz_1.png"
            for line in process.stdout.splitlines():
                if line.strip().startswith("- "):
                    # Construct absolute path based on DIAGRAM_DIR from config
                    # Assuming the script outputs paths relative to the workspace root
                    # and DIAGRAM_DIR is also relative to workspace root or an absolute path.
                    relative_img_path = line.strip().lstrip("- ")
                    # Check if DIAGRAM_DIR is already part of the path
                    if str(DIAGRAM_DIR) in relative_img_path:
                         img_path = Path(relative_img_path)
                    else:
                         img_path = DIAGRAM_DIR.parent / relative_img_path # Go up one level from DIAGRAM_DIR if it's like 'tryneway/diagrams'
                    
                    if img_path.exists():
                        image_paths.append(img_path.resolve())
                    else:
                        # Fallback: try resolving relative to current working directory if not found
                        # This assumes the script might be run from the workspace root.
                        ws_root_path = Path.cwd() / relative_img_path
                        if ws_root_path.exists():
                            image_paths.append(ws_root_path.resolve())
                        else:
                            print(f"Warning: Could not find generated image: {img_path} or {ws_root_path}")
            print(f"Found visualization images: {image_paths}")

        elif script_name == "generate_diagram.py":
            # generate_diagram.py prints paths like "Diagram saved to: /path/to/diagram.png"
            # And the diagram name is based on input PDF name + _flowchart_diagram.png
            # e.g., attention_flowchart_diagram.png
            # It saves to config.OUTPUT_DIR / "diagrams"
            expected_diagram_name = f"{pdf_input_path.stem}_flowchart_diagram.png"
            img_path = OUTPUT_DIR_GEN_DIAGRAM / expected_diagram_name
            if img_path.exists():
                image_paths.append(img_path.resolve())
            else:
                print(f"Warning: Could not find generated diagram: {img_path}")
            print(f"Found flowchart diagram: {image_paths}")
            
        return image_paths

    except FileNotFoundError:
        print(f"Error: Python interpreter '{sys.executable}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while running {script_name}: {e}")
        return None


def create_pdf_from_images(image_paths: List[Path], output_pdf_path: Path):
    """Creates a PDF document from a list of image paths."""
    if not image_paths:
        print("No images provided to create PDF.")
        return

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False, margin=0) # Disable auto page break initially
    pdf.add_page()

    current_y = pdf.t_margin # Start at top margin
    image_padding_mm = 5 # Padding between images

    for image_path in image_paths:
        if not image_path.exists():
            print(f"Warning: Image not found at {image_path}, skipping.")
            continue
        
        try:
            with Image.open(image_path) as img:
                width_px, height_px = img.size
            
            dpi = 96
            width_mm = (width_px / dpi) * 25.4
            height_mm = (height_px / dpi) * 25.4

            page_width_mm = pdf.w - pdf.l_margin - pdf.r_margin
            page_height_mm = pdf.h - pdf.t_margin - pdf.b_margin
            
            aspect_ratio = width_mm / height_mm

            # Scale image to fit page width
            display_width_mm = page_width_mm
            display_height_mm = display_width_mm / aspect_ratio

            # If image is too tall even after scaling to page width, scale by height instead
            if display_height_mm > page_height_mm : # Check if it would fit if it was the only image
                display_height_mm = page_height_mm
                display_width_mm = display_height_mm * aspect_ratio
                # Recenter if scaled by height
                if display_width_mm > page_width_mm: # Should not happen if logic is correct
                    display_width_mm = page_width_mm
                    display_height_mm = display_width_mm / aspect_ratio


            # Check if image fits on current page
            if current_y + display_height_mm + image_padding_mm > (pdf.h - pdf.b_margin):
                pdf.add_page()
                current_y = pdf.t_margin # Reset Y position for new page

            # Center the image horizontally
            x_pos = (pdf.w - display_width_mm) / 2
            
            pdf.image(str(image_path), x=x_pos, y=current_y, w=display_width_mm, h=display_height_mm)
            current_y += display_height_mm + image_padding_mm # Update Y position for next image
            print(f"Added {image_path.name} to PDF at y={current_y:.2f}mm.")

        except Exception as e:
            print(f"Error adding image {image_path} to PDF: {e}")

    try:
        pdf.output(str(output_pdf_path))
        print(f"Successfully created PDF: {output_pdf_path}")
    except Exception as e:
        print(f"Error saving PDF {output_pdf_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run visualization and diagram generation scripts and combine outputs into a PDF.")
    parser.add_argument("pdf_input_path", type=Path, help="Path to the input PDF file.")
    args = parser.parse_args()

    if not args.pdf_input_path.exists():
        print(f"Error: Input PDF file not found: {args.pdf_input_path}")
        sys.exit(1)

    all_image_paths = []

    # Run generate_diagram.py
    diagram_images = run_script("generate_diagram.py", args.pdf_input_path)
    if diagram_images:
        all_image_paths.extend(diagram_images)

    # Run pdf_to_visualization.py
    visualization_images = run_script("pdf_to_visualization.py", args.pdf_input_path)
    if visualization_images:
        all_image_paths.extend(visualization_images)
    
    if not all_image_paths:
        print("No images were generated by the scripts. PDF will not be created.")
        sys.exit(1)
        
    # Deduplicate paths if any script somehow produced the same path (unlikely with current naming)
    unique_image_paths = sorted(list(set(all_image_paths)))
    print(f"\nCollected unique image paths for PDF: {unique_image_paths}")

    output_pdf_filename = f"{args.pdf_input_path.stem}_combined_report.pdf"
    # Save PDF in the same directory as the input PDF, or a dedicated output dir
    output_pdf_path = args.pdf_input_path.parent / output_pdf_filename
    # Alternatively, save to a fixed output directory:
    # output_pdf_path = DIAGRAM_DIR.parent / "combined_reports" / output_pdf_filename
    # (output_pdf_path.parent).mkdir(parents=True, exist_ok=True)


    create_pdf_from_images(unique_image_paths, output_pdf_path)

if __name__ == "__main__":
    # Ensure DIAGRAM_DIR and OUTPUT_DIR_GEN_DIAGRAM exist before running
    DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_GEN_DIAGRAM.mkdir(parents=True, exist_ok=True)
    main()