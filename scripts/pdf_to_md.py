import argparse
import pymupdf4llm
import os
import sys

def convert_pdf_to_markdown(pdf_path, output_path=None):
    """
    Converts a PDF file to Markdown using PyMuPDF4LLM.
    """
    # 1. Validate input file
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' was not found.")
        return

    if not pdf_path.lower().endswith('.pdf'):
        print("Error: The input file must be a PDF.")
        return

    # 2. Determine output path if not provided
    if output_path is None:
        # Replace .pdf extension with .md
        output_path = os.path.splitext(pdf_path)[0] + ".md"

    print(f"Processing: {pdf_path}...")

    try:
        # 3. Perform the conversion
        # pymupdf4llm extracts text, identifies headers/tables, and formats as MD
        md_text = pymupdf4llm.to_markdown(pdf_path)

        # 4. Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_text)

        print(f"Success! Markdown saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Convert a PDF file to Markdown.")
    parser.add_argument("pdf_file", help="Path to the input PDF file")
    parser.add_argument("-o", "--output", help="Path to the output MD file (optional)", default=None)

    args = parser.parse_args()

    convert_pdf_to_markdown(args.pdf_file, args.output)