"""
pdf_to_images.py
Converts source PDFs (Vidhan Sabha proceedings, debates, question-answer
registers, bills, etc.) into per-page images, which is the input format
every downstream retrieval and generation script expects.

Usage:
    python pdf_to_images.py --input_dir /path/to/pdfs --output_dir /path/to/page_images

Requires: pdf2image (and poppler-utils installed on the system:
    sudo apt-get install poppler-utils)
"""

import argparse
import os
from pathlib import Path

from pdf2image import convert_from_path

from config import RAW_PDF_DIR, PAGE_IMAGE_DIR


def convert_pdfs_to_images(input_dir: str, output_dir: str, dpi: int = 200) -> list:
    """
    Convert every PDF in input_dir into individual page images saved under
    output_dir/<pdf_stem>/page_0001.png, page_0002.png, ...

    Returns a list of dicts: {"doc_id": str, "page_num": int, "image_path": str}
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    pdf_paths = sorted(Path(input_dir).glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {input_dir}. Add source files there first.")
        return manifest

    for pdf_path in pdf_paths:
        doc_id = pdf_path.stem
        doc_out_dir = Path(output_dir) / doc_id
        doc_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Rendering {pdf_path.name} at {dpi} DPI ...")
        pages = convert_from_path(str(pdf_path), dpi=dpi)

        for i, page in enumerate(pages, start=1):
            image_path = doc_out_dir / f"page_{i:04d}.png"
            page.save(image_path, "PNG")
            manifest.append({
                "doc_id": doc_id,
                "page_num": i,
                "image_path": str(image_path),
            })

        print(f"  -> {len(pages)} pages written to {doc_out_dir}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs into page images.")
    parser.add_argument("--input_dir", default=RAW_PDF_DIR,
                         help="Directory containing source PDF files.")
    parser.add_argument("--output_dir", default=PAGE_IMAGE_DIR,
                         help="Directory to write rendered page images to.")
    parser.add_argument("--dpi", type=int, default=200,
                         help="Render resolution. 200-300 DPI is a good balance "
                              "of legibility vs. file size for dense text/tables.")
    args = parser.parse_args()

    manifest = convert_pdfs_to_images(args.input_dir, args.output_dir, args.dpi)
    print(f"\nDone. {len(manifest)} total pages rendered from "
          f"{len(set(m['doc_id'] for m in manifest))} documents.")


if __name__ == "__main__":
    main()
