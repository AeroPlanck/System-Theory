"""Audit final PDF structure and produce contact sheets from Poppler PNGs."""
from pathlib import Path
import argparse
import json

from pypdf import PdfReader
from pypdf.generic import ContentStream
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def inspect():
    documents = []
    for name in ("Methods Appendix.pdf", "PRL.pdf"):
        doc = PdfReader(ROOT/name)
        figures = []
        for number, page in enumerate(doc.pages, 1):
            text = page.extract_text()
            if "Figure " in text or "FIG." in text:
                figures.append(number)
        pages = []
        for number, page in enumerate(doc.pages, 1):
            pages.append({"page": number, "image_objects": len(page.images),
                          "text_characters": len(page.extract_text())})
        documents.append({"file": name, "page_count": len(doc.pages), "pages": pages,
                          "pages_with_figure_text": figures})
    assets = []
    for path in sorted((ROOT/"Figures/HighResolution20260903").glob("*.pdf")):
        doc = PdfReader(path)
        assert len(doc.pages) == 1
        page = doc.pages[0]
        images = len(page.images)
        assert images == 0, f"Unexpected raster object in vector redraw: {path.name}"
        operations = ContentStream(page.get_contents(), doc).operations
        paths = sum(op in (b"m", b"l", b"c", b"re") for _, op in operations)
        assert paths > 0, f"No vector drawings: {path.name}"
        assets.append({"file": path.name, "width_inches": float(page.mediabox.width)/72,
                       "height_inches": float(page.mediabox.height)/72, "raster_objects": images,
                       "vector_path_operations": paths})
    report = {"documents": documents, "vector_assets": assets}
    (ROOT/"qa"/"pdf_structure_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def contact_sheets():
    for prefix in ("methods", "prl"):
        pages = sorted((ROOT/"qa").glob(f"{prefix}-*.png"))
        for first in range(0, len(pages), 9):
            sheet = Image.new("RGB", (1320, 1810), "#e0e3e8")
            draw = ImageDraw.Draw(sheet)
            for cell, path in enumerate(pages[first:first+9]):
                x, y = (cell%3)*440, (cell//3)*600
                with Image.open(path) as page:
                    page.thumbnail((428, 565))
                    sheet.paste(page, (x+(440-page.width)//2, y+27))
                draw.text((x+12, y+7), path.stem, fill="black")
            sheet.save(ROOT/"qa"/f"{prefix}_contact_{first//9+1}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contacts-only", action="store_true")
    args = parser.parse_args()
    if not args.contacts_only:
        inspect()
    contact_sheets()
