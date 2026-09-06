"""Run with the bundled Python (pypdf) after redraw_theory_figures.py."""
from pathlib import Path
import json
from pypdf import PdfReader
from pypdf.generic import ContentStream

ROOT = Path(__file__).resolve().parent


def embedded(font):
    font = font.get_object()
    descriptor = font.get("/FontDescriptor")
    if descriptor:
        descriptor = descriptor.get_object()
        if any(name in descriptor for name in ("/FontFile", "/FontFile2", "/FontFile3")):
            return True
    return all(embedded(child) for child in font.get("/DescendantFonts", [])) if font.get("/DescendantFonts") else False


def main():
    manifest = json.loads((ROOT/"Theory_Figure_Verification.json").read_text(encoding="utf-8"))
    checks = []
    for figure in manifest["figures"]:
        reader = PdfReader(ROOT/figure["pdf"])
        assert len(reader.pages) == 1
        page = reader.pages[0]
        resources = page["/Resources"].get_object()
        fonts = resources["/Font"].get_object()
        operations = ContentStream(page.get_contents(),reader).operations
        paths = sum(op in (b"m", b"l", b"c", b"re") for _,op in operations)
        assert len(page.images) == 0, "Raster image found in purported vector figure"
        assert all(embedded(font) for font in fonts.values()), "Unembedded font"
        assert paths > 0 and len(page.extract_text()) > 10
        checks.append({"pdf":figure["pdf"], "pages":1, "raster_images":0,
                       "all_fonts_embedded":True,"font_resources":len(fonts),
                       "vector_path_operations":paths,
                       "page_size_points":[float(page.mediabox.width),float(page.mediabox.height)],
                       "extracted_text_characters":len(page.extract_text()),
                       "PNG_dpi":figure["png_dpi"]})
    (ROOT/"Theory_PDF_Structure_QA.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2))


if __name__ == "__main__":
    main()
