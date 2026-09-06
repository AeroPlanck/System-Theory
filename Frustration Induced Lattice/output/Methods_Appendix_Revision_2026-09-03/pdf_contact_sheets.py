"""Contact sheets of Poppler-rendered pages for whole-document visual QA."""
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
pages = sorted(root.glob("qa_page-*.png"))
for first in range(0, len(pages), 12):
    sheet = Image.new("RGB", (1240, 1690), "#d8dce0")
    draw = ImageDraw.Draw(sheet)
    for cell, path in enumerate(pages[first:first+12]):
        x, y = (cell % 4)*310, (cell // 4)*560
        with Image.open(path) as page:
            page.thumbnail((298, 520))
            sheet.paste(page, (x+6, y+25))
        draw.text((x+10,y+7), f"Page {first+cell+1}", fill="black")
    sheet.save(root/f"qa_contact_{first//12+1}.png")
