import os
import fitz
from PIL import Image
from typing import List
from utils import resize_image

def process_pdf_file(pdf_uploaded_file, embed_image, base_output_folder: str = "pdf_pages") -> tuple[list[str], list]:
    page_paths: List[str] = []
    page_embs: List = []
    pdf_name = pdf_uploaded_file.name
    out_dir = os.path.join(base_output_folder, os.path.splitext(pdf_name)[0])
    os.makedirs(out_dir, exist_ok=True)
    try:
        doc = fitz.open(stream=pdf_uploaded_file.read(), filetype="pdf")
        for i, page in enumerate(doc.pages()):
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = resize_image(img)
            path = os.path.join(out_dir, f"page_{i+1}.png")
            img.save(path, "PNG")
            emb = embed_image(img)
            page_paths.append(path)
            page_embs.append(emb)
        doc.close()
        return page_paths, page_embs
    except Exception:
        return [], []