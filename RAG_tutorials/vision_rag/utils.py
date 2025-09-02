import io, base64
from PIL import Image

MAX_PIXELS = 1568*1568

def resize_image(pil_image: Image.Image) -> Image.Image:
    w, h = pil_image.size
    if w*h > MAX_PIXELS:
        scale = (MAX_PIXELS/(w*h))**0.5
        new_w, new_h = int(w*scale), int(h*scale)
        pil_image = pil_image.copy()
        pil_image.thumbnail((new_w, new_h))
    return pil_image

def image_file_to_data_uri(path: str) -> str:
    img = Image.open(path)
    return pil_to_data_uri(img)

def pil_to_data_uri(img: Image.Image) -> str:
    fmt = img.format if img.format else "PNG"
    img = resize_image(img)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    print(f"data:image/{fmt.lower()};base64,{b64}")
    return f"data:image/{fmt.lower()};base64,{b64}"