import os
import re
import json
import time
import cv2
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import google.generativeai as genai
from pdf2image import convert_from_path
import tempfile

# -------------------- CONFIG --------------------
POPPLER_PATH = r"C:\Users\HP\Downloads\Release-25.07.0-0.zip\poppler-25.07.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Gemini API key
GEMINI_API_KEY = "api key here "
genai.configure(api_key=GEMINI_API_KEY)

# Input and output paths
file_path = r"C:\Users\HP\PycharmProjects\OCR\regcard6.png"
output_dir = "extracted_registration_output"
os.makedirs(output_dir, exist_ok=True)


# -------------------- PDF HANDLERS --------------------
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    image_files = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_filename = os.path.join(output_dir, f"page{page_num + 1}_img{img_index + 1}.png")
            if pix.n < 5:
                pix.save(img_filename)
            else:
                pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(img_filename)
            image_files.append(img_filename)
    return image_files


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        for img in images:
            text += pytesseract.image_to_string(img)
    return text


# -------------------- IMAGE HANDLERS --------------------
def extract_text_from_image(img_path):
    img = Image.open(img_path)
    return pytesseract.image_to_string(img)


def save_image(img_path):
    img = Image.open(img_path)
    save_path = os.path.join(output_dir, os.path.basename(img_path))
    img.save(save_path)
    return save_path


# -------------------- GEMINI AI EXTRACTION --------------------
def extract_registration_fields_with_gemini(text, image_path=None):
    """
    Use Gemini 2.5 Flash to extract structured registration fields.
    Automatically converts PDF to image if needed.
    """
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    prompt = f"""
    You are a smart document parser.
    Extract the following fields from a vehicle registration card:
    - First_Name
    - Last_Name
    - DOB
    - Vehicle_Make
    - Vehicle_Model
    - Vehicle_Color
    - Expiry_Date

    Rules:
    - Return all dates in MM/DD/YYYY format if possible.
    - If a field is missing or unclear, return null.
    - Return ONLY valid JSON (no extra text, no explanation).

    OCR Extracted Text:
    {text}
    """

    # --- Handle image input ---
    parts = [prompt]
    temp_image_path = None

    # Convert PDF to image if necessary
    if image_path and image_path.lower().endswith(".pdf"):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_images = convert_from_path(image_path, dpi=300, poppler_path=POPPLER_PATH)
                if pdf_images:
                    temp_image_path = os.path.join(tmpdir, "first_page.jpg")
                    pdf_images[0].save(temp_image_path, "JPEG")
                    with open(temp_image_path, "rb") as img:
                        parts.append({"mime_type": "image/jpeg", "data": img.read()})
        except Exception as e:
            print(f"[Warning] Could not convert PDF to image: {e}")

    elif image_path and image_path.lower().endswith((".jpg", ".jpeg", ".png")):
        with open(image_path, "rb") as img:
            parts.append({"mime_type": "image/jpeg", "data": img.read()})

    # --- Gemini call ---
    response = model.generate_content(parts)
    output_text = response.text.strip()

    # --- Parse JSON safely ---
    try:
        return json.loads(output_text)
    except Exception:
        match = re.search(r"\{.*\}", output_text, re.S)
        return json.loads(match.group(0)) if match else {"raw_output": output_text}


def extract_document(file_path):
    """
    Extracts text + image + structured data from a file (PDF or image).
    """
    ext = file_path.lower().split('.')[-1]
    text = ""
    images = []

    # PDF case
    if ext == "pdf":
        text = extract_text_from_pdf(file_path)
        images = extract_images_from_pdf(file_path)

    # Image case
    elif ext in ["png", "jpeg", "jpg"]:
        text = extract_text_from_image(file_path)
        saved_img = save_image(file_path)
        images = [saved_img]

    else:
        raise ValueError("Unsupported file type: " + ext)

    print("Extracting structured fields using Gemini 2.5 Flash...")

    # ✅ Now Gemini will get the image too (even if PDF)
    gemini_fields = extract_registration_fields_with_gemini(text, file_path)

    return {
        "fields": gemini_fields,
        "images": images
    }
# -------------------- RUN --------------------
if __name__ == "__main__":
    extracted_data = extract_document(file_path)

    output_json = os.path.join(output_dir, "registration_data.json")
    with open(output_json, "w") as f:
        json.dump(extracted_data, f, indent=4)

    print("\n===== Structured Fields (via Gemini) =====")
    print(json.dumps(extracted_data["fields"], indent=2))

    print("\n===== Extracted Images =====")
    for img_file in extracted_data["images"]:
        print(img_file)

    print("\nJSON saved at:", output_json)
