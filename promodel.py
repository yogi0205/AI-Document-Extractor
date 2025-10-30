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
from dotenv import load_dotenv
import google.generativeai as genai

# -------------------- CONFIG --------------------
POPPLER_PATH = r"C:\Users\HP\Downloads\Release-25.07.0-0.zip\poppler-25.07.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Gemini API key
load_dotenv(".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ Gemini API key not found. Please set GEMINI_API_KEY in .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# Input and output
file_path = r"C:\Users\HP\PycharmProjects\OCR\dlsample\dlsample10.jpg"
output_dir = "extracted_output"
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
    except:
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


def merge_overlapping_boxes(boxes, overlap_thresh=0.3):
    """
    Merge overlapping or touching bounding boxes into a single box.
    """
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [boxes[0]]

    for current in boxes[1:]:
        last = merged[-1]
        (x1, y1, w1, h1) = last
        (x2, y2, w2, h2) = current

        # Compute overlap
        overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
        overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)
        overlap_area = max(0, overlap_x) * max(0, overlap_y)
        area1 = w1 * h1
        area2 = w2 * h2
        overlap_ratio = overlap_area / float(min(area1, area2))

        if overlap_ratio > overlap_thresh:
            # Merge them
            start_x = min(x1, x2)
            start_y = min(y1, y2)
            end_x = max(x1 + w1, x2 + w2)
            end_y = max(y1 + h1, y2 + h2)
            merged[-1] = (start_x, start_y, end_x - start_x, end_y - start_y)
        else:
            merged.append(current)

    return merged


def extract_license_photo(img_path):
    """
    Detect and extract the license photo region cleanly (single image only).
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,  # more strict, fewer false detections
        minSize=(80, 80)
    )

    # If no face detected, fallback to contour method
    if len(faces) == 0:
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.7 < aspect_ratio < 1.3 and w * h > 1000:
                faces = [(x, y, w, h)]
                break

    # --- Fix 1: Merge overlapping detections
    faces = merge_overlapping_boxes(faces)

    # --- Fix 2: Keep only the largest detected face/photo
    if len(faces) > 1:
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        faces = [faces[0]]

    # --- Crop and save detected photo
    saved_photos = []
    for i, (x, y, w, h) in enumerate(faces):
        photo_crop = img[y:y + h, x:x + w]
        timestamp = int(time.time())
        photo_path = os.path.join(output_dir, f"license_photo_{timestamp}_{i + 1}.png")
        cv2.imwrite(photo_path, photo_crop)
        saved_photos.append(photo_path)

    return saved_photos


# -------------------- GEMINI AI EXTRACTION (PRO) --------------------
def extract_fields_with_gemini(text, image_path=None):
    """
    Use Gemini 2.5 Pro to intelligently extract structured fields.
    """

    model = genai.GenerativeModel("gemini-2.5-pro")

    prompt = f"""
    You are a highly capable document parser.
    Extract the following fields from a driver's license or ID document:
    - First_Name
    - Last_Name
    - DL_Number
    - DOB
    - Issue_Date
    - Expiry_Date
    - Address
    - City
    - State
    - Postal_Code

    Rules:
    - All dates must be in MM/DD/YYYY format if possible.
    - Use null if any field is missing or unclear.
    - Address should only include street and house number, not city/state.
    - Return ONLY valid JSON (no explanations, no text before or after JSON).
    - Extract the date format accurately.
    OCR Extracted Text:
    {text}
    """

    # Include image if available
    parts = [prompt]
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as img:
            parts.append({"mime_type": "image/jpeg", "data": img.read()})

    # Send to Gemini 2.5 Pro
    response = model.generate_content(parts)
    output_text = response.text.strip()

    # Parse clean JSON
    try:
        json_data = json.loads(output_text)
    except:
        match = re.search(r"\{.*\}", output_text, re.S)
        json_data = json.loads(match.group(0)) if match else {"raw_output": output_text}

    return json_data


# -------------------- MAIN FUNCTION --------------------
def extract_document(file_path):
    ext = file_path.lower().split('.')[-1]
    text = ""
    images = []

    if ext == "pdf":
        text = extract_text_from_pdf(file_path)
        images = extract_images_from_pdf(file_path)
    elif ext in ["png", "jpeg", "jpg"]:
        text = extract_text_from_image(file_path)
        saved_img = save_image(file_path)
        images = [saved_img]
        photo_files = extract_license_photo(file_path)
        images.extend(photo_files)
    else:
        raise ValueError("Unsupported file type: " + ext)

    # Use Gemini 2.5 Pro to extract structured data
    print("Extracting structured fields using Gemini 2.5 Pro...")
    gemini_fields = extract_fields_with_gemini(text, file_path)

    result = {
        "fields": gemini_fields,
        "images": images
    }
    return result


# -------------------- RUN --------------------
if __name__ == "__main__":
    extracted_data = extract_document(file_path)

    output_json = os.path.join(output_dir, "extracted_data.json")
    with open(output_json, "w") as f:
        json.dump(extracted_data, f, indent=4)

    print("\n===== Structured Fields (via Gemini 2.5 Pro) =====")
    print(json.dumps(extracted_data["fields"], indent=2))

    print("\n===== Extracted Images =====")
    for img_file in extracted_data["images"]:
        print(img_file)

    print("\nJSON saved at:",output_json)