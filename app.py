import os
import re
import json
import time
import cv2
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import google.generativeai as genai

# -------------------- CONFIG --------------------
POPPLER_PATH = r"C:\Users\HP\Downloads\Release-25.07.0-0.zip\poppler-25.07.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Gemini API key (keep it directly here)
GEMINI_API_KEY = "api key here "
genai.configure(api_key=GEMINI_API_KEY)

OUTPUT_DIR = "extracted_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------- COMMON HELPERS --------------------
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    image_files = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_filename = os.path.join(OUTPUT_DIR, f"page{page_num + 1}_img{img_index + 1}.png")
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


def extract_text_from_image(img_path):
    img = Image.open(img_path)
    return pytesseract.image_to_string(img)


def save_image(img_path):
    img = Image.open(img_path)
    save_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    img.save(save_path)
    return save_path


# -------------------- LICENSE PHOTO EXTRACTION --------------------
def extract_license_photo(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.7 < aspect_ratio < 1.3 and w * h > 1000:
                faces = [(x, y, w, h)]
                break

    saved_photos = []
    for i, (x, y, w, h) in enumerate(faces):
        photo_crop = img[y:y + h, x:x + w]
        timestamp = int(time.time())
        photo_path = os.path.join(OUTPUT_DIR, f"license_photo_{timestamp}_{i + 1}.png")
        cv2.imwrite(photo_path, photo_crop)
        saved_photos.append(photo_path)
    return saved_photos


# -------------------- GEMINI EXTRACTORS --------------------
def extract_registration_fields_with_gemini(text, image_path=None):
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
    - Return all dates in MM/DD/YYYY format.
    - If missing, return null.
    - Return ONLY valid JSON.
    OCR Text: {text}
    """

    parts = [prompt]
    if image_path and image_path.lower().endswith((".jpg", ".jpeg", ".png")):
        with open(image_path, "rb") as img:
            parts.append({"mime_type": "image/jpeg", "data": img.read()})
    elif image_path and image_path.lower().endswith(".pdf"):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_images = convert_from_path(image_path, dpi=300, poppler_path=POPPLER_PATH)
                if pdf_images:
                    temp_image_path = os.path.join(tmpdir, "first_page.jpg")
                    pdf_images[0].save(temp_image_path, "JPEG")
                    with open(temp_image_path, "rb") as img:
                        parts.append({"mime_type": "image/jpeg", "data": img.read()})
        except Exception as e:
            st.warning(f"PDF to image conversion failed: {e}")

    response = model.generate_content(parts)
    output_text = response.text.strip()

    try:
        return json.loads(output_text)
    except Exception:
        match = re.search(r"\{.*\}", output_text, re.S)
        return json.loads(match.group(0)) if match else {"raw_output": output_text}


def extract_license_fields_with_gemini(text, image_path=None):
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    prompt = f"""
    You are a smart document parser.
    Extract the following fields from a driver's license:
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
    - Dates must be in MM/DD/YYYY format.
    - Return null if any field missing.
    - Return only valid JSON.
    OCR Text: {text}
    """

    parts = [prompt]
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as img:
            parts.append({"mime_type": "image/jpeg", "data": img.read()})

    response = model.generate_content(parts)
    output_text = response.text.strip()

    try:
        return json.loads(output_text)
    except Exception:
        match = re.search(r"\{.*\}", output_text, re.S)
        return json.loads(match.group(0)) if match else {"raw_output": output_text}


# -------------------- MAIN EXTRACT FUNCTION --------------------
def extract_document(file_path, doc_type):
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
    else:
        raise ValueError("Unsupported file type: " + ext)

    if doc_type == "Registration Card":
        gemini_fields = extract_registration_fields_with_gemini(text, file_path)
    else:
        if ext in ["png", "jpeg", "jpg"]:
            photo_files = extract_license_photo(file_path)
            images.extend(photo_files)
        gemini_fields = extract_license_fields_with_gemini(text, file_path)

    return {"fields": gemini_fields, "images": images}


# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="AI-Based Document Extractor", layout="centered")

# Custom dark theme CSS
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #0d1117;
            color: #e6edf3;
        }
        [data-testid="stHeader"] {
            background: none;
        }
        h1, h2, h3, h4 {
            color: #e6edf3;
        }
        div.stSelectbox, div.stFileUploader {
            background-color: #161b22 !important;
            border-radius: 10px;
            padding: 12px;
        }
        .stDownloadButton button {
            background-color: #238636 !important;
            color: white !important;
            border-radius: 8px;
        }
        .stDownloadButton button:hover {
            background-color: #2ea043 !important;
        }
        .stButton button {
            background-color: #1f6feb !important;
            color: white !important;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #388bfd !important;
        }
        .stAlert {
            background-color: #161b22 !important;
            border-left: 4px solid #2ea043 !important;
        }
    </style>
""", unsafe_allow_html=True)

# App layout
st.title("🧠 AI-Based Document Extractor")
st.caption("Upload a document to automatically extract key information using AI and OCR.")

doc_type = st.selectbox("📄 Select Document Type:", ["Registration Card", "Driving License"])
uploaded_file = st.file_uploader("📂 Upload Image or PDF File", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    with st.spinner(f"🔍 Extracting data from {doc_type}... Please wait."):
        extracted_data = extract_document(temp_path, doc_type)
        time.sleep(1.5)

    st.success("✅ Extraction Completed!")

    st.subheader("🧾 Structured Output (JSON)")
    st.json(extracted_data["fields"])

    st.subheader("📷 Extracted Images")
    for img_file in extracted_data["images"]:
        st.image(img_file, use_container_width=True)

    output_json = os.path.join(OUTPUT_DIR, f"{doc_type.lower().replace(' ', '_')}_data.json")
    with open(output_json, "w") as f:
        json.dump(extracted_data, f, indent=4)

    st.download_button(
        label="💾 Download Extracted Data (JSON)",
        data=json.dumps(extracted_data, indent=4),
        file_name=f"{doc_type.lower().replace(' ', '_')}_data.json",
        mime="application/json"
    )
