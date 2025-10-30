import streamlit as st
import os
import io
import json
import base64
import tempfile
from pathlib import Path
from typing import List, Optional
from PIL import Image
import pytesseract
import cv2
import numpy as np
import re
from datetime import datetime

# Try importing Gemini client
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# -------------------------
# Helper: load pages from uploaded file
# -------------------------
def load_images_from_upload(uploaded_file) -> List[Image.Image]:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        img = Image.open(uploaded_file).convert("RGB")
        return [img]
    elif suffix == ".pdf":
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(uploaded_file.read(), dpi=300)
        return [page.convert("RGB") for page in pages]
    else:
        st.error("Unsupported file type.")
        return []

# -------------------------
# OCR with pytesseract
# -------------------------
def ocr_image(pil_img: Image.Image) -> str:
    gray = pil_img.convert("L")
    return pytesseract.image_to_string(gray, lang='eng')

# -------------------------
# Detect a face/photo region and crop it
# -------------------------
def detect_and_crop_photo(pil_img: Image.Image) -> Optional[str]:
    np_img = np.array(pil_img)
    img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        st.warning("Haar cascade not found.")
        return None

    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=3, minSize=(40,40))

    if len(faces) == 0:
        h, w = gray.shape
        heuristic_crop = img_bgr[int(0.05*h):int(0.6*h), int(0.02*w):int(0.4*w)]
        if heuristic_crop.size == 0:
            return None
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        cv2.imwrite(temp_path, heuristic_crop)
        return temp_path

    best = max(faces, key=lambda r: r[2]*r[3])
    x, y, wf, hf = best
    padx, pady = int(wf*0.15), int(hf*0.15)
    crop = img_bgr[y-pady:y+hf+pady, x-padx:x+wf+padx]
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    cv2.imwrite(temp_path, crop)
    return temp_path

# -------------------------
# Local regex DL parser (robust)
# -------------------------
def local_parse_dl_text(ocr_text: str) -> dict:
    out = {}
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    ocr_up = ocr_text.upper()
    bad_tokens = {"FNIMA", "LNCARDHOLDER", "CARDHOLDER", "LICENSE", "END", "EXP"}

    # --------------------------
    # License number
    # --------------------------
    m = re.search(r'\b[A-Z0-9]{1,2}\d{5,9}\b', ocr_text)
    if m:
        lic = m.group(0).upper()
        if lic[0] in ['0', '1', 'I']:
            lic = 'L' + lic[1:]
        out['license_number'] = lic

    # --------------------------
    # Dates
    # --------------------------
    current_year = datetime.now().year
    date_matches = re.findall(r'\b(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12][0-9]|3[01])[\/\-](\d{2,4})\b', ocr_text)
    parsed_dates = []
    for m in date_matches:
        month, day, year_raw = int(m[0]), int(m[1]), m[2]
        year = int(year_raw)
        if year < 100:
            year += 1900 if year > 25 else 2000
        if year > current_year + 5:
            continue
        parsed_dates.append(f"{month:02}/{day:02}/{year:04}")
    if parsed_dates:
        out['dates_found'] = parsed_dates
        years = [int(d.split('/')[-1]) for d in parsed_dates]
        dob_idx = years.index(min(years))
        exp_idx = years.index(max(years))
        out['dob'] = parsed_dates[dob_idx]
        out['expiration_date'] = parsed_dates[exp_idx]
        remaining = [d for i, d in enumerate(parsed_dates) if i not in (dob_idx, exp_idx)]
        if remaining:
            out['issue_date'] = remaining[0]

    # --------------------------
    # Name extraction
    # --------------------------
    fn, ln_name = None, None
    # Look for FN / LN patterns
    for ln in lines[:15]:
        line_up = ln.upper()
        fn_match = re.search(r'\bFN[:\s\-]*([A-Z]+)\b', line_up)
        ln_match = re.search(r'\bLN[:\s\-]*([A-Z]+)\b', line_up)
        if fn_match:
            candidate_fn = fn_match.group(1).title()
            if candidate_fn.upper() not in bad_tokens:
                fn = candidate_fn
        if ln_match:
            candidate_ln = ln_match.group(1).title()
            if candidate_ln.upper() not in bad_tokens:
                ln_name = candidate_ln
        if fn or ln_name:
            break

    # Fallback: first line with 2+ uppercase tokens
    if not fn or not ln_name:
        for ln in lines[:12]:
            tokens = re.findall(r"[A-Z]{2,}", ln.upper())
            if len(tokens) >= 2:
                first, last = tokens[0].title(), tokens[1].title()
                if first not in bad_tokens:
                    fn = first
                if last not in bad_tokens:
                    ln_name = last
                if fn or ln_name:
                    break

    if fn:
        out['first_name'] = fn
    if ln_name:
        out['last_name'] = ln_name

    # --------------------------
    # Address
    # --------------------------
    for ln in lines:
        if re.search(r'\d{1,5}\s+[A-Za-z]', ln):
            out['address'] = ln.title()
            break

    # City/State/Postal
    for ln in lines:
        m = re.search(r'([A-Z][a-zA-Z]+)[, ]+\s*([A-Z]{2})\s*(\d{5})', ln)
        if m:
            out['city'] = m.group(1).title()
            out['state'] = m.group(2)
            out['postal_code'] = m.group(3)
            break

    # --------------------------
    # Other fields
    # --------------------------
    for ln in lines:
        ln_upper = ln.upper()
        # Class
        cls = re.search(r'CLASS[:\s]*([A-Z0-9])', ln_upper)
        if cls:
            out['class'] = cls.group(1)
        # Sex
        sex = re.search(r'SEX[:\s]*([FM])', ln_upper)
        if sex:
            out['sex'] = sex.group(1)
        # Height
        ht = re.search(r"(\d{1,2}'-\d{2}\")", ln)
        if ht:
            out['height'] = ht.group(1)
        # Weight
        wt = re.search(r"(\d{2,3}\s?LB)", ln_upper)
        if wt:
            out['weight'] = wt.group(1)
        # Hair
        hr = re.search(r'HAIR[:\s\-]*([A-Z]+)', ln_upper)
        if hr:
            out['hair'] = hr.group(1)
        # Eyes
        ey = re.search(r'EYES[:\s\-]*([A-Z]+)', ln_upper)
        if ey:
            out['eyes'] = ey.group(1)

    return out

# -------------------------
# Gemini extraction
# -------------------------
def call_gemini_extract(ocr_text: str, image_b64: Optional[str] = None) -> dict:
    if not GENAI_AVAILABLE:
        st.warning("Gemini client not available, using local parse only.")
        return {}

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    prompt = f"""
Extract structured data from this driver's license OCR text. Return JSON only.

Fields:
license_number, first_name, last_name, dob, expiration_date, issue_date,
address, city, state, postal_code, class, sex, height, weight, hair, eyes.

OCR text:
\"\"\"{ocr_text}\"\"\"
"""
    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
    text = getattr(response, "text", str(response))

    try:
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
    except Exception as e:
        st.warning(f"Gemini parse error: {e}")
    return {}

# -------------------------
# Streamlit App
# -------------------------
st.title("🪪 Driver’s License Data Extractor")

uploaded = st.file_uploader("Upload a Driver’s License (Image or PDF)", type=["png", "jpg", "jpeg", "pdf"])
use_gemini = st.toggle("Use Gemini AI Extraction", value=True)

if uploaded:
    with st.spinner("Processing..."):
        images = load_images_from_upload(uploaded)
        if not images:
            st.stop()
        page0 = images[0]
        st.image(page0, caption="Uploaded Document", use_container_width=True)

        ocr_text = ocr_image(page0)
        st.subheader("🧾 OCR Text")
        st.text_area("Extracted OCR Text", ocr_text, height=250)

        photo_path = detect_and_crop_photo(page0)
        if photo_path:
            st.image(photo_path, caption="Detected Photo", width=200)

        parsed_local = local_parse_dl_text(ocr_text)
        final_data = {}

        if use_gemini:
            try:
                img_b64 = None
                if photo_path and os.path.exists(photo_path):
                    with open(photo_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode("ascii")[:6000]
                final_data = call_gemini_extract(ocr_text, img_b64)
            except Exception as e:
                st.warning(f"Gemini failed: {e}")
                final_data = parsed_local
        else:
            final_data = parsed_local

        # Fill missing fields from local parse
        for k, v in parsed_local.items():
            if k not in final_data or final_data.get(k) is None:
                final_data[k] = v

        st.subheader("📊 Extracted Data")
        st.json(final_data)

        with open("extracted_data.json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2)
        st.success("✅ Extraction Complete — saved to extracted_data.json")
