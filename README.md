#  AI-Based Document Extractor  

An **AI-powered intelligent document automation system** built using **Python**, **Streamlit**, and **Google Gemini AI**, designed to perform **OCR-based text extraction** and **structured field mapping** from documents like **Driving Licenses** and **Vehicle Registration Cards**.  

---

###  **Key Features**  

-  **AI-driven document parsing** using Gemini for structured text interpretation  
-  Upload and process **PDFs or images** via Streamlit’s file uploader  
-  **OCR-based data extraction** using Tesseract and PyMuPDF for enhanced accuracy  
-  **Image preprocessing** and **photo region detection** using OpenCV  
-  Generates clean, structured **JSON output** for seamless data integration  
-  Download extracted JSON and images for downstream automation workflows  

---

###  **Tech Stack**  

- **Languages & Frameworks:** Python, Streamlit  
- **Libraries:** OpenCV, PyMuPDF, pdf2image, pytesseract  
- **AI Integration:** Google Gemini API for semantic field extraction  
- **Utilities:** dotenv for secure key handling, pandas for structured data formatting  

---
###  **Project Highlights** 

1. Built an AI-ML integrated document processing pipeline combining OCR and LLMs

2. Enhanced data accuracy using hybrid vision-text AI techniques

3. Developed a clean and modern Streamlit-based UI for interactive document uploads

4. Implemented secure API key management using dotenv for data privacy

---

###  **Setup Instructions**  

```bash
# Clone the repository  
git clone https://github.com/yourusername/AI-Document-Extractor.git  
cd AI-Document-Extractor  

# Install dependencies  
pip install -r requirements.txt  

# Create .env file with your API key  
GEMINI_API_KEY=your_api_key_here  

# Run the Streamlit app  
streamlit run app.py 

```





