🧠 AI-Based Document Extractor

An intelligent document processing system built using Python, Streamlit, and Google Gemini AI, designed to automatically extract structured data from Driver’s Licenses and Vehicle Registration Cards.

🚀 Features

Upload and process images or PDFs via Streamlit interface

Auto-detects document type (Registration Card / Driving License)

Extracts key fields using OCR (Tesseract) and Gemini AI

Detects and crops driver’s photo using OpenCV

Outputs clean, structured JSON data

Downloadable extraction results

🧰 Tech Stack

Python, Streamlit, OpenCV, PyPDF2, PyMuPDF, pdf2image, pytesseract

Google Generative AI (Gemini) for intelligent text interpretation

dotenv for secure API key management

⚙️ Setup Instructions

Clone the repository

git clone https://github.com/yourusername/AI-Document-Extractor.git
cd AI-Document-Extractor


Install dependencies

pip install -r requirements.txt


Create a .env file in the project root and add your Gemini API key:

GEMINI_API_KEY=your_api_key_here


Run the app

streamlit run app.py
