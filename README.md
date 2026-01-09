=========================================================
🎓 JEE MATH MENTOR: MULTIMODAL AI AGENT
=========================================================

Project Overview:
-----------------
The JEE Math Mentor is an advanced, multi-agent AI system 
built to provide pedagogical solutions to JEE-level math 
problems. It leverages LangGraph for orchestration, 
Groq (Llama 3) for reasoning, and ChromaDB for RAG 
(Retrieval-Augmented Generation).

Features:
---------
* ⌨️ Text: Direct input of mathematical queries.
* 🖼️ Image: OCR extraction of math from photos (Pix2Text).
* 🎤 Audio: Voice-to-text transcription (Faster-Whisper).
* 🧠 RAG: Uses local PDF notes for accurate JEE theory.
* 🛡️ Verifier Loop: Multi-step reasoning with human-in-the-loop.

=========================================================
📂 FOLDER STRUCTURE
=========================================================

math-mentor-app/
│
├── data/                   # Input PDFs for training/notes
│   └── jee-math-notes.pdf
│
├── src/                    # Source Code
│   ├── agents/             # Logic for Parser, Solver, Verifier
│   └── image_audio_input.py # Multimodal processing
│
├── vectorstore/            # Persistent ChromaDB storage
│   └── chromadb/
│
├── app.py                  # Main UI (Streamlit)
├── data_ingestion.py       # Script to populate VectorDB
├── .env                    # Secrets (API Keys)
└── requirements.txt        # Python Dependencies

=========================================================
🛠️ INSTALLATION & SETUP
=========================================================

1. Initialize Virtual Environment (Recommended):
   --------------------------------------------
   python -m venv venv
   
   # Windows:
   venv\Scripts\activate
   
   # Mac/Linux:
   source venv/bin/activate

2. Install Dependencies:
   ---------------------
   pip install --upgrade pip
   pip install -r requirements.txt

3. System Dependencies:
   --------------------
   Ensure the following are installed on your OS:
   - FFmpeg (for Audio processing)
   - Tesseract-OCR (for Image processing)

4. Environment Variables:
   ----------------------
   Create a '.env' file in the root directory:
   GROQ_API_KEY=your_api_key_here

=========================================================
🚀 EXECUTION GUIDE
=========================================================

Step 1: Ingest Data (RAG Setup)
-------------------------------
Run this once to convert your PDFs into searchable memory:
> python data_ingestion.py

Step 2: Run the Application
---------------------------
Start the Streamlit interface:
> streamlit run app.py

Step 3: Accessing the App
-------------------------
Open your browser and navigate to the local URL provided 
(usually http://localhost:8501).

=========================================================
🧭 NAVIGATION IN APP
=========================================================
- Sidebar: Choose "Image" or "Audio" to upload files.
- Chat Box: Enter or edit extracted text.
- Status Box: Watch the agents (Parser -> Solver -> Verifier).
- Memory Toast: Notifications when a problem is saved 
  to ChromaDB for future learning.

=========================================================
🛡️ TROUBLESHOOTING & REPAIRING DATABASE
=========================================================
⚠️ IMPORTANT: IF VECTOR DATABASE IS NOT WORKING
If the agent cannot find theorems or the RAG system fails, 
your local database might be missing or corrupted. 

To fix this, follow these navigation steps:
1. Open your terminal/command prompt.
2. Navigate to your project root folder (math-mentor-app).
3. Run the ingestion script to recreate the database:
   > python data_ingestion.py

This will process the PDF notes in the /data folder and 
automatically create the /vectorstore/chromadb directory.

- Chroma Lock: If "Chroma instance exists," restart app.py.
- Model Download: First run of Pix2Text or Whisper will 
  download model weights (~1GB-2GB).
- API Errors: Check your GROQ_API_KEY in the .env file.

=========================================================
Generated on: 2026-01-09
=========================================================
