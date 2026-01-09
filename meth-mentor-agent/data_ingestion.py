
import logging
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


os.environ['ANONYMIZED_TELEMETRY'] = 'False'
# Configure basic logging to output to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pdf_file = 'jee-math-notes.pdf'


def load_and_chunk_pdf(pdf_file: str) -> List[Document]:
    """
    Loads a PDF and processes it into chunks with detailed logging for traceability.
    """
    
    current_dir = Path(__file__).parent
    pdf_path = current_dir / "data" / pdf_file
    pdf_path_str = str(pdf_path.resolve())
    print(pdf_path_str)
    # STAGE 1: Document Loading
    logger.info(f"Starting ingestion for: {pdf_file}") # Traceability log [cite: 112]
    
    try:
        pdf_loader = PyMuPDF4LLMLoader(pdf_path_str)
        docs = pdf_loader.load()
        logger.info(f"Successfully loaded {len(docs)} pages from PDF.") # Observability log [cite: 112]
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise

    # STAGE 2: Semantic Header Splitting
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
        ("####", "Header_4"),
        ("#####", "Header_5"),
        ("######", "Header_6"),
    ]

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False 
    )

    semantic_docs = []
    logger.info("Splitting document by semantic headers...") # Decision log [cite: 113]
    
    for doc in docs:
        header_splits = header_splitter.split_text(doc.page_content)
        for split in header_splits:
            split.metadata.update(doc.metadata)
            for key, value in split.metadata.items():
                if key.startswith("Header_") and isinstance(value, str):
                # Remove asterisks and clean up whitespace
                    split.metadata[key] = value.replace("**", "").strip().lower()
            semantic_docs.append(split)
    
    logger.info(f"Identified {len(semantic_docs)} structural sections (Headers).")

    # STAGE 3: Recursive Character Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""] 
    )

    final_chunks = text_splitter.split_documents(semantic_docs)
    
    # Final Summary Log for Ingestion Traceability [cite: 112, 114]
    logger.info(f"Ingestion complete: {len(final_chunks)} chunks generated.")
    
    return final_chunks

def create_vector_store(pdf_file_name: str):
    """
    Requirement 1.2: Creates vector embeddings and a persistent vector database.
    
    This function:
    1. Initializes a sentence-transformer embedding model.
    2. Calls the chunking pipeline to get processed document segments.
    3. Indexes the segments into ChromaDB with persistent storage.
    
    Args:
        pdf_file_name (str): The name of the source PDF to process and index.
    """
    
    # 1. Setup Persistent Storage Paths
    # Using 'vectorstore' at the root level as per project structure requirements
    vectorstore_foldername = "vectorstore"
    current_dir = Path(__file__).parent
    vector_db_path = current_dir / vectorstore_foldername / "chroma_db"
    
    # Ensure the directory exists before attempting to write to it
    vector_db_path.parent.mkdir(parents=True, exist_ok=True)
    vector_db_path_str = str(vector_db_path.resolve())
    print(vector_db_path_str)
    # 2. Initialize Embedding Model
    # Using 'all-MiniLM-L6-v2' for a balance of speed and performance [cite: 42]
    logger.info("Initializing HuggingFace embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Retrieve Chunks from Ingestion Pipeline
    # This fulfills the 'modular' software engineering requirement [cite: 7, 21]
    logger.info(f"Requesting chunks for document: {pdf_file_name}")
    chunks = load_and_chunk_pdf(pdf_file=pdf_file_name)
    
    if not chunks:
        logger.error("No chunks received from the loading function. Aborting indexing.")
        return

    # 4. Create and Persist Vector Database
    # Storing vectors in Chroma to support similarity search and filtering [cite: 43, 45]
    logger.info(f"Indexing {len(chunks)} chunks into ChromaDB at {vector_db_path_str}...")
    
    try:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=vector_db_path_str
        )
        
        
        logger.info("Vector database successfully created and persisted.")
        
    except Exception as e:
        logger.error(f"Failed to create vector store: {str(e)}")
        raise

   



   

