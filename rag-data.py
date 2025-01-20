from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
import nltk
import logging
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
class Config:
    CHROMA_PATH = Path("chroma")
    DATA_PATH = Path("/home/brthomps/working/foamy-stuff")
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 100
    ALLOWED_EXTENSIONS = [".md"]  # Add more extensions as needed

def setup_environment() -> bool:
    """Initialize environment and dependencies."""
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            logging.error("OpenAI API key not found in environment variables")
            return False

        openai.api_key = api_key

        # Download NLTK data
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logging.error(f"Failed to download NLTK data: {e}")
            return False

        return True

    except Exception as e:
        logging.error(f"Environment setup failed: {e}")
        return False

def load_documents() -> Optional[List[Document]]:
    """Load documents from the data directory."""
    try:
        if not Config.DATA_PATH.exists():
            logging.error(f"Data directory not found: {Config.DATA_PATH}")
            return None

        loader = DirectoryLoader(
            str(Config.DATA_PATH),
            glob="*.[mM][dD]",  # Case-insensitive .md extension
            recursive=True 
        )
        documents = loader.load()
        
        if not documents:
            logging.warning("No documents found in data directory")
            return None

        logging.info(f"Successfully loaded {len(documents)} documents")
        return documents

    except Exception as e:
        logging.error(f"Error loading documents: {e}")
        return None

def split_text(documents: List[Document]) -> Optional[List[Document]]:
    """Split documents into chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logging.error(f"Error splitting documents: {e}")
        return None

def save_to_chroma(chunks: List[Document]) -> bool:
    """Save document chunks to Chroma database."""
    try:
        # Clear out the database first
        if Config.CHROMA_PATH.exists():
            shutil.rmtree(Config.CHROMA_PATH)

        # Create a new DB from the documents
        db = Chroma.from_documents(
            chunks, 
            OpenAIEmbeddings(), 
            persist_directory=str(Config.CHROMA_PATH)
        )
        db.persist()
        logging.info(f"Saved {len(chunks)} chunks to {Config.CHROMA_PATH}")
        return True

    except Exception as e:
        logging.error(f"Error saving to Chroma: {e}")
        return False

def generate_data_store() -> bool:
    """Main function to generate the data store."""
    documents = load_documents()
    if not documents:
        return False

    chunks = split_text(documents)
    if not chunks:
        return False

    return save_to_chroma(chunks)

def main():
    if not setup_environment():
        logging.error("Failed to setup environment. Exiting.")
        return

    if generate_data_store():
        logging.info("Successfully generated data store")
    else:
        logging.error("Failed to generate data store")

if __name__ == "__main__":
    main()
