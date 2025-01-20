import os
import argparse
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

CHROMA_PATH = "chroma"
SIMILARITY_THRESHOLD = 0.7

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def setup_database():
    """Initialize the Chroma database with embeddings."""
    try:
        embedding_function = OpenAIEmbeddings()
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    except Exception as e:
        logging.error(f"Failed to setup database: {e}")
        raise

def query_database(db, query_text, k=3):
    """Search the database for relevant documents."""
    try:
        results = db.similarity_search_with_relevance_scores(query_text, k=k)
        if not results or results[0][1] < SIMILARITY_THRESHOLD:
            logging.warning("No relevant results found")
            return None
        return results
    except Exception as e:
        logging.error(f"Error querying database: {e}")
        return None

def generate_response(context_text, query_text):
    """Generate a response using the ChatOpenAI model."""
    try:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        model = ChatOpenAI()
        # Using invoke instead of deprecated predict method
        response = model.invoke(prompt)
        
        return response.content
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        raise

def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    
    try:
        # Setup and query the database
        db = setup_database()
        results = query_database(db, args.query_text)
        
        if not results:
            print("Unable to find matching results.")
            return

        # Generate context from results
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Generate and format response
        response_text = generate_response(context_text, args.query_text)
        sources = [doc.metadata.get("source", "Unknown source") for doc, _score in results]
        
        # Print results
        print("\nResponse:", response_text)
        print("\nSources:", sources)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
