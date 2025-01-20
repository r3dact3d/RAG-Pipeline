from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import openai
import os
import numpy as np
import logging
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingComparator:
    def __init__(self):
        """Initialize the embedding comparator with OpenAI embeddings."""
        self._setup_environment()
        self.embedding_function = OpenAIEmbeddings()
        self.evaluator = load_evaluator("pairwise_embedding_distance")

    def _setup_environment(self) -> None:
        """Set up environment variables and API key."""
        load_dotenv()
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        openai.api_key = api_key

    def get_embedding(self, word: str) -> Optional[List[float]]:
        """Get embedding vector for a single word."""
        try:
            vector = self.embedding_function.embed_query(word)
            logging.info(f"Generated embedding for '{word}' with dimension {len(vector)}")
            return vector
        except Exception as e:
            logging.error(f"Error generating embedding for '{word}': {e}")
            return None

    def compare_words(self, word1: str, word2: str) -> Optional[dict]:
        """Compare two words and return detailed comparison metrics."""
        try:
            # Get embeddings
            vector1 = self.get_embedding(word1)
            vector2 = self.get_embedding(word2)
            
            if not vector1 or not vector2:
                return None

            # Calculate various similarity metrics
            cosine_sim = cosine_similarity([vector1], [vector2])[0][0]
            
            # Get evaluator results
            eval_results = self.evaluator.evaluate_string_pairs(
                prediction=word1,
                prediction_b=word2
            )

            return {
                "words": (word1, word2),
                "cosine_similarity": cosine_sim,
                "evaluator_score": eval_results["score"],
                "evaluator_metadata": eval_results.get("metadata", {}),
                "vector_dimensions": len(vector1)
            }
        except Exception as e:
            logging.error(f"Error comparing words '{word1}' and '{word2}': {e}")
            return None

    def compare_multiple_words(self, words: List[str]) -> Optional[dict]:
        """Compare multiple words and their relationships."""
        try:
            results = {}
            for i, word1 in enumerate(words):
                for word2 in words[i+1:]:
                    comparison = self.compare_words(word1, word2)
                    if comparison:
                        results[f"{word1}-{word2}"] = comparison
            return results
        except Exception as e:
            logging.error(f"Error in multiple word comparison: {e}")
            return None

def plot_similarity_matrix(words: List[str], comparisons: dict) -> None:
    """Plot a similarity matrix for the compared words."""
    n = len(words)
    similarity_matrix = np.zeros((n, n))
    
    # Fill the similarity matrix
    for i, word1 in enumerate(words):
        similarity_matrix[i, i] = 1.0  # Self-similarity is 1
        for j, word2 in enumerate(words[i+1:], i+1):
            key = f"{word1}-{word2}"
            if key in comparisons:
                sim = comparisons[key]["cosine_similarity"]
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Matrix is symmetric

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    
    # Add labels
    plt.xticks(range(n), words, rotation=45)
    plt.yticks(range(n), words)
    
    # Add values in cells
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                    ha='center', va='center')
    
    plt.title('Word Similarity Matrix')
    plt.tight_layout()
    plt.show()

def main():
    try:
        comparator = EmbeddingComparator()
        
        # Example words to compare
        words = ["apple", "iphone", "computer", "fruit", "technology"]
        logging.info(f"Comparing words: {words}")
        
        # Perform comparisons
        comparisons = comparator.compare_multiple_words(words)
        
        if comparisons:
            # Print detailed results
            print("\nDetailed Comparison Results:")
            print("-" * 50)
            for pair, results in comparisons.items():
                print(f"\nComparing: {pair}")
                print(f"Cosine Similarity: {results['cosine_similarity']:.4f}")
                print(f"Evaluator Score: {results['evaluator_score']:.4f}")
                print(f"Vector Dimensions: {results['vector_dimensions']}")
            
            # Plot similarity matrix
            plot_similarity_matrix(words, comparisons)
        
    except Exception as e:
        logging.error(f"An error occurred in main: {e}")
        raise

if __name__ == "__main__":
    main()
