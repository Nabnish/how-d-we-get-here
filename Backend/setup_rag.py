"""
Setup RAG Pipeline: Fetch PubMed articles → Create FAISS index → Ready for queries
"""

import sys
import os

# Add Backend to path
sys.path.insert(0, os.path.dirname(__file__))

from pdfimport import PubMedAPI
from vectorConvert import build_index_for_query, answer_with_pubmed

def setup_rag_index(search_query: str, max_results: int = 10, email: str = None):
    """
    Main function to set up RAG:
    1. Fetch articles from PubMed
    2. Create embeddings and FAISS index
    3. Save index files (pubmed_faiss.index, pubmed_meta.pkl)
    
    Args:
        search_query: What to search for (e.g., "diabetes treatment")
        max_results: How many articles to fetch
        email: Your email (required by NCBI for polite API use)
    """
    
    print(f"🔍 Setting up RAG index for: '{search_query}'")
    print(f"📊 Fetching up to {max_results} articles from PubMed...\n")
    
    try:
        result = build_index_for_query(
            query=search_query,
            max_results=max_results,
            email=email or "your_email@example.com"
        )
        
        print("✅ RAG Index created successfully!")
        print(f"   - Indexed chunks: {result['indexed_chunks']}")
        print(f"   - Articles fetched: {result['articles_indexed']}")
        print(f"   - Index file: {result['index_file']}")
        print(f"   - Meta file: {result['meta_file']}\n")
        
        return result
        
    except Exception as e:
        print(f"❌ Error creating RAG index: {e}")
        raise

def query_rag(question: str) -> str:
    """
    Query the RAG system with a question.
    
    Args:
        question: Your question about the indexed documents
    
    Returns:
        Answer from the LLM based on retrieved context
    """
    
    print(f"\n❓ Question: {question}")
    print("⏳ Retrieving context and generating answer...\n")
    
    try:
        answer = answer_with_pubmed(question)
        print(f"💡 Answer:\n{answer}\n")
        return answer
    except Exception as e:
        print(f"❌ Error querying RAG: {e}")
        raise

if __name__ == "__main__":
    # ========================================
    # STEP 1: Create RAG Index
    # ========================================
    setup_rag_index(
        search_query="diabetes treatment",      # ⬅️ CHANGE THIS
        max_results=5,                           # ⬅️ CHANGE THIS
        email="hrithikmadhu2008@gmail.com"      # ⬅️ CHANGE THIS to your email
    )
    
    # ========================================
    # STEP 2: Query the RAG System
    # ========================================
    query_rag("What are the main treatment options?")
    query_rag("What are the side effects?")
    query_rag("Is this suitable for Type 2 diabetes?")
