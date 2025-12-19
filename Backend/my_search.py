from setup_rag import setup_rag_index, query_rag
import sys

if len(sys.argv) < 2:
    print("Usage: python my_search.py 'your search query'")
    sys.exit(1)

search_term = sys.argv[1]

setup_rag_index(
    search_query=search_term,
    max_results=10,
    email="your_email@example.com"
)

# Ask multiple questions
query_rag("What are the main findings?")
query_rag("What are the limitations?")
query_rag("What future research is needed?")