"""
Test building index from a sample uploaded document and querying it.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from vectorConvert import build_index_from_text, answer_with_pubmed, extractive_answer

SAMPLE_DOC = {
    "text": (
        "Diabetes Management:\n\n"
        "Metformin is commonly used as first-line therapy for type 2 diabetes. "
        "It reduces hepatic glucose production and improves peripheral insulin sensitivity. "
        "Side effects include GI upset and potential B12 deficiency.\n\n"
        "SGLT2 inhibitors reduce cardiovascular events and provide renal protection."
    ),
    "title": "Sample Diabetes Doc",
    "source": "test"
}

print("Building index from sample document...")
res = build_index_from_text([SAMPLE_DOC])
print("Indexing result:", res)

questions = [
    "What are the main treatments mentioned?",
    "What are metformin side effects?",
]

for q in questions:
    print("\nQuery:", q)
    try:
        # Use fast extractive fallback for specific queries to avoid slow CPU LLM runs
        if "metformin" in q.lower():
            ans = extractive_answer(q)
        else:
            ans = answer_with_pubmed(q)
        print("Answer:\n", ans)
    except Exception as e:
        print("Error while querying:", e)

print("\nTest completed.")
