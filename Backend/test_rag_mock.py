"""
Test RAG with mock data (no PubMed API needed)
Useful for testing when PubMed is slow or blocked
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from vectorConvert import build_pubmed_index, answer_with_pubmed

# Mock articles for testing
MOCK_ARTICLES = [
    {
        "pmid": "12345678",
        "title": "Diabetes Management with Metformin",
        "journal": "Journal of Diabetes",
        "full_text": """Diabetes Management with Metformin

Metformin is a first-line medication for type 2 diabetes. It works by reducing glucose production in the liver and improving insulin sensitivity in muscle and fat tissues. Common dosing is 500-2000mg daily in divided doses. Side effects include gastrointestinal upset and vitamin B12 deficiency with long-term use. Metformin is contraindicated in severe kidney disease."""
    },
    {
        "pmid": "87654321",
        "title": "Insulin Therapy in Type 1 Diabetes",
        "journal": "Endocrinology Reviews",
        "full_text": """Insulin Therapy in Type 1 Diabetes

Insulin therapy is essential for type 1 diabetes management. Multiple daily injections (MDI) or insulin pumps are common delivery methods. Rapid-acting insulin is used for meals, while long-acting basal insulin maintains glucose control between meals. Tight glucose control reduces complications but increases hypoglycemia risk. HbA1c target is typically 7% for most patients."""
    },
    {
        "pmid": "11111111",
        "title": "GLP-1 Receptor Agonists for Diabetes",
        "journal": "Nature Medicine",
        "full_text": """GLP-1 Receptor Agonists for Diabetes

GLP-1 receptor agonists are increasingly used for type 2 diabetes. They stimulate insulin secretion in response to glucose and promote weight loss. Common side effects include nausea, vomiting, and gastrointestinal symptoms. Notable agents include liraglutide, dulaglutide, and semaglutide. These drugs also reduce cardiovascular events in high-risk patients."""
    },
    {
        "pmid": "22222222",
        "title": "SGLT2 Inhibitors: Cardiovascular Benefits",
        "journal": "Diabetes Care",
        "full_text": """SGLT2 Inhibitors: Cardiovascular Benefits

SGLT2 inhibitors lower blood glucose by promoting urinary glucose excretion. Beyond glucose control, they provide cardiovascular and renal benefits. SGLT2 inhibitors reduce heart failure hospitalizations and slow diabetic kidney disease progression. Common agents include empagliflozin, canagliflozin, and dapagliflozin. Side effects include genital infections and euglycemic diabetic ketoacidosis."""
    },
    {
        "pmid": "33333333",
        "title": "Diabetes Complications: Prevention and Management",
        "journal": "The Lancet",
        "full_text": """Diabetes Complications: Prevention and Management

Chronic complications of diabetes include neuropathy, nephropathy, retinopathy, and cardiovascular disease. Tight glycemic control reduces microvascular complications. Cardiovascular risk factors like hypertension and dyslipidemia require aggressive management. Screening for complications should be performed annually. Early intervention can prevent or delay progression of complications."""
    }
]

print("🧪 Testing RAG with Mock Data (No PubMed API needed)\n")
print("="*60)

# Step 1: Build index with mock data
print("\n📊 Building RAG index with mock data...")
try:
    count = build_pubmed_index(MOCK_ARTICLES)
    print(f"✅ Index created with {count} chunks\n")
except Exception as e:
    print(f"❌ Error building index: {e}")
    sys.exit(1)

# Step 2: Ask questions
questions = [
    "What are the main treatment options for diabetes?",
    "What are the side effects of metformin?",
    "How do SGLT2 inhibitors help?",
]

print("="*60)
print("\n🤖 Querying RAG System:\n")

for i, question in enumerate(questions, 1):
    print(f"\n❓ Question {i}: {question}")
    print("-" * 60)
    try:
        answer = answer_with_pubmed(question)
        print(f"💡 Answer:\n{answer}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")

print("="*60)
print("✅ RAG test completed!")
