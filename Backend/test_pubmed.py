"""
Quick test to debug PubMed API
"""
from pdfimport import PubMedAPI

print("Testing PubMed API...\n")

pubmed = PubMedAPI(email="test@example.com")

# Test 1: Search for IDs
print("1. Searching for PubMed IDs...")
pmids = pubmed.search(query="diabetes treatment", max_results=5)
print(f"   Found {len(pmids)} PMIDs: {pmids}\n")

if pmids:
    # Test 2: Fetch details
    print("2. Fetching article details...")
    articles = pubmed.fetch_details(pmids)
    print(f"   Got {len(articles)} articles\n")
    
    if articles:
        print("3. Sample article:")
        article = articles[0]
        print(f"   Title: {article['title'][:100]}...")
        print(f"   PMID: {article['pmid']}")
        print(f"   Journal: {article['journal']}")
else:
    print("❌ No PMIDs returned. PubMed API might be down or blocking requests.")
