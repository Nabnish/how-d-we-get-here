import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import time

class PubMedAPI:
    """
    A class to interact with the NCBI PubMed API (E-utilities).
    No API key required - it's free and open.
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, email: Optional[str] = None, tool: str = "PubMedAPI"):
        """
        Initialize the PubMed API client.
        
        Args:
            email: Your email (optional but recommended by NCBI)
            tool: Name of your tool/application
        """
        self.email = email
        self.tool = tool
        self.params = {
            "tool": tool,
            "email": email if email else "example@example.com"
        }
    
    def search(self, query: str, max_results: int = 10, retstart: int = 0) -> List[str]:
        """
        Search PubMed for articles matching the query.
        
        Args:
            query: Search query (e.g., "diabetes treatment", "cancer[Title]")
            max_results: Maximum number of results to return
            retstart: Starting position for results (for pagination)
        
        Returns:
            List of PubMed IDs (PMIDs)
        """
        search_url = f"{self.BASE_URL}/esearch.fcgi"
        
        params = {
            **self.params,
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retstart": retstart,
            "retmode": "json"
        }
        
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            pmids = data.get("esearchresult", {}).get("idlist", [])
            return pmids
        except requests.exceptions.RequestException as e:
            print(f"Error searching PubMed: {e}")
            return []
    
    def fetch_details(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch detailed information for given PubMed IDs.
        
        Args:
            pmids: List of PubMed IDs
        
        Returns:
            List of dictionaries containing article details
        """
        if not pmids:
            return []
        
        fetch_url = f"{self.BASE_URL}/efetch.fcgi"
        
        params = {
            **self.params,
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        
        try:
            response = requests.get(fetch_url, params=params)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            articles = []
            
            for article in root.findall(".//PubmedArticle"):
                article_data = self._parse_article(article)
                if article_data:
                    articles.append(article_data)
            
            return articles
        except requests.exceptions.RequestException as e:
            print(f"Error fetching details: {e}")
            return []
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            return []
    
    def _parse_article(self, article: ET.Element) -> Optional[Dict]:
        """
        Parse a single article from XML.
        
        Args:
            article: XML element representing a PubMed article
        
        Returns:
            Dictionary with article details
        """
        try:
            # Extract PMID
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "N/A"
            
            # Extract titlell
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "N/A"
            
            # Extract abstract
            abstract_parts = []
            for abstract_text in article.findall(".//AbstractText"):
                label = abstract_text.get("Label", "")
                text = abstract_text.text if abstract_text.text else ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available"
            
            # Extract authors
            authors = []
            for author in article.findall(".//Author"):
                last_name = author.find("LastName")
                first_name = author.find("ForeName")
                initials = author.find("Initials")
                
                if last_name is not None:
                    name_parts = [last_name.text]
                    if first_name is not None:
                        name_parts.append(first_name.text)
                    elif initials is not None:
                        name_parts.append(initials.text)
                    authors.append(" ".join(name_parts))
            
            # Extract journal
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "N/A"
            
            # Extract publication date
            pub_date_elem = article.find(".//PubDate")
            pub_date = "N/A"
            if pub_date_elem is not None:
                year_elem = pub_date_elem.find("Year")
                month_elem = pub_date_elem.find("Month")
                if year_elem is not None:
                    pub_date = year_elem.text
                    if month_elem is not None:
                        pub_date = f"{month_elem.text} {pub_date}"
            
            # Extract DOI
            doi_elem = article.find(".//ELocationID[@EIdType='doi']")
            doi = doi_elem.text if doi_elem is not None else "N/A"
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "journal": journal,
                "publication_date": pub_date,
                "doi": doi,
                "full_text": f"{title}\n\nAbstract:\n{abstract}"
            }
        except Exception as e:
            print(f"Error parsing article: {e}")
            return None
    
    def search_and_fetch(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search PubMed and fetch full details in one call.
        
        Args:
            query: Search query
            max_results: Maximum number of results
        
        Returns:
            List of article dictionaries
        """
        pmids = self.search(query, max_results=max_results)
        if not pmids:
            return []
        
        # Be polite - add a small delay to avoid rate limiting
        time.sleep(0.34)  # NCBI recommends max 3 requests per second
        
        return self.fetch_details(pmids)
    
    def get_full_text_link(self, pmid: str) -> Optional[str]:
        """
        Get the full-text link for a PubMed article (if available).
        
        Args:
            pmid: PubMed ID
        
        Returns:
            URL to full-text article or None
        """
        fetch_url = f"{self.BASE_URL}/efetch.fcgi"
        
        params = {
            **self.params,
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        }
        
        try:
            response = requests.get(fetch_url, params=params)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            # Look for full-text links
            for link in root.findall(".//ELocationID[@EIdType='pii']"):
                return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        except Exception as e:
            print(f"Error getting full-text link: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # ============================================
    # CHANGE THIS: Replace with your email address
    # ============================================
    pubmed = PubMedAPI(email="your_email@example.com")  # ⬅️ CHANGE THIS LINE
    
    # ============================================
    # CHANGE THIS: Modify the search query to what you want
    # ============================================
    search_query = "diabetes treatment"  # ⬅️ CHANGE THIS to your search term
    max_results = 5  # ⬅️ CHANGE THIS to how many results you want
    
    print(f"Searching for articles about '{search_query}'...")
    articles = pubmed.search_and_fetch(search_query, max_results=max_results)
    
    print(f"\nFound {len(articles)} articles:\n")
    
    for i, article in enumerate(articles, 1):
        print(f"{i}. {article['title']}")
        print(f"   Authors: {', '.join(article['authors'][:3])}...")
        print(f"   Journal: {article['journal']}")
        print(f"   Year: {article['publication_date']}")
        print(f"   PMID: {article['pmid']}")
        print(f"   Abstract preview: {article['abstract'][:200]}...")
        print()
    
    # Example 2: Search with specific filters (optional - you can delete this section)
    # Uncomment and modify the lines below if you want a second search:
    # print("\n" + "="*50)
    # print("Searching for recent articles...")
    # recent_articles = pubmed.search_and_fetch("your_query[Title] AND 2023[PDAT]", max_results=3)
    # 
    # for article in recent_articles:
    #     print(f"\nTitle: {article['title']}")
    #     print(f"Link: https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/")

