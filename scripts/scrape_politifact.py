import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from database import collection  # MongoDB collection import
from preprocess import preprocess  # Import text preprocessing

# ✅ Define label mapping (1 = Fake, 0 = Real)
label_mapping = {
    "pants-fire": 1,
    "false": 1,
    "barely-true": 1,
    "half-true": 1,  # Let's consider it Fake
    "mostly-true": 0,
    "true": 0
}

def scrape_politifact_covid(min_claims=50, max_pages=50):
    """Scrapes Politifact for COVID-related fact-checks, assigns labels, and stores in MongoDB."""
    covid_keywords = ["COVID", "coronavirus", "pandemic", "vaccine", "mask", "quarantine", "lockdown"]
    base_url = "https://www.politifact.com/factchecks/list/?page="
    claims_data = []
    page = 1  

    while len(claims_data) < min_claims and page <= max_pages:
        url = base_url + str(page)
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to fetch {url}, skipping...")
            page += 1
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        fact_checks = soup.find_all("article", class_="m-statement")

        for fact in fact_checks:
            try:
                claim_text = fact.find("div", class_="m-statement__quote").text.strip()
                verdict = fact.find("div", class_="m-statement__meter").find("img")["alt"]
                source_link = "https://www.politifact.com" + fact.find("a")["href"]
                date = fact.find("footer", class_="m-statement__footer").text.strip()

                # ✅ Filter only COVID-related claims
                if any(keyword.lower() in claim_text.lower() for keyword in covid_keywords):
                    # ✅ Map labels (1 = Fake, 0 = Real)
                    is_fake = label_mapping.get(verdict.lower(), None)

                    # ✅ Preprocess text
                    cleaned_text = preprocess(claim_text)

                    # ✅ Only store if label is recognized
                    if is_fake is not None:
                        doc = {
                            "Claim": claim_text,
                            "Label": verdict,
                            "is_fake": is_fake,  # Binary label
                            "clean_text": cleaned_text,
                            "Source": source_link,
                            "Date": date
                        }
                        claims_data.append(doc)

            except AttributeError:
                continue

        print(f"Scraped {len(claims_data)} COVID claims from {page} pages...")
        page += 1
        time.sleep(1)  

    # ✅ Insert into MongoDB
    if claims_data:
        collection.insert_many(claims_data)
        print(f"✅ Inserted {len(claims_data)} new labeled claims into MongoDB!")

scrape_politifact_covid()
