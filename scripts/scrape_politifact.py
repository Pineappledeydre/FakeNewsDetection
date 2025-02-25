import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import datetime
from database import collection  # MongoDB collection import
from preprocess import preprocess  # Import text preprocessing
from classify_news import predict_fake  # Import classification function

# ✅ Define label mapping (1 = Fake, 0 = Real)
label_mapping = {
    "pants-fire": 1,
    "false": 1,
    "barely-true": 1,
    "half-true": 1,  # Considered Fake
    "mostly-true": 0,
    "true": 0
}

def scrape_politifact_covid(min_claims=50, max_pages=50):
    """Scrapes Politifact for COVID-related fact-checks, assigns labels, classifies, and stores in MongoDB."""
    covid_keywords = ["COVID", "coronavirus", "pandemic", "vaccine", "mask", "quarantine", "lockdown"]
    base_url = "https://www.politifact.com/factchecks/list/?page="
    new_claims = []
    page = 1  

    while len(new_claims) < min_claims and page <= max_pages:
        url = base_url + str(page)
        response = requests.get(url)

        if response.status_code != 200:
            print(f"⚠️ Failed to fetch {url}, skipping...")
            page += 1
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        fact_checks = soup.find_all("article", class_="m-statement")

        for fact in fact_checks:
            try:
                claim_text = fact.find("div", class_="m-statement__quote").text.strip()
                verdict = fact.find("div", class_="m-statement__meter").find("img")["alt"].strip().lower()
                source_link = "https://www.politifact.com" + fact.find("a")["href"]
                date_text = fact.find("footer", class_="m-statement__footer").text.strip()

                # ✅ Convert date to standardized format
                try:
                    date = datetime.datetime.strptime(date_text, "%B %d, %Y").strftime("%Y-%m-%d")
                except ValueError:
                    date = "Unknown"

                # ✅ Filter only COVID-related claims
                if any(keyword.lower() in claim_text.lower() for keyword in covid_keywords):
                    is_fake = label_mapping.get(verdict, None)
                    cleaned_text = preprocess(claim_text)  # ✅ Preprocess before classification

                    # ✅ Check if claim already exists in MongoDB
                    if collection.find_one({"Claim": claim_text}):
                        print(f"🔄 Claim already exists: {claim_text[:50]}... Skipping.")
                        continue

                    # ✅ Only process if label is recognized
                    if is_fake is not None:
                        # ✅ Classify using the BERT model
                        probability_fake = predict_fake(cleaned_text)
                        probability_real = 1 - probability_fake
                        predicted_label = "Fake" if probability_fake > 0.5 else "Real"

                        # ✅ Create document with classification
                        doc = {
                            "Claim": claim_text,
                            "Label": verdict.capitalize(),
                            "is_fake": is_fake,
                            "clean_text": cleaned_text,
                            "Source": source_link,
                            "Date": date,
                            "probability_fake": probability_fake,
                            "probability_real": probability_real,
                            "predicted_label": predicted_label
                        }
                        new_claims.append(doc)

            except AttributeError:
                continue

        print(f"📌 Scraped {len(new_claims)} new claims from {page} pages...")
        page += 1
        time.sleep(1)  # Be respectful to Politifact's server

    # ✅ Insert classified claims into MongoDB
    if new_claims:
        collection.insert_many(new_claims)
        print(f"✅ Inserted {len(new_claims)} classified claims into MongoDB!")

def fetch_new_politifact_claims(min_claims=10, max_pages=12):
    """Fetch and store new classified Politifact claims in MongoDB."""
    scrape_politifact_covid(min_claims=min_claims, max_pages=max_pages)
