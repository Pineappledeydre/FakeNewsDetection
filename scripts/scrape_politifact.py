import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from scripts.preprocess import preprocess

def scrape_politifact_covid(min_claims=50, max_pages=50):
    """Scrapes COVID-19 fact-checked claims from Politifact."""
    covid_keywords = ["COVID", "coronavirus", "pandemic", "vaccine", "mask", "lockdown"]
    base_url = "https://www.politifact.com/factchecks/list/?page="
    claims_data = []
    page = 1

    while len(claims_data) < min_claims and page <= max_pages:
        url = base_url + str(page)
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
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
                    claims_data.append([claim_text, verdict, source_link, date])

            except AttributeError:
                continue

        print(f"Scraped {len(claims_data)} COVID claims from {page} pages...")
        page += 1
        time.sleep(1)

    df = pd.DataFrame(claims_data, columns=["Claim", "Label", "Source", "Date"])
    df["clean_text"] = df["Claim"].apply(preprocess)
    df.to_csv("data/covid_politifact_claims.csv", index=False)
    return df
