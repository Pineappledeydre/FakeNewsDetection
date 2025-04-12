# Fake News Detector – COVID-19 Edition

This project is a **Streamlit-based web application** that uses a fine-tuned BERT model to classify fact-checked claims from **[Politifact](https://www.politifact.com/)** as **fake** or **real**, with a focus on COVID-19-related misinformation. It also allows users to input their own claims and get predictions, while maintaining a MongoDB database of all classified entries.

---

## Features

- Scrapes and classifies live claims from Politifact in real time  
- Uses a fine-tuned BERT model to classify fake vs real news  
- Visualizes prediction statistics, including fake/real distribution and model accuracy  
- Allows users to submit and classify their own claims  
- Stores all results in MongoDB for long-term tracking and analysis  

---

## Tech Stack

- **Frontend:** Streamlit  
- **ML Model:** Fine-tuned `bert-base-uncased`  
- **Backend:** MongoDB  
- **Scraping:** Custom scraper for Politifact  
- **Preprocessing:** Custom tokenization and cleaning  
- **Visualization:** Matplotlib & Streamlit UI  

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fake-news-covid19.git
cd fake-news-covid19
```

### 2. Set up environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file or use Streamlit Secrets to store your MongoDB URI:

```
MONGO_URI=your_mongodb_uri
```

You can also add this in `.streamlit/secrets.toml`:

```toml
MONGO_URI = "your_mongodb_uri"
```

### 3. Add the model

Place your fine-tuned model file in the following path:

```
models/bert_finetuned_model.pth
```

---

## Running the App

```bash
streamlit run app.py
```

---

## Key Components

### Fetch New Claims  
Scrape and classify recent fact-checked claims from Politifact by selecting the number of claims and pages to fetch.

### Latest Predictions  
View the most recent classified claims with predicted labels and fake news probability.

### Fake vs Real Distribution  
See the overall distribution of predicted labels and model accuracy.

### Submit Your Own Claim  
Input any claim and see the classification results along with probabilities.

---

## Folder Structure

```
├── app.py                       # Main Streamlit app
├── models/
│   └── bert_finetuned_model.pth
├── scripts/
│   ├── classify_news.py         # Model + Tokenizer
│   ├── database.py              # MongoDB helpers
│   ├── preprocess.py            # Text cleaning
│   └── scrape_politifact.py     # Politifact scraper
└── requirements.txt
```

---

## Notes

- Model must be placed manually (`models/bert_finetuned_model.pth`)
- Claims are stored in MongoDB under `FakeNewsDB > PolitifactClaims`
- The app will show errors if `MONGO_URI` or the model is missing

---

## Future Improvements

- Add user authentication  
- Enable editing/deleting of submitted claims  
- Display time series of misinformation trends  
- Add model explanation (SHAP, LIME)
