# 🛒 Product Review Sentiment Dashboard

A beautiful, modern product review sentiment analysis dashboard built with a **FastAPI** backend and a **Tailwind CSS** frontend. Analyze customer reviews in bulk or one at a time, visualize sentiment trends, and get actionable improvement suggestions.

---

## ✨ Features

- 📄 **Single Review Analysis** — Paste any review and get instant sentiment + wordcloud
- 📊 **Bulk CSV Analysis** — Upload a CSV file and analyze hundreds of reviews at once
- 💡 **Suggestions Tab** — Category breakdown chart + improvement suggestions from negative reviews
- 🤖 **AI Suggestion Chatbot** — Ask questions about your review data
- 🌗 **Dark Mode UI** — Glassmorphism design with Tailwind CSS via CDN

---

## 🗂️ Project Structure

```
sentiment analysis/
├── api.py              # FastAPI backend (sentiment logic, endpoints)
├── app.py              # Original Streamlit app (legacy)
├── index.html          # Frontend UI (Tailwind CSS, Chart.js)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🚀 How to Run

### Prerequisites

- **Python 3.9+** installed
- **pip** available in your terminal

---

### Step 1 — Install Dependencies

Open a terminal in the project folder and run:

```powershell
pip install -r requirements.txt
```

---

### Step 2 — Start the Backend API

Run the FastAPI server:

```powershell
uvicorn api:app --reload
```

You should see output like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

> ⚠️ **Keep this terminal open.** The backend must be running for the frontend to work.

---

### Step 3 — Open the Frontend

Simply open `index.html` directly in your browser:

- **Option A**: Double-click `index.html` in File Explorer
- **Option B**: Right-click → *Open with* → your browser

The app will load at a local `file://` URL (no extra server needed for the frontend).

---

## 📋 Using the App

### 📄 Single Review Tab
1. Type or paste any review text into the text area
2. Click **Analyze Review**
3. See the sentiment result (Positive / Neutral / Negative) and a wordcloud

### 📊 Bulk Analysis Tab
1. In the **sidebar**, click *Click to upload* and select a `.csv` file
   - The CSV **must contain a column named `review`**
2. (Optional) Add a keyword filter or toggle sentiment checkboxes
3. Click **Analyze Bulk Data**
4. View metrics, sentiment distribution chart, and wordclouds

### 💡 Suggestions Tab
- After a bulk analysis, switch to the **Suggestions** tab
- See a bar chart of the most complained-about categories
- Read actionable improvement suggestions based on negative reviews

### 🤖 Chatbot
- Click the **robot icon** (bottom-right corner)
- Ask questions like:
  - *"What are the main issues?"*
  - *"Is there a problem with quality?"*
  - *"What about battery?"*

---

## 📁 CSV Format

Your CSV file must have at least one column named `review`. Example:

```csv
review
"The battery life is terrible, only lasts 3 hours."
"Great camera quality! Really happy with this purchase."
"Delivery was late and packaging was damaged."
"Decent product for the price."
```

---

## 🛑 Stopping the App

Press `Ctrl + C` in the terminal where `uvicorn` is running to stop the backend server.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Backend API framework |
| `uvicorn` | ASGI server to run FastAPI |
| `python-multipart` | Handles file uploads |
| `pandas` | CSV reading and data processing |
| `nltk` | VADER sentiment analysis |
| `wordcloud` | Wordcloud image generation |
| `matplotlib` | Image rendering for wordclouds |
| `scikit-learn` | ML utilities |

Frontend uses **CDN links only** (no npm required):
- [Tailwind CSS](https://cdn.tailwindcss.com)
- [Chart.js](https://cdn.jsdelivr.net/npm/chart.js)
- [Font Awesome](https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css)
- [Google Fonts – Inter](https://fonts.google.com/specimen/Inter)
