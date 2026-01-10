from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from wordcloud import WordCloud
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import io
import base64
import json
import traceback

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

app = FastAPI(title="Sentiment Analysis API")

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

# Enable CORS for local HTML file access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_sentiment(text):
    score = sia.polarity_scores(str(text))['compound']
    if score >= 0.05: return "Positive"
    elif score <= -0.05: return "Negative"
    else: return "Neutral"

def get_emoji(sentiment):
    return {"Positive":"😊","Neutral":"😐","Negative":"😡"}.get(sentiment, "")

def create_wordcloud_base64(text):
    if not text or not str(text).strip():
        return None
    try:
        wc = WordCloud(width=500, height=300, background_color='#1e293b', colormap='Blues').generate(str(text))
        img = io.BytesIO()
        wc.to_image().save(img, format='PNG')
        return base64.b64encode(img.getvalue()).decode('utf-8')
    except Exception as e:
        print("Wordcloud generation error:", e)
        return None

class SingleReviewRequest(BaseModel):
    text: str

@app.post("/analyze_single")
def analyze_single(req: SingleReviewRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty review")
    
    sentiment = get_sentiment(req.text)
    emoji = get_emoji(sentiment)
    wc_base64 = create_wordcloud_base64(req.text)
    
    return {
        "sentiment": sentiment,
        "emoji": emoji,
        "wordcloud": wc_base64
    }

@app.post("/analyze_csv")
async def analyze_csv(
    file: UploadFile = File(...),
    keyword_filter: str = Form(""),
    sentiment_filter: str = Form("[]")
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
        
        if 'review' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain column 'review'.")
            
        df['sentiment'] = df['review'].apply(get_sentiment)
        
        # Apply filters
        filtered_df = df
        if keyword_filter:
            filtered_df = filtered_df[filtered_df['review'].str.contains(keyword_filter, case=False, na=False)]
        
        sentiments_to_include = json.loads(sentiment_filter)
        if sentiments_to_include:
            filtered_df = filtered_df[filtered_df['sentiment'].isin(sentiments_to_include)]
            
        total = len(filtered_df)
        pos = int((filtered_df['sentiment']=="Positive").sum())
        neu = int((filtered_df['sentiment']=="Neutral").sum())
        neg = int((filtered_df['sentiment']=="Negative").sum())
        
        # Pie chart data
        sentiment_counts = filtered_df['sentiment'].value_counts().to_dict()
        
        # Wordclouds
        wordclouds = {}
        for s_type in ["Positive", "Neutral", "Negative"]:
            text = " ".join(filtered_df[filtered_df['sentiment']==s_type]['review'].astype(str))
            wc = create_wordcloud_base64(text) if text.strip() else None
            wordclouds[s_type] = wc
            
        # Suggestions
        negative_reviews = filtered_df[filtered_df['sentiment']=="Negative"]['review'].tolist()
        categories = {
            'camera':['camera','photo','lens'],
            'price':['price','expensive','cost','value'],
            'battery':['battery','charge','power'],
            'delivery':['delivery','shipping','late','packaging'],
            'quality':['quality','material','build','durable']
        }
        cat_counter = Counter()
        for review in negative_reviews:
            rev = str(review).lower()
            for cat, keywords in categories.items():
                if any(k in rev for k in keywords):
                    cat_counter[cat]+=1
                    
        total_neg = len(negative_reviews)
        suggestions = []
        cat_chart_data = {"labels": [], "values": []}
        if total_neg > 0:
            for cat, count in cat_counter.most_common(10):
                percent = round((count/total_neg)*100)
                suggestions.append(f"⚠️ {percent}% of negative reviews mention {cat}. Consider improving {cat}.")
                cat_chart_data["labels"].append(cat)
                cat_chart_data["values"].append(count)

        # Positive category analysis (best features)
        positive_reviews = filtered_df[filtered_df['sentiment']=="Positive"]['review'].tolist()
        pos_counter = Counter()
        for review in positive_reviews:
            rev = str(review).lower()
            for cat, keywords in categories.items():
                if any(k in rev for k in keywords):
                    pos_counter[cat] += 1

        total_pos = len(positive_reviews)
        positive_cat_data = {"labels": [], "values": [], "percentages": []}
        for cat, count in pos_counter.most_common():
            pct = round((count / total_pos) * 100) if total_pos > 0 else 0
            positive_cat_data["labels"].append(cat)
            positive_cat_data["values"].append(count)
            positive_cat_data["percentages"].append(pct)

        # Neutral category analysis
        neutral_reviews = filtered_df[filtered_df['sentiment']=="Neutral"]['review'].tolist()
        neu_counter = Counter()
        for review in neutral_reviews:
            rev = str(review).lower()
            for cat, keywords in categories.items():
                if any(k in rev for k in keywords):
                    neu_counter[cat] += 1

        total_neu_r = len(neutral_reviews)
        neutral_cat_data = {"labels": [], "values": [], "percentages": []}
        for cat, count in neu_counter.most_common():
            pct = round((count / total_neu_r) * 100) if total_neu_r > 0 else 0
            neutral_cat_data["labels"].append(cat)
            neutral_cat_data["values"].append(count)
            neutral_cat_data["percentages"].append(pct)

        return {
            "metrics": {"total": total, "Positive": pos, "Neutral": neu, "Negative": neg},
            "pie_chart": sentiment_counts,
            "wordclouds": wordclouds,
            "suggestions": suggestions,
            "category_chart": cat_chart_data,
            "positive_categories": positive_cat_data,
            "neutral_categories": neutral_cat_data
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    message: str
    context: str           # suggestions text
    metrics: dict = {}     # {total, Positive, Neutral, Negative}
    categories: dict = {}  # negative: {labels, values}
    positive_categories: dict = {}  # {labels, values, percentages}
    neutral_categories: dict = {}   # {labels, values, percentages}

@app.post("/chat")
def chat(req: ChatRequest):
    msg = req.message.lower().strip()
    metrics = req.metrics
    categories = req.categories
    pos_cats = req.positive_categories
    neu_cats = req.neutral_categories
    suggestions_text = req.context.strip()
    has_data = bool(metrics) and metrics.get("total", 0) > 0

    # ── Helper ──────────────────────────────────────────────────────────────
    def pct(key):
        total = metrics.get("total", 1) or 1
        return round(metrics.get(key, 0) / total * 100)

    def top_category():
        labels = categories.get("labels", [])
        values = categories.get("values", [])
        if labels and values:
            idx = values.index(max(values))
            return labels[idx], values[idx]
        return None, None

    def cat_pct(cat_name):
        labels = categories.get("labels", [])
        values = categories.get("values", [])
        neg = metrics.get("Negative", 0) or 1
        if cat_name in labels:
            v = values[labels.index(cat_name)]
            return round(v / neg * 100), v
        return None, None

    # ── Greetings ────────────────────────────────────────────────────────────
    greet_words = ["hello", "hi", "hey", "howdy", "what's up", "sup"]
    if any(w in msg for w in greet_words):
        if has_data:
            return {"reply": f"👋 Hello! I've analysed your data — <b>{metrics['total']} reviews</b> in total. Ask me anything about sentiments, categories, or what to improve!"}
        return {"reply": "👋 Hello! I'm your Review Intelligence Bot. Please upload and analyse a CSV file first, then I can answer questions about your reviews."}

    # ── No data yet ──────────────────────────────────────────────────────────
    if not has_data:
        return {"reply": "📂 No data analysed yet. Please upload a CSV file from the sidebar and click <b>Analyze Bulk Data</b> first."}

    # ── Overall summary ──────────────────────────────────────────────────────
    summary_words = ["summary", "overview", "tell me", "what do you know", "overall", "describe", "explain", "analyse", "analyze"]
    if any(w in msg for w in summary_words):
        top_cat, top_val = top_category()
        cat_note = f" The top complaint category is <b>{top_cat}</b> ({top_val} mentions)." if top_cat else ""
        return {"reply": (
            f"📊 <b>Summary of your {metrics['total']} reviews:</b><br>"
            f"• ✅ <b>{metrics['Positive']} Positive</b> ({pct('Positive')}%)<br>"
            f"• 😐 <b>{metrics['Neutral']} Neutral</b> ({pct('Neutral')}%)<br>"
            f"• ❌ <b>{metrics['Negative']} Negative</b> ({pct('Negative')}%)<br>"
            f"{cat_note}"
        )}

    # ── Positive sentiment ───────────────────────────────────────────────────
    if any(w in msg for w in ["positive", "good review", "happy customer", "satisfied", "praise"]):
        p = pct('Positive')
        if p >= 60:
            return {"reply": f"😊 Great news! <b>{metrics['Positive']} reviews ({p}%)</b> are positive. Your customers are largely satisfied. Keep up the quality!"}
        elif p >= 40:
            return {"reply": f"😊 <b>{metrics['Positive']} reviews ({p}%)</b> are positive — a moderate satisfaction rate. There's room to push this higher by addressing negative feedback."}
        else:
            return {"reply": f"⚠️ Only <b>{metrics['Positive']} reviews ({p}%)</b> are positive. This is quite low — focus on resolving the top complaints to improve satisfaction."}

    # ── Negative sentiment ───────────────────────────────────────────────────
    if any(w in msg for w in ["negative", "bad review", "complaint", "unhappy", "dissatisfied", "problem", "issue", "worst"]):
        n = pct('Negative')
        top_cat, top_val = top_category()
        cat_note = f" The biggest driver of negativity is <b>{top_cat}</b> with {top_val} mentions." if top_cat else ""
        if n >= 40:
            return {"reply": f"😡 <b>{metrics['Negative']} reviews ({n}%)</b> are negative — this needs urgent attention!{cat_note} Review the Suggestions tab for detailed action points."}
        else:
            return {"reply": f"😡 <b>{metrics['Negative']} reviews ({n}%)</b> are negative.{cat_note} Keep an eye on these and address the complaints proactively."}

    # ── Neutral sentiment ────────────────────────────────────────────────────
    if any(w in msg for w in ["neutral", "mixed", "indifferent", "average", "neutral review", "neutral field"]):
        n = pct('Neutral')
        neu_labels = neu_cats.get("labels", [])
        neu_values = neu_cats.get("values", [])
        neu_pcts   = neu_cats.get("percentages", [])
        if neu_labels:
            lines = "".join(
                f"• <b>{l.capitalize()}</b>: {v} mentions ({p}%)<br>"
                for l, v, p in zip(neu_labels, neu_values, neu_pcts)
            )
            return {"reply": (
                f"😐 <b>{metrics['Neutral']} reviews ({n}%)</b> are neutral.<br><br>"
                f"<b>Topics mentioned in neutral reviews:</b><br>{lines}"
                f"<br>These are areas customers feel <i>mixed</i> about — improving them could shift opinions positive!"
            )}
        return {"reply": f"😐 <b>{metrics['Neutral']} reviews ({n}%)</b> are neutral. These customers are on the fence — no specific categories stood out in their mixed feedback."}

    # ── Best features / what customers love ─────────────────────────────────
    if any(w in msg for w in ["best feature", "best aspect", "loved", "love", "what do customer like", "what is good", "good about", "customers like", "customers love", "top feature", "praised", "strength"]):
        pos_labels = pos_cats.get("labels", [])
        pos_values = pos_cats.get("values", [])
        pos_pcts   = pos_cats.get("percentages", [])
        if pos_labels:
            lines = "".join(
                f"• ⭐ <b>{l.capitalize()}</b>: praised in {v} positive reviews ({p}%)<br>"
                for l, v, p in zip(pos_labels, pos_values, pos_pcts)
            )
            top = pos_labels[0].capitalize()
            return {"reply": (
                f"🌟 <b>Top features customers love (from {metrics['Positive']} positive reviews):</b><br><br>"
                f"{lines}<br>"
                f"<b>{top}</b> is your strongest point — keep it up!"
            )}
        return {"reply": f"✅ No specific features were highlighted in positive reviews. Customers seem generally satisfied without focusing on a specific attribute — that's still good!"}

    # ── Total / count ────────────────────────────────────────────────────────
    if any(w in msg for w in ["how many", "total", "count", "number of review"]):
        return {"reply": f"📋 You have a total of <b>{metrics['total']} reviews</b> analysed: {metrics['Positive']} positive, {metrics['Neutral']} neutral, and {metrics['Negative']} negative."}

    # ── Specific categories (cross-sentiment) ────────────────────────────────
    cat_keywords = {
        "quality": ["quality", "build", "material", "durable", "durability"],
        "price":   ["price", "expensive", "cost", "value", "cheap", "affordable"],
        "battery": ["battery", "charge", "charging", "power"],
        "camera":  ["camera", "photo", "picture", "lens", "image"],
        "delivery":["delivery", "shipping", "ship", "package", "packaging", "late", "arrived"],
    }
    for cat, words in cat_keywords.items():
        if any(w in msg for w in words):
            neg_labels = categories.get("labels", [])
            neg_values = categories.get("values", [])
            pos_labels = pos_cats.get("labels", [])
            pos_values = pos_cats.get("values", [])
            neu_labels = neu_cats.get("labels", [])
            neu_values = neu_cats.get("values", [])

            neg_count = neg_values[neg_labels.index(cat)] if cat in neg_labels else 0
            pos_count = pos_values[pos_labels.index(cat)] if cat in pos_labels else 0
            neu_count = neu_values[neu_labels.index(cat)] if cat in neu_labels else 0

            total_neg = metrics.get("Negative", 1) or 1
            total_pos = metrics.get("Positive", 1) or 1

            if neg_count == 0 and pos_count == 0 and neu_count == 0:
                return {"reply": f"🔍 <b>{cat.capitalize()}</b> is not significantly mentioned in any reviews in your dataset."}

            parts = []
            if pos_count:
                parts.append(f"• ✅ <b>Positive</b>: praised in {pos_count} reviews ({round(pos_count/total_pos*100)}% of positive)")
            if neu_count:
                parts.append(f"• 😐 <b>Neutral</b>: mentioned in {neu_count} neutral reviews")
            if neg_count:
                neg_p = round(neg_count / total_neg * 100)
                severity = "🔴 High priority" if neg_p >= 30 else "🟡 Watch closely"
                parts.append(f"• ❌ <b>Negative</b>: complained in {neg_count} reviews ({neg_p}% of negatives) — {severity}")

            return {"reply": (
                f"📌 <b>{cat.capitalize()} — Full Sentiment Breakdown:</b><br><br>"
                + "<br>".join(parts)
            )}

    # ── Top / worst issue ────────────────────────────────────────────────────
    if any(w in msg for w in ["top issue", "biggest issue", "worst issue", "most complained", "main problem", "biggest problem", "top complaint", "most mentioned"]):
        top_cat, top_val = top_category()
        if top_cat:
            neg = metrics.get("Negative", 1) or 1
            p = round(top_val / neg * 100)
            return {"reply": f"🔴 The <b>most complained-about category</b> is <b>{top_cat.capitalize()}</b> with <b>{top_val} mentions</b> ({p}% of negative reviews). Focus your improvement efforts here first."}
        return {"reply": "✅ No specific complaint category stands out — negative reviews don't focus on a single recurring issue."}

    # ── Improvement / suggestions ────────────────────────────────────────────
    if any(w in msg for w in ["improve", "suggestion", "recommend", "fix", "what should", "advice", "action", "next step"]):
        if suggestions_text and "No data" not in suggestions_text:
            return {"reply": f"💡 <b>Recommended actions based on your data:</b><br>{suggestions_text.replace(chr(10), '<br>')}"}
        return {"reply": "💡 No specific improvement areas detected — your negative reviews don't cluster around common complaint categories. Keep monitoring!"}

    # ── Percentage questions ─────────────────────────────────────────────────
    if any(w in msg for w in ["percent", "percentage", "ratio", "proportion", "breakdown"]):
        return {"reply": (
            f"📈 <b>Sentiment breakdown ({metrics['total']} reviews):</b><br>"
            f"• Positive: <b>{pct('Positive')}%</b> ({metrics['Positive']})<br>"
            f"• Neutral: <b>{pct('Neutral')}%</b> ({metrics['Neutral']})<br>"
            f"• Negative: <b>{pct('Negative')}%</b> ({metrics['Negative']})"
        )}

    # ── Comparison ───────────────────────────────────────────────────────────
    if any(w in msg for w in ["compare", "vs", "versus", "difference", "more positive", "more negative"]):
        diff = metrics['Positive'] - metrics['Negative']
        if diff > 0:
            return {"reply": f"📊 Positive reviews outweigh negative ones by <b>{diff} reviews</b> ({pct('Positive')}% vs {pct('Negative')}%). Overall sentiment leans positive!"}
        elif diff < 0:
            return {"reply": f"📊 Negative reviews outnumber positive ones by <b>{abs(diff)} reviews</b> ({pct('Negative')}% vs {pct('Positive')}%). This needs attention!"}
        else:
            return {"reply": f"📊 Positive and negative reviews are exactly <b>equal</b> ({metrics['Positive']} each). Sentiment is perfectly balanced."}

    # ── Fallback ─────────────────────────────────────────────────────────────
    top_cat, top_val = top_category()
    pos_labels = pos_cats.get("labels", [])
    best_feature = pos_labels[0].capitalize() if pos_labels else None
    lines = [
        f"• Total reviews: <b>{metrics['total']}</b>",
        f"• Positive rate: <b>{pct('Positive')}%</b>",
    ]
    if best_feature:
        lines.append(f"• Customers love: <b>{best_feature}</b>")
    if top_cat:
        lines.append(f"• Top complaint: <b>{top_cat.capitalize()}</b> ({top_val} mentions)")
    lines.append("")
    lines.append("Try: <i>'Best feature?'</i>, <i>'Tell me about battery'</i>, <i>'What should I improve?'</i>, or <i>'Neutral reviews?'</i>")
    return {"reply": "🤔 I didn't quite get that, but here's a quick snapshot:<br>" + "<br>".join(lines)}
