
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import seaborn as sns

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Advanced Product Review Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🛒 Product Review Sentiment Dashboard</h1>", unsafe_allow_html=True)

# --- Sidebar Filters ---
st.sidebar.header("Filters & Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV (must contain 'review')", type="csv")
keyword_filter = st.sidebar.text_input("Search Keyword")
sentiment_filter = st.sidebar.multiselect("Filter Sentiment", ["Positive","Neutral","Negative"], default=["Positive","Neutral","Negative"])

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📄 Single Review", "📊 Bulk Analysis", "💡 Suggestions"])

# --- Function to get sentiment ---
def get_sentiment(text):
    score = sia.polarity_scores(str(text))['compound']
    if score >= 0.05: return "Positive"
    elif score <= -0.05: return "Negative"
    else: return "Neutral"

# --- Load Data ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    if 'review' not in df.columns:
        st.error("CSV must contain column 'review'.")
    else:
        df['sentiment'] = df['review'].apply(get_sentiment)
        df['emoji'] = df['sentiment'].map({"Positive":"😊","Neutral":"😐","Negative":"😡"})

        # Apply filters
        filtered_df = df
        if keyword_filter:
            filtered_df = filtered_df[filtered_df['review'].str.contains(keyword_filter, case=False, na=False)]
        if sentiment_filter:
            filtered_df = filtered_df[filtered_df['sentiment'].isin(sentiment_filter)]

        # --- Tab 1: Single Review ---
        with tab1:
            st.markdown("### 🔎 Analyze Single Review")
            review_input = st.text_area("Enter your review text here:")
            if st.button("Analyze Review"):
                if review_input.strip() != "":
                    sentiment = get_sentiment(review_input)
                    emoji = {"Positive":"😊","Neutral":"😐","Negative":"😡"}[sentiment]
                    st.markdown(f"**Sentiment:** {sentiment} {emoji}")
                    wc = WordCloud(width=400,height=200,background_color='white').generate(review_input)
                    st.image(wc.to_array())
                else:
                    st.warning("Please enter a review to analyze.")

        # --- Tab 2: Bulk Analysis ---
        with tab2:
            st.markdown("### 📊 Bulk Review Analysis")
            col1,col2,col3,col4 = st.columns(4)
            total = len(filtered_df)
            pos = (filtered_df['sentiment']=="Positive").sum()
            neu = (filtered_df['sentiment']=="Neutral").sum()
            neg = (filtered_df['sentiment']=="Negative").sum()
            col1.metric("Total Reviews", total)
            col2.metric("Positive", f"{pos} 😊")
            col3.metric("Neutral", f"{neu} 😐")
            col4.metric("Negative", f"{neg} 😡")

            # Pie chart
            fig, ax = plt.subplots(figsize=(3,3))
            sentiment_counts = filtered_df['sentiment'].value_counts()
            wedges, texts, autotexts = ax.pie(
                sentiment_counts,
                labels=sentiment_counts.index,
                autopct='%1.0f%%',
                startangle=90,
                wedgeprops={'width':0.5, 'edgecolor':'black'},
                colors=['#4CAF50','#FFC107','#F44336']
            )
            for t in texts: t.set_fontsize(6)
            for a in autotexts: a.set_fontsize(5)
            ax.axis('equal')
            st.pyplot(fig)

            # Wordclouds
            st.markdown("### Wordclouds by Sentiment")
            col1,col2,col3 = st.columns(3)
            for sentiment_type,col in zip(["Positive","Neutral","Negative"],[col1,col2,col3]):
                text = " ".join(filtered_df[filtered_df['sentiment']==sentiment_type]['review'].astype(str))
                if text:
                    col.markdown(f"**{sentiment_type} Reviews**")
                    wc = WordCloud(width=300,height=150,background_color='white').generate(text)
                    col.image(wc.to_array())

        # --- Tab 3: Suggestions & Category Insights ---
        with tab3:
            st.markdown("### 💡 Suggestions & Category Insights")
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
                rev = review.lower()
                for cat, keywords in categories.items():
                    if any(k in rev for k in keywords):
                        cat_counter[cat]+=1

            if cat_counter:
                fig2, ax2 = plt.subplots(figsize=(5,2))
                ax2.barh(list(cat_counter.keys()), list(cat_counter.values()), color="#FF6F61")
                ax2.set_xlabel("Negative Mentions")
                ax2.set_ylabel("Category")
                ax2.set_title("Most Complained Categories")
                st.pyplot(fig2)

            total_neg = len(negative_reviews)
            suggestions = []
            for cat, count in cat_counter.most_common(10):
                percent = round((count/total_neg)*100)
                suggestions.append(f"⚠️ {percent}% of negative reviews mention {cat}. Consider improving {cat}.")
            for s in suggestions:
                st.write(s)