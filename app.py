
import streamlit as st
from transformers import pipeline, AutoTokenizer
import requests
from bs4 import BeautifulSoup
import torch

# Page configuration
st.set_page_config(page_title="Robo-Advisor", page_icon="🤖", layout="wide")

# Cache the models to avoid reloading
@st.cache_resource
def load_summarization_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    tokenizer.model_max_length = 1024
    return pipeline("summarization", model="facebook/bart-large-cnn", tokenizer=tokenizer)

@st.cache_resource
def load_sentiment_model():
    # Load your sentiment analysis model here
    # Replace with your actual model path/name
    return pipeline("sentiment-analysis")

# Text summarization function
def text_summarization(url, model):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        text_content = "
".join([p.get_text() for p in paragraphs])
        
        if not text_content.strip():
            text_content = soup.get_text()
        
        # Summarize the text
        summary = model(
            text_content,
            max_length=150,
            min_length=50,
            truncation=True
        )[0]["summary_text"]
        
        return summary, text_content
    except Exception as e:
        return None, str(e)

# Sentiment analysis function
def analyze_sentiment(text, model):
    try:
        result = model(text)[0]
        return result['label'], result['score']
    except Exception as e:
        return None, str(e)

# Generate advice based on sentiment
def generate_advice(sentiment, score):
    advice_map = {
        'POSITIVE': "📈 The article shows positive sentiment. Consider this as a favorable indicator, but always conduct thorough research before making investment decisions.",
        'NEGATIVE': "📉 The article shows negative sentiment. Exercise caution and consider diversifying your portfolio. Consult with a financial advisor for personalized guidance.",
        'NEUTRAL': "➡️ The article shows neutral sentiment. Monitor the situation closely and gather more information before making any decisions."
    }
    
    # Map sentiment labels (adjust based on your model's output)
    sentiment_upper = sentiment.upper()
    if 'POSITIVE' in sentiment_upper or sentiment == '2':
        return advice_map['POSITIVE']
    elif 'NEGATIVE' in sentiment_upper or sentiment == '0':
        return advice_map['NEGATIVE']
    else:
        return advice_map['NEUTRAL']

# Main app
def main():
    st.title("🤖 Robo-Advisor")
    st.markdown("### Financial Article Analysis & Investment Advice")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("This robo-advisor analyzes financial articles through:
"
                "1. Article Input (URL)
"
                "2. Text Summarization
"
                "3. Sentiment Analysis
"
                "4. Investment Advice Generation")
    
    # Main content
    tab1, tab2 = st.tabs(["📊 Analyze Article", "ℹ️ How It Works"])
    
    with tab1:
        # Input section
        st.subheader("1️⃣ Article Input")
        url_input = st.text_input("Enter the URL of a financial article:", 
                                   placeholder="https://example.com/financial-article")
        
        analyze_button = st.button("🔍 Analyze Article", type="primary")
        
        if analyze_button and url_input:
            with st.spinner("Processing article..."):
                # Load models
                summarization_model = load_summarization_model()
                sentiment_model = load_sentiment_model()
                
                # Step 2: Summarization
                st.subheader("2️⃣ Text Summarization")
                summary, original_text = text_summarization(url_input, summarization_model)
                
                if summary:
                    st.success("Summary generated successfully!")
                    st.write(summary)
                    
                    with st.expander("View Original Text"):
                        st.text_area("Original Article Text", original_text, height=200)
                    
                    # Step 3: Sentiment Analysis
                    st.subheader("3️⃣ Sentiment Analysis")
                    sentiment, confidence = analyze_sentiment(summary, sentiment_model)
                    
                    if sentiment:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Sentiment", sentiment)
                        with col2:
                            st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Step 4: Investment Advice
                        st.subheader("4️⃣ Investment Advice")
                        advice = generate_advice(sentiment, confidence)
                        st.info(advice)
                        
                        # Additional disclaimer
                        st.warning("⚠️ **Disclaimer**: This advice is generated by AI and should not be considered as professional financial advice. Always consult with a qualified financial advisor before making investment decisions.")
                    else:
                        st.error(f"Sentiment analysis failed: {confidence}")
                else:
                    st.error(f"Failed to fetch or summarize the article: {original_text}")
        
        elif analyze_button and not url_input:
            st.warning("Please enter a URL to analyze.")
    
    with tab2:
        st.subheader("How the Robo-Advisor Works")
