!pip install streamlit
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def main():
    tokenizer = AutoTokenizer.from_pretrained("kenwuhj/CustomModel_ZA_sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("kenwuhj/CustomModel_ZA_sentiment")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    st.title("Sentiment Analysis with HuggingFace Spaces")
    st.write("Enter a sentence to analyze its sentiment:")

    user_input = st.text_input("Input text for sentiment analysis:", label_visibility="hidden")
    if user_input:
        result = sentiment_pipeline(user_input)
        sentiment = result[0]["label"]
        confidence = result[0]["score"]

        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
