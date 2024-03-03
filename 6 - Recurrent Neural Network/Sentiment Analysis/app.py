import streamlit as st
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tqdm import tqdm
import re

# Function to preprocess the input text
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s\']', '', text)
    filtered_tokens = [token for token in text.split() if len(token) > 1]
    processed_text = ' '.join(filtered_tokens)
    return processed_text

def prepare_data(df):
    processed_tweets = []
    for line in tqdm(df["Tweet"]):
        processed_tweets.append(preprocess_text(line))
    return processed_tweets

# Load the pre-trained model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    # Load the DataFrame
    df = pd.read_csv("twitter_training.csv", header=None, names=["ID", "Game", "Sentiment", "Tweet"]).dropna()
    tweets = prepare_data(df)

    # Preprocess the data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    tweet_sequences = pad_sequences(tokenizer.texts_to_sequences(tweets), padding='post')

    # Load the pre-trained model
    try:
        model = load_model("model.h5")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model = None

    return tokenizer, tweet_sequences, model

# Streamlit app
def main():
    st.title("Sentiment Analysis App")

    # Load model and tokenizer
    tokenizer, tweet_sequences, model = load_model_and_tokenizer()

    # Input text area
    text_input = st.text_area("Enter your text here:")

    if st.button("Predict") and model is not None:
        # Preprocess input text
        processed_text = preprocess_text(text_input)

        # Convert to sequence and pad
        seq = pad_sequences(tokenizer.texts_to_sequences([processed_text]), maxlen=tweet_sequences.shape[1], padding='post')

        # Predict sentiment
        prediction = model.predict(seq)
        map = {0:"Positive", 1:"Neutral", 2:"Negative", 3:"Irrevelent"}
        st.write("Prediction:", map[np.argmax(prediction)])

if __name__ == "__main__":
    main()