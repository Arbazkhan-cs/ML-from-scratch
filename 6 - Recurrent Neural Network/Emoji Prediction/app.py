import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle, json, re
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

nltk.download("stopwords")
nltk.download('punkt')

# Load the pre-trained model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("emojiMap.json", "r") as f:
        map_ = json.load(f)

    model = load_model("model.h5")

    return tokenizer, model, map_


def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.strip()
    return tweet

def remove_stopwords(text):
  stop_words = set(stopwords.words('english'))
  tokens = word_tokenize(text) # same as split function
  filteredTokens = [token.lower() for token in tokens if token.lower() not in stop_words]
  filteredTokens = [preprocess_tweet(token) for token in filteredTokens if len(token) > 1]
  res = ' '.join(filteredTokens)
  return res

def prediction(text, tokenizer, model, map_):
  preprocess_text = [remove_stopwords(text)]
  seq = tokenizer.texts_to_sequences(preprocess_text)
  seq = pad_sequences(seq, padding="post", maxlen=32)
  predict = model.predict(seq)
  label = np.argmax(predict)
  return map_[f'{label}']

# Streamlit app
def main():
    st.title("Sentiment Analysis App")
    tokenizer, model, map_ = load_model_and_tokenizer()
    text_input = st.text_area("Enter your text here:")

    if st.button("Predict") and model is not None:
        result = prediction(text_input, tokenizer, model, map_)
        st.success(result)
    else:
        st.write("Please enter some text to analyze the sentiment.")

if __name__ == "__main__":
    main()