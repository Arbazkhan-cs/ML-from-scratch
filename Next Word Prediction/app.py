import streamlit as st
import numpy as np
from keras.utils import pad_sequences
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import pickle

def load_model_tokenizer():
    # Load tokenizer and model
    with open("./Preprocessing_Modeling/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    max_len = 340  # Define max_len according to your data
    model = load_model("./Preprocessing_Modeling/model.h5")  # Load your trained model here
    return model, tokenizer, max_len

model, tokenizer, max_len = load_model_tokenizer()

# Define integer_to_text function
def integer_to_text(tokenizer, label):
    for idx, val in tokenizer.word_index.items():
        if val == label:
            return idx
    return None

# Define generate_next_word function
def generate_next_word(text, next_words):
    for _ in range(next_words):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], padding="post", maxlen=max_len)[0]
        # Reshape the sequence to add a batch dimension
        seq = np.reshape(seq, (1, len(seq), 1))  # Assuming feature_dim is 1
        label_probabilities = model.predict(seq, verbose=False).flatten()

        # Get the indices of the top 3 probabilities
        top_3_indices = np.argsort(label_probabilities)[-3:][::-1]

        # Randomly select one of the top 3 indices
        selected_index = np.random.choice(top_3_indices)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == selected_index:
                output_word = word
                break
        text += " " + output_word
    return text

# Streamlit app
def main():
    st.title("Text Generation App")

    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose an option", ["Integer to Text", "Generate Next Word"])

    if option == "Integer to Text":
        st.header("Convert Integer to Text")
        label = st.number_input("Enter the integer label:", min_value=1, step=1)
        if st.button("Convert"):
            if tokenizer is not None:
                text = integer_to_text(tokenizer, label)
                st.write(f"Text: {text}")
            else:
                st.error("Tokenizer is not loaded.")

    elif option == "Generate Next Word":
        st.header("Generate Next Word")
        input_text = st.text_input("Enter input text:")
        next_words = st.number_input("Enter number of next words to generate:", min_value=1, step=1)
        if st.button("Generate"):
            if tokenizer is not None and model is not None:
                generated_text = generate_next_word(input_text, next_words)
                st.write(f"Generated Text: {generated_text}")
            else:
                st.error("Tokenizer or model is not loaded.")

if __name__ == "__main__":
    main()
