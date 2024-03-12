import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import streamlit as st

# Load the pre-trained model and tokenizer
def load_model_and_tokenizer():
    model = load_model("./Preprocessing_Modeling/model.h5")

    with open("./Preprocessing_Modeling/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    max_len = 340
    return model, tokenizer, max_len

model, tokenizer, max_len = load_model_and_tokenizer()

def generate_next_words(input_text, num_words):
    generated_text = input_text
    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([input_text])[0]
        seq = pad_sequences([seq], padding="post", maxlen=max_len)[0]
        # Reshape the sequence to add a batch dimension
        seq = np.reshape(seq, (1, len(seq), 1))  # Assuming input_dim is 1
        y_hat = model.predict(seq, verbose=False).flatten()
        top_indices = np.argsort(y_hat)[-2:][::-1]  # Get top 2 indices
        selected_index = np.random.choice(top_indices)  # Randomly choose one of the top 2 indices
        next_word = tokenizer.index_word.get(selected_index, "")
        next_word = next_word.replace("…",  "")
        input_text += " " + next_word
        generated_text += " " + next_word
    return generated_text

# Streamlit app
def main():
    st.title('Text Generator')

    # Input text and number of suggestions
    input_text = st.text_input('Input Text:', value='', key='input_text')
    num_words = st.slider('Number of Words:', min_value=1, max_value=50, value=3)

    # Generate suggestions when button is clicked
    if st.button('Generate Text'):
        description = generate_next_words(input_text, num_words)
        st.write(description)

if __name__ == '__main__':
    main()
