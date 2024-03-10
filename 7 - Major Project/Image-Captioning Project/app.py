import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model

# Load the pre-trained model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    model = load_model("./Model Building/model.h5")

    with open("./Preprocessing/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("./Preprocessing/all_captions.pkl", "rb") as f:
        all_captions = pickle.load(f)

    max_len = max(len(token.split()) for token in all_captions)

    model_vgg16 = VGG16()
    model_vgg16 = Model(inputs=model_vgg16.inputs, outputs=model_vgg16.layers[-2].output)
    return model, tokenizer, max_len, model_vgg16


def idx_to_integer(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def get_image_feature(img, model_vgg16):
    image = load_img(img, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model_vgg16.predict(image, verbose=0)
    return feature


# generating captions for the image
import numpy as np


def predict_captions(image, model, tokenizer, max_len, model_vgg16):
    image = get_image_feature(image, model_vgg16)
    in_text = "<startseq>"
    predict_text = ""
    for i in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len, padding="post")
        y_hat = model.predict([image, seq], verbose=0).flatten()
        top_indices = np.argsort(y_hat)[-3:][::-1]
        top_probabilities = y_hat[top_indices]
        chosen_index = np.random.choice(top_indices, p=top_probabilities / np.sum(top_probabilities))
        word = idx_to_integer(chosen_index, tokenizer)
        if word is None:
            break
        if word == "endseq":
            in_text += ' <endseq>'
            break
        in_text += ' ' + word
        predict_text += word + ' '
    return predict_text


def main():
    st.title("Image Captioning Application")

    # Load model and tokenizer
    model, tokenizer, max_len, model_vgg16 = load_model_and_tokenizer()

    # File uploader for image
    st.sidebar.title("Upload Image")
    uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.sidebar.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.sidebar.write("")
        if st.sidebar.button("Predict"):
            st.sidebar.write("Generating captions...")
            # Predict captions for uploaded image
            predicted_caption = predict_captions(uploaded_image, model, tokenizer, max_len, model_vgg16)
            st.markdown(f"<b><p style='font-size: 30px;'>Caption:</p><b>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 20px;'>It seems like {predicted_caption}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
