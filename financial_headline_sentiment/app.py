import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.title("News Headline Sentiment Analysis")

st.header("Please input the headline news:")

userInput = st.text_input('Input here')


path_tokenizer = 'tokenizer.pickle'
with open(path_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

@st.cache(allow_output_mutation=True)
def teachable_machine_classification(text, weights_file):
    # Load the model
    #model = tf.saved_model.load(weights_file)
    model = tf.keras.models.load_model(weights_file)
    
    text_series = pd.Series(text)
    
    input_tokenized = tokenizer.texts_to_sequences(text_series)
    input_pad = pad_sequences(input_tokenized, maxlen=100, padding='post')

    # run the inference
    predictions = model.predict(input_pad)
    predictions = predictions.argmax(axis=1)

    return  predictions



if st.button('Predict'):
    st.write("Predicting. Please wait... ")
    label = teachable_machine_classification(userInput, 'model_glove_conv1d_gap')
    
    if label == 0:
        st.write("It's a negative sentiment")
    elif label == 1:
        st.write("It's a neutral sentiment")
    elif label == 2:
        st.write("It's a positive sentiment")
    else:
        st.write("error")
else:
    st.write('')


