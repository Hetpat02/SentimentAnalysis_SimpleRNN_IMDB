import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence


word_index = imdb.get_word_index()
reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])

model = load_model('simple_rnn_imdb.h5')

#helper function
def decode_review(review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in review])

def preprocess_text(text):
    words = text.lower().split()
    review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([review], maxlen=500)
    return padded_review


#prediction function
def predict_senti(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)

    # sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]


#streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as either positive or negative.')

#user inp
user_inp = st.text_area('Movie Review:')

if st.button('Classify'):
    if user_inp:
        preprocessed_inp = preprocess_text(user_inp)

        #make prediction
        prediction = model.predict(preprocessed_inp)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        #display result
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction[0][0]}')

    else:
        st.write('Please enter a review to classify.')
