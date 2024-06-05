# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import pickle
import streamlit as st
import time

file = open('data.txt', 'r')
data = file.read()
file.close()


import os
import tensorflow

if os.path.exists("next_word_model.keras"):
    True

else:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([data])
    
    input_sequences = []
    for sentence in data.split('\n'):
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:i + 1])

    max_len = max([len(x) for x in input_sequences])

    padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

    X = padded_input_sequences[:, :-1]
    y = padded_input_sequences[:, -1]

    from tensorflow.keras.utils import to_categorical
    y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, 100,input_length=max_len - 1))
    model.add(LSTM(150))
    model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, epochs=100)

    model.save('next_word_model.keras')
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


model = load_model("next_word_model.keras")
with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)

input_sequences = []
for sentence in data.split('\n'):
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:i + 1])

max_len = max([len(x) for x in input_sequences])

def jadu(model, tokenizer, text, num_words):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == np.argmax(predicted):
                predicted_word = word
                break
        text += " " + predicted_word
    return text


st.title("Next Word Prediction")

input_text = st.text_input("Enter the initial text:")
num_words = st.slider('Number of Words:', min_value=1, max_value=50, step=1)

if st.button("Predict"):
    if input_text:
        result = jadu(model, tokenizer, input_text, num_words)
        st.write(result)
    else:
        st.write("Please enter some initial text.")
