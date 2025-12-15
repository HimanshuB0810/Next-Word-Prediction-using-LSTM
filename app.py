import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model=load_model("next_word_lstm.h5")

# load the tokenizer
with open("tokenizer.pkl","rb") as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):] # Ensure the sequence length matches max_sequence_len
    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicated=model.predict(token_list,verbose=0)
    predicated_word_index=np.argmax(predicated,axis=1) #Returns the indices of the maximum values along an axis.
    for word,index in tokenizer.word_index.items():
        if index==predicated_word_index:
            return word
    return None

# Streamlit app
st.title("Next Word Predication With LSTM")
input_text=st.text_input("Enter the Sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len=model.input_shape[1]+1 # model.input_shape will be (None, 13)
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next Word: {next_word}")
