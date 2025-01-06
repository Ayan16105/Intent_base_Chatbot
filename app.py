import pickle
import random
import ssl
import nltk
import streamlit as st
ssl._create_default_https_context=ssl._create_unverified_context
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import json
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


stopwords = set(stopwords.words('english'))
def data_preprocess(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    process_token = []
    for word in tokens:
        if word not in string.punctuation and word not in stopwords:
            lem_word = lemmatizer.lemmatize(word)
            process_token.append(lem_word)
    
    return process_token
def load_model_and_vect():
    with open('chatbot_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('chatbot_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


with open('intents.json') as f:
    df = json.load(f)


tags = []

for intent in df:
    for ptrn in intent['patterns']:
        tags.append(intent['tag'])

def chatbot(user_inp):
    model, vectorizer = load_model_and_vect()
    user_inp = data_preprocess(user_inp)
    user_inp_vectorized = vectorizer.transform([' '.join(user_inp)])
    tag = model.predict(user_inp_vectorized)[0]
    for intent in df:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    return "Sorry, I didn't understand that."
def chatbot(user_inp):
    user_inp = vectorizer.transform([])
    tag = model.predict(user_inp_vectorized)[0]
    for intent in df:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    return "Sorry, I didn't understand that."



st.title("Intent Base ChatBot")


if "messages" not in st.session_state:
    st.session_state.messages=[]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Ask me here...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})    

    ans = chatbot(prompt)
    with st.chat_message("assistant"):
        st.markdown(ans)
    st.session_state.messages.append({"role":"assistant","content":ans})
            

