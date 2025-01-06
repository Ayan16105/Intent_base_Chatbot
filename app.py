import pickle
import random
import ssl
import nltk
import re  # Importing the re library for regex
import streamlit as st
import string
import json
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Bypass punkt dependency by using a simple regex tokenizer
nltk.data.path.append(r'C:\Users\HP\AppData\Roaming\nltk_data')
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords')
nltk.download('wordnet')  # Download WordNet corpus to fix the lemmatization issue

# Initialize stopwords
stopwords_set = set(stopwords.words('english'))

# Simple regex-based tokenizer
def regex_tokenize(text):
    # Regex pattern to match words, including words with apostrophes (e.g., "don't", "I'm")
    return re.findall(r'\b\w+\b', text.lower())

def data_preprocess(text):
    lemmatizer = WordNetLemmatizer()
    tokens = regex_tokenize(text)  # Using the regex tokenizer
    process_token = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in string.punctuation and word not in stopwords_set
    ]
    return process_token

def load_model_and_vect():
    # Load the trained model and vectorizer
    with open('chatbot_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('chatbot_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Load model and vectorizer at the start
model, vectorizer = load_model_and_vect()

# Load intents
with open('intents.json') as f:
    df = json.load(f)

def chatbot(user_inp):
    # Process the user input
    user_inp = data_preprocess(user_inp)
    user_inp_vectorized = vectorizer.transform([' '.join(user_inp)])
    
    # Predict the intent
    tag = model.predict(user_inp_vectorized)[0]
    
    # Find the response
    for intent in df:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "Sorry, I didn't understand that."

# Streamlit application
st.title("Intent-Based ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me here...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})    

    # Get chatbot response
    ans = chatbot(prompt)
    with st.chat_message("assistant"):
        st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
