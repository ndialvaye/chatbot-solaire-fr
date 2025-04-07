import streamlit as st
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

def load_text():
    if not os.path.exists("corpus_chatbot_fr.txt"):
        st.error("Le fichier corpus_chatbot_fr.txt est introuvable.")
        st.stop()
    with open("corpus_chatbot_fr.txt", "r", encoding="utf-8") as f:
        return f.read()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words("french"))
    return ' '.join([w for w in words if w not in stop_words])

def get_best_response(user_input, sentences, processed_sentences):
    cleaned_input = preprocess(user_input)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(processed_sentences + [cleaned_input])
    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1])
    idx = np.argmax(cosine_sim)
    score = cosine_sim[0][idx]
    if score < 0.1:
        return "Je ne suis pas sûr de comprendre. Peux-tu reformuler ?"
    return sentences[idx]

def main():
    st.set_page_config(page_title="Chatbot Solaire 🇫🇷", page_icon="🌞")
    st.title("🤖 Chatbot sur le Système Solaire (Français)")
    st.write("Pose une question sur l'espace ou les planètes !")

    text = load_text()
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    processed_sentences = [preprocess(s) for s in sentences]

    with st.form("chat_form"):
        user_input = st.text_input("Votre question :")
        submit = st.form_submit_button("Envoyer")

    if submit and user_input:
        response = get_best_response(user_input, sentences, processed_sentences)
        st.success("Réponse :")
        st.write(response)

if __name__ == "__main__":
    main()
