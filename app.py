import string
import numpy as np
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# FAQ dataset
faqs = [
    {
        "question": "What is artificial intelligence?",
        "answer": "Artificial Intelligence, or AI, is technology that enables computers to perform tasks that normally require human intelligence."
    },
    {
        "question": "What is machine learning?",
        "answer": "Machine Learning is a branch of AI that allows computers to learn patterns from data and make predictions or decisions."
    },
    {
        "question": "What is deep learning?",
        "answer": "Deep Learning is a type of machine learning that uses neural networks with multiple layers to learn complex patterns."
    },
    {
        "question": "What is NLP?",
        "answer": "NLP stands for Natural Language Processing. It allows computers to understand, process, and generate human language."
    },
    {
        "question": "What is a chatbot?",
        "answer": "A chatbot is a software application that communicates with users through text or voice and provides automated responses."
    },
    {
        "question": "How does AI work?",
        "answer": "AI systems use algorithms, data, and computational methods to identify patterns and perform tasks or make predictions."
    },
    {
        "question": "What is supervised learning?",
        "answer": "Supervised learning is a machine learning approach where a model learns from labeled training data."
    },
    {
        "question": "What is unsupervised learning?",
        "answer": "Unsupervised learning finds patterns or structures in data without using labeled outputs."
    },
    {
        "question": "What is a neural network?",
        "answer": "A neural network is a machine learning model inspired by the human brain that consists of interconnected processing units called neurons."
    },
    {
        "question": "What is computer vision?",
        "answer": "Computer vision is a field of AI that enables computers to understand and analyze images and videos."
    },
    {
        "question": "What programming languages are used for AI?",
        "answer": "Python is one of the most commonly used programming languages for AI. Other languages include Java, C++, and R."
    },
    {
        "question": "What is training data?",
        "answer": "Training data is a collection of examples used to teach a machine learning model how to identify patterns."
    },
    {
        "question": "What is an AI model?",
        "answer": "An AI model is a computational system trained using data to perform tasks such as classification, prediction, or language processing."
    },
    {
        "question": "What is generative AI?",
        "answer": "Generative AI is a type of artificial intelligence that can create new content such as text, images, audio, or code."
    },
    {
        "question": "What is the difference between AI and ML?",
        "answer": "AI is the broader concept of machines performing intelligent tasks, while Machine Learning is a subset of AI that learns from data."
    }
]


questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]


# NLP preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = nltk.word_tokenize(text)

    words = [
        word for word in words
        if word.isalpha() and word not in stop_words
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


# Process FAQ questions
processed_questions = [
    preprocess_text(question)
    for question in questions
]


# TF-IDF
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(
    processed_questions
)


# Find best answer
def get_answer(user_question):

    if not user_question.strip():
        return "Please enter a question."

    processed_input = preprocess_text(user_question)

    user_vector = vectorizer.transform(
        [processed_input]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )[0]

    best_match_index = int(
        np.argmax(similarity_scores)
    )

    best_score = similarity_scores[best_match_index]

    if best_score < 0.15:
        return "I'm sorry, I couldn't find a suitable answer. Please try asking your question in a different way."

    return answers[best_match_index]


# Streamlit UI
st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI FAQ Chatbot")

st.write(
    "Ask questions about Artificial Intelligence, "
    "Machine Learning, NLP, and related topics."
)

user_question = st.text_input(
    "Your Question",
    placeholder="Ask your question here..."
)

if st.button("Ask"):
    if user_question:
        answer = get_answer(user_question)

        st.subheader("Chatbot Response")
        st.success(answer)

    else:
        st.warning("Please enter a question.")