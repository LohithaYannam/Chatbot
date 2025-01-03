import os
import nltk
import ssl
import json
import random
import csv
import datetime
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Handle SSL certificates if necessary
ssl._create_default_https_context = ssl._create_unverified_context

# Make sure NLTK resources are available
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = "intents.json"
with open(file_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    return "I'm sorry, I didn't understand that."

counter = 0

def main():
    global counter
    st.title("Intents of Chatbot using NLP")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)  # Use selectbox instead of selection

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the chatbot")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)

            # Get the chatbot response
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save the user input and the chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        
        # Display conversation history
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond.")

        st.subheader("Project Overview:")
        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm are used to train the chatbot on labelled intents.
        2. The chatbot interface is built using the Streamlit web framework.
        """)

        st.subheader("Dataset:")
        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The components are:
        - **Intents**: The intent of the user input (e.g., "greeting", "budget", "about").
        - **Entities**: The entities extracted from the user input (e.g., "Hi", "How do I create a budget?").
        - **Text**: The user input text.
        """)

        st.subheader("Streamlit Chatbot Interface:")
        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box and displays responses from the chatbot.")

        st.subheader("Conclusion:")
        st.write("In this project, a chatbot is built that can understand and respond to user inputs based on predefined intents.")

if __name__ == "__main__":
    main()


