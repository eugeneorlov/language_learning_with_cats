import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Set page config
st.set_page_config(page_title="Language Bot", page_icon="🐱", layout="centered")

# Header with image
st.image("resources/language_learning_with_cats.png", use_container_width=True)
st.title("🐱 Language Bot")
st.subheader("Your AI-powered assistant for learning languages")

# Language picker
language = st.radio("Pick language", ["German", "French", "Spanish"])

# Subtitle
st.subheader("What do you want to learn?")

# Define prompts
prompts = {
    "Vocabulary": f"Create a vocabulary quiz in {language} focusing on job interview topics. The quiz should be Fill-in-the-Blanks type and include 10 questions. Provide an answer key.",
    "Grammar": f"Explain the {language} basic grammar concepts.",
    "Conversational practice": f"Let’s role-play in {language}. Please start the conversation."
}

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Reset and set new context based on button click
col1, col2, col3 = st.columns(3)
selected_task = None
with col1:
    if st.button("Vocabulary"):
        st.session_state.messages = [
            {"role": "system", "content": prompts["Vocabulary"]},
            {"role": "assistant", "content": "📘 Context set: Vocabulary practice."}
        ]
        selected_task = "Vocabulary"
with col2:
    if st.button("Grammar"):
        st.session_state.messages = [
            {"role": "system", "content": prompts["Grammar"]},
            {"role": "assistant", "content": "✍️ Context set: Grammar practice."}
        ]
        selected_task = "Grammar"
with col3:
    if st.button("Conversational practice"):
        st.session_state.messages = [
            {"role": "system", "content": prompts["Conversational practice"]},
            {"role": "assistant", "content": "💬 Context set: Conversational practice."}
        ]
        selected_task = "Conversational practice"

st.divider()

# Chat UI
st.subheader("💬 Chat with Language Bot")

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # Call OpenAI API (new v1.0+ interface)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages
    )

    assistant_msg = response.choices[0].message.content
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_msg})
    st.chat_message("assistant").markdown(assistant_msg)
