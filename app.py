import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
from pathlib import Path
import pandas as pd
import io
from openai import OpenAI

# --- Initialize OpenAI client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set page config
st.set_page_config(page_title="Language Learning Assistant",
                   page_icon="🐱🎤", layout="centered")

# Header with image
st.title("🐱🎤 AI Language Learning Assistant")
st.image("resources/language_learning_with_cats.png", use_container_width=True)

# --- Language Selection ---
language = st.selectbox("Choose your practice language:", [
                        "German", "French", "Spanish"])

# --- Feedback Mode ---
feedback_mode = st.checkbox(
    "Enable feedback (grammar + vocabulary explanations)", value=True)

# --- Conversation Log ---
if "log" not in st.session_state:
    st.session_state.log = []

# --- WebRTC Audio Streamer ---
ctx = webrtc_streamer(
    key="speech-demo",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,  # larger buffer to avoid queue overflow
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# --- Record & Process Audio on Button Click ---
if ctx.audio_receiver:
    if st.button("🎙️ Transcribe & Chat"):
        # Save recorded audio to WAV
        wav_path = Path("input.wav")
        with open(wav_path, "wb") as f:
            for frame in ctx.audio_receiver.get_frames(timeout=1):
                f.write(frame.to_ndarray().tobytes())

        # --- Step 1: Transcribe Speech ---
        with open(wav_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f
            )
        user_text = transcription.text
        st.write(f"**You said:** {user_text}")

        # --- Step 2: Chat with GPT (correction + feedback) ---
        prompt = f"""
You are a helpful language tutor. The learner is practicing {language}.
1. Correct their mistakes politely.
2. Respond in {language}.
3. If feedback mode is enabled, explain grammar and vocabulary.
Learner said: "{user_text}"
Feedback mode: {feedback_mode}
"""
        chat_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        assistant_text = chat_response.choices[0].message.content
        st.write(f"**Assistant ({language}):** {assistant_text}")

        # --- Step 3: Convert GPT reply to Speech ---
        speech_file = Path("output.mp3")
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=assistant_text,
        ) as r:
            r.stream_to_file(speech_file)

        st.audio(str(speech_file), format="audio/mp3")

        # --- Step 4: Log Conversation ---
        st.session_state.log.append({
            "Language": language,
            "User": user_text,
            "Assistant": assistant_text,
            "FeedbackMode": feedback_mode
        })

# --- Downloadable CSV Log ---
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Conversation Log (CSV)",
        data=csv_buffer.getvalue(),
        file_name="language_learning_log.csv",
        mime="text/csv"
    )
