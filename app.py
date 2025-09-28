import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import openai
import pandas as pd
from pathlib import Path
import io

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set page config
st.set_page_config(page_title="Language Bot",
                   page_icon="🐱🎤", layout="centered")

# Header with image
st.image("resources/language_learning_with_cats.png", use_container_width=True)
st.title("🐱🎤 AI Language Learning Assistant")

# --- 1. Language Selection ---
language = st.selectbox("Choose your practice language:", [
                        "German", "French", "Spanish"])

# --- 2. Feedback Mode Toggle ---
feedback_mode = st.checkbox(
    "Enable feedback (grammar + vocabulary explanations)", value=True)

# --- Conversation Log (kept in session state) ---
if "log" not in st.session_state:
    st.session_state.log = []

# --- WebRTC audio input ---
ctx = webrtc_streamer(
    key="speech-demo",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if ctx.audio_receiver:
    if st.button("🎙️ Transcribe & Chat"):
        # Save recorded audio to file
        wav_path = Path("input.wav")
        with open(wav_path, "wb") as f:
            for frame in ctx.audio_receiver.get_frames(timeout=1):
                f.write(frame.to_ndarray().tobytes())

        # --- Step 1: Transcribe Speech ---
        with open(wav_path, "rb") as f:
            transcription = openai.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f
            )
        user_text = transcription.text
        st.write(f"**You said:** {user_text}")

        # --- Step 2: Send to GPT with error correction + feedback ---
        prompt = f"""
You are a helpful language tutor. The learner is practicing {language}.
1. Correct their mistakes politely.
2. Respond in {language}.
3. If feedback mode is enabled, explain grammar and vocabulary.
Learner said: "{user_text}"
Feedback mode: {feedback_mode}
"""
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        assistant_text = response.choices[0].message.content
        st.write(f"**Assistant ({language}):** {assistant_text}")

        # --- Step 3: Convert GPT reply to Speech ---
        speech_file = Path("output.mp3")
        with openai.audio.speech.with_streaming_response.create(
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

# --- Progress Tracking: CSV Download ---
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
