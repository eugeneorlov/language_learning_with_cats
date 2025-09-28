import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import soundfile as sf
from pathlib import Path
import pandas as pd
import io
from openai import OpenAI

# --- Initialize OpenAI client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Language Bot",
                   page_icon="🐱🎤", layout="centered")
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
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# --- Record & Process Audio on Button Click ---
if ctx.audio_receiver:
    if st.button("🎙️ Transcribe & Chat"):
        wav_path = Path("input.wav")

        # --- Step 1: Collect frames and save proper WAV ---
        frames = [frame.to_ndarray()
                  for frame in ctx.audio_receiver.get_frames(timeout=1)]
        if len(frames) == 0:
            st.warning("No audio frames captured. Please try again.")
        else:
            audio_data = np.hstack(frames)

            # Transpose if shape is (channels, samples)
            if audio_data.shape[0] < audio_data.shape[1]:
                audio_data = audio_data.T

            # Convert to float32 [-1,1]
            if audio_data.dtype != np.float32:
                if np.issubdtype(audio_data.dtype, np.integer):
                    audio_data = audio_data.astype(
                        np.float32) / np.iinfo(audio_data.dtype).max
                else:
                    audio_data = audio_data.astype(np.float32)

            # Save WAV in PCM_16
            sf.write(wav_path, audio_data, samplerate=48000, subtype="PCM_16")

            # --- Step 2: Transcribe Speech ---
            with open(wav_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=f
                )
            user_text = transcription.text
            st.write(f"**You said:** {user_text}")

            # --- Step 3: Chat with GPT ---
            prompt_system = f"""
You are a helpful language tutor. The learner is practicing {language}.
1. Correct their mistakes politely.
2. Respond in {language}.
3. If feedback mode is enabled, explain grammar and vocabulary.
"""
            prompt_user = f"Learner said: {user_text}\nFeedback mode: {feedback_mode}"

            chat_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user}
                ]
            )
            assistant_text = chat_response.choices[0].message.content
            st.write(f"**Assistant ({language}):** {assistant_text}")

            # --- Step 4: Convert GPT reply to Speech (TTS) ---
            speech_file = Path("output.mp3")
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=assistant_text,
            ) as tts_stream:
                tts_stream.stream_to_file(speech_file)

            st.audio(str(speech_file), format="audio/mp3")

            # --- Step 5: Log Conversation ---
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
