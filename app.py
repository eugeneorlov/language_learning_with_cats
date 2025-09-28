import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import soundfile as sf
from pathlib import Path
import pandas as pd
import io
from openai import OpenAI
from datetime import datetime

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Language Bot",
                   page_icon="🐱🎤", layout="centered")
st.title("🐱🎤 AI Language Learning Assistant")

# --- Language and Feedback ---
language = st.selectbox("Choose your practice language:", [
                        "German", "English", "French", "Spanish"])
feedback_mode = st.checkbox(
    "Enable feedback (grammar + vocabulary explanations)", value=True)

# --- Conversation Log ---
if "log" not in st.session_state:
    st.session_state.log = []

# --- Chat Input ---
st.subheader("💬 Chat with AI")
chat_input = st.text_input("Type your message here and press Enter:")


def process_user_input(user_text):
    """Send user text to GPT, get assistant response, play TTS, log conversation."""
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

    # TTS
    speech_file = Path("output.mp3")
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=assistant_text,
    ) as tts_stream:
        tts_stream.stream_to_file(speech_file)

    st.audio(str(speech_file), format="audio/mp3")

    # Log
    st.session_state.log.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Language": language,
        "User": user_text,
        "Assistant": assistant_text,
        "FeedbackMode": feedback_mode
    })

    return assistant_text


# --- Handle chat input ---
if chat_input:
    process_user_input(chat_input)

# --- Voice Interaction ---
st.subheader("🎤 Voice Conversation")
col1, col2 = st.columns(2)
start_button = col1.button("Start Recording")
stop_button = col2.button("Stop & Send Audio")

if "recording" not in st.session_state:
    st.session_state.recording = False

if start_button:
    st.session_state.recording = True

if stop_button:
    st.session_state.recording = False
    if "webrtc_ctx" in st.session_state and st.session_state.webrtc_ctx.audio_receiver:
        frames = [frame.to_ndarray(
        ) for frame in st.session_state.webrtc_ctx.audio_receiver.get_frames(timeout=1)]
        if len(frames) > 0:
            audio_data = np.hstack(frames)
            if audio_data.shape[0] < audio_data.shape[1]:
                audio_data = audio_data.T
            if audio_data.dtype != np.float32:
                if np.issubdtype(audio_data.dtype, np.integer):
                    audio_data = audio_data.astype(
                        np.float32) / np.iinfo(audio_data.dtype).max
                else:
                    audio_data = audio_data.astype(np.float32)
            wav_path = Path("input.wav")
            sf.write(wav_path, audio_data, samplerate=48000, subtype="PCM_16")

            # Transcribe audio
            with open(wav_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=f
                )
            user_text = transcription.text
            st.write(f"**You said:** {user_text}")

            # Process and log
            process_user_input(user_text)

# --- WebRTC streamer ---
if st.session_state.recording:
    st.session_state.webrtc_ctx = webrtc_streamer(
        key="voice-demo",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

# --- Display full conversation history ---
st.subheader("📝 Conversation History")
for entry in st.session_state.log:
    st.markdown(
        f"**[{entry['Timestamp']}] You ({entry['Language']}):** {entry['User']}")
    st.markdown(f"**Assistant:** {entry['Assistant']}\n---")

# --- Download CSV ---
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
