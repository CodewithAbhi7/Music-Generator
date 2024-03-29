!pip install transformers
import streamlit as st
from transformers import pipeline
import scipy
import os
import wave

# Load MusicGen pipeline
synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")

MUSIC_FOLDER = "../music"  # Name of the folder to save audio files

def create_audio(user_input, file_path):
        with st.spinner("Generating audio..."):
            music = synthesiser(user_input, forward_params={"do_sample": True})
            audio_data = music["audio"]
            sampling_rate = music["sampling_rate"]
            scipy.io.wavfile.write(file_path, rate=sampling_rate, data=audio_data)
            st.success(f"Audio generated and saved as {file_path}.")

def app():
    st.title("MusicGen Streamlit App")

    user_input = st.text_area("Enter music description (e.g., 'lo-fi music with a soothing melody'):")
    
    # User input for the desired filename
    filename = st.text_input("Enter the desired music filename (without extension):", "musicgen_out")

    if st.button("Generate Audio"):
        if user_input:

            # Ensure the 'music' folder exists
            os.makedirs(MUSIC_FOLDER, exist_ok=True)
            
            file_path = os.path.join(MUSIC_FOLDER, f"{filename}.wav")
            
            create_audio(user_input, file_path)

            # Play audio in the app
            st.audio(file_path, format="audio/wav")

    
    st.header("Play generated Audio Files")

    # Display a list of available audio files in the 'music' folder
    audio_files = os.listdir(MUSIC_FOLDER)
    selected_file = st.selectbox("Select an audio file to play:", audio_files, index=0)

    if selected_file:
        selected_file_path = os.path.join(MUSIC_FOLDER, selected_file)
        st.audio(selected_file_path, format="audio/wav", start_time=0)


if __name__ == "__main__":
    app()
