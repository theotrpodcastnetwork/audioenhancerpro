import numpy as np
import soundfile as sf
import streamlit as st
import librosa
import librosa.display
import noisereduce as nr
from scipy.signal import butter, filtfilt, lfilter
import pyloudnorm as pyln
from pydub import AudioSegment
import torch

# =============================================================================
# Load Silero VAD model (cached to avoid reloading on every run)
# =============================================================================
@st.cache_resource
def load_vad_model():
    """
    Load the Silero VAD model and return the model along with the get_speech_timestamps function.
    This function uses torch.hub to load the model from the repository.
    """
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=True)
    # utils is a tuple; the first element is the get_speech_timestamps function
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps

def apply_vad(audio, sr, model, get_speech_timestamps):
    """
    Apply Silero VAD to detect speech segments in the audio.
    
    Parameters:
        audio (np.array): 1D audio signal.
        sr (int): Sampling rate of the audio.
        model: Loaded Silero VAD model.
        get_speech_timestamps: Function to obtain speech segments.
        
    Returns:
        np.array: Concatenated speech segments from the input audio.
    """
    # Get timestamps for speech segments.
    # Each element in speech_timestamps is a dictionary with keys 'start' and 'end' (sample indices).
    speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=sr)
    
    if len(speech_timestamps) == 0:
        # If no speech is detected, return the original audio
        return audio
    
    # Concatenate all detected speech segments into one array
    speech_audio = np.concatenate([audio[segment['start']:segment['end']] for segment in speech_timestamps])
    return speech_audio

# =============================================================================
# Existing enhancement functions
# =============================================================================
def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

def parametric_eq(audio, sr):
    # Boost midrange frequencies for clearer speech using a bandpass filter.
    b, a = butter(2, [300/(sr/2), 3000/(sr/2)], btype='band')
    return filtfilt(b, a, audio)

def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def compress_audio(audio, sr):
    meter = pyln.Meter(sr) 
    loudness = meter.integrated_loudness(audio)
    return pyln.normalize.loudness(audio, loudness, -20.0)  # Target -20 LUFS

# =============================================================================
# Main audio enhancement pipeline with optional Silero VAD preprocessing
# =============================================================================
def enhance_audio(input_file, output_file, apply_vad_flag=False):
    # Load audio with its original sampling rate
    audio, sr = librosa.load(input_file, sr=None)
    
    # If enabled, apply Silero VAD to extract only the speech segments (trim silence/background)
    if apply_vad_flag:
        model, get_speech_timestamps = load_vad_model()
        audio = apply_vad(audio, sr, model, get_speech_timestamps)
    
    # Apply the enhancement processing steps:
    audio = reduce_noise(audio, sr)
    audio = parametric_eq(audio, sr)
    audio = butter_lowpass_filter(audio, cutoff=8000, fs=sr)
    audio = normalize_audio(audio)
    audio = compress_audio(audio, sr)
    
    # Optionally, further trim any extra silence from the beginning or end
    audio, _ = librosa.effects.trim(audio)
    
    # Save the enhanced audio as a 16-bit PCM WAV file
    sf.write(output_file, audio, sr, format='WAV', subtype='PCM_16')
    print(f'Enhanced audio saved to {output_file}')

# =============================================================================
# Streamlit app interface
# =============================================================================
def main():
    st.title("Audio Enhancer Pro")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
    
    # Checkbox to optionally apply Silero VAD for speech detection/trimming
    apply_vad_flag = st.checkbox("Apply Silero VAD (Trim Silence / Extract Speech)", value=False)
    
    if uploaded_file is not None:
        input_file = "temp_input.wav"
        output_file = "enhanced_output.wav"
        
        # Save the uploaded file to a temporary location
        with open(input_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(input_file, format='audio/wav')
        
        if st.button("Enhance Audio"):
            enhance_audio(input_file, output_file, apply_vad_flag=apply_vad_flag)
            st.success("Audio enhancement complete!")
            st.audio(output_file, format='audio/wav')

if __name__ == '__main__':
    main()
