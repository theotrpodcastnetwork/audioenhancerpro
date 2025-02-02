import numpy as np
import soundfile as sf
import streamlit as st
import librosa
import librosa.display
import noisereduce as nr
from scipy.signal import butter, filtfilt, lfilter
import pyloudnorm as pyln
from pydub import AudioSegment

def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

def parametric_eq(audio, sr):
    # Boost midrange frequencies for clearer speech
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

def enhance_audio(input_file, output_file):
    audio, sr = librosa.load(input_file, sr=None)
    
    # Apply enhancements
    audio = reduce_noise(audio, sr)
    audio = parametric_eq(audio, sr)
    audio = butter_lowpass_filter(audio, cutoff=8000, fs=sr)
    audio = normalize_audio(audio)
    audio = compress_audio(audio, sr)
    
    # Trim extra silence
    audio, _ = librosa.effects.trim(audio)
    
    # Save output with correct format
    sf.write(output_file, audio, sr, format='WAV', subtype='PCM_16')
    print(f'Enhanced audio saved to {output_file}')

def main():
    st.title("Audio Enhancer Pro")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
    
    if uploaded_file is not None:
        input_file = "temp_input.wav"
        output_file = "enhanced_output.wav"
        
        with open(input_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(input_file, format='audio/wav')
        
        if st.button("Enhance Audio"):
            enhance_audio(input_file, output_file)
            st.success("Audio enhancement complete!")
            st.audio(output_file, format='audio/wav')

if __name__ == '__main__':
    main() 
