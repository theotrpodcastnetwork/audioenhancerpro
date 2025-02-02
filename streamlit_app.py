import numpy as np
import soundfile as sf
import streamlit as st
import librosa
import librosa.display
import noisereduce as nr
from scipy.signal import butter, filtfilt, lfilter
import pyloudnorm as pyln
from pydub import AudioSegment

def reduce_noise(audio, sr, prop_decrease=1.0):
    """
    Reduce noise using the noisereduce library.
    
    Parameters:
        audio (np.array): Audio signal.
        sr (int): Sampling rate.
        prop_decrease (float): Controls the aggressiveness of noise reduction.
                               Lower values result in less aggressive reduction.
    Returns:
        np.array: Denoised audio signal.
    """
    return nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease)

def parametric_eq(audio, sr, f0=1500, Q=1.0, gain_db=6.0):
    """
    Applies a peaking (bell) filter to boost midrange frequencies.
    
    This filter boosts frequencies around f0 (default 1500 Hz) without cutting out high frequencies.
    
    Parameters:
        audio (np.array): Audio signal.
        sr (int): Sampling rate.
        f0 (float): Center frequency for the boost (Hz).
        Q (float): Quality factor (controls the bandwidth of the boost).
        gain_db (float): Boost gain in decibels.
        
    Returns:
        np.array: Equalized audio signal.
    """
    # Calculate filter coefficients based on the Audio EQ Cookbook formulas
    A = 10**(gain_db / 40)  # A = sqrt(10^(gain/20))
    omega = 2 * np.pi * f0 / sr
    alpha = np.sin(omega) / (2 * Q)
    
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(omega)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(omega)
    a2 = 1 - alpha / A
    
    # Normalize the coefficients
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
    
    # Apply the filter using zero-phase filtering (filtfilt)
    return filtfilt(b, a, audio)

def normalize_audio(audio):
    """Normalize audio so its maximum absolute amplitude is 1."""
    return audio / np.max(np.abs(audio))

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a Butterworth low-pass filter.
    
    Parameters:
        data (np.array): Audio signal.
        cutoff (float): Cutoff frequency (Hz).
        fs (int): Sampling rate.
        order (int): Filter order.
        
    Returns:
        np.array: Filtered audio signal.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def compress_audio(audio, sr):
    """
    Compress (normalize) audio to a target loudness of -20 LUFS.
    
    Parameters:
        audio (np.array): Audio signal.
        sr (int): Sampling rate.
        
    Returns:
        np.array: Loudness-normalized audio signal.
    """
    meter = pyln.Meter(sr) 
    loudness = meter.integrated_loudness(audio)
    return pyln.normalize.loudness(audio, loudness, -20.0)  # Target -20 LUFS

def enhance_audio(input_file, output_file, cutoff=12000, noise_prop=1.0, eq_gain=6.0):
    """
    Enhance the input audio by applying noise reduction, EQ, low-pass filtering,
    normalization, compression, and trimming.
    
    Parameters:
        input_file (str): Path to the input audio file.
        output_file (str): Path for the output enhanced audio.
        cutoff (float): Low-pass filter cutoff frequency in Hz.
        noise_prop (float): Aggressiveness of noise reduction.
        eq_gain (float): Midrange boost gain in dB.
    """
    # Load audio using its original sampling rate
    audio, sr = librosa.load(input_file, sr=None)
    
    # Apply processing steps:
    audio = reduce_noise(audio, sr, prop_decrease=noise_prop)
    audio = parametric_eq(audio, sr, gain_db=eq_gain)  # Boost midrange without cutting high frequencies
    audio = butter_lowpass_filter(audio, cutoff=cutoff, fs=sr)
    audio = normalize_audio(audio)
    audio = compress_audio(audio, sr)
    
    # Trim extra silence from beginning and end
    audio, _ = librosa.effects.trim(audio)
    
    # Save the enhanced audio as a 16-bit PCM WAV file
    sf.write(output_file, audio, sr, format='WAV', subtype='PCM_16')
    print(f"Enhanced audio saved to {output_file}")

def main():
    st.title("Audio Enhancer Pro")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
    
    # Allow user to adjust parameters
    cutoff = st.slider("Select low-pass filter cutoff frequency (Hz)", 
                       min_value=8000, max_value=20000, value=12000, step=500)
    noise_prop = st.slider("Noise Reduction Aggressiveness (0.1 = low, 1.0 = high)", 
                           min_value=0.1, max_value=1.0, value=0.8, step=0.1)
    eq_gain = st.slider("Midrange EQ Boost (dB)", 
                        min_value=0.0, max_value=12.0, value=6.0, step=0.5)
    
    if uploaded_file is not None:
        input_file = "temp_input.wav"
        output_file = "enhanced_output.wav"
        
        # Save the uploaded file temporarily
        with open(input_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(input_file, format="audio/wav")
        
        if st.button("Enhance Audio"):
            enhance_audio(input_file, output_file, cutoff=cutoff, noise_prop=noise_prop, eq_gain=eq_gain)
            st.success("Audio enhancement complete!")
            st.audio(output_file, format="audio/wav")

if __name__ == '__main__':
    main()
