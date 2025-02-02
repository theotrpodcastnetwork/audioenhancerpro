import numpy as np
import soundfile as sf
import streamlit as st
import librosa
import librosa.display
import noisereduce as nr
from scipy.signal import butter, filtfilt, lfilter
import pyloudnorm as pyln
from pydub import AudioSegment

def reduce_noise(audio, sr, noise_duration=0.5, prop_decrease=1.0):
    """
    Reduce background noise using noisereduce.
    
    If noise_duration is set to a positive value, it extracts that duration
    from the beginning of the audio as a noise sample. If noise_duration is 0,
    it lets noisereduce automatically estimate the noise profile.
    
    Parameters:
        audio (np.array): Audio signal.
        sr (int): Sampling rate.
        noise_duration (float): Duration (in seconds) of noise sample to extract.
                                Set to 0 for automatic noise estimation.
        prop_decrease (float): Controls the aggressiveness of noise reduction.
    Returns:
        np.array: Denoised audio signal.
    """
    if noise_duration > 0:
        # Extract a noise clip from the beginning of the audio
        n_samples = int(noise_duration * sr)
        noise_clip = audio[:n_samples]
        return nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip, prop_decrease=prop_decrease)
    else:
        # Automatic noise estimation without a manually defined noise sample
        return nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease)

def parametric_eq(audio, sr, f0=1500, Q=1.0, gain_db=6.0):
    A = 10 ** (gain_db / 40)
    omega = 2 * np.pi * f0 / sr
    alpha = np.sin(omega) / (2 * Q)
    
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(omega)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(omega)
    a2 = 1 - alpha / A
    
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
    
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
    return pyln.normalize.loudness(audio, loudness, -20.0)

def enhance_audio(input_file, output_file, cutoff=12000, noise_prop=1.0, eq_gain=6.0, noise_duration=0.0):
    """
    Enhance the audio by applying noise reduction, EQ, filtering, normalization,
    compression, and silence trimming.
    
    Parameters:
        input_file (str): Path to input audio file.
        output_file (str): Path to save enhanced audio.
        cutoff (float): Low-pass filter cutoff frequency (Hz).
        noise_prop (float): Noise reduction aggressiveness.
        eq_gain (float): Midrange EQ boost in dB.
        noise_duration (float): Duration (in seconds) to sample for background noise.
                                Set to 0 to auto-detect noise.
    """
    audio, sr = librosa.load(input_file, sr=None)
    
    # Apply noise reduction (automatic if noise_duration == 0)
    audio = reduce_noise(audio, sr, noise_duration=noise_duration, prop_decrease=noise_prop)
    
    audio = parametric_eq(audio, sr, gain_db=eq_gain)
    audio = butter_lowpass_filter(audio, cutoff=cutoff, fs=sr)
    audio = normalize_audio(audio)
    audio = compress_audio(audio, sr)
    audio, _ = librosa.effects.trim(audio)
    
    sf.write(output_file, audio, sr, format='WAV', subtype='PCM_16')
    print(f"Enhanced audio saved to {output_file}")

def main():
    st.title("Audio Enhancer Pro")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
    
    cutoff = st.slider("Low-Pass Filter Cutoff Frequency (Hz)", 
                       min_value=8000, max_value=20000, value=12000, step=500)
    noise_prop = st.slider("Noise Reduction Aggressiveness (0.1 = low, 1.0 = high)", 
                           min_value=0.1, max_value=1.0, value=0.8, step=0.1)
    eq_gain = st.slider("Midrange EQ Boost (dB)", 
                        min_value=0.0, max_value=12.0, value=6.0, step=0.5)
    
    # Option for automatic noise estimation: setting duration to 0 triggers auto-detection.
    noise_duration = st.number_input("Noise Sample Duration (seconds, 0 for auto)", 
                                     min_value=0.0, max_value=5.0, value=0.0, step=0.1,
                                     help="Set to 0 to let the algorithm automatically estimate background noise.")
    
    if uploaded_file is not None:
        input_file = "temp_input.wav"
        output_file = "enhanced_output.wav"
        
        with open(input_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(input_file, format="audio/wav")
        
        if st.button("Enhance Audio"):
            enhance_audio(input_file, output_file, cutoff=cutoff, noise_prop=noise_prop, 
                          eq_gain=eq_gain, noise_duration=noise_duration)
            st.success("Audio enhancement complete!")
            st.audio(output_file, format="audio/wav")

if __name__ == '__main__':
    main()
