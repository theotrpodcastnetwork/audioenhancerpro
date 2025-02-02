import numpy as np
import soundfile as sf
import streamlit as st
import librosa
import noisereduce as nr
from scipy.signal import butter, filtfilt, lfilter
import pyloudnorm as pyln
import webrtcvad

# ------------------------------------------------------------------
# WebRTC VAD Implementation
# ------------------------------------------------------------------
def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    Generates audio frames from a 1-D numpy array.
    Each frame is of duration 'frame_duration_ms' milliseconds.
    """
    n_samples = int(sample_rate * frame_duration_ms / 1000)
    offset = 0
    while offset + n_samples <= len(audio):
        yield audio[offset:offset + n_samples]
        offset += n_samples

def apply_webrtc_vad(audio, sample_rate, frame_duration_ms=30, vad_mode=3):
    """
    Apply WebRTC VAD to extract speech segments from audio.
    
    Parameters:
        audio (np.array): A 1D numpy array containing float audio samples (range -1 to 1).
        sample_rate (int): The sampling rate of the audio.
        frame_duration_ms (int): Duration (in milliseconds) for each frame (10, 20, or 30 ms).
        vad_mode (int): Aggressiveness of the VAD (0 is least, 3 is most aggressive).
        
    Returns:
        np.array: Concatenated speech frames as a float32 numpy array (range -1 to 1).
    """
    # Convert float audio (range -1, 1) to 16-bit PCM (range -32768, 32767)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Initialize WebRTC VAD and set its mode
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)
    
    speech_frames = []
    # Process the audio in frames
    for frame in frame_generator(frame_duration_ms, audio_int16, sample_rate):
        frame_bytes = frame.tobytes()
        if vad.is_speech(frame_bytes, sample_rate):
            speech_frames.append(frame)
    
    if not speech_frames:
        # If no speech is detected, return the original audio
        return audio
    
    # Concatenate speech frames into one array
    speech_audio_int16 = np.concatenate(speech_frames)
    # Convert back to float32 in range -1 to 1
    speech_audio = speech_audio_int16.astype(np.float32) / 32767.0
    return speech_audio

# ------------------------------------------------------------------
# Existing Audio Enhancement Functions
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Main Audio Enhancement Pipeline
# ------------------------------------------------------------------
def enhance_audio(input_file, output_file, apply_vad_flag=False):
    # Load audio using librosa (preserving the original sampling rate)
    audio, sr = librosa.load(input_file, sr=None)
    
    # If enabled, apply WebRTC VAD to extract speech segments
    if apply_vad_flag:
        audio = apply_webrtc_vad(audio, sr)
    
    # Apply the rest of the enhancement steps
    audio = reduce_noise(audio, sr)
    audio = parametric_eq(audio, sr)
    audio = butter_lowpass_filter(audio, cutoff=8000, fs=sr)
    audio = normalize_audio(audio)
    audio = compress_audio(audio, sr)
    
    # Trim any extra silence from the beginning and end
    audio, _ = librosa.effects.trim(audio)
    
    # Save the enhanced audio as a 16-bit PCM WAV file
    sf.write(output_file, audio, sr, format='WAV', subtype='PCM_16')
    st.write(f'Enhanced audio saved to {output_file}')

# ------------------------------------------------------------------
# Streamlit Interface
# ------------------------------------------------------------------
def main():
    st.title("Audio Enhancer Pro")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
    
    # Checkbox to optionally apply WebRTC VAD for speech extraction
    apply_vad_flag = st.checkbox("Apply WebRTC VAD (Extract Speech)", value=False)
    
    if uploaded_file is not None:
        input_file = "temp_input.wav"
        output_file = "enhanced_output.wav"
        
        # Save the uploaded file temporarily
        with open(input_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(input_file, format='audio/wav')
        
        if st.button("Enhance Audio"):
            enhance_audio(input_file, output_file, apply_vad_flag=apply_vad_flag)
            st.success("Audio enhancement complete!")
            st.audio(output_file, format='audio/wav')

if __name__ == '__main__':
    main()
