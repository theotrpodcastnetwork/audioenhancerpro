import librosa
import librosa.display
import soundfile as sf
import numpy as np
import noisereduce as nr
import os
from pydub import AudioSegment

def load_audio(file_path):
    """Load an audio file."""
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def noise_reduction(audio, sr):
    """Apply AI-based noise reduction."""
    reduced_audio = nr.reduce_noise(y=audio, sr=sr)
    return reduced_audio

def enhance_voice(audio, sr):
    """Apply dynamic EQ and de-esser."""
    audio = librosa.effects.preemphasis(audio)
    return audio

def save_audio(output_path, audio, sr):
    """Save processed audio file."""
    sf.write(output_path, audio, sr)

def process_audio(input_path, output_path):
    """Main function to process an audio file."""
    audio, sr = load_audio(input_path)
    audio = noise_reduction(audio, sr)
    audio = enhance_voice(audio, sr)
    save_audio(output_path, audio, sr)
    print(f"Processed file saved: {output_path}")

# Example usage
if __name__ == "__main__":
    input_file = "input_audio.wav"
    output_file = "enhanced_audio.wav"
    process_audio(input_file, output_file)
