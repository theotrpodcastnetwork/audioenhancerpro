import os
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, filtfilt
from pydub import AudioSegment
from flask import Flask, request, send_file, render_template
import io

app = Flask(__name__)

# 🔹 Process audio in chunks to optimize memory usage
def ai_reduce_noise(audio, sr, chunk_size=2):
    """Apply AI-based noise reduction in small chunks to prevent crashes on low-memory instances."""
    chunk_samples = int(chunk_size * sr)
    enhanced_audio = np.zeros_like(audio)

    for start in range(0, len(audio), chunk_samples):
        end = min(start + chunk_samples, len(audio))
        enhanced_audio[start:end] = nr.reduce_noise(y=audio[start:end], sr=sr, prop_decrease=0.8)

    return enhanced_audio

# 🔹 Normalize Audio
def normalize_audio(audio):
    """Normalize full-length audio."""
    return audio / np.max(np.abs(audio))

# 🔹 Lowpass Filter
def butter_lowpass_filter(audio, cutoff, sr, order=5):
    """Apply a lowpass filter to remove high-frequency noise."""
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, audio)

# 🔹 AI-Based Loudness Compression
def ai_compress_audio(audio, target_loudness=-20.0):
    """Apply AI-based loudness compression."""
    audio = audio / np.max(np.abs(audio))
    gain = 10**(target_loudness / 20)
    return audio * gain

# 🔹 Full AI Enhancement Pipeline
def enhance_audio(audio_data, sr):
    """Apply all enhancement steps to audio."""
    audio = ai_reduce_noise(audio_data, sr)
    audio = butter_lowpass_filter(audio, cutoff=8000, sr=sr)
    audio = normalize_audio(audio)
    audio = ai_compress_audio(audio)
    return audio

# 🔹 Upload & Process Audio API
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Get the file from the request
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        # Convert to WAV format (if not already)
        input_buffer = io.BytesIO(file.read())
        audio_segment = AudioSegment.from_file(input_buffer, format=file.filename.split(".")[-1])
        input_buffer = io.BytesIO()
        audio_segment.export(input_buffer, format="wav")
        input_buffer.seek(0)

        # Read full audio data
        with sf.SoundFile(input_buffer) as audio_file:
            audio = audio_file.read(dtype="float32")
            sr = audio_file.samplerate

        # Enhance audio
        enhanced_audio = enhance_audio(audio, sr)

        # Save enhanced audio to buffer
        output_buffer = io.BytesIO()
        sf.write(output_buffer, enhanced_audio, sr, format="WAV", subtype="PCM_16")
        output_buffer.seek(0)

        return send_file(output_buffer, as_attachment=True, download_name="enhanced_audio.wav", mimetype="audio/wav")

    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Audio Enhancer</title>
    </head>
    <body>
        <h1>Upload an audio file to enhance</h1>
        <form action="/" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Enhance Audio">
        </form>
    </body>
    </html>
    '''

# 🔹 Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
