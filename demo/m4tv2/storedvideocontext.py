# pip3 install openai-whisper ffmpeg-python sentence-transformers numpy


import subprocess
import whisper
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Load Whisper model and SentenceTransformer model
whisper_model = whisper.load_model("base")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set your input MP4 file
input_mp4 = "input.mp4"
output_dir = "news_segments"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Extract Audio from MP4 File Using FFmpeg
def extract_audio(input_file, output_audio):
    command = [
        "ffmpeg",
        "-i", input_file,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # Convert to PCM
        "-ar", "16000",  # Sample rate
        "-ac", "1",  # Mono channel
        output_audio
    ]
    subprocess.run(command, check=True)

# Step 2: Transcribe Audio Using Whisper
def transcribe_audio(audio_file):
    result = whisper_model.transcribe(audio_file)
    return result

# Step 3: Segment Transcription Based on Context
def segment_transcription(transcription):
    segments = transcription['segments']
    sentences = [segment['text'] for segment in segments]
    start_times = [segment['start'] for segment in segments]

    # Calculate embeddings for each sentence
    embeddings = embedding_model.encode(sentences)

    # Identify context changes based on embedding similarity
    context_boundaries = [0]  # Start with the first sentence
    threshold = 0.7  # Similarity threshold to determine if context has shifted

    for i in range(1, len(embeddings)):
        similarity = np.dot(embeddings[i - 1], embeddings[i]) / (np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]))
        if similarity < threshold:
            context_boundaries.append(i)

    # Create segments based on context boundaries
    segmented_transcriptions = []
    for i in range(len(context_boundaries)):
        start_idx = context_boundaries[i]
        end_idx = context_boundaries[i + 1] if i + 1 < len(context_boundaries) else len(sentences)

        segment_text = " ".join(sentences[start_idx:end_idx])
        start_time = start_times[start_idx]
        end_time = start_times[end_idx - 1] if end_idx - 1 < len(start_times) else None

        segmented_transcriptions.append({
            "start_time": start_time,
            "end_time": end_time,
            "text": segment_text
        })

    return segmented_transcriptions

# Step 4: Save Each Segment as a Separate File
def save_segments(input_video, segments):
    for i, segment in enumerate(segments):
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        segment_file = os.path.join(output_dir, f"segment_{i + 1}.mp4")

        # Define FFmpeg command to cut out the segment
        command = [
            "ffmpeg",
            "-i", input_video,
            "-ss", str(start_time)  # Start time
        ]
        if end_time is not None:
            duration = end_time - start_time
            command += ["-t", str(duration)]  # Duration

        command += ["-c", "copy", segment_file]

        # Run FFmpeg to extract the segment
        subprocess.run(command, check=True)
        print(f"Saved segment: {segment_file}")

# Main function to handle all steps
def main():
    # Define paths
    audio_file = "output.wav"

    # Step 1: Extract audio from MP4
    print("Extracting audio from MP4...")
    # extract_audio(input_mp4, audio_file)

    # Step 2: Transcribe audio using Whisper
    print("Transcribing audio using Whisper...")
    transcription = transcribe_audio(audio_file)
    print(" -------- transcription --------- ")
    print(transcription)

    # Step 3: Segment the transcription based on context
    print("Segmenting the transcription based on context...")
    segments = segment_transcription(transcription)

    print(" -------- segments --------- ")
    print(segments)
    # Step 4: Save each segment as a separate video file
    print("Saving each segment as a separate video file...")
    save_segments(input_mp4, segments)

    # Clean up temporary audio file
    os.remove(audio_file)

if __name__ == "__main__":
    main()
