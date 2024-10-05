import subprocess
import whisper
import os
import numpy as np
import boto3
from sentence_transformers import SentenceTransformer

# AWS S3 configuration
s3_bucket_name = "for-ott-ssai-input"
s3_object_key = "micro-content/stored-video/input.mp4"
local_mp4_path = "input.mp4"

# Load Whisper model and SentenceTransformer model
whisper_model = whisper.load_model("base")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set output directory for segments
output_dir = "news_segments"
os.makedirs(output_dir, exist_ok=True)

# Initialize Boto3 client
s3 = boto3.client('s3')

# Step 1: Download MP4 File from S3 Using Boto3
def download_from_s3(bucket_name, object_key, local_file):
    s3.download_file(bucket_name, object_key, local_file)
    print(f"Downloaded {object_key} from S3 bucket {bucket_name} to {local_file}")

# Step 2: Extract Audio from MP4 File Using FFmpeg
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

# Step 3: Transcribe Audio Using Whisper
def transcribe_audio(audio_file):
    result = whisper_model.transcribe(audio_file)
    return result

# Step 4: Segment Transcription Based on Context
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

# Step 5: Save Each Segment as a Separate File
def save_segments(input_video, segments):
    segment_files = []
    for i, segment in enumerate(segments):
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        segment_file = os.path.join(output_dir, f"segment_{i + 1}.mp4")
        segment_files.append(segment_file)

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

    return segment_files

# Step 6: Upload Segments to S3 Using Boto3
def upload_segments_to_s3(bucket_name, segment_files):
    for segment_file in segment_files:
        segment_key = os.path.join("segments", os.path.basename(segment_file))
        s3.upload_file(segment_file, bucket_name, segment_key)
        print(f"Uploaded {segment_file} to S3 bucket {bucket_name} as {segment_key}")

# Main function to handle all steps
def main():
    # Step 1: Download the MP4 file from S3
    print("Downloading MP4 file from S3...")
    download_from_s3(s3_bucket_name, s3_object_key, local_mp4_path)

    # Define paths
    audio_file = "temp_audio.wav"

    # Step 2: Extract audio from MP4
    print("Extracting audio from MP4...")
    extract_audio(local_mp4_path, audio_file)

    # Step 3: Transcribe audio using Whisper
    print("Transcribing audio using Whisper...")
    transcription = transcribe_audio(audio_file)

    # Step 4: Segment the transcription based on context
    print("Segmenting the transcription based on context...")
    segments = segment_transcription(transcription)

    # Step 5: Save each segment as a separate video file
    print("Saving each segment as a separate video file...")
    segment_files = save_segments(local_mp4_path, segments)

    # Step 6: Upload segments to S3
    print("Uploading segments to S3...")
    upload_segments_to_s3(s3_bucket_name, segment_files)

    # Clean up temporary audio file
    os.remove(audio_file)

if __name__ == "__main__":
    main()
