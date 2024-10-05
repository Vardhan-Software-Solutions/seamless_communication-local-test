import subprocess
import whisper
import os
import re
from datetime import datetime

# Load Whisper model
model = whisper.load_model("base")

# Set your input MP4 file
input_mp4 = "input-news.mp4"
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
    # Transcribe audio file using Whisper
    result = model.transcribe(audio_file)
    return result

# Step 3: Segment Transcription into Smaller News Items
def segment_transcription(transcription):
    # Split the transcription into segments based on certain keywords
    split_keywords = ["breaking news", "in other news", "next up", "top story"]

    # Create segments
    segments = []
    current_segment = {"start_time": 0, "text": ""}
    lines = transcription['segments']

    for line in lines:
        text = line['text']
        start_time = line['start']
        
        # If the line contains any of the split keywords, save the current segment and start a new one
        if any(keyword in text.lower() for keyword in split_keywords):
            if current_segment['text']:
                segments.append(current_segment)
            current_segment = {"start_time": start_time, "text": text}
        else:
            current_segment["text"] += " " + text

    # Append the final segment
    if current_segment['text']:
        segments.append(current_segment)

    return segments

# Step 4: Save Each Segment as a Separate File
def save_segments(input_video, segments):
    for i, segment in enumerate(segments):
        start_time = segment["start_time"]
        end_time = segments[i + 1]["start_time"] if i + 1 < len(segments) else None
        segment_file = os.path.join(output_dir, f"segment_{i + 1}.mp4")
        
        # Define FFmpeg command to cut out the segment
        command = [
            "ffmpeg",
            "-i", input_video,
            "-ss", str(start_time),  # Start time
        ]
        if end_time is not None:
            duration = end_time - start_time
            command += ["-t", str(duration)]  # Duration
        
        command += [segment_file]
        
        # Run FFmpeg to extract the segment
        subprocess.run(command, check=True)
        print(f"Saved segment: {segment_file}")

# Main function to handle all steps
def main():
    # Define paths
    audio_file = "temp_audio1.wav"

    # Step 1: Extract audio from MP4
    extract_audio(input_mp4, audio_file)

    print("  -EXTRACTION DONE ----------------- ")
    # Step 2: Transcribe audio using Whisper
    transcription = transcribe_audio(audio_file)

    print("  ------------------ ")
    print(transcription)

    # Step 3: Segment the transcription into news items
    segments = segment_transcription(transcription)

    # Step 4: Save each segment as a separate video file
    save_segments(input_mp4, segments)

    # Clean up temporary audio file
    os.remove(audio_file)

if __name__ == "__main__":
    main()
