# pip3 install openai-whisper ffmpeg-python sentence-transformers numpy
# pip3 install boto3 openai-whisper ffmpeg-python sentence-transformers
# pip3 install tiktoken
# pip3 install transformers torch sentencepiece
# pip3 install -U tokenizers
# pip3 install transformers -U
# pip3 install datetime
# pip3 install re

import subprocess
import whisper
import os
import re
import numpy as np
from openai import OpenAI
import tiktoken
import boto3
from sentence_transformers import SentenceTransformer
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# AWS S3 configuration
s3_bucket_name = "for-ott-ssai-input"
s3_object_key = "micro-content/stored-video/input.mp4"
local_mp4_path = "input.mp4"

# openai.api_key = os.getenv("OPENAI_API_KEY")

gptClient = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)



# Load Whisper model and SentenceTransformer model
whisper_model = whisper.load_model("turbo")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set output directory for segments
output_dir = "news_segments"
os.makedirs(output_dir, exist_ok=True)


# tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
TOKEN_LIMIT = 4096

# Initialize Boto3 client
s3 = boto3.client('s3')







# def segment_transcription_with_mistral(transcription):

#     # Load Mistral or other open LLM model and tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
#     model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

#     text = transcription['text']

#     # Create the prompt for Mistral-like LLM
#     prompt = (
#         "The following is a transcription of a news broadcast. "
#         "Please divide the transcription into context-based segments. "
#         "Provide the segments in a structured format:\n\n"
#         f"{text}\n"
#         "Divide the above transcription into distinct topics and provide structured segments."
#     )

#     # Tokenize the prompt
#     inputs = tokenizer(prompt, return_tensors="pt")
#     input_ids = inputs.input_ids

#     # Generate output using the Mistral model
#     with torch.no_grad():
#         output = model.generate(input_ids, max_length=1024, num_return_sequences=1)

#     # Decode the output into text
#     output_text = tokenizer.decode(output[0], skip_special_tokens=True)

#     # Split the output text into segments (assuming a structured output is returned)
#     segments = []  # Parse the output into segments based on your format requirements
#     current_segment = {}
#     for line in output_text.splitlines():
#         if line.startswith("Start Time:"):
#             if current_segment:
#                 segments.append(current_segment)
#             current_segment = {"start_time": line.replace("Start Time:", "").strip()}
#         elif line.startswith("End Time:"):
#             current_segment["end_time"] = line.replace("End Time:", "").strip()
#         elif line.startswith("Text:"):
#             current_segment["text"] = line.replace("Text:", "").strip()
#     if current_segment:
#         segments.append(current_segment)

#     return segments







# Step 4: Use GPT-4 to Determine Context-Based Segments
def segment_transcription_with_gpt(transcription):
    text = transcription['text']

    # Create a prompt for GPT-4
    prompt = (
       "Persona: You work as an intern for a news agency Context: You are responsible for splitting a large text containing several disparate news segments into smaller individual and continuous text segments. Task: Identify each change in context to a new topic in the provided text. Output Format: 1. A list of each individual change in context, each representing a new news segment.2. For each segment, provide:a. Title – A concise title for the segment.b. Starting Point – The point in the transcript where the segment begins.c. Summary – A brief description summarizing the content of the segment. d. HashTags - whatever possible hashtags that can be created "
       "Here is the transcription:\n"
        f"{text}\n"
    )


    # segments = transcription['segments']  # Get the segments with timing info

    # # Build a structured prompt including start/end times and text from Whisper
    # prompt = "The following is a transcription of a news broadcast. Please group the transcription into context-based segments. You will be given start and end times for each portion of the text. Group the segments by context, but retain the original start and end times. Provide the result in the following JSON format:\n\n"

    # prompt += "[\n"
    # for segment in segments:
    #     start_time = segment['start']  # Start time in seconds
    #     end_time = segment.get('end', None)  # End time in seconds
    #     text = segment['text']
    #     prompt += f"  {{'start_time': {start_time}, 'end_time': {end_time}, 'text': '{text.strip()}'}},\n"
    # prompt += "]\n"


    response = gptClient.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
                # "content": "Hello!"
            }
            ],
            model="gpt-4",
        )

    print("response ", response)
    # Extract the generated content from the response
    segments_text = response.choices[0].message.content
    
    # Convert the response to a usable format (assuming it's a JSON-like output)
    import ast
    try:
        segments = ast.literal_eval(segments_text)
    except (SyntaxError, ValueError) as e:
        raise ValueError("The response format from GPT could not be parsed. Please check the response for errors.")

    return segments


# Step 1: Download MP4 File from S3 Using Boto3
def download_from_s3(bucket_name, object_key, local_file):
    s3.download_file(bucket_name, object_key, local_file)
    print(f"Downloaded {object_key} from S3 bucket {bucket_name} to {local_file}")

# Step 2: Extract Audio from MP4 File Using FFmpeg
def extract_audio(input_file, output_audio, duration=60):
    command = [
        "ffmpeg",
        "-i", input_file,
        # "-t", str(duration),  # Duration of extraction
        "-vn",  # No video
        "-acodec", "libmp3lame",  # Use MP3 codec
        "-b:a", "64k",  # Audio bitrate (can go as low as 32k for smaller files)
        "-ar", "16000",  # Sample rate
        "-ac", "1",  # Mono channel
         output_audio
    ]
    # command1 = [
    #     "ffmpeg",
    #     "-i", input_file,
    #       "-t", str(duration),  # Duration of extraction (1 minute)
    #     "-vn",  # No video
    #     "-acodec", "pcm_s16le",  # Convert to PCM
    #     "-ar", "16000",  # Sample rate
    #     "-ac", "1",  # Mono channel
    #     output_audio
    # ]
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

# Utility function to convert "HH:MM:SS" time format to seconds
def time_to_seconds(time_str):
    """Convert time string of format 'HH:MM:SS' or 'MM:SS' or 'H:MM' to total seconds."""
    time_parts = time_str.split(':')

    if len(time_parts) == 3:
        # Time format is HH:MM:SS
        hours, minutes, seconds = time_parts
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    elif len(time_parts) == 2:
        # Time format is MM:SS or H:MM
        minutes, seconds = time_parts
        total_seconds = int(minutes) * 60 + int(seconds)
    elif len(time_parts) == 1:
        # Time format is SS (only seconds)
        total_seconds = int(time_parts[0])
    else:
        raise ValueError(f"Unrecognized time format: {time_str}")

    return total_seconds

def match_text_with_first_second_timing(second_array, first_array):
    # Create an empty list to store the results
    matched_segments = []

    # Iterate over second array
    for second in second_array:
        # Clean the text of the second array element
        clean_second_text = clean_text(second["text"])
        
        # Initialize variables to track start and end time
        first_occurrence_start = None
        last_occurrence_end = None
        
        # Iterate over first array
        for first in first_array:
            # Clean the text of the first array element
            clean_first_text = clean_text(first["text"])
            
            # Check if any part of the cleaned second array's text matches cleaned first array's text
            if clean_second_text.find(clean_first_text) != -1 or clean_first_text.find(clean_second_text) != -1:
                # If it's the first match, set the start time
                if first_occurrence_start is None:
                    first_occurrence_start = first["start"]
                # Update the end time with the last match's end time
                last_occurrence_end = first["end"]
        
        # If a match is found, add it to the results
        if first_occurrence_start is not None and last_occurrence_end is not None:
            matched_segments.append({
                "start": first_occurrence_start,  # Start of the first occurrence
                "end": last_occurrence_end,      # End of the last occurrence
                "text": second["text"]
            })

    # Print or return the matched segments
    print(matched_segments)
    return matched_segments



def clean_text(text):
    """
    Function to preprocess the text by removing punctuation and converting to lowercase.
    """
    return re.sub(r'[^a-zA-Z\s]', '', text).lower()

def match_text_with_timing(gpt_text, transcription_segments):
    """
    Matches GPT text with transcription segments and aggregates the start and end times.
    """
    cleaned_gpt_text = clean_text(gpt_text)
    first_occurrence_start = None
    last_occurrence_end = None

    for transcription in transcription_segments:
        cleaned_transcription_text = clean_text(transcription["text"])

        # Check if the cleaned GPT text matches any part of the transcription text
        if cleaned_gpt_text.find(cleaned_transcription_text) != -1 or cleaned_transcription_text.find(cleaned_gpt_text) != -1:
            if first_occurrence_start is None:
                first_occurrence_start = transcription["start"]
            last_occurrence_end = transcription["end"]

    return first_occurrence_start, last_occurrence_end

def save_segments(input_video, gpt_segments, transcription_segments):
    """
    This function saves video segments based on the GPT-generated context-based
    segments by matching them with the Whisper transcription's start and end times.
    """
    segment_files = []

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, gpt_segment in enumerate(gpt_segments):
        gpt_text = gpt_segment["text"].strip()  # GPT's generated text

        # Match the GPT segment text with transcription timing and aggregate start/end times
        start_time, end_time = match_text_with_timing(gpt_text, transcription_segments)

        # If no match found, continue to the next GPT segment
        if start_time is None or end_time is None:
            print(f"Could not match GPT segment: {gpt_text}")
            continue

        segment_file = os.path.join(output_dir, f"segment_{i + 1}.mp4")
        segment_files.append(segment_file)

        # Define FFmpeg command to cut out the segment
        command = [
            "ffmpeg",
            "-i", input_video,
            "-ss", str(start_time)  # Start time in seconds
        ]

        # Calculate the duration and add to the FFmpeg command
        duration = end_time - start_time
        command += ["-t", str(duration)]  # Duration in seconds

        command += ["-c", "copy", segment_file]

        # Run FFmpeg to extract the segment
        try:
            subprocess.run(command, check=True)
            print(f"Saved segment: {segment_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting segment {i + 1}: {e}")

    return segment_files

def match_text_with_timingOLD(gpt_segment_text, transcription_segments):
    """
    This function matches GPT-generated text with the transcription segments
    from Whisper and returns the start and end time that best match the text.
    """
    # Initialize start and end times
    matched_start_time = None
    matched_end_time = None

    # Loop over transcription segments
    for trans_segment in transcription_segments:
        trans_text = trans_segment['text'].strip()

        # Check if GPT-generated text matches the transcription segment
        if gpt_segment_text in trans_text or trans_text in gpt_segment_text:
            # Match found, assign start and end times
            matched_start_time = trans_segment['start']
            matched_end_time = trans_segment['end']
            break

    return matched_start_time, matched_end_time

def save_identified_segmentsOld(input_video, gpt_segments, transcription_segments):
    matched_segments = match_text_with_first_second_timing(gpt_segments, transcription_segments)

# def save_segmentsOld(input_video, gpt_segments, transcription_segments):
#     """
#     This function saves video segments based on the GPT-generated context-based
#     segments by matching them with the Whisper transcription's start and end times.
#     """
#     segment_files = []
#     for i, gpt_segment in enumerate(gpt_segments):
#         gpt_text = gpt_segment["text"].strip()  # GPT's generated text

#         # Match the GPT segment text with transcription timing
#         # start_time, end_time = match_text_with_timing(gpt_text, transcription_segments)
#         # start_time, end_time = match_text_with_first_second_timing(gpt_text, transcription_segments)

#         if start_time is None or end_time is None:
#             print(f"Could not match GPT segment: {gpt_text}")
#             continue

#         segment_file = os.path.join(output_dir, f"segment_{i + 1}.mp4")
#         segment_files.append(segment_file)

#         # Define FFmpeg command to cut out the segment
#         command = [
#             "ffmpeg",
#             "-i", input_video,
#             "-ss", str(start_time)  # Start time in seconds
#         ]
#         if end_time is not None:
#             duration = end_time - start_time  # Calculate duration
#             command += ["-t", str(duration)]  # Duration in seconds

#         command += ["-c", "copy", segment_file]

#         # Run FFmpeg to extract the segment
#         subprocess.run(command, check=True)
#         print(f"Saved segment: {segment_file}")

#     return segment_files

# # Step 5: Save Each Segment as a Separate File
# def save_segmentsOLD(input_video, segments):
#     segment_files = []
#     for i, segment in enumerate(segments):
#         start_time = time_to_seconds(segment["start_time"])
#         end_time = time_to_seconds(segment["end_time"])
#         segment_file = os.path.join(output_dir, f"segment_{i + 1}.mp4")
#         segment_files.append(segment_file)

#         # Define FFmpeg command to cut out the segment
#         command = [
#             "ffmpeg",
#             "-i", input_video,
#             "-ss", str(start_time)  # Start time
#         ]
#         if end_time is not None:
#             duration = end_time - start_time
#             command += ["-t", str(duration)]  # Duration

#         command += ["-c", "copy", segment_file]

#         # Run FFmpeg to extract the segment
#         subprocess.run(command, check=True)
#         print(f"Saved segment: {segment_file}")

#     return segment_files

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
    audio_file = "output_min_1.mp3"

    # Step 2: Extract audio from MP4
    print("Extracting audio from MP4...")
    extract_audio(local_mp4_path, audio_file)

    # Step 3: Transcribe audio using Whisper
    print("Transcribing audio using Whisper...")
    transcription = transcribe_audio(audio_file)
    print("Transcribing---------------")
    print(transcription)


    
    # print("Segmenting the transcription using GPT-4...")
    segments = segment_transcription_with_gpt(transcription)
    print(segments)
    # transcription_segments = transcription['segments']
    # print("Saving each segment as a separate video file...")
    # segment_files = save_segments(local_mp4_path, segments,transcription_segments)
    # print("Uploading segments to S3...")
    # upload_segments_to_s3(s3_bucket_name, segment_files)
    os.remove(audio_file)

if __name__ == "__main__":
    main()
