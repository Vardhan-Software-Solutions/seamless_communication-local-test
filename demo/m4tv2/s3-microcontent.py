# pip3 install openai-whisper ffmpeg-python sentence-transformers numpy
# pip3 install boto3 openai-whisper ffmpeg-python sentence-transformers
# pip3 install tiktoken
# pip3 install transformers torch sentencepiece
# pip3 install -U tokenizers
# pip3 install transformers -U


# pip3 install openai-whisper ffmpeg-python boto3 datetime re
# pip3 install boto3
# pip3 install datetime
# pip3 install re
import json
import subprocess
import whisper
import os
import re
from openai import OpenAI
import boto3
from datetime import datetime


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
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set output directory for segments
output_dir = "news_segments"
os.makedirs(output_dir, exist_ok=True)


# tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
TOKEN_LIMIT = 4096

# Initialize Boto3 client
s3 = boto3.client('s3')






# Step 4: Use GPT-4 to Determine Context-Based Segments
def segment_transcription_with_gpt(transcription):
    text = transcription['text']

    # Create a prompt for GPT-4
    prompt = (
       "Persona: You work as an intern for a news agency Context: You are responsible for splitting a large text containing several disparate news segments into smaller individual and continuous text segments.\n"
       "Remember not to merge two different news associated to same person if context is different\n"
       "Task: Identify each change in context to a new topic in the provided text. \n"
       "Output Format: \n"
       "1. A list of each individual change in context, each representing a new news segment.\n"
       "2. For each segment, provide:\n"
       "a. title – A concise title for the segment.\n"
       "b. startText – The point in the transcript where the segment begins.\n"
       "c. endText – The point in the transcript where the segment ends.\n"
       "d. description – A brief description summarizing the content of the segment."
       "e. hashTags - whatever possible hashtags that can be created "
        "f. text: Actual Full Text without any changes "
        "Provide the result only in valid JSON format\n"
        f"{text}\n"
    )

    response = gptClient.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
            ],
            model="gpt-4o",
            response_format={ "type": "json_object" }
        )

    content1 = response.choices[0].message.content
    content = json.loads(content1)
    chatSegments = response.choices[0].message.content
    if 'segments' in content:
        chatSegments = content['segments']
    print("\n response-segments-text:: \n\n\n ", chatSegments)
    return chatSegments


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
   
    subprocess.run(command, check=True)

# Step 3: Transcribe Audio Using Whisper
def transcribe_audio(audio_file):
    result = whisper_model.transcribe(audio_file)
    return result


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

# def match_text_with_first_second_timing(second_array, first_array):
#     # Create an empty list to store the results
#     matched_segments = []

#     # Iterate over second array
#     for second in second_array:
#         # Clean the text of the second array element
#         clean_second_text = clean_text(second["text"])
        
#         # Initialize variables to track start and end time
#         first_occurrence_start = None
#         last_occurrence_end = None
        
#         # Iterate over first array
#         for first in first_array:
#             # Clean the text of the first array element
#             clean_first_text = clean_text(first["text"])
            
#             # Check if any part of the cleaned second array's text matches cleaned first array's text
#             if clean_second_text.find(clean_first_text) != -1 or clean_first_text.find(clean_second_text) != -1:
#                 # If it's the first match, set the start time
#                 if first_occurrence_start is None:
#                     first_occurrence_start = first["start"]
#                 # Update the end time with the last match's end time
#                 last_occurrence_end = first["end"]
        
#         # If a match is found, add it to the results
#         if first_occurrence_start is not None and last_occurrence_end is not None:
#             matched_segments.append({
#                 "start": first_occurrence_start,  # Start of the first occurrence
#                 "end": last_occurrence_end,      # End of the last occurrence
#                 "text": second["text"]
#             })

#     # Print or return the matched segments
#     print(matched_segments)
#     return matched_segments


def get_start_end_time(item1,array2):
    start_pos = None
    end_pos = None
        
        # Compare startText with text in array2 to find start position
    for item2 in array2:
        if clean_text(item1["startText"]).strip() in clean_text(item2["text"]).strip():
            start_pos = item2["start"]
            break
    # Compare endText with text in array2 to find end position
    for item2 in array2:
        if clean_text(item1["endText"]).strip() in clean_text(item2["text"]).strip():
            end_pos = item2["end"]
            break

return start_pos, end_pos


def find_start_end(array1, array2):
    result = []

    for item1 in array1:
        start_pos = None
        end_pos = None
        
        # Compare startText with text in array2 to find start position
        for item2 in array2:
            if item1["startText"].strip() == item2["text"].strip():
                start_pos = item2["start"]
                break

        # Compare endText with text in array2 to find end position
        for item2 in array2:
            if item1["endText"].strip() == item2["text"].strip():
                end_pos = item2["end"]
                break
        
        result.append({
            "startText": item1["startText"],
            "endText": item1["endText"],
            "description": item1["description"],
            "hashTags": item1["hashTags"],
            "title": item1["title"],
            "text":item1["text"],
            "start": start_pos,
            "end": end_pos
        })
    
    return result

# Call the function and print the result
# matched_data = find_start_end(array1, array2)
# print(matched_data)




def clean_text(text):
    """
    Function to preprocess the text by removing punctuation and converting to lowercase.
    """
    return re.sub(r'[^a-zA-Z\s]', '', text).lower()

# def match_text_with_timing(gpt_text, transcription_segments):
#     """
#     Matches GPT text with transcription segments and aggregates the start and end times.
#     """
#     cleaned_gpt_text = clean_text(gpt_text)
#     first_occurrence_start = None
#     last_occurrence_end = None

#     for transcription in transcription_segments:
#         cleaned_transcription_text = clean_text(transcription["text"])

#         # Check if the cleaned GPT text matches any part of the transcription text
#         if cleaned_gpt_text.find(cleaned_transcription_text) != -1 or cleaned_transcription_text.find(cleaned_gpt_text) != -1:
#             if first_occurrence_start is None:
#                 first_occurrence_start = transcription["start"]
#             last_occurrence_end = transcription["end"]

#     return first_occurrence_start, last_occurrence_end

def save_segments(input_video, gpt_segments, transcription_segments):
    """
    This function saves video segments based on the GPT-generated context-based
    segments by matching them with the Whisper transcription's start and end times.
    """
    segment_files = []

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, gpt_segment in gpt_segments:
        # gpt_text = gpt_segment["text"].strip()  # GPT's generated text

        # Match the GPT segment text with transcription timing and aggregate start/end times
        # start_time, end_time = match_text_with_timing(gpt_text, transcription_segments)
        start_time, end_time = get_start_end_time(gpt_segment,transcription_segments)

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
    # download_from_s3(s3_bucket_name, s3_object_key, local_mp4_path)

    # Define paths
    audio_file = "output_min_1.mp3"

    # Step 2: Extract audio from MP4
    print("Extracting audio from MP4...")
    # extract_audio(local_mp4_path, audio_file)

    # Step 3: Transcribe audio using Whisper
    print("Transcribing audio using Whisper...")
    transcription = transcribe_audio(audio_file)
    print("Transcribing---------------")
    print(transcription)


    
    # print("Segmenting the transcription using GPT-4...")
    segments = segment_transcription_with_gpt(transcription)
    print("final segments from GPT ",segments)
    transcription_segments = transcription['segments']
    print("Saving each segment as a separate video file...")
    # segment_files = save_segments(local_mp4_path, segments,transcription_segments)
    print("Uploading segments to S3...")
    # upload_segments_to_s3(s3_bucket_name, segment_files)
    os.remove(audio_file)

if __name__ == "__main__":
    main()
