import os
import re
import subprocess
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Function to detect silence and get timestamps
def detect_silence(audio_file, silence_threshold='-50dB', silence_duration=1.0):
    """
    Run FFmpeg's silencedetect filter to detect silence in an audio file.
    
    :param audio_file: The path to the audio file.
    :param silence_threshold: Threshold for silence detection (default is -50dB).
    :param silence_duration: Minimum silence duration to consider a break (in seconds).
    :return: List of timestamps where silence starts and ends.
    """
    silence_command = [
        'ffmpeg', '-i', audio_file, '-af',
        f'silencedetect=n={silence_threshold}:d={silence_duration}',
        '-f', 'null', '-'
    ]

    # Run the command and capture the output
    result = subprocess.run(silence_command, stderr=subprocess.PIPE, text=True)
    output = result.stderr
    
    # Parse silence start and end times from the FFmpeg output
    silence_times = []
    for line in output.splitlines():
        silence_start_match = re.search(r'silence_start: (\d+(\.\d+)?)', line)
        silence_end_match = re.search(r'silence_end: (\d+(\.\d+)?)', line)
        
        if silence_start_match:
            silence_start = float(silence_start_match.group(1))
            silence_times.append(silence_start)
        
        if silence_end_match:
            silence_end = float(silence_end_match.group(1))
            silence_times.append(silence_end)
    
    return silence_times

# Function to break audio into chunks based on silence timestamps
def split_audio_by_silence(audio_file, silence_times, output_folder='audio_chunks'):
    """
    Split the audio file into chunks based on silence timestamps.
    
    :param audio_file: The path to the audio file.
    :param silence_times: List of timestamps where silence starts and ends.
    :param output_folder: Folder where the chunks will be saved.
    :return: List of paths to the generated audio chunks.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # FFmpeg command to split audio based on start and end times
    chunk_files = []
    previous_time = 0
    for i, silence_time in enumerate(silence_times):
        chunk_file = f"{output_folder}/chunk_{i:03d}.mp3"
        ffmpeg_split_command = [
            'ffmpeg', '-i', audio_file, '-ss', str(previous_time), '-to', str(silence_time),
            '-c', 'copy', chunk_file
        ]
        subprocess.run(ffmpeg_split_command)
        chunk_files.append(chunk_file)
        previous_time = silence_time
    
    # Handle the last chunk after the last silence
    last_chunk_file = f"{output_folder}/chunk_{len(silence_times):03d}.mp3"
    ffmpeg_split_command = [
        'ffmpeg', '-i', audio_file, '-ss', str(previous_time), '-c', 'copy', last_chunk_file
    ]
    subprocess.run(ffmpeg_split_command)
    chunk_files.append(last_chunk_file)
    
    return chunk_files


if __name__ == '__main__':
    # Fetch configuration from environment variables
    AUDIO_FILE = os.getenv('AUDIO_FILE')
    OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', 'audio_chunks')
    
    # Silence detection parameters
    SILENCE_THRESHOLD = os.getenv('SILENCE_THRESHOLD', '-50dB')
    SILENCE_DURATION = float(os.getenv('SILENCE_DURATION', 1.0))

    # Ensure the audio file is provided
    if not AUDIO_FILE:
        print("Error: AUDIO_FILE environment variable is required.")
        exit(1)

    # Step 1: Detect silence and get timestamps
    print("Detecting silence in the audio file...")
    silence_times = detect_silence(AUDIO_FILE, SILENCE_THRESHOLD, SILENCE_DURATION)
    print(f"Silence detected at: {silence_times}")
    
    # Step 2: Split the audio based on the detected silence
    print("Splitting audio based on silence...")
    chunk_files = split_audio_by_silence(AUDIO_FILE, silence_times, OUTPUT_FOLDER)
    
    # Output the results
    print(f"Audio split into {len(chunk_files)} chunks:")
    for chunk in chunk_files:
        print(f" - {chunk}")
