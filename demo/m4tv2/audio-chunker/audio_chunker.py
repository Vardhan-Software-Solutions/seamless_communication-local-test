# pip install python-dotenv

import os
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Function to get the size of a file in MB
def get_file_size_in_mb(file_path):
    file_size_bytes = os.path.getsize(file_path)
    return file_size_bytes / (1024 * 1024)

# Function to break audio based on silence detection
def break_audio_by_silence(audio_file, silence_threshold='-50dB', silence_duration=1.0, output_folder='audio_chunks'):
    """
    Break audio into smaller files based on silence detection.
    
    :param audio_file: The path to the audio file.
    :param silence_threshold: Threshold for silence detection (default is -50dB).
    :param silence_duration: Minimum silence duration to consider a break (in seconds).
    :param output_folder: Folder where the chunks will be saved.
    :return: List of paths to the generated audio chunks.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # FFmpeg command to detect silence and split audio based on silence
    silence_command = f"""
    ffmpeg -i {audio_file} -af silencedetect=n={silence_threshold}:d={silence_duration} -f segment -segment_times silence -c copy {output_folder}/chunk_%03d.mp3
    """
    os.system(silence_command)
    
    # Collect all generated chunk files
    return [f"{output_folder}/{f}" for f in os.listdir(output_folder) if f.endswith('.mp3')]

# Function to break a large audio chunk into smaller files based on file size
def break_audio_by_size(audio_file, max_size_mb=10, output_folder='audio_chunks_by_size'):
    """
    Break audio into smaller files based on file size.
    
    :param audio_file: The path to the audio file.
    :param max_size_mb: Maximum size of each chunk in MB (default is 10MB).
    :param output_folder: Folder where the chunks will be saved.
    :return: List of paths to the generated audio chunks.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert MB to bytes for FFmpeg
    size_limit = max_size_mb * 1024 * 1024  # Convert MB to bytes
    size_command = f"""
    ffmpeg -i {audio_file} -fs {size_limit} -f segment -segment_time 600 {output_folder}/chunk_%03d.mp3
    """
    os.system(size_command)
    
    # Collect all generated chunk files
    return [f"{output_folder}/{f}" for f in os.listdir(output_folder) if f.endswith('.mp3')]

# Main function to handle both silence and size-based chunking
def process_audio(audio_file, silence_threshold, silence_duration, max_size_mb, output_folder):
    # Step 1: Break the audio by silence
    print(f"Breaking audio based on silence from file: {audio_file}")
    silence_chunks = break_audio_by_silence(
        audio_file,
        silence_threshold=silence_threshold,
        silence_duration=silence_duration,
        output_folder=output_folder
    )
    
    final_chunks = []
    
    # Step 2: Check each chunk for size and break further if needed
    for chunk in silence_chunks:
        chunk_size = get_file_size_in_mb(chunk)
        
        if chunk_size > max_size_mb:
            print(f"Chunk {chunk} exceeds size limit ({chunk_size:.2f} MB). Breaking it further...")
            
            # Create a temp folder for splitting large chunks
            temp_folder = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(chunk))[0]}_by_size")
            size_chunks = break_audio_by_size(chunk, max_size_mb=max_size_mb, output_folder=temp_folder)
            final_chunks.extend(size_chunks)
            
            # Clean up the original large chunk
            os.remove(chunk)
        else:
            final_chunks.append(chunk)
    
    return final_chunks


if __name__ == '__main__':
    # Fetch configuration from environment variables
    AUDIO_FILE = os.getenv('AUDIO_FILE')
    OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', 'audio_chunks')
    
    # Silence detection parameters
    SILENCE_THRESHOLD = os.getenv('SILENCE_THRESHOLD', '-50dB')
    SILENCE_DURATION = float(os.getenv('SILENCE_DURATION', 1.0))
    
    # Size-based chunking parameter
    MAX_SIZE_MB = int(os.getenv('MAX_SIZE_MB', 10))

    if not AUDIO_FILE:
        print("AUDIO_FILE environment variable is required.")
        exit(1)

    # Process the audio with silence and size-based chunking
    final_chunks = process_audio(
        audio_file=AUDIO_FILE,
        silence_threshold=SILENCE_THRESHOLD,
        silence_duration=SILENCE_DURATION,
        max_size_mb=MAX_SIZE_MB,
        output_folder=OUTPUT_FOLDER
    )

    # Output the results
    print(f"Audio broken into {len(final_chunks)} chunks:")
    for chunk in final_chunks:
        print(f" - {chunk}")
