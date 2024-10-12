import os
import gc
import soundfile as sf
from tqdm import tqdm
from huggingface_hub import HfApi, HfFolder


# MOORE UTIL FUNCTIONS
def create_audio_file(example, audio_column, output_dir, index):
    """
    Creates a single audio file from the 'audio' column of an example and returns the file path.
    """
    # Construct the output file path
    audio_filename = f"audio_{index}.wav"
    audio_filepath = os.path.join(output_dir, audio_filename)

    # If file does not exist, write the audio data to the file
    if not os.path.isfile(audio_filepath):
        # Extract audio data and sample rate from the example
        audio_data = example[audio_column]['array']
        sample_rate = example[audio_column]['sampling_rate']

        # Save the audio file
        sf.write(audio_filepath, audio_data, sample_rate)

    #return {"audio_file_path": audio_filepath}
    return {"audio_filename": audio_filename}


def batch_create_audio_files_and_update_dataset(dataset, audio_column, output_dir):
    """
    Maps over the dataset, creates audio files and updates the dataset with the file paths.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Use the .map() function to process the dataset and create audio files
    dataset_with_audio_paths = dataset.map(
        lambda example, idx: create_audio_file(example, audio_column, output_dir, idx),
        with_indices=True,  # Pass example indices to the map function
        num_proc=32
    )

    return dataset_with_audio_paths


def create_audio_files_and_update_dataset(dataset, audio_column, output_dir):
    """
    Create audio files from the 'audio' column of a Hugging Face dataset and update the dataset with file paths.

    Parameters:
    - dataset: The input dataset that contains the 'audio' column.
    - audio_column: The name of the column containing the audio data (datasets.Audio feature).
    - output_dir: The directory where audio files will be saved.

    Returns:
    - The updated dataset with the 'audio' column containing the file paths of saved audio files.
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare a list to hold the file paths, to avoid modifying the dataset in-place
    #audio_file_paths = []
    audio_file_names = []

    for index, example in tqdm(enumerate(dataset), total=len(dataset), desc="Creating audio files", unit="file"):
        audio_filename = f"audio_{index}.wav"
        audio_filepath = os.path.join(output_dir, audio_filename)

        if os.path.isfile(audio_filepath):
            audio_file_names.append(audio_filepath)
            continue

        audio_data = example[audio_column]['array']
        # Typically, the sample rate should also be retrieved from the dataset
        sample_rate = example[audio_column]['sampling_rate']

        # Save the audio file
        #sf.write(audio_filepath, audio_data, sample_rate)
        sf.write(audio_filename, audio_data, sample_rate)

        # Append the file path to the list
        #audio_file_paths.append(audio_filepath)
        audio_file_names.append(audio_filename)

        # Option to clear memory if needed, uncomment if large arrays are involved
        del audio_data
        gc.collect()

    # Update the dataset with the new file paths
    #dataset = dataset.add_column("audio_file_path", audio_file_paths)
    dataset = dataset.add_column("audio_filename", audio_filename)

    return dataset


# Function to create the metadata file
def create_metadata_file(dataset, output_dir='MyTTSDataSet', filename='metadata.csv'):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the path to the metadata file
    metadata_path = os.path.join(output_dir, filename)

    # Open the metadata file in write mode
    with open(metadata_path, 'w', encoding='utf-8') as f:
        # Iterate over each item in the dataset
        f.write(f"audio_file|text|speaker_name\n")
        for item in dataset:
            # Your dataset should have an 'audio' column with a dictionary containing the file path and 'array' for the audio data
            #audio_path = item['audio_file_path'].replace(".wav", "")
            audio_filename = item['audio_filename']
            text = item['text'].replace(" ", " ").replace(" ", " ").replace("\n", " ").replace("\\", " ")
            normalized_text = text
            speaker_id = item['speaker_id']
            lang = item['lang']

            # Write the formatted data to the metadata file
            f.write(f"wavs/{audio_filename}|{str(text)}|{str(speaker_id)}\n")

    return metadata_path
