from datasets import load_dataset, Audio
from huggingface_hub import login
from dotenv import load_dotenv
import os
load_dotenv()

from moore_utils import *


HF_DATASET_NAME         = "ArissBandoss/moore-tts-full-dataset"
HF_DATASET_AUDIO_COLUMN = "denoised_audio"
#SAMPLING_RATE           =  16000

HF_DATASET_CACHE_DIR    = "/teamspace/studios/this_studio/IMS-Toucan/.cache"
DATASET_PATH            = "/teamspace/studios/this_studio/IMS-Toucan/dataset"
AUDIO_FILES_DIR         = "/teamspace/studios/this_studio/IMS-Toucan/dataset/wavs"

HF_TOKEN                = os.getenv("HF_TOKEN")
METADATA_FILE_NAME      = "metadata.csv"

# Log in to Hugging Face using the token
login(token=HF_TOKEN)


# Load the dataset splits
print(f"\n\n========== Loading dataset from Hugging Face repo {HF_DATASET_NAME} ========")
dataset = load_dataset(
    HF_DATASET_NAME, 
    #download_mode="force_redownload",
    # cache_dir=HF_DATASET_CACHE_DIR,
    revision="e9c0251804007c5f528e3d970813b19afe4f744b"
)

# Preprocessing
#dataset = dataset.cast_column(HF_DATASET_AUDIO_COLUMN, Audio(sampling_rate=SAMPLING_RATE))
dataset['train'] = dataset['train'].add_column('lang', ['mos'] * len(dataset['train']))


print(f"\n\n============== Creating audio files at... {DATASET_PATH} ================")
dataset = batch_create_audio_files_and_update_dataset(
    dataset, 
    audio_column=HF_DATASET_AUDIO_COLUMN,
    output_dir=AUDIO_FILES_DIR
)

create_metadata_file(
    dataset['train'], 
    output_dir=DATASET_PATH, 
    filename=METADATA_FILE_NAME
)
