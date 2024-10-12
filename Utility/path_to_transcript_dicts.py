import glob
import json
import os
import csv
import random
import xml.etree.ElementTree as ET
from csv import DictReader
from pathlib import Path


# HELPER FUNCTIONS
def moore_audio_metadata_to_dict():
    root = "/teamspace/studios/this_studio/IMS-Toucan/dataset"
    csv_files=[
                "/teamspace/studios/this_studio/IMS-Toucan/dataset/metadata.csv", 
                "/teamspace/studios/this_studio/IMS-Toucan/dataset/metadata_val.csv"
               ]
    path_to_transcript = dict()
    
    # Loop through each csv file
    for csv_file in csv_files:
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='|')
            for row in reader:
                if len(row) != 3:  # Ensure the row has all three fields (audio, text, speaker)
                    continue
                audio_rel_path, text, speaker_name = row
                
                # Construct the absolute path to the audio file
                abs_audio_path = os.path.join(root, audio_rel_path)
                
                # Add to the dictionary
                path_to_transcript[abs_audio_path] = text
    
    return path_to_transcript


def split_dictionary_into_chunks(input_dict, split_n):
    res = []
    new_dict = {}
    elements_per_dict = (len(input_dict.keys()) // split_n) + 1
    for k, v in input_dict.items():
        if len(new_dict) < elements_per_dict:
            new_dict[k] = v
        else:
            res.append(new_dict)
            new_dict = {k: v}
    res.append(new_dict)
    return res


def limit_to_n(path_to_transcript, n=40000):
    # deprecated, we now just use the whole thing always, because there's a critical mass of data
    limited_dict = dict()
    if len(path_to_transcript.keys()) > n:
        for key in random.sample(list(path_to_transcript.keys()), n):
            limited_dict[key] = path_to_transcript[key]
        return limited_dict
    else:
        return path_to_transcript


def build_path_to_transcript_integration_test(re_cache=True):
    root = "/mount/resources/speech/corpora/NancyKrebs"
    path_to_transcript = dict()
    with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n")[:500]:
        if line.strip() != "":
            norm_transcript = line.split("|")[1]
            wav_path = os.path.join(root, "wav", line.split("|")[0] + ".wav")
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


def build_path_to_transcript_multi_ling_librispeech_template(root):
    """
    https://arxiv.org/abs/2012.03411
    """
    path_to_transcript = dict()
    with open(os.path.join(root, "transcripts.txt"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            fields = line.split("\t")
            wav_folders = fields[0].split("_")
            wav_path = f"{root}/audio/{wav_folders[0]}/{wav_folders[1]}/{fields[0]}.flac"
            path_to_transcript[wav_path] = fields[1]
    return path_to_transcript


def build_path_to_transcript_libritts_all_clean():
    root = "/mount/resources/speech/corpora/LibriTTS_R/"
    path_train = "/mount/resources/speech/corpora/LibriTTS_R/"  # using all files from the "clean" subsets from LibriTTS-R https://arxiv.org/abs/2305.18802
    path_to_transcript = dict()
    for speaker in os.listdir(path_train):
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if file.endswith("normalized.txt"):
                    with open(os.path.join(path_train, speaker, chapter, file), 'r', encoding='utf8') as tf:
                        transcript = tf.read()
                    wav_file = file.split(".")[0] + ".wav"
                    path_to_transcript[os.path.join(path_train, speaker, chapter, wav_file)] = transcript
    return path_to_transcript
    

# OTHER LANGUAGES
def build_path_to_transcript_bibletts_akuapem_twi():
    path_to_transcript = dict()
    root = '/resources/speech/corpora/BibleTTS/akuapem-twi'
    for split in ['train', 'dev', 'test']:
        for book in Path(root, split).glob('*'):
            for textfile in book.glob('*.txt'):
                with open(textfile, 'r', encoding='utf-8') as f:
                    text = ' '.join([line.strip() for line in f])  # should usually be only one line anyway
                path_to_transcript[textfile.with_suffix('.flac')] = text

    return path_to_transcript


def build_path_to_transcript_african_voices_template(root):
    path_to_transcript = dict()

    with open(Path(root, 'txt.done.data'), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\\"', "'").split('"')
            text = line[1]
            file = line[0].split()[-1]
            path_to_transcript[str(Path(root, 'wav', f'{file}.wav'))] = text

    return path_to_transcript

def build_path_to_transcript_african_voices_hausa_cmv():
    main_root = '/resources/speech/corpora/AfricanVoices'
    path_to_transcript = build_path_to_transcript_african_voices_template(f'{main_root}/hau_cmv_f')
    path_to_transcript.update(build_path_to_transcript_african_voices_template(f'{main_root}/hau_cmv_m'))
    return path_to_transcript


def build_path_to_transcript_fleurs_template(root):
    path_to_transcript = dict()

    for split in ['train', 'dev', 'test']:
        with open(Path(root, f'{split}.tsv'), 'r', encoding='utf-8') as f:
            reader = DictReader(f, delimiter='\t', fieldnames=['id', 'filename', 'transcription_raw',
                                                               'transcription', 'words', 'speaker', 'gender'])
            for row in reader:
                path_to_transcript[str(Path(root, 'audio', split, row['filename']))] = row['transcription_raw'].strip()

    return path_to_transcript

def build_path_to_transcript_fleurs_fula():
    root = '/resources/speech/corpora/fleurs/ff_sn'
    return build_path_to_transcript_fleurs_template(root)


def build_path_to_transcript_mms_template(lang, root='/resources/speech/corpora/mms_synthesized_bible_speech'):
    path_to_transcript = dict()

    i = 0
    with open(Path(root, 'bible_texts', f'{lang}.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            path = Path(root, 'bible_audios', lang, f'{i}.wav')
            if path.exists():
                path_to_transcript[str(path)] = line.strip()
                i += 1

    return path_to_transcript


if __name__ == '__main__':
    pass
