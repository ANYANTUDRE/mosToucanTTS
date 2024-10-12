import os
import torch
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface


def read_texts(sentence, filename, model_id=None, device="cpu", language="mos", speaker_reference=None, duration_scaling_factor=1.0):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename, duration_scaling_factor=duration_scaling_factor, prosody_creativity=0.0)
    del tts


def moore_test(version, model_id=None, exec_device="cpu", speaker_reference=None):
    os.makedirs("audios", exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["""M yet-y lame tɩ kuga sẽn dɩgã, Wẽnd tõe n yiisa a Abraam kamb b pʋsẽ. 
                            Lar zoe n dɩga tɩɩga sɛɛga; tɩɩg ninga fãa sẽn pa womda bi-sõma, 
                            b na n kɛɛg-a lame n yõog bugum. 11Mam yẽ kõt-y la soob koomẽ y zʋg 
                            tekr yĩnga la mam sẽn deng a soab taoorã yɩɩda maam; paam n lodg a 
                            neodr gĩs menga looga maam; yẽ na n soo yãmb ne a Vʋʋsem Sõng la bugum. 
                            A zãada kaorg a nugẽ n na n yelg a kood zɛɛgẽ; 
                            ka-bɛɛla na n suii baoorẽ, la fɩɩba na n yõoga bugum ning sẽn ka kiida pʋgẽ."""],
               filename=f"audios/{model_id}_english_test_{version}.wav",
               device=exec_device,
               language="mos",
               speaker_reference=speaker_reference)


def all_test(version, model_id=None, exec_device="cpu", speaker_reference=None):
    moore_test(version, model_id, exec_device, speaker_reference)
    #french_test(version, model_id, exec_device, speaker_reference)


if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")

    os.makedirs("audios/speaker_references/", exist_ok=True)
    merged_speaker_references = ["audios/speaker_references/" + ref for ref in os.listdir("audios/speaker_references/")]

    moore_test( version="version_11",
                model_id="mos_ToucanTTS",  # use the finetuned model
                exec_device=exec_device,
                speaker_reference=merged_speaker_references if merged_speaker_references != [] else None)