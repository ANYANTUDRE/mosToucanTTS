# Training / Finetuning TTS model using IMS Toucan

## What is IMS Toucan?
IMS Toucan is a **toolkit for training, using, and teaching state-of-the-art Text-to-Speech Synthesis**, developed at the Institute for Natural Language Processing (IMS), University of Stuttgart, Germany, official home of the massively multilingual ToucanTTS system. 

<br>

![image](https://github.com/DigitalPhonetics/IMS-Toucan/blob/MassiveScaleToucan/Utility/toucan.png)

--- 
<br>


## Features
- Support for Text-to-Speech models for over **7000 Languages**.
- The system is **fast, controllable, and doesn't require a ton of compute.**
- Even if **this repo shows the finetuning for Moore language**, it can be adapted for training/finetuning for any language.

--- 
<br>


## I. Installation 🦉

### Basic Requirements

- **Python recommended version:** `3.10`
- **Hardware requirements:** 
  - **For training / finetuning:** you should have at least one cuda enabled GPU.
  - **For inference:** you don't need a GPU.

- **System/OS packages requirements:**   
If you're using Linux, you should have the following packages installed (on most distributions they come pre-installed):

```
libsndfile1
espeak-ng
ffmpeg
libasound-dev
libportaudio2
libsqlite3-dev
```

### Instructions for installation
To install this toolkit, follow these instructions step by step. 
These assume that your working in a **Linux-based environment like Lighning AI**.

1. **Clone the repository and navigate to the cloned directory:**
   ```bash
      git clone https://github.com/ANYANTUDRE/mosToucanTTS-.git
      cd mosToucanTTS-
   ```

2. **Create and activate** a [**virtual environment**](https://docs.python.org/3/library/venv.html):
    ```bash
      python -m venv .env
      source .env/bin/activate
    ```

3. **Install System/OS packages:**
    ```bash
      sudo su
      sudo apt-get install libsndfile1 espeak-ng ffmpeg libasound-dev libportaudio2 libsqlite3-dev
    ```

4. **Install all the requirements / librairies:**
   ```bash
     pip install --no-cache-dir -r requirements.txt
   ```


### Storage configuration
If you don't want the pretrained and trained models as well as the cache files resulting from preprocessing your
datasets to be stored in the default subfolders, you can set corresponding directories globally by editing `Utility/storage_config.py` to suit your needs (the path can be relative to the repository root directory or absolute).

--- 
<br>


## II. Dataset Preparation
The following steps assume that your dataset is hosted on Hugging Face Hub:

1. **First create a `.env` file where you'll store your Hugging Face token:**   
Here is what it should contain:
   ```bash
      HF_TOKEN="your_hugging_face_token_here"
   ```

2. **Download dataset from Hugging Face and create metadata files**:   
Open the `moore_data_prep.py` script and modify `HF_DATASET_NAME` variable to the corresponding dataset id and `HF_DATASET_AUDIO_COLUMN` to corresponding audio colum. You may also modify the other variables to suit your needs.   
Then run:
   ```bash
      python moore_data_prep.py
   ```

After this is executed correctly, yould see a `dataset` folder where there are all the audio files and a metadata file with transcriptions.

3. **Creating a new Recipe🐣:**   
In the directory called *Utility* there is a file called
`path_to_transcript_dicts.py`. 


In this file you should write a **function that returns a dictionary** that has all the absolute paths to each of the audio files in your dataset as strings as the keys and the textual transcriptions of the corresponding audios as the values.  
 
For reference, check ` moore_audio_metadata_to_dict()` function for Moore language.


## III. Training / Finetuning Pipeline

### Finetuning a pretrained model
Again, follow these instructions:

- **Step 1: Creating a new Recipe 🐣**   
  Go to the directory *TrainingInterfaces/Recipes*. In there, make a copy of the `moore_simple_finetuning.py`.

  We will use this copy as reference and only make the necessary changes to use the new dataset. 
  - Find the call(s) to the `prepare_tts_corpus` function. Replace the `path_to_transcript_dict` used there with the one(s) you just created. Then change the name of the corresponding cache directory to something that makes sense for the dataset.
  - Also look out for the variable `save_dir`, which is where the checkpoints will be saved to. This is a default value; you can overwrite it when calling the pipeline later using a command line argument in case you want to fine-tune from a checkpoint and thus save into a different directory.
  - Finally, change the `lang` argument in the creation of the dataset and in the call to the train loop function to the **ISO 639-3 language ID** (like `mos` in our case) that matches your data.


- **Step 2: Adapt `run_training_pipeline.py` script**   
Once this is complete, we are almost done, now we just need to make it available to the `run_training_pipeline.py` file in the top level. 

In said file, import the `run` function from the pipeline you just created and give it a meaningful name. Now in the `pipeline_dict`, add your imported function as value and use as key a shorthand that makes sense.


- **Step 3: Training a Model 🦜**   
Once you have a recipe built, training is super easy:
  ```bash
  python run_training_pipeline.py <shorthand of the pipeline>
  ```

You can supply any of the following arguments, but don't have to (although for training you should definitely specify at least a GPU ID).

  ```bash
  --gpu_id <ID of the GPU you wish to use, as displayed with nvidia-smi, default is cpu. If multiple GPUs are provided (comma separated), then distributed training will be used, but the script has to be started with torchrun.> 

  --resume_checkpoint <path to a checkpoint to load>

  --resume (if this is present, the furthest checkpoint available will be loaded automatically)

  --finetune (if this is present, the provided checkpoint will be fine-tuned on the data from this pipeline)

  --model_save_dir <path to a directory where the checkpoints should be saved>

  --wandb (if this is present, the logs will be synchronized to your weights&biases account, if you are logged in on the command line)

  --wandb_resume_id <the id of the run you want to resume, if you are using weights&biases (you can find the id in the URL of the run)>
  ```

**For multi-GPU training**, you have to supply multiple GPU ids (comma separated) and start the script with torchrun. You also have to specify the number of GPUs. This has to match the number of IDs that you supply.
  ```bash
  torchrun --standalone --nproc_per_node=4 --nnodes=1 run_training_pipeline.py <shorthand of the pipeline> --gpu_id "0,1,2,3"
  ```

After every epoch (or alternatively after certain step counts), some logs will be written to the console and to the Weights and Biases website, if you are logged in and set the flag. 


### Training a model from scratch
If you want to train from scratch, have a look at a `moore_training_from_scratch.py` and look at the arguments used there.  

The training steps are almost similar to the above ones.



### **Some notes:** 
- If you get **cuda out of memory errors, you need to decrease the `*batchsize*`** in the arguments of the call to the `training_loop` in the pipeline you are running. Try decreasing the `*batchsize*` in small steps until you get no more out of cuda memory errors.

- In the directory you specified for saving, **checkpoint files and spectrogram visualization**
data will appear. Since the checkpoints are quite big, only the five most recent ones will be kept. 

- The amount of **training steps highly depends on the data** you are using and whether you're finetuning from a pretrained checkpoint or training from scratch. The fewer data you have, the fewer steps you should take to prevent a possible collapse. If you want to stop earlier, just kill the process, since everything is daemonic all the child-processes should die with it. 

- In case there are some ghost-processes left behind, you can use the following command to find them and kill them manually.
  ```bash
    fuser -v /dev/nvidia*
  ```

- Whenever a checkpoint is saved, a compressed version that can be used for inference is also created, which is named `*best.py*`

--- 
<br>


## IV. Inference 🦢
You can load your trained models, or the pretrained provided one, using the `InferenceInterfaces/ToucanTTSInterface.py`.
Simply create an object from it with the proper directory handle identifying the model you want to use. The rest should work out in the background. You might want to set a language
embedding or a speaker embedding using the *set_language* and *set_speaker_embedding* functions. Most things should be self-explanatory.

An *InferenceInterface* contains two methods to create audio from text. They are *read_to_file* and
*read_aloud*.

- *read_to_file*: takes as input a list of strings and a filename. It will synthesize the sentences in the list and concatenate them with a short pause inbetween and write them to the filepath you supply as the other argument.

- *read_aloud*: takes just a string, which it will then convert to speech and immediately play using the system's speakers. If you set the optional argument *view* to *True*, a visualization will pop up, that you need to close for the program to continue.

The first method use is demonstrated in *run_text_to_file_reader.py*.

There are simple scaling parameters to control the duration, the variance of the pitch curve and the variance of the energy curve. You can either change them in the code when using the interactive demo or the reader, or you can simply pass them to the interface when you use it in your own code.

To change the language of the model and see which languages are available in our pretrained model,
[have a look at the list linked here](https://github.com/DigitalPhonetics/IMS-Toucan/blob/feb573ca630823974e6ced22591ab41cdfb93674/Utility/language_list.md)

--- 
<br>


## V. References / Links 🦚 🐧

- [IMS-Toucan GitHub repository](https://github.com/DigitalPhonetics/IMS-Toucan/tree/MassiveScaleToucan)

- [Interactive massively-multi-lingual demo on Hugging Face🤗](https://huggingface.co/spaces/Flux9665/MassivelyMultilingualTTS)

- Introducing the first TTS System in over 7000 languages [[associated code and models]](https://github.com/DigitalPhonetics/IMS-Toucan/releases/tag/v3.0)

```
@inproceedings{lux2024massive,
  year         = 2024,
  title        = {{Meta Learning Text-to-Speech Synthesis in over 7000 Languages}},
  author       = {Florian Lux and Sarina Meyer and Lyonel Behringer and Frank Zalkow and Phat Do and Matt Coler and  Emanuël A. P. Habets and Ngoc Thang Vu},
  booktitle    = {Interspeech}
  publisher    = {ISCA}
}
```
