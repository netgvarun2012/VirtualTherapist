import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import time
from matplotlib import cm
import soundfile as sf
import sounddevice as sd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torch.nn.functional as F
import streamlit as st
import tempfile
import noisereduce as nr
import altair as alt
import pyaudio
import wave
import whisper
from transformers import (
    HubertForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    AutoModel,
    AutoTokenizer,
    HubertForSequenceClassification
)
from transformers import AutoTokenizer, AutoModelForCausalLM
  
emo2promptMapping = {
'Angry':'ANGRY', 
'Calm':'CALM',
'Disgust':'DISGUSTED', 
'Fearful':'FEARFUL',
'Happy': 'HAPPY', 
'Sad': 'SAD',
'Surprised': 'SURPRISED'
}

# Check if GPU (cuda) is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#Load speech to text model
speech_model = whisper.load_model("base")

#Define Labels related info
num_labels=7
label_mapping = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'sad', 'surprised']

# Get the path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create the path to the file in the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, "../EmotionDetector/Models/"))
#file_path = os.path.join(parent_dir, "MultiModal/MultiModal_model_state_dict.pth")

# Define your model name from the Hugging Face model hub
model_weights_path = "https://huggingface.co/netgvarun2005/MultiModalBertHubert/blob/main/MultiModal_model_state_dict.pth"

# GenAI model
parent_dir2 = os.path.abspath(os.path.join(current_dir, "../GenAI/"))


# Emo Detector
model_id = "facebook/hubert-base-ls960"
bert_model_name = "bert-base-uncased"
tokenizerDir = os.path.join(parent_dir, 'Tokenizer\\')


def config():
    # Loading Image using PIL
    im = Image.open('./icon.png')
    
    # Set the page configuration with the title and icon
    st.set_page_config(page_title="Virtual Therapist", page_icon=im)
    
    # Add custom CSS styles
    st.markdown("""
        <style>
        .mobile-screen {
            border: 2px solid black;            
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start; /* Align content to the top */
            height: 20vh;
            padding: 20px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    # Render mobile screen container and its content
    st.sidebar.title("Sound Recorder")

    # Define a custom style for your title
    title_style = """
    <style>
        h1 {
            font-family: 'Comic Sans MS', cursive, sans-serif;
            color: blue;
            font-size: 22px; /* Add font size here */
        }
    </style>
    """

    # Display the title with the custom style
    st.markdown(title_style, unsafe_allow_html=True)
    st.markdown("# WELCOME! HOW ARE YOU FEELING? PLEASE RECORD AN AUDIO!", unsafe_allow_html=True)
    st.markdown("# BASED ON YOUR EMOTIONAL STATE, I WILL SUGGEST SOME TIPS!", unsafe_allow_html=True)

    
    return


class MultimodalModel(nn.Module):
    '''
    Custom PyTorch model that takes as input both the audio features and the text embeddings, and concatenates the last hidden states from the Hubert and BERT models.
    '''
    def __init__(self, bert_model_name, num_labels):
        super().__init__()
        self.hubert = HubertForSequenceClassification.from_pretrained("netgvarun2005/HubertStandaloneEmoDetector", num_labels=num_labels).hubert
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.hubert.config.hidden_size + self.bert.config.hidden_size, num_labels)

    def forward(self, input_values, text):
        hubert_output = self.hubert(input_values).last_hidden_state

        bert_output = self.bert(text).last_hidden_state

        # Apply mean pooling along the sequence dimension
        hubert_output = hubert_output.mean(dim=1)
        bert_output = bert_output.mean(dim=1)

        concat_output = torch.cat((hubert_output, bert_output), dim=-1)
        logits = self.classifier(concat_output)
        return logits


def speechtoText(wavfile):
    return speech_model.transcribe(wavfile)['text']

def resampleaudio(wavfile):
    audio, sr = librosa.load(wavfile, sr=None)

    # Set the desired target sample rate
    target_sample_rate = 16000

    # Resample the audio to the target sample rate
    resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
    
    sf.write(wavfile,resampled_audio, target_sample_rate)
    return wavfile


def noiseReduction(wavfile):
    audio, sr = librosa.load(wavfile, sr=None)

    # Set parameters for noise reduction
    n_fft = 2048  # FFT window size
    hop_length = 512  # Hop length for STFT

    # Perform noise reduction
    reduced_noise = nr.reduce_noise(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # Save the denoised audio to a new WAV file
    sf.write(wavfile,reduced_noise, sr)
    return wavfile


def removeSilence(wavfile):
    # Load the audio file
    audio_file = wavfile

    audio, sr = librosa.load(audio_file, sr=None)

    # Split the audio file based on silence
    clips = librosa.effects.split(audio, top_db=40)

    # Combine the audio clips
    non_silent_audio = []
    for start, end in clips:
        non_silent_audio.extend(audio[start:end])


    # Save the audio without silence to a new WAV file
    sf.write(wavfile,non_silent_audio, sr)
    return wavfile

def preprocessWavFile(wavfile):
    resampledwavfile = resampleaudio(wavfile)
    denoised_file = noiseReduction(resampledwavfile)
    return removeSilence(denoised_file)

@st.cache(allow_output_mutation=True)
def load_model():
    # Load the model
    multiModel = MultimodalModel(bert_model_name, num_labels)

    # Load the model weights directly from Hugging Face Spaces
    multiModel.load_state_dict(torch.hub.load_state_dict_from_url(model_weights_path, map_location=device), strict=False)

   # multiModel.load_state_dict(torch.load(file_path + "/MultiModal_model_state_dict.pth",map_location=device),strict=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizerDir) 

    # GenAI
    tokenizer_gpt = AutoTokenizer.from_pretrained(os.path.join(parent_dir2,"Tokenizer"), pad_token='<|pad|>',bos_token='<|startoftext|>',eos_token='<|endoftext|>')
    model_gpt = AutoModelForCausalLM.from_pretrained("netgvarun2005/GPTVirtualTherapist")
   
    return multiModel,tokenizer,model_gpt,tokenizer_gpt


def predict(audio_array,multiModal_model,key,tokenizer,text):    
    input_text = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    input_audio = feature_extractor(
        raw_speech=audio_array,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )
    logits = multiModal_model(input_audio["input_values"], input_text["input_ids"])

    probabilities = F.softmax(logits, dim=1).to_dense()
    _, predicted = torch.max(probabilities, 1)
    class_prob = probabilities.tolist()
    class_prob = class_prob[0]
    class_prob = [round(value, 2) for value in class_prob]
    maxVal = np.argmax(class_prob)

    # Display the final transcript
    if label_mapping[predicted] == "":
        st.write("Inference impossible, a problem occurred with your audio or your parameters, we apologize :(")

    return (label_mapping[maxVal]).capitalize()

def record_audio(output_file, duration=5):
    # st.sidebar.markdown("Recording...")
    sd.wait()  # Wait for microphone to start
    sd.wait()  # Wait for microphone to start
    time.sleep(0.4)

    st.sidebar.markdown("<p style='font-size: 14px; font-weight: bold;'>Recording...</p>", unsafe_allow_html=True)

    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 2
    fs = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []

    for _ in range(int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    time.sleep(0.5)
    # st.sidebar.markdown("Recording finished!")   
    st.sidebar.markdown("<p style='font-size: 14px; font-weight: bold;'>Recording finished!</p>", unsafe_allow_html=True)

    time.sleep(0.5)
 
def GenerateText(emo,gpt_tokenizer,gpt_model):
    prompt  = f'<startoftext>{emo2promptMapping[emo]}:' 

    generated = gpt_tokenizer(prompt, return_tensors="pt").input_ids

    sample_outputs = gpt_model.generate(generated, do_sample=True, top_k=50,
                                    max_length=20, top_p=0.95, temperature=0.2, num_return_sequences=10,no_repeat_ngram_size=1)

    # Extract and split the generated text into words
    outputs = set([gpt_tokenizer.decode(sample_output, skip_special_tokens=True).split(':')[-1] for sample_output in sample_outputs])        
    for i, sample_output in enumerate(outputs):
        st.write(f"<span style='font-size: 18px; font-family: Arial, sans-serif; font-weight: bold;'>{i+1}: {sample_output}</span>", unsafe_allow_html=True)
        time.sleep(0.5)        


def process_file(ser_model,tokenizer,gpt_model,gpt_tokenizer):
    emo = ""
    button_label = "Show Helpful Tips"
    recorded = False  # Initialize the recording state as False

    if 'stage' not in st.session_state:
        st.session_state.stage = 0

    def set_stage(stage):
        st.session_state.stage = stage

   # Add custom CSS styles
    st.markdown("""
        <style>
            .stRecordButton {
                width: 50px;
                height: 50px;
                border-radius: 50px;
                background-color: red;
                color: black; /* Text color */
                font-size: 16px;
                font-weight: bold;
                border: 2px solid white; /* Solid border */
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                cursor: pointer;
                transition: background-color 0.2s;
                display: flex;
                justify-content: center;
                align-items: center;
            }

            .stRecordButton:hover {
                background-color: darkred; /* Change background color on hover */
            }
        </style>
    """, unsafe_allow_html=True)

    if st.sidebar.button("Record a 4 sec audio!", key="record_button", help="Click to start recording", on_click=set_stage, args=(1,)):
    # Your button click action here

        # Apply bold styling to the button label
        st.sidebar.markdown("<span style='font-weight: bolder;'>Record a 4 sec audio!</span>", unsafe_allow_html=True)


        # recorded = True  # Set the recording state to True after recording

        # Add your audio recording code here
        output_wav_file = "output.wav"
        record_audio(output_wav_file, duration=4)
        
        # # Use a div to encapsulate the audio element and apply the border
        with st.sidebar.markdown('<div class="audio-container">', unsafe_allow_html=True):
            # Play recorded sound
            st.audio(output_wav_file, format="wav")    

        audio_array, sr = librosa.load(preprocessWavFile(output_wav_file), sr=None)
        st.sidebar.markdown("<p style='font-size: 14px; font-weight: bold;'>Generating transcriptions! Please wait...</p>", unsafe_allow_html=True)

        transcription = speechtoText(output_wav_file)
        
        emo = predict(audio_array,ser_model,2,tokenizer,transcription)
        
        # Display the transcription in a textbox
        st.sidebar.text_area("Transcription", transcription, height=25)        

        txt = f"You seem to be <b>{(emo2promptMapping[emo]).capitalize()}!</b>\n Click on 'Show Helpful Tips' button to proceed further."
        st.markdown(f"<div class='mobile-screen' style='font-size: 24px;'>{txt} </div>", unsafe_allow_html=True)

        # Store the value of emo in the session state
        st.session_state.emo = emo

    if st.session_state.stage > 0:
        if st.button(button_label,on_click=set_stage, args=(2,)):
            # Retrieve prompt from the emotion
            emo = st.session_state.emo
            GenerateText(emo,gpt_tokenizer,gpt_model)  

if __name__ == '__main__':
    config()
    ser_model,tokenizer,gpt_model,gpt_tokenizer = load_model()
    process_file(ser_model,tokenizer,gpt_model,gpt_tokenizer)
