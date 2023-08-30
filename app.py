# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import time
from matplotlib import cm
import soundfile as sf
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import streamlit as st
import tempfile
import noisereduce as nr
import pyaudio
import wave
import whisper
from transformers import (
    HubertForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    AutoModel,
    AutoTokenizer,
    HubertForSequenceClassification,
    AutoModelForCausalLM
)
from streamlit.components.v1 import html

# Mapping Hubert model's output to GPT input
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

# Define the model's name from the Hugging Face model hub
model_weights_path = "https://huggingface.co/netgvarun2005/MultiModalBertHubert/resolve/main/MultiModal_model_state_dict.pth"

# Model name initialization
model_id = "facebook/hubert-base-ls960"
bert_model_name = "bert-base-uncased"


def open_page(url):
    """ 
    Function to invoke javascript code to redirect to an external URL.

    Parameters: 
        External URL to redirect to.

    Returns: 
        None
    """ 
    open_script= """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)

def config():
    """
    Configure the Streamlit application settings and styles.

    This function sets the page configuration, including the title and icon, adds custom CSS styles
    for specific elements, and defines a custom style for the application title.

    Parameters:
        None

    Returns:
        None
    """
    # Loading Image using PIL
    im = Image.open('./config/icon.png')
    
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

@st.cache_resource(show_spinner=False)
def speechtoText(wavfile):
    """
    Convert speech from a WAV audio file to text using a pre-trained Whisper ASR model.

    This function takes a WAV audio file as input and utilizes a pre-trained Whisper ASR model
    to transcribe the speech into text.

    Parameters:
        wavfile (str): The file path to the input WAV audio file.

    Returns:
        str: The transcribed text from the speech in the audio file.
    """
    return speech_model.transcribe(wavfile)['text']

def resampleaudio(wavfile):
    """
    Resample an audio file to a target sample rate and save it back to the same file.

    This function loads an audio file in WAV format, resamples it to the specified target sample rate,
    and then saves the resampled audio back to the same file, overwriting the original content.

    Parameters:
        wavfile (str): The file path to the input WAV audio file.

    Returns:
        str: The file path to the resampled WAV audio file.
    """
    audio, sr = librosa.load(wavfile, sr=None)

    # Set the desired target sample rate
    target_sample_rate = 16000

    # Resample the audio to the target sample rate
    resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
   
    # Write to the original file 
    sf.write(wavfile,resampled_audio, target_sample_rate)
    return wavfile


def noiseReduction(wavfile):
    """
    Apply noise reduction to an audio file and save the denoised audio back to the same file.

    This function loads an audio file in WAV format, performs noise reduction using the specified parameters,
    and then saves the denoised audio back to the same file, overwriting the original content.

    Parameters:
        wavfile (str): The file path to the input WAV audio file.

    Returns:
        str: The file path to the denoised WAV audio file.
    """
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
    """
    Remove silence from an audio file and save the trimmed audio back to the same file.

    This function loads an audio file in WAV format, identifies and removes silence based on a specified threshold,
    and then saves the trimmed audio back to the same file, overwriting the original content.

    Parameters:
        wavfile (str): The file path to the input WAV audio file.

    Returns:
        str: The file path to the audio file with silence removed.
    """
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
    """
    Perform a series of audio preprocessing steps on a WAV file.

    This function takes an input WAV audio file, applies a series of preprocessing steps,
    including resampling, noise reduction, and silence removal, and returns the path to the
    preprocessed audio file.

    Parameters:
        wavfile (str): The file path to the input WAV audio file.

    Returns:
        str: The file path to the preprocessed WAV audio file.
    """
    resampledwavfile = resampleaudio(wavfile)
    denoised_file = noiseReduction(resampledwavfile)
    return removeSilence(denoised_file)

@st.cache_resource()
def load_model():
    """
    Load and configure various models and tokenizers for a multi-modal application.

    This function loads a multi-modal model and its weights from a specified source,
    initializes tokenizers for the model and an additional language model, and returns
    these components for use in a multi-modal application.

    Returns:
        tuple: A tuple containing the following components:
            - multiModel (MultimodalModel): The multi-modal model.
            - tokenizer (AutoTokenizer): Tokenizer for the multi-modal model.
            - model_gpt (AutoModelForCausalLM): Language model for text generation.
            - tokenizer_gpt (AutoTokenizer): Tokenizer for the language model.
    """
    # Load the model
    multiModel = MultimodalModel(bert_model_name, num_labels)

    # Load the model weights directly from Hugging Face Spaces
    multiModel.load_state_dict(torch.hub.load_state_dict_from_url(model_weights_path, map_location=device), strict=False)

   # multiModel.load_state_dict(torch.load(file_path + "/MultiModal_model_state_dict.pth",map_location=device),strict=False)
    tokenizer = AutoTokenizer.from_pretrained("netgvarun2005/MultiModalBertHubertTokenizer") 

    # GenAI
    tokenizer_gpt = AutoTokenizer.from_pretrained("netgvarun2005/GPTTherapistDeepSpeedTokenizer", pad_token='<|pad|>',bos_token='<|startoftext|>',eos_token='<|endoftext|>')
    model_gpt = AutoModelForCausalLM.from_pretrained("netgvarun2005/GPTTherapistDeepSpeedModel")
   
    return multiModel,tokenizer,model_gpt,tokenizer_gpt


def predict(audio_array,multiModal_model,key,tokenizer,text):    
    """
    Perform multimodal prediction using an audio feature array and text input.

    This function takes an audio feature array and text as input, tokenizes the text,
    extracts audio features, and uses a multi-modal model to predict a class label based on
    the combined audio and text inputs.

    Parameters:
        audio_array (numpy.ndarray): A numpy array containing audio features.
        multiModal_model: The multi-modal model for prediction.
        key: A key for identifying the model (e.g., model_id).
        tokenizer: Tokenizer for processing the text input.
        text (str): The input text for prediction.

    Returns:
        str: The predicted class label.
    """
    # Tokenize the input text
    input_text = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Extract audio features using a feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    input_audio = feature_extractor(
        raw_speech=audio_array,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )

    # Make predictions with the multi-modal model
    logits = multiModal_model(input_audio["input_values"], input_text["input_ids"])

    # Calculate class probabilities
    probabilities = F.softmax(logits, dim=1).to_dense()
    _, predicted = torch.max(probabilities, 1)
    class_prob = probabilities.tolist()
    class_prob = class_prob[0]
    class_prob = [round(value, 2) for value in class_prob]
    maxVal = np.argmax(class_prob)

    # Display the final transcript and handle inference issues
    if label_mapping[predicted] == "":
        st.write("Inference impossible, a problem occurred with your audio or your parameters, we apologize :(")

    return (label_mapping[maxVal]).capitalize()

def GenerateText(emo,gpt_tokenizer,gpt_model):
    """
    Generate text based on a given emotion using a GPT-2 model.

    This function takes an emotion as input, generates text based on the emotion prompt,
    and displays multiple generated text samples.

    Parameters:
        emo (str): The emotion for which text should be generated.
        gpt_tokenizer: Tokenizer for processing the GPT-2 model input.
        gpt_model: The GPT-2 model for text generation.

    Returns:
        None
    """
    # Create a prompt based on the input emotion
    prompt  = f'<startoftext>{emo2promptMapping[emo]}:' 

    # Tokenize the prompt and convert it to input tensors
    generated = gpt_tokenizer(prompt, return_tensors="pt").input_ids

    # Move the generated tensor and GPT model to the specified device (e.g., GPU)
    generated = generated.to(device)
    gpt_model.to(device)

    # Generate multiple text samples based on the prompt
    sample_outputs = gpt_model.generate(generated, do_sample=True, top_k=50,
                                    max_length=30, top_p=0.95, temperature=1.1, num_return_sequences=10)#,no_repeat_ngram_size=1)

    # Extract and split the generated text into words
    outputs = set([gpt_tokenizer.decode(sample_output, skip_special_tokens=True).split(':')[-1] for sample_output in sample_outputs])        

    # Display the generated text samples with a delay for readability
    for i, sample_output in enumerate(outputs):
        st.write(f"<span style='font-size: 18px; font-family: Arial, sans-serif; font-weight: bold;'>{i+1}: {sample_output}</span>", unsafe_allow_html=True)
        time.sleep(0.5)        


def process_file(ser_model,tokenizer,gpt_model,gpt_tokenizer):
    """
    Process and analyze an uploaded WAV file, generating transcriptions and helpful tips.

    This function allows users to upload a WAV audio file, processes the file to obtain transcriptions,
    predicts the user's emotional state, and displays helpful tips based on the predicted emotion.

    Parameters:
        ser_model: The emotion analysis model for predicting emotions.
        tokenizer: Tokenizer for processing text inputs.
        gpt_model: The GPT-3 model for generating text.
        gpt_tokenizer: Tokenizer for processing GPT-3 model inputs.

    Returns:
        None
    """
    emo = ""
    button_label = "Show Helpful Tips"
    uploaded_file = st.file_uploader("Upload your file! It should be .wav", type=["wav"])

    if uploaded_file is not None:
        # Read the content of the uploaded file
        audio_content = uploaded_file.read()
        # Display audio file
        st.audio(audio_content, format="audio/wav")

        # Save the audio content to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            #print(f'temp_filename is {temp_filename}\n')
            temp_file.write(audio_content)

            try:
                audio_array, sr = librosa.load(preprocessWavFile(temp_filename), sr=None)
                with st.spinner(st.markdown("<p style='font-size: 16px; font-weight: bold;'>Generating transcriptions in the side pane! Please wait...</p>", unsafe_allow_html=True)):
                    transcription = speechtoText(temp_filename)
                    emo = predict(audio_array,ser_model,2,tokenizer,transcription)
                    # Display the transcription in a textbox
                    st.sidebar.text_area("Transcription", transcription, height=25)      
            except:
                st.write("Inference impossible, a problem occurred with your audio or your parameters, we apologize :(")
  

            txt = f"You seem to be <b>{(emo2promptMapping[emo]).capitalize()}!</b>\n Click on 'Show Helpful Tips' button to proceed further."
            st.markdown(f"<div class='mobile-screen' style='font-size: 24px;'>{txt} </div>", unsafe_allow_html=True)

            # Store the value of emo in the session state
            st.session_state.emo = emo
            if st.button(button_label):
                with st.spinner(st.markdown("<p style='font-size: 16px; font-weight: bold;'>Generating tips (it may take upto 3-4 mins depending upon network speed). Please wait...</p>", unsafe_allow_html=True)):
                    # Retrieve prompt from the emotion
                    emo = st.session_state.emo
                    # Call the function for GENAI
                    GenerateText(emo,gpt_tokenizer,gpt_model)

def main():
    """
    Main function for running a Streamlit-based multi-modal text generation application.

    This function configures the Streamlit application, loads necessary models and tokenizers,
    and allows users to process audio files to generate transcriptions and helpful tips.

    Returns:
        None
    """
    config()
    if st.sidebar.button("**Open External Audio Recorder!**"):
        open_page("https://voice-recorder-online.com/")

    # Load the models, and tokenizers
    ser_model,tokenizer,gpt_model,gpt_tokenizer = load_model()

    # Process and analyze uploaded audio files
    process_file(ser_model,tokenizer,gpt_model,gpt_tokenizer)


if __name__ == '__main__':
    main()
