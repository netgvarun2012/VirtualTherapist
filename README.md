---
title: VirtualTherapist
emoji: ⚡
colorFrom: yellow
colorTo: blue
sdk: streamlit
sdk_version: 1.26.0
app_file: app.py
pinned: false
license: mit
---

# VIRTUAL THERAPIST - INTELLIGENT EMOTION RECOGNIZER AND THERAPEUTIC MESSAGES GENERATOR.

Check out the app at https://huggingface.co/spaces/netgvarun2005/VirtualTherapist


# Table of Contents
  * [Introduction](#introduction)
  * [Demo Videos](#demos)
  * [Setup and Requirements](#installation)
  * [Working Pipeline](#pipeline)
  * [Tutorials](#tutorials)
  * [Dataset](#dataset)
  * [Usage](#usage)

# Introduction <a id="introduction"></a>

This repository details the technical and functional aspects of *'Virtual Therapist'* app - an Intelligent speech and text input based assistant that can decipher emotions and generate **therapeutic messages** based on the Emotional state of the user.

Emotions recognized - *Angry*,*Sad*,*Fearful*,*Happy*,*Disgusted*,*Surprised*,*Calm* with ~80% accuracy.

Generative AI finetuning of GPTNeo 1.3B parameter model - perplexity obtained 1.502 corresponding to the training loss of '0.407200'.


# Reocrded Demos <a id="demos"></a>

https://github.com/netgvarun2012/VirtualTherapist/assets/93938450/f5cfe70a-ae62-43dd-a8c2-752d5c47dad5

https://github.com/netgvarun2012/VirtualTherapist/assets/93938450/b52c4244-0b8c-4a88-9034-a940e4df8bd2

https://github.com/netgvarun2012/VirtualTherapist/assets/93938450/ce71ea03-3fe1-46eb-a7d2-36676860fc6c

# Setup and Requirements <a id="installation"></a>
For a list of required linux packages see the *packages.txt*
or just install them all at once using pip.
```
sudo apt install packages.txt
```

For a list of required python packages see the *requirements.txt*
or just install them all at once using pip.
```
pip install -r requirements.txt
```

### Application Features:
- Using the web-app, a user can upload an audio snippet or record an audio snippet (extenal website link provided).
- User can playback the recorded/uploaded audio.
- AI in the background upon rececing the audio input generates english transcriptions in real-time.
- MultiModal AI system then uses both the Speech input and the Text input to predict emotion of the user out of 7 pre-defined classes of **"Anger", "Sad", "Disgusted", "Happy", "Calm", "Surprised" and "Fearful"**.
- After deciphering the emotion, Generative AI component of the system takes the predicted emotion labels as prompt and generates helpful tips accordingly.
- There is an added feature of choosing between "Balanced" and "Creative" one liner recommendations.

### Methodology:

- In terms of GEN-AI component, following was achieved:
  - A dataset was created manually by requesting CHATGPT to generate one-liner tips based on 7 different emotion categories.
  - This dataset was augmented using [pegasus_paraphrase model](https://huggingface.co/tuner007/pegasus_paraphrase) to generate paraphrased instances.
  - [GPT-Neo 1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B) model was fine-tuned using Emotion labels as prompts and pytorch as backend.
  - [Deep Speed](https://github.com/microsoft/DeepSpeed) library was utilized to optimize and speed up training of 1.3Billion parameters model.
    
- In terms of Speech procesing and modelling , following was achieved:
  - 5 publicly available speech Emotion datasets were concatenated to create a robust dataset.
  - [Librosa](https://librosa.org/) library was used heavily to do preprocessing like **'Sample rate adjustment', 'Noise reduction', 'Silence removal' and 'Short audio removal'** on raw audio files.
  - [Whisper](https://github.com/openai/whisper) model was used to generate English language transcriptions of the preprocessed files.
  - [Hubeet](https://huggingface.co/docs/transformers/model_doc/hubert) model was fine-tuned with a classification head on preprocessed audio and emotion labels in supervised manner.
  - [BERT](https://huggingface.co/docs/transformers/model_doc/bert) was trained on text transcrition embeddings.
  - Finally, a MultiModal architecture was created and finetuned jointly by concatenating Hubert and BERT embeddings.
  - More information on the whole process can be found [here](https://github.com/netgvarun2012/VirtualTherapist/blob/main/documentation/Speech_and_Text_based_MultiModal_Emotion_Recognizer.pdf).

### Deployment:
  - Lightweight Streamlit app as a front-end was chosen due to its simplicity and rapid development capabilities, allowing for quick prototyping and user-friendly interaction with complex data and models.
  - All the models were deployed on Hugging Face's Model Hub, a platform known for its accessibility, scalability, and collaborative environment for sharing and accessing state-of-the-art models and solutions.
  - Finally, Hugging Face Spaces was used as the cloud hosting platform ,as it could host such big deep learning models in a very const effective manner.
  - Github repo was utilized for sharing files between local and huggingface repo.


## Highlights:
- This project demonstrates my past experience in working with Generative AI in realms of *'Finetuning'*, *'prompt engineering'* and *'deployment'*.
- I have developed industry ready apps like **'Predictive Writing'** (Please refer to my [Medium Blog](https://medium.com/@sharmavarun.cs/predictive-writing-using-gpt-transformer-a042d37f7fb3) on this).
- I have also worked extensively on exploring the customization of the core generative AI transformers libraries and other capabilities - for e.g.  integrated [GeDi](https://blog.salesforceairesearch.com/gedi/) model with GPT-neo model for coherence song lyrics generation while working for NUS computing and submitted a paper in ACMMM 23 conference.
- I have experience in using ChatGPT, Llama, Alpaca, T5 models for LLMs and other Natural language processing tasks.
- I have exposure to working with latest advances in LLM spaces such as using DeepSpeed library, LORA (low-rank adaption),QLORA, PEFT libraries to optimize finetuning of LLMs.
- This project also demonstrates my strong experience in working with Speech datasets and speech processing concepts.
- Also, it demoonstrates my expertise in working with Deep learning, machine learning and other data science concepts.
- I have used python scripts,streamlit, github,huggingface spaces to develop an EndtoEnd application demonstrating my deployment related skills as well.



    

