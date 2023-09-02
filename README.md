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

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


## Repo for a "VirtualTherapist" project utilizing Generative AI, Speech and Text Emotion detection modes to suggest helpful tips based on the gauged emotion of the user.

### Main Features:
- Using the web-app, a user can upload an audio snippet or record an audio snippet.
- User can playback the audio.
- AI in the background upon rececing the audio input generates english transcriptions in real-time.
- MultiModal AI system then uses both the Speech input and the Text input to predict emotion of the user out of 7 pre-defined classes of "Anger", "Sad", "Disgusted", "Happy", "Calm", "Surprised" and "Fearful".
- After deciphering the emotion, Generative AI component of the system takes the predicted emotion labels as prompt and generates helpful tips accordingly.
- There is an added feature of choosing between "Balanced" and "Creative" one liner recommendations.

### Methodology:

- In terms of GEN-AI component, following was achieved:
  - A dataset was created manually by requesting CHATGPT to generate one-liner tips based on 7 different emotion categories.
  - This dataset was augmented using [pegasus_paraphrase model](https://huggingface.co/tuner007/pegasus_paraphrase) to generate paraphrased instances.
  - [GPT-Neo 1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B) model was fine-tuned using Emotion labels as prompts.
  - [Deep Speed](https://github.com/microsoft/DeepSpeed) library was utilized to optimize and speed up training.
    
- In terms of Speech procesing and modelling , following was achieved:
  - 5 publicly available speech Emotion datasets were concatenated to create a robust dataset.
  - [Librosa](https://librosa.org/) library was used heavily to do preprocessing like 'Sample rate adjustment', 'Noise reduction', 'Silence removal' and 'Short audio removal' on raw audio files.
  - [Whisper](https://github.com/openai/whisper) model was used to generate English language transcriptions of the preprocessed files.
  - [Hubeet](https://huggingface.co/docs/transformers/model_doc/hubert) model was fine-tuned with a classification head on preprocessed audio and emotion labels in supervised manner.
  - [BERT](https://huggingface.co/docs/transformers/model_doc/bert) was trained on text transcrition embeddings.
  - Finally, a MultiModal architecture was created and finetuned jointly by concatenating Hubert and BERT embeddings.
  - More information on the whole process can be found [here](https://github.com/netgvarun2012/VirtualTherapist/blob/main/documentation/Speech_and_Text_based_MultiModal_Emotion_Recognizer.pdf).

### Deployment:
  - Lightweight Streamlit app as a front-end was chosen due to its simplicity and rapid development capabilities, allowing for quick prototyping and user-friendly interaction with complex data and models.
  - All the models were deployed on Hugging Face's Model Hub, a platform known for its accessibility, scalability, and collaborative environment for sharing and accessing state-of-the-art models and solutions.
  - Finally, Hugging Face Spaces was used as the cloud hosting platform , providing a convenient and collaborative environment for hosting, sharing, and showcasing the models, datasets, and applications in an accessible and user-friendly manner.
  - Github repo was utilized for sharing files between local and huggingface repo.
    
- The competition required the contestants to establish the stock timing model, spontaneously find the best trading opportunity to complete the trading and strive for the lowest overall trading cost of the stock.
- The competition offered 500 stocks, each stock must complete buy and sell of 100 shares a day, and each trading of the number of shares can be distributed freely.
- The trading time of each stock is from 9:30 to 11:30 and 13:00 to 15:00 daily.
- Contestants needed to select several optimal time points for each stock to trade within the trading time. "Buy low, Sell high".

After thorough literature review, We zeroed in on using Deep Re-inforcement learning method called PPO (Proximal Policy Optimizer).

<img width="326" alt="image" src="https://user-images.githubusercontent.com/93938450/235061302-81cc709d-5d89-459b-984c-39715e910e28.png">

As per the innovation, we brought to the competition, we introduced the following:
- Custom Reward function.

- Relevant State variables.

- Short-selling.

- Explainability of results.


PPO algorithm by OPENAI uses a neural network to repesent the policy function (mapping between states and actios):

<img width="327" alt="image" src="https://user-images.githubusercontent.com/93938450/235061516-7a55dd36-e961-48c5-a2fc-c58a61af21fd.png">


