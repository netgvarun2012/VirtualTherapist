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
- After deciphering the emotion, Generative AI component of the system takes the predicted emotion labels as prompt and generated helpful tips accordingly.
- There is an added feature of requesting "Balanced" vs "Creative" one liner recommendations.

### Methodology:
- In terms of GEN-AI component, following was achieved:
  - A dataset was created manually by requesting CHATGPT to generate one-liner tips based on 7 different emotion categories.
  - This dataset was augmented using pegasus_paraphrase model <https://huggingface.co/tuner007/pegasus_paraphrase>
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


