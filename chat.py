import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import azure.cognitiveservices.speech as speechsdk 




# Set up the subscription info for Azure Speech Service
speech_key = "81cc4d4e364e42619db777546d05dc04"
service_region = "eastus"

# Create the speech config
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

# Create a speech recognizer and synthesizer
def create_speech_recognizer():
    audio_config = speechsdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    return speech_recognizer

def create_speech_synthesizer():
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    return speech_synthesizer

# Initialize the recognizer and synthesizer
speech_recognizer = create_speech_recognizer()
speech_synthesizer = create_speech_synthesizer()

# Load the trained model as before
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Iris"

def get_response(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "I'm not sure I understand. Could you rephrase that?"

print("Welcome back, i am Iris a Medical Assistant! (Say 'quit' to exit)")

while True:
    try:
        print("Listening...")
        result = speech_recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"You: {result.text}")
            if result.text.lower() == "quit":
                break

            response = get_response(result.text)
            print(f"{bot_name}: {response}")
            speech_synthesizer.speak_text_async(response)

        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized, please try again.")

        else:
            print(f"Error: {result.reason}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
