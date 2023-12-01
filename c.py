import random
import json
import torch
import azure.cognitiveservices.speech as speechsdk

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Set up the Azure Speech SDK
speech_key, service_region = "81cc4d4e364e42619db777546d05dc04", "eastus"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

# Create a recognizer and synthesizer with the given settings
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

# Load the intents file and model
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Iris"

# Function to recognize speech
def recognize_speech():
    print("Listening...")
    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
    return ""

# Function to synthesize speech
def synthesize_speech(text):
    result = speech_synthesizer.speak_text_async(text).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Text-to-Speech failed with status: {}".format(result.reason))

# Greet the user first
synthesize_speech(f"Welcome back, I am {bot_name}, your voice-activated assistant.")

# Main chatbot interaction loop
while True:
    sentence = recognize_speech()
    if not sentence:
        continue

    print(f"You: {sentence}")

    # Check for the quit command
    if sentence.lower() == "shutdown":
        synthesize_speech("Goodbye boss.")
        break

    # If the user wants to initiate a Google search
    if sentence.lower().startswith("search for"):
        search_query = sentence.lower().replace("search for", "", 1).strip()
        synthesize_speech(f"Initiating a Google search for {search_query}")
        # Code to perform the Google search would go here
        # For example, you might use a library or API that performs the search and fetches results
        print(f"Search results for {search_query}: [Search results not implemented]")
        continue

    # Processing the speech text
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                print(f"{bot_name}: {response}")
                synthesize_speech(response)
    else:
        synthesize_speech("I don't understand...")
