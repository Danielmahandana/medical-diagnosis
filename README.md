# Dignostics-labs
Speech-based Medical Assistant - Iris

This Python code serves as a speech-based medical assistant named Iris. Iris interacts with users through speech recognition and synthesis, providing medical assistance based on predefined intents and responses.

Overview

The code leverages Azure Cognitive Services for speech recognition and synthesis functionalities. It utilizes a pre-trained neural network model to understand user queries related to medical assistance.

Requirements

To run this code, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- Azure Cognitive Services SDK for Speech

Getting Started

1. Clone the repository or download the code.
2. Install the required Python packages using `pip`:

    ```bash
    pip install torch azure-cognitiveservices-speech
    ```

3. Obtain Azure Speech Service subscription key and region and replace the `speech_key` and `service_region` variables in the code with your credentials.

4. Place the necessary model files (`intents.json` and `data.pth`) in the same directory as the script.

5. Run the script:

    ```bash
    python chat.py
    ```
Code Structure

- `model.py`: Defines the neural network model (`NeuralNet`) used for intent classification.
- `nltk_utils.py`: Provides utility functions for tokenization and bag-of-words representation.
- `chat.py`: Main script implementing the speech-based medical assistant using Azure Cognitive Services.

Usage

Once the script is running, Iris will greet the user and listen for queries. Users can interact with Iris by speaking queries related to medical assistance. Iris will respond based on recognized intents and provide appropriate responses.

Commands

- To exit the program, say "quit".
Note

Ensure a reliable internet connection for Azure Cognitive Services usage.

