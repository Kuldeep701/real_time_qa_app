#!/bin/bash
set -e

echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip git pulseaudio ffmpeg portaudio19-dev libasound2-dev

echo "Upgrading pip and installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "Downloading NLTK data..."
python3 -c "
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
"

echo "Setup complete!"
