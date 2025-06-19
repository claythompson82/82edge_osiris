# scripts/setup_voice.sh
#!/usr/bin/env bash
set -e
sudo apt-get -qq update
sudo apt-get -y install ffmpeg portaudio19-dev  # for audio

pip install --upgrade --no-cache-dir \
  torch==2.7.1+cpu torchaudio==2.7.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu

pip install --no-cache-dir git+https://github.com/openai/whisper@main

# install Chatterbox without version pin issues
pip install --no-deps --no-cache-dir \
  git+https://github.com/resemble-ai/chatterbox@eb90621

pip install sounddevice websockets pydub
