# Live Closed Captioning App

A real-time closed captioning application that captures speech from your Mac's microphone and displays live captions. The app uses Google's speech recognition API and can optionally enhance transcriptions with Anthropic's Claude API for improved accuracy.

## Features

- **Real-time captioning**: Live transcription from microphone input
- **Clean interface**: Easy-to-read captions with timestamps
- **Claude enhancement**: Optional AI-powered transcription correction
- **Simple controls**: Start/stop captioning and clear display
- **Mac optimized**: Designed specifically for macOS microphone access

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up Claude API for enhanced transcription:
   ```bash
   export ANTHROPIC_API_KEY="your_api_key_here"
   ```

## Usage

1. Run the application:
   ```bash
   python caption_app.py
   ```

2. Click "Start Captioning" to begin listening
3. Speak clearly into your Mac's microphone
4. Watch captions appear in real-time
5. Click "Stop Captioning" when finished

## Requirements

- Python 3.7+
- macOS with built-in microphone
- Internet connection (for speech recognition APIs)

## Dependencies

- `SpeechRecognition`: Audio capture and speech recognition
- `PyAudio`: Microphone access
- `anthropic`: Claude API integration (optional)
- `tkinter`: GUI framework (included with Python)

## API Keys

The app works without Claude API using Google's free speech recognition. For enhanced accuracy:

1. Get an API key from [Anthropic Console](https://console.anthropic.com)
2. Set the `ANTHROPIC_API_KEY` environment variable
3. Restart the application

## Troubleshooting

- **Microphone not working**: Ensure microphone permissions are granted in System Preferences
- **Poor transcription quality**: Speak clearly and reduce background noise
- **Claude not working**: Verify API key is correctly set in environment variables

## Privacy

- Audio is processed in real-time and not stored
- Claude API calls are only made for transcription enhancement
- No audio data is saved to disk
