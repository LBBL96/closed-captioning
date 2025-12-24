#!/usr/bin/env python3
"""
Web-based Real-time Closed Captioning App
Uses microphone input to transcribe speech and display captions in real-time.
Integrates Claude API for enhanced transcription accuracy.
"""

import speech_recognition as sr
import threading
import queue
import time
import os
import json
import tempfile
from anthropic import Anthropic
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def setup_google_credentials():
    """Create credentials file from environment variables"""
    # Check if credentials are in env vars
    project_id = os.getenv('GOOGLE_PROJECT_ID')
    private_key = os.getenv('GOOGLE_PRIVATE_KEY')
    client_email = os.getenv('GOOGLE_CLIENT_EMAIL')
    
    if project_id and private_key and client_email:
        # Create temporary credentials file
        creds = {
            "type": "service_account",
            "project_id": project_id,
            "private_key": private_key.replace('\\n', '\n'),
            "client_email": client_email,
            "token_uri": "https://oauth2.googleapis.com/token"
        }
        
        # Write to temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(creds, temp_file)
        temp_file.close()
        
        # Set environment variable for Google SDK
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file.name
        print(f"Google credentials loaded from environment variables")
        return True
    return False

# Try to set up Google credentials
setup_google_credentials()

class WebCaptionApp:
    def __init__(self):
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'caption-app-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Thread-safe queue for communication
        self.text_queue = queue.Queue()
        
        # Claude client (will be initialized if API key is provided)
        self.claude_client = None
        self.use_claude = False
        
        # Audio processing thread
        self.is_running = False
        self.audio_thread = None
        
        # Setup routes and events
        self.setup_routes()
        
        # Adjust for ambient noise
        self.calibrate_microphone()
        
    def setup_routes(self):
        """Setup Flask routes and SocketIO events"""
        
        @self.app.route('/')
        def index():
            return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Closed Captioning</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #000;
            color: #fff;
            margin: 0;
            padding: 20px;
            height: 100vh;
            overflow: hidden;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .header h1 {
            font-size: 24px;
            font-weight: bold;
            margin: 0;
        }
        .caption-area {
            height: 300px;
            min-height: 200px;
            max-height: 50vh;
            background-color: #111;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            overflow-y: auto;
            font-size: 24px;
            line-height: 1.5;
            margin-bottom: 20px;
        }
        .caption-item {
            margin-bottom: 15px;
            padding: 10px;
            background-color: #222;
            border-radius: 4px;
        }
        .timestamp {
            color: #888;
            font-size: 16px;
            margin-bottom: 5px;
        }
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            align-items: center;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .btn-start {
            background-color: #2ecc71;
            color: white;
        }
        .btn-stop {
            background-color: #e74c3c;
            color: white;
        }
        .btn-clear {
            background-color: #95a5a6;
            color: white;
        }
        .status {
            color: #95a5a6;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Live Captions</h1>
        </div>
        
        <div class="caption-area" id="captionArea">
            <!-- Captions will appear here -->
        </div>
        
        <div class="controls">
            <button id="toggleBtn" class="btn btn-start">Start Captioning</button>
            <button id="clearBtn" class="btn btn-clear">Clear</button>
            <span id="status" class="status">Ready</span>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script>
        const socket = io();
        const toggleBtn = document.getElementById('toggleBtn');
        const clearBtn = document.getElementById('clearBtn');
        const status = document.getElementById('status');
        const captionArea = document.getElementById('captionArea');
        
        let isRunning = false;
        
        // Socket events
        socket.on('caption', function(data) {
            const captionItem = document.createElement('div');
            captionItem.className = 'caption-item';
            captionItem.innerHTML = `
                <div class="timestamp">[${data.timestamp}]</div>
                <div>${data.text}</div>
            `;
            captionArea.appendChild(captionItem);
            captionArea.scrollTop = captionArea.scrollHeight;
        });
        
        socket.on('status', function(data) {
            status.textContent = data.status;
        });
        
        // Button events
        toggleBtn.addEventListener('click', function() {
            if (!isRunning) {
                socket.emit('start');
                toggleBtn.textContent = 'Stop Captioning';
                toggleBtn.className = 'btn btn-stop';
                isRunning = true;
            } else {
                socket.emit('stop');
                toggleBtn.textContent = 'Start Captioning';
                toggleBtn.className = 'btn btn-start';
                isRunning = false;
            }
        });
        
        clearBtn.addEventListener('click', function() {
            captionArea.innerHTML = '';
            socket.emit('clear');
        });
    </script>
</body>
</html>
            ''')
        
        @self.socketio.on('start')
        def handle_start():
            self.start_captioning()
        
        @self.socketio.on('stop')
        def handle_stop():
            self.stop_captioning()
        
        @self.socketio.on('clear')
        def handle_clear():
            self.clear_captions()
        
        # Background thread to emit captions
        def emit_captions():
            while True:
                try:
                    text = self.text_queue.get_nowait()
                    timestamp = time.strftime("%H:%M:%S")
                    self.socketio.emit('caption', {'text': text, 'timestamp': timestamp})
                except queue.Empty:
                    pass
                self.socketio.sleep(0.1)
        
        self.socketio.start_background_task(emit_captions)
        
    def calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        print("Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        # Wait longer before considering silence as end of phrase
        self.recognizer.pause_threshold = 1.5  # seconds of silence before phrase ends
        self.recognizer.non_speaking_duration = 0.5  # minimum silence length
        print("Microphone calibrated")
        
    def start_captioning(self):
        """Start the captioning process"""
        if not self.is_running:
            self.is_running = True
            self.socketio.emit('status', {'status': 'Listening...'})
            
            # Start audio processing thread
            self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
            self.audio_thread.start()
            
    def stop_captioning(self):
        """Stop the captioning process"""
        self.is_running = False
        self.socketio.emit('status', {'status': 'Stopped'})
        
    def clear_captions(self):
        """Clear all captions"""
        # This is handled on the client side
        pass
        
    def process_audio(self):
        """Process audio in separate thread"""
        print("Audio processing started...")
        while self.is_running:
            try:
                with self.microphone as source:
                    # Listen for audio - 6 seconds captures most sentences
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=6)
                    print(f"Audio captured, length: {len(audio.frame_data)} bytes")
                    
                # Recognize speech using Google's free API
                try:
                    print("Sending to Google Speech API...")
                    text = self.recognizer.recognize_google(audio)
                    print(f"Recognized: {text}")
                    
                    # Put text in queue immediately for real-time display
                    self.text_queue.put(text)
                    
                    # Note: Claude enhancement disabled for speed
                    # To enable, uncomment below (adds ~1s delay per caption)
                    # if self.use_claude and self.claude_client:
                    #     enhanced_text = self.enhance_with_claude(text)
                    #     if enhanced_text:
                    #         self.text_queue.put(f"[Enhanced] {enhanced_text}")
                    
                except sr.UnknownValueError:
                    # Speech was unintelligible
                    print("Could not understand audio")
                except sr.RequestError as e:
                    # API error
                    print(f"Google API error: {e}")
                    self.text_queue.put(f"API Error: {e}")
                    
            except sr.WaitTimeoutError:
                # Timeout, continue listening
                print("Listening timeout, retrying...")
            except Exception as e:
                # Other errors
                print(f"Error in audio processing: {e}")
                self.text_queue.put(f"Error: {e}")
                
    def enhance_with_claude(self, text):
        """Enhance transcription using Claude API"""
        try:
            message = self.claude_client.messages.create(
                model="claude-haiku-4-5-20250214",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": f"Please correct any transcription errors in this text and return only the corrected version: {text}"
                }]
            )
            return message.content[0].text.strip()
        except Exception as e:
            print(f"Claude API error: {e}")
            return text
            
    def setup_claude(self, api_key):
        """Setup Claude API client"""
        try:
            self.claude_client = Anthropic(api_key=api_key)
            self.use_claude = True
            print("Claude API initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Claude API: {e}")
            self.use_claude = False
            
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Start the web application"""
        # Try to load Claude API key from environment variable
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            self.setup_claude(api_key)
        else:
            print("ANTHROPIC_API_KEY environment variable not found. Running without Claude enhancement.")
            
        print(f"Starting web captioning app at http://{host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)

if __name__ == "__main__":
    app = WebCaptionApp()
    app.run(port=5050, debug=True)
