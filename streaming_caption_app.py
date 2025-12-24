#!/usr/bin/env python3
"""
Real-time Streaming Closed Captioning App with Speaker Diarization
Uses Google Cloud Speech-to-Text streaming API for instant captions.
Speaker labels are added with a slight delay via batch processing.
"""

import os
import sys
import json
import tempfile
import threading
import queue
import time
import wave
import io
import pyaudio
from google.cloud import speech
from flask import Flask, render_template_string
from flask_socketio import SocketIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_google_credentials():
    """Create credentials file from environment variables"""
    project_id = os.getenv('GOOGLE_PROJECT_ID')
    private_key = os.getenv('GOOGLE_PRIVATE_KEY')
    client_email = os.getenv('GOOGLE_CLIENT_EMAIL')
    
    if project_id and private_key and client_email:
        creds = {
            "type": "service_account",
            "project_id": project_id,
            "private_key": private_key.replace('\\n', '\n'),
            "client_email": client_email,
            "token_uri": "https://oauth2.googleapis.com/token"
        }
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(creds, temp_file)
        temp_file.close()
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file.name
        print("Google credentials loaded from environment variables")
        return True
    
    print("Warning: Google Cloud credentials not found in .env")
    return False

# Setup credentials before importing speech client
setup_google_credentials()

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms chunks
GAIN = 3.0  # Audio amplification factor (increase for distant speakers)

import struct

def amplify_audio(audio_data, gain=GAIN):
    """Amplify audio data by a gain factor"""
    # Unpack 16-bit audio samples
    samples = struct.unpack(f'<{len(audio_data)//2}h', audio_data)
    # Amplify and clip to prevent overflow
    amplified = []
    for sample in samples:
        new_sample = int(sample * gain)
        # Clip to 16-bit range
        new_sample = max(-32768, min(32767, new_sample))
        amplified.append(new_sample)
    # Pack back to bytes
    return struct.pack(f'<{len(amplified)}h', *amplified)

class MicrophoneStream:
    """Opens a recording stream as a generator yielding audio chunks."""
    
    def __init__(self, rate=RATE, chunk=CHUNK, gain=GAIN):
        self._rate = rate
        self._chunk = chunk
        self._gain = gain
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        # Amplify audio for better sensitivity to distant speakers
        amplified = amplify_audio(in_data, self._gain)
        self._buff.put(amplified)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


class StreamingCaptionApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'caption-app-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.is_running = False
        self.stream_thread = None
        self.diarization_thread = None
        
        # Audio buffer for diarization
        self.audio_buffer = []
        self.transcript_buffer = []  # Store transcripts with timestamps
        self.caption_id = 0
        
        # Speaker tracking
        self.last_speaker = None
        self.speaker_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        self.speaker_names = ['Speaker 1', 'Speaker 2', 'Speaker 3', 'Speaker 4', 'Speaker 5']
        
        self.setup_routes()
        
    def setup_routes(self):
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
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
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
            height: 600px;
            max-height: 80vh;
            background-color: #111;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            overflow-y: auto;
            font-size: 28px;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        .interim {
            color: #888;
        }
        .final {
            color: #fff;
            margin-bottom: 10px;
        }
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
        }
        .btn-start { background-color: #2ecc71; color: white; }
        .btn-stop { background-color: #e74c3c; color: white; }
        .btn-clear { background-color: #95a5a6; color: white; }
        .status { color: #95a5a6; font-size: 14px; margin-left: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 Live Captions (Streaming)</h1>
        </div>
        
        <div class="caption-area" id="captionArea">
            <div id="finalText"></div>
            <div id="interimText" class="interim"></div>
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
        const finalText = document.getElementById('finalText');
        const interimText = document.getElementById('interimText');
        const captionArea = document.getElementById('captionArea');
        
        let isRunning = false;
        
        socket.on('interim', function(data) {
            interimText.textContent = data.text;
            captionArea.scrollTop = captionArea.scrollHeight;
        });
        
        socket.on('final', function(data) {
            const div = document.createElement('div');
            div.className = 'caption-line';
            div.id = 'caption-' + data.id;
            div.style.marginBottom = '8px';
            div.innerHTML = '<span class="caption-text">' + data.text + '</span>';
            finalText.appendChild(div);
            interimText.textContent = '';
            captionArea.scrollTop = captionArea.scrollHeight;
        });
        
        socket.on('speaker_update', function(data) {
            const div = document.getElementById('caption-' + data.id);
            if (div) {
                // Set color on the text span inside
                const textSpan = div.querySelector('.caption-text');
                if (textSpan) {
                    textSpan.style.color = data.color;
                }
                div.title = data.name;
                div.dataset.speaker = data.speaker;
                
                // Check previous sibling's speaker to determine if we need a break
                const prevDiv = div.previousElementSibling;
                const prevSpeaker = prevDiv ? prevDiv.dataset.speaker : null;
                
                // Add visual break if speaker changed
                if (prevSpeaker && prevSpeaker !== String(data.speaker)) {
                    div.style.marginTop = '25px';
                    div.style.borderTop = '2px solid #444';
                    div.style.paddingTop = '15px';
                }
                
                // Add speaker label at the beginning if not already there
                if (!div.querySelector('.speaker-label')) {
                    const label = document.createElement('span');
                    label.className = 'speaker-label';
                    label.style.color = data.color;
                    label.style.fontSize = '18px';
                    label.style.fontWeight = 'bold';
                    label.style.marginRight = '10px';
                    label.textContent = data.name + ':';
                    div.insertBefore(label, div.firstChild);
                }
                
                captionArea.scrollTop = captionArea.scrollHeight;
            }
        });
        
        socket.on('status', function(data) {
            status.textContent = data.status;
        });
        
        socket.on('error', function(data) {
            status.textContent = 'Error: ' + data.message;
        });
        
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
            finalText.innerHTML = '';
            interimText.textContent = '';
        });
    </script>
</body>
</html>
            ''')
        
        @self.socketio.on('start')
        def handle_start():
            self.start_streaming()
        
        @self.socketio.on('stop')
        def handle_stop():
            self.stop_streaming()
    
    def start_streaming(self):
        if not self.is_running:
            # Wait for any previous threads to finish
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=2)
            if self.diarization_thread and self.diarization_thread.is_alive():
                self.diarization_thread.join(timeout=1)
            
            self.is_running = True
            self.audio_buffer = []
            self.transcript_buffer = []
            self.caption_id = 0
            self.socketio.emit('status', {'status': 'Listening...'})
            self.stream_thread = threading.Thread(target=self.stream_audio, daemon=True)
            self.stream_thread.start()
            # Start diarization thread
            self.diarization_thread = threading.Thread(target=self.run_diarization, daemon=True)
            self.diarization_thread.start()
    
    def stop_streaming(self):
        self.is_running = False
        # Wait briefly for threads to stop
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2)
        self.socketio.emit('status', {'status': 'Stopped'})
    
    def stream_audio(self):
        """Stream audio to Google Cloud Speech-to-Text"""
        try:
            client = speech.SpeechClient()
            
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                language_code="en-US",
                enable_automatic_punctuation=True,
            )
            
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
            )
            
            with MicrophoneStream(RATE, CHUNK) as stream:
                audio_generator = stream.generator()
                
                def audio_with_buffer():
                    for content in audio_generator:
                        self.audio_buffer.append(content)
                        yield content
                
                requests = (
                    speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_with_buffer()
                )
                
                responses = client.streaming_recognize(streaming_config, requests)
                
                for response in responses:
                    if not self.is_running:
                        break
                        
                    if not response.results:
                        continue
                    
                    result = response.results[0]
                    if not result.alternatives:
                        continue
                    
                    transcript = result.alternatives[0].transcript
                    
                    if result.is_final:
                        caption_id = self.caption_id
                        self.caption_id += 1
                        self.transcript_buffer.append({
                            'id': caption_id,
                            'text': transcript,
                            'time': time.time()
                        })
                        self.socketio.emit('final', {'id': caption_id, 'text': transcript})
                        print(f"Final [{caption_id}]: {transcript}")
                    else:
                        self.socketio.emit('interim', {'text': transcript})
                        
        except Exception as e:
            print(f"Streaming error: {e}")
            self.socketio.emit('error', {'message': str(e)})
            self.socketio.emit('status', {'status': f'Error: {e}'})
        
        self.is_running = False
        self.socketio.emit('status', {'status': 'Stopped'})
    
    def run_diarization(self):
        """Run speaker diarization on buffered audio periodically"""
        print("Diarization thread started")
        processed_captions = set()
        
        while self.is_running:
            time.sleep(10)  # Process every 10 seconds
            
            # Need unprocessed captions
            unprocessed = [t for t in self.transcript_buffer if t['id'] not in processed_captions]
            if not unprocessed:
                continue
                
            # Get all audio data collected so far
            audio_data = b''.join(self.audio_buffer)
            
            # Need at least 10 seconds of audio for good diarization
            if len(audio_data) < RATE * 2 * 10:
                print(f"Waiting for more audio for diarization...")
                continue
            
            try:
                print(f"Running diarization on {len(audio_data)} bytes of audio...")
                client = speech.SpeechClient()
                
                audio = speech.RecognitionAudio(content=audio_data)
                
                diarization_config = speech.SpeakerDiarizationConfig(
                    enable_speaker_diarization=True,
                    min_speaker_count=2,
                    max_speaker_count=2,
                )
                
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=RATE,
                    language_code="en-US",
                    enable_automatic_punctuation=True,
                    diarization_config=diarization_config,
                )
                
                response = client.recognize(config=config, audio=audio)
                
                if response.results:
                    result = response.results[-1]
                    if result.alternatives:
                        words = result.alternatives[0].words
                        print(f"Got {len(words)} words with speaker info")
                        
                        # Build word sequence with speakers
                        word_list = []
                        for word_info in words:
                            speaker = getattr(word_info, 'speaker_tag', 1)
                            word_list.append({
                                'word': word_info.word.lower(),
                                'speaker': speaker
                            })
                        
                        # Match transcripts to word sequences
                        word_idx = 0
                        for item in self.transcript_buffer:
                            if item['id'] in processed_captions:
                                continue
                                
                            item_words = item['text'].lower().split()
                            speaker_votes = {}
                            
                            # Find words from this transcript in the word list
                            for iw in item_words:
                                clean_iw = ''.join(c for c in iw if c.isalnum())
                                if not clean_iw:
                                    continue
                                # Search nearby in word list
                                for wi in range(max(0, word_idx-5), min(len(word_list), word_idx+20)):
                                    if word_list[wi]['word'].startswith(clean_iw[:3]):
                                        s = word_list[wi]['speaker']
                                        speaker_votes[s] = speaker_votes.get(s, 0) + 1
                                        word_idx = wi + 1
                                        break
                            
                            if speaker_votes:
                                # Use most common speaker
                                speaker = max(speaker_votes, key=speaker_votes.get)
                                color = self.speaker_colors[(speaker - 1) % len(self.speaker_colors)]
                                name = self.speaker_names[(speaker - 1) % len(self.speaker_names)]
                                
                                self.socketio.emit('speaker_update', {
                                    'id': item['id'],
                                    'speaker': speaker,
                                    'color': color,
                                    'name': name
                                })
                                print(f"{name} -> [{item['id']}]: {item['text'][:40]}...")
                                processed_captions.add(item['id'])
                else:
                    print("No diarization results returned")
                                        
            except Exception as e:
                print(f"Diarization error: {e}")
                import traceback
                traceback.print_exc()
    
    def run(self, host='127.0.0.1', port=5050, debug=False):
        print(f"Starting streaming caption app at http://{host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    app = StreamingCaptionApp()
    app.run(debug=True)
