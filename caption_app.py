#!/usr/bin/env python3
"""
Real-time Closed Captioning App
Uses microphone input to transcribe speech and display captions in real-time.
Integrates Claude API for enhanced transcription accuracy.
"""

import speech_recognition as sr
import threading
import queue
import tkinter as tk
from tkinter import scrolledtext, font
import time
import os
from anthropic import Anthropic
import json

class CaptionApp:
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Live Closed Captioning")
        self.root.geometry("800x400")
        self.root.configure(bg='black')
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Thread-safe queue for communication
        self.text_queue = queue.Queue()
        
        # Claude client (will be initialized if API key is provided)
        self.claude_client = None
        self.use_claude = False
        
        # Setup GUI
        self.setup_gui()
        
        # Audio processing thread
        self.is_running = False
        self.audio_thread = None
        
        # Adjust for ambient noise
        self.calibrate_microphone()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = tk.Frame(self.root, bg='black')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title label
        title_font = font.Font(family="Arial", size=16, weight="bold")
        title_label = tk.Label(main_frame, text="Live Captions", 
                               font=title_font, fg='white', bg='black')
        title_label.pack(pady=(0, 10))
        
        # Caption display area
        self.caption_text = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            font=font.Font(family="Arial", size=24),
            bg='black',
            fg='white',
            height=10,
            padx=10,
            pady=10
        )
        self.caption_text.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg='black')
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Start/Stop button
        self.start_button = tk.Button(
            control_frame,
            text="Start Captioning",
            command=self.toggle_captioning,
            font=font.Font(family="Arial", size=12),
            bg='#2ecc71',
            fg='white',
            padx=20,
            pady=10
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        clear_button = tk.Button(
            control_frame,
            text="Clear",
            command=self.clear_captions,
            font=font.Font(family="Arial", size=12),
            bg='#e74c3c',
            fg='white',
            padx=20,
            pady=10
        )
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(
            control_frame,
            text="Ready",
            font=font.Font(family="Arial", size=10),
            fg='#95a5a6',
            bg='black'
        )
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Schedule GUI updates
        self.update_gui()
        
    def calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        self.status_label.config(text="Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        self.status_label.config(text="Ready")
        
    def toggle_captioning(self):
        """Start or stop captioning"""
        if not self.is_running:
            self.start_captioning()
        else:
            self.stop_captioning()
            
    def start_captioning(self):
        """Start the captioning process"""
        self.is_running = True
        self.start_button.config(text="Stop Captioning", bg='#e74c3c')
        self.status_label.config(text="Listening...")
        
        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.audio_thread.start()
        
    def stop_captioning(self):
        """Stop the captioning process"""
        self.is_running = False
        self.start_button.config(text="Start Captioning", bg='#2ecc71')
        self.status_label.config(text="Stopped")
        
    def clear_captions(self):
        """Clear all captions"""
        self.caption_text.delete(1.0, tk.END)
        
    def process_audio(self):
        """Process audio in separate thread"""
        while self.is_running:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                # Recognize speech using Google's free API
                try:
                    text = self.recognizer.recognize_google(audio)
                    
                    # Enhance with Claude if available
                    if self.use_claude and self.claude_client:
                        enhanced_text = self.enhance_with_claude(text)
                        if enhanced_text:
                            text = enhanced_text
                    
                    # Put text in queue for GUI update
                    self.text_queue.put(text)
                    
                except sr.UnknownValueError:
                    # Speech was unintelligible
                    pass
                except sr.RequestError as e:
                    # API error
                    self.text_queue.put(f"API Error: {e}")
                    
            except sr.WaitTimeoutError:
                # Timeout, continue listening
                pass
            except Exception as e:
                # Other errors
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
            
    def update_gui(self):
        """Update GUI from queue"""
        try:
            while True:
                text = self.text_queue.get_nowait()
                # Add timestamp and text to caption display
                timestamp = time.strftime("%H:%M:%S")
                self.caption_text.insert(tk.END, f"[{timestamp}] {text}\n\n")
                self.caption_text.see(tk.END)
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(100, self.update_gui)
        
    def setup_claude(self, api_key):
        """Setup Claude API client"""
        try:
            self.claude_client = Anthropic(api_key=api_key)
            self.use_claude = True
            print("Claude API initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Claude API: {e}")
            self.use_claude = False
            
    def run(self):
        """Start the application"""
        # Try to load Claude API key from environment variable
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            self.setup_claude(api_key)
        else:
            print("ANTHROPIC_API_KEY environment variable not found. Running without Claude enhancement.")
            
        self.root.mainloop()

if __name__ == "__main__":
    app = CaptionApp()
    app.run()
