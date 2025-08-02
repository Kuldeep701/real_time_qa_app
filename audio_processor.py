"""
Audio Processing Module
Handles system audio capture, voice activity detection, and speech-to-text conversion
Optimized for Windows 11 WSL2 and Ubuntu environments
"""
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import webrtcvad
import whisper
from typing import Optional, Callable
import logging
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles real-time audio capture, voice activity detection, and speech-to-text conversion
    """
    
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self.config = Config()
        
        # Audio settings optimized for speech recognition
        self.sample_rate = self.config.AUDIO_SAMPLE_RATE
        self.channels = self.config.AUDIO_CHANNELS
        self.chunk_size = self.config.AUDIO_CHUNK_SIZE
        
        # Initialize VAD (Voice Activity Detection) for better question segmentation
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # Aggressiveness mode: 0-3 (2 is balanced)
        
        # Initialize Whisper model for speech-to-text
        logger.info(f"Loading Whisper model: {self.config.WHISPER_MODEL}")
        self.whisper_model = whisper.load_model(self.config.WHISPER_MODEL)
        
        # Threading controls for concurrent processing
        self.is_recording = False
        self.audio_thread = None
        self.processing_thread = None
        self.audio_queue = queue.Queue()
        
        # Audio buffer for speech detection and processing
        self.audio_buffer = []
        self.speech_frames = []
        self.silence_start = None
        self.is_speech_active = False
        
        # Audio stream handle
        self.stream = None
        
    def start_recording(self):
        """Start audio recording and processing with error handling"""
        if self.is_recording:
            logger.warning("Recording is already active")
            return
            
        self.is_recording = True
        logger.info("Starting audio recording...")
        
        try:
            # Get and log default input device info
            device_info = sd.query_devices(kind='input')
            logger.info(f"Using audio device: {device_info['name']}")
            
            # Start audio capture thread
            self.audio_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
            self.audio_thread.start()
            
            # Start audio processing thread
            self.processing_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
            self.processing_thread.start()
            
            logger.info("Audio recording started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start audio recording: {e}")
            self.is_recording = False
            raise
    
    def stop_recording(self):
        """Stop audio recording and processing safely"""
        if not self.is_recording:
            logger.warning("Recording is not active")
            return
            
        logger.info("Stopping audio recording...")
        self.is_recording = False
        
        # Close audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        # Wait for threads to finish with timeout
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
            
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            
        logger.info("Audio recording stopped")
    
    def _audio_capture_loop(self):
        """Main audio capture loop running in separate thread"""
        def audio_callback(indata, frames, time, status):
            """Callback for audio stream - runs in real-time"""
            if status:
                logger.warning(f"Audio stream status: {status}")
            
            # Convert to the format expected by webrtcvad (16-bit PCM)
            audio_data = (indata[:, 0] * 32767).astype(np.int16)
            
            # Add to processing queue
            self.audio_queue.put(audio_data.tobytes())
        
        try:
            # Create audio stream with optimized settings
            self.stream = sd.InputStream(
                callback=audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.float32
            )
            
            self.stream.start()
            logger.info("Audio stream started")
            
            # Keep the stream alive
            while self.is_recording:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in audio capture loop: {e}")
            self.is_recording = False
    
    def _process_audio_loop(self):
        """Process audio data for voice activity detection and speech recognition"""
        frame_duration_ms = 30  # 30ms frames for VAD (webrtcvad requirement)
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        
        while self.is_recording:
            try:
                # Get audio data from queue with timeout
                audio_data = self.audio_queue.get(timeout=1.0)
                
                # Convert bytes back to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Process in 30ms frames for VAD
                for i in range(0, len(audio_array), frame_size):
                    frame = audio_array[i:i + frame_size]
                    
                    if len(frame) < frame_size:
                        continue
                        
                    # Voice Activity Detection
                    is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                    
                    if is_speech:
                        if not self.is_speech_active:
                            logger.info("Speech detected - starting recording")
                            self.is_speech_active = True
                            self.speech_frames = []
                            
                        self.speech_frames.extend(frame)
                        self.silence_start = None
                        
                    else:
                        if self.is_speech_active:
                            if self.silence_start is None:
                                self.silence_start = time.time()
                            elif time.time() - self.silence_start > self.config.MIN_SILENCE_DURATION:
                                # End of speech detected - process the complete question
                                logger.info("End of speech detected - processing...")
                                self._process_speech_segment()
                                self.is_speech_active = False
                                self.silence_start = None
                                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
    
    def _process_speech_segment(self):
        """Process a complete speech segment with Whisper STT"""
        if not self.speech_frames:
            return
            
        try:
            # Convert to numpy array and normalize for Whisper
            audio_np = np.array(self.speech_frames, dtype=np.float32) / 32768.0
            
            # Ensure minimum length for reliable transcription
            if len(audio_np) < self.sample_rate * 0.5:  # Less than 0.5 seconds
                logger.debug("Audio segment too short, skipping")
                return
                
            # Pad or trim to 30 seconds (Whisper requirement)
            if len(audio_np) > self.sample_rate * 30:
                audio_np = audio_np[:self.sample_rate * 30]
            else:
                # Pad with zeros if needed
                pad_length = self.sample_rate * 30 - len(audio_np)
                audio_np = np.pad(audio_np, (0, pad_length), mode='constant')
            
            logger.info("Transcribing audio with Whisper...")
            
            # Transcribe with Whisper - optimized settings for questions
            result = self.whisper_model.transcribe(
                audio_np,
                language='en',  # Can be made configurable
                temperature=0.0,  # More deterministic results
                no_speech_threshold=0.6,
                logprob_threshold=-1.0
            )
            
            text = result.get('text', '').strip()
            confidence = 1.0 - abs(result.get('avg_logprob', 0.0))  # Rough confidence estimate
            
            if text and len(text.split()) >= 3:  # At least 3 words for valid question
                logger.info(f"Transcribed: '{text}' (confidence: {confidence:.2f})")
                
                # Send to callback for question detection
                self.callback(text)
            else:
                logger.debug("Transcription result too short or empty")
                
        except Exception as e:
            logger.error(f"Error processing speech segment: {e}")
    
    def is_active(self) -> bool:
        """Check if audio processing is currently active"""
        return self.is_recording
    
    def get_status(self) -> dict:
        """Get current status information for monitoring"""
        return {
            'is_recording': self.is_recording,
            'is_speech_active': self.is_speech_active,
            'queue_size': self.audio_queue.qsize(),
            'sample_rate': self.sample_rate,
            'channels': self.channels
        }
