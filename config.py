"""
Configuration file for Real-time Q&A Application
Centralized configuration management with environment variable support
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Perplexity API settings
    PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY', '')
    PERPLEXITY_BASE_URL = 'https://api.perplexity.ai'
    PERPLEXITY_MODEL = os.getenv('PERPLEXITY_MODEL', 'sonar-pro')  # Using sonar-pro for your Pro subscription
    
    # Audio settings optimized for speech recognition
    AUDIO_SAMPLE_RATE = 16000  # 16kHz for speech recognition
    AUDIO_CHANNELS = 1  # Mono audio
    AUDIO_CHUNK_SIZE = 1024
    AUDIO_FORMAT = 'int16'
    
    # Speech-to-text settings
    WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')  # Can be: tiny, base, small, medium, large
    MIN_SILENCE_DURATION = 1.5  # seconds
    SPEECH_THRESHOLD = 0.3  # Voice activity detection threshold
    
    # Question detection settings
    QUESTION_CONFIDENCE_THRESHOLD = 0.7
    MIN_QUESTION_LENGTH = 5  # minimum words in question
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # Docker/WSL audio settings
    PULSE_SERVER = os.getenv('PULSE_SERVER', '/mnt/wslg/PulseServer')
