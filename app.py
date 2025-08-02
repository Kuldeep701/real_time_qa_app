"""
Main Flask Application
Real-time Question Answering System using system audio capture and Perplexity API
"""
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import threading
import logging
import time
from datetime import datetime
from typing import Dict, List
import json

# Import our custom modules
from audio_processor import AudioProcessor
from question_detector import QuestionDetector, QuestionInfo
from perplexity_client import PerplexityClient, PerplexityResponse
from config import Config

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app with security considerations
app = Flask(__name__)
app.config.from_object(Config)

# Initialize SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Global components
audio_processor = None
question_detector = None
perplexity_client = None

# Store recent questions and answers with size limit
recent_qa_pairs = []
MAX_HISTORY = 50

class QASystem:
    """Main Question-Answer system coordinator with error handling"""
    
    def __init__(self):
        self.is_active = False
        self.stats = {
            'questions_processed': 0,
            'answers_generated': 0,
            'start_time': None,
            'errors': 0
        }
        self.error_recovery_attempts = 0
        self.max_recovery_attempts = 3
    
    def start(self):
        """Start the Q&A system with comprehensive error handling"""
        global audio_processor, question_detector, perplexity_client
        
        if self.is_active:
            logger.warning("Q&A system is already running")
            return False
        
        try:
            logger.info("Starting Q&A system...")
            
            # Initialize NLP component first (fastest to fail)
            question_detector = QuestionDetector()
            
            # Initialize and test Perplexity client
            perplexity_client = PerplexityClient()
            if not perplexity_client.test_connection():
                raise Exception("Failed to connect to Perplexity API - check your API key")
            
            # Initialize audio processor with callback (most complex component)
            audio_processor = AudioProcessor(callback=self.on_speech_detected)
            audio_processor.start_recording()
            
            self.is_active = True
            self.stats['start_time'] = datetime.now()
            self.error_recovery_attempts = 0
            
            logger.info("Q&A system started successfully")
            
            # Emit success status to all connected clients
            socketio.emit('system_status', {
                'status': 'active',
                'message': 'Q&A system is now listening for questions...'
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Q&A system: {e}")
            self.stats['errors'] += 1
            self.error_recovery_attempts += 1
            
            # Emit error status
            socketio.emit('system_status', {
                'status': 'error',
                'message': f'Failed to start system: {str(e)}'
            })
            
            return False
    
    def stop(self):
        """Stop the Q&A system safely"""
        global audio_processor
        
        if not self.is_active:
            logger.warning("Q&A system is not running")
            return False
        
        try:
            logger.info("Stopping Q&A system...")
            
            if audio_processor:
                audio_processor.stop_recording()
            
            self.is_active = False
            
            logger.info("Q&A system stopped successfully")
            
            socketio.emit('system_status', {
                'status': 'inactive',
                'message': 'Q&A system has been stopped.'
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Q&A system: {e}")
            return False
    
    def on_speech_detected(self, text: str):
        """Callback function when speech is detected and transcribed"""
        try:
            logger.info(f"Processing speech: '{text}'")
            
            # Emit transcription to frontend immediately
            socketio.emit('speech_detected', {
                'text': text,
                'timestamp': datetime.now().isoformat()
            })
            
            # Detect if it's a question using NLP
            question_info = question_detector.detect_question(text)
            
            if question_info and question_info.is_valid:
                logger.info(f"Question detected: {question_info.text}")
                self.stats['questions_processed'] += 1
                
                # Emit question detection
                socketio.emit('question_detected', {
                    'question': question_info.text,
                    'confidence': question_info.confidence,
                    'type': question_info.question_type,
                    'keywords': question_info.keywords,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Process the question asynchronously to maintain responsiveness
                threading.Thread(
                    target=self.process_question_async,
                    args=(question_info,),
                    daemon=True
                ).start()
            else:
                confidence = question_info.confidence if question_info else 0
                logger.info(f"Not a valid question (confidence: {confidence:.2f})")
                
        except Exception as e:
            logger.error(f"Error processing speech: {e}")
            self.stats['errors'] += 1
    
    def process_question_async(self, question_info: QuestionInfo):
        """Process a question asynchronously with error handling"""
        try:
            logger.info(f"Getting answer for: {question_info.text}")
            
            # Emit processing status
            socketio.emit('processing_question', {
                'question': question_info.text,
                'status': 'searching'
            })
            
            # Get answer from Perplexity with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = perplexity_client.ask_question(question_info.text)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"API attempt {attempt + 1} failed, retrying...")
                    time.sleep(1)
            
            if response.success:
                self.stats['answers_generated'] += 1
                
                # Store in history
                qa_pair = {
                    'question': question_info.text,
                    'answer': response.answer,
                    'sources': response.sources,
                    'confidence': question_info.confidence,
                    'processing_time': response.processing_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Manage history size
                recent_qa_pairs.append(qa_pair)
                if len(recent_qa_pairs) > MAX_HISTORY:
                    recent_qa_pairs.pop(0)
                
                # Emit answer to frontend
                socketio.emit('answer_received', qa_pair)
                
                logger.info(f"Answer generated successfully (time: {response.processing_time:.2f}s)")
                
            else:
                logger.error(f"Failed to get answer: {response.error_message}")
                self.stats['errors'] += 1
                
                socketio.emit('answer_error', {
                    'question': question_info.text,
                    'error': response.error_message,
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            self.stats['errors'] += 1
            
            socketio.emit('answer_error', {
                'question': question_info.text,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

# Initialize the Q&A system
qa_system = QASystem()

# Flask routes for REST API
@app.route('/')
def index():
    """Main page with responsive interface"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get comprehensive system status"""
    audio_status = audio_processor.get_status() if audio_processor else {}
    
    return jsonify({
        'system_active': qa_system.is_active,
        'audio_status': audio_status,
        'stats': qa_system.stats,
        'recent_questions': len(recent_qa_pairs)
    })

@app.route('/api/history')
def get_history():
    """Get recent Q&A history"""
    return jsonify({
        'qa_pairs': recent_qa_pairs[-20:],  # Return last 20 pairs
        'total_count': len(recent_qa_pairs)
    })

@app.route('/api/start', methods=['POST'])
def start_system():
    """Start the Q&A system"""
    success = qa_system.start()
    return jsonify({'success': success})

@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Stop the Q&A system"""
    success = qa_system.stop()
    return jsonify({'success': success})

# SocketIO event handlers for real-time communication
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected")
    emit('connected', {'status': 'Connected to Q&A system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")

@socketio.on('request_status')
def handle_status_request():
    """Handle status request from client"""
    audio_status = audio_processor.get_status() if audio_processor else {}
    
    emit('status_update', {
        'system_active': qa_system.is_active,
        'audio_status': audio_status,
        'stats': qa_system.stats
    })

# Application entry point with error handling
if __name__ == '__main__':
    try:
        logger.info("Starting Real-time Q&A Application")
        logger.info(f"Server will run on {Config.HOST}:{Config.PORT}")
        
        # Run with SocketIO for real-time features
        socketio.run(
            app,
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        qa_system.stop()
    except Exception as e:
        logger.error(f"Application error: {e}")
        qa_system.stop()
