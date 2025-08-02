"""
Perplexity API Client
Handles communication with Perplexity's Sonar Pro API for real-time question answering
Optimized for your Pro subscription with proper error handling and rate limiting
"""
import requests
import json
import logging
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class PerplexityResponse:
    """Response from Perplexity API with comprehensive metadata"""
    answer: str
    sources: List[str]
    citations: List[Dict]
    model_used: str
    tokens_used: int
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class PerplexityClient:
    """
    Client for interacting with Perplexity's Sonar Pro API
    Optimized for real-time Q&A with proper error handling
    """
    
    def __init__(self):
        self.config = Config()
        self.api_key = self.config.PERPLEXITY_API_KEY
        self.base_url = self.config.PERPLEXITY_BASE_URL
        self.model = self.config.PERPLEXITY_MODEL
        
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY is required but not provided")
        
        # Session for connection pooling and efficiency
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'RealTimeQA/1.0'
        })
        
        # Rate limiting to respect API limits
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
        # Request timeout settings
        self.timeout = 30  # seconds
        
    def ask_question(self, question: str, context: Optional[str] = None) -> PerplexityResponse:
        """
        Ask a question using Perplexity's Sonar Pro API with comprehensive error handling
        
        Args:
            question: The question to ask
            context: Optional context to provide with the question
            
        Returns:
            PerplexityResponse object with the answer and metadata
        """
        start_time = time.time()
        
        try:
            # Apply rate limiting
            self._apply_rate_limit()
            
            # Prepare the request payload
            messages = self._prepare_messages(question, context)
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,  # Lower temperature for more focused answers
                "max_tokens": 1500,  # Reasonable limit for answers
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
                "stream": False
            }
            
            logger.info(f"Sending request to Perplexity API: {question[:100]}...")
            
            # Make the API request with timeout
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                return self._parse_success_response(response.json(), processing_time)
            elif response.status_code == 429:
                error_msg = "Rate limit exceeded - please wait before making more requests"
                logger.error(error_msg)
                return self._create_error_response(error_msg, processing_time)
            elif response.status_code == 401:
                error_msg = "Invalid API key - please check your Perplexity API key"
                logger.error(error_msg)
                return self._create_error_response(error_msg, processing_time)
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return self._create_error_response(error_msg, processing_time)
                
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout - Perplexity API took longer than {self.timeout}s to respond"
            logger.error(error_msg)
            return self._create_error_response(error_msg, time.time() - start_time)
            
        except requests.exceptions.ConnectionError:
            error_msg = "Connection error - Unable to reach Perplexity API"
            logger.error(error_msg)
            return self._create_error_response(error_msg, time.time() - start_time)
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return self._create_error_response(error_msg, time.time() - start_time)
    
    def _prepare_messages(self, question: str, context: Optional[str] = None) -> List[Dict]:
        """Prepare messages for the API request with optimal formatting"""
        messages = []
        
        # System message for better responses tailored to Q&A
        system_message = (
            "You are a helpful AI assistant that provides accurate, concise, and well-sourced answers. "
            "Focus on giving direct answers to questions while citing relevant sources. "
            "Keep responses informative but not overly long. "
            "For technical questions, provide clear explanations. "
            "Always cite your sources when possible."
        )
        
        messages.append({
            "role": "system",
            "content": system_message
        })
        
        # Add context if provided (useful for follow-up questions)
        if context:
            messages.append({
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}"
            })
        else:
            messages.append({
                "role": "user",
                "content": question
            })
        
        return messages
    
    def _parse_success_response(self, response_data: Dict, processing_time: float) -> PerplexityResponse:
        """Parse a successful API response with comprehensive error handling"""
        try:
            choice = response_data.get('choices', [{}])[0]
            message = choice.get('message', {})
            answer = message.get('content', 'No answer provided')
            
            # Extract usage information for monitoring
            usage = response_data.get('usage', {})
            tokens_used = usage.get('total_tokens', 0)
            
            # Extract citations/sources if available (Sonar Pro feature)
            citations = response_data.get('citations', [])
            sources = []
            
            # Try to extract sources from citations
            for citation in citations:
                if isinstance(citation, dict):
                    url = citation.get('url', '')
                    title = citation.get('title', '')
                    if url:
                        sources.append(url)
                elif isinstance(citation, str):
                    sources.append(citation)
            
            logger.info(f"Received answer: {answer[:100]}... (tokens: {tokens_used}, time: {processing_time:.2f}s)")
            
            return PerplexityResponse(
                answer=answer,
                sources=sources,
                citations=citations,
                model_used=self.model,
                tokens_used=tokens_used,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Error parsing API response: {str(e)}"
            logger.error(error_msg)
            return self._create_error_response(error_msg, processing_time)
    
    def _create_error_response(self, error_message: str, processing_time: float) -> PerplexityResponse:
        """Create a standardized error response"""
        return PerplexityResponse(
            answer="I'm sorry, I encountered an error while trying to answer your question. Please try again.",
            sources=[],
            citations=[],
            model_used=self.model,
            tokens_used=0,
            processing_time=processing_time,
            success=False,
            error_message=error_message
        )
    
    def _apply_rate_limit(self):
        """Apply rate limiting to avoid overwhelming the API"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def test_connection(self) -> bool:
        """Test the connection to Perplexity API with a simple question"""
        try:
            response = self.ask_question("What is 2+2?")
            return response.success
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_available_models(self) -> Optional[List[str]]:
        """Get list of available models (if supported by API)"""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                return [model.get('id') for model in models_data.get('data', [])]
            return None
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return None
