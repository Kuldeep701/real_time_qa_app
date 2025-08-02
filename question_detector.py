"""
Question Detection Module
Analyzes text to determine if it contains a question and extracts relevant information
Uses NLP techniques for accurate question identification
"""
import re
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
import nltk
from nltk.tokenize import word_tokenize
from nltk.pos_tag import pos_tag
from nltk.chunk import ne_chunk
from config import Config

# Download required NLTK data with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker', quiet=True)

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

logger = logging.getLogger(__name__)

@dataclass
class QuestionInfo:
    """Information about a detected question"""
    text: str
    confidence: float
    question_type: str  # 'wh', 'yes_no', 'choice', 'other'
    keywords: List[str]
    entities: List[str]
    is_valid: bool

class QuestionDetector:
    """
    Detects questions in text and extracts relevant information using NLP
    """
    
    def __init__(self):
        self.config = Config()
        
        # Question word patterns for different types
        self.wh_words = {
            'what', 'when', 'where', 'who', 'whom', 'whose', 'why', 'which', 'how'
        }
        
        # Comprehensive question indicators with regex patterns
        self.question_indicators = [
            r'\b(?:what|when|where|who|whom|whose|why|which|how)\b.*\?',
            r'\b(?:is|are|was|were|do|does|did|will|would|could|should|can|may|might)\b.*\?',
            r'\b(?:tell me|explain|describe|define|list|show me|help me)\b.*\?*',
            r'.*\?$',  # Ends with question mark
            r'\b(?:can you|could you|would you|will you)\b.*',  # Polite requests
        ]
        
        # Compile regex patterns for efficiency
        self.question_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.question_indicators]
        
        # Stop words for keyword extraction (expanded list)
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'can', 'could', 'should', 'would', 'may', 'might', 'must',
            'shall', 'will', 'please', 'tell', 'show', 'help'
        }
    
    def detect_question(self, text: str) -> Optional[QuestionInfo]:
        """
        Detect if the text contains a question and extract information
        
        Args:
            text: The text to analyze
            
        Returns:
            QuestionInfo object if a question is detected, None otherwise
        """
        if not text or len(text.strip()) < self.config.MIN_QUESTION_LENGTH:
            return None
        
        text = text.strip()
        
        # Initialize confidence and question type
        confidence = 0.0
        question_type = 'other'
        
        # Pattern-based detection with weighted scoring
        for i, pattern in enumerate(self.question_patterns):
            if pattern.search(text):
                # Different patterns have different weights
                weights = [0.4, 0.3, 0.2, 0.3, 0.25]  # Corresponding to pattern importance
                confidence += weights[min(i, len(weights)-1)]
        
        # Heuristic-based detection for additional confidence
        confidence += self._calculate_heuristic_confidence(text)
        
        # Determine question type
        question_type = self._classify_question_type(text)
        
        # Extract keywords and entities for better context
        keywords = self._extract_keywords(text)
        entities = self._extract_entities(text)
        
        # Determine if it's a valid question based on confidence and length
        is_valid = (
            confidence >= self.config.QUESTION_CONFIDENCE_THRESHOLD and
            len(text.split()) >= self.config.MIN_QUESTION_LENGTH and
            len(keywords) > 0  # Must have some meaningful keywords
        )
        
        if confidence > 0.1:  # Log potential questions for debugging
            logger.info(f"Question analysis - Text: '{text}', Confidence: {confidence:.2f}, Type: {question_type}")
        
        return QuestionInfo(
            text=text,
            confidence=confidence,
            question_type=question_type,
            keywords=keywords,
            entities=entities,
            is_valid=is_valid
        )
    
    def _calculate_heuristic_confidence(self, text: str) -> float:
        """Calculate confidence based on various heuristics"""
        confidence = 0.0
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return 0.0
        
        # Question mark at the end (strong indicator)
        if text.endswith('?'):
            confidence += 0.4
        
        # WH-words at the beginning (very strong indicator)
        if words[0] in self.wh_words:
            confidence += 0.3
        
        # Auxiliary verbs at the beginning (yes/no questions)
        auxiliary_verbs = {'is', 'are', 'was', 'were', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might'}
        if words[0] in auxiliary_verbs:
            confidence += 0.2
        
        # Question words anywhere in text (weaker indicator)
        for word in self.wh_words:
            if word in text_lower:
                confidence += 0.1
                break  # Only count once
        
        # Imperative question patterns
        imperative_patterns = ['tell me', 'explain', 'describe', 'define', 'show me', 'help me']
        for pattern in imperative_patterns:
            if pattern in text_lower:
                confidence += 0.2
                break
        
        # Rising intonation indicators
        if '?' in text:
            confidence += 0.1
        
        # Polite request patterns
        polite_patterns = ['can you', 'could you', 'would you', 'will you', 'please']
        for pattern in polite_patterns:
            if pattern in text_lower:
                confidence += 0.15
                break
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _classify_question_type(self, text: str) -> str:
        """Classify the type of question for better processing"""
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return 'other'
        
        # WH-questions (what, when, where, etc.)
        if words[0] in self.wh_words or any(word in self.wh_words for word in words[:3]):
            return 'wh'
        
        # Yes/No questions (starting with auxiliary verbs)
        auxiliary_verbs = {'is', 'are', 'was', 'were', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might'}
        if words[0] in auxiliary_verbs:
            return 'yes_no'
        
        # Choice questions (containing "or")
        if ' or ' in text_lower:
            return 'choice'
        
        # Imperative questions
        imperative_starters = ['tell', 'explain', 'describe', 'define', 'show', 'help', 'list']
        if words[0] in imperative_starters:
            return 'imperative'
        
        return 'other'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from the question for context"""
        try:
            # Tokenize and POS tag
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # Extract nouns, verbs, and adjectives, excluding stop words
            keywords = []
            for word, pos in pos_tags:
                if (pos.startswith('N') or pos.startswith('V') or pos.startswith('J')) and \
                   word.isalpha() and \
                   word not in self.stop_words and \
                   len(word) > 2:
                    keywords.append(word)
            
            return keywords[:10]  # Return top 10 keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            # Fallback: simple word extraction
            words = [word.lower() for word in text.split() 
                    if word.isalpha() and len(word) > 2 and word.lower() not in self.stop_words]
            return words[:10]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from the question for better context"""
        try:
            # Tokenize and POS tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Named entity chunking
            chunks = ne_chunk(pos_tags, binary=False)
            
            entities = []
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity = ' '.join([token for token, pos in chunk.leaves()])
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
