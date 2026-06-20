"""
Tokenizer for Sparse Retrieval Service.

This module provides text tokenization functionality for BM25 indexing and querying.
It supports various normalization techniques, stop word removal, stemming, and
custom dictionary support.

Architecture Notes:
- Configurable tokenization pipeline
- Support for recruiter-specific terminology
- Phrase preservation for multi-term queries
- Stop word removal
- Stemming support

SOLID Principles Applied:
- Single Responsibility: Handles only tokenization
- Open/Closed: Open for new tokenization strategies
- Dependency Inversion: Depends on tokenization interface
"""

import logging
import re
import string
from typing import List, Set, Optional, Dict
from collections import defaultdict

logger = logging.getLogger(__name__)


class Tokenizer:
    """
    Text tokenizer for BM25 indexing and querying.
    
    This class provides comprehensive text tokenization with normalization,
    stop word removal, stemming, and custom dictionary support.
    
    Tokenization Pipeline:
        1. Text normalization (lowercase, strip whitespace)
        2. Punctuation removal
        3. Stop word removal
        4. Stemming (optional)
        5. Custom dictionary expansion
        6. Phrase preservation
    
    BM25 Context:
        BM25 relies on term frequency (TF) and inverse document frequency (IDF).
        Proper tokenization is crucial for accurate TF/IDF calculations.
        - Lowercasing ensures case-insensitive matching
        - Stop word removal reduces noise from common terms
        - Stemming groups morphological variants (e.g., "python" and "pythonic")
        - Custom dictionary ensures recruiter-specific terms are preserved
    """
    
    # Default stop words for English
    DEFAULT_STOP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'when', 'where', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
        'don', 'should', 'now', 'i', 'you', 'your', 'we', 'our', 'they', 'their',
        'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'would', 'could', 'should', 'may', 'might', 'must', 'shall'
    }
    
    def __init__(
        self,
        stop_words: Optional[Set[str]] = None,
        use_stemming: bool = False,
        custom_dictionary: Optional[Dict[str, str]] = None,
        preserve_phrases: bool = True,
        phrase_delimiter: str = '"'
    ):
        """
        Initialize the tokenizer.
        
        Args:
            stop_words: Set of stop words to remove (default: DEFAULT_STOP_WORDS)
            use_stemming: Whether to use stemming (default: False)
            custom_dictionary: Dictionary mapping terms to preferred forms
                             e.g., {"ml": "machine learning", "ai": "artificial intelligence"}
            preserve_phrases: Whether to preserve quoted phrases (default: True)
            phrase_delimiter: Character used to delimit phrases (default: '"')
        """
        self.stop_words = stop_words if stop_words is not None else self.DEFAULT_STOP_WORDS
        self.use_stemming = use_stemming
        self.custom_dictionary = custom_dictionary if custom_dictionary is not None else {}
        self.preserve_phrases = preserve_phrases
        self.phrase_delimiter = phrase_delimiter
        
        # Initialize stemmer if needed
        self.stemmer = None
        if self.use_stemming:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
                logger.info("Tokenizer initialized with NLTK Porter Stemmer")
            except ImportError:
                logger.warning("NLTK not available, stemming disabled")
                self.use_stemming = False
        
        logger.info(
            f"Tokenizer initialized: stop_words={len(self.stop_words)}, "
            f"use_stemming={self.use_stemming}, "
            f"custom_dictionary={len(self.custom_dictionary)}"
        )
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a list of tokens.
        
        This method applies the complete tokenization pipeline:
        1. Extract and preserve phrases
        2. Normalize text (lowercase, strip)
        3. Remove punctuation
        4. Split into tokens
        5. Remove stop words
        6. Apply custom dictionary
        7. Apply stemming (if enabled)
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not text or not text.strip():
            return []
        
        # Step 1: Extract phrases if preservation is enabled
        phrases = []
        text_without_phrases = text
        
        if self.preserve_phrases:
            phrases, text_without_phrases = self._extract_phrases(text)
        
        # Step 2: Normalize text
        text_normalized = text_without_phrases.lower().strip()
        
        # Step 3: Remove punctuation
        text_no_punct = self._remove_punctuation(text_normalized)
        
        # Step 4: Split into tokens
        tokens = text_no_punct.split()
        
        # Step 5: Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Step 6: Apply custom dictionary
        tokens = self._apply_custom_dictionary(tokens)
        
        # Step 7: Apply stemming if enabled
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Add preserved phrases back
        if phrases:
            tokens.extend(phrases)
        
        logger.debug(f"Tokenized text: {len(tokens)} tokens")
        
        return tokens
    
    def _extract_phrases(self, text: str) -> tuple[List[str], str]:
        """
        Extract quoted phrases from text.
        
        Args:
            text: Text to extract phrases from
            
        Returns:
            Tuple of (list of phrases, text without phrases)
        """
        phrases = []
        text_without_phrases = text
        
        # Find all quoted phrases
        pattern = f'{self.phrase_delimiter}(.*?){self.phrase_delimiter}'
        matches = re.findall(pattern, text)
        
        for match in matches:
            # Normalize the phrase
            phrase = match.lower().strip()
            # Remove punctuation from phrase
            phrase = self._remove_punctuation(phrase)
            # Replace spaces with underscores for phrase preservation
            phrase = phrase.replace(' ', '_')
            if phrase:
                phrases.append(phrase)
        
        # Remove phrases from text
        text_without_phrases = re.sub(pattern, '', text)
        
        return phrases, text_without_phrases
    
    def _remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text.
        
        Args:
            text: Text to remove punctuation from
            
        Returns:
            Text without punctuation
        """
        # Remove punctuation but preserve underscores for phrases
        translator = str.maketrans('', '', string.punctuation.replace('_', ''))
        return text.translate(translator)
    
    def _apply_custom_dictionary(self, tokens: List[str]) -> List[str]:
        """
        Apply custom dictionary to expand or normalize tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with dictionary applied
        """
        expanded_tokens = []
        
        for token in tokens:
            # Check if token has a custom mapping
            if token in self.custom_dictionary:
                # Expand to the preferred form
                preferred_form = self.custom_dictionary[token]
                # Tokenize the preferred form
                expanded = preferred_form.lower().split()
                expanded_tokens.extend(expanded)
            else:
                expanded_tokens.append(token)
        
        return expanded_tokens
    
    def tokenize_query(self, query: str) -> List[str]:
        """
        Tokenize a search query.
        
        This method is specifically designed for query tokenization and may
        apply different rules than document tokenization (e.g., preserving
        more terms for better recall).
        
        Args:
            query: Search query to tokenize
            
        Returns:
            List of query tokens
        """
        logger.debug(f"Tokenizing query: {query[:50]}...")
        return self.tokenize(query)
    
    def tokenize_document(self, text: str) -> List[str]:
        """
        Tokenize a document for indexing.
        
        This method is specifically designed for document tokenization and
        applies the complete tokenization pipeline.
        
        Args:
            text: Document text to tokenize
            
        Returns:
            List of document tokens
        """
        logger.debug(f"Tokenizing document: {len(text)} characters")
        return self.tokenize(text)
    
    def add_stop_words(self, words: Set[str]) -> None:
        """
        Add custom stop words to the tokenizer.
        
        Args:
            words: Set of stop words to add
        """
        self.stop_words.update(words)
        logger.info(f"Added {len(words)} custom stop words")
    
    def add_custom_dictionary(self, dictionary: Dict[str, str]) -> None:
        """
        Add entries to the custom dictionary.
        
        Args:
            dictionary: Dictionary mapping terms to preferred forms
        """
        self.custom_dictionary.update(dictionary)
        logger.info(f"Added {len(dictionary)} entries to custom dictionary")
    
    def get_vocabulary(self, texts: List[str]) -> Set[str]:
        """
        Get the vocabulary from a list of texts.
        
        Args:
            texts: List of texts to extract vocabulary from
            
        Returns:
            Set of unique tokens (vocabulary)
        """
        vocabulary = set()
        
        for text in texts:
            tokens = self.tokenize_document(text)
            vocabulary.update(tokens)
        
        logger.info(f"Extracted vocabulary: {len(vocabulary)} unique tokens")
        
        return vocabulary
