import re
import logging
import json
import time
import os
from pathlib import Path
import pickle
import hashlib
from datetime import datetime, timedelta
from collections import Counter, defaultdict

# Configure logging
logger = logging.getLogger("TextProcessor")

class HTMLFilter:
    """Simple HTML filter to extract text from HTML content."""
    
    def __init__(self):
        self.text = ""
    
    def feed(self, data):
        """
        Process HTML content and extract plain text.
        
        Args:
            data: HTML content string
        """
        if not data:
            self.text = ""
            return
        
        # Remove HTML tags
        text = re.sub(r'<[^>]*>', ' ', data)
        # Replace HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        self.text = text


class TextProcessor:
    """Simple text processing class without NLTK dependencies."""
    
    def __init__(self):
        """Initialize text processing components."""
        self.stop_words = {
            'the', 'is', 'in', 'to', 'and', 'of', 'a', 'for', 'on', 'with',
            'this', 'that', 'it', 'at', 'from', 'by', 'be', 'was', 'as', 'are',
            'have', 'had', 'has', 'not', 'but', 'what', 'all', 'were', 'when',
            'we', 'there', 'been', 'if', 'more', 'no', 'or', 'about', 'which',
            'their', 'will', 'one', 'would', 'so', 'you', 'an', 'they', 'these',
            # Domain-specific stop words
            'jira', 'confluence', 'page', 'issue', 'ticket', 'project', 'user',
            'atlassian', 'key', 'id', 'com', 'org', 'http', 'https', 'www'
        }
        
        logger.info("Initialized simple text processor (no NLTK)")
    
    def preprocess_text(self, text):
        """
        Simple text preprocessing without NLTK.
        
        Args:
            text: Input text to process
            
        Returns:
            Dict containing processed text data
        """
        if not text or not isinstance(text, str):
            return {
                "original": "",
                "tokens": [],
                "filtered_tokens": [],
                "lemmatized_tokens": [],  # Will be same as filtered_tokens
                "stemmed_tokens": [],     # Will be same as filtered_tokens
                "segments": [],
                "pos_tags": []
            }
        
        # Simple tokenization by splitting on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stop words
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        
        # Simple sentence segmentation
        segments = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        return {
            "original": text,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "lemmatized_tokens": filtered_tokens,  # Use same as filtered tokens
            "stemmed_tokens": filtered_tokens,     # Use same as filtered tokens
            "segments": segments,
            "pos_tags": []  # Empty list instead of POS tags
        }
    
    def extract_keywords(self, text, top_n=10):
        """
        Extract the most important keywords from text without NLTK.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        processed = self.preprocess_text(text)
        
        # Count word frequencies
        word_freq = {}
        for token in processed["filtered_tokens"]:
            if token in word_freq:
                word_freq[token] += 1
            else:
                word_freq[token] = 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N keywords
        return [word for word, freq in sorted_words[:top_n]]
    
    def segment_text(self, text, max_segment_length=500, overlap=50):
        """
        Segment text into smaller chunks (simplified).
        
        Args:
            text: Input text to segment
            max_segment_length: Maximum character length for each segment
            overlap: Number of characters to overlap between segments
            
        Returns:
            List of text segments
        """
        if not text:
            return []
            
        # Simply split text by sentences (periods followed by space)
        segments = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        result = []
        current = ""
        
        for segment in segments:
            if len(current) + len(segment) <= max_segment_length:
                current += ". " + segment if current else segment
            else:
                if current:
                    result.append(current)
                    # Create overlap with previous segment
                    if len(current) > overlap:
                        current = current[-overlap:]
                    else:
                        current = ""
                current += segment
        
        if current:
            result.append(current)
            
        return result
    
    def preprocess_question(self, question):
        """
        Preprocess a question (simplified).
        
        Args:
            question: User question
            
        Returns:
            Dict with processed question information
        """
        # Simple processing
        processed = self.preprocess_text(question)
        
        # Simple question type detection
        question_words = {"what", "who", "where", "when", "why", "how", "which", "can", "do", "is", "are", "will"}
        tokens = question.lower().split()
        
        question_type = "unknown"
        if tokens and tokens[0] in question_words:
            question_type = tokens[0]
        
        # Extract keywords
        keywords = self.extract_keywords(question, top_n=5)
        
        # Simple multi-part question detection
        multi_part_patterns = [
            r'\d+\s*\.\s+',  # Numbered list (1. 2. etc)
            r'first.*?second',
            r'part\s+\d+',
            r'multiple questions',
            r'several questions',
            r'and also',
            r';',
            r'\?.*?\?'  # Multiple question marks
        ]
        
        is_multi_part = any(re.search(pattern, question, re.IGNORECASE) for pattern in multi_part_patterns)
        
        return {
            "processed": processed,
            "type": question_type,
            "keywords": keywords,
            "is_multi_part": is_multi_part,
            "entities": []  # Empty list instead of entities
        }
    
    def split_multi_part_question(self, question):
        """
        Split a multi-part question (simplified).
        
        Args:
            question: Multi-part question
            
        Returns:
            List of individual questions
        """
        # Method 1: Split by question marks followed by space or sentence start
        parts = re.split(r'\?\s+', question)
        
        # Make sure each part ends with a question mark
        for i, part in enumerate(parts):
            if i < len(parts) - 1 and not part.endswith('?'):
                parts[i] = part + '?'
        
        # Filter out empty parts and strip whitespace
        parts = [part.strip() for part in parts if part.strip()]
        
        # If splitting by question marks didn't work well, try numbering patterns
        if len(parts) <= 1:
            # Match numbered patterns like "1. First question 2. Second question"
            numbered_parts = re.split(r'\d+\s*\.\s+', question)
            
            # Remove empty parts and strip whitespace
            numbered_parts = [part.strip() for part in numbered_parts if part.strip()]
            
            if len(numbered_parts) > 1:
                parts = numbered_parts
        
        # If we still don't have multiple parts, try other delimiters
        if len(parts) <= 1:
            other_delimiters = [';', 'and also,', 'additionally,', 'moreover,', 'furthermore,']
            for delimiter in other_delimiters:
                if delimiter in question.lower():
                    parts = question.split(delimiter)
                    parts = [part.strip() for part in parts if part.strip()]
                    break
        
        # If still no success, default to the original question
        if len(parts) <= 1:
            return [question]
        
        return parts


class ContentIndex:
    """An indexing system for content with search capabilities."""
    
    def __init__(self, index_dir=None):
        """
        Initialize the content index.
        
        Args:
            index_dir: Directory to store index files
        """
        self.index_dir = Path(index_dir) if index_dir else Path("./content_index")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize index structures
        self.token_index = defaultdict(list)  # Maps tokens to document IDs
        self.document_metadata = {}  # Maps document IDs to metadata
        self.text_processor = TextProcessor()
        
        # Load existing index if available
        self.load_index()
        
        logger.info(f"Initialized content index in {self.index_dir}")
    
    def index_document(self, document):
        """
        Index a document for search.
        
        Args:
            document: Document dict with metadata, content, etc.
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract document ID and content
            metadata = document.get("metadata", {})
            doc_id = metadata.get("id")
            content = document.get("content", "")
            source_type = document.get("source_type", "unknown")
            
            if not doc_id or not content:
                logger.warning("Document missing ID or content, skipping indexing")
                return False
            
            # Process the content
            processed = self.text_processor.preprocess_text(content)
            
            # Store metadata
            self.document_metadata[doc_id] = {
                "id": doc_id,
                "title": metadata.get("title", "Untitled"),
                "url": metadata.get("url", ""),
                "source_type": source_type,
                "last_indexed": datetime.now().isoformat()
            }
            
            # Index tokens
            for token in processed["lemmatized_tokens"]:
                if doc_id not in self.token_index[token]:
                    self.token_index[token].append(doc_id)
            
            # Save the index
            self.save_index()
            
            logger.info(f"Indexed document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            return False
    
    def search(self, query, max_results=10):
        """
        Search for documents matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of document IDs sorted by relevance
        """
        if not self.token_index:
            logger.warning("Index is empty, cannot search")
            return []
        
        # Process the query
        processed = self.text_processor.preprocess_text(query)
        search_tokens = processed["lemmatized_tokens"]
        
        if not search_tokens:
            logger.warning("No valid search tokens in query")
            return []
        
        # Score documents based on token matches
        scores = defaultdict(int)
        
        # Weight different tokens
        for token in search_tokens:
            matching_docs = self.token_index.get(token, [])
            
            for doc_id in matching_docs:
                # Increase score for each matching token
                scores[doc_id] += 1
        
        # Sort by score (descending)
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top results
        top_docs = [doc_id for doc_id, score in sorted_results[:max_results]]
        
        logger.info(f"Search found {len(top_docs)} results for query: {query}")
        return top_docs
    
    def get_document_metadata(self, doc_id):
        """
        Get metadata for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document metadata or None if not found
        """
        return self.document_metadata.get(doc_id)
    
    def save_index(self):
        """Save the index to disk."""
        try:
            # Save token index
            token_index_path = self.index_dir / "token_index.pkl"
            with open(token_index_path, 'wb') as f:
                pickle.dump(dict(self.token_index), f)
            
            # Save document metadata
            metadata_path = self.index_dir / "document_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.document_metadata, f)
            
            logger.info(f"Saved index with {len(self.token_index)} tokens and {len(self.document_metadata)} documents")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self):
        """Load the index from disk."""
        try:
            # Load token index
            token_index_path = self.index_dir / "token_index.pkl"
            if token_index_path.exists():
                with open(token_index_path, 'rb') as f:
                    self.token_index = defaultdict(list, pickle.load(f))
            
            # Load document metadata
            metadata_path = self.index_dir / "document_metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.document_metadata = pickle.load(f)
            
            logger.info(f"Loaded index with {len(self.token_index)} tokens and {len(self.document_metadata)} documents")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            self.token_index = defaultdict(list)
            self.document_metadata = {}
            return False
    
    def clear_index(self):
        """Clear the index."""
        self.token_index = defaultdict(list)
        self.document_metadata = {}
        
        # Remove index files
        token_index_path = self.index_dir / "token_index.pkl"
        metadata_path = self.index_dir / "document_metadata.pkl"
        
        if token_index_path.exists():
            os.remove(token_index_path)
        
        if metadata_path.exists():
            os.remove(metadata_path)
        
        logger.info("Index cleared")
        return True


class Cache:
    """Advanced caching system with flexible expiration policies."""
    
    def __init__(self, cache_dir=None, default_expiry=86400):
        """
        Initialize the cache system.
        
        Args:
            cache_dir: Directory to store cache files
            default_expiry: Default expiry time in seconds (24 hours)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./enhanced_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_expiry = default_expiry
        self.memory_cache = {}
        self.metadata = {}
        self.metadata_file = self.cache_dir / "metadata.pkl"
        
        # Load metadata if it exists
        self.load_metadata()
        
        logger.info(f"Initialized enhanced cache in {self.cache_dir}")
    
    def get_cache_path(self, key):
        """Get the file path for a cache item."""
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed_key}.pkl"
    
    def is_cache_valid(self, key):
        """Check if a cache item exists and is still valid (not expired)."""
        # Check memory cache first
        if key in self.memory_cache:
            # Check if expired
            metadata = self.metadata.get(key, {})
            expiry_time = metadata.get("expiry_time", 0)
            
            if expiry_time == 0 or time.time() < expiry_time:
                return True
            else:
                # Expired, remove from memory cache
                del self.memory_cache[key]
        
        # Check file cache
        cache_path = self.get_cache_path(key)
        if not cache_path.exists():
            return False
        
        # Check if expired based on metadata
        metadata = self.metadata.get(key, {})
        expiry_time = metadata.get("expiry_time", 0)
        
        if expiry_time == 0:
            # No expiry time, check file modification time
            mtime = cache_path.stat().st_mtime
            if time.time() - mtime > self.default_expiry:
                return False
            return True
        
        # Check if expired based on expiry time
        return time.time() < expiry_time
    
    def get(self, key, default=None):
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default if not found or expired
        """
        if not self.is_cache_valid(key):
            return default
        
        # Try memory cache first
        if key in self.memory_cache:
            logger.debug(f"Cache hit (memory): {key}")
            return self.memory_cache[key]
        
        # Try file cache
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
                
                # Update memory cache
                self.memory_cache[key] = value
                
                logger.debug(f"Cache hit (file): {key}")
                return value
        except Exception as e:
            logger.error(f"Error reading cache for {key}: {str(e)}")
            return default
    
    def set(self, key, value, expiry=None, metadata=None):
        """
        Store a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expiry: Expiry time in seconds (None for default)
            metadata: Additional metadata to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate expiry time
            expiry_time = 0
            if expiry is not None:
                expiry_time = time.time() + expiry
            elif self.default_expiry > 0:
                expiry_time = time.time() + self.default_expiry
            
            # Update metadata
            item_metadata = metadata or {}
            item_metadata["expiry_time"] = expiry_time
            item_metadata["cached_at"] = time.time()
            self.metadata[key] = item_metadata
            
            # Update memory cache
            self.memory_cache[key] = value
            
            # Update file cache
            cache_path = self.get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Save metadata
            self.save_metadata()
            
            logger.debug(f"Cached: {key}")
            return True
        except Exception as e:
            logger.error(f"Error writing cache for {key}: {str(e)}")
            return False
    
    def delete(self, key):
        """
        Delete a cache item.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            # Remove from metadata
            if key in self.metadata:
                del self.metadata[key]
                self.save_metadata()
            
            # Remove from file cache
            cache_path = self.get_cache_path(key)
            if cache_path.exists():
                os.remove(cache_path)
            
            logger.debug(f"Deleted from cache: {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting cache for {key}: {str(e)}")
            return False
    
    def clear(self, pattern=None):
        """
        Clear cache items.
        
        Args:
            pattern: Regex pattern to match keys (None for all)
            
        Returns:
            Number of items cleared
        """
        try:
            count = 0
            keys_to_delete = []
            
            # Find keys to delete
            if pattern:
                regex = re.compile(pattern)
                for key in self.metadata:
                    if regex.search(key):
                        keys_to_delete.append(key)
            else:
                keys_to_delete = list(self.metadata.keys())
            
            # Delete each key
            for key in keys_to_delete:
                if self.delete(key):
                    count += 1
            
            logger.info(f"Cleared {count} cache items")
            return count
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return 0
    
    def save_metadata(self):
        """Save metadata to disk."""
        try:
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            return True
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            return False
    
    def load_metadata(self):
        """Load metadata from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded cache metadata with {len(self.metadata)} entries")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            self.metadata = {}
            return False
