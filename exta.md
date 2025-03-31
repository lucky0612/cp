# Environment Configuration

# Application settings
FLASK_ENV=development
FLASK_APP=app.py
SECRET_KEY=your-secret-key-here
DEBUG=True
PORT=5000

# Google Cloud / Vertex AI settings
PROJECT_ID=prj-dv-cws-4363
REGION=us-central1
MODEL_NAME=gemini-2.0-flash-001

# Confluence settings
CONFLUENCE_URL=https://cmegroup.atlassian.net
CONFLUENCE_USERNAME=your-username
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_SSL_VERIFY=False

# JIRA settings
JIRA_URL=https://cmegroup-jira.atlassian.net
JIRA_USERNAME=your-username
JIRA_API_TOKEN=your-api-token
JIRA_SSL_VERIFY=False

# Remedy settings
REMEDY_URL=https://cmegroup-restapi.onbmc.com
REMEDY_USERNAME=your-username
REMEDY_PASSWORD=your-password
REMEDY_SSL_VERIFY=False

# RAG Engine settings
CACHE_DIR=./cache
EMBEDDINGS_DIMENSION=768
EMBEDDINGS_CACHE_SIZE=1000
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5




















import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Application settings
FLASK_ENV = os.getenv("FLASK_ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")
PORT = int(os.getenv("PORT", 5000))

# Google Cloud / Vertex AI settings
PROJECT_ID = os.getenv("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.getenv("REGION", "us-central1")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-001")

# Confluence settings
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", "https://cmegroup.atlassian.net")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_SSL_VERIFY = os.getenv("CONFLUENCE_SSL_VERIFY", "False").lower() == "true"

# JIRA settings
JIRA_URL = os.getenv("JIRA_URL", "https://cmegroup-jira.atlassian.net")
JIRA_USERNAME = os.getenv("JIRA_USERNAME")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_SSL_VERIFY = os.getenv("JIRA_SSL_VERIFY", "False").lower() == "true"

# Remedy settings
REMEDY_URL = os.getenv("REMEDY_URL", "https://cmegroup-restapi.onbmc.com")
REMEDY_USERNAME = os.getenv("REMEDY_USERNAME")
REMEDY_PASSWORD = os.getenv("REMEDY_PASSWORD")
REMEDY_SSL_VERIFY = os.getenv("REMEDY_SSL_VERIFY", "False").lower() == "true"

# RAG Engine settings
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
EMBEDDINGS_DIMENSION = int(os.getenv("EMBEDDINGS_DIMENSION", 768))
EMBEDDINGS_CACHE_SIZE = int(os.getenv("EMBEDDINGS_CACHE_SIZE", 1000))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enterprise_rag.log"),
        logging.StreamHandler()
    ]
)

def get_logger(name):
    """Get a logger with the specified name."""
    return logging.getLogger(name)























import logging
from config import get_logger

# Create a centralized logger
logger = get_logger("enterprise_rag")

def setup_module_logger(module_name):
    """Create a logger for a specific module."""
    return get_logger(f"enterprise_rag.{module_name}")




















from utils.logger import logger, setup_module_logger
from utils.cache import Cache, get_cache
from utils.content_parser import (
    extract_text_from_html, 
    extract_tables_from_html, 
    clean_text,
    extract_text_from_confluence_content,
    extract_code_blocks
)

__all__ = [
    'logger', 
    'setup_module_logger',
    'Cache',
    'get_cache',
    'extract_text_from_html',
    'extract_tables_from_html',
    'clean_text',
    'extract_text_from_confluence_content',
    'extract_code_blocks'
]





















import os
import json
import pickle
import hashlib
import time
from collections import OrderedDict
from threading import Lock
from utils.logger import setup_module_logger
from config import CACHE_DIR

logger = setup_module_logger("cache")

class Cache:
    """A thread-safe LRU cache implementation with disk persistence."""
    
    def __init__(self, name, max_size=1000, cache_dir=CACHE_DIR):
        self.name = name
        self.max_size = max_size
        self.cache_dir = os.path.join(cache_dir, name)
        self.cache = OrderedDict()
        self.lock = Lock()
        self.metadata_file = os.path.join(self.cache_dir, "metadata.json")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load cache metadata if it exists
        self._load_metadata()
        
    def _load_metadata(self):
        """Load cache metadata from disk."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                # Only load metadata, not the actual cache values
                for key, info in metadata.items():
                    self.cache[key] = None  # Just placeholder, actual data loaded on demand
        except Exception as e:
            logger.error(f"Error loading cache metadata: {str(e)}")
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            metadata = {k: {"timestamp": time.time()} for k in self.cache.keys()}
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {str(e)}")
    
    def _get_item_path(self, key):
        """Get file path for a cached item."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.pkl")
    
    def _save_item(self, key, value):
        """Save an item to disk."""
        try:
            item_path = self._get_item_path(key)
            with open(item_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Error saving cache item: {str(e)}")
    
    def _load_item(self, key):
        """Load an item from disk."""
        try:
            item_path = self._get_item_path(key)
            if os.path.exists(item_path):
                with open(item_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cache item: {str(e)}")
        return None
    
    def get(self, key, default=None):
        """Get an item from the cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                
                # Load from disk if not loaded yet
                if value is None:
                    value = self._load_item(key)
                
                self.cache[key] = value
                return value
            return default
    
    def set(self, key, value):
        """Add an item to the cache."""
        with self.lock:
            # If key exists, remove it first
            if key in self.cache:
                self.cache.pop(key)
            
            # Add to cache
            self.cache[key] = value
            
            # Save to disk
            self._save_item(key, value)
            
            # If cache is too large, remove oldest item
            if len(self.cache) > self.max_size:
                oldest_key, _ = self.cache.popitem(last=False)
                try:
                    # Remove from disk as well
                    os.remove(self._get_item_path(oldest_key))
                except:
                    pass
            
            # Update metadata
            self._save_metadata()
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            
            # Remove all files in cache directory
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except:
                        pass
            
            # Update metadata
            self._save_metadata()

# Cache instances
_caches = {}

def get_cache(name, max_size=1000):
    """Get or create a cache with the specified name."""
    if name not in _caches:
        _caches[name] = Cache(name, max_size)
    return _caches[name]


















import re
import html2text
from bs4 import BeautifulSoup
from utils.logger import setup_module_logger

logger = setup_module_logger("content_parser")

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Replace multiple newlines with a single one
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single one
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_text_from_html(html_content):
    """Extract plain text from HTML content."""
    if not html_content:
        return ""
    
    try:
        # Use html2text to convert HTML to markdown
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = False
        converter.ignore_tables = False
        
        # Convert HTML to markdown
        markdown_text = converter.handle(html_content)
        
        # Clean the text
        return clean_text(markdown_text)
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {str(e)}")
        # Fallback to BeautifulSoup if html2text fails
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            return clean_text(soup.get_text())
        except Exception as e2:
            logger.error(f"BeautifulSoup fallback failed: {str(e2)}")
            return ""

def extract_tables_from_html(html_content):
    """Extract tables from HTML content as markdown tables."""
    if not html_content:
        return []
    
    tables = []
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        html_tables = soup.find_all('table')
        
        for table in html_tables:
            rows = table.find_all('tr')
            if not rows:
                continue
            
            markdown_table = []
            
            # Process header row
            header_cells = rows[0].find_all(['th', 'td'])
            if header_cells:
                header = "|" + "|".join([clean_text(cell.get_text()) for cell in header_cells]) + "|"
                markdown_table.append(header)
                
                # Add separator row
                separator = "|" + "|".join(["---" for _ in header_cells]) + "|"
                markdown_table.append(separator)
            
            # Process data rows
            for row in rows[1:]:
                cells = row.find_all('td')
                if cells:
                    data_row = "|" + "|".join([clean_text(cell.get_text()) for cell in cells]) + "|"
                    markdown_table.append(data_row)
            
            if len(markdown_table) > 1:  # Only add if we have a header and at least one data row
                tables.append("\n".join(markdown_table))
        
        return tables
    except Exception as e:
        logger.error(f"Error extracting tables from HTML: {str(e)}")
        return []

def extract_code_blocks(text):
    """Extract code blocks from markdown or HTML content."""
    if not text:
        return []
    
    code_blocks = []
    
    # Find markdown code blocks (```...```)
    markdown_pattern = r'```(?:\w+)?\s*(.*?)\s*```'
    markdown_matches = re.findall(markdown_pattern, text, re.DOTALL)
    code_blocks.extend(markdown_matches)
    
    # Find HTML code blocks (<pre>...</pre> or <code>...</code>)
    try:
        soup = BeautifulSoup(text, 'html.parser')
        
        # Extract from <pre> tags
        for pre in soup.find_all('pre'):
            code_blocks.append(pre.get_text())
        
        # Extract from <code> tags that are not inside <pre>
        for code in soup.find_all('code'):
            if not code.find_parent('pre'):
                code_blocks.append(code.get_text())
    except:
        pass
    
    return [clean_text(block) for block in code_blocks if clean_text(block)]

def extract_text_from_confluence_content(content):
    """Extract text from Confluence content, handling specific Confluence formats."""
    if not content:
        return ""
    
    # Handle Confluence storage format (XML-like)
    content_text = ""
    
    try:
        # Try to parse as HTML/XML
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract text from specific Confluence elements
        for macro in soup.find_all('ac:structured-macro'):
            macro_name = macro.get('ac:name', '')
            
            if macro_name == 'code':
                # Handle code blocks
                code_param = macro.find('ac:parameter', {'ac:name': 'language'})
                language = code_param.text if code_param else ""
                
                code_body = macro.find('ac:plain-text-body')
                if code_body:
                    content_text += f"\n```{language}\n{code_body.text}\n```\n"
            
            elif macro_name == 'note' or macro_name == 'info' or macro_name == 'warning':
                # Handle note/info/warning macros
                title_param = macro.find('ac:parameter', {'ac:name': 'title'})
                title = f"**{title_param.text}**\n" if title_param else ""
                
                rich_body = macro.find('ac:rich-text-body')
                if rich_body:
                    body_text = rich_body.get_text()
                    content_text += f"\n> {title}{body_text}\n"
        
        # If we couldn't extract specific content, fall back to regular HTML extraction
        if not content_text:
            content_text = extract_text_from_html(content)
        
        return clean_text(content_text)
    except Exception as e:
        logger.error(f"Error extracting text from Confluence content: {str(e)}")
        # Fall back to regular HTML extraction
        return extract_text_from_html(content)



















from rag_engine.processor import RAGProcessor
from rag_engine.chunking import chunk_text, chunk_document
from rag_engine.embedding import (
    get_embeddings, 
    embed_query, 
    embed_documents
)
from rag_engine.retrieval import (
    RetrievalEngine,
    combine_search_results
)
from rag_engine.gemini_integration import GeminiLLM

__all__ = [
    'RAGProcessor',
    'chunk_text',
    'chunk_document',
    'get_embeddings',
    'embed_query',
    'embed_documents',
    'RetrievalEngine',
    'combine_search_results',
    'GeminiLLM'
]














import re
import nltk
from typing import List, Dict, Any, Tuple
from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import setup_module_logger

logger = setup_module_logger("chunking")

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        logger.warning("Failed to download NLTK punkt. Will use basic chunking instead.")

def detect_sentence_boundaries(text: str) -> List[int]:
    """
    Detect sentence boundaries in text using NLTK if available.
    
    Args:
        text: The text to analyze
        
    Returns:
        List of indices indicating sentence boundaries
    """
    try:
        sentences = nltk.sent_tokenize(text)
        boundaries = []
        current_idx = 0
        
        for sentence in sentences:
            # Find the sentence in the original text
            # (needed because sent_tokenize might normalize whitespace)
            start_idx = text.find(sentence, current_idx)
            if start_idx == -1:
                # If exact match not found, try a more flexible approach
                # This handles cases where whitespace might differ
                words = re.escape(sentence.strip()).split(r'\s+')
                pattern = r'\s+'.join(words)
                match = re.search(pattern, text[current_idx:])
                if match:
                    start_idx = current_idx + match.start()
                else:
                    # Skip this sentence if we can't find it
                    continue
            
            # Update the current index and add the boundary
            current_idx = start_idx + len(sentence)
            boundaries.append(current_idx)
        
        return boundaries
    except Exception as e:
        logger.warning(f"Error detecting sentence boundaries: {str(e)}. Using fallback method.")
        return []

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks, trying to preserve sentence boundaries.
    
    Args:
        text: The text to split
        chunk_size: The target size of each chunk
        chunk_overlap: The overlap between consecutive chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Get sentence boundaries
    sentence_boundaries = detect_sentence_boundaries(text)
    
    # If sentence detection failed, use a simpler approach
    if not sentence_boundaries:
        # Split by newlines and then recombine to respect paragraph boundaries
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph would exceed chunk size, 
            # and we already have content, start a new chunk
            if current_chunk and len(current_chunk) + len(para) > chunk_size:
                chunks.append(current_chunk)
                # Start next chunk with overlap from previous chunk
                if len(current_chunk) > chunk_overlap:
                    current_chunk = current_chunk[-chunk_overlap:] + '\n' + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += '\n'
                current_chunk += para
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    # Create chunks based on sentence boundaries
    chunks = []
    current_chunk_start = 0
    
    while current_chunk_start < len(text):
        # Determine end of this chunk (target position)
        target_end = current_chunk_start + chunk_size
        
        if target_end >= len(text):
            # If we're near the end, just take the rest of the text
            chunks.append(text[current_chunk_start:])
            break
        
        # Find the closest sentence boundary to our target
        closest_boundary = None
        for boundary in sentence_boundaries:
            if boundary > current_chunk_start and boundary <= target_end:
                closest_boundary = boundary
            elif boundary > target_end:
                break
        
        if closest_boundary:
            # We found a sentence boundary in range
            chunks.append(text[current_chunk_start:closest_boundary])
            current_chunk_start = closest_boundary - chunk_overlap
            current_chunk_start = max(0, current_chunk_start)
        else:
            # No sentence boundary found, just use the target end
            chunks.append(text[current_chunk_start:target_end])
            current_chunk_start = target_end - chunk_overlap
    
    return chunks

def chunk_document(document: Dict[str, Any], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split a document into chunks, preserving metadata.
    
    Args:
        document: A dictionary with at least 'content' and 'metadata' keys
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks with preserved metadata and added chunk info
    """
    if not document or 'content' not in document:
        return []
    
    content = document.get('content', '')
    metadata = document.get('metadata', {})
    source = document.get('source', '')
    
    # Skip empty documents
    if not content.strip():
        return []
    
    # Chunk the text
    text_chunks = chunk_text(content, chunk_size, chunk_overlap)
    
    # Create document chunks with metadata
    doc_chunks = []
    for i, chunk_text in enumerate(text_chunks):
        chunk = {
            'content': chunk_text,
            'metadata': {
                **metadata,
                'chunk_index': i,
                'chunk_count': len(text_chunks)
            },
            'source': source
        }
        doc_chunks.append(chunk)
    
    return doc_chunks






















import os
import numpy as np
from typing import List, Dict, Any, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from config import EMBEDDINGS_DIMENSION, EMBEDDINGS_CACHE_SIZE
from utils.logger import setup_module_logger
from utils.cache import get_cache

logger = setup_module_logger("embedding")

# Initialize cache
embeddings_cache = get_cache("embeddings", max_size=EMBEDDINGS_CACHE_SIZE)

# Global models for reuse
tfidf_vectorizer = None
svd_transformer = None
actual_dimension = EMBEDDINGS_DIMENSION  # This will be adjusted based on available features

def _get_or_create_models(texts: List[str] = None) -> Tuple[TfidfVectorizer, TruncatedSVD]:
    """
    Get existing models or create new ones based on input texts.
    
    Args:
        texts: Optional list of texts to fit models on
        
    Returns:
        Tuple of (tfidf_vectorizer, svd_transformer)
    """
    global tfidf_vectorizer, svd_transformer, actual_dimension
    
    # If no texts provided, return existing models (may be None)
    if not texts:
        return tfidf_vectorizer, svd_transformer
    
    # Create and fit TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        analyzer='word',
        stop_words='english',
        token_pattern=r'\b\w+\b',  # Match whole words
        min_df=1,  # Include terms that appear in at least 1 document
        max_df=0.9,  # Exclude terms that appear in more than 90% of documents
        ngram_range=(1, 2)  # Include unigrams and bigrams
    )
    
    try:
        # Fit the vectorizer
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        
        # Determine the actual SVD dimension based on available features
        n_features = tfidf_matrix.shape[1]
        actual_dimension = min(EMBEDDINGS_DIMENSION, n_features)
        
        logger.info(f"TF-IDF vectorizer created with {n_features} features. Using dimension: {actual_dimension}")
        
        # Create and fit SVD transformer if we have enough features
        if actual_dimension > 0:
            svd_transformer = TruncatedSVD(n_components=actual_dimension, random_state=42)
            svd_transformer.fit(tfidf_matrix)
            logger.info(f"SVD transformer created with {actual_dimension} components")
        else:
            logger.warning("Not enough features for SVD. Will use TF-IDF directly.")
            svd_transformer = None
    except Exception as e:
        logger.error(f"Error creating embedding models: {str(e)}")
        # Ensure we have at least a vectorizer
        if tfidf_vectorizer is None:
            tfidf_vectorizer = TfidfVectorizer()
    
    return tfidf_vectorizer, svd_transformer

def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        Numpy array of embeddings, shape (len(texts), dimension)
    """
    if not texts:
        return np.zeros((0, EMBEDDINGS_DIMENSION))
    
    # Check cache first for all texts
    cached_embeddings = []
    texts_to_embed = []
    text_indices = []
    
    for i, text in enumerate(texts):
        if not text or not text.strip():
            # Handle empty text
            cached_embeddings.append(np.zeros(EMBEDDINGS_DIMENSION))
            continue
            
        # Create a cache key
        cache_key = f"embedding:{hash(text)}"
        
        # Try to get from cache
        embedding = embeddings_cache.get(cache_key)
        
        if embedding is not None:
            # Ensure consistent dimensionality
            if len(embedding) < EMBEDDINGS_DIMENSION:
                # Pad with zeros if needed
                padding = np.zeros(EMBEDDINGS_DIMENSION - len(embedding))
                embedding = np.concatenate([embedding, padding])
            elif len(embedding) > EMBEDDINGS_DIMENSION:
                # Truncate if needed
                embedding = embedding[:EMBEDDINGS_DIMENSION]
                
            cached_embeddings.append(embedding)
        else:
            # Need to generate embedding
            cached_embeddings.append(None)  # Placeholder
            texts_to_embed.append(text)
            text_indices.append(i)
    
    # If we have texts to embed, generate embeddings
    if texts_to_embed:
        # Get or create models
        vectorizer, svd = _get_or_create_models(texts_to_embed)
        
        try:
            # Generate TF-IDF vectors
            tfidf_vectors = vectorizer.transform(texts_to_embed)
            
            # Generate embeddings using SVD if available
            if svd is not None:
                new_embeddings = svd.transform(tfidf_vectors)
            else:
                # Use TF-IDF directly, normalized
                new_embeddings = tfidf_vectors.toarray()
                norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                new_embeddings = new_embeddings / norms
            
            # Ensure consistent dimensionality
            padded_embeddings = []
            for embedding in new_embeddings:
                if len(embedding) < EMBEDDINGS_DIMENSION:
                    # Pad with zeros
                    padding = np.zeros(EMBEDDINGS_DIMENSION - len(embedding))
                    embedding = np.concatenate([embedding, padding])
                elif len(embedding) > EMBEDDINGS_DIMENSION:
                    # Truncate
                    embedding = embedding[:EMBEDDINGS_DIMENSION]
                
                padded_embeddings.append(embedding)
            
            # Cache and update results
            for i, idx in enumerate(text_indices):
                embedding = padded_embeddings[i]
                cached_embeddings[idx] = embedding
                
                # Create a cache key
                cache_key = f"embedding:{hash(texts[idx])}"
                
                # Add to cache
                embeddings_cache.set(cache_key, embedding)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Create zero embeddings for failures
            for idx in text_indices:
                if cached_embeddings[idx] is None:
                    cached_embeddings[idx] = np.zeros(EMBEDDINGS_DIMENSION)
    
    # Combine all embeddings and ensure dimensionality
    final_embeddings = np.array([
        embedding if embedding is not None else np.zeros(EMBEDDINGS_DIMENSION)
        for embedding in cached_embeddings
    ])
    
    # Make sure we have the right shape
    if final_embeddings.shape[1] != EMBEDDINGS_DIMENSION:
        # Pad or truncate as needed
        padded = np.zeros((len(final_embeddings), EMBEDDINGS_DIMENSION))
        for i, embedding in enumerate(final_embeddings):
            if len(embedding) > EMBEDDINGS_DIMENSION:
                padded[i] = embedding[:EMBEDDINGS_DIMENSION]
            else:
                padded[i, :len(embedding)] = embedding
        final_embeddings = padded
    
    return final_embeddings

def embed_query(query: str) -> np.ndarray:
    """
    Get embedding for a single query string.
    
    Args:
        query: The query text to embed
        
    Returns:
        Numpy array of the query embedding
    """
    if not query or not query.strip():
        return np.zeros(EMBEDDINGS_DIMENSION)
    
    embeddings = get_embeddings([query])
    return embeddings[0]

def embed_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Embed a list of documents, adding embeddings to each document.
    
    Args:
        documents: List of document dictionaries, each with at least a 'content' key
        
    Returns:
        List of documents with embeddings added
    """
    if not documents:
        return []
    
    # Extract text content from documents
    texts = [doc.get('content', '') for doc in documents]
    
    # Get embeddings for all texts
    all_embeddings = get_embeddings(texts)
    
    # Add embeddings to documents
    embedded_docs = []
    for i, doc in enumerate(documents):
        embedded_doc = doc.copy()
        embedded_doc['embedding'] = all_embeddings[i]
        embedded_docs.append(embedded_doc)
    
    return embedded_docs


















import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from config import TOP_K_RESULTS, CACHE_DIR
from utils.logger import setup_module_logger
from utils.cache import get_cache
from rag_engine.embedding import embed_query

logger = setup_module_logger("retrieval")

class RetrievalEngine:
    """Retrieval engine for finding relevant documents using hybrid search."""
    
    def __init__(self, source_id: str = "default"):
        """
        Initialize the retrieval engine.
        
        Args:
            source_id: ID of the data source (used for caching)
        """
        self.source_id = source_id
        self.documents = []
        self.index = None
        self.tfidf_vectorizer = None
        self.embeddings = None
        self.document_cache = get_cache(f"documents_{source_id}")
        self.index_path = os.path.join(CACHE_DIR, f"index_{source_id}.faiss")
        self.metadata_path = os.path.join(CACHE_DIR, f"index_metadata_{source_id}.pkl")
        
        # Try to load existing index
        self._load_index()
    
    def _load_index(self) -> bool:
        """
        Load index from disk if available.
        
        Returns:
            True if index was loaded successfully, False otherwise
        """
        try:
            # Check if both index and metadata files exist
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Extract data from metadata
                self.documents = metadata.get('documents', [])
                
                # Only load index if we have documents
                if self.documents:
                    # Load FAISS index
                    self.index = faiss.read_index(self.index_path)
                    
                    # Load TF-IDF vectorizer
                    self.tfidf_vectorizer = metadata.get('tfidf_vectorizer')
                    
                    # Load embeddings
                    self.embeddings = metadata.get('embeddings')
                    
                    logger.info(f"Loaded index for source {self.source_id} with {len(self.documents)} documents")
                    return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
        
        return False
    
    def _save_index(self):
        """Save index to disk."""
        try:
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            metadata = {
                'documents': self.documents,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'embeddings': self.embeddings
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved index for source {self.source_id} with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def update_documents(self, documents: List[Dict[str, Any]]):
        """
        Update the document index with new documents.
        
        Args:
            documents: List of document dictionaries with 'content', 'metadata', and 'embedding' keys
        """
        if not documents:
            return
        
        # Ensure all documents have embeddings
        if any('embedding' not in doc for doc in documents):
            logger.error("Some documents do not have embeddings. Please embed documents before updating the index.")
            return
        
        # Ensure all documents have an ID
        for i, doc in enumerate(documents):
            if 'id' not in doc.get('metadata', {}):
                doc['metadata']['id'] = f"{self.source_id}_{len(self.documents) + i}"
        
        # Extract embeddings as numpy array
        embeddings = np.array([doc['embedding'] for doc in documents], dtype=np.float32)
        
        if self.index is None:
            # Create new index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.embeddings = embeddings
            self.documents = documents
            
            # Add all documents to index
            self.index.add(embeddings)
        else:
            # Append to existing index
            self.index.add(embeddings)
            
            # Append to documents and embeddings
            self.documents.extend(documents)
            if self.embeddings is None:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])
        
        # Update TF-IDF vectorizer for keyword search
        corpus = [doc.get('content', '') for doc in self.documents]
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            stop_words='english'
        )
        self.tfidf_vectorizer.fit(corpus)
        
        # Save updated index
        self._save_index()
    
    def similarity_search(self, query: str, k: int = TOP_K_RESULTS, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search using embeddings.
        
        Args:
            query: The query string
            k: Number of results to return
            threshold: Optional similarity threshold
            
        Returns:
            List of document dictionaries with added similarity scores
        """
        if not self.index or not self.documents:
            return []
        
        # Get query embedding
        query_embedding = embed_query(query)
        
        # Convert to float32 and reshape to 2D array (batch of 1)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search the index
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding, k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
                
            distance = distances[0][i]
            similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
            
            # Apply threshold if provided
            if threshold is not None and similarity < threshold:
                continue
            
            # Get document
            doc = self.documents[idx].copy()
            doc['similarity'] = float(similarity)
            results.append(doc)
        
        return results
    
    def keyword_search(self, query: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """
        Perform keyword search using TF-IDF.
        
        Args:
            query: The query string
            k: Number of results to return
            
        Returns:
            List of document dictionaries with added BM25 scores
        """
        if not self.tfidf_vectorizer or not self.documents:
            return []
        
        try:
            # Transform query to TF-IDF vector
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Transform all documents
            doc_vectors = self.tfidf_vectorizer.transform([doc.get('content', '') for doc in self.documents])
            
            # Calculate dot product for similarity
            scores = (query_vector * doc_vectors.T).toarray().flatten()
            
            # Get top-k indices
            k = min(k, len(self.documents))
            top_indices = np.argsort(scores)[-k:][::-1]
            
            # Get results
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    doc = self.documents[idx].copy()
                    doc['bm25_score'] = float(scores[idx])
                    results.append(doc)
            
            return results
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []
    
    def hybrid_search(self, query: str, k: int = TOP_K_RESULTS, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: The query string
            k: Number of results to return
            semantic_weight: Weight for semantic search (1-semantic_weight = keyword weight)
            
        Returns:
            List of document dictionaries with added hybrid scores
        """
        # Get results from both methods
        semantic_results = self.similarity_search(query, k=k*2)
        keyword_results = self.keyword_search(query, k=k*2)
        
        return combine_search_results(
            semantic_results, 
            keyword_results, 
            k=k, 
            semantic_weight=semantic_weight
        )
    
    def search(self, query: str, search_type: str = "hybrid", k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query.
        
        Args:
            query: The query string
            search_type: Type of search to perform ('semantic', 'keyword', or 'hybrid')
            k: Number of results to return
            
        Returns:
            List of document dictionaries with added scores
        """
        if search_type == "semantic":
            return self.similarity_search(query, k=k)
        elif search_type == "keyword":
            return self.keyword_search(query, k=k)
        else:  # hybrid is the default
            return self.hybrid_search(query, k=k)

def combine_search_results(
    semantic_results: List[Dict[str, Any]], 
    keyword_results: List[Dict[str, Any]], 
    k: int = TOP_K_RESULTS, 
    semantic_weight: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Combine results from semantic and keyword search.
    
    Args:
        semantic_results: Results from semantic search
        keyword_results: Results from keyword search
        k: Number of results to return
        semantic_weight: Weight for semantic search (1-semantic_weight = keyword weight)
        
    Returns:
        Combined and reranked results
    """
    # Normalize weights to sum to 1.0
    keyword_weight = 1.0 - semantic_weight
    
    # Create dictionaries to track scores
    combined_scores = {}
    
    # Process semantic results
    for doc in semantic_results:
        doc_id = doc['metadata'].get('id')
        if not doc_id:
            continue
            
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                'doc': doc,
                'semantic_score': 0.0,
                'keyword_score': 0.0,
                'combined_score': 0.0
            }
        
        combined_scores[doc_id]['semantic_score'] = doc.get('similarity', 0.0)
    
    # Process keyword results
    for doc in keyword_results:
        doc_id = doc['metadata'].get('id')
        if not doc_id:
            continue
            
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                'doc': doc,
                'semantic_score': 0.0,
                'keyword_score': 0.0,
                'combined_score': 0.0
            }
        
        combined_scores[doc_id]['keyword_score'] = doc.get('bm25_score', 0.0)
    
    # Normalize scores
    semantic_scores = [entry['semantic_score'] for entry in combined_scores.values()]
    keyword_scores = [entry['keyword_score'] for entry in combined_scores.values()]
    
    if semantic_scores:
        max_semantic = max(semantic_scores) or 1.0
        min_semantic = min(semantic_scores) or 0.0
    else:
        max_semantic, min_semantic = 1.0, 0.0
        
    if keyword_scores:
        max_keyword = max(keyword_scores) or 1.0
        min_keyword = min(keyword_scores) or 0.0
    else:
        max_keyword, min_keyword = 1.0, 0.0
    
    # Calculate combined scores
    for doc_id, entry in combined_scores.items():
        # Normalize semantic score
        normalized_semantic = 0.0
        if max_semantic > min_semantic:
            normalized_semantic = (entry['semantic_score'] - min_semantic) / (max_semantic - min_semantic)
        
        # Normalize keyword score
        normalized_keyword = 0.0
        if max_keyword > min_keyword:
            normalized_keyword = (entry['keyword_score'] - min_keyword) / (max_keyword - min_keyword)
        
        # Calculate combined score
        combined_score = (semantic_weight * normalized_semantic) + (keyword_weight * normalized_keyword)
        entry['combined_score'] = combined_score
    
    # Sort by combined score
    sorted_results = sorted(
        combined_scores.values(), 
        key=lambda x: x['combined_score'], 
        reverse=True
    )
    
    # Get top-k results
    top_results = sorted_results[:k]
    
    # Format results
    final_results = []
    for entry in top_results:
        doc = entry['doc'].copy()
        doc['semantic_score'] = float(entry['semantic_score'])
        doc['keyword_score'] = float(entry['keyword_score'])
        doc['combined_score'] = float(entry['combined_score'])
        final_results.append(doc)
    
    return final_results






















import os
import time
from typing import List, Dict, Any, Optional, Tuple, Generator

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel, Part
from google.api_core.exceptions import GoogleAPICallError

from config import PROJECT_ID, REGION, MODEL_NAME
from utils.logger import setup_module_logger

logger = setup_module_logger("gemini_integration")

class GeminiLLM:
    """Integration with Google's Gemini models via Vertex AI."""
    
    def __init__(self, project_id: str = PROJECT_ID, region: str = REGION, model_name: str = MODEL_NAME):
        """
        Initialize the Gemini LLM integration.
        
        Args:
            project_id: Google Cloud project ID
            region: Google Cloud region
            model_name: Gemini model name
        """
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        self.model = None
        
        # Initialize Vertex AI
        self._initialize_vertex()
    
    def _initialize_vertex(self):
        """Initialize Vertex AI and set up the model."""
        try:
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.region)
            
            # Set up the model
            self.model = GenerativeModel(self.model_name)
            
            logger.info(f"Initialized Gemini model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Vertex AI: {str(e)}")
            raise
    
    def _build_prompt(self, query: str, context: List[Dict[str, Any]], conversation: List[Dict[str, str]] = None) -> str:
        """
        Build a prompt for the LLM using query and context.
        
        Args:
            query: The user's query
            context: List of context documents from RAG
            conversation: Optional conversation history
            
        Returns:
            Formatted prompt string
        """
        # Start with system instructions
        prompt = """You are a helpful assistant that provides accurate information based on the provided context. 
If you don't know the answer based on the context, admit that you don't know rather than making up information.
Always cite your sources when providing information.

For this query, I'll provide context from relevant documents. Use this context to answer the question.
If the context doesn't contain enough information to answer the question, say so.

"""
        
        # Add conversation history if provided
        if conversation:
            prompt += "Previous conversation:\n"
            for message in conversation:
                role = message.get("role", "")
                content = message.get("content", "")
                if role and content:
                    prompt += f"{role.capitalize()}: {content}\n"
            prompt += "\n"
        
        # Add context from documents
        if context:
            prompt += "Context information:\n"
            for i, doc in enumerate(context, 1):
                content = doc.get("content", "").strip()
                source = doc.get("source", "Unknown source")
                metadata = doc.get("metadata", {})
                
                prompt += f"[Document {i}] "
                if "title" in metadata:
                    prompt += f"Title: {metadata['title']} | "
                prompt += f"Source: {source}\n"
                prompt += f"{content}\n\n"
        else:
            prompt += "No specific context found for this query.\n\n"
        
        # Add the query
        prompt += f"Question: {query}\n\n"
        prompt += "Answer: "
        
        return prompt
    
    def generate_response(
        self, 
        query: str, 
        context: List[Dict[str, Any]], 
        conversation: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        stream: bool = False
    ) -> str:
        """
        Generate a response using the Gemini model.
        
        Args:
            query: The user's query
            context: List of context documents from RAG
            conversation: Optional conversation history
            temperature: Temperature for generation (0.0 to 1.0)
            max_output_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated response
        """
        if not self.model:
            try:
                self._initialize_vertex()
            except Exception as e:
                return f"Error connecting to Gemini: {str(e)}"
        
        # Build the prompt
        prompt = self._build_prompt(query, context, conversation)
        
        # Configure generation parameters
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=max_output_tokens,
        )
        
        try:
            if stream:
                return self._generate_streaming(prompt, generation_config)
            else:
                return self._generate_non_streaming(prompt, generation_config)
        except GoogleAPICallError as e:
            logger.error(f"Gemini API error: {str(e)}")
            return f"Error generating response: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in Gemini generation: {str(e)}")
            return "Sorry, I encountered an unexpected error while generating a response."
    
    def _generate_non_streaming(self, prompt: str, generation_config: GenerationConfig) -> str:
        """Generate a non-streaming response."""
        try:
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Extract text from response
            return response.text
        except Exception as e:
            logger.error(f"Error in non-streaming generation: {str(e)}")
            raise
    
    def _generate_streaming(self, prompt: str, generation_config: GenerationConfig) -> Generator[str, None, None]:
        """Generate a streaming response."""
        try:
            # Generate streaming response
            response_stream = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True
            )
            
            # Return generator
            for chunk in response_stream:
                if hasattr(chunk, 'text'):
                    yield chunk.text
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    if hasattr(chunk.candidates[0], 'content'):
                        content = chunk.candidates[0].content
                        if hasattr(content, 'parts') and content.parts:
                            for part in content.parts:
                                if hasattr(part, 'text'):
                                    yield part.text
        except Exception as e:
            logger.error(f"Error in streaming generation: {str(e)}")
            yield f"Error: {str(e)}"




























from typing import List, Dict, Any, Optional, Tuple, Union, Generator
import time
import json

from utils.logger import setup_module_logger
from rag_engine.chunking import chunk_document
from rag_engine.embedding import embed_documents
from rag_engine.retrieval import RetrievalEngine, combine_search_results
from rag_engine.gemini_integration import GeminiLLM

logger = setup_module_logger("rag_processor")

class RAGProcessor:
    """Main processor for RAG operations."""
    
    def __init__(self):
        """Initialize the RAG processor."""
        self.engines = {}
        self.llm = GeminiLLM()
        logger.info("RAG Processor initialized")
    
    def get_retrieval_engine(self, source_id: str) -> RetrievalEngine:
        """
        Get a retrieval engine for a specific source.
        
        Args:
            source_id: ID of the data source
            
        Returns:
            RetrievalEngine instance
        """
        if source_id not in self.engines:
            self.engines[source_id] = RetrievalEngine(source_id)
        return self.engines[source_id]
    
    def process_documents(self, documents: List[Dict[str, Any]], source_id: str) -> int:
        """
        Process documents and add them to the index.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' keys
            source_id: ID of the data source
            
        Returns:
            Number of documents processed
        """
        if not documents:
            return 0
        
        # Get retrieval engine
        engine = self.get_retrieval_engine(source_id)
        
        # Chunk documents
        chunked_docs = []
        for doc in documents:
            chunked_docs.extend(chunk_document(doc))
        
        # Embed chunked documents
        embedded_docs = embed_documents(chunked_docs)
        
        # Update the retrieval engine
        engine.update_documents(embedded_docs)
        
        return len(embedded_docs)
    
    def search(
        self, 
        query: str, 
        source_ids: List[str] = None, 
        search_type: str = "hybrid", 
        top_k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for documents matching the query across specified sources.
        
        Args:
            query: The query string
            source_ids: List of source IDs to search, or None for all sources
            search_type: Type of search to perform ('semantic', 'keyword', or 'hybrid')
            top_k: Number of results to return per source
            
        Returns:
            Dictionary mapping source IDs to search results
        """
        # If no source_ids provided, use all available sources
        if not source_ids:
            source_ids = list(self.engines.keys())
        
        # If still no sources, return empty results
        if not source_ids:
            return {}
        
        # Search each source
        results_by_source = {}
        for source_id in source_ids:
            engine = self.get_retrieval_engine(source_id)
            results = engine.search(query, search_type=search_type, k=top_k)
            results_by_source[source_id] = results
        
        return results_by_source
    
    def combined_search(
        self, 
        query: str, 
        source_ids: List[str] = None, 
        search_type: str = "hybrid", 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search across multiple sources and combine results.
        
        Args:
            query: The query string
            source_ids: List of source IDs to search, or None for all sources
            search_type: Type of search to perform ('semantic', 'keyword', or 'hybrid')
            top_k: Number of results to return in total
            
        Returns:
            Combined search results from all sources
        """
        # Get results by source
        results_by_source = self.search(query, source_ids, search_type, top_k=top_k*2)
        
        # Combine all results
        all_results = []
        for source_id, results in results_by_source.items():
            all_results.extend(results)
        
        # Sort by similarity score
        if search_type == "semantic":
            all_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        elif search_type == "keyword":
            all_results.sort(key=lambda x: x.get('bm25_score', 0), reverse=True)
        else:  # hybrid
            all_results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        # Return top-k results
        return all_results[:top_k]
    
    def generate_response(
        self, 
        query: str, 
        source_ids: List[str] = None, 
        search_type: str = "hybrid", 
        top_k: int = 5,
        conversation: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response using RAG.
        
        Args:
            query: The query string
            source_ids: List of source IDs to search, or None for all sources
            search_type: Type of search to perform ('semantic', 'keyword', or 'hybrid')
            top_k: Number of context documents to use
            conversation: Optional conversation history
            temperature: Temperature for generation
            max_output_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated response or a generator for streaming
        """
        # Search for context
        context = self.combined_search(query, source_ids, search_type, top_k)
        
        # Generate response
        return self.llm.generate_response(
            query=query,
            context=context,
            conversation=conversation,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            stream=stream
        )

























from data_sources.confluence.connector import ConfluenceConnector
from data_sources.jira.connector import JIRAConnector
from data_sources.remedy.connector import RemedyConnector





















from data_sources.confluence.client import ConfluenceClient
from data_sources.confluence.parser import extract_structure_from_page
from data_sources.confluence.connector import ConfluenceConnector

















import requests
import logging
import urllib3
from typing import Dict, List, Any, Optional, Tuple, Union
from urllib.parse import quote

from config import CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN, CONFLUENCE_SSL_VERIFY
from utils.logger import setup_module_logger

logger = setup_module_logger("confluence_client")

class ConfluenceClient:
    """Client for Confluence REST API operations with comprehensive error handling."""
    
    def __init__(self, base_url=None, username=None, api_token=None, ssl_verify=None):
        """
        Initialize the Confluence client with server and authentication details.
        
        Args:
            base_url: The base URL of the Confluence instance (e.g., https://example.atlassian.net)
            username: Email address for authentication
            api_token: API token for authentication
            ssl_verify: Whether to verify SSL certificates
        """
        self.base_url = base_url or CONFLUENCE_URL
        self.base_url = self.base_url.rstrip('/')  # Remove trailing slash if present
        self.username = username or CONFLUENCE_USERNAME
        self.api_token = api_token or CONFLUENCE_API_TOKEN
        self.ssl_verify = ssl_verify if ssl_verify is not None else CONFLUENCE_SSL_VERIFY
        
        # Handle SSL verification
        if not self.ssl_verify:
            # Disable SSL warnings if verification is disabled
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL certificate verification is disabled for Confluence.")
    
    def test_connection(self) -> bool:
        """
        Test the connection to Confluence API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/content",
                auth=(self.username, self.api_token),
                verify=self.ssl_verify,
                params={"limit": 1}
            )
            
            if response.status_code == 200:
                logger.info("Connection to Confluence successful")
                return True
            else:
                logger.error(f"Failed to connect to Confluence. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Confluence: {str(e)}")
            return False
    
    def get_content_by_id(self, content_id: str, expand: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a specific content by its ID.
        
        Args:
            content_id: The ID of the content to retrieve
            expand: Optional list of properties to expand in the result
            
        Returns:
            Content data or None if not found/error
        """
        if not content_id:
            logger.error("No content ID provided")
            return None
        
        logger.info(f"Fetching content with ID: {content_id}")
        
        try:
            params = {}
            if expand:
                params["expand"] = ",".join(expand)
                
            response = requests.get(
                f"{self.base_url}/rest/api/content/{content_id}",
                auth=(self.username, self.api_token),
                verify=self.ssl_verify,
                params=params
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get content. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting content: {str(e)}")
            return None
    
    def get_page_content(self, page_id: str, expand: bool = True, storage_metadata: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get a page with its body and metadata.
        
        Args:
            page_id: The ID of the page to retrieve
            expand: Whether to expand the page content
            storage_metadata: Whether to include storage metadata
            
        Returns:
            Page data or None if not found/error
        """
        if not page_id:
            return None
        
        logger.info(f"Fetching page content for ID: {page_id}")
        
        # Define what to expand
        expand_params = []
        if expand:
            expand_params.extend(["body.storage", "metadata.labels"])
        if storage_metadata:
            expand_params.append("body.storage.metadata")
        
        return self.get_content_by_id(page_id, expand=expand_params)
    
    def search_content(self, cql: str = None, title: str = None, content_type: str = "page", limit: int = 10, expand: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for content using CQL or specific parameters.
        
        Args:
            cql: Confluence Query Language string
            title: Title to search for
            content_type: Type of content to search for (default: page)
            limit: Maximum number of results to return
            expand: Optional list of properties to expand in the result
            
        Returns:
            List of matching content items
        """
        logger.info(f"Searching for content with {'CQL' if cql else 'title'}: {cql if cql else title}")
        
        try:
            # Build query parameters
            params = {
                "limit": limit
            }
            
            # Add CQL if provided
            if cql:
                params["cql"] = cql
            
            # Build simple query if CQL not provided
            query_parts = []
            if title:
                # Remove special characters in title
                safe_title = title.replace('"', '\\"')
                query_parts.append(f'title ~ "{safe_title}"')
            
            if query_parts:
                query = " AND ".join(query_parts)
                params["cql"] = query + f" AND type = {content_type}"
            
            # Add expansion parameters
            if expand:
                params["expand"] = ",".join(expand)
            
            # Make the request
            response = requests.get(
                f"{self.base_url}/rest/api/content/search",
                auth=(self.username, self.api_token),
                params=params,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                results = response.json()
                if "results" in results:
                    logger.info(f"Found {len(results['results'])} matching content items")
                    return results["results"]
                else:
                    logger.warning("Search response doesn't contain results")
                    return []
            else:
                logger.error(f"Search failed with status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error searching content: {str(e)}")
            return []
    
    def get_all_spaces(self, limit: int = 25, start: int = 0) -> List[Dict[str, Any]]:
        """
        Get all spaces with pagination handling.
        
        Args:
            limit: Maximum number of results per request
            start: Starting index for pagination
            
        Returns:
            List of spaces
        """
        logger.info("Retrieving all spaces")
        
        all_spaces = []
        start = 0
        limit = 25  # Confluence API commonly uses 25 as default
        
        while True:
            try:
                logger.info(f"Fetching spaces...")
                response = requests.get(
                    f"{self.base_url}/rest/api/space",
                    auth=(self.username, self.api_token),
                    params={
                        "limit": limit,
                        "start": start
                    },
                    verify=self.ssl_verify
                )
                
                if response.status_code == 200:
                    spaces = response.json()
                    if not spaces.get("results"):
                        break
                    
                    all_spaces.extend(spaces.get("results", []))
                    logger.info(f"Retrieved {len(all_spaces)} spaces so far")
                    
                    # Check if there are more spaces
                    if len(spaces.get("results", [])) < limit:
                        break
                    
                    # Check for "_links" for a "next" link
                    links = spaces.get("_links", {})
                    if not links.get("next"):
                        break
                    
                    start += limit
                else:
                    logger.error(f"Get spaces failed with status code: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    break
            except Exception as e:
                logger.error(f"Error retrieving spaces: {str(e)}")
                break
        
        logger.info(f"Retrieved a total of {len(all_spaces)} spaces")
        return all_spaces
    
    def get_all_content(self, content_type: str = "page", limit: int = 25) -> List[Dict[str, Any]]:
        """
        Get all content of specified type with pagination handling.
        
        Args:
            content_type: Type of content to search for
            limit: Maximum number of results per request
            
        Returns:
            List of content items
        """
        logger.info(f"Retrieving all {content_type} content")
        
        all_content = []
        start = 0
        
        while True:
            try:
                response = requests.get(
                    f"{self.base_url}/rest/api/content",
                    auth=(self.username, self.api_token),
                    params={
                        "type": content_type,
                        "limit": limit,
                        "start": start
                    },
                    verify=self.ssl_verify
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if not data.get("results"):
                        break
                    
                    all_content.extend(data.get("results", []))
                    logger.info(f"Retrieved {len(all_content)} {content_type} items so far")
                    
                    # Check if there are more pages
                    if len(data.get("results", [])) < limit:
                        break
                    
                    start += limit
                else:
                    logger.error(f"Failed to get {content_type} content. Status code: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    break
            except Exception as e:
                logger.error(f"Error retrieving {content_type} content: {str(e)}")
                break
        
        logger.info(f"Retrieved a total of {len(all_content)} {content_type} items")
        return all_content

















from bs4 import BeautifulSoup
import re
from typing import Dict, List, Any, Optional, Tuple

from utils.logger import setup_module_logger
from utils.content_parser import (
    extract_text_from_html, 
    extract_tables_from_html,
    extract_code_blocks,
    extract_text_from_confluence_content
)

logger = setup_module_logger("confluence_parser")

def extract_structure_from_page(page_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured content from a Confluence page.
    
    Args:
        page_data: Page data from Confluence API
        
    Returns:
        Structured page content with metadata
    """
    if not page_data:
        return {}
    
    try:
        # Extract basic metadata
        result = {
            "id": page_data.get("id"),
            "type": page_data.get("type"),
            "title": page_data.get("title"),
            "version": page_data.get("version", {}).get("number"),
            "created": page_data.get("history", {}).get("createdDate"),
            "updated": page_data.get("history", {}).get("lastUpdated", {}).get("when"),
            "creator": page_data.get("history", {}).get("createdBy", {}).get("displayName"),
            "updater": page_data.get("history", {}).get("lastUpdated", {}).get("by", {}).get("displayName"),
            "space": {
                "key": page_data.get("space", {}).get("key"),
                "name": page_data.get("space", {}).get("name")
            },
            "content": "",
            "tables": [],
            "code_blocks": []
        }
        
        # Extract labels
        labels = []
        metadata = page_data.get("metadata", {})
        if "labels" in metadata and "results" in metadata["labels"]:
            for label in metadata["labels"]["results"]:
                labels.append(label.get("name"))
        result["labels"] = labels
        
        # Extract content (Confluence stores HTML in body.storage.value)
        body = page_data.get("body", {})
        storage = body.get("storage", {})
        html_content = storage.get("value", "")
        
        if html_content:
            # Extract main text content
            result["content"] = extract_text_from_confluence_content(html_content)
            
            # Extract tables
            result["tables"] = extract_tables_from_html(html_content)
            
            # Extract code blocks
            result["code_blocks"] = extract_code_blocks(html_content)
        
        return result
    except Exception as e:
        logger.error(f"Error extracting structure from page: {str(e)}")
        return {
            "id": page_data.get("id", "unknown"),
            "title": page_data.get("title", "unknown"),
            "content": "",
            "error": str(e)
        }





















from typing import List, Dict, Any, Optional

from data_sources.confluence.client import ConfluenceClient
from data_sources.confluence.parser import extract_structure_from_page
from utils.logger import setup_module_logger

logger = setup_module_logger("confluence_connector")

class ConfluenceConnector:
    """Connector for Confluence as a RAG data source."""
    
    def __init__(self, base_url=None, username=None, api_token=None, ssl_verify=None):
        """
        Initialize the Confluence connector.
        
        Args:
            base_url: Base URL of the Confluence instance
            username: Username for authentication
            api_token: API token for authentication
            ssl_verify: Whether to verify SSL certificates
        """
        self.client = ConfluenceClient(base_url, username, api_token, ssl_verify)
        self.source_id = "confluence"
    
    def test_connection(self) -> bool:
        """
        Test the connection to Confluence.
        
        Returns:
            True if connection is successful, False otherwise
        """
        return self.client.test_connection()
    
    def get_document(self, page_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single document from Confluence.
        
        Args:
            page_id: ID of the page to retrieve
            
        Returns:
            Document dictionary or None if not found/error
        """
        # Get page content from Confluence
        page_data = self.client.get_page_content(page_id)
        if not page_data:
            logger.error(f"Could not retrieve page with ID: {page_id}")
            return None
        
        # Parse page data to extract structured content
        structured_data = extract_structure_from_page(page_data)
        
        # Format as a document for RAG
        document = {
            "content": structured_data.get("content", ""),
            "metadata": {
                "id": structured_data.get("id"),
                "title": structured_data.get("title"),
                "type": "confluence_page",
                "space_key": structured_data.get("space", {}).get("key"),
                "space_name": structured_data.get("space", {}).get("name"),
                "version": structured_data.get("version"),
                "created": structured_data.get("created"),
                "updated": structured_data.get("updated"),
                "creator": structured_data.get("creator"),
                "updater": structured_data.get("updater"),
                "labels": structured_data.get("labels", []),
                "tables_count": len(structured_data.get("tables", [])),
                "code_blocks_count": len(structured_data.get("code_blocks", []))
            },
            "source": f"confluence:page:{page_id}"
        }
        
        # Add tables as separate sections if available
        if structured_data.get("tables"):
            document["tables"] = structured_data["tables"]
        
        # Add code blocks as separate sections if available
        if structured_data.get("code_blocks"):
            document["code_blocks"] = structured_data["code_blocks"]
        
        return document
    
    def search_documents(self, query: str = None, cql: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents in Confluence.
        
        Args:
            query: Text query to search for
            cql: Confluence Query Language string for advanced search
            limit: Maximum number of results to return
            
        Returns:
            List of document dictionaries
        """
        # Search for content in Confluence
        if cql:
            search_results = self.client.search_content(cql=cql, limit=limit)
        else:
            search_results = self.client.search_content(title=query, limit=limit)
        
        # Extract documents from search results
        documents = []
        for result in search_results:
            # Get the full page content
            page_id = result.get("id")
            if not page_id:
                continue
                
            document = self.get_document(page_id)
            if document:
                documents.append(document)
        
        return documents
    
    def get_all_documents(self, space_key: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all documents from Confluence, optionally filtered by space.
        
        Args:
            space_key: Optional key of the space to get documents from
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of document dictionaries
        """
        # Build CQL query
        cql = "type=page"
        if space_key:
            cql += f" AND space={space_key}"
        
        # Get all pages using the search endpoint with CQL
        search_results = self.client.search_content(cql=cql, limit=limit)
        
        # Extract documents from search results
        documents = []
        for result in search_results:
            # Get the full page content
            page_id = result.get("id")
            if not page_id:
                continue
                
            document = self.get_document(page_id)
            if document:
                documents.append(document)
                
            # Stop if we've reached the limit
            if len(documents) >= limit:
                break
        
        return documents




















from data_sources.jira.client import JIRAClient
from data_sources.jira.connector import JIRAConnector

















import requests
import urllib3
import json
from typing import List, Dict, Any, Optional
import time

from config import JIRA_URL, JIRA_USERNAME, JIRA_API_TOKEN, JIRA_SSL_VERIFY
from utils.logger import setup_module_logger

logger = setup_module_logger("jira_client")

class JIRAClient:
    """Client for JIRA REST API operations with comprehensive error handling."""
    
    def __init__(self, base_url=None, username=None, api_token=None, ssl_verify=None):
        """
        Initialize the JIRA client with server and authentication details.
        
        Args:
            base_url: The base URL of the JIRA instance (e.g., https://example.atlassian.net)
            username: Email address for authentication
            api_token: API token for authentication
            ssl_verify: Whether to verify SSL certificates
        """
        self.base_url = base_url or JIRA_URL
        self.base_url = self.base_url.rstrip('/')  # Remove trailing slash if present
        self.username = username or JIRA_USERNAME
        self.api_token = api_token or JIRA_API_TOKEN
        self.ssl_verify = ssl_verify if ssl_verify is not None else JIRA_SSL_VERIFY
        
        # Handle SSL verification
        if not self.ssl_verify:
            # Disable SSL warnings if verification is disabled
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL certificate verification is disabled for JIRA.")
    
    def test_connection(self) -> bool:
        """
        Test the connection to JIRA API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing connection to JIRA...")
            response = requests.get(
                f"{self.base_url}/rest/api/2/serverInfo",
                auth=(self.username, self.api_token),
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                server_info = response.json()
                logger.info(f"Connection to JIRA successful!")
                logger.info(f"Server version: {server_info.get('version', 'Unknown')}")
                return True
            else:
                logger.error(f"✗ Failed to connect to JIRA. Check log for details.")
                logger.error(f"Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
        except Exception as e:
            logger.error(f"✗ Error connecting to JIRA: {str(e)}")
            return False
    
    def get_issue(self, issue_key: str, fields=None, expand=None) -> Optional[Dict[str, Any]]:
        """
        Get a specific issue by its key.
        
        Args:
            issue_key: The Issue key (e.g., DEMO-1)
            fields: Comma-separated string of field names to include
            expand: Comma-separated list of sections to expand
            
        Returns:
            Issue data or None if not found/error
        """
        if not issue_key:
            logger.error("No issue key provided")
            return None
        
        logger.info(f"Fetching issue: {issue_key}")
        
        # Build query parameters
        params = {}
        if fields:
            params["fields"] = fields
        if expand:
            params["expand"] = expand
        
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/2/issue/{issue_key}",
                auth=(self.username, self.api_token),
                params=params,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully retrieved issue: {issue_key}")
                return response.json()
            else:
                logger.error(f"Failed to get issue {issue_key}. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting issue {issue_key}: {str(e)}")
            return None
    
    def search_issues(self, jql: str, max_results: int = 100, fields: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for issues using JQL.
        
        Args:
            jql: JQL query string
            max_results: Maximum number of results to return
            fields: List of field names to include
            
        Returns:
            List of matching issues
        """
        logger.info(f"Searching issues with JQL: {jql}")
        
        # Build query parameters
        params = {
            "jql": jql,
            "maxResults": max_results
        }
        
        if fields:
            params["fields"] = ",".join(fields)
        
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/2/search",
                auth=(self.username, self.api_token),
                params=params,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                data = response.json()
                issues = data.get("issues", [])
                logger.info(f"Search returned {len(issues)} issues (total: {data.get('total', 'unknown')})")
                return issues
            else:
                logger.error(f"Failed to search issues. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error searching issues: {str(e)}")
            return []
    
    def get_all_issues(self, jql: str, fields: List[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get all issues matching a JQL query, handling pagination.
        
        Args:
            jql: JQL query string
            fields: List of field names to include
            limit: Maximum total number of issues to retrieve
            
        Returns:
            List of all matching issues
        """
        logger.info(f"Retrieving all issues matching: {jql}")
        
        all_issues = []
        start_at = 0
        page_size = 100  # JIRA API limit is usually 100
        
        while True:
            try:
                # Build query parameters
                params = {
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": min(page_size, limit - len(all_issues))
                }
                
                if fields:
                    params["fields"] = ",".join(fields)
                
                response = requests.get(
                    f"{self.base_url}/rest/api/2/search",
                    auth=(self.username, self.api_token),
                    params=params,
                    verify=self.ssl_verify
                )
                
                if response.status_code == 200:
                    data = response.json()
                    issues = data.get("issues", [])
                    
                    if not issues:
                        break
                    
                    all_issues.extend(issues)
                    logger.info(f"Retrieved {len(all_issues)} issues so far")
                    
                    # Check if we've reached our limit or all available issues
                    if len(all_issues) >= limit or len(all_issues) >= data.get("total", 0):
                        break
                    
                    start_at += len(issues)
                else:
                    logger.error(f"Failed to get issues. Status code: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    break
            except Exception as e:
                logger.error(f"Error retrieving issues: {str(e)}")
                break
        
        logger.info(f"Retrieved a total of {len(all_issues)} issues")
        return all_issues
    
    def get_issue_types(self) -> List[Dict[str, Any]]:
        """
        Get all issue types defined in the JIRA instance.
        
        Returns:
            List of issue type dictionaries
        """
        try:
            logger.info("Fetching issue types...")
            response = requests.get(
                f"{self.base_url}/rest/api/2/issuetype",
                auth=(self.username, self.api_token),
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                issue_types = response.json()
                logger.info(f"Successfully retrieved {len(issue_types)} issue types")
                return issue_types
            else:
                logger.error(f"Failed to get issue types. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error retrieving issue types: {str(e)}")
            return []
    
    def get_projects(self) -> List[Dict[str, Any]]:
        """
        Get all projects visible to the authenticated user.
        
        Returns:
            List of project dictionaries
        """
        try:
            logger.info("Fetching projects...")
            response = requests.get(
                f"{self.base_url}/rest/api/2/project",
                auth=(self.username, self.api_token),
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                projects = response.json()
                logger.info(f"Successfully retrieved {len(projects)} projects")
                return projects
            else:
                logger.error(f"Failed to get projects. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error retrieving projects: {str(e)}")
            return []
    
    def create_issue(self, project_key: str, issue_type: str, summary: str, description: str, fields: Dict = None) -> Optional[Dict[str, Any]]:
        """
        Create a new issue.
        
        Args:
            project_key: The project key
            issue_type: Issue type name or ID
            summary: Issue summary
            description: Issue description
            fields: Additional fields to set
            
        Returns:
            Created issue data or None if error
        """
        logger.info(f"Creating issue in project {project_key} of type {issue_type}")
        
        # Base issue data
        issue_data = {
            "fields": {
                "project": {
                    "key": project_key
                },
                "summary": summary,
                "description": description,
                "issuetype": {
                    "name": issue_type
                }
            }
        }
        
        # Add additional fields if provided
        if fields:
            issue_data["fields"].update(fields)
        
        try:
            response = requests.post(
                f"{self.base_url}/rest/api/2/issue",
                auth=(self.username, self.api_token),
                json=issue_data,
                verify=self.ssl_verify
            )
            
            if response.status_code == 201:
                logger.info(f"Successfully created issue")
                return response.json()
            else:
                logger.error(f"Failed to create issue. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error creating issue: {str(e)}")
            return None





















from typing import List, Dict, Any, Optional
import re
from datetime import datetime
import logging

from data_sources.jira.client import JIRAClient
from utils.content_parser import clean_text, extract_text_from_html
from utils.logger import setup_module_logger

logger = setup_module_logger("jira_connector")

class JIRAConnector:
    """Connector for JIRA as a RAG data source."""
    
    def __init__(self, base_url=None, username=None, api_token=None, ssl_verify=None):
        """
        Initialize the JIRA connector.
        
        Args:
            base_url: Base URL of the JIRA instance
            username: Username for authentication
            api_token: API token for authentication
            ssl_verify: Whether to verify SSL certificates
        """
        self.client = JIRAClient(base_url, username, api_token, ssl_verify)
        self.source_id = "jira"
    
    def test_connection(self) -> bool:
        """
        Test the connection to JIRA.
        
        Returns:
            True if connection is successful, False otherwise
        """
        return self.client.test_connection()
    
    def _process_issue_content(self, issue: Dict[str, Any]) -> str:
        """
        Process raw issue data into a textual format for embedding.
        
        Args:
            issue: Raw issue data from JIRA API
            
        Returns:
            Processed content as a string
        """
        content_parts = []
        
        # Extract fields
        fields = issue.get("fields", {})
        
        # Add summary
        if fields.get("summary"):
            content_parts.append(f"Summary: {fields['summary']}")
        
        # Add description (handles Jira markup)
        if fields.get("description"):
            description_text = fields["description"]
            # Handle if description is in Atlassian Document Format (JSON)
            if isinstance(description_text, dict) and "content" in description_text:
                # Simplified extraction from ADF
                adf_content = []
                for content_block in description_text.get("content", []):
                    if content_block.get("type") == "paragraph":
                        for text_node in content_block.get("content", []):
                            if text_node.get("type") == "text":
                                adf_content.append(text_node.get("text", ""))
                description_text = "\n".join(adf_content)
            
            content_parts.append(f"Description: {clean_text(description_text)}")
        
        # Add issue type
        if fields.get("issuetype", {}).get("name"):
            content_parts.append(f"Issue Type: {fields['issuetype']['name']}")
        
        # Add status
        if fields.get("status", {}).get("name"):
            content_parts.append(f"Status: {fields['status']['name']}")
        
        # Add priority
        if fields.get("priority", {}).get("name"):
            content_parts.append(f"Priority: {fields['priority']['name']}")
        
        # Add assignee
        if fields.get("assignee", {}).get("displayName"):
            content_parts.append(f"Assignee: {fields['assignee']['displayName']}")
        
        # Add reporter
        if fields.get("reporter", {}).get("displayName"):
            content_parts.append(f"Reporter: {fields['reporter']['displayName']}")
        
        # Add project
        if fields.get("project", {}).get("name"):
            content_parts.append(f"Project: {fields['project']['name']} ({fields['project'].get('key', '')})")
        
        # Add components
        if fields.get("components"):
            components = [c.get("name", "") for c in fields["components"] if c.get("name")]
            if components:
                content_parts.append(f"Components: {', '.join(components)}")
        
        # Add labels
        if fields.get("labels"):
            content_parts.append(f"Labels: {', '.join(fields['labels'])}")
        
        # Add fixVersions
        if fields.get("fixVersions"):
            versions = [v.get("name", "") for v in fields["fixVersions"] if v.get("name")]
            if versions:
                content_parts.append(f"Fix Versions: {', '.join(versions)}")
        
        # Add comments
        if "comment" in fields and "comments" in fields["comment"]:
            comments = fields["comment"]["comments"]
            if comments:
                content_parts.append("\nComments:")
                for comment in comments:
                    author = comment.get("author", {}).get("displayName", "Unknown")
                    created = comment.get("created", "")
                    body = comment.get("body", "")
                    
                    # Handle if comment body is in Atlassian Document Format
                    if isinstance(body, dict) and "content" in body:
                        # Simplified extraction
                        adf_content = []
                        for content_block in body.get("content", []):
                            if content_block.get("type") == "paragraph":
                                for text_node in content_block.get("content", []):
                                    if text_node.get("type") == "text":
                                        adf_content.append(text_node.get("text", ""))
                        body = "\n".join(adf_content)
                    
                    comment_text = f"Comment by {author} on {created}:\n{clean_text(body)}"
                    content_parts.append(comment_text)
        
        # Combine all parts
        return "\n\n".join(content_parts)
    
    def get_document(self, issue_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a single document from JIRA.
        
        Args:
            issue_key: Key of the issue to retrieve (e.g., PROJ-123)
            
        Returns:
            Document dictionary or None if not found/error
        """
        # Define fields to retrieve
        fields = [
            "summary", "description", "issuetype", "status", "priority", 
            "assignee", "reporter", "project", "created", "updated",
            "components", "labels", "fixVersions", "comment"
        ]
        
        # Get issue from JIRA
        issue = self.client.get_issue(issue_key, fields=",".join(fields))
        if not issue:
            logger.error(f"Could not retrieve issue with key: {issue_key}")
            return None
        
        # Process issue data
        content = self._process_issue_content(issue)
        
        # Format as a document for RAG
        document = {
            "content": content,
            "metadata": {
                "id": issue.get("id"),
                "key": issue.get("key"),
                "type": "jira_issue",
                "issue_type": issue.get("fields", {}).get("issuetype", {}).get("name"),
                "status": issue.get("fields", {}).get("status", {}).get("name"),
                "priority": issue.get("fields", {}).get("priority", {}).get("name"),
                "project_key": issue.get("fields", {}).get("project", {}).get("key"),
                "project_name": issue.get("fields", {}).get("project", {}).get("name"),
                "created": issue.get("fields", {}).get("created"),
                "updated": issue.get("fields", {}).get("updated"),
                "assignee": issue.get("fields", {}).get("assignee", {}).get("displayName"),
                "reporter": issue.get("fields", {}).get("reporter", {}).get("displayName"),
                "summary": issue.get("fields", {}).get("summary"),
                "labels": issue.get("fields", {}).get("labels", []),
                "components": [c.get("name") for c in issue.get("fields", {}).get("components", [])],
                "fix_versions": [v.get("name") for v in issue.get("fields", {}).get("fixVersions", [])]
            },
            "source": f"jira:issue:{issue_key}"
        }
        
        return document
    
    def search_documents(self, jql: str = None, text_query: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search for documents in JIRA.
        
        Args:
            jql: JQL query string for searching
            text_query: Text to search for (will be converted to JQL)
            limit: Maximum number of results to return
            
        Returns:
            List of document dictionaries
        """
        # Build JQL if text query is provided
        if text_query and not jql:
            # Escape special characters
            escaped_query = text_query.replace('"', '\\"')
            jql = f'text ~ "{escaped_query}" ORDER BY updated DESC'
        
        # Use default JQL if none provided
        if not jql:
            jql = 'order by updated DESC'
        
        # Define fields to retrieve
        fields = [
            "summary", "description", "issuetype", "status", "priority", 
            "assignee", "reporter", "project", "created", "updated",
            "components", "labels", "fixVersions"
        ]
        
        # Search for issues in JIRA
        issues = self.client.search_issues(jql, max_results=limit, fields=fields)
        
        # Convert issues to documents
        documents = []
        for issue in issues:
            # Process issue data
            content = self._process_issue_content(issue)
            
            # Format as a document for RAG
            document = {
                "content": content,
                "metadata": {
                    "id": issue.get("id"),
                    "key": issue.get("key"),
                    "type": "jira_issue",
                    "issue_type": issue.get("fields", {}).get("issuetype", {}).get("name"),
                    "status": issue.get("fields", {}).get("status", {}).get("name"),
                    "priority": issue.get("fields", {}).get("priority", {}).get("name"),
                    "project_key": issue.get("fields", {}).get("project", {}).get("key"),
                    "project_name": issue.get("fields", {}).get("project", {}).get("name"),
                    "created": issue.get("fields", {}).get("created"),
                    "updated": issue.get("fields", {}).get("updated"),
                    "assignee": issue.get("fields", {}).get("assignee", {}).get("displayName"),
                    "reporter": issue.get("fields", {}).get("reporter", {}).get("displayName"),
                    "summary": issue.get("fields", {}).get("summary"),
                    "labels": issue.get("fields", {}).get("labels", []),
                    "components": [c.get("name") for c in issue.get("fields", {}).get("components", [])],
                    "fix_versions": [v.get("name") for v in issue.get("fields", {}).get("fixVersions", [])]
                },
                "source": f"jira:issue:{issue.get('key')}"
            }
            
            documents.append(document)
        
        return documents
    
    def get_all_documents(self, project_key: str = None, issue_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all documents from JIRA, optionally filtered by project and issue type.
        
        Args:
            project_key: Optional key of the project to get issues from
            issue_type: Optional issue type to filter by
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of document dictionaries
        """
        # Build JQL
        jql_parts = []
        
        if project_key:
            jql_parts.append(f'project = "{project_key}"')
        
        if issue_type:
            jql_parts.append(f'issuetype = "{issue_type}"')
        
        # Add ordering
        jql_parts.append('ORDER BY updated DESC')
        
        # Combine JQL parts
        jql = " AND ".join(jql_parts)
        
        # Use search_documents to get the results
        return self.search_documents(jql=jql, limit=limit)




















from data_sources.remedy.client import RemedyClient
from data_sources.remedy.connector import RemedyConnector

















import json
import requests
import logging
import urllib3
import getpass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from urllib.parse import quote

from config import REMEDY_URL, REMEDY_USERNAME, REMEDY_PASSWORD, REMEDY_SSL_VERIFY
from utils.logger import setup_module_logger

logger = setup_module_logger("remedy_client")

class RemedyClient:
    """
    Client for BMC Remedy REST API operations with comprehensive error handling and
    advanced querying.
    """
    
    def __init__(self, server_url=None, username=None, password=None, ssl_verify=None):
        """
        Initialize the Remedy client with server and authentication details.
        
        Args:
            server_url: The base URL of the Remedy server (e.g., https://cmegroup-restapi.onbmc.com)
            username: Username for authentication (will prompt if None)
            password: Password for authentication (will prompt if None)
            ssl_verify: Whether to verify SSL certificates (set to False to disable verification)
        """
        self.server_url = server_url or REMEDY_URL
        self.server_url = self.server_url.rstrip('/')  # Remove trailing slash if present
        self.username = username or REMEDY_USERNAME
        self.password = password or REMEDY_PASSWORD
        self.token = None
        self.token_type = "AR-JWT"
        
        # Handle SSL verification
        if ssl_verify is None:
            ssl_verify = REMEDY_SSL_VERIFY
            
        if ssl_verify is False:
            # Disable SSL warnings if verification is disabled
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL certificate verification is disabled.")
        self.ssl_verify = ssl_verify
        
        logger.info(f"Initialized Remedy client for {self.server_url}")
    
    def login(self) -> Tuple[int, str]:
        """
        Log in to Remedy and get authentication token.
        
        Returns:
            tuple: (returnVal, token) where returnVal is 1 on success, -1 on failure
        """
        if not self.username:
            self.username = input("Enter Username: ")
        if not self.password:
            self.password = getpass.getpass(prompt="Enter Password: ")
        
        logger.info(f"Attempting to login as {self.username}")
        url = f"{self.server_url}/api/jwt/login"
        payload = {"username": self.username, "password": self.password}
        headers = {"content-type": "application/x-www-form-urlencoded"}
        
        try:
            r = requests.post(url, data=payload, headers=headers, verify=self.ssl_verify)
            if r.status_code == 200:
                self.token = r.text
                logger.info("Login successful")
                return 1, self.token
            else:
                logger.error(f"Login failed with status code: {r.status_code}")
                print(f"Failure...")
                print(f"Status Code: {r.status_code}")
                return -1, r.text
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return -1, str(e)
    
    def logout(self) -> bool:
        """
        Log out from Remedy and invalidate the token.
        
        Returns:
            bool: True on success, False on failure
        """
        if not self.token:
            logger.warning("Cannot logout: No active token")
            return False
        
        logger.info("Logging out and invalidating token")
        url = f"{self.server_url}/api/jwt/logout"
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        try:
            r = requests.post(url, headers=headers, verify=self.ssl_verify)
            if r.status_code == 204 or r.status_code == 200:
                logger.info("Logout successful")
                self.token = None
                return True
            else:
                logger.error(f"Logout failed with status code: {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return False
    
    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific incident by its ID.
        
        Args:
            incident_id: The Incident Number (e.g., INC000001482087)
            
        Returns:
            dict: Incident data or None if not found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return None
        
        logger.info(f"Fetching incident: {incident_id}")
        
        # Create qualified query
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incident data
        result = self.query_form("HPD:Help Desk", qualified_query, fields)
        if result and "entries" in result and len(result["entries"]) > 0:
            logger.info(f"Successfully retrieved incident: {incident_id}")
            return result["entries"][0]
        else:
            logger.error(f"Incident not found or error: {incident_id}")
            return None
    
    def get_incidents_by_date(self, date: str, status: str = None, owner_group: str = None) -> List[Dict[str, Any]]:
        """
        Get all incidents submitted on a specific date.
        
        Args:
            date: The submission date in YYYY-MM-DD format
            status: Optional status filter (e.g., "Closed", "Open")
            owner_group: Optional owner group filter
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return []
        
        logger.info(f"Fetching incidents for date: {date}")
        
        # Parse the date and create date range (entire day)
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            start_datetime = date_obj.strftime("%Y-%m-%d 00:00:00.000")
            end_datetime = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00.000")
            
            # Create qualified query
            query_parts = [f"'Submit Date' >= \"{start_datetime}\" AND 'Submit Date' < \"{end_datetime}\""]
            
            # Add status filter if provided
            if status:
                query_parts.append(f"'Status'=\"{status}\"")
                
            # Add owner group filter if provided
            if owner_group:
                query_parts.append(f"'Owner Group'=\"{owner_group}\"")
                
            qualified_query = " AND ".join(query_parts)
            
            # Fields to retrieve
            fields = [
                "Assignee", "Incident Number", "Description", "Status", "Owner",
                "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
                "Priority", "Environment", "Summary", "Support Group Name",
                "Request Assignee", "Work Order ID", "Request Manager"
            ]
            
            # Get the incidents
            result = self.query_form("HPD:Help Desk", qualified_query, fields)
            if result and "entries" in result:
                logger.info(f"Retrieved {len(result['entries'])} incidents for date {date}")
                return result["entries"]
            else:
                logger.warning(f"No incidents found for date {date} or error occurred")
                return []
        except ValueError:
            logger.error(f"Invalid date format: {date}. Use YYYY-MM-DD.")
            return []
    
    def get_incidents_by_status(self, status: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get incidents by their status.
        
        Args:
            status: The status to filter by (e.g., "Open", "Closed", "Resolved")
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return []
        
        logger.info(f"Fetching incidents with status: {status}")
        
        # Create qualified query
        qualified_query = f"'Status'=\"{status}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents with status {status}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found with status {status} or error occurred")
            return []
    
    def get_incidents_by_assignee(self, assignee: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get incidents assigned to a specific person.
        
        Args:
            assignee: The assignee name
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return []
        
        logger.info(f"Fetching incidents assigned to: {assignee}")
        
        # Create qualified query
        qualified_query = f"'Assignee'=\"{assignee}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents assigned to {assignee}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found assigned to {assignee} or error occurred")
            return []
    
    def query_form(self, form_name: str, qualified_query: str = None, fields: List[str] = None, limit: int = 100) -> Optional[Dict[str, Any]]:
        """
        Query a Remedy form with optional filters and field selection.
        
        Args:
            form_name: The name of the form to query (e.g., "HPD:Help Desk")
            qualified_query: Optional qualified query string for filtering
            fields: Optional list of fields to retrieve
            limit: Maximum number of records to retrieve
            
        Returns:
            dict: Query result or None if error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return None
        
        logger.info(f"Querying form: {form_name}")
        
        # Build URL
        url = f"{self.server_url}/api/arsys/v1/entry/{form_name}"
        
        # Build headers
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        # Build query parameters
        params = {}
        if qualified_query:
            params["q"] = qualified_query
        if fields:
            params["fields"] = ",".join(fields)
        if limit:
            params["limit"] = limit
        
        # Make the request
        try:
            r = requests.get(url, headers=headers, params=params, verify=self.ssl_verify)
            if r.status_code == 200:
                result = r.json()
                logger.info(f"Successfully queried form {form_name} and got {len(result.get('entries', []))} results")
                return result
            else:
                logger.error(f"Query failed with status code: {r.status_code}")
                logger.error(f"Headers: {r.headers}")
                logger.error(f"Response: {r.text}")
                return None
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return None
    
    def create_incident(self, summary: str, description: str, impact: str = "4-Minor/Localized", urgency: str = "4-Low",
                       reported_source: str = "Direct Input", service_type: str = "User Service Restoration",
                       assigned_group: str = None) -> Optional[Dict[str, Any]]:
        """
        Create a new incident in Remedy.
        
        Args:
            summary: Incident summary/title
            description: Detailed description
            impact: Impact level (1-5)
            urgency: Urgency level (1-4)
            reported_source: How the incident was reported
            service_type: Type of service
            assigned_group: Group to assign the incident to
            
        Returns:
            dict: Created incident data or None if error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return None
        
        logger.info(f"Creating new incident: {summary}")
        
        # Build URL
        url = f"{self.server_url}/api/arsys/v1/entry/HPD:Help Desk"
        
        # Build headers
        headers = {
            "Authorization": f"{self.token_type} {self.token}",
            "Content-Type": "application/json"
        }
        
        # Build incident data
        incident_data = {
            "values": {
                "Summary": summary,
                "Description": description,
                "Impact": impact,
                "Urgency": urgency,
                "Reported Source": reported_source,
                "Service Type": service_type
            }
        }
        
        if assigned_group:
            incident_data["values"]["Assigned Group"] = assigned_group
        
        # Make the request
        try:
            r = requests.post(url, headers=headers, json=incident_data, verify=self.ssl_verify)
            if r.status_code == 201:
                result = r.json()
                logger.info(f"Successfully created incident: {result.get('values', {}).get('Incident Number')}")
                return result
            else:
                logger.error(f"Create incident failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return None
        except Exception as e:
            logger.error(f"Create incident error: {str(e)}")
            return None
    
    def update_incident(self, incident_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing incident.
        
        Args:
            incident_id: The Incident Number to update
            update_data: Dictionary of fields to update
            
        Returns:
            bool: True on success, False on failure
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return False
        
        logger.info(f"Updating incident: {incident_id}")
        
        # Build URL
        url = f"{self.server_url}/api/arsys/v1/entry/HPD:Help Desk"
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        url = f"{url}?q={quote(qualified_query)}"
        
        # Build headers
        headers = {
            "Authorization": f"{self.token_type} {self.token}",
            "Content-Type": "application/json"
        }
        
        # Build update data
        payload = {
            "values": update_data
        }
        
        # Make the request
        try:
            r = requests.put(url, headers=headers, json=payload, verify=self.ssl_verify)
            if r.status_code == 204:
                logger.info(f"Successfully updated incident: {incident_id}")
                return True
            else:
                logger.error(f"Update incident failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return False
        except Exception as e:
            logger.error(f"Update incident error: {str(e)}")
            return False
    
    def get_incident_history(self, incident_id: str) -> List[Dict[str, Any]]:
        """
        Get the history of changes for a specific incident.
        
        Args:
            incident_id: The Incident Number
            
        Returns:
            list: History entries or empty list if none found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return []
        
        logger.info(f"Fetching history for incident: {incident_id}")
        
        # Build URL for history form
        url = f"{self.server_url}/api/arsys/v1/entry/HPD:Help Desk History"
        
        # Qualified query to filter by incident number
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        
        # Headers
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        # Query parameters
        params = {
            "q": qualified_query,
            "fields": "History Date Time,Action,Description,Status,Changed By,Assigned Group"
        }
        
        # Make the request
        try:
            r = requests.get(url, headers=headers, params=params, verify=self.ssl_verify)
            if r.status_code == 200:
                result = r.json()
                logger.info(f"Successfully retrieved history for incident {incident_id} with {len(result.get('entries', []))} entries")
                return result.get("entries", [])
            else:
                logger.error(f"Get history failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return []
        except Exception as e:
            logger.error(f"Get history error: {str(e)}")
            return []
    
    def process_incident_for_rag(self, incident: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process an incident into a format suitable for RAG indexing.
        
        Args:
            incident: Raw incident data from Remedy API
            
        Returns:
            dict: Processed incident with metadata and content
        """
        if not incident or "values" not in incident:
            return None
        
        values = incident.get("values", {})
        
        # Extract metadata
        metadata = {
            "incident_number": values.get("Incident Number"),
            "status": values.get("Status"),
            "priority": values.get("Priority"),
            "impact": values.get("Impact"),
            "assignee": values.get("Assignee"),
            "owner": values.get("Owner"),
            "owner_group": values.get("Owner Group"),
            "assigned_group": values.get("Assigned Group"),
            "submitter": values.get("Submitter"),
            "submit_date": values.get("Submit Date"),
            "summary": values.get("Summary")
        }
        
        # Build content for embedding
        content_parts = []
        if values.get("Summary"):
            content_parts.append(f"Summary: {values.get('Summary')}")
        if values.get("Description"):
            content_parts.append(f"Description: {values.get('Description')}")
        if values.get("Status"):
            content_parts.append(f"Status: {values.get('Status')}")
        if values.get("Priority"):
            content_parts.append(f"Priority: {values.get('Priority')}")
        if values.get("Impact"):
            content_parts.append(f"Impact: {values.get('Impact')}")
        if values.get("Assignee"):
            content_parts.append(f"Assigned to: {values.get('Assignee')}")
        if values.get("Owner Group"):
            content_parts.append(f"Owner Group: {values.get('Owner Group')}")
        
        # Combine content parts into a single text
        content = "\n".join(content_parts)
        
        return {
            "metadata": metadata,
            "content": content,
            "raw_data": values
        }






















from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from data_sources.remedy.client import RemedyClient
from utils.logger import setup_module_logger

logger = setup_module_logger("remedy_connector")

class RemedyConnector:
    """Connector for Remedy as a RAG data source."""
    
    def __init__(self, server_url=None, username=None, password=None, ssl_verify=None):
        """
        Initialize the Remedy connector.
        
        Args:
            server_url: Base URL of the Remedy server
            username: Username for authentication
            password: Password for authentication
            ssl_verify: Whether to verify SSL certificates
        """
        self.client = RemedyClient(server_url, username, password, ssl_verify)
        self.source_id = "remedy"
        self.is_logged_in = False
    
    def login(self) -> bool:
        """
        Log in to Remedy.
        
        Returns:
            True if login is successful, False otherwise
        """
        status, _ = self.client.login()
        self.is_logged_in = status == 1
        return self.is_logged_in
    
    def logout(self) -> bool:
        """
        Log out from Remedy.
        
        Returns:
            True if logout is successful, False otherwise
        """
        if self.is_logged_in:
            success = self.client.logout()
            if success:
                self.is_logged_in = False
            return success
        return True  # Already logged out
    
    def ensure_login(self) -> bool:
        """
        Ensure the client is logged in.
        
        Returns:
            True if logged in, False otherwise
        """
        if not self.is_logged_in:
            return self.login()
        return True
    
    def get_document(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single document from Remedy.
        
        Args:
            incident_id: ID of the incident to retrieve (e.g., INC000001234567)
            
        Returns:
            Document dictionary or None if not found/error
        """
        # Ensure logged in
        if not self.ensure_login():
            logger.error("Could not log in to Remedy")
            return None
        
        # Get incident from Remedy
        incident = self.client.get_incident(incident_id)
        if not incident:
            logger.error(f"Could not retrieve incident: {incident_id}")
            return None
        
        # Format as a document for RAG
        document = self._format_incident_as_document(incident)
        
        return document
    
    def _format_incident_as_document(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a Remedy incident as a document for RAG.
        
        Args:
            incident: Incident data from Remedy API
            
        Returns:
            Document dictionary
        """
        values = incident.get("values", {})
        
        # Build content
        content_parts = []
        
        # Add summary
        if values.get("Summary"):
            content_parts.append(f"Summary: {values.get('Summary')}")
        
        # Add description
        if values.get("Description"):
            content_parts.append(f"Description: {values.get('Description')}")
        
        # Add status
        if values.get("Status"):
            content_parts.append(f"Status: {values.get('Status')}")
        
        # Add priority
        if values.get("Priority"):
            content_parts.append(f"Priority: {values.get('Priority')}")
        
        # Add impact
        if values.get("Impact"):
            content_parts.append(f"Impact: {values.get('Impact')}")
        
        # Add submitter
        if values.get("Submitter"):
            content_parts.append(f"Submitted by: {values.get('Submitter')}")
        
        # Add submit date
        if values.get("Submit Date"):
            content_parts.append(f"Submitted on: {values.get('Submit Date')}")
        
        # Add assignee
        if values.get("Assignee"):
            content_parts.append(f"Assigned to: {values.get('Assignee')}")
        
        # Add owner group
        if values.get("Owner Group"):
            content_parts.append(f"Owner Group: {values.get('Owner Group')}")
        
        # Add assigned group
        if values.get("Assigned Group"):
            content_parts.append(f"Assigned Group: {values.get('Assigned Group')}")
        
        # Combine content parts
        content = "\n\n".join(content_parts)
        
        # Create document
        document = {
            "content": content,
            "metadata": {
                "id": values.get("Incident Number"),
                "title": values.get("Summary"),
                "type": "remedy_incident",
                "status": values.get("Status"),
                "priority": values.get("Priority"),
                "impact": values.get("Impact"),
                "assignee": values.get("Assignee"),
                "submitter": values.get("Submitter"),
                "submit_date": values.get("Submit Date"),
                "owner_group": values.get("Owner Group"),
                "assigned_group": values.get("Assigned Group")
            },
            "source": f"remedy:incident:{values.get('Incident Number')}"
        }
        
        return document
    
    def search_documents(self, query: str = None, status: str = None, assignee: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents in Remedy.
        
        Args:
            query: Text query to search for in summary (optional)
            status: Status to filter by (optional)
            assignee: Assignee to filter by (optional)
            limit: Maximum number of results to return
            
        Returns:
            List of document dictionaries
        """
        # Ensure logged in
        if not self.ensure_login():
            logger.error("Could not log in to Remedy")
            return []
        
        # Build query
        query_parts = []
        
        if query:
            # Escape quotes
            query = query.replace('"', '\\"')
            query_parts.append(f"'Summary' LIKE \"%{query}%\"")
        
        if status:
            query_parts.append(f"'Status'=\"{status}\"")
        
        if assignee:
            query_parts.append(f"'Assignee'=\"{assignee}\"")
        
        # Combine query parts
        qualified_query = " AND ".join(query_parts) if query_parts else None
        
        # Define fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Query Remedy
        result = self.client.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        
        # Format results as documents
        documents = []
        if result and "entries" in result:
            for incident in result["entries"]:
                document = self._format_incident_as_document(incident)
                documents.append(document)
        
        return documents
    
    def get_documents_by_date(self, date: str, status: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get documents from Remedy for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
            status: Status to filter by (optional)
            limit: Maximum number of results to return
            
        Returns:
            List of document dictionaries
        """
        # Ensure logged in
        if not self.ensure_login():
            logger.error("Could not log in to Remedy")
            return []
        
        # Get incidents by date
        incidents = self.client.get_incidents_by_date(date, status)
        
        # Format results as documents
        documents = []
        for incident in incidents[:limit]:
            document = self._format_incident_as_document(incident)
            documents.append(document)
        
        return documents
    
    def get_documents_by_status(self, status: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get documents from Remedy with a specific status.
        
        Args:
            status: Status to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of document dictionaries
        """
        # Ensure logged in
        if not self.ensure_login():
            logger.error("Could not log in to Remedy")
            return []
        
        # Get incidents by status
        incidents = self.client.get_incidents_by_status(status, limit=limit)
        
        # Format results as documents
        documents = []
        for incident in incidents:
            document = self._format_incident_as_document(incident)
            documents.append(document)
        
        return documents
    
    def get_documents_by_assignee(self, assignee: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get documents from Remedy assigned to a specific person.
        
        Args:
            assignee: Assignee to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of document dictionaries
        """
        # Ensure logged in
        if not self.ensure_login():
            logger.error("Could not log in to Remedy")
            return []
        
        # Get incidents by assignee
        incidents = self.client.get_incidents_by_assignee(assignee, limit=limit)
        
        # Format results as documents
        documents = []
        for incident in incidents:
            document = self._format_incident_as_document(incident)
            documents.append(document)
        
        return documents


















from api.routes import register_routes
from api.response_formatter import format_response, stream_response























from flask import Blueprint, request, jsonify, Response, stream_with_context
import json
from typing import Dict, List, Any, Optional

from rag_engine.processor import RAGProcessor
from data_sources.confluence.connector import ConfluenceConnector
from data_sources.jira.connector import JIRAConnector
from data_sources.remedy.connector import RemedyConnector
from utils.logger import setup_module_logger
from api.response_formatter import format_response, stream_response

logger = setup_module_logger("api_routes")

# Initialize connectors and processor
rag_processor = RAGProcessor()
confluence_connector = ConfluenceConnector()
jira_connector = JIRAConnector()
remedy_connector = RemedyConnector()

# Create blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

@api_bp.route('/api/sources', methods=['GET'])
def get_sources():
    """Get available data sources."""
    sources = [
        {"id": "confluence", "name": "Confluence", "type": "wiki"},
        {"id": "jira", "name": "JIRA", "type": "issue_tracker"},
        {"id": "remedy", "name": "Remedy", "type": "ticket_system"}
    ]
    return jsonify({"sources": sources})

@api_bp.route('/api/confluence/test', methods=['GET'])
def test_confluence():
    """Test Confluence connection."""
    success = confluence_connector.test_connection()
    return jsonify({"success": success})

@api_bp.route('/api/jira/test', methods=['GET'])
def test_jira():
    """Test JIRA connection."""
    success = jira_connector.test_connection()
    return jsonify({"success": success})

@api_bp.route('/api/remedy/test', methods=['GET'])
def test_remedy():
    """Test Remedy connection."""
    remedy = RemedyConnector()  # Create a new instance for the test
    success = remedy.login()
    if success:
        remedy.logout()
    return jsonify({"success": success})

@api_bp.route('/api/index/confluence', methods=['POST'])
def index_confluence():
    """Index Confluence content."""
    data = request.json or {}
    space_key = data.get('space_key')
    limit = data.get('limit', 100)
    
    # Get documents from Confluence
    documents = confluence_connector.get_all_documents(space_key, limit=limit)
    
    # Process documents
    count = rag_processor.process_documents(documents, "confluence")
    
    return jsonify({
        "success": True,
        "indexed_count": count,
        "source": "confluence"
    })

@api_bp.route('/api/index/jira', methods=['POST'])
def index_jira():
    """Index JIRA content."""
    data = request.json or {}
    project_key = data.get('project_key')
    issue_type = data.get('issue_type')
    limit = data.get('limit', 100)
    
    # Get documents from JIRA
    documents = jira_connector.get_all_documents(project_key, issue_type, limit=limit)
    
    # Process documents
    count = rag_processor.process_documents(documents, "jira")
    
    return jsonify({
        "success": True,
        "indexed_count": count,
        "source": "jira"
    })

@api_bp.route('/api/index/remedy', methods=['POST'])
def index_remedy():
    """Index Remedy content."""
    data = request.json or {}
    status = data.get('status')
    limit = data.get('limit', 100)
    
    # Login to Remedy
    if not remedy_connector.ensure_login():
        return jsonify({
            "success": False,
            "error": "Failed to log in to Remedy"
        }), 500
    
    # Get documents from Remedy
    documents = remedy_connector.get_documents_by_status(status, limit=limit) if status else []
    
    # Process documents
    count = rag_processor.process_documents(documents, "remedy")
    
    # Logout from Remedy
    remedy_connector.logout()
    
    return jsonify({
        "success": True,
        "indexed_count": count,
        "source": "remedy"
    })

@api_bp.route('/api/search', methods=['POST'])
def search():
    """Search for documents across sources."""
    data = request.json or {}
    query = data.get('query', '')
    sources = data.get('sources', [])
    search_type = data.get('search_type', 'hybrid')
    top_k = data.get('top_k', 5)
    
    if not query:
        return jsonify({
            "success": False,
            "error": "Query is required"
        }), 400
    
    # If no sources specified, use all
    if not sources:
        sources = ["confluence", "jira", "remedy"]
    
    # Perform search
    results = rag_processor.search(query, sources, search_type, top_k)
    
    # Format results for API response
    formatted_results = format_response(results)
    
    return jsonify({
        "success": True,
        "query": query,
        "results": formatted_results
    })

@api_bp.route('/api/chat', methods=['POST'])
def chat():
    """Chat with RAG-augmented LLM."""
    data = request.json or {}
    query = data.get('query', '')
    sources = data.get('sources', [])
    conversation = data.get('conversation', [])
    stream = data.get('stream', False)
    temperature = data.get('temperature', 0.7)
    top_k = data.get('top_k', 5)
    
    if not query:
        return jsonify({
            "success": False,
            "error": "Query is required"
        }), 400
    
    # If stream is requested, use streaming response
    if stream:
        def generate():
            # Generate streaming response
            for chunk in rag_processor.generate_response(
                query=query,
                source_ids=sources,
                conversation=conversation,
                temperature=temperature,
                top_k=top_k,
                stream=True
            ):
                yield stream_response(chunk)
        
        return Response(
            stream_with_context(generate()),
            content_type='text/event-stream'
        )
    
    # Otherwise, generate full response
    response = rag_processor.generate_response(
        query=query,
        source_ids=sources,
        conversation=conversation,
        temperature=temperature,
        top_k=top_k,
        stream=False
    )
    
    return jsonify({
        "success": True,
        "query": query,
        "response": response
    })

def register_routes(app):
    """Register API routes with the Flask app."""
    app.register_blueprint(api_bp)




















import json
from typing import Dict, List, Any, Union

def format_response(results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Format search results for API response.
    
    Args:
        results: Search results by source ID
        
    Returns:
        Formatted results list
    """
    formatted_results = []
    
    for source_id, source_results in results.items():
        for result in source_results:
            # Extract metadata
            metadata = result.get("metadata", {})
            
            # Format source
            source = result.get("source", "")
            source_parts = source.split(":", 2) if source else []
            source_type = source_parts[0] if len(source_parts) > 0 else ""
            source_subtype = source_parts[1] if len(source_parts) > 1 else ""
            source_id = source_parts[2] if len(source_parts) > 2 else ""
            
            # Create formatted result
            formatted_result = {
                "content": result.get("content", ""),
                "source": {
                    "type": source_type,
                    "subtype": source_subtype,
                    "id": source_id
                },
                "metadata": metadata
            }
            
            # Add scores if available
            if "similarity" in result:
                formatted_result["semantic_score"] = result["similarity"]
            if "bm25_score" in result:
                formatted_result["keyword_score"] = result["bm25_score"]
            if "combined_score" in result:
                formatted_result["score"] = result["combined_score"]
            
            formatted_results.append(formatted_result)
    
    # Sort by score if available
    if formatted_results and "score" in formatted_results[0]:
        formatted_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return formatted_results

def stream_response(chunk: str) -> str:
    """
    Format a response chunk for server-sent events.
    
    Args:
        chunk: Text chunk from streaming response
        
    Returns:
        Formatted SSE event
    """
    # Use JSON to escape special characters
    data = json.dumps({"text": chunk})
    return f"data: {data}\n\n"




















from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from datetime import datetime
import json

from config import PORT, DEBUG, SECRET_KEY, FLASK_ENV
from utils.logger import logger
from api import register_routes

def create_app():
    """Create and configure the Flask application."""
    # Create Flask app
    app = Flask(__name__, 
                static_folder='static',
                template_folder='templates')
    
    # Configure app
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['DEBUG'] = DEBUG
    app.config['ENV'] = FLASK_ENV
    
    # Register API routes
    register_routes(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register web routes
    register_web_routes(app)
    
    logger.info(f"Application initialized in {FLASK_ENV} mode")
    
    return app

def register_error_handlers(app):
    """Register error handlers for the app."""
    @app.errorhandler(404)
    def not_found_error(error):
        if request.path.startswith('/api/'):
            return jsonify({"error": "Not found"}), 404
        return render_template('error.html', error="Page not found"), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {str(error)}")
        if request.path.startswith('/api/'):
            return jsonify({"error": "Internal server error"}), 500
        return render_template('error.html', error="Internal server error"), 500

def register_web_routes(app):
    """Register web routes for the app."""
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/chat')
    def chat():
        return render_template('chat.html')
    
    @app.route('/admin')
    def admin():
        return render_template('admin.html')
    
    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory(os.path.join(app.root_path, 'static', 'images'),
                                   'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Create app instance
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)

















from app import app
from config import PORT

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)















<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Enterprise RAG System{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">
    {% block extra_css %}{% endblock %}
    <link rel="icon" href="{{ url_for('static', filename='images/favicon.ico') }}">
</head>
<body>
    <header>
        <nav class="navbar">
            <div class="container">
                <div class="navbar-brand">
                    <a href="/" class="logo">
                        <span class="logo-text">EnterpriseRAG</span>
                    </a>
                </div>
                <div class="navbar-menu">
                    <ul class="navbar-nav">
                        <li class="nav-item"><a href="/" class="nav-link">Home</a></li>
                        <li class="nav-item"><a href="/chat" class="nav-link">Chat</a></li>
                        <li class="nav-item"><a href="/admin" class="nav-link">Admin</a></li>
                    </ul>
                </div>
            </div>
        </nav>
    </header>

    <main>
        <div class="container">
            {% block content %}{% endblock %}
        </div>
    </main>

    <footer>
        <div class="container">
            <p>&copy; {{ now.year }} Enterprise RAG System. All rights reserved.</p>
        </div>
    </footer>

    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>


















{% extends "base.html" %}

{% block title %}Enterprise RAG System - Home{% endblock %}

{% block content %}
<div class="hero">
    <div class="hero-content">
        <h1 class="hero-title">Enterprise RAG System</h1>
        <p class="hero-description">A Retrieval-Augmented Generation system that integrates with Confluence, JIRA, and Remedy for intelligent responses.</p>
        <div class="hero-buttons">
            <a href="/chat" class="button primary">Start Chat</a>
            <a href="/admin" class="button secondary">Admin Panel</a>
        </div>
    </div>
</div>

<div class="features">
    <div class="feature-container">
        <div class="feature">
            <div class="feature-icon confluence-icon"></div>
            <h2>Confluence Integration</h2>
            <p>Access knowledge from your Confluence pages for comprehensive answers to your questions.</p>
        </div>
        
        <div class="feature">
            <div class="feature-icon jira-icon"></div>
            <h2>JIRA Integration</h2>
            <p>Get insights from JIRA issues and track project progress through natural language queries.</p>
        </div>
        
        <div class="feature">
            <div class="feature-icon remedy-icon"></div>
            <h2>Remedy Integration</h2>
            <p>Retrieve information from Remedy tickets and get assistance with incident management.</p>
        </div>
    </div>
</div>

<div class="section">
    <h2 class="section-title">How It Works</h2>
    <div class="workflow">
        <div class="workflow-step">
            <div class="step-number">1</div>
            <h3>Ask a Question</h3>
            <p>Enter your question in natural language through the chat interface.</p>
        </div>
        <div class="workflow-step">
            <div class="step-number">2</div>
            <h3>Retrieve Context</h3>
            <p>The system searches for relevant information across Confluence, JIRA, and Remedy.</p>
        </div>
        <div class="workflow-step">
            <div class="step-number">3</div>
            <h3>Generate Response</h3>
            <p>Using Google's Gemini API, the system generates accurate responses based on retrieved context.</p>
        </div>
    </div>
</div>
{% endblock %}
















{% extends "base.html" %}

{% block title %}Enterprise RAG System - Chat{% endblock %}

{% block extra_css %}
<link rel="stylesheet" href="{{ url_for('static', filename='css/chat.css') }}">
{% endblock %}

{% block content %}
<div class="chat-container">
    <div class="chat-sidebar">
        <div class="data-sources">
            <h3>Data Sources</h3>
            <div class="source-options">
                <label class="source-option">
                    <input type="checkbox" name="source" value="confluence" checked>
                    <span class="source-name">Confluence</span>
                </label>
                <label class="source-option">
                    <input type="checkbox" name="source" value="jira" checked>
                    <span class="source-name">JIRA</span>
                </label>
                <label class="source-option">
                    <input type="checkbox" name="source" value="remedy" checked>
                    <span class="source-name">Remedy</span>
                </label>
            </div>
        </div>
        
        <div class="chat-settings">
            <h3>Settings</h3>
            <div class="setting">
                <label for="temperature">Temperature</label>
                <input type="range" id="temperature" name="temperature" min="0" max="1" step="0.1" value="0.7">
                <span class="setting-value" id="temperature-value">0.7</span>
            </div>
            <div class="setting">
                <label for="top-k">Top K Results</label>
                <input type="range" id="top-k" name="top-k" min="1" max="10" step="1" value="5">
                <span class="setting-value" id="top-k-value">5</span>
            </div>
            <div class="setting">
                <label class="switch">
                    <input type="checkbox" id="stream" name="stream" checked>
                    <span class="slider round"></span>
                    <span class="switch-label">Stream Response</span>
                </label>
            </div>
        </div>
        
        <div class="chat-info">
            <h3>About</h3>
            <p>This chat interface uses Retrieval-Augmented Generation (RAG) to provide accurate answers from your enterprise data sources.</p>
            <p>Responses are generated using Google's Gemini API and enhanced with contextual information from Confluence, JIRA, and Remedy.</p>
        </div>
    </div>
    
    <div class="chat-main">
        <div class="chat-header">
            <h2>Enterprise Chat Assistant</h2>
        </div>
        
        <div class="chat-messages" id="chat-messages">
            <div class="message system">
                <div class="message-content">
                    <p>Hello! I'm your Enterprise Assistant. I can help you find information from Confluence, JIRA, and Remedy. What would you like to know?</p>
                </div>
            </div>
        </div>
        
        <div class="chat-input">
            <form id="chat-form">
                <input type="text" id="user-input" placeholder="Type your question here..." autocomplete="off">
                <button type="submit" id="send-button">
                    <span class="send-icon"></span>
                </button>
            </form>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script src="{{ url_for('static', filename='js/chat.js') }}"></script>
{% endblock %}


















{% extends "base.html" %}

{% block title %}Enterprise RAG System - Admin{% endblock %}

{% block content %}
<div class="admin-container">
    <h1 class="page-title">Admin Dashboard</h1>
    
    <div class="admin-cards">
        <div class="admin-card">
            <div class="card-header">
                <h2>Confluence Indexing</h2>
                <span class="status-indicator" id="confluence-status">Unknown</span>
            </div>
            <div class="card-body">
                <div class="form-group">
                    <label for="confluence-space">Space Key (optional)</label>
                    <input type="text" id="confluence-space" placeholder="Leave empty for all spaces">
                </div>
                <div class="form-group">
                    <label for="confluence-limit">Document Limit</label>
                    <input type="number" id="confluence-limit" value="100" min="1" max="500">
                </div>
                <div class="card-actions">
                    <button id="test-confluence" class="button secondary">Test Connection</button>
                    <button id="index-confluence" class="button primary">Index Content</button>
                </div>
            </div>
        </div>
        
        <div class="admin-card">
            <div class="card-header">
                <h2>JIRA Indexing</h2>
                <span class="status-indicator" id="jira-status">Unknown</span>
            </div>
            <div class="card-body">
                <div class="form-group">
                    <label for="jira-project">Project Key (optional)</label>
                    <input type="text" id="jira-project" placeholder="Leave empty for all projects">
                </div>
                <div class="form-group">
                    <label for="jira-type">Issue Type (optional)</label>
                    <input type="text" id="jira-type" placeholder="Bug, Story, Task, etc.">
                </div>
                <div class="form-group">
                    <label for="jira-limit">Document Limit</label>
                    <input type="number" id="jira-limit" value="100" min="1" max="500">
                </div>
                <div class="card-actions">
                    <button id="test-jira" class="button secondary">Test Connection</button>
                    <button id="index-jira" class="button primary">Index Content</button>
                </div>
            </div>
        </div>
        
        <div class="admin-card">
            <div class="card-header">
                <h2>Remedy Indexing</h2>
                <span class="status-indicator" id="remedy-status">Unknown</span>
            </div>
            <div class="card-body">
                <div class="form-group">
                    <label for="remedy-status-filter">Status (optional)</label>
                    <select id="remedy-status-filter">
                        <option value="">All Statuses</option>
                        <option value="Open">Open</option>
                        <option value="Assigned">Assigned</option>
                        <option value="In Progress">In Progress</option>
                        <option value="Resolved">Resolved</option>
                        <option value="Closed">Closed</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="remedy-limit">Document Limit</label>
                    <input type="number" id="remedy-limit" value="100" min="1" max="500">
                </div>
                <div class="card-actions">
                    <button id="test-remedy" class="button secondary">Test Connection</button>
                    <button id="index-remedy" class="button primary">Index Content</button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="indexing-status">
        <h2>Indexing Status</h2>
        <div class="status-table-container">
            <table class="status-table">
                <thead>
                    <tr>
                        <th>Source</th>
                        <th>Status</th>
                        <th>Documents</th>
                        <th>Last Updated</th>
                    </tr>
                </thead>
                <tbody id="indexing-status-table">
                    <tr>
                        <td>Confluence</td>
                        <td>Not Indexed</td>
                        <td>0</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>JIRA</td>
                        <td>Not Indexed</td>
                        <td>0</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>Remedy</td>
                        <td>Not Indexed</td>
                        <td>0</td>
                        <td>-</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script src="{{ url_for('static', filename='js/admin.js') }}"></script>
{% endblock %}


















{% extends "base.html" %}

{% block title %}Enterprise RAG System - Error{% endblock %}

{% block content %}
<div class="error-container">
    <div class="error-content">
        <h1 class="error-title">Oops! Something went wrong.</h1>
        <p class="error-message">{{ error }}</p>
        <a href="/" class="button primary">Go Back Home</a>
    </div>
</div>
{% endblock %}

















/* Main Stylesheet */

/* Variables */
:root {
    --primary-color: #4f46e5;
    --primary-hover: #4338ca;
    --secondary-color: #64748b;
    --secondary-hover: #475569;
    --light-bg: #f1f5f9;
    --dark-bg: #1e293b;
    --text-color: #334155;
    --light-text: #f8fafc;
    --border-color: #e2e8f0;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    --border-radius: 6px;
    --box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --transition: all 0.3s ease;
    
    /* Brand colors */
    --confluence-color: #0052cc;
    --jira-color: #2684ff;
    --remedy-color: #6554c0;
}

/* Reset */
*, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html {
    font-size: 16px;
    scroll-behavior: smooth;
}

body {
    font-family: var(--font-family);
    color: var(--text-color);
    line-height: 1.6;
    background-color: var(--light-bg);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Container */
.container {
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* Navigation */
.navbar {
    background-color: white;
    box-shadow: var(--box-shadow);
    padding: 1rem 0;
    position: sticky;
    top: 0;
    z-index: 100;
}

.navbar-brand {
    display: flex;
    align-items: center;
}

.logo {
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.logo-text {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--primary-color), var(--jira-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.navbar .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.navbar-nav {
    display: flex;
    list-style: none;
    gap: 2rem;
}

.nav-link {
    text-decoration: none;
    color: var(--text-color);
    font-weight: 500;
    transition: var(--transition);
    padding: 0.5rem 0;
    position: relative;
}

.nav-link:hover,
.nav-link:focus {
    color: var(--primary-color);
}

.nav-link::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 0;
    height: 2px;
    background-color: var(--primary-color);
    transition: var(--transition);
}

.nav-link:hover::after,
.nav-link:focus::after {
    width: 100%;
}

/* Main Content */
main {
    flex: 1;
    padding: 2rem 0;
}

/* Footer */
footer {
    background-color: white;
    padding: 2rem 0;
    text-align: center;
    margin-top: auto;
    border-top: 1px solid var(--border-color);
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    color: var(--dark-bg);
    line-height: 1.2;
    margin-bottom: 1rem;
}

h1 {
    font-size: 2.5rem;
}

h2 {
    font-size: 2rem;
}

h3 {
    font-size: 1.5rem;
}

p {
    margin-bottom: 1rem;
}

/* Buttons */
.button {
    display: inline-block;
    padding: 0.75rem 1.5rem;
    border-radius: var(--border-radius);
    font-weight: 600;
    text-decoration: none;
    text-align: center;
    cursor: pointer;
    border: none;
    transition: var(--transition);
}

.button.primary {
    background-color: var(--primary-color);
    color: white;
}

.button.primary:hover,
.button.primary:focus {
    background-color: var(--primary-hover);
}

.button.secondary {
    background-color: var(--secondary-color);
    color: white;
}

.button.secondary:hover,
.button.secondary:focus {
    background-color: var(--secondary-hover);
}

/* Hero Section */
.hero {
    background: linear-gradient(135deg, var(--dark-bg), var(--primary-color));
    color: var(--light-text);
    padding: 4rem 0;
    margin-bottom: 3rem;
    border-radius: var(--border-radius);
    text-align: center;
}

.hero-content {
    max-width: 800px;
    margin: 0 auto;
    padding: 0 1rem;
}

.hero-title {
    font-size: 3rem;
    margin-bottom: 1.5rem;
    color: var(--light-text);
}

.hero-description {
    font-size: 1.25rem;
    margin-bottom: 2rem;
    opacity: 0.9;
}

.hero-buttons {
    display: flex;
    justify-content: center;
    gap: 1rem;
}

/* Features */
.features {
    margin-bottom: 3rem;
}

.feature-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
}

.feature {
    background-color: white;
    border-radius: var(--border-radius);
    padding: 2rem;
    box-shadow: var(--box-shadow);
    text-align: center;
    transition: var(--transition);
}

.feature:hover {
    transform: translateY(-5px);
}

.feature-icon {
    width: 60px;
    height: 60px;
    margin: 0 auto 1.5rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    color: white;
}

.confluence-icon {
    background-color: var(--confluence-color);
}

.jira-icon {
    background-color: var(--jira-color);
}

.remedy-icon {
    background-color: var(--remedy-color);
}

.confluence-icon::before {
    content: "C";
}

.jira-icon::before {
    content: "J";
}

.remedy-icon::before {
    content: "R";
}

/* Section */
.section {
    margin-bottom: 3rem;
}

.section-title {
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    padding-bottom: 1rem;
}

.section-title::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 80px;
    height: 3px;
    background-color: var(--primary-color);
}

/* Workflow */
.workflow {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
}

.workflow-step {
    text-align: center;
    padding: 2rem;
    background-color: white;
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    position: relative;
}

.step-number {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background-color: var(--primary-color);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    margin: 0 auto 1rem;
}

/* Error page */
.error-container {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 60vh;
    text-align: center;
}

.error-content {
    max-width: 600px;
    padding: 2rem;
    background-color: white;
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
}

.error-title {
    color: var(--error-color);
}

.error-message {
    margin-bottom: 2rem;
}

/* Admin Dashboard */
.admin-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
}

.page-title {
    margin-bottom: 2rem;
    text-align: center;
}

.admin-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-bottom: 3rem;
}

.admin-card {
    background-color: white;
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    overflow: hidden;
}

.card-header {
    padding: 1.5rem;
    background-color: var(--dark-bg);
    color: var(--light-text);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.card-header h2 {
    margin: 0;
    color: var(--light-text);
    font-size: 1.25rem;
}

.status-indicator {
    padding: 0.25rem 0.75rem;
    border-radius: var(--border-radius);
    font-size: 0.875rem;
    font-weight: bold;
}

.status-indicator.connected {
    background-color: var(--success-color);
}

.status-indicator.error {
    background-color: var(--error-color);
}

.card-body {
    padding: 1.5rem;
}

.form-group {
    margin-bottom: 1.5rem;
}

.form-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
}

.form-group input,
.form-group select {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    font-family: var(--font-family);
}

.card-actions {
    display: flex;
    gap: 1rem;
}

.status-table-container {
    background-color: white;
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    overflow: hidden;
}

.status-table {
    width: 100%;
    border-collapse: collapse;
}

.status-table th,
.status-table td {
    padding: 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

.status-table th {
    background-color: var(--dark-bg);
    color: var(--light-text);
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .navbar .container {
        flex-direction: column;
        gap: 1rem;
    }
    
    .navbar-nav {
        width: 100%;
        justify-content: center;
    }
    
    .hero-title {
        font-size: 2.25rem;
    }
    
    .hero-description {
        font-size: 1rem;
    }
    
    .hero-buttons {
        flex-direction: column;
        width: 100%;
        max-width: 300px;
        margin: 0 auto;
    }
    
    .button {
        width: 100%;
    }
    
    .card-actions {
        flex-direction: column;
    }
}






















/* Chat Stylesheet */

/* Variables */
:root {
    --chat-bg: #f8fafc;
    --chat-bubble-user: #4f46e5;
    --chat-bubble-assistant: #ffffff;
    --chat-text-user: #ffffff;
    --chat-text-assistant: #334155;
    --chat-bubble-system: #f3f4f6;
    --chat-text-system: #6b7280;
    --chat-input-bg: #ffffff;
    --sidebar-bg: #f1f5f9;
    --sidebar-border: #e2e8f0;
}

/* Chat Container */
.chat-container {
    display: flex;
    height: calc(100vh - 160px);
    background-color: var(--chat-bg);
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    overflow: hidden;
    margin-top: 1rem;
}

/* Chat Sidebar */
.chat-sidebar {
    width: 300px;
    background-color: var(--sidebar-bg);
    border-right: 1px solid var(--sidebar-border);
    padding: 1.5rem;
    overflow-y: auto;
}

.chat-sidebar h3 {
    font-size: 1.125rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--sidebar-border);
}

.data-sources,
.chat-settings,
.chat-info {
    margin-bottom: 2rem;
}

.source-options {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.source-option {
    display: flex;
    align-items: center;
    cursor: pointer;
}

.source-option input[type="checkbox"] {
    margin-right: 0.5rem;
}

.source-name {
    font-weight: 500;
}

.setting {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}

.setting label {
    margin-bottom: 0.5rem;
    font-weight: 500;
}

.setting input[type="range"] {
    -webkit-appearance: none;
    width: 100%;
    height: 6px;
    background: var(--border-color);
    border-radius: 5px;
    outline: none;
}

.setting input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    background: var(--primary-color);
    border-radius: 50%;
    cursor: pointer;
}

.setting-value {
    align-self: flex-end;
    font-size: 0.875rem;
    color: var(--secondary-color);
}

/* Switch Toggle */
.switch {
    position: relative;
    display: inline-block;
    width: 48px;
    height: 24px;
    margin-right: 0.5rem;
}

.switch input {
    opacity: 0;
    width: 0;
    height: 0;
}

.switch-label {
    cursor: pointer;
    font-weight: 500;
}

.slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: var(--secondary-color);
    transition: var(--transition);
    border-radius: 34px;
}

.slider:before {
    position: absolute;
    content: "";
    height: 18px;
    width: 18px;
    left: 3px;
    bottom: 3px;
    background-color: white;
    transition: var(--transition);
    border-radius: 50%;
}

input:checked + .slider {
    background-color: var(--primary-color);
}

input:checked + .slider:before {
    transform: translateX(24px);
}

/* Chat Main */
.chat-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    max-width: calc(100% - 300px);
}

.chat-header {
    padding: 1rem 1.5rem;
    background-color: white;
    border-bottom: 1px solid var(--sidebar-border);
    display: flex;
    align-items: center;
}

.chat-header h2 {
    margin: 0;
    font-size: 1.25rem;
}

.chat-messages {
    flex: 1;
    padding: 1.5rem;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.message {
    max-width: 80%;
    padding: 1rem;
    border-radius: 1rem;
    position: relative;
    animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.message.user {
    align-self: flex-end;
    background-color: var(--chat-bubble-user);
    color: var(--chat-text-user);
    border-bottom-right-radius: 0.25rem;
}

.message.assistant {
    align-self: flex-start;
    background-color: var(--chat-bubble-assistant);
    color: var(--chat-text-assistant);
    border-bottom-left-radius: 0.25rem;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

.message.system {
    align-self: center;
    background-color: var(--chat-bubble-system);
    color: var(--chat-text-system);
    max-width: 90%;
    text-align: center;
}

.message-content p {
    margin: 0;
}

.message-content p:not(:last-child) {
    margin-bottom: 0.75rem;
}

.message-metadata {
    font-size: 0.75rem;
    opacity: 0.7;
    margin-top: 0.5rem;
    text-align: right;
}

.message.assistant .message-metadata {
    text-align: left;
}

.message-sources {
    margin-top: 0.75rem;
    border-top: 1px solid rgba(0, 0, 0, 0.1);
    padding-top: 0.75rem;
    font-size: 0.875rem;
}

.message-source {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    background-color: rgba(0, 0, 0, 0.05);
    border-radius: 4px;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    font-size: 0.75rem;
}

.message-source.confluence {
    background-color: rgba(0, 82, 204, 0.1);
    color: var(--confluence-color);
}

.message-source.jira {
    background-color: rgba(38, 132, 255, 0.1);
    color: var(--jira-color);
}

.message-source.remedy {
    background-color: rgba(101, 84, 192, 0.1);
    color: var(--remedy-color);
}

/* Chat Input */
.chat-input {
    padding: 1rem 1.5rem;
    background-color: white;
    border-top: 1px solid var(--sidebar-border);
}

#chat-form {
    display: flex;
    gap: 0.75rem;
}

#user-input {
    flex: 1;
    padding: 0.75rem 1rem;
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    font-family: var(--font-family);
    font-size: 1rem;
}

#user-input:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2);
}

#send-button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    border-radius: var(--border-radius);
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: var(--transition);
}

#send-button:hover {
    background-color: var(--primary-hover);
}

.send-icon {
    position: relative;
    width: 16px;
    height: 16px;
}

.send-icon:before {
    content: "";
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(-45deg);
    width: 16px;
    height: 2px;
    background-color: white;
}

.send-icon:after {
    content: "";
    position: absolute;
    top: 9px;
    left: 1px;
    width: 8px;
    height: 8px;
    border-left: 2px solid white;
    border-bottom: 2px solid white;
    transform: rotate(45deg);
}

/* Loading indicator */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.5rem 1rem;
    background-color: var(--chat-bubble-assistant);
    border-radius: 1rem;
    width: fit-content;
    align-self: flex-start;
    margin-top: 0.5rem;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

.typing-dot {
    width: 8px;
    height: 8px;
    background-color: var(--secondary-color);
    border-radius: 50%;
    animation: typing 1s infinite ease-in-out;
}

.typing-dot:nth-child(1) {
    animation-delay: 0s;
}

.typing-dot:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-dot:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typing {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-5px);
    }
}

/* Responsive adjustments */
@media (max-width: 992px) {
    .chat-container {
        flex-direction: column;
        height: auto;
    }
    
    .chat-sidebar {
        width: 100%;
        border-right: none;
        border-bottom: 1px solid var(--sidebar-border);
        padding: 1rem;
    }
    
    .chat-main {
        max-width: 100%;
        height: 60vh;
    }
}

@media (max-width: 576px) {
    .message {
        max-width: 90%;
    }
}
