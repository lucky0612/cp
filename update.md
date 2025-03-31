# RAG System Directory Structure

```
enterprise_rag/
│
├── .env                         # Environment variables
├── requirements.txt             # Project dependencies (with exact versions)
├── config.py                    # Configuration management
├── app.py                       # Main Flask application entry point
├── run.py                       # Application launcher
│
├── static/                      # Static files
│   ├── css/
│   │   ├── main.css             # Main stylesheet
│   │   └── chat.css             # Chat interface styling
│   ├── js/
│   │   ├── main.js              # Main JavaScript
│   │   └── chat.js              # Chat functionality
│   └── images/                  # UI images and icons
│
├── templates/                   # Flask templates
│   ├── base.html                # Base template with common structure
│   ├── index.html               # Home page
│   ├── chat.html                # Chat interface
│   └── error.html               # Error page
│
├── data_sources/                # Data source connectors
│   ├── __init__.py
│   ├── confluence/              # Confluence integration
│   │   ├── __init__.py
│   │   ├── client.py            # Confluence API client
│   │   ├── parser.py            # Content parsing utilities
│   │   └── connector.py         # Confluence connector for RAG
│   ├── jira/                    # JIRA integration
│   │   ├── __init__.py
│   │   ├── client.py            # JIRA API client
│   │   └── connector.py         # JIRA connector for RAG
│   └── remedy/                  # Remedy integration
│       ├── __init__.py
│       ├── client.py            # Remedy API client
│       └── connector.py         # Remedy connector for RAG
│
├── rag_engine/                  # RAG implementation
│   ├── __init__.py
│   ├── chunking.py              # Document chunking strategies
│   ├── embedding.py             # Embedding utilities
│   ├── retrieval.py             # Vector search and hybrid retrieval
│   ├── gemini_integration.py    # Gemini API integration
│   └── processor.py             # Main RAG processor
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── logger.py                # Logging configuration
│   ├── cache.py                 # Caching mechanisms
│   └── content_parser.py        # Content parsing utilities
│
└── api/                         # API endpoints
    ├── __init__.py
    ├── routes.py                # API routes
    └── response_formatter.py    # Response formatting
```



















# Environment Configuration

# System
DEBUG=True
LOG_LEVEL=INFO
CACHE_TIMEOUT=3600
CACHE_TYPE=simple

# Vertex AI / Google Cloud
PROJECT_ID=prj-dv-cws-4363
REGION=us-central1
MODEL_NAME=gemini-2.0-flash-001

# Confluence
CONFLUENCE_URL=https://cmegroup.atlassian.net
CONFLUENCE_USERNAME=your_username
CONFLUENCE_TOKEN=your_api_token

# JIRA
JIRA_URL=https://cmegroup.atlassian.net
JIRA_USERNAME=your_username
JIRA_TOKEN=your_api_token

# Remedy
REMEDY_URL=https://cmegroup-restapi.onbmc.com
REMEDY_USERNAME=your_username
REMEDY_PASSWORD=your_password
REMEDY_SSL_VERIFY=False

# RAG Configuration
VECTOR_DIMENSION=768
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
HYBRID_ALPHA=0.5  # Weight between vector and BM25 search (0-1)













anyio==4.9.0
beautifulsoup4==4.13.3
blinker==1.9.0
cachelib==0.13.0
cachetools==5.5.2
certifi==2025.1.31
charset-normalizer==3.4.1
click==8.1.8
defusedxml==0.7.1
docstring_parser==0.16
faiss-cpu==1.10.0
filelock==3.18.0
Flask==3.1.0
flask-caching==2.1.0
fspec==2023.3.2
google-api-core==2.94.1
google-api-python-client==2.160.0
google-auth==2.30.0
google-auth-httplib2==0.2.0
google-cloud-aiplatform==1.71.1
google-cloud-bigquery==3.31.0
google-cloud-core==2.4.1
google-cloud-resource-manager==1.14.0
google-cloud-storage==2.19.0
google-crc32c==1.7.1
google-genai==1.8.0
google-generativeai==0.4.0
googleapis-common-protos==1.69.2
grpcio==1.71.0
grpcio-status==1.71.0
html2text==2024.2.26
httpcore==1.0.7
httplib2==0.22.0
httpx==0.28.1
huggingface-hub==0.20.1
idna==3.10
itsdangerous==2.2.0
Jinja2==3.1.6
joblib==1.2.0
MarkupSafe==1.0.2
mpmath==1.0
networkx==3.9.1
nltk==3.9.1
numpy==1.25.2.4
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvcc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.1.3
nvidia-cuda-nvrtc-cu12==12.2.140
nvidia-cudnn-cu12==9.1.0.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
oauthlib==3.3.2
packaging==24.0.2
Pillow==11.0.0
proto-pydantic==0.6.1
protobuf==5.29.4
py4j==0.10.1
pydantic==2.11.1
pydantic_core==2.33.0
PyJWT==2.10.1
python-dateutil==2.9.0.post0
python-dotenv==1.1.0
PyYAML==6.0.2
rank4classes==0.2.12
requests==2.32.0
requests-toolbelt==1.0.0
rsa==4.9
safetensors==0.5.3
scikit-learn==1.6.1
scipy==1.15.2
sentence-transformers==4.0.1
setuptools==78.1.0
shapely==2.0.7
six==1.17.0
sniffio==1.3.1
sympy==1.13.1
tenacity==9.0.0
threadpoolctl==3.6.0
tokenizers==0.21.5
torch==2.6.0
transformers==4.50.3
triton==3.2.0
typing-extensions==4.13.0
typing-inspect==0.9.0
tzdata==2025.2
urllib3==2.3.0
vertexai==1.71.1
websockets==13.0.1
Werkzeug==3.1.3














"""
Configuration management for Enterprise RAG System.
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# System Config
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
LOG_LEVEL = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', 3600))
CACHE_TYPE = os.getenv('CACHE_TYPE', 'simple')

# Vertex AI / Google Cloud
PROJECT_ID = os.getenv('PROJECT_ID', 'prj-dv-cws-4363')
REGION = os.getenv('REGION', 'us-central1')
MODEL_NAME = os.getenv('MODEL_NAME', 'gemini-2.0-flash-001')

# Confluence
CONFLUENCE_URL = os.getenv('CONFLUENCE_URL', 'https://cmegroup.atlassian.net')
CONFLUENCE_USERNAME = os.getenv('CONFLUENCE_USERNAME')
CONFLUENCE_TOKEN = os.getenv('CONFLUENCE_TOKEN')

# JIRA
JIRA_URL = os.getenv('JIRA_URL', 'https://cmegroup.atlassian.net')
JIRA_USERNAME = os.getenv('JIRA_USERNAME')
JIRA_TOKEN = os.getenv('JIRA_TOKEN')

# Remedy
REMEDY_URL = os.getenv('REMEDY_URL', 'https://cmegroup-restapi.onbmc.com')
REMEDY_USERNAME = os.getenv('REMEDY_USERNAME')
REMEDY_PASSWORD = os.getenv('REMEDY_PASSWORD')
REMEDY_SSL_VERIFY = os.getenv('REMEDY_SSL_VERIFY', 'False').lower() == 'true'

# RAG Configuration
VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION', 768))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', 5))
HYBRID_ALPHA = float(os.getenv('HYBRID_ALPHA', 0.5))

# System Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for directory in [CACHE_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Flask Configuration
SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(24).hex())

















"""
Main Flask application entry point for Enterprise RAG System.
"""
import os
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache

import config
from utils.logger import setup_logging
from api.routes import register_api_routes

def create_app():
    """
    Application factory function to create and configure the Flask app.
    """
    # Initialize Flask app
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Setup logging
    setup_logging(app)
    app.logger.info("Starting Enterprise RAG System")
    
    # Configure caching
    cache = Cache(app, config={
        'CACHE_TYPE': config.CACHE_TYPE,
        'CACHE_DEFAULT_TIMEOUT': config.CACHE_TIMEOUT,
        'CACHE_DIR': config.CACHE_DIR if config.CACHE_TYPE == 'filesystem' else None
    })
    app.cache = cache
    
    # Register API routes
    register_api_routes(app)
    
    # Register error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('error.html', error=e, status_code=404), 404
        
    @app.errorhandler(500)
    def server_error(e):
        app.logger.error(f"Server error: {str(e)}")
        return render_template('error.html', error=e, status_code=500), 500
    
    # Main routes
    @app.route('/')
    def index():
        """Home page route."""
        return render_template('index.html')
    
    @app.route('/chat')
    def chat():
        """Chat interface route."""
        return render_template('chat.html')
    
    @app.route('/health')
    def health():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "version": "1.0.0"})
    
    return app

# Create application instance
app = create_app()

if __name__ == '__main__':
    # This is only used for local development
    # For production, use gunicorn or a proper WSGI server
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=config.DEBUG)

















"""
Application launcher for Enterprise RAG System.
This is the entry point for running the application.
"""
import os
from app import app

if __name__ == '__main__':
    # Get port from environment variable or use default 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Start the Flask application
    app.run(host='0.0.0.0', port=port)










"""
Logging configuration for Enterprise RAG System.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
import config

def setup_logging(app):
    """
    Configure logging for the application.
    
    Args:
        app: Flask application instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Set log level
    app.logger.setLevel(config.LOG_LEVEL)
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure file handler for logging
    file_handler = RotatingFileHandler(
        os.path.join(config.LOG_DIR, 'app.log'),
        maxBytes=10485760,  # 10 MB
        backupCount=10
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(config.LOG_LEVEL)
    
    # Configure console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(config.LOG_LEVEL)
    
    # Remove existing handlers if any
    for handler in app.logger.handlers:
        app.logger.removeHandler(handler)
    
    # Add handlers to the logger
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    
    # Add logger to Flask app
    app.logger.info("Logging configured successfully")

def get_logger(name):
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the module
        
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)
    
    # Return logger if handlers already exist
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(config.LOG_LEVEL)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    
    return logger

















"""
Utility package for Enterprise RAG System.
"""















"""
Caching utilities for Enterprise RAG System.
"""
import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
import config
from utils.logger import get_logger

logger = get_logger(__name__)

class CacheManager:
    """
    Cache manager to handle caching of data.
    Supports both memory and file-based caching.
    """
    def __init__(self, cache_dir=None, default_timeout=3600):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            default_timeout: Default cache timeout in seconds
        """
        self.memory_cache = {}
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.default_timeout = default_timeout
        
        # Ensure cache directory exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            
        logger.info(f"Cache manager initialized with directory: {self.cache_dir}")
    
    def _get_cache_key(self, key):
        """Generate a cache key hash."""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
    
    def _get_cache_file_path(self, key):
        """Get the file path for a cache key."""
        cache_key = self._get_cache_key(key)
        return os.path.join(self.cache_dir, f"{cache_key}.cache")
    
    def get(self, key, default=None):
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached data or default value
        """
        # Check memory cache first
        cache_key = self._get_cache_key(key)
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            if entry['expires'] > datetime.now():
                logger.debug(f"Cache hit for key: {key}")
                return entry['data']
            else:
                # Remove expired item
                del self.memory_cache[cache_key]
        
        # Check file cache
        cache_file = self._get_cache_file_path(key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                    
                if entry['expires'] > datetime.now():
                    # Update memory cache
                    self.memory_cache[cache_key] = entry
                    logger.debug(f"File cache hit for key: {key}")
                    return entry['data']
                else:
                    # Remove expired cache file
                    os.remove(cache_file)
            except (pickle.PickleError, IOError) as e:
                logger.error(f"Error reading cache file: {e}")
        
        logger.debug(f"Cache miss for key: {key}")
        return default
    
    def set(self, key, value, timeout=None):
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Data to cache
            timeout: Cache timeout in seconds
        """
        timeout = timeout or self.default_timeout
        expires = datetime.now() + timedelta(seconds=timeout)
        entry = {'data': value, 'expires': expires}
        
        # Update memory cache
        cache_key = self._get_cache_key(key)
        self.memory_cache[cache_key] = entry
        
        # Update file cache
        cache_file = self._get_cache_file_path(key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Cached data for key: {key}")
        except (pickle.PickleError, IOError) as e:
            logger.error(f"Error writing to cache file: {e}")
    
    def delete(self, key):
        """
        Delete an item from the cache.
        
        Args:
            key: Cache key
        """
        cache_key = self._get_cache_key(key)
        
        # Remove from memory cache
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
        
        # Remove from file cache
        cache_file = self._get_cache_file_path(key)
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                logger.debug(f"Deleted cache for key: {key}")
            except IOError as e:
                logger.error(f"Error deleting cache file: {e}")
    
    def clear(self):
        """Clear all cache."""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear file cache
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except IOError as e:
                    logger.error(f"Error deleting cache file: {e}")
        
        logger.info("Cache cleared")

# Global cache instance
cache_manager = CacheManager(
    cache_dir=config.CACHE_DIR,
    default_timeout=config.CACHE_TIMEOUT
)














"""
Content parsing utilities for Enterprise RAG System.
"""
import re
import html2text
from bs4 import BeautifulSoup
from utils.logger import get_logger

logger = get_logger(__name__)

class ContentParser:
    """
    Parser for different content types including HTML, Markdown, and plain text.
    Handles extraction of text, tables, code blocks, and other content formats.
    """
    
    def __init__(self):
        """Initialize the content parser."""
        self.html_parser = html2text.HTML2Text()
        self.html_parser.ignore_links = False
        self.html_parser.ignore_images = False
        self.html_parser.ignore_tables = False
        self.html_parser.body_width = 0  # No wrapping
        
    def clean_text(self, text):
        """
        Clean text by removing extra whitespace and normalizing line breaks.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace multiple newlines with double newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Trim leading/trailing whitespace
        return text.strip()
    
    def html_to_text(self, html_content):
        """
        Convert HTML to markdown text.
        
        Args:
            html_content: HTML content
            
        Returns:
            Markdown text
        """
        if not html_content:
            return ""
        
        try:
            markdown_text = self.html_parser.handle(html_content)
            return self.clean_text(markdown_text)
        except Exception as e:
            logger.error(f"Error converting HTML to text: {e}")
            # Fallback to BeautifulSoup for basic text extraction
            soup = BeautifulSoup(html_content, 'html.parser')
            return self.clean_text(soup.get_text())
    
    def extract_tables(self, html_content):
        """
        Extract tables from HTML content.
        
        Args:
            html_content: HTML content
            
        Returns:
            List of tables as markdown formatted strings
        """
        if not html_content:
            return []
        
        tables = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            table_elements = soup.find_all('table')
            
            for i, table in enumerate(table_elements):
                # Convert the table to markdown using html2text
                table_html = str(table)
                markdown_table = self.html_parser.handle(table_html)
                
                # Clean and add to list
                if markdown_table.strip():
                    tables.append(self.clean_text(markdown_table))
            
            logger.debug(f"Extracted {len(tables)} tables from HTML content")
            return tables
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            return []
    
    def extract_code_blocks(self, html_content):
        """
        Extract code blocks from HTML content.
        
        Args:
            html_content: HTML content
            
        Returns:
            List of code blocks with their language if available
        """
        if not html_content:
            return []
        
        code_blocks = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract from <pre><code> elements
            code_elements = soup.find_all('pre')
            for pre in code_elements:
                code = pre.find('code')
                if code:
                    lang = None
                    if code.get('class'):
                        # Try to extract language from class
                        for cls in code.get('class'):
                            if cls.startswith('language-'):
                                lang = cls.replace('language-', '')
                                break
                    
                    code_blocks.append({
                        'code': self.clean_text(code.get_text()),
                        'language': lang
                    })
            
            # Extract from <div class="code-block"> elements (Confluence format)
            div_code_blocks = soup.find_all('div', class_='code-block')
            for div in div_code_blocks:
                lang = None
                if div.get('data-lang'):
                    lang = div.get('data-lang')
                
                code_blocks.append({
                    'code': self.clean_text(div.get_text()),
                    'language': lang
                })
            
            logger.debug(f"Extracted {len(code_blocks)} code blocks from HTML content")
            return code_blocks
        except Exception as e:
            logger.error(f"Error extracting code blocks: {e}")
            return []
    
    def extract_lists(self, html_content):
        """
        Extract lists from HTML content.
        
        Args:
            html_content: HTML content
            
        Returns:
            List of lists as markdown formatted strings
        """
        if not html_content:
            return []
        
        lists = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract ordered and unordered lists
            for list_tag in soup.find_all(['ul', 'ol']):
                # Check if it's a top-level list
                if not list_tag.parent or list_tag.parent.name not in ['ul', 'ol', 'li']:
                    list_html = str(list_tag)
                    list_markdown = self.html_parser.handle(list_html)
                    
                    if list_markdown.strip():
                        lists.append(self.clean_text(list_markdown))
            
            logger.debug(f"Extracted {len(lists)} lists from HTML content")
            return lists
        except Exception as e:
            logger.error(f"Error extracting lists: {e}")
            return []
    
    def extract_headings(self, html_content):
        """
        Extract headings from HTML content.
        
        Args:
            html_content: HTML content
            
        Returns:
            List of headings with their level
        """
        if not html_content:
            return []
        
        headings = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for i in range(1, 7):  # h1 to h6
                for heading in soup.find_all(f'h{i}'):
                    headings.append({
                        'text': self.clean_text(heading.get_text()),
                        'level': i
                    })
            
            logger.debug(f"Extracted {len(headings)} headings from HTML content")
            return headings
        except Exception as e:
            logger.error(f"Error extracting headings: {e}")
            return []
    
    def extract_images(self, html_content):
        """
        Extract image references from HTML content.
        
        Args:
            html_content: HTML content
            
        Returns:
            List of image details (alt text, src)
        """
        if not html_content:
            return []
        
        images = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for img in soup.find_all('img'):
                alt_text = img.get('alt', '')
                src = img.get('src', '')
                
                if src:
                    images.append({
                        'alt_text': alt_text,
                        'src': src
                    })
            
            logger.debug(f"Extracted {len(images)} image references from HTML content")
            return images
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            return []
    
    def parse_confluence_content(self, page_content):
        """
        Parse Confluence page content.
        
        Args:
            page_content: Confluence page content object
            
        Returns:
            Parsed content with text and structured elements
        """
        if not page_content or 'body' not in page_content or 'storage' not in page_content['body']:
            return {'text': "", 'elements': []}
        
        try:
            html_content = page_content['body']['storage']['value']
            content_type = page_content['body']['storage'].get('representation', 'storage')
            
            # If not HTML, return as is
            if content_type != 'storage':
                return {'text': html_content, 'elements': []}
            
            # Extract full text
            text = self.html_to_text(html_content)
            
            # Extract structured elements
            elements = []
            
            # Add tables
            tables = self.extract_tables(html_content)
            for table in tables:
                elements.append({
                    'type': 'table',
                    'content': table
                })
            
            # Add code blocks
            code_blocks = self.extract_code_blocks(html_content)
            for code_block in code_blocks:
                elements.append({
                    'type': 'code',
                    'content': code_block['code'],
                    'language': code_block['language']
                })
            
            # Add lists
            lists = self.extract_lists(html_content)
            for list_content in lists:
                elements.append({
                    'type': 'list',
                    'content': list_content
                })
            
            # Add headings
            headings = self.extract_headings(html_content)
            for heading in headings:
                elements.append({
                    'type': 'heading',
                    'content': heading['text'],
                    'level': heading['level']
                })
            
            # Add image references
            images = self.extract_images(html_content)
            for image in images:
                elements.append({
                    'type': 'image',
                    'alt_text': image['alt_text'],
                    'src': image['src']
                })
            
            logger.info(f"Parsed Confluence content with {len(elements)} structured elements")
            return {
                'text': text,
                'elements': elements
            }
            
        except Exception as e:
            logger.error(f"Error parsing Confluence content: {e}")
            return {'text': "", 'elements': []}
    
    def parse_jira_description(self, description):
        """
        Parse JIRA issue description.
        
        Args:
            description: JIRA issue description
            
        Returns:
            Parsed description text
        """
        if not description:
            return ""
        
        try:
            # Check if the description is already in HTML format
            if description.startswith('<') and '>' in description:
                return self.html_to_text(description)
            
            # Otherwise, treat as plain text or Jira's wiki markup
            return self.clean_text(description)
        except Exception as e:
            logger.error(f"Error parsing JIRA description: {e}")
            return description

# Global content parser instance
content_parser = ContentParser()















"""
Google Gemini API integration for Enterprise RAG System.
"""
import json
from datetime import datetime
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

import config
from utils.logger import get_logger
from utils.cache import cache_manager

logger = get_logger(__name__)

class GeminiManager:
    """
    Manager class for Google Gemini LLM integration.
    Handles prompting, response generation, and context management.
    """
    
    def __init__(self):
        """Initialize the Gemini Manager."""
        self.initialized = False
        self.model_name = config.MODEL_NAME
        self.project_id = config.PROJECT_ID
        self.location = config.REGION
        self.model = None
        
        # Initialize Vertex AI
        self._initialize()
    
    def _initialize(self):
        """Initialize the Vertex AI client and Gemini model."""
        try:
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            
            # Create model instance
            self.model = GenerativeModel(self.model_name)
            
            logger.info(f"Successfully initialized Gemini model: {self.model_name}")
            self.initialized = True
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            self.initialized = False
    
    def _build_system_prompt(self):
        """
        Build the system prompt for Gemini.
        
        Returns:
            str: System prompt
        """
        return """
You are an AI assistant for a corporate environment, designed to provide helpful and accurate information based on the provided context.
Your task is to help users find information from corporate knowledge sources like Confluence, JIRA, and Remedy.

Guidelines:
1. Answer questions based on the context provided. If the answer is not in the context, say so.
2. Be concise and focus on the user's question.
3. Cite sources when appropriate by mentioning the source name and link if available.
4. For technical questions, provide clear explanations with code examples when relevant.
5. When discussing dates or deadlines, clearly specify the timeframe.
6. If you encounter information that appears outdated, mention this possibility.
7. Format your responses with markdown when appropriate for readability.
8. When providing code, use proper code blocks with language specification.
9. If the context contains different or conflicting information, acknowledge this and present multiple perspectives.

Current date: {current_date}
"""
    
    def _build_context_prompt(self, context_items):
        """
        Build the context prompt from retrieved chunks.
        
        Args:
            context_items: List of context items with source information
            
        Returns:
            str: Context prompt
        """
        if not context_items:
            return "No relevant context found."
        
        context_text = "Here is the context information from various sources:\n\n"
        
        for i, item in enumerate(context_items, 1):
            source = item.get('source', 'Unknown source')
            source_type = item.get('source_type', 'Unknown type')
            content = item.get('content', '')
            source_link = item.get('source_link', '')
            
            context_text += f"[{i}] {source} ({source_type})"
            if source_link:
                context_text += f" [Link: {source_link}]"
            context_text += "\n"
            
            # Add content with indentation
            for line in content.split('\n'):
                context_text += f"    {line}\n"
            
            context_text += "\n"
        
        return context_text
    
    def generate_response(self, query, context_items, conversation_history=None, stream=False):
        """
        Generate a response using Gemini with RAG context.
        
        Args:
            query: User query
            context_items: Retrieved context items
            conversation_history: Optional conversation history
            stream: Whether to stream the response
            
        Returns:
            Response text or a generator for streaming
        """
        if not self.initialized:
            logger.warning("Gemini model not initialized. Attempting to reinitialize.")
            self._initialize()
            if not self.initialized:
                return "Sorry, I'm currently unable to access the AI service. Please try again later."
        
        # Build the full prompt
        system_prompt = self._build_system_prompt().format(
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        context_prompt = self._build_context_prompt(context_items)
        
        full_prompt = f"{system_prompt}\n\n{context_prompt}\n\nUser question: {query}\n\nAnswer:"
        
        # Set up generation config
        generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
        )
        
        try:
            # Cache key for non-streaming responses
            cache_key = f"gemini_response:{hash(full_prompt)}" if not stream else None
            
            # Check cache for non-streaming responses
            if not stream and cache_key:
                cached_response = cache_manager.get(cache_key)
                if cached_response:
                    logger.info("Using cached response for query")
                    return cached_response
            
            logger.info(f"Generating response for query: {query}")
            
            if stream:
                return self._generate_streaming_response(full_prompt, generation_config)
            else:
                response = self._generate_complete_response(full_prompt, generation_config)
                
                # Cache the response
                if cache_key:
                    cache_manager.set(cache_key, response, timeout=3600)
                
                return response
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    def _generate_complete_response(self, prompt, generation_config):
        """Generate a complete response."""
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Extract text from response
        if response and hasattr(response, 'text'):
            return response.text
        else:
            logger.error("Empty or invalid response from Gemini")
            return "Sorry, I couldn't generate a response. Please try again."
    
    def _generate_streaming_response(self, prompt, generation_config):
        """Generate a streaming response."""
        response_generator = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True
        )
        
        # Return a generator that yields text chunks
        for chunk in response_generator:
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text

# Global Gemini manager instance
gemini_manager = GeminiManager()














"""
RAG Engine package for Enterprise RAG System.
"""

















"""
Document chunking strategies for Enterprise RAG System.
"""
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from utils.logger import get_logger
import config

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    pass  # Handle silently, log below

logger = get_logger(__name__)

class DocumentChunker:
    """
    Handles document chunking strategies for RAG.
    Supports various chunking methods including fixed size,
    semantic boundary detection, and hierarchical chunking.
    """
    
    def __init__(self, chunk_size=None, chunk_overlap=None):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Size of chunks in characters or tokens
            chunk_overlap: Overlap between chunks in characters or tokens
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        # Ensure NLTK resources are downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.warning("NLTK punkt tokenizer not found, attempting to download...")
            try:
                nltk.download('punkt', quiet=True)
                logger.info("Successfully downloaded NLTK punkt tokenizer")
            except Exception as e:
                logger.error(f"Failed to download NLTK resources: {e}")
    
    def chunk_by_fixed_size(self, text, token_based=False):
        """
        Chunk document by fixed size with overlap.
        
        Args:
            text: Document text
            token_based: If True, use token count instead of character count
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        
        if token_based:
            # Tokenize the text
            tokens = word_tokenize(text)
            
            # Create chunks by token count
            start = 0
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunks.append(' '.join(chunk_tokens))
                
                # Move to next chunk with overlap
                start = end - self.chunk_overlap
        else:
            # Create chunks by character count
            start = 0
            text_len = len(text)
            
            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                
                # If we're not at the end, try to find a good breaking point
                if end < text_len:
                    # Look for a period, question mark, or exclamation point followed by a space
                    good_break = max(
                        text.rfind('. ', start, end),
                        text.rfind('? ', start, end),
                        text.rfind('! ', start, end),
                        text.rfind('\n', start, end)
                    )
                    
                    if good_break != -1 and good_break > start:
                        end = good_break + 1
                
                chunks.append(text[start:end])
                
                # Move to next chunk with overlap
                start = end - self.chunk_overlap
        
        logger.debug(f"Created {len(chunks)} fixed-size chunks from text")
        return chunks
    
    def chunk_by_semantic_boundaries(self, text):
        """
        Chunk document by semantic boundaries (paragraphs, sentences).
        
        Args:
            text: Document text
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        current_chunk = ""
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph is short, add it directly to the current chunk
            if len(paragraph) < self.chunk_size / 2:
                if len(current_chunk + " " + paragraph) <= self.chunk_size:
                    current_chunk += " " + paragraph if current_chunk else paragraph
                else:
                    chunks.append(current_chunk)
                    current_chunk = paragraph
            else:
                # For long paragraphs, split into sentences
                sentences = sent_tokenize(paragraph)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    # If sentence would make chunk too large, start a new chunk
                    if len(current_chunk + " " + sentence) <= self.chunk_size:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        chunks.append(current_chunk)
                        current_chunk = sentence
        
        # Add the final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.debug(f"Created {len(chunks)} semantic-boundary chunks from text")
        return chunks
    
    def chunk_hierarchically(self, text, metadata=None):
        """
        Create hierarchical chunks (document → section → paragraph).
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of chunks with hierarchical metadata
        """
        if not text:
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Split document into sections based on headings
        section_pattern = r'(?:^|\n)#{1,6}\s+(.+?)(?:\n|$)'
        sections = re.split(section_pattern, text)
        headings = re.findall(section_pattern, text)
        
        # Ensure we have a heading for each section (use empty for first if needed)
        if len(sections) > len(headings) + 1:
            headings = [""] + headings
        
        # Process each section
        for i, section_text in enumerate(sections):
            if not section_text.strip():
                continue
            
            # Get current section heading
            section_heading = headings[i - 1] if i > 0 and i - 1 < len(headings) else ""
            
            # Split section into paragraphs
            paragraphs = re.split(r'\n\s*\n', section_text)
            
            # Create a chunk for each paragraph with hierarchical metadata
            for j, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Skip if paragraph is too large and chunk it further
                if len(paragraph) > self.chunk_size:
                    sub_chunks = self.chunk_by_semantic_boundaries(paragraph)
                    for k, sub_chunk in enumerate(sub_chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update({
                            'section': section_heading,
                            'paragraph_index': f"{j}.{k}",
                            'hierarchical_level': 'paragraph'
                        })
                        
                        chunks.append({
                            'text': sub_chunk,
                            'metadata': chunk_metadata
                        })
                else:
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'section': section_heading,
                        'paragraph_index': j,
                        'hierarchical_level': 'paragraph'
                    })
                    
                    chunks.append({
                        'text': paragraph,
                        'metadata': chunk_metadata
                    })
        
        logger.debug(f"Created {len(chunks)} hierarchical chunks from text")
        return chunks
    
    def chunk_document(self, document, method='semantic', metadata=None):
        """
        Chunk a document using the specified method.
        
        Args:
            document: Document text
            method: Chunking method ('fixed', 'semantic', or 'hierarchical')
            metadata: Document metadata
            
        Returns:
            List of chunks with metadata
        """
        text = document.get('text', '') if isinstance(document, dict) else document
        metadata = metadata or {}
        
        if isinstance(document, dict) and 'metadata' in document:
            metadata.update(document['metadata'])
        
        # Choose chunking method
        if method == 'fixed':
            text_chunks = self.chunk_by_fixed_size(text)
        elif method == 'semantic':
            text_chunks = self.chunk_by_semantic_boundaries(text)
        elif method == 'hierarchical':
            return self.chunk_hierarchically(text, metadata)
        else:
            logger.warning(f"Unknown chunking method: {method}. Using semantic chunking.")
            text_chunks = self.chunk_by_semantic_boundaries(text)
        
        # Add metadata to each chunk
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = i
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return chunks

# Global document chunker instance
document_chunker = DocumentChunker(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP
)
















"""
Embedding utilities for Enterprise RAG System.
"""
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger
from utils.cache import cache_manager
import config

logger = get_logger(__name__)

class EmbeddingManager:
    """
    Handles text embedding generation and management.
    Uses sentence-transformers for embedding generation.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.vector_dimension = config.VECTOR_DIMENSION
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence-transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Successfully loaded embedding model: {self.model_name}")
            
            # Update vector dimension based on the model
            self.vector_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Vector dimension: {self.vector_dimension}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")
    
    def get_embedding(self, text):
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.vector_dimension)
        
        # Check cache first
        cache_key = f"embedding:{self.model_name}:{hash(text)}"
        cached_embedding = cache_manager.get(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate embedding if not in cache
        if self.model is None:
            self._initialize_model()
        
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            
            # Cache the embedding
            cache_manager.set(cache_key, embedding)
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.vector_dimension)
    
    def get_embeddings(self, texts, batch_size=32):
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for embedding generation
            
        Returns:
            List of numpy.ndarray: Embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts and check cache
        non_empty_texts = []
        cached_embeddings = {}
        indices_to_generate = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                cached_embeddings[i] = np.zeros(self.vector_dimension)
            else:
                cache_key = f"embedding:{self.model_name}:{hash(text)}"
                cached_embedding = cache_manager.get(cache_key)
                
                if cached_embedding is not None:
                    cached_embeddings[i] = cached_embedding
                else:
                    non_empty_texts.append(text)
                    indices_to_generate.append(i)
        
        # Generate embeddings for texts not in cache
        if non_empty_texts:
            if self.model is None:
                self._initialize_model()
            
            try:
                new_embeddings = self.model.encode(
                    non_empty_texts,
                    batch_size=batch_size,
                    show_progress_bar=False
                )
                
                # Cache the new embeddings
                for j, idx in enumerate(indices_to_generate):
                    embedding = new_embeddings[j]
                    text = texts[idx]
                    cache_key = f"embedding:{self.model_name}:{hash(text)}"
                    cache_manager.set(cache_key, embedding)
                    cached_embeddings[idx] = embedding
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                # Use zero vectors for failed embeddings
                for idx in indices_to_generate:
                    cached_embeddings[idx] = np.zeros(self.vector_dimension)
        
        # Reconstruct the ordered list of embeddings
        embeddings = [cached_embeddings[i] for i in range(len(texts))]
        return embeddings
    
    def embed_chunks(self, chunks):
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of chunks with added embeddings
        """
        if not chunks:
            return []
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.get_embeddings(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding['embedding'] = embedding
            embedded_chunks.append(chunk_with_embedding)
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return embedded_chunks
    
    def save_embeddings(self, embeddings, filepath):
        """
        Save embeddings to a file.
        
        Args:
            embeddings: List of embeddings or dict mapping IDs to embeddings
            filepath: Output file path
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved embeddings to {filepath}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def load_embeddings(self, filepath):
        """
        Load embeddings from a file.
        
        Args:
            filepath: Input file path
            
        Returns:
            Loaded embeddings
        """
        try:
            with open(filepath, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"Loaded embeddings from {filepath}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None
    
    def get_query_embedding(self, query):
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            numpy.ndarray: Query embedding vector
        """
        return self.get_embedding(query)

# Global embedding manager instance
embedding_manager = EmbeddingManager()
























"""
Retrieval components for Enterprise RAG System.
"""
import os
import pickle
import numpy as np
import faiss
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

import config
from utils.logger import get_logger
from utils.cache import cache_manager
from rag_engine.embedding import embedding_manager

# Download required NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    pass  # Handle silently, log below

logger = get_logger(__name__)

class RetrievalEngine:
    """
    Handles document retrieval using vector similarity search and BM25 ranking.
    Supports hybrid retrieval combining multiple ranking methods.
    """
    
    def __init__(self):
        """Initialize the retrieval engine."""
        self.vector_index = None
        self.chunks = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Ensure NLTK resources are downloaded
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.warning("NLTK resources not found, attempting to download...")
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                logger.info("Successfully downloaded NLTK resources")
            except Exception as e:
                logger.error(f"Failed to download NLTK resources: {e}")
    
    def initialize_vector_index(self, dimension=None):
        """
        Initialize the FAISS vector index.
        
        Args:
            dimension: Dimension of the embedding vectors
        """
        dimension = dimension or embedding_manager.vector_dimension
        
        try:
            # Create a new FAISS index for L2 similarity
            self.vector_index = faiss.IndexFlatL2(dimension)
            logger.info(f"Initialized FAISS vector index with dimension {dimension}")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            self.vector_index = None
    
    def add_documents(self, chunks):
        """
        Add document chunks to the retrieval engine.
        
        Args:
            chunks: List of document chunks with embeddings
        """
        if not chunks:
            logger.warning("No chunks provided to add to retrieval engine")
            return
        
        # Store the chunks
        self.chunks = chunks
        
        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in chunks], dtype=np.float32)
        
        # Initialize vector index if needed
        if self.vector_index is None:
            self.initialize_vector_index(embeddings.shape[1])
        
        # Add embeddings to the index
        try:
            self.vector_index.add(embeddings)
            logger.info(f"Added {len(chunks)} chunks to the vector index")
        except Exception as e:
            logger.error(f"Error adding embeddings to vector index: {e}")
        
        # Initialize TF-IDF for BM25-like search
        self._initialize_tfidf([chunk['text'] for chunk in chunks])
    
    def _initialize_tfidf(self, texts):
        """
        Initialize TF-IDF vectorizer for lexical search.
        
        Args:
            texts: List of document texts
        """
        try:
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),
                max_features=10000
            )
            
            # Fit and transform the texts
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info("Initialized TF-IDF vectorizer for lexical search")
        except Exception as e:
            logger.error(f"Error initializing TF-IDF vectorizer: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def save_index(self, directory):
        """
        Save the retrieval index and related data.
        
        Args:
            directory: Directory to save the index files
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save FAISS index
            index_path = os.path.join(directory, 'vector_index.faiss')
            faiss.write_index(self.vector_index, index_path)
            
            # Save chunks
            chunks_path = os.path.join(directory, 'chunks.pkl')
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            # Save TF-IDF vectorizer and matrix
            tfidf_path = os.path.join(directory, 'tfidf.pkl')
            with open(tfidf_path, 'wb') as f:
                pickle.dump((self.tfidf_vectorizer, self.tfidf_matrix), f)
            
            logger.info(f"Saved retrieval index to {directory}")
        except Exception as e:
            logger.error(f"Error saving retrieval index: {e}")
    
    def load_index(self, directory):
        """
        Load the retrieval index and related data.
        
        Args:
            directory: Directory containing the index files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load FAISS index
            index_path = os.path.join(directory, 'vector_index.faiss')
            self.vector_index = faiss.read_index(index_path)
            
            # Load chunks
            chunks_path = os.path.join(directory, 'chunks.pkl')
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            # Load TF-IDF vectorizer and matrix
            tfidf_path = os.path.join(directory, 'tfidf.pkl')
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer, self.tfidf_matrix = pickle.load(f)
            
            logger.info(f"Loaded retrieval index from {directory}")
            return True
        except Exception as e:
            logger.error(f"Error loading retrieval index: {e}")
            return False
    
    def vector_search(self, query_embedding, k=None):
        """
        Perform vector similarity search.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        k = k or config.TOP_K_RESULTS
        
        if self.vector_index is None or not self.chunks:
            logger.warning("Vector index not initialized or no chunks available")
            return []
        
        try:
            # Reshape query embedding for FAISS
            query_embedding = np.array([query_embedding], dtype=np.float32)
            
            # Perform the search
            distances, indices = self.vector_index.search(query_embedding, k)
            
            # Format the results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                # Skip invalid indices
                if idx < 0 or idx >= len(self.chunks):
                    continue
                
                # Calculate cosine similarity (1 - distance_normalized)
                # FAISS uses L2 distance, so we convert to a similarity score
                max_distance = np.sqrt(2)  # Max L2 distance for normalized vectors
                similarity = 1 - (distance / max_distance)
                
                results.append((self.chunks[idx], float(similarity)))
            
            logger.debug(f"Vector search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            return []
    
    def bm25_search(self, query, k=None):
        """
        Perform BM25-like lexical search using TF-IDF.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        k = k or config.TOP_K_RESULTS
        
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None or not self.chunks:
            logger.warning("TF-IDF not initialized or no chunks available")
            return []
        
        try:
            # Transform query to TF-IDF space
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarity scores
            similarity_scores = (self.tfidf_matrix @ query_vector.T).toarray().flatten()
            
            # Get the top k results
            top_indices = similarity_scores.argsort()[-k:][::-1]
            
            # Format the results
            results = []
            for idx in top_indices:
                if similarity_scores[idx] > 0:
                    results.append((self.chunks[idx], float(similarity_scores[idx])))
            
            logger.debug(f"BM25 search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error performing BM25 search: {e}")
            return []
    
    def hybrid_search(self, query, query_embedding=None, k=None, alpha=None):
        """
        Perform hybrid search combining vector and BM25 search.
        
        Args:
            query: Query text
            query_embedding: Query embedding vector (optional)
            k: Number of results to return
            alpha: Weight for vector search (0-1)
            
        Returns:
            List of (chunk, score) tuples
        """
        k = k or config.TOP_K_RESULTS
        alpha = alpha or config.HYBRID_ALPHA
        
        # Generate query embedding if not provided
        if query_embedding is None:
            query_embedding = embedding_manager.get_query_embedding(query)
        
        # Cache key for hybrid search
        cache_key = f"hybrid_search:{hash(query)}:{hash(str(query_embedding))}:{k}:{alpha}"
        cached_results = cache_manager.get(cache_key)
        if cached_results:
            return cached_results
        
        # Perform vector search
        vector_results = self.vector_search(query_embedding, k=k*2)
        
        # Perform BM25 search
        bm25_results = self.bm25_search(query, k=k*2)
        
        # Combine results using the specified alpha
        combined_scores = {}
        
        # Add vector search results
        for chunk, score in vector_results:
            chunk_id = id(chunk)  # Use object id as a unique identifier
            combined_scores[chunk_id] = {
                'chunk': chunk,
                'vector_score': score,
                'bm25_score': 0,
                'combined_score': alpha * score
            }
        
        # Add BM25 search results
        for chunk, score in bm25_results:
            chunk_id = id(chunk)
            if chunk_id in combined_scores:
                combined_scores[chunk_id]['bm25_score'] = score
                combined_scores[chunk_id]['combined_score'] += (1 - alpha) * score
            else:
                combined_scores[chunk_id] = {
                    'chunk': chunk,
                    'vector_score': 0,
                    'bm25_score': score,
                    'combined_score': (1 - alpha) * score
                }
        
        # Sort by combined score and take top k
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:k]
        
        # Format final results
        results = [(item['chunk'], item['combined_score']) for item in sorted_results]
        
        # Cache the results
        cache_manager.set(cache_key, results, timeout=3600)
        
        logger.info(f"Hybrid search returned {len(results)} results with alpha={alpha}")
        return results
    
    def retrieve(self, query, search_type='hybrid', k=None):
        """
        Retrieve documents for a query.
        
        Args:
            query: Query text
            search_type: Type of search ('vector', 'bm25', or 'hybrid')
            k: Number of results to return
            
        Returns:
            List of retrieved document chunks with scores
        """
        k = k or config.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = embedding_manager.get_query_embedding(query)
        
        # Perform the appropriate search
        if search_type == 'vector':
            results = self.vector_search(query_embedding, k=k)
        elif search_type == 'bm25':
            results = self.bm25_search(query, k=k)
        else:  # hybrid is the default
            results = self.hybrid_search(query, query_embedding, k=k)
        
        # Format the results for the RAG engine
        retrieved_chunks = []
        for chunk, score in results:
            retrieved_chunk = {
                'content': chunk['text'],
                'metadata': chunk.get('metadata', {}),
                'source': chunk.get('metadata', {}).get('source', 'Unknown'),
                'source_type': chunk.get('metadata', {}).get('source_type', 'Unknown'),
                'source_link': chunk.get('metadata', {}).get('source_link', ''),
                'score': score
            }
            retrieved_chunks.append(retrieved_chunk)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query}")
        return retrieved_chunks

# Global retrieval engine instance
retrieval_engine = RetrievalEngine()
















"""
Main RAG processor for Enterprise RAG System.
"""
import time
from concurrent.futures import ThreadPoolExecutor
import config
from utils.logger import get_logger
from utils.cache import cache_manager
from rag_engine.chunking import document_chunker
from rag_engine.embedding import embedding_manager
from rag_engine.retrieval import retrieval_engine
from rag_engine.gemini_integration import gemini_manager

logger = get_logger(__name__)

class RAGProcessor:
    """
    Main RAG processor handling the end-to-end flow:
    1. Document ingestion and chunking
    2. Embedding generation
    3. Retrieval
    4. Response generation using LLM
    """
    
    def __init__(self):
        """Initialize the RAG processor."""
        self.is_initialized = False
        self.document_sources = {}
        
    def initialize(self):
        """Initialize the RAG processor components."""
        if self.is_initialized:
            return
        
        try:
            # Initialization tasks are performed when needed
            # Document chunker, embedding manager, retrieval engine and gemini manager
            # are already instantiated as global instances
            
            self.is_initialized = True
            logger.info("RAG processor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG processor: {e}")
            self.is_initialized = False
    
    def register_document_source(self, source_id, source_manager):
        """
        Register a document source.
        
        Args:
            source_id: Unique identifier for the source
            source_manager: Source manager instance
        """
        self.document_sources[source_id] = source_manager
        logger.info(f"Registered document source: {source_id}")
    
    def process_documents(self, documents, source_type=None):
        """
        Process documents for RAG.
        
        Args:
            documents: List of documents to process
            source_type: Type of source (e.g., 'confluence', 'jira', 'remedy')
            
        Returns:
            List of processed chunks with embeddings
        """
        if not documents:
            logger.warning("No documents provided for processing")
            return []
        
        # Step 1: Chunk documents
        start_time = time.time()
        chunks = []
        
        for doc in documents:
            # Add source type to metadata
            if isinstance(doc, dict) and 'metadata' in doc and source_type:
                doc['metadata']['source_type'] = source_type
            
            # Chunk the document using semantic boundaries by default
            doc_chunks = document_chunker.chunk_document(doc, method='semantic')
            chunks.extend(doc_chunks)
        
        chunk_time = time.time() - start_time
        logger.info(f"Chunked {len(documents)} documents into {len(chunks)} chunks in {chunk_time:.2f}s")
        
        # Step 2: Generate embeddings
        start_time = time.time()
        chunks_with_embeddings = embedding_manager.embed_chunks(chunks)
        embed_time = time.time() - start_time
        logger.info(f"Generated embeddings for {len(chunks)} chunks in {embed_time:.2f}s")
        
        return chunks_with_embeddings
    
    def index_documents(self, chunks_with_embeddings, overwrite=False):
        """
        Index document chunks for retrieval.
        
        Args:
            chunks_with_embeddings: List of chunks with embeddings
            overwrite: Whether to overwrite existing index
        """
        if not chunks_with_embeddings:
            logger.warning("No chunks provided for indexing")
            return
        
        # If overwrite, create a new index
        if overwrite or retrieval_engine.vector_index is None:
            retrieval_engine.initialize_vector_index()
        
        # Add chunks to the index
        retrieval_engine.add_documents(chunks_with_embeddings)
    
    def retrieve_relevant_chunks(self, query, search_type='hybrid', top_k=None):
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            search_type: Type of search ('vector', 'bm25', or 'hybrid')
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks
        """
        top_k = top_k or config.TOP_K_RESULTS
        
        # Check if retrieval engine is ready
        if retrieval_engine.vector_index is None or not retrieval_engine.chunks:
            logger.warning("Retrieval engine not initialized with documents")
            return []
        
        # Cache key for retrieval
        cache_key = f"retrieval:{hash(query)}:{search_type}:{top_k}"
        cached_results = cache_manager.get(cache_key)
        if cached_results:
            return cached_results
        
        # Retrieve chunks
        start_time = time.time()
        retrieved_chunks = retrieval_engine.retrieve(query, search_type, k=top_k)
        retrieval_time = time.time() - start_time
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.2f}s using {search_type} search")
        
        # Cache the results
        cache_manager.set(cache_key, retrieved_chunks, timeout=1800)  # 30 minutes
        
        return retrieved_chunks
    
    def generate_response(self, query, context_items, conversation_history=None, stream=False):
        """
        Generate a response using the Gemini LLM.
        
        Args:
            query: User query
            context_items: Retrieved context items
            conversation_history: Optional conversation history
            stream: Whether to stream the response
            
        Returns:
            Generated response or generator
        """
        start_time = time.time()
        
        # Generate response
        response = gemini_manager.generate_response(
            query, 
            context_items, 
            conversation_history=conversation_history,
            stream=stream
        )
        
        if not stream:
            generation_time = time.time() - start_time
            logger.info(f"Generated response in {generation_time:.2f}s")
        
        return response
    
    def process_query(self, query, sources=None, search_type='hybrid', top_k=None, stream=False):
        """
        Process a user query through the entire RAG pipeline.
        
        Args:
            query: User query
            sources: List of source types to query ('confluence', 'jira', 'remedy')
            search_type: Type of search ('vector', 'bm25', or 'hybrid')
            top_k: Number of chunks to retrieve
            stream: Whether to stream the response
            
        Returns:
            Generated response and retrieved chunks
        """
        self.initialize()
        
        # If no specific sources are provided, use all available
        if not sources:
            sources = list(self.document_sources.keys())
        elif isinstance(sources, str):
            sources = [sources]
        
        # Retrieve chunks from specified sources
        if sources and self.document_sources:
            # Check if we need to retrieve and index documents
            if not retrieval_engine.chunks:
                self._retrieve_and_index_sources(sources)
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve_relevant_chunks(query, search_type, top_k)
        
        # Generate response
        response = self.generate_response(query, retrieved_chunks, stream=stream)
        
        return {
            'response': response,
            'context': retrieved_chunks
        }
    
    def _retrieve_and_index_sources(self, sources):
        """
        Retrieve documents from sources and index them.
        
        Args:
            sources: List of source types to retrieve from
        """
        all_chunks = []
        
        # Use ThreadPoolExecutor for parallel document retrieval
        with ThreadPoolExecutor(max_workers=min(len(sources), 3)) as executor:
            future_to_source = {}
            
            # Start document retrieval for each source
            for source in sources:
                if source in self.document_sources:
                    source_manager = self.document_sources[source]
                    future = executor.submit(source_manager.get_documents)
                    future_to_source[future] = source
            
            # Process results as they complete
            for future in future_to_source:
                source = future_to_source[future]
                try:
                    documents = future.result()
                    if documents:
                        # Process and chunk documents
                        chunks = self.process_documents(documents, source_type=source)
                        all_chunks.extend(chunks)
                        logger.info(f"Retrieved and processed {len(documents)} documents from {source}")
                except Exception as e:
                    logger.error(f"Error retrieving documents from {source}: {e}")
        
        # Index all chunks
        if all_chunks:
            self.index_documents(all_chunks)
            logger.info(f"Indexed {len(all_chunks)} chunks from {len(sources)} sources")
    
    def load_index(self, directory):
        """
        Load a saved retrieval index.
        
        Args:
            directory: Directory containing the index files
            
        Returns:
            bool: True if successful, False otherwise
        """
        return retrieval_engine.load_index(directory)
    
    def save_index(self, directory):
        """
        Save the current retrieval index.
        
        Args:
            directory: Directory to save the index files
        """
        retrieval_engine.save_index(directory)

# Global RAG processor instance
rag_processor = RAGProcessor()















"""
Confluence API client for Enterprise RAG System.
"""
import requests
import logging
import json
from html.parser import HTMLParser

import config
from utils.logger import get_logger
from utils.cache import cache_manager

logger = get_logger(__name__)

# Disable insecure request warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class HTMLFilter(HTMLParser):
    """Filter to extract text from HTML."""
    def __init__(self, data):
        super().__init__()
        self.text = []
        self.data = data + " "
    
    def handle_data(self, data):
        self.text.append(data + " ")

class ConfluenceClient:
    """
    Client for Confluence REST API operations with comprehensive error handling.
    """
    
    def __init__(self, base_url=None, username=None, api_token=None, ssl_verify=True):
        """
        Initialize the Confluence client with server and authentication details.
        
        Args:
            base_url: The base URL of the Confluence server
            username: Username for authentication
            api_token: API token for authentication
            ssl_verify: Whether to verify SSL certificates
        """
        self.base_url = base_url or config.CONFLUENCE_URL
        self.username = username or config.CONFLUENCE_USERNAME
        self.api_token = api_token or config.CONFLUENCE_TOKEN
        self.ssl_verify = ssl_verify
        
        # Configure headers
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized Confluence client for {self.base_url}")
    
    def test_connection(self):
        """
        Test the connection to Confluence server.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Testing connection to Confluence...")
            response = requests.get(
                f"{self.base_url}/rest/api/server-info",
                auth=(self.username, self.api_token),
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                server_info = response.json()
                logger.info(f"Connection to Confluence successful! Server version: {server_info.get('version', 'Unknown')}")
                return True
            else:
                logger.error(f"Failed to connect to Confluence. Check log for details.")
                if hasattr(response, 'text') and response.text:
                    logger.error(f"Status code: {response.status_code}")
                    logger.error(f"Response content: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_issue(self, issue_key, fields=None, expand=None):
        """
        Get a specific issue by its key.
        
        Args:
            issue_key: The Issue Key (e.g., DEMO-1)
            fields: Comma-separated list of field names to include
            expand: Comma-separated list of sections to expand
            
        Returns:
            dict: Issue data or None if not found/error
        """
        cache_key = f"confluence_issue:{issue_key}:{fields}:{expand}"
        cached_issue = cache_manager.get(cache_key)
        if cached_issue:
            return cached_issue
        
        logger.info(f"Fetching issue: {issue_key}")
        
        # Build the URL
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}"
        
        # Build query parameters
        params = {}
        if fields:
            params["fields"] = fields
        if expand:
            params["expand"] = expand
        
        try:
            response = requests.get(
                url,
                auth=(self.username, self.api_token),
                params=params,
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            issue = response.json()
            
            # Cache the result
            cache_manager.set(cache_key, issue, timeout=3600)  # 1 hour
            
            logger.info(f"Successfully retrieved issue: {issue_key}")
            return issue
        except requests.RequestException as e:
            logger.error(f"Failed to get issue {issue_key}: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return None
    
    def search_content(self, cql, max_results=50, start=0, expand=None, limit=50, fields=None):
        """
        Search for content using CQL (Confluence Query Language).
        
        Args:
            cql: Confluence Query Language string
            max_results: Maximum number of results to return
            start: Starting index for pagination
            expand: Sections to expand in the results
            limit: Maximum number of results per request
            fields: Fields to include in the results
            
        Returns:
            dict: Search results or None if error
        """
        cache_key = f"confluence_search:{hash(cql)}:{max_results}:{start}:{expand}:{limit}:{fields}"
        cached_results = cache_manager.get(cache_key)
        if cached_results:
            return cached_results
        
        logger.info(f"Searching content with CQL: {cql}")
        
        # Build the URL
        url = f"{self.base_url}/rest/api/content/search"
        
        # Build query parameters
        params = {
            "cql": cql,
            "limit": min(limit, 100),  # Ensure limit is not too high
            "start": start
        }
        
        if expand:
            params["expand"] = expand
        
        if fields:
            params["fields"] = fields
        
        try:
            response = requests.get(
                url,
                auth=(self.username, self.api_token),
                params=params,
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            search_results = response.json()
            
            # If max_results is larger than limit, we may need multiple requests
            results = search_results.get("results", [])
            total = search_results.get("size", 0)
            
            # Continue fetching if we need more results and there are more available
            next_start = start + len(results)
            while len(results) < max_results and next_start < total:
                # Update start parameter for next request
                params["start"] = next_start
                
                # Make the next request
                next_response = requests.get(
                    url,
                    auth=(self.username, self.api_token),
                    params=params,
                    headers=self.headers,
                    verify=self.ssl_verify
                )
                
                next_response.raise_for_status()
                next_results = next_response.json()
                
                # Add new results to the list
                new_results = next_results.get("results", [])
                results.extend(new_results)
                
                # Update next_start for potential next iteration
                next_start += len(new_results)
                
                # Break if we got fewer results than requested (end of results)
                if len(new_results) < params["limit"]:
                    break
            
            # Trim results to max_results
            if len(results) > max_results:
                results = results[:max_results]
            
            # Update the results in the search_results dictionary
            search_results["results"] = results
            search_results["size"] = len(results)
            
            # Cache the result
            cache_manager.set(cache_key, search_results, timeout=1800)  # 30 minutes
            
            logger.info(f"Search returned {len(results)} results")
            return search_results
        except requests.RequestException as e:
            logger.error(f"Failed to search content: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_content(self, content_id, expand=None, version=None):
        """
        Get a specific content by its ID.
        
        Args:
            content_id: The ID of the content to retrieve
            expand: Comma-separated list of properties to expand
            version: Version number to retrieve
            
        Returns:
            dict: Content data or None if not found/error
        """
        cache_key = f"confluence_content:{content_id}:{expand}:{version}"
        cached_content = cache_manager.get(cache_key)
        if cached_content:
            return cached_content
        
        logger.info(f"Fetching content: {content_id}")
        
        # Build the URL
        url = f"{self.base_url}/rest/api/content/{content_id}"
        
        # Build query parameters
        params = {}
        if expand:
            params["expand"] = expand
        if version:
            params["version"] = version
        
        try:
            response = requests.get(
                url,
                auth=(self.username, self.api_token),
                params=params,
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            content = response.json()
            
            # Cache the result
            cache_manager.set(cache_key, content, timeout=3600)  # 1 hour
            
            logger.info(f"Successfully retrieved content: {content_id}")
            return content
        except requests.RequestException as e:
            logger.error(f"Failed to get content {content_id}: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_page_content(self, page_id, expand="body.storage,metadata.labels"):
        """
        Get a page's content in a suitable format for RAG.
        This extracts and processes the content to be more suitable for embeddings.
        
        Args:
            page_id: The ID of the page
            expand: Properties to expand in the request
            
        Returns:
            dict: Processed page content with text and metadata
        """
        try:
            # Get the page content with expanded body
            page = self.get_content(page_id, expand=expand)
            if not page:
                return None
            
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "space": page.get("_expandable", {}).get("space", "").split("/")[-1] if "_expandable" in page and "space" in page["_expandable"] else ""
            }
            
            # Add labels as metadata if available
            if "metadata" in page and "labels" in page["metadata"]:
                labels = [label.get("name") for label in page["metadata"]["labels"].get("results", [])]
                metadata["labels"] = labels
            
            # Get raw text
            content = ""
            if "body" in page and "storage" in page["body"]:
                html_content = page["body"]["storage"]["value"]
                
                # Convert HTML to plain text using the HTMLFilter
                html_filter = HTMLFilter("")
                html_filter.feed(html_content)
                content = "".join(html_filter.text)
            
            # Add URL for reference
            page_url = f"{self.base_url}/display/{metadata['space']}/{page_id}"
            metadata["url"] = page_url
            
            return {
                "metadata": metadata,
                "content": content,
                "raw_html": page.get("body", {}).get("storage", {}).get("value", "")
            }
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None
    
    def get_all_content(self, content_type="page", limit=100, start=0, expand=None, space_key=None):
        """
        Get all content of a specific type with pagination handling.
        
        Args:
            content_type: Type of content to retrieve (default: page)
            limit: Maximum number of results per request
            start: Starting index for pagination
            expand: Properties to expand in the request
            space_key: Optional space key to filter content
            
        Returns:
            list: List of content items
        """
        cache_key = f"confluence_all_content:{content_type}:{limit}:{start}:{expand}:{space_key}"
        cached_content = cache_manager.get(cache_key)
        if cached_content:
            return cached_content
        
        logger.info(f"Retrieving all {content_type} content")
        
        # Build query parameters
        params = {
            "type": content_type,
            "limit": min(limit, 100),  # Ensure limit is not too high
            "start": start
        }
        
        if expand:
            params["expand"] = expand
        
        if space_key:
            params["spaceKey"] = space_key
        
        all_content = []
        
        try:
            # Build the URL
            url = f"{self.base_url}/rest/api/content"
            
            while True:
                response = requests.get(
                    url,
                    auth=(self.username, self.api_token),
                    params=params,
                    headers=self.headers,
                    verify=self.ssl_verify
                )
                
                response.raise_for_status()
                data = response.json()
                
                # Add results to the list
                results = data.get("results", [])
                all_content.extend(results)
                
                # Break if we've reached the end of the content
                if len(results) < params["limit"]:
                    break
                
                # Update start parameter for next request
                params["start"] = params["start"] + len(results)
            
            # Cache the result
            cache_manager.set(cache_key, all_content, timeout=1800)  # 30 minutes
            
            logger.info(f"Retrieved a total of {len(all_content)} {content_type} content items")
            return all_content
        except requests.RequestException as e:
            logger.error(f"Failed to get all content: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return all_content  # Return what we've got so far
    
    def get_spaces(self, limit=100, start=0, expand=None):
        """
        Get all spaces with pagination handling.
        
        Args:
            limit: Maximum number of results per request
            start: Starting index for pagination
            expand: Properties to expand in the request
            
        Returns:
            list: List of spaces
        """
        cache_key = f"confluence_spaces:{limit}:{start}:{expand}"
        cached_spaces = cache_manager.get(cache_key)
        if cached_spaces:
            return cached_spaces
        
        logger.info("Retrieving all spaces")
        
        # Build query parameters
        params = {
            "limit": min(limit, 100),  # Ensure limit is not too high
            "start": start
        }
        
        if expand:
            params["expand"] = expand
        
        all_spaces = []
        
        try:
            # Build the URL
            url = f"{self.base_url}/rest/api/space"
            
            while True:
                response = requests.get(
                    url,
                    auth=(self.username, self.api_token),
                    params=params,
                    headers=self.headers,
                    verify=self.ssl_verify
                )
                
                response.raise_for_status()
                data = response.json()
                
                # Add results to the list
                results = data.get("results", [])
                all_spaces.extend(results)
                
                # Break if we've reached the end of the spaces
                if len(results) < params["limit"]:
                    break
                
                # Update start parameter for next request
                params["start"] = params["start"] + len(results)
            
            # Cache the result
            cache_manager.set(cache_key, all_spaces, timeout=3600)  # 1 hour
            
            logger.info(f"Retrieved a total of {len(all_spaces)} spaces")
            return all_spaces
        except requests.RequestException as e:
            logger.error(f"Failed to get spaces: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return all_spaces  # Return what we've got so far















"""
Confluence connector for Enterprise RAG System.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import config
from utils.logger import get_logger
from utils.cache import cache_manager
from utils.content_parser import content_parser
from data_sources.confluence.client import ConfluenceClient

logger = get_logger(__name__)

class ConfluenceConnector:
    """
    Connector for retrieving and processing Confluence content for RAG.
    Handles content retrieval, processing, and caching.
    """
    
    def __init__(self, base_url=None, username=None, api_token=None, ssl_verify=None):
        """
        Initialize the Confluence connector.
        
        Args:
            base_url: Confluence base URL
            username: Confluence username
            api_token: Confluence API token
            ssl_verify: Whether to verify SSL certificates
        """
        self.client = ConfluenceClient(
            base_url=base_url,
            username=username,
            api_token=api_token,
            ssl_verify=ssl_verify if ssl_verify is not None else True
        )
        self.spaces_to_include = []
        self.spaces_to_exclude = []
        self.content_types = ["page"]
        
        logger.info("Initialized Confluence connector")
    
    def set_spaces(self, include_spaces=None, exclude_spaces=None):
        """
        Set spaces to include or exclude.
        
        Args:
            include_spaces: List of space keys to include
            exclude_spaces: List of space keys to exclude
        """
        if include_spaces:
            self.spaces_to_include = include_spaces
        
        if exclude_spaces:
            self.spaces_to_exclude = exclude_spaces
        
        logger.info(f"Updated space filters: include={self.spaces_to_include}, exclude={self.spaces_to_exclude}")
    
    def set_content_types(self, content_types):
        """
        Set content types to retrieve.
        
        Args:
            content_types: List of content types (e.g., page, blogpost)
        """
        self.content_types = content_types
        logger.info(f"Updated content types to retrieve: {content_types}")
    
    def test_connection(self):
        """
        Test the connection to Confluence.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        return self.client.test_connection()
    
    def get_spaces(self):
        """
        Get all available spaces.
        
        Returns:
            list: List of space objects
        """
        spaces = self.client.get_spaces(limit=100, expand="description")
        
        # Filter spaces if needed
        if self.spaces_to_include:
            spaces = [space for space in spaces if space.get("key") in self.spaces_to_include]
        
        if self.spaces_to_exclude:
            spaces = [space for space in spaces if space.get("key") not in self.spaces_to_exclude]
        
        return spaces
    
    def get_space_content(self, space_key, content_type="page", limit=100, expand="body.storage"):
        """
        Get all content from a specific space.
        
        Args:
            space_key: Space key
            content_type: Content type (e.g., page, blogpost)
            limit: Maximum number of results per request
            expand: Properties to expand in the response
            
        Returns:
            list: List of content objects
        """
        cache_key = f"confluence_space_content:{space_key}:{content_type}:{limit}:{expand}"
        cached_content = cache_manager.get(cache_key)
        if cached_content:
            return cached_content
        
        logger.info(f"Retrieving {content_type} content from space: {space_key}")
        
        content_items = self.client.get_all_content(
            content_type=content_type,
            limit=limit,
            expand=expand,
            space_key=space_key
        )
        
        # Cache the results
        cache_manager.set(cache_key, content_items, timeout=1800)  # 30 minutes
        
        return content_items
    
    def search_content(self, query, max_results=50, content_type=None, space_key=None):
        """
        Search for content using CQL.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            content_type: Content type filter
            space_key: Space key filter
            
        Returns:
            list: List of search results
        """
        # Build CQL query
        cql_parts = []
        
        # Add text query
        if query:
            # Escape special characters in query
            query = query.replace('"', '\\"')
            cql_parts.append(f'text ~ "{query}"')
        
        # Add content type filter
        if content_type:
            cql_parts.append(f'type = "{content_type}"')
        elif self.content_types:
            type_conditions = [f'type = "{ct}"' for ct in self.content_types]
            cql_parts.append(f"({' OR '.join(type_conditions)})")
        
        # Add space filter
        if space_key:
            cql_parts.append(f'space = "{space_key}"')
        elif self.spaces_to_include:
            space_conditions = [f'space = "{space}"' for space in self.spaces_to_include]
            cql_parts.append(f"({' OR '.join(space_conditions)})")
        
        # Combine all parts with AND
        cql = " AND ".join(cql_parts)
        
        # If cql is empty, get all content of the specified types
        if not cql and self.content_types:
            type_conditions = [f'type = "{ct}"' for ct in self.content_types]
            cql = f"({' OR '.join(type_conditions)})"
        
        # Search for content
        search_results = self.client.search_content(
            cql=cql,
            max_results=max_results,
            expand="body.storage",
            limit=25  # Limit per request
        )
        
        if not search_results or "results" not in search_results:
            return []
        
        return search_results["results"]
    
    def process_content_item(self, content_item):
        """
        Process a content item for RAG.
        
        Args:
            content_item: Confluence content item
            
        Returns:
            dict: Processed document for RAG
        """
        try:
            # Extract content ID and title
            content_id = content_item.get("id")
            title = content_item.get("title", "Untitled")
            
            # Get full content with expanded body if not already expanded
            if "body" not in content_item or "storage" not in content_item.get("body", {}):
                content_item = self.client.get_content(content_id, expand="body.storage,metadata.labels")
            
            # Skip if still no body
            if not content_item or "body" not in content_item or "storage" not in content_item.get("body", {}):
                logger.warning(f"No body content for {content_id}: {title}")
                return None
            
            # Extract HTML content
            html_content = content_item["body"]["storage"]["value"]
            
            # Parse content using content parser
            parsed_content = content_parser.parse_confluence_content(content_item)
            
            # Extract space key
            if "_expandable" in content_item and "space" in content_item["_expandable"]:
                space_path = content_item["_expandable"]["space"]
                space_key = space_path.split("/")[-1] if space_path else "Unknown"
            else:
                space_key = "Unknown"
            
            # Extract labels
            labels = []
            if "metadata" in content_item and "labels" in content_item["metadata"]:
                labels_data = content_item["metadata"]["labels"].get("results", [])
                labels = [label.get("name") for label in labels_data]
            
            # Create source link
            source_link = f"{self.client.base_url}/display/{space_key}/{content_id}"
            
            # Create document for RAG
            document = {
                "text": parsed_content["text"],
                "metadata": {
                    "source": title,
                    "source_type": "confluence",
                    "source_id": content_id,
                    "source_link": source_link,
                    "space_key": space_key,
                    "labels": labels,
                    "elements": parsed_content.get("elements", []),
                    "last_modified": content_item.get("version", {}).get("when"),
                    "creator": content_item.get("version", {}).get("by", {}).get("displayName")
                }
            }
            
            logger.debug(f"Processed Confluence content: {title} ({content_id})")
            return document
        except Exception as e:
            logger.error(f"Error processing content item: {str(e)}")
            return None
    
    def get_documents(self, max_documents=100, recent_days=None):
        """
        Get documents from Confluence for RAG.
        
        Args:
            max_documents: Maximum number of documents to retrieve
            recent_days: Only fetch documents updated in recent days
            
        Returns:
            list: List of processed documents for RAG
        """
        logger.info("Retrieving documents from Confluence")
        
        # Cache key for documents
        cache_key = f"confluence_documents:{max_documents}:{recent_days}:{','.join(self.spaces_to_include)}:{','.join(self.content_types)}"
        cached_documents = cache_manager.get(cache_key)
        if cached_documents:
            return cached_documents
        
        # Build CQL query for recent content if needed
        cql_parts = []
        
        # Add content type filter
        if self.content_types:
            type_conditions = [f'type = "{ct}"' for ct in self.content_types]
            cql_parts.append(f"({' OR '.join(type_conditions)})")
        
        # Add space filter
        if self.spaces_to_include:
            space_conditions = [f'space = "{space}"' for space in self.spaces_to_include]
            cql_parts.append(f"({' OR '.join(space_conditions)})")
        
        # Add date filter if recent_days is specified
        if recent_days:
            # Calculate the date threshold
            threshold_date = (datetime.now() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
            cql_parts.append(f'lastmodified >= "{threshold_date}"')
        
        # Combine all parts with AND
        cql = " AND ".join(cql_parts)
        
        # If no specific filters are defined, just order by last modified
        if not cql:
            cql = "order by lastmodified desc"
        else:
            cql += " order by lastmodified desc"
        
        # Search for content
        search_results = self.client.search_content(
            cql=cql,
            max_results=max_documents,
            expand="body.storage,metadata.labels",
            limit=25  # Limit per request
        )
        
        if not search_results or "results" not in search_results:
            logger.warning("No content found in Confluence")
            return []
        
        content_items = search_results["results"]
        logger.info(f"Found {len(content_items)} Confluence content items")
        
        # Process content items in parallel
        documents = []
        with ThreadPoolExecutor(max_workers=min(len(content_items), 10)) as executor:
            futures = [executor.submit(self.process_content_item, item) for item in content_items]
            
            for future in futures:
                document = future.result()
                if document:
                    documents.append(document)
        
        logger.info(f"Processed {len(documents)} documents from Confluence")
        
        # Cache the results
        cache_manager.set(cache_key, documents, timeout=1800)  # 30 minutes
        
        return documents

# Global Confluence connector instance
confluence_connector = ConfluenceConnector(ssl_verify=False)






















"""
Confluence integration package for Enterprise RAG System.
"""
from data_sources.confluence.client import ConfluenceClient
from data_sources.confluence.connector import ConfluenceConnector
















"""
JIRA API client for Enterprise RAG System.
"""
import requests
import logging
import json
from html.parser import HTMLParser

import config
from utils.logger import get_logger
from utils.cache import cache_manager

logger = get_logger(__name__)

# Disable insecure request warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class HTMLFilter(HTMLParser):
    """Filter to extract text from HTML."""
    def __init__(self, data):
        super().__init__()
        self.text = []
        self.data = data + " "
    
    def handle_data(self, data):
        self.text.append(data + " ")

class JiraClient:
    """
    Client for JIRA REST API operations with comprehensive error handling.
    """
    
    def __init__(self, base_url=None, username=None, api_token=None, ssl_verify=True):
        """
        Initialize the JIRA client with server and authentication details.
        
        Args:
            base_url: The base URL of the JIRA server
            username: Username for authentication
            api_token: API token for authentication
            ssl_verify: Whether to verify SSL certificates
        """
        self.base_url = base_url or config.JIRA_URL
        self.username = username or config.JIRA_USERNAME
        self.api_token = api_token or config.JIRA_TOKEN
        self.ssl_verify = ssl_verify
        
        # Configure headers
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized JIRA client for {self.base_url}")
    
    def test_connection(self):
        """
        Test the connection to JIRA server.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Testing connection to JIRA...")
            response = requests.get(
                f"{self.base_url}/rest/api/2/serverInfo",
                auth=(self.username, self.api_token),
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                server_info = response.json()
                logger.info(f"Connection to JIRA successful! Server version: {server_info.get('version', 'Unknown')}")
                return True
            else:
                logger.error(f"Failed to connect to JIRA. Check log for details.")
                if hasattr(response, 'text') and response.text:
                    logger.error(f"Status code: {response.status_code}")
                    logger.error(f"Response content: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_issue(self, issue_key, fields=None, expand=None):
        """
        Get a specific issue by its key.
        
        Args:
            issue_key: The Issue Key (e.g., DEMO-1)
            fields: Comma-separated list of field names to include
            expand: Comma-separated list of sections to expand
            
        Returns:
            dict: Issue data or None if not found/error
        """
        cache_key = f"jira_issue:{issue_key}:{fields}:{expand}"
        cached_issue = cache_manager.get(cache_key)
        if cached_issue:
            return cached_issue
        
        logger.info(f"Fetching issue: {issue_key}")
        
        # Build the URL
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}"
        
        # Build query parameters
        params = {}
        if fields:
            params["fields"] = fields
        if expand:
            params["expand"] = expand
        
        try:
            response = requests.get(
                url,
                auth=(self.username, self.api_token),
                params=params,
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            issue = response.json()
            
            # Cache the result
            cache_manager.set(cache_key, issue, timeout=3600)  # 1 hour
            
            logger.info(f"Successfully retrieved issue: {issue_key}")
            return issue
        except requests.RequestException as e:
            logger.error(f"Failed to get issue {issue_key}: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return None
    
    def search_issues(self, jql, max_results=50, start_at=0, fields=None, expand=None):
        """
        Search for issues using JQL (JIRA Query Language).
        
        Args:
            jql: JIRA Query Language string
            max_results: Maximum number of results to return
            start_at: Starting index for pagination
            fields: Fields to include in the response
            expand: Sections to expand in the response
            
        Returns:
            dict: Search results or None if error
        """
        cache_key = f"jira_search:{hash(jql)}:{max_results}:{start_at}:{fields}:{expand}"
        cached_results = cache_manager.get(cache_key)
        if cached_results:
            return cached_results
        
        logger.info(f"Searching issues with JQL: {jql}")
        
        # Build the URL
        url = f"{self.base_url}/rest/api/2/search"
        
        # Build request payload
        payload = {
            "jql": jql,
            "maxResults": min(max_results, 100),  # Ensure max_results is not too high
            "startAt": start_at
        }
        
        if fields:
            payload["fields"] = fields.split(",") if isinstance(fields, str) else fields
        
        if expand:
            payload["expand"] = expand
        
        try:
            response = requests.post(
                url,
                auth=(self.username, self.api_token),
                json=payload,
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            search_results = response.json()
            
            # If total is greater than max_results and more than what we've got,
            # we need to make additional requests
            total = search_results.get("total", 0)
            issues = search_results.get("issues", [])
            
            # Continue fetching if we need more issues
            current_count = len(issues)
            next_start_at = start_at + current_count
            
            while current_count < max_results and next_start_at < total:
                # Update startAt for next request
                payload["startAt"] = next_start_at
                
                # Make the next request
                next_response = requests.post(
                    url,
                    auth=(self.username, self.api_token),
                    json=payload,
                    headers=self.headers,
                    verify=self.ssl_verify
                )
                
                next_response.raise_for_status()
                next_results = next_response.json()
                next_issues = next_results.get("issues", [])
                
                # Add new issues to the list
                issues.extend(next_issues)
                
                # Break if we got fewer issues than requested
                if len(next_issues) == 0:
                    break
                
                # Update for next iteration
                current_count = len(issues)
                next_start_at += len(next_issues)
            
            # Ensure we don't exceed max_results
            if len(issues) > max_results:
                issues = issues[:max_results]
            
            # Update issues in search_results
            search_results["issues"] = issues
            
            # Cache the results
            cache_manager.set(cache_key, search_results, timeout=1800)  # 30 minutes
            
            logger.info(f"Search returned {len(issues)} issues")
            return search_results
        except requests.RequestException as e:
            logger.error(f"Failed to search issues: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_projects(self):
        """
        Get all projects.
        
        Returns:
            list: List of projects or empty list if error
        """
        cache_key = "jira_projects"
        cached_projects = cache_manager.get(cache_key)
        if cached_projects:
            return cached_projects
        
        logger.info("Fetching projects...")
        
        # Build the URL
        url = f"{self.base_url}/rest/api/2/project"
        
        try:
            response = requests.get(
                url,
                auth=(self.username, self.api_token),
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            projects = response.json()
            
            # Cache the result
            cache_manager.set(cache_key, projects, timeout=3600)  # 1 hour
            
            logger.info(f"Successfully retrieved {len(projects)} projects")
            return projects
        except requests.RequestException as e:
            logger.error(f"Failed to get projects: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return []
    
    def get_issue_types(self):
        """
        Get all issue types.
        
        Returns:
            list: List of issue types or empty list if error
        """
        cache_key = "jira_issue_types"
        cached_types = cache_manager.get(cache_key)
        if cached_types:
            return cached_types
        
        logger.info("Fetching issue types...")
        
        # Build the URL
        url = f"{self.base_url}/rest/api/2/issuetype"
        
        try:
            response = requests.get(
                url,
                auth=(self.username, self.api_token),
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            issue_types = response.json()
            
            # Cache the result
            cache_manager.set(cache_key, issue_types, timeout=3600)  # 1 hour
            
            logger.info(f"Successfully retrieved {len(issue_types)} issue types")
            return issue_types
        except requests.RequestException as e:
            logger.error(f"Failed to get issue types: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return []
    
    def create_issue(self, project_key, issue_type, summary, description, fields=None):
        """
        Create a new issue.
        
        Args:
            project_key: The project key
            issue_type: The issue type name or ID
            summary: The issue summary
            description: The issue description
            fields: Dictionary of additional fields to set
            
        Returns:
            dict: Created issue data or None if error
        """
        logger.info(f"Creating issue in project {project_key} of type {issue_type}")
        
        # Build the URL
        url = f"{self.base_url}/rest/api/2/issue"
        
        # Build request payload
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
            for field, value in fields.items():
                issue_data["fields"][field] = value
        
        try:
            response = requests.post(
                url,
                auth=(self.username, self.api_token),
                json=issue_data,
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            created_issue = response.json()
            
            logger.info(f"Successfully created issue: {created_issue.get('key')}")
            return created_issue
        except requests.RequestException as e:
            logger.error(f"Failed to create issue: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_issue_content(self, issue_key):
        """
        Get an issue's content in a suitable format for RAG.
        
        Args:
            issue_key: The Issue Key
            
        Returns:
            dict: Processed issue content with text and metadata
        """
        try:
            # Get the issue with all relevant fields
            fields_to_include = "summary,description,status,assignee,reporter,priority,labels,components,fixVersions,resolution,comment"
            issue = self.get_issue(issue_key, fields=fields_to_include)
            if not issue:
                return None
            
            # Extract metadata
            metadata = {
                "key": issue.get("key"),
                "id": issue.get("id"),
                "type": issue.get("fields", {}).get("issuetype", {}).get("name")
            }
            
            # Add fields as metadata
            fields = issue.get("fields", {})
            if "status" in fields:
                metadata["status"] = fields["status"].get("name")
            
            if "priority" in fields:
                metadata["priority"] = fields["priority"].get("name")
            
            if "assignee" in fields and fields["assignee"]:
                metadata["assignee"] = fields["assignee"].get("displayName")
            
            if "reporter" in fields and fields["reporter"]:
                metadata["reporter"] = fields["reporter"].get("displayName")
            
            if "labels" in fields:
                metadata["labels"] = fields["labels"]
            
            if "created" in fields:
                metadata["created"] = fields["created"]
            
            if "updated" in fields:
                metadata["updated"] = fields["updated"]
            
            # Build the content
            content_parts = []
            
            if "summary" in fields:
                content_parts.append(f"Summary: {fields['summary']}")
            
            if "description" in fields and fields["description"]:
                content_parts.append(f"Description: {fields['description']}")
            
            # Add comments if available
            if "comment" in fields and "comments" in fields["comment"]:
                comments = fields["comment"]["comments"]
                for comment in comments:
                    author = comment.get("author", {}).get("displayName", "Unknown")
                    created = comment.get("created", "")
                    body = comment.get("body", "")
                    content_parts.append(f"Comment by {author} on {created}: {body}")
            
            # Join all parts with newlines
            content = "\n\n".join(content_parts)
            
            # Add URL for reference
            issue_url = f"{self.base_url}/browse/{issue_key}"
            metadata["url"] = issue_url
            
            return {
                "metadata": metadata,
                "content": content
            }
        except Exception as e:
            logger.error(f"Error processing issue content: {str(e)}")
            return None



















"""
JIRA connector for Enterprise RAG System.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import config
from utils.logger import get_logger
from utils.cache import cache_manager
from utils.content_parser import content_parser
from data_sources.jira.client import JiraClient

logger = get_logger(__name__)

class JiraConnector:
    """
    Connector for retrieving and processing JIRA content for RAG.
    Handles issue retrieval, processing, and caching.
    """
    
    def __init__(self, base_url=None, username=None, api_token=None, ssl_verify=None):
        """
        Initialize the JIRA connector.
        
        Args:
            base_url: JIRA base URL
            username: JIRA username
            api_token: JIRA API token
            ssl_verify: Whether to verify SSL certificates
        """
        self.client = JiraClient(
            base_url=base_url,
            username=username,
            api_token=api_token,
            ssl_verify=ssl_verify if ssl_verify is not None else True
        )
        self.projects_to_include = []
        self.issue_types_to_include = []
        self.statuses_to_include = []
        
        logger.info("Initialized JIRA connector")
    
    def set_filters(self, projects=None, issue_types=None, statuses=None):
        """
        Set filters for issue retrieval.
        
        Args:
            projects: List of project keys to include
            issue_types: List of issue types to include
            statuses: List of statuses to include
        """
        if projects:
            self.projects_to_include = projects
        
        if issue_types:
            self.issue_types_to_include = issue_types
        
        if statuses:
            self.statuses_to_include = statuses
        
        logger.info(f"Updated JIRA filters: projects={self.projects_to_include}, " +
                   f"issue_types={self.issue_types_to_include}, statuses={self.statuses_to_include}")
    
    def test_connection(self):
        """
        Test the connection to JIRA.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        return self.client.test_connection()
    
    def get_projects(self):
        """
        Get all available projects.
        
        Returns:
            list: List of project objects
        """
        projects = self.client.get_projects()
        
        # Filter projects if needed
        if self.projects_to_include:
            projects = [project for project in projects if project.get("key") in self.projects_to_include]
        
        return projects
    
    def get_issue_types(self):
        """
        Get all available issue types.
        
        Returns:
            list: List of issue type objects
        """
        issue_types = self.client.get_issue_types()
        
        # Filter issue types if needed
        if self.issue_types_to_include:
            issue_types = [issue_type for issue_type in issue_types 
                          if issue_type.get("name") in self.issue_types_to_include]
        
        return issue_types
    
    def search_issues(self, jql=None, max_results=50):
        """
        Search for issues using JQL.
        
        Args:
            jql: JQL query string
            max_results: Maximum number of results
            
        Returns:
            list: List of issues
        """
        # Build JQL query if not provided
        if not jql:
            jql_parts = []
            
            # Add project filter
            if self.projects_to_include:
                project_conditions = [f'project = "{project}"' for project in self.projects_to_include]
                jql_parts.append(f"({' OR '.join(project_conditions)})")
            
            # Add issue type filter
            if self.issue_types_to_include:
                type_conditions = [f'issuetype = "{issue_type}"' for issue_type in self.issue_types_to_include]
                jql_parts.append(f"({' OR '.join(type_conditions)})")
            
            # Add status filter
            if self.statuses_to_include:
                status_conditions = [f'status = "{status}"' for status in self.statuses_to_include]
                jql_parts.append(f"({' OR '.join(status_conditions)})")
            
            # Combine all parts with AND
            jql = " AND ".join(jql_parts)
            
            # Add order by clause
            if jql:
                jql += " ORDER BY updated DESC"
            else:
                jql = "ORDER BY updated DESC"
        
        # Get relevant fields
        fields = "summary,description,status,assignee,reporter,priority,labels,created,updated"
        
        # Search for issues
        search_results = self.client.search_issues(
            jql=jql,
            max_results=max_results,
            fields=fields
        )
        
        if not search_results or "issues" not in search_results:
            return []
        
        return search_results["issues"]
    
    def process_issue(self, issue):
        """
        Process an issue for RAG.
        
        Args:
            issue: JIRA issue object
            
        Returns:
            dict: Processed document for RAG
        """
        try:
            # Extract issue key
            issue_key = issue.get("key")
            
            # Get full issue content
            issue_content = self.client.get_issue_content(issue_key)
            if not issue_content:
                logger.warning(f"No content for issue: {issue_key}")
                return None
            
            # Extract fields
            fields = issue.get("fields", {})
            summary = fields.get("summary", "Untitled")
            
            # Create source link
            source_link = f"{self.client.base_url}/browse/{issue_key}"
            
            # Create document for RAG
            document = {
                "text": issue_content["content"],
                "metadata": {
                    "source": f"{issue_key}: {summary}",
                    "source_type": "jira",
                    "source_id": issue_key,
                    "source_link": source_link,
                    "project": issue_key.split("-")[0] if "-" in issue_key else "",
                    "status": fields.get("status", {}).get("name", "Unknown"),
                    "assignee": fields.get("assignee", {}).get("displayName", "Unassigned") if fields.get("assignee") else "Unassigned",
                    "created": fields.get("created"),
                    "updated": fields.get("updated"),
                    "priority": fields.get("priority", {}).get("name") if fields.get("priority") else None,
                    "issue_type": fields.get("issuetype", {}).get("name") if fields.get("issuetype") else None,
                    "labels": fields.get("labels", [])
                }
            }
            
            logger.debug(f"Processed JIRA issue: {issue_key}")
            return document
        except Exception as e:
            logger.error(f"Error processing issue: {str(e)}")
            return None
    
    def get_documents(self, max_documents=100, recent_days=None):
        """
        Get documents from JIRA for RAG.
        
        Args:
            max_documents: Maximum number of documents to retrieve
            recent_days: Only fetch documents updated in recent days
            
        Returns:
            list: List of processed documents for RAG
        """
        logger.info("Retrieving documents from JIRA")
        
        # Cache key for documents
        cache_key = f"jira_documents:{max_documents}:{recent_days}:{','.join(self.projects_to_include)}:{','.join(self.issue_types_to_include)}"
        cached_documents = cache_manager.get(cache_key)
        if cached_documents:
            return cached_documents
        
        # Build JQL query
        jql_parts = []
        
        # Add project filter
        if self.projects_to_include:
            project_conditions = [f'project = "{project}"' for project in self.projects_to_include]
            jql_parts.append(f"({' OR '.join(project_conditions)})")
        
        # Add issue type filter
        if self.issue_types_to_include:
            type_conditions = [f'issuetype = "{issue_type}"' for issue_type in self.issue_types_to_include]
            jql_parts.append(f"({' OR '.join(type_conditions)})")
        
        # Add status filter
        if self.statuses_to_include:
            status_conditions = [f'status = "{status}"' for status in self.statuses_to_include]
            jql_parts.append(f"({' OR '.join(status_conditions)})")
        
        # Add date filter if recent_days is specified
        if recent_days:
            # Calculate the date threshold
            threshold_date = (datetime.now() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
            jql_parts.append(f'updated >= "{threshold_date}"')
        
        # Combine all parts with AND
        jql = " AND ".join(jql_parts)
        
        # Add order by clause
        if jql:
            jql += " ORDER BY updated DESC"
        else:
            jql = "ORDER BY updated DESC"
        
        # Search for issues
        issues = self.search_issues(jql=jql, max_results=max_documents)
        
        if not issues:
            logger.warning("No issues found in JIRA")
            return []
        
        logger.info(f"Found {len(issues)} JIRA issues")
        
        # Process issues in parallel
        documents = []
        with ThreadPoolExecutor(max_workers=min(len(issues), 10)) as executor:
            futures = [executor.submit(self.process_issue, issue) for issue in issues]
            
            for future in futures:
                document = future.result()
                if document:
                    documents.append(document)
        
        logger.info(f"Processed {len(documents)} documents from JIRA")
        
        # Cache the results
        cache_manager.set(cache_key, documents, timeout=1800)  # 30 minutes
        
        return documents

# Global JIRA connector instance
jira_connector = JiraConnector(ssl_verify=False)




















"""
JIRA integration package for Enterprise RAG System.
"""
from data_sources.jira.client import JiraClient
from data_sources.jira.connector import JiraConnector













"""
Remedy API client for Enterprise RAG System.
"""
import os
import requests
import json
import logging
import time
import getpass
from urllib.parse import quote
from datetime import datetime, timedelta

import config
from utils.logger import get_logger
from utils.cache import cache_manager

logger = get_logger(__name__)

# Disable insecure request warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class RemedyClient:
    """
    Client for BMC Remedy REST API operations with comprehensive error handling.
    """
    
    def __init__(self, server_url=None, username=None, password=None, ssl_verify=None):
        """
        Initialize the Remedy client with server and authentication details.
        
        Args:
            server_url: The base URL of the Remedy server
            username: Username for authentication
            password: Password for authentication
            ssl_verify: Whether to verify SSL certificates
        """
        self.server_url = server_url or config.REMEDY_URL
        self.server_url = self.server_url.rstrip('/')
        self.username = username or config.REMEDY_USERNAME
        self.password = password or config.REMEDY_PASSWORD
        
        # Handle SSL verification
        if ssl_verify is None:
            self.ssl_verify = config.REMEDY_SSL_VERIFY
        else:
            self.ssl_verify = ssl_verify
            
        # Authentication token
        self.token = None
        self.token_type = "AR-JWT"
        
        logger.info(f"Initialized Remedy client for {self.server_url}")
    
    def login(self):
        """
        Log in to Remedy and get authentication token.
        
        Returns:
            tuple: (returnVal, token) where returnVal is 1 on success, -1 on failure
        """
        # Prompt for credentials if not provided
        if not self.username:
            self.username = input("Enter Remedy Username: ")
        if not self.password:
            self.password = getpass.getpass(prompt="Enter Remedy Password: ")
        
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
    
    def logout(self):
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
    
    def _ensure_login(self):
        """
        Ensure that the client is logged in.
        
        Returns:
            bool: True if login successful, False otherwise
        """
        if self.token:
            return True
        
        status, _ = self.login()
        return status == 1
    
    def get_incident(self, incident_id):
        """
        Get a specific incident by its ID.
        
        Args:
            incident_id: The Incident Number (e.g., INC000001482087)
            
        Returns:
            dict: Incident data or None if not found/error
        """
        if not self._ensure_login():
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
    
    def get_incidents_by_date(self, date, status=None, owner_group=None):
        """
        Get all incidents submitted on a specific date.
        
        Args:
            date: The submission date in YYYY-MM-DD format
            status: Optional status filter (e.g., "Closed", "Open")
            owner_group: Optional owner group filter
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self._ensure_login():
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
    
    def get_incidents_by_status(self, status, limit=100):
        """
        Get incidents by their status.
        
        Args:
            status: The status to filter by (e.g., "Open", "Closed", "Resolved")
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self._ensure_login():
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
    
    def get_incidents_by_assignee(self, assignee, limit=100):
        """
        Get incidents assigned to a specific person.
        
        Args:
            assignee: The assignee name
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self._ensure_login():
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
    
    def query_form(self, form_name, qualified_query=None, fields=None, limit=100):
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
        if not self._ensure_login():
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
                logger.error(f"Response: {r.text}")
                return None
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return None
    
    def create_incident(self, summary, description, impact="4-Minor/Localized", urgency="4-Low",
                       reported_source="Direct Input", service_type="User Service Restoration",
                       assigned_group=None):
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
        if not self._ensure_login():
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
    
    def update_incident(self, incident_id, update_data):
        """
        Update an existing incident.
        
        Args:
            incident_id: The Incident Number to update
            update_data: Dictionary of fields to update
            
        Returns:
            bool: True on success, False on failure
        """
        if not self._ensure_login():
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
    
    def get_incident_history(self, incident_id):
        """
        Get the history of changes for a specific incident.
        
        Args:
            incident_id: The Incident Number
            
        Returns:
            list: History entries or empty list if none found/error
        """
        if not self._ensure_login():
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
    
    def process_incident_for_rag(self, incident):
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















"""
Remedy connector for Enterprise RAG System.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import config
from utils.logger import get_logger
from utils.cache import cache_manager
from utils.content_parser import content_parser
from data_sources.remedy.client import RemedyClient

logger = get_logger(__name__)

class RemedyConnector:
    """
    Connector for retrieving and processing Remedy content for RAG.
    Handles incident retrieval, processing, and caching.
    """
    
    def __init__(self, server_url=None, username=None, password=None, ssl_verify=None):
        """
        Initialize the Remedy connector.
        
        Args:
            server_url: Remedy server URL
            username: Remedy username
            password: Remedy password
            ssl_verify: Whether to verify SSL certificates
        """
        self.client = RemedyClient(
            server_url=server_url,
            username=username,
            password=password,
            ssl_verify=ssl_verify if ssl_verify is not None else False
        )
        self.statuses_to_include = []
        self.owner_groups_to_include = []
        self.impact_levels_to_include = []
        
        logger.info("Initialized Remedy connector")
    
    def set_filters(self, statuses=None, owner_groups=None, impact_levels=None):
        """
        Set filters for incident retrieval.
        
        Args:
            statuses: List of statuses to include
            owner_groups: List of owner groups to include
            impact_levels: List of impact levels to include
        """
        if statuses:
            self.statuses_to_include = statuses
        
        if owner_groups:
            self.owner_groups_to_include = owner_groups
        
        if impact_levels:
            self.impact_levels_to_include = impact_levels
        
        logger.info(f"Updated Remedy filters: statuses={self.statuses_to_include}, " +
                   f"owner_groups={self.owner_groups_to_include}, impact_levels={self.impact_levels_to_include}")
    
    def test_connection(self):
        """
        Test the connection to Remedy.
        
        Returns:
            bool: True if login successful, False otherwise
        """
        status, _ = self.client.login()
        if status == 1:
            self.client.logout()
            return True
        return False
    
    def get_incident(self, incident_id):
        """
        Get a specific incident by ID.
        
        Args:
            incident_id: Incident number (e.g., INC000001234)
            
        Returns:
            dict: Processed incident for RAG or None if not found
        """
        # Get raw incident data
        incident = self.client.get_incident(incident_id)
        if not incident:
            return None
        
        # Process incident for RAG
        processed_incident = self.client.process_incident_for_rag(incident)
        
        return processed_incident
    
    def search_incidents(self, query=None, status=None, owner_group=None, max_results=50):
        """
        Search for incidents using various criteria.
        
        Args:
            query: Text query to search in incident fields
            status: Status filter
            owner_group: Owner group filter
            max_results: Maximum number of results
            
        Returns:
            list: List of incidents
        """
        # Build the qualified query
        query_parts = []
        
        # Add text search if provided
        if query:
            # Simple implementation - search in Summary and Description
            query_parts.append(f"('Summary' LIKE \"%{query}%\" OR 'Description' LIKE \"%{query}%\")")
        
        # Add status filter
        if status:
            query_parts.append(f"'Status'=\"{status}\"")
        elif self.statuses_to_include:
            status_conditions = [f"'Status'=\"{status}\"" for status in self.statuses_to_include]
            query_parts.append(f"({' OR '.join(status_conditions)})")
        
        # Add owner group filter
        if owner_group:
            query_parts.append(f"'Owner Group'=\"{owner_group}\"")
        elif self.owner_groups_to_include:
            group_conditions = [f"'Owner Group'=\"{group}\"" for group in self.owner_groups_to_include]
            query_parts.append(f"({' OR '.join(group_conditions)})")
        
        # Add impact filter
        if self.impact_levels_to_include:
            impact_conditions = [f"'Impact'=\"{impact}\"" for impact in self.impact_levels_to_include]
            query_parts.append(f"({' OR '.join(impact_conditions)})")
        
        # Combine all parts with AND
        qualified_query = " AND ".join(query_parts) if query_parts else None
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Query the form
        result = self.client.query_form("HPD:Help Desk", qualified_query, fields, limit=max_results)
        
        if not result or "entries" not in result:
            return []
        
        return result["entries"]
    
    def process_incidents(self, incidents):
        """
        Process a list of incidents for RAG.
        
        Args:
            incidents: List of raw incidents
            
        Returns:
            list: List of processed incidents for RAG
        """
        processed_incidents = []
        
        for incident in incidents:
            processed = self.client.process_incident_for_rag(incident)
            if processed:
                processed_incidents.append(processed)
        
        return processed_incidents
    
    def get_documents(self, max_documents=100, recent_days=None):
        """
        Get documents from Remedy for RAG.
        
        Args:
            max_documents: Maximum number of documents to retrieve
            recent_days: Only fetch documents updated in recent days
            
        Returns:
            list: List of processed documents for RAG
        """
        logger.info("Retrieving documents from Remedy")
        
        # Cache key for documents
        cache_key = f"remedy_documents:{max_documents}:{recent_days}:{','.join(self.statuses_to_include)}:{','.join(self.owner_groups_to_include)}"
        cached_documents = cache_manager.get(cache_key)
        if cached_documents:
            return cached_documents
        
        # Login to Remedy
        login_status, _ = self.client.login()
        if login_status != 1:
            logger.error("Failed to login to Remedy")
            return []
        
        try:
            # Build query for incidents
            query_parts = []
            
            # Add status filter
            if self.statuses_to_include:
                status_conditions = [f"'Status'=\"{status}\"" for status in self.statuses_to_include]
                query_parts.append(f"({' OR '.join(status_conditions)})")
            
            # Add owner group filter
            if self.owner_groups_to_include:
                group_conditions = [f"'Owner Group'=\"{group}\"" for group in self.owner_groups_to_include]
                query_parts.append(f"({' OR '.join(group_conditions)})")
            
            # Add impact filter
            if self.impact_levels_to_include:
                impact_conditions = [f"'Impact'=\"{impact}\"" for impact in self.impact_levels_to_include]
                query_parts.append(f"({' OR '.join(impact_conditions)})")
            
            # Add date filter if recent_days is specified
            if recent_days:
                # Calculate the date threshold
                threshold_date = (datetime.now() - timedelta(days=recent_days)).strftime("%Y-%m-%d %H:%M:%S.000")
                query_parts.append(f"'Submit Date' >= \"{threshold_date}\"")
            
            # Combine all parts with AND
            qualified_query = " AND ".join(query_parts) if query_parts else None
            
            # Fields to retrieve
            fields = [
                "Assignee", "Incident Number", "Description", "Status", "Owner",
                "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
                "Priority", "Environment", "Summary", "Support Group Name",
                "Request Assignee", "Work Order ID", "Request Manager"
            ]
            
            # Query for incidents
            result = self.client.query_form("HPD:Help Desk", qualified_query, fields, limit=max_documents)
            
            if not result or "entries" not in result:
                logger.warning("No incidents found in Remedy")
                return []
            
            incidents = result["entries"]
            logger.info(f"Found {len(incidents)} Remedy incidents")
            
            # Process incidents in parallel
            documents = []
            with ThreadPoolExecutor(max_workers=min(len(incidents), 10)) as executor:
                futures = [executor.submit(self.client.process_incident_for_rag, incident) for incident in incidents]
                
                for future in futures:
                    document = future.result()
                    if document:
                        # Format document for RAG
                        rag_document = {
                            "text": document["content"],
                            "metadata": {
                                "source": f"Incident {document['metadata']['incident_number']}",
                                "source_type": "remedy",
                                "source_id": document['metadata']['incident_number'],
                                "source_link": f"{self.client.server_url}/arsys/forms/onbmc-s/HPD%3AHelp+Desk/Default+Administrator+View/?mode=search&F304255500={document['metadata']['incident_number']}",
                                "status": document['metadata']['status'],
                                "priority": document['metadata']['priority'],
                                "impact": document['metadata']['impact'],
                                "assignee": document['metadata']['assignee'],
                                "owner_group": document['metadata']['owner_group'],
                                "submit_date": document['metadata']['submit_date']
                            }
                        }
                        documents.append(rag_document)
            
            logger.info(f"Processed {len(documents)} documents from Remedy")
            
            # Cache the results
            cache_manager.set(cache_key, documents, timeout=1800)  # 30 minutes
            
            return documents
        finally:
            # Always logout
            self.client.logout()

# Global Remedy connector instance
remedy_connector = RemedyConnector(ssl_verify=False)






















"""
Remedy integration package for Enterprise RAG System.
"""
from data_sources.remedy.client import RemedyClient
from data_sources.remedy.connector import RemedyConnector



















"""
Data sources package for Enterprise RAG System.
"""
from data_sources.confluence import ConfluenceConnector, ConfluenceClient
from data_sources.jira import JiraConnector, JiraClient
from data_sources.remedy import RemedyConnector, RemedyClient












"""
API routes for Enterprise RAG System.
"""
import json
from flask import Blueprint, request, jsonify, Response, stream_with_context

import config
from utils.logger import get_logger
from rag_engine.processor import rag_processor
from data_sources.confluence import confluence_connector
from data_sources.jira import jira_connector
from data_sources.remedy import remedy_connector

logger = get_logger(__name__)

# API Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

def register_api_routes(app):
    """
    Register API routes with the Flask app.
    
    Args:
        app: Flask application instance
    """
    # Register the blueprint
    app.register_blueprint(api_bp)
    
    # Register data sources with the RAG processor
    rag_processor.register_document_source('confluence', confluence_connector)
    rag_processor.register_document_source('jira', jira_connector)
    rag_processor.register_document_source('remedy', remedy_connector)
    
    logger.info("API routes registered")

# Health check endpoint
@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0'
    })

# Chat endpoint for question answering
@api_bp.route('/chat', methods=['POST'])
def chat():
    """
    Process a chat query and generate a response.
    """
    try:
        data = request.json
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Invalid request. Missing query parameter.'
            }), 400
        
        query = data.get('query')
        sources = data.get('sources', None)  # Optional sources parameter
        search_type = data.get('search_type', 'hybrid')  # Default to hybrid search
        stream = data.get('stream', False)  # Stream parameter
        
        logger.info(f"Processing chat query: {query}")
        
        # Process the query through the RAG pipeline
        if stream:
            # Stream the response
            def generate_streaming_response():
                result = rag_processor.process_query(
                    query=query,
                    sources=sources,
                    search_type=search_type,
                    stream=True
                )
                
                # Stream response chunks
                for chunk in result['response']:
                    yield chunk
            
            return Response(
                stream_with_context(generate_streaming_response()),
                content_type='text/plain'
            )
        else:
            # Process query and get response
            result = rag_processor.process_query(
                query=query,
                sources=sources,
                search_type=search_type
            )
            
            # Return the response with context
            return jsonify({
                'response': result['response'],
                'context': result['context']
            })
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}")
        return jsonify({
            'error': f"An error occurred: {str(e)}"
        }), 500

# Sources endpoint to get available data sources
@api_bp.route('/sources', methods=['GET'])
def get_sources():
    """
    Get available data sources and their status.
    """
    try:
        sources = {
            'confluence': {
                'name': 'Confluence',
                'available': confluence_connector.test_connection()
            },
            'jira': {
                'name': 'JIRA',
                'available': jira_connector.test_connection()
            },
            'remedy': {
                'name': 'Remedy',
                'available': remedy_connector.test_connection()
            }
        }
        
        return jsonify({
            'sources': sources
        })
    except Exception as e:
        logger.error(f"Error getting sources: {str(e)}")
        return jsonify({
            'error': f"An error occurred: {str(e)}"
        }), 500

# Index endpoint to trigger document indexing
@api_bp.route('/index', methods=['POST'])
def index_documents():
    """
    Trigger document indexing for selected sources.
    """
    try:
        data = request.json
        sources = data.get('sources', None)
        max_documents = data.get('max_documents', 100)
        recent_days = data.get('recent_days', None)
        
        logger.info(f"Indexing documents from sources: {sources}")
        
        # Use all sources if none specified
        if not sources:
            sources = ['confluence', 'jira', 'remedy']
        
        # Collect all documents
        all_documents = []
        
        # Process each source
        for source in sources:
            if source == 'confluence':
                documents = confluence_connector.get_documents(
                    max_documents=max_documents,
                    recent_days=recent_days
                )
                all_documents.extend(documents)
            elif source == 'jira':
                documents = jira_connector.get_documents(
                    max_documents=max_documents,
                    recent_days=recent_days
                )
                all_documents.extend(documents)
            elif source == 'remedy':
                documents = remedy_connector.get_documents(
                    max_documents=max_documents,
                    recent_days=recent_days
                )
                all_documents.extend(documents)
        
        # Process and index the documents
        chunks = rag_processor.process_documents(all_documents)
        rag_processor.index_documents(chunks)
        
        logger.info(f"Indexed {len(chunks)} document chunks from {len(all_documents)} documents")
        
        return jsonify({
            'status': 'success',
            'documents_indexed': len(all_documents),
            'chunks_indexed': len(chunks)
        })
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        return jsonify({
            'error': f"An error occurred: {str(e)}"
        }), 500

# Confluence search endpoint
@api_bp.route('/search/confluence', methods=['GET'])
def search_confluence():
    """
    Search for content in Confluence.
    """
    try:
        query = request.args.get('query', '')
        limit = int(request.args.get('limit', 10))
        
        # Search Confluence
        results = confluence_connector.client.search_content(
            cql=f'text ~ "{query}"',
            max_results=limit,
            expand="body.storage"
        )
        
        if not results or 'results' not in results:
            return jsonify({
                'results': []
            })
        
        # Format the results
        formatted_results = []
        for item in results['results']:
            # Get space key from _expandable.space
            space_key = ""
            if '_expandable' in item and 'space' in item['_expandable']:
                space_path = item['_expandable']['space']
                space_key = space_path.split('/')[-1] if space_path else ""
            
            # Create result item
            result_item = {
                'id': item.get('id'),
                'title': item.get('title'),
                'type': item.get('type'),
                'space_key': space_key,
                'link': f"{confluence_connector.client.base_url}/display/{space_key}/{item.get('id')}"
            }
            
            # Add excerpt if available
            if 'body' in item and 'storage' in item['body']:
                html_content = item['body']['storage']['value']
                
                # Extract first 200 characters of text
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                text = soup.get_text()
                result_item['excerpt'] = text[:200] + '...' if len(text) > 200 else text
            
            formatted_results.append(result_item)
        
        return jsonify({
            'results': formatted_results
        })
    except Exception as e:
        logger.error(f"Error searching Confluence: {str(e)}")
        return jsonify({
            'error': f"An error occurred: {str(e)}"
        }), 500

# JIRA search endpoint
@api_bp.route('/search/jira', methods=['GET'])
def search_jira():
    """
    Search for issues in JIRA.
    """
    try:
        query = request.args.get('query', '')
        limit = int(request.args.get('limit', 10))
        
        # Build JQL
        jql = f'text ~ "{query}" ORDER BY updated DESC'
        
        # Search JIRA
        results = jira_connector.client.search_issues(
            jql=jql,
            max_results=limit,
            fields="summary,description,status,assignee,reporter,priority,created,updated,issueType"
        )
        
        if not results or 'issues' not in results:
            return jsonify({
                'results': []
            })
        
        # Format the results
        formatted_results = []
        for issue in results['issues']:
            # Create result item
            result_item = {
                'key': issue.get('key'),
                'id': issue.get('id'),
                'summary': issue.get('fields', {}).get('summary'),
                'status': issue.get('fields', {}).get('status', {}).get('name'),
                'priority': issue.get('fields', {}).get('priority', {}).get('name') if issue.get('fields', {}).get('priority') else None,
                'created': issue.get('fields', {}).get('created'),
                'updated': issue.get('fields', {}).get('updated'),
                'link': f"{jira_connector.client.base_url}/browse/{issue.get('key')}"
            }
            
            # Add assignee if available
            if 'assignee' in issue.get('fields', {}) and issue['fields']['assignee']:
                result_item['assignee'] = issue['fields']['assignee'].get('displayName')
            else:
                result_item['assignee'] = 'Unassigned'
            
            # Add excerpt from description
            description = issue.get('fields', {}).get('description', '')
            if description:
                result_item['excerpt'] = description[:200] + '...' if len(description) > 200 else description
            
            formatted_results.append(result_item)
        
        return jsonify({
            'results': formatted_results
        })
    except Exception as e:
        logger.error(f"Error searching JIRA: {str(e)}")
        return jsonify({
            'error': f"An error occurred: {str(e)}"
        }), 500

# Remedy search endpoint
@api_bp.route('/search/remedy', methods=['GET'])
def search_remedy():
    """
    Search for incidents in Remedy.
    """
    try:
        query = request.args.get('query', '')
        limit = int(request.args.get('limit', 10))
        
        # Check if Remedy connector is available
        if not remedy_connector.test_connection():
            return jsonify({
                'error': 'Remedy connection not available'
            }), 503
        
        # Login to Remedy
        login_status, _ = remedy_connector.client.login()
        if login_status != 1:
            return jsonify({
                'error': 'Failed to login to Remedy'
            }), 503
        
        try:
            # Search for incidents
            incidents = remedy_connector.search_incidents(
                query=query,
                max_results=limit
            )
            
            # Format the results
            formatted_results = []
            for incident in incidents:
                # Extract values
                values = incident.get('values', {})
                
                # Create result item
                incident_number = values.get('Incident Number')
                result_item = {
                    'incident_number': incident_number,
                    'summary': values.get('Summary'),
                    'status': values.get('Status'),
                    'priority': values.get('Priority'),
                    'impact': values.get('Impact'),
                    'assignee': values.get('Assignee'),
                    'owner_group': values.get('Owner Group'),
                    'submit_date': values.get('Submit Date'),
                    'link': f"{remedy_connector.client.server_url}/arsys/forms/onbmc-s/HPD%3AHelp+Desk/Default+Administrator+View/?mode=search&F304255500={incident_number}"
                }
                
                # Add excerpt from description
                description = values.get('Description', '')
                if description:
                    result_item['excerpt'] = description[:200] + '...' if len(description) > 200 else description
                
                formatted_results.append(result_item)
            
            return jsonify({
                'results': formatted_results
            })
        finally:
            # Always logout
            remedy_connector.client.logout()
    except Exception as e:
        logger.error(f"Error searching Remedy: {str(e)}")
        return jsonify({
            'error': f"An error occurred: {str(e)}"
        }), 500

















"""
API package for Enterprise RAG System.
"""
from api.routes import register_api_routes
















"""
Response formatting utilities for Enterprise RAG System.
"""
import json
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)

class ResponseFormatter:
    """
    Formats API responses for the Enterprise RAG System.
    """
    
    @staticmethod
    def format_chat_response(response, context=None):
        """
        Format a chat response with context citations.
        
        Args:
            response: Generated response text
            context: Context items used for generation
            
        Returns:
            dict: Formatted response
        """
        if not response:
            return {"error": "No response generated"}
        
        # Process citations if context is provided
        sources = []
        if context:
            # Extract unique sources from context
            seen_sources = set()
            for item in context:
                source = item.get("source")
                source_type = item.get("source_type")
                source_link = item.get("source_link")
                source_id = item.get("source_id")
                
                # Skip if missing key information
                if not source or not source_type:
                    continue
                
                # Create a unique key for the source
                source_key = f"{source_type}:{source_id}" if source_id else f"{source_type}:{source}"
                
                # Only add if not already seen
                if source_key not in seen_sources:
                    sources.append({
                        "source": source,
                        "type": source_type,
                        "link": source_link,
                        "id": source_id
                    })
                    seen_sources.add(source_key)
        
        # Create the formatted response
        formatted_response = {
            "response": response,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
        
        return formatted_response
    
    @staticmethod
    def format_error_response(error_message):
        """
        Format an error response.
        
        Args:
            error_message: Error message
            
        Returns:
            dict: Formatted error response
        """
        return {
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def format_search_results(results, source_type):
        """
        Format search results.
        
        Args:
            results: List of search results
            source_type: Type of source (confluence, jira, remedy)
            
        Returns:
            dict: Formatted search results
        """
        formatted_results = {
            "source_type": source_type,
            "count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        return formatted_results

# Global response formatter instance
response_formatter = ResponseFormatter()

















<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Enterprise RAG System{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">
    {% block extra_css %}{% endblock %}
</head>
<body>
    <!-- Navigation Bar -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-gradient">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">
                <i class="fas fa-robot me-2"></i>Enterprise RAG
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link {% if request.path == '/' %}active{% endif %}" href="{{ url_for('index') }}">
                            <i class="fas fa-home me-1"></i> Home
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link {% if request.path == '/chat' %}active{% endif %}" href="{{ url_for('chat') }}">
                            <i class="fas fa-comment-dots me-1"></i> Chat
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#" data-bs-toggle="modal" data-bs-target="#aboutModal">
                            <i class="fas fa-info-circle me-1"></i> About
                        </a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- Main Content Area -->
    <div class="container mt-4">
        {% block content %}{% endblock %}
    </div>

    <!-- Footer -->
    <footer class="footer mt-5 py-3 bg-light">
        <div class="container text-center">
            <span class="text-muted">Enterprise RAG System &copy; {{ now.year }}</span>
        </div>
    </footer>

    <!-- About Modal -->
    <div class="modal fade" id="aboutModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">About Enterprise RAG</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p>Enterprise RAG is a Retrieval-Augmented Generation system that integrates with:</p>
                    <ul>
                        <li><strong>Confluence</strong> - For company documentation and knowledge base</li>
                        <li><strong>JIRA</strong> - For project management and issue tracking</li>
                        <li><strong>Remedy</strong> - For incident management and IT service desk</li>
                    </ul>
                    <p>The system uses Google's Gemini AI to provide accurate answers to your questions based on company data.</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-primary" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked@5.1.0/marked.min.js"></script>
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block extra_scripts %}{% endblock %}
</body>
</html>
















{% extends "base.html" %}

{% block title %}Enterprise RAG System - Home{% endblock %}

{% block extra_css %}
<style>
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: #4A6FDC;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #3a4daa 0%, #6c88e0 100%);
        color: white;
        padding: 4rem 0;
        border-radius: 10px;
        margin-bottom: 3rem;
    }
    
    .source-card {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    
    .source-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
    
    .card-header {
        font-weight: 600;
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #3a4daa 0%, #6c88e0 100%);
        border: none;
    }
    
    .btn-primary:hover {
        background: linear-gradient(135deg, #2c3d8a 0%, #5a76ca 100%);
    }
</style>
{% endblock %}

{% block content %}
<!-- Hero Section -->
<div class="hero-section text-center mb-5 shadow">
    <div class="container py-5">
        <h1 class="display-4 fw-bold mb-4">Enterprise Intelligence at Your Fingertips</h1>
        <p class="lead mb-4">Access and query your organization's knowledge across Confluence, JIRA, and Remedy with natural language.</p>
        <a href="{{ url_for('chat') }}" class="btn btn-light btn-lg px-4 py-2">
            <i class="fas fa-comment-dots me-2"></i>Start Chatting
        </a>
    </div>
</div>

<!-- Features Section -->
<div class="container mb-5">
    <h2 class="text-center mb-5">How It Works</h2>
    <div class="row g-4">
        <div class="col-md-4 text-center">
            <div class="feature-icon">
                <i class="fas fa-search"></i>
            </div>
            <h3>Intelligent Retrieval</h3>
            <p>Our system searches across your enterprise data sources to find the most relevant information.</p>
        </div>
        <div class="col-md-4 text-center">
            <div class="feature-icon">
                <i class="fas fa-brain"></i>
            </div>
            <h3>AI-Powered Analysis</h3>
            <p>Google's Gemini AI processes the retrieved information to generate accurate and contextual responses.</p>
        </div>
        <div class="col-md-4 text-center">
            <div class="feature-icon">
                <i class="fas fa-comment-alt"></i>
            </div>
            <h3>Natural Conversation</h3>
            <p>Ask questions in plain language and receive clear, sourced answers with citations.</p>
        </div>
    </div>
</div>

<!-- Data Sources Section -->
<div class="container mb-5">
    <h2 class="text-center mb-5">Integrated Data Sources</h2>
    <div class="row g-4">
        <div class="col-md-4">
            <div class="card source-card h-100">
                <div class="card-header bg-primary text-white">
                    <i class="fas fa-book me-2"></i>Confluence
                </div>
                <div class="card-body">
                    <h5 class="card-title">Company Knowledge Base</h5>
                    <p class="card-text">Access documentation, guidelines, procedures, and company knowledge stored in Confluence.</p>
                    <p class="card-text text-muted"><small>Examples: Project documentation, technical guides, company policies</small></p>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card source-card h-100">
                <div class="card-header bg-primary text-white">
                    <i class="fas fa-tasks me-2"></i>JIRA
                </div>
                <div class="card-body">
                    <h5 class="card-title">Project & Issue Tracking</h5>
                    <p class="card-text">Query information about projects, tasks, bugs, and other issues tracked in JIRA.</p>
                    <p class="card-text text-muted"><small>Examples: Project status, issue details, feature requests</small></p>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card source-card h-100">
                <div class="card-header bg-primary text-white">
                    <i class="fas fa-headset me-2"></i>Remedy
                </div>
                <div class="card-body">
                    <h5 class="card-title">Service Management</h5>
                    <p class="card-text">Get information about incidents, service requests, and IT service management data from Remedy.</p>
                    <p class="card-text text-muted"><small>Examples: Incident details, request status, service metrics</small></p>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Call to Action -->
<div class="container mb-5 text-center">
    <div class="card p-5 bg-light">
        <h2>Ready to get started?</h2>
        <p class="lead mb-4">Ask questions in natural language and get accurate answers based on your company's data.</p>
        <div>
            <a href="{{ url_for('chat') }}" class="btn btn-primary btn-lg px-4 py-2">
                <i class="fas fa-comment-dots me-2"></i>Go to Chat Interface
            </a>
        </div>
    </div>
</div>
{% endblock %}
















{% extends "base.html" %}

{% block title %}Enterprise RAG System - Chat{% endblock %}

{% block extra_css %}
<style>
    body {
        overflow-x: hidden;
    }
    
    .chat-container {
        height: calc(100vh - 200px);
        display: flex;
        flex-direction: column;
    }
    
    .chat-header {
        background: linear-gradient(135deg, #3a4daa 0%, #6c88e0 100%);
        color: white;
        padding: 1rem;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }
    
    .chat-messages {
        flex-grow: 1;
        overflow-y: auto;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        background-color: #f8f9fa;
    }
    
    .message {
        max-width: 80%;
        margin-bottom: 1rem;
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        position: relative;
        line-height: 1.5;
    }
    
    .user-message {
        align-self: flex-end;
        background-color: #dcf8c6;
        border-bottom-right-radius: 0.2rem;
    }
    
    .assistant-message {
        align-self: flex-start;
        background-color: white;
        border-bottom-left-radius: 0.2rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .message-content {
        word-wrap: break-word;
    }
    
    .message-time {
        display: block;
        font-size: 0.7rem;
        margin-top: 0.3rem;
        text-align: right;
        color: #777;
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding: 0.75rem 1rem;
        background-color: white;
        border-radius: 1rem;
        border-bottom-left-radius: 0.2rem;
        width: fit-content;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .typing-bubble {
        display: inline-block;
        width: 8px;
        height: 8px;
        margin-right: 3px;
        border-radius: 50%;
        background-color: #6c757d;
        animation: typing-bubble 1.2s infinite ease-in-out;
    }
    
    .typing-bubble:nth-child(1) {
        animation-delay: 0s;
    }
    
    .typing-bubble:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-bubble:nth-child(3) {
        animation-delay: 0.4s;
        margin-right: 0;
    }
    
    @keyframes typing-bubble {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-5px);
        }
    }
    
    .chat-input-container {
        display: flex;
        padding: 1rem;
        background-color: white;
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
        box-shadow: 0 -1px 5px rgba(0, 0, 0, 0.05);
    }
    
    .chat-input {
        flex-grow: 1;
        border: 1px solid #ced4da;
        border-radius: 1.5rem;
        padding: 0.75rem 1rem;
        resize: none;
    }
    
    .chat-input:focus {
        outline: none;
        border-color: #4A6FDC;
        box-shadow: 0 0 0 0.2rem rgba(74, 111, 220, 0.25);
    }
    
    .chat-send-btn {
        margin-left: 0.5rem;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #3a4daa 0%, #6c88e0 100%);
        border: none;
        color: white;
        cursor: pointer;
    }
    
    .chat-send-btn:disabled {
        background: #c5c5c5;
        cursor: not-allowed;
    }
    
    .chat-send-btn:not(:disabled):hover {
        background: linear-gradient(135deg, #2c3d8a 0%, #5a76ca 100%);
    }
    
    .sources-container {
        margin-top: 1rem;
        padding: 0.75rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        font-size: 0.85rem;
    }
    
    .sources-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    
    .source-item {
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        background-color: white;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .source-type {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .source-type-confluence {
        background-color: #0052CC;
        color: white;
    }
    
    .source-type-jira {
        background-color: #0052CC;
        color: white;
    }
    
    .source-type-remedy {
        background-color: #FF5630;
        color: white;
    }
    
    .source-link {
        color: #4A6FDC;
        text-decoration: none;
    }
    
    .source-link:hover {
        text-decoration: underline;
    }
    
    .welcome-message {
        padding: 2rem;
        text-align: center;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .welcome-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: #4A6FDC;
    }
    
    .data-source-selector {
        margin-bottom: 1rem;
    }
    
    .source-checkbox {
        margin-right: 0.5rem;
    }
    
    code {
        background-color: #f1f1f1;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.9em;
        color: #e83e8c;
    }
    
    pre {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        overflow-x: auto;
    }
    
    blockquote {
        border-left: 3px solid #4A6FDC;
        padding-left: 1rem;
        color: #6c757d;
        margin: 1rem 0;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
    }
    
    table, th, td {
        border: 1px solid #dee2e6;
    }
    
    th, td {
        padding: 0.5rem;
        text-align: left;
    }
    
    th {
        background-color: #f8f9fa;
    }
</style>
{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-12 mb-4">
        <div class="card chat-container">
            <div class="chat-header">
                <div class="d-flex justify-content-between align-items-center">
                    <h5 class="mb-0">
                        <i class="fas fa-robot me-2"></i>Enterprise RAG Assistant
                    </h5>
                    <div class="form-check form-switch">
                        <input class="form-check-input" type="checkbox" id="streamingToggle" checked>
                        <label class="form-check-label text-white" for="streamingToggle">Streaming</label>
                    </div>
                </div>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="welcome-message">
                    <div class="welcome-icon">
                        <i class="fas fa-robot"></i>
                    </div>
                    <h4>Welcome to Enterprise RAG Assistant</h4>
                    <p>Ask me anything about your organization's knowledge across Confluence, JIRA, and Remedy.</p>
                    <div class="data-source-selector">
                        <div class="form-check form-check-inline">
                            <input class="form-check-input source-checkbox" type="checkbox" id="confluenceSource" value="confluence" checked>
                            <label class="form-check-label" for="confluenceSource">Confluence</label>
                        </div>
                        <div class="form-check form-check-inline">
                            <input class="form-check-input source-checkbox" type="checkbox" id="jiraSource" value="jira" checked>
                            <label class="form-check-label" for="jiraSource">JIRA</label>
                        </div>
                        <div class="form-check form-check-inline">
                            <input class="form-check-input source-checkbox" type="checkbox" id="remedySource" value="remedy" checked>
                            <label class="form-check-label" for="remedySource">Remedy</label>
                        </div>
                    </div>
                    <p class="text-muted small">Example questions you can ask:</p>
                    <div class="d-flex flex-wrap justify-content-center gap-2 mb-3">
                        <button class="btn btn-sm btn-outline-primary example-question">What are the recent incidents in Remedy?</button>
                        <button class="btn btn-sm btn-outline-primary example-question">Where can I find documentation on the RAG system?</button>
                        <button class="btn btn-sm btn-outline-primary example-question">What's the status of JIRA ticket DEMO-123?</button>
                    </div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <textarea class="chat-input" id="chatInput" placeholder="Type your question here..." rows="1"></textarea>
                <button class="chat-send-btn" id="sendBtn" disabled>
                    <i class="fas fa-paper-plane"></i>
                </button>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_scripts %}
<script>
    document.addEventListener('DOMContentLoaded', function() {
        const chatMessages = document.getElementById('chatMessages');
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        const streamingToggle = document.getElementById('streamingToggle');
        const exampleQuestions = document.querySelectorAll('.example-question');
        
        // Auto-adjust textarea height
        chatInput.addEventListener('input', function() {
            chatInput.style.height = 'auto';
            chatInput.style.height = (chatInput.scrollHeight) + 'px';
            
            // Enable/disable send button based on input content
            sendBtn.disabled = chatInput.value.trim() === '';
        });
        
        // Handle enter key to send message
        chatInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!sendBtn.disabled) {
                    sendMessage();
                }
            }
        });
        
        // Send button click handler
        sendBtn.addEventListener('click', sendMessage);
        
        // Example question click handler
        exampleQuestions.forEach(button => {
            button.addEventListener('click', function() {
                chatInput.value = this.textContent;
                chatInput.dispatchEvent(new Event('input'));
                sendMessage();
            });
        });
        
        // Function to send user message and get response
        function sendMessage() {
            const userMessage = chatInput.value.trim();
            if (!userMessage) return;
            
            // Add user message to chat
            addMessage(userMessage, 'user');
            
            // Clear input and reset height
            chatInput.value = '';
            chatInput.style.height = 'auto';
            sendBtn.disabled = true;
            
            // Get selected data sources
            const selectedSources = [];
            document.querySelectorAll('.source-checkbox:checked').forEach(cb => {
                selectedSources.push(cb.value);
            });
            
            // Show typing indicator
            showTypingIndicator();
            
            // Prepare request data
            const requestData = {
                query: userMessage,
                sources: selectedSources,
                stream: streamingToggle.checked
            };
            
            // Send request to API
            if (streamingToggle.checked) {
                // Streaming response
                fetchStreamingResponse(requestData);
            } else {
                // Non-streaming response
                fetchCompleteResponse(requestData);
            }
        }
        
        // Function to fetch streaming response
        function fetchStreamingResponse(requestData) {
            const assistantMessageId = 'assistant-' + Date.now();
            addEmptyAssistantMessage(assistantMessageId);
            
            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            }).then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                
                // Remove typing indicator
                removeTypingIndicator();
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                
                const processChunk = ({ done, value }) => {
                    if (done) {
                        return;
                    }
                    
                    // Decode chunk and append to buffer
                    buffer += decoder.decode(value, { stream: true });
                    
                    // Update message content
                    const messageEl = document.getElementById(assistantMessageId);
                    if (messageEl) {
                        const contentEl = messageEl.querySelector('.message-content');
                        contentEl.innerHTML = formatMessage(buffer);
                    }
                    
                    // Continue reading
                    return reader.read().then(processChunk);
                };
                
                return reader.read().then(processChunk);
            }).catch(error => {
                console.error('Error:', error);
                removeTypingIndicator();
                addMessage('Sorry, an error occurred. Please try again.', 'assistant');
            });
        }
        
        // Function to fetch complete response
        function fetchCompleteResponse(requestData) {
            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            }).then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            }).then(data => {
                // Remove typing indicator
                removeTypingIndicator();
                
                // Add response to chat
                addMessage(data.response, 'assistant', data.context);
            }).catch(error => {
                console.error('Error:', error);
                removeTypingIndicator();
                addMessage('Sorry, an error occurred. Please try again.', 'assistant');
            });
        }
        
        // Function to add message to chat
        function addMessage(message, sender, context = null) {
            const messageEl = document.createElement('div');
            messageEl.className = `message ${sender}-message`;
            messageEl.id = `${sender}-${Date.now()}`;
            
            const contentEl = document.createElement('div');
            contentEl.className = 'message-content';
            contentEl.innerHTML = formatMessage(message);
            
            const timeEl = document.createElement('div');
            timeEl.className = 'message-time';
            timeEl.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            
            messageEl.appendChild(contentEl);
            messageEl.appendChild(timeEl);
            
            // Add sources if present (for assistant messages)
            if (sender === 'assistant' && context && context.length > 0) {
                const sourcesContainer = document.createElement('div');
                sourcesContainer.className = 'sources-container';
                
                const sourcesTitle = document.createElement('div');
                sourcesTitle.className = 'sources-title';
                sourcesTitle.innerHTML = `<i class="fas fa-bookmark me-1"></i>Sources (${context.length})`;
                sourcesContainer.appendChild(sourcesTitle);
                
                // Deduplicate sources
                const uniqueSources = {};
                context.forEach(item => {
                    const sourceKey = `${item.source_type}-${item.source}`;
                    if (!uniqueSources[sourceKey]) {
                        uniqueSources[sourceKey] = item;
                    }
                });
                
                // Add each unique source
                Object.values(uniqueSources).forEach(source => {
                    const sourceItem = document.createElement('div');
                    sourceItem.className = 'source-item';
                    
                    let sourceTypeClass = 'source-type-confluence';
                    if (source.source_type === 'jira') {
                        sourceTypeClass = 'source-type-jira';
                    } else if (source.source_type === 'remedy') {
                        sourceTypeClass = 'source-type-remedy';
                    }
                    
                    sourceItem.innerHTML = `
                        <span class="source-type ${sourceTypeClass}">${source.source_type}</span>
                        <span class="source-name">${source.source}</span>
                        ${source.source_link ? `<a href="${source.source_link}" target="_blank" class="source-link ms-2"><i class="fas fa-external-link-alt"></i></a>` : ''}
                    `;
                    
                    sourcesContainer.appendChild(sourceItem);
                });
                
                messageEl.appendChild(sourcesContainer);
            }
            
            chatMessages.appendChild(messageEl);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Function to add empty assistant message for streaming
        function addEmptyAssistantMessage(id) {
            const messageEl = document.createElement('div');
            messageEl.className = 'message assistant-message';
            messageEl.id = id;
            
            const contentEl = document.createElement('div');
            contentEl.className = 'message-content';
            
            const timeEl = document.createElement('div');
            timeEl.className = 'message-time';
            timeEl.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            
            messageEl.appendChild(contentEl);
            messageEl.appendChild(timeEl);
            
            chatMessages.appendChild(messageEl);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Function to show typing indicator
        function showTypingIndicator() {
            const typingIndicator = document.createElement('div');
            typingIndicator.className = 'typing-indicator';
            typingIndicator.id = 'typingIndicator';
            
            typingIndicator.innerHTML = `
                <div class="typing-bubble"></div>
                <div class="typing-bubble"></div>
                <div class="typing-bubble"></div>
            `;
            
            chatMessages.appendChild(typingIndicator);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Function to remove typing indicator
        function removeTypingIndicator() {
            const typingIndicator = document.getElementById('typingIndicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }
        
        // Function to format message with markdown
        function formatMessage(message) {
            // Convert markdown to HTML using marked library
            return marked.parse(message);
        }
    });
</script>
{% endblock %}


















{% extends "base.html" %}

{% block title %}Error {{ status_code }} - Enterprise RAG System{% endblock %}

{% block extra_css %}
<style>
    .error-container {
        text-align: center;
        padding: 5rem 1rem;
    }
    
    .error-code {
        font-size: 8rem;
        font-weight: 700;
        margin-bottom: 0;
        background: linear-gradient(135deg, #3a4daa 0%, #6c88e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
    }
    
    .error-title {
        font-size: 2rem;
        margin-bottom: 1.5rem;
        color: #343a40;
    }
    
    .error-message {
        font-size: 1.1rem;
        color: #6c757d;
        max-width: 600px;
        margin: 0 auto 2rem;
    }
    
    .error-icon {
        font-size: 5rem;
        margin-bottom: 1.5rem;
        color: #6c88e0;
    }
</style>
{% endblock %}

{% block content %}
<div class="error-container">
    {% if status_code == 404 %}
        <div class="error-icon">
            <i class="fas fa-map-signs"></i>
        </div>
        <h1 class="error-code">404</h1>
        <h2 class="error-title">Page Not Found</h2>
        <p class="error-message">
            Oops! The page you are looking for does not exist or has been moved.
        </p>
    {% elif status_code == 500 %}
        <div class="error-icon">
            <i class="fas fa-exclamation-triangle"></i>
        </div>
        <h1 class="error-code">500</h1>
        <h2 class="error-title">Internal Server Error</h2>
        <p class="error-message">
            Sorry, something went wrong on our end. We're working to fix the issue.
        </p>
    {% elif status_code == 403 %}
        <div class="error-icon">
            <i class="fas fa-lock"></i>
        </div>
        <h1 class="error-code">403</h1>
        <h2 class="error-title">Access Forbidden</h2>
        <p class="error-message">
            You don't have permission to access this resource.
        </p>
    {% else %}
        <div class="error-icon">
            <i class="fas fa-exclamation-circle"></i>
        </div>
        <h1 class="error-code">{{ status_code }}</h1>
        <h2 class="error-title">An Error Occurred</h2>
        <p class="error-message">
            {{ error }}
        </p>
    {% endif %}
    
    <div class="mt-4">
        <a href="{{ url_for('index') }}" class="btn btn-primary">
            <i class="fas fa-home me-2"></i>Back to Home
        </a>
    </div>
    
    {% if config.DEBUG %}
        <div class="alert alert-warning mt-5 mx-auto" style="max-width: 800px; text-align: left;">
            <h5 class="alert-heading"><i class="fas fa-bug me-2"></i>Debug Information</h5>
            <hr>
            <pre>{{ error }}</pre>
        </div>
    {% endif %}
</div>
{% endblock %}




















/* 
 * Main Stylesheet for Enterprise RAG System
 * Features a professional, cool and serene design with gradients
 */

:root {
  /* Color Palette */
  --primary-color: #4361ee;
  --primary-light: #4895ef;
  --primary-dark: #3a0ca3;
  --secondary-color: #4cc9f0;
  --accent-color: #7209b7;
  --success-color: #06d6a0;
  --warning-color: #ffd166;
  --danger-color: #ef476f;
  --gray-100: #f8f9fa;
  --gray-200: #e9ecef;
  --gray-300: #dee2e6;
  --gray-400: #ced4da;
  --gray-500: #adb5bd;
  --gray-600: #6c757d;
  --gray-700: #495057;
  --gray-800: #343a40;
  --gray-900: #212529;
  
  /* Gradients */
  --gradient-primary: linear-gradient(135deg, var(--primary-color), var(--primary-light));
  --gradient-secondary: linear-gradient(135deg, var(--secondary-color), var(--primary-light));
  --gradient-accent: linear-gradient(135deg, var(--accent-color), var(--primary-dark));
  
  /* Shadows */
  --shadow-sm: 0 .125rem .25rem rgba(0, 0, 0, .075);
  --shadow-md: 0 .5rem 1rem rgba(0, 0, 0, .15);
  --shadow-lg: 0 1rem 3rem rgba(0, 0, 0, .175);
  
  /* Spacing */
  --spacing-xs: 0.25rem;
  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 1.5rem;
  --spacing-xl: 3rem;
  
  /* Border radius */
  --border-radius-sm: 0.25rem;
  --border-radius-md: 0.5rem;
  --border-radius-lg: 1rem;
  --border-radius-xl: 1.5rem;
}

/* Base Styles */
body {
  font-family: 'Inter', 'Segoe UI', Roboto, -apple-system, BlinkMacSystemFont, sans-serif;
  color: var(--gray-800);
  background-color: var(--gray-100);
  line-height: 1.6;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

a {
  color: var(--primary-color);
  text-decoration: none;
  transition: color 0.3s;
}

a:hover {
  color: var(--primary-dark);
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
  font-weight: 600;
  line-height: 1.2;
  margin-bottom: 1rem;
  color: var(--gray-900);
}

.text-primary { color: var(--primary-color) !important; }
.text-secondary { color: var(--secondary-color) !important; }
.text-accent { color: var(--accent-color) !important; }
.text-success { color: var(--success-color) !important; }
.text-warning { color: var(--warning-color) !important; }
.text-danger { color: var(--danger-color) !important; }

/* Navigation */
.navbar {
  padding: 1rem 0;
  box-shadow: var(--shadow-sm);
}

.navbar-dark {
  background: var(--gradient-primary);
}

.navbar-brand {
  font-weight: 700;
  font-size: 1.5rem;
  color: white;
}

.navbar-nav .nav-link {
  font-weight: 500;
  padding: 0.5rem 1rem;
  color: rgba(255, 255, 255, 0.85);
  transition: color 0.3s;
}

.navbar-nav .nav-link:hover,
.navbar-nav .nav-link.active {
  color: white;
}

/* Container */
.container {
  max-width: 1200px;
  padding: 0 15px;
}

/* Main content */
.main-content {
  flex: 1;
  padding: var(--spacing-lg) 0;
}

/* Cards */
.card {
  border: none;
  border-radius: var(--border-radius-md);
  box-shadow: var(--shadow-sm);
  transition: transform 0.3s, box-shadow 0.3s;
  overflow: hidden;
  margin-bottom: var(--spacing-lg);
  background-color: white;
}

.card:hover {
  transform: translateY(-5px);
  box-shadow: var(--shadow-md);
}

.card-header {
  background-color: white;
  border-bottom: 1px solid var(--gray-200);
  font-weight: 600;
  padding: 1rem 1.5rem;
}

.card-body {
  padding: 1.5rem;
}

.card-title {
  font-size: 1.25rem;
  margin-bottom: 0.75rem;
}

.card-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  color: var(--primary-color);
}

/* Feature Cards */
.feature-card {
  text-align: center;
  padding: 2rem;
}

/* Buttons */
.btn {
  font-weight: 500;
  padding: 0.5rem 1.25rem;
  border-radius: var(--border-radius-md);
  transition: all 0.3s;
}

.btn-primary {
  background: var(--gradient-primary);
  border: none;
}

.btn-primary:hover {
  background: var(--gradient-primary);
  opacity: 0.9;
  transform: translateY(-2px);
}

.btn-outline-primary {
  border-color: var(--primary-color);
  color: var(--primary-color);
}

.btn-outline-primary:hover {
  background-color: var(--primary-color);
  color: white;
}

.btn-secondary {
  background: var(--gradient-secondary);
  border: none;
}

.btn-accent {
  background: var(--gradient-accent);
  border: none;
  color: white;
}

/* Icons */
.icon-primary { color: var(--primary-color); }
.icon-secondary { color: var(--secondary-color); }
.icon-accent { color: var(--accent-color); }
.icon-success { color: var(--success-color); }
.icon-warning { color: var(--warning-color); }
.icon-danger { color: var(--danger-color); }

/* Badges */
.badge {
  font-weight: 500;
  padding: 0.35em 0.65em;
  border-radius: var(--border-radius-sm);
}

.badge-primary {
  background-color: var(--primary-color);
  color: white;
}

.badge-secondary {
  background-color: var(--secondary-color);
  color: white;
}

.badge-accent {
  background-color: var(--accent-color);
  color: white;
}

/* Alerts */
.alert {
  border: none;
  border-radius: var(--border-radius-md);
  padding: 1rem 1.5rem;
}

.alert-primary {
  background-color: rgba(67, 97, 238, 0.15);
  color: var(--primary-dark);
}

.alert-secondary {
  background-color: rgba(76, 201, 240, 0.15);
  color: var(--secondary-color);
}

.alert-danger {
  background-color: rgba(239, 71, 111, 0.15);
  color: var(--danger-color);
}

/* Forms */
.form-control {
  border-radius: var(--border-radius-md);
  padding: 0.75rem 1rem;
  border: 1px solid var(--gray-300);
  transition: border-color 0.3s, box-shadow 0.3s;
}

.form-control:focus {
  border-color: var(--primary-light);
  box-shadow: 0 0 0 0.25rem rgba(67, 97, 238, 0.25);
}

.form-label {
  font-weight: 500;
  margin-bottom: 0.5rem;
  color: var(--gray-700);
}

/* Source Citation */
.source-citation {
  padding: 0.75rem 1rem;
  background-color: var(--gray-100);
  border-radius: var(--border-radius-md);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
}

.source-citation i {
  margin-right: 0.75rem;
  font-size: 1.25rem;
  color: var(--primary-color);
}

.source-citation a {
  color: var(--primary-color);
  font-weight: 500;
}

/* Footer */
.footer {
  margin-top: auto;
  padding: 1.5rem 0;
  background-color: var(--gray-200);
  border-top: 1px solid var(--gray-300);
}

/* Animations */
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.fade-in {
  animation: fadeIn 0.5s ease-out forwards;
}

/* Loading indicator */
.loading {
  display: inline-block;
  width: 1.5rem;
  height: 1.5rem;
  border: 3px solid rgba(0, 0, 0, 0.2);
  border-radius: 50%;
  border-top-color: var(--primary-color);
  animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .container {
    padding: 0 10px;
  }
  
  .card {
    margin-bottom: 1rem;
  }
  
  .navbar-brand {
    font-size: 1.25rem;
  }
}

/* Image styling */
.img-fluid {
  max-width: 100%;
  height: auto;
  border-radius: var(--border-radius-md);
}

/* Hero section */
.hero {
  padding: 3rem 0;
  text-align: center;
  background: var(--gradient-primary);
  color: white;
  border-radius: var(--border-radius-lg);
  margin-bottom: 2rem;
}

.hero-title {
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
}

.hero-subtitle {
  font-size: 1.25rem;
  margin-bottom: 2rem;
  opacity: 0.9;
}

/* Section styling */
.section {
  padding: 3rem 0;
}

.section-title {
  text-align: center;
  margin-bottom: 2rem;
  position: relative;
}

.section-title:after {
  content: '';
  display: block;
  width: 50px;
  height: 3px;
  background: var(--primary-color);
  margin: 0.5rem auto 0;
  border-radius: 3px;
}

/* Modal styling */
.modal-content {
  border: none;
  border-radius: var(--border-radius-md);
  box-shadow: var(--shadow-lg);
}

.modal-header {
  background: var(--gradient-primary);
  color: white;
  border-bottom: none;
  border-top-left-radius: var(--border-radius-md);
  border-top-right-radius: var(--border-radius-md);
}

.modal-title {
  font-weight: 600;
}

.modal-footer {
  border-top: 1px solid var(--gray-200);
}

/* Table styling */
.table {
  border-collapse: separate;
  border-spacing: 0;
  width: 100%;
}

.table th {
  background-color: var(--gray-100);
  font-weight: 600;
  text-align: left;
  padding: 0.75rem 1rem;
  border-bottom: 2px solid var(--gray-300);
}

.table td {
  padding: 0.75rem 1rem;
  border-bottom: 1px solid var(--gray-200);
  vertical-align: middle;
}

.table-striped tbody tr:nth-of-type(odd) {
  background-color: rgba(0, 0, 0, 0.02);
}

.table-hover tbody tr:hover {
  background-color: rgba(0, 0, 0, 0.04);
}

/* Code blocks from markdown */
pre {
  background-color: var(--gray-800);
  color: var(--gray-100);
  padding: 1rem;
  border-radius: var(--border-radius-md);
  overflow-x: auto;
  margin: 1rem 0;
}

code {
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  font-size: 0.9em;
}

/* For syntax highlighting */
.hljs-keyword { color: #c678dd; }
.hljs-string { color: #98c379; }
.hljs-comment { color: #7f848e; font-style: italic; }
.hljs-number { color: #d19a66; }
.hljs-literal { color: #56b6c2; }
.hljs-tag { color: #e06c75; }
.hljs-attr { color: #d19a66; }
.hljs-selector { color: #e06c75; }
.hljs-meta { color: #56b6c2; }
.hljs-builtin-name { color: #e6c07b; }
.hljs-built_in { color: #e6c07b; }
.hljs-name { color: #e06c75; }


















/* 
 * Chat Interface Stylesheet for Enterprise RAG System
 * Features a professional, cool and serene design with gradients
 */

/* Chat container */
.chat-container {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 200px);
  min-height: 500px;
  background-color: white;
  border-radius: var(--border-radius-lg);
  box-shadow: var(--shadow-md);
  overflow: hidden;
}

/* Chat header */
.chat-header {
  background: var(--gradient-primary);
  color: white;
  padding: 1rem 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.chat-header h2 {
  margin: 0;
  font-size: 1.25rem;
  color: white;
}

.chat-header .chat-controls {
  display: flex;
  gap: 0.75rem;
}

.chat-header .btn {
  padding: 0.375rem 0.75rem;
  color: white;
  background-color: rgba(255, 255, 255, 0.15);
  border: none;
  border-radius: var(--border-radius-md);
  transition: background-color 0.3s;
}

.chat-header .btn:hover {
  background-color: rgba(255, 255, 255, 0.25);
}

/* Chat messages area */
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  background-color: var(--gray-100);
}

/* Message bubble styling */
.message {
  display: flex;
  max-width: 85%;
  animation: fadeIn 0.3s ease-out forwards;
}

/* User message */
.message.user {
  align-self: flex-end;
}

.message.user .message-content {
  background: var(--gradient-primary);
  color: white;
  border-radius: var(--border-radius-md) 0 var(--border-radius-md) var(--border-radius-md);
  padding: 0.75rem 1rem;
  box-shadow: var(--shadow-sm);
}

/* Assistant message */
.message.assistant {
  align-self: flex-start;
}

.message.assistant .message-content {
  background-color: white;
  border-radius: 0 var(--border-radius-md) var(--border-radius-md) var(--border-radius-md);
  padding: 0.75rem 1rem;
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--gray-200);
}

/* Message bubble animations */
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.fade-in {
  animation: fadeIn 0.3s ease-out forwards;
}

/* Message metadata */
.message-meta {
  font-size: 0.75rem;
  color: var(--gray-500);
  margin-top: 0.25rem;
  display: flex;
  align-items: center;
}

.message-meta i {
  margin-right: 0.25rem;
}

/* Message content */
.message-content {
  word-break: break-word;
  line-height: 1.5;
}

/* Markdown rendering in messages */
.message-content p {
  margin: 0 0 0.75rem 0;
}

.message-content p:last-child {
  margin-bottom: 0;
}

.message-content a {
  color: inherit;
  text-decoration: underline;
}

.message-content ul, 
.message-content ol {
  margin: 0.5rem 0;
  padding-left: 1.5rem;
}

.message-content h1, 
.message-content h2, 
.message-content h3, 
.message-content h4 {
  margin: 1rem 0 0.5rem 0;
}

/* Code blocks in messages */
.message-content pre {
  margin: 0.75rem 0;
  border-radius: var(--border-radius-sm);
  max-width: 100%;
  overflow-x: auto;
}

.message-content code {
  font-size: 0.85rem;
}

.message.user .message-content pre {
  background-color: rgba(0, 0, 0, 0.2);
}

.message.assistant .message-content pre {
  background-color: var(--gray-800);
  color: var(--gray-100);
}

/* Chat input area */
.chat-input-container {
  padding: 1rem;
  background-color: white;
  border-top: 1px solid var(--gray-200);
  display: flex;
  align-items: flex-end;
  gap: 0.75rem;
}

.chat-input {
  flex: 1;
  border-radius: var(--border-radius-md);
  border: 1px solid var(--gray-300);
  padding: 0.75rem 1rem;
  resize: none;
  min-height: 42px;
  max-height: 150px;
  transition: border-color 0.3s, box-shadow 0.3s;
}

.chat-input:focus {
  border-color: var(--primary-light);
  box-shadow: 0 0 0 0.25rem rgba(67, 97, 238, 0.25);
  outline: none;
}

.chat-submit {
  background: var(--gradient-primary);
  color: white;
  border: none;
  border-radius: var(--border-radius-md);
  padding: 0.75rem 1.25rem;
  font-weight: 500;
  cursor: pointer;
  transition: opacity 0.3s, transform 0.3s;
}

.chat-submit:hover {
  opacity: 0.9;
  transform: translateY(-2px);
}

.chat-submit:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

/* Sources display */
.source-container {
  margin-top: 0.75rem;
  background-color: rgba(255, 255, 255, 0.8);
  border-radius: var(--border-radius-md);
  padding: 0.75rem;
  border: 1px solid var(--gray-200);
}

.source-title {
  font-weight: 600;
  font-size: 0.85rem;
  color: var(--gray-700);
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
}

.source-title i {
  margin-right: 0.5rem;
  color: var(--primary-color);
}

.source-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 0;
  padding: 0;
  list-style: none;
}

.source-item {
  font-size: 0.8rem;
  padding: 0.35rem 0.75rem;
  background-color: var(--gray-100);
  border-radius: var(--border-radius-sm);
  display: flex;
  align-items: center;
  gap: 0.35rem;
  border: 1px solid var(--gray-200);
  transition: background-color 0.3s;
}

.source-item:hover {
  background-color: var(--gray-200);
}

.source-item i {
  color: var(--primary-color);
  font-size: 0.85rem;
}

.source-item a {
  color: var(--gray-800);
  text-decoration: none;
}

.source-item a:hover {
  text-decoration: underline;
}

/* Source type badges */
.source-badge {
  display: inline-flex;
  align-items: center;
  padding: 0.125rem 0.5rem;
  border-radius: 50px;
  font-size: 0.7rem;
  font-weight: 500;
  margin-right: 0.5rem;
}

.source-badge.confluence {
  background-color: rgba(76, 154, 255, 0.15);
  color: #1a73e8;
}

.source-badge.jira {
  background-color: rgba(0, 102, 204, 0.15);
  color: #0052cc;
}

.source-badge.remedy {
  background-color: rgba(245, 130, 31, 0.15);
  color: #f5821f;
}

/* Typing indicator */
.typing-indicator {
  display: inline-flex;
  align-items: center;
  padding: 0.5rem 1rem;
  background-color: white;
  border-radius: var(--border-radius-md);
  box-shadow: var(--shadow-sm);
  margin-top: 0.5rem;
  animation: fadeIn 0.3s;
}

.typing-indicator span {
  font-size: 0.85rem;
  color: var(--gray-600);
  margin-left: 0.5rem;
}

.typing-indicator .dots {
  display: inline-flex;
  align-items: center;
}

.typing-indicator .dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background-color: var(--primary-color);
  margin-right: 4px;
  animation: typingAnimation 1.4s infinite ease-in-out;
}

.typing-indicator .dot:nth-child(1) {
  animation-delay: 0s;
}

.typing-indicator .dot:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator .dot:nth-child(3) {
  animation-delay: 0.4s;
  margin-right: 0;
}

@keyframes typingAnimation {
  0%, 60%, 100% {
    transform: translateY(0);
  }
  30% {
    transform: translateY(-5px);
  }
}

/* Data source selection */
.source-selector {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}

.source-selector-label {
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--gray-700);
  margin-right: 0.5rem;
  display: flex;
  align-items: center;
}

.source-option {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.4rem 0.75rem;
  border-radius: var(--border-radius-md);
  font-size: 0.85rem;
  cursor: pointer;
  border: 1px solid var(--gray-300);
  transition: all 0.3s;
}

.source-option:hover {
  background-color: var(--gray-100);
}

.source-option.active {
  background-color: var(--primary-color);
  color: white;
  border-color: var(--primary-color);
}

.source-option i {
  font-size: 0.9rem;
}

/* Conversation history */
.conversation-history {
  font-size: 0.85rem;
  color: var(--gray-600);
  text-align: center;
  margin: 1rem 0;
  position: relative;
}

.conversation-history:before,
.conversation-history:after {
  content: '';
  display: block;
  height: 1px;
  background-color: var(--gray-300);
  width: 42%;
  position: absolute;
  top: 50%;
}

.conversation-history:before {
  left: 0;
}

.conversation-history:after {
  right: 0;
}

.conversation-history span {
  background-color: var(--gray-100);
  padding: 0 1rem;
  position: relative;
  z-index: 1;
}

/* File upload */
.file-upload {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
  color: var(--gray-700);
  padding: 0.35rem 0.75rem;
  border-radius: var(--border-radius-md);
  background-color: var(--gray-100);
  border: 1px dashed var(--gray-300);
  cursor: pointer;
  transition: all 0.3s;
}

.file-upload:hover {
  background-color: var(--gray-200);
  border-color: var(--gray-400);
}

.file-upload i {
  color: var(--primary-color);
  font-size: 1rem;
}

.file-upload input {
  display: none;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
  .chat-container {
    height: calc(100vh - 150px);
  }
  
  .message {
    max-width: 90%;
  }
  
  .source-item {
    padding: 0.25rem 0.5rem;
  }
  
  .chat-input-container {
    padding: 0.75rem;
  }
  
  .chat-input {
    padding: 0.5rem 0.75rem;
  }
  
  .chat-submit {
    padding: 0.5rem 1rem;
  }
}

/* Responsive adjustments for very small screens */
@media (max-width: 480px) {
  .message {
    max-width: 95%;
  }
  
  .chat-header h2 {
    font-size: 1rem;
  }
  
  .chat-messages {
    padding: 1rem;
  }
  
  .source-container {
    padding: 0.5rem;
  }
  
  .source-list {
    flex-direction: column;
    gap: 0.25rem;
  }
  
  .source-selector {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .source-selector-label {
    margin-bottom: 0.5rem;
  }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
  .chat-messages {
    background-color: var(--gray-800);
  }
  
  .message.assistant .message-content {
    background-color: var(--gray-700);
    color: white;
    border-color: var(--gray-600);
  }
  
  .source-container {
    background-color: rgba(40, 40, 40, 0.8);
    border-color: var(--gray-600);
  }
  
  .source-title {
    color: var(--gray-300);
  }
  
  .source-item {
    background-color: var(--gray-700);
    border-color: var(--gray-600);
  }
  
  .source-item a {
    color: var(--gray-300);
  }
  
  .chat-input-container {
    background-color: var(--gray-800);
    border-color: var(--gray-700);
  }
  
  .chat-input {
    background-color: var(--gray-700);
    color: white;
    border-color: var(--gray-600);
  }
  
  .typing-indicator {
    background-color: var(--gray-700);
  }
  
  .typing-indicator span {
    color: var(--gray-300);
  }
}

/* Chat options menu */
.chat-options {
  position: relative;
}

.chat-options-toggle {
  background: none;
  border: none;
  color: white;
  cursor: pointer;
  padding: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.chat-options-menu {
  position: absolute;
  top: 100%;
  right: 0;
  width: 200px;
  background-color: white;
  border-radius: var(--border-radius-md);
  box-shadow: var(--shadow-md);
  overflow: hidden;
  z-index: 100;
  transform-origin: top right;
  transition: transform 0.3s, opacity 0.3s;
  transform: scale(0.95);
  opacity: 0;
  pointer-events: none;
}

.chat-options-menu.active {
  transform: scale(1);
  opacity: 1;
  pointer-events: auto;
}

.chat-options-item {
  padding: 0.75rem 1rem;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  cursor: pointer;
  transition: background-color 0.3s;
}

.chat-options-item:hover {
  background-color: var(--gray-100);
}

.chat-options-item i {
  color: var(--primary-color);
  font-size: 1rem;
}

/* Conversation tools */
.conversation-tools {
  display: flex;
  justify-content: flex-end;
  margin-top: 0.5rem;
  gap: 0.5rem;
}

.tool-button {
  background: none;
  border: none;
  color: var(--gray-500);
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.35rem 0.5rem;
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  transition: all 0.3s;
}

.tool-button:hover {
  color: var(--primary-color);
  background-color: rgba(67, 97, 238, 0.1);
}

.tool-button i {
  font-size: 0.9rem;
}

/* Download conversation button */
.download-conversation {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
  padding: 0.5rem 0.75rem;
  border-radius: var(--border-radius-md);
  background-color: var(--gray-200);
  color: var(--gray-700);
  border: none;
  cursor: pointer;
  transition: all 0.3s;
}

.download-conversation:hover {
  background-color: var(--gray-300);
}

.download-conversation i {
  font-size: 0.9rem;
}



















/**
 * Main JavaScript functionality for the Enterprise RAG System
 */

// Global variables
const apiBaseUrl = '/api';

// Initialize on document ready
document.addEventListener('DOMContentLoaded', function() {
    initializeTooltips();
    checkSourcesStatus();
    setupThemeToggle();
    handleFlashMessages();
    initMarkdown();
});

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
}

/**
 * Check and display status of data sources
 */
function checkSourcesStatus() {
    const sourcesStatusContainer = document.getElementById('sources-status');
    if (!sourcesStatusContainer) return;

    fetchWithTimeout(`${apiBaseUrl}/sources`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
    }, 5000)
    .then(response => response.json())
    .then(data => {
        if (data.sources) {
            let html = '';
            for (const [sourceId, sourceInfo] of Object.entries(data.sources)) {
                const status = sourceInfo.available ? 
                    '<span class="badge bg-success">Available</span>' : 
                    '<span class="badge bg-danger">Unavailable</span>';
                html += `
                <div class="source-item d-flex justify-content-between align-items-center mb-2">
                    <div>
                        <i class="source-icon fa ${getSourceIcon(sourceId)} me-2"></i>
                        <strong>${sourceInfo.name}</strong>
                    </div>
                    <div>${status}</div>
                </div>`;
            }
            sourcesStatusContainer.innerHTML = html;
        }
    })
    .catch(error => {
        console.error('Error checking sources:', error);
        sourcesStatusContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle me-2"></i>
                Error checking data sources. Please try again.
            </div>`;
    });
}

/**
 * Get Font Awesome icon for a source
 */
function getSourceIcon(sourceId) {
    switch (sourceId) {
        case 'confluence':
            return 'fa-book';
        case 'jira':
            return 'fa-ticket-alt';
        case 'remedy':
            return 'fa-tools';
        default:
            return 'fa-database';
    }
}

/**
 * Setup theme toggle functionality
 */
function setupThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');
    if (!themeToggle) return;

    // Check for saved theme preference or use preferred color scheme
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        document.body.classList.add('dark-theme');
        themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    } else {
        themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    }

    // Toggle theme on click
    themeToggle.addEventListener('click', function() {
        document.body.classList.toggle('dark-theme');
        
        if (document.body.classList.contains('dark-theme')) {
            localStorage.setItem('theme', 'dark');
            themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
        } else {
            localStorage.setItem('theme', 'light');
            themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
        }
    });
}

/**
 * Handle flash messages/notifications
 */
function handleFlashMessages() {
    const flashMessages = document.querySelectorAll('.flash-message');
    
    flashMessages.forEach(message => {
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            message.classList.add('fade-out');
            setTimeout(() => message.remove(), 500);
        }, 5000);
        
        // Allow manual dismiss
        const dismissBtn = message.querySelector('.dismiss');
        if (dismissBtn) {
            dismissBtn.addEventListener('click', () => {
                message.classList.add('fade-out');
                setTimeout(() => message.remove(), 500);
            });
        }
    });
}

/**
 * Initialize markdown rendering
 */
function initMarkdown() {
    // Configure marked options if available
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            renderer: new marked.Renderer(),
            highlight: function(code, language) {
                if (typeof hljs !== 'undefined') {
                    const validLanguage = hljs.getLanguage(language) ? language : 'plaintext';
                    return hljs.highlight(validLanguage, code).value;
                }
                return code;
            },
            gfm: true,
            breaks: true,
            sanitize: false
        });
    }
}

/**
 * Fetch with timeout for better error handling
 */
function fetchWithTimeout(url, options, timeout = 10000) {
    return Promise.race([
        fetch(url, options),
        new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Request timed out')), timeout)
        )
    ]);
}

/**
 * Show a user notification
 */
function showNotification(message, type = 'info') {
    const notificationContainer = document.getElementById('notification-container') || 
                               document.createElement('div');
    
    if (!document.getElementById('notification-container')) {
        notificationContainer.id = 'notification-container';
        notificationContainer.className = 'notification-container';
        document.body.appendChild(notificationContainer);
    }
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    const iconClass = type === 'success' ? 'fa-check-circle' :
                    type === 'warning' ? 'fa-exclamation-triangle' :
                    type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle';
    
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${iconClass} notification-icon"></i>
            <span>${message}</span>
        </div>
        <button class="notification-dismiss"><i class="fas fa-times"></i></button>
    `;
    
    notificationContainer.appendChild(notification);
    
    // Add slide-in effect
    setTimeout(() => notification.classList.add('show'), 10);
    
    // Auto-dismiss after 5 seconds
    const dismissTimeout = setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 5000);
    
    // Manual dismiss
    const dismissBtn = notification.querySelector('.notification-dismiss');
    dismissBtn.addEventListener('click', () => {
        clearTimeout(dismissTimeout);
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    });
}

/**
 * Format date and time
 */
function formatDateTime(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleString();
}

/**
 * Format a source link for display
 */
function formatSourceLink(source) {
    if (!source) return '';
    
    let icon = 'fa-link';
    let typeLabel = 'Source';
    
    if (source.type === 'confluence') {
        icon = 'fa-book';
        typeLabel = 'Confluence';
    } else if (source.type === 'jira') {
        icon = 'fa-ticket-alt';
        typeLabel = 'JIRA';
    } else if (source.type === 'remedy') {
        icon = 'fa-tools';
        typeLabel = 'Remedy';
    }
    
    return `
        <a href="${source.link}" class="source-link" target="_blank" 
           data-bs-toggle="tooltip" title="${typeLabel}">
            <i class="fas ${icon} me-1"></i>
            ${source.source}
        </a>
    `;
}

/**
 * Truncate text to specified length
 */
function truncateText(text, maxLength = 100) {
    if (!text || text.length <= maxLength) return text;
    return text.substr(0, maxLength - 3) + '...';
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    return navigator.clipboard.writeText(text)
        .then(() => {
            showNotification('Copied to clipboard!', 'success');
            return true;
        })
        .catch(err => {
            console.error('Error copying to clipboard:', err);
            showNotification('Failed to copy to clipboard', 'error');
            return false;
        });
}

/**
 * Download text as file
 */
function downloadAsFile(content, filename, contentType = 'text/plain') {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 100);
}



















/**
 * Chat interface functionality for the Enterprise RAG System
 */

// Chat state
const chatState = {
    messages: [],
    isProcessing: false,
    selectedSources: [],
    searchType: 'hybrid',
    streamResponse: true,
    maxContext: 10,
    currentReader: null
};

// Initialize chat interface
document.addEventListener('DOMContentLoaded', function() {
    initializeChatInterface();
});

/**
 * Initialize chat interface and event handlers
 */
function initializeChatInterface() {
    // Initialize elements
    const chatContainer = document.getElementById('chat-container');
    if (!chatContainer) return;
    
    // Initialize chat display, only if on chat page
    initializeChatDisplay();
    
    // Set up event listeners
    setupChatInputHandlers();
    setupSourceSelectionHandlers();
    setupOptionHandlers();
    
    // Load chat history from localStorage
    loadChatHistory();
    
    // Check sources status for source selection
    checkAndUpdateSources();
}

/**
 * Initialize chat message display
 */
function initializeChatDisplay() {
    const chatMessages = document.getElementById('chat-messages');
    const welcomeMessage = {
        role: 'assistant',
        content: "Hello! I'm your Enterprise RAG Assistant. I can help you find information from Confluence, JIRA, and Remedy. What would you like to know?",
        timestamp: new Date().toISOString()
    };
    
    // Add welcome message
    appendMessageToChat(welcomeMessage);
    chatState.messages.push(welcomeMessage);
}

/**
 * Setup event handlers for the chat input
 */
function setupChatInputHandlers() {
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    
    // Send message on Enter (but allow Shift+Enter for newlines)
    chatInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Send message on button click
    sendButton.addEventListener('click', sendMessage);
    
    // Adjust textarea height as user types
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
}

/**
 * Setup source selection handlers
 */
function setupSourceSelectionHandlers() {
    const sourceCheckboxes = document.querySelectorAll('.source-checkbox');
    
    sourceCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            updateSelectedSources();
        });
    });
    
    // Select all/none buttons
    const selectAllBtn = document.getElementById('select-all-sources');
    const selectNoneBtn = document.getElementById('select-none-sources');
    
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', function() {
            sourceCheckboxes.forEach(cb => cb.checked = true);
            updateSelectedSources();
        });
    }
    
    if (selectNoneBtn) {
        selectNoneBtn.addEventListener('click', function() {
            sourceCheckboxes.forEach(cb => cb.checked = false);
            updateSelectedSources();
        });
    }
    
    // Initial update of selected sources
    updateSelectedSources();
}

/**
 * Update selected sources based on checkboxes
 */
function updateSelectedSources() {
    const sourceCheckboxes = document.querySelectorAll('.source-checkbox:checked');
    chatState.selectedSources = Array.from(sourceCheckboxes).map(cb => cb.value);
    
    // Update the source indicators
    const sourceIndicators = document.getElementById('selected-sources');
    if (sourceIndicators) {
        if (chatState.selectedSources.length === 0) {
            sourceIndicators.textContent = 'All sources';
        } else {
            sourceIndicators.textContent = chatState.selectedSources
                .map(s => s.charAt(0).toUpperCase() + s.slice(1))
                .join(', ');
        }
    }
}

/**
 * Check and update sources in the interface
 */
function checkAndUpdateSources() {
    fetch(`${apiBaseUrl}/sources`)
        .then(response => response.json())
        .then(data => {
            if (data.sources) {
                // Update source checkboxes
                for (const [sourceId, sourceInfo] of Object.entries(data.sources)) {
                    const checkbox = document.querySelector(`.source-checkbox[value="${sourceId}"]`);
                    if (checkbox) {
                        checkbox.disabled = !sourceInfo.available;
                        
                        // Update the label
                        const label = checkbox.closest('label');
                        if (label) {
                            if (!sourceInfo.available) {
                                label.classList.add('text-muted');
                                label.setAttribute('data-bs-toggle', 'tooltip');
                                label.setAttribute('title', 'Source currently unavailable');
                            } else {
                                label.classList.remove('text-muted');
                                label.removeAttribute('data-bs-toggle');
                                label.removeAttribute('title');
                            }
                        }
                    }
                }
                
                // Reinitialize tooltips
                initializeTooltips();
                
                // Update selected sources
                updateSelectedSources();
            }
        })
        .catch(error => {
            console.error('Error checking sources:', error);
        });
}

/**
 * Setup option handlers (search type, streaming)
 */
function setupOptionHandlers() {
    // Search type selector
    const searchTypeSelect = document.getElementById('search-type');
    if (searchTypeSelect) {
        searchTypeSelect.addEventListener('change', function() {
            chatState.searchType = this.value;
        });
    }
    
    // Streaming toggle
    const streamToggle = document.getElementById('stream-toggle');
    if (streamToggle) {
        streamToggle.addEventListener('change', function() {
            chatState.streamResponse = this.checked;
        });
    }
    
    // Clear chat button
    const clearChatBtn = document.getElementById('clear-chat');
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', clearChat);
    }
}

/**
 * Send a message to the API
 */
function sendMessage() {
    const chatInput = document.getElementById('chat-input');
    const message = chatInput.value.trim();
    
    if (!message || chatState.isProcessing) return;
    
    // Disable input during processing
    chatState.isProcessing = true;
    chatInput.disabled = true;
    
    // Reset input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    // Create user message
    const userMessage = {
        role: 'user',
        content: message,
        timestamp: new Date().toISOString()
    };
    
    // Display user message
    appendMessageToChat(userMessage);
    
    // Add to messages history
    chatState.messages.push(userMessage);
    
    // Create a placeholder for the assistant's response
    const responseId = `response-${Date.now()}`;
    const assistantPlaceholder = {
        id: responseId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        sources: []
    };
    
    // Display assistant placeholder with typing indicator
    appendMessageToChat(assistantPlaceholder, true);
    
    // Prepare request data
    const requestData = {
        query: message,
        sources: chatState.selectedSources.length > 0 ? chatState.selectedSources : null,
        search_type: chatState.searchType,
        stream: chatState.streamResponse
    };
    
    // Handle streaming or non-streaming response
    if (chatState.streamResponse) {
        handleStreamingResponse(requestData, responseId);
    } else {
        handleNonStreamingResponse(requestData, responseId);
    }
}

/**
 * Handle streaming response from the API
 */
function handleStreamingResponse(requestData, responseId) {
    // Start the fetch request
    fetch(`${apiBaseUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        // Get the streaming response
        const reader = response.body.getReader();
        chatState.currentReader = reader;
        
        // Create a decoder for the stream
        const decoder = new TextDecoder();
        let responseText = '';
        
        // Function to process stream chunks
        function processStream({ done, value }) {
            // If stream is done
            if (done) {
                // Update the state
                chatState.isProcessing = false;
                document.getElementById('chat-input').disabled = false;
                
                // Remove typing indicator
                const responseElement = document.getElementById(responseId);
                if (responseElement) {
                    const typingIndicator = responseElement.querySelector('.typing-indicator');
                    if (typingIndicator) typingIndicator.remove();
                }
                
                // Add the complete message to history
                const assistantMessage = {
                    role: 'assistant',
                    content: responseText,
                    timestamp: new Date().toISOString(),
                    sources: []  // No sources in streaming mode
                };
                chatState.messages.push(assistantMessage);
                
                // Save chat history
                saveChatHistory();
                
                return;
            }
            
            // Decode the chunk and append to response text
            const chunk = decoder.decode(value, { stream: true });
            responseText += chunk;
            
            // Update the UI with the current response text
            const responseElement = document.getElementById(responseId);
            if (responseElement) {
                const contentElement = responseElement.querySelector('.message-content');
                if (contentElement) {
                    // Render markdown
                    contentElement.innerHTML = marked.parse(responseText);
                    
                    // Scroll to the bottom
                    scrollToBottom();
                }
            }
            
            // Continue reading the stream
            return reader.read().then(processStream);
        }
        
        // Start processing the stream
        return reader.read().then(processStream);
    })
    .catch(error => {
        console.error('Error sending message:', error);
        
        // Update UI to show error
        const responseElement = document.getElementById(responseId);
        if (responseElement) {
            const contentElement = responseElement.querySelector('.message-content');
            if (contentElement) {
                contentElement.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
                
                // Remove typing indicator
                const typingIndicator = responseElement.querySelector('.typing-indicator');
                if (typingIndicator) typingIndicator.remove();
            }
        }
        
        // Reset state
        chatState.isProcessing = false;
        document.getElementById('chat-input').disabled = false;
    });
}

/**
 * Handle non-streaming response from the API
 */
function handleNonStreamingResponse(requestData, responseId) {
    fetch(`${apiBaseUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // Update the state
        chatState.isProcessing = false;
        document.getElementById('chat-input').disabled = false;
        
        // Create assistant message
        const assistantMessage = {
            role: 'assistant',
            content: data.response,
            timestamp: new Date().toISOString(),
            sources: data.context || []
        };
        
        // Update the placeholder message
        updateAssistantMessage(responseId, assistantMessage);
        
        // Add to messages history
        chatState.messages.push(assistantMessage);
        
        // Save chat history
        saveChatHistory();
    })
    .catch(error => {
        console.error('Error sending message:', error);
        
        // Update UI to show error
        const responseElement = document.getElementById(responseId);
        if (responseElement) {
            const contentElement = responseElement.querySelector('.message-content');
            if (contentElement) {
                contentElement.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
                
                // Remove typing indicator
                const typingIndicator = responseElement.querySelector('.typing-indicator');
                if (typingIndicator) typingIndicator.remove();
            }
        }
        
        // Reset state
        chatState.isProcessing = false;
        document.getElementById('chat-input').disabled = false;
    });
}

/**
 * Update an existing assistant message (used for non-streaming mode)
 */
function updateAssistantMessage(messageId, message) {
    const messageElement = document.getElementById(messageId);
    if (!messageElement) return;
    
    // Update content
    const contentElement = messageElement.querySelector('.message-content');
    if (contentElement) {
        // Render markdown
        contentElement.innerHTML = marked.parse(message.content);
        
        // Remove typing indicator
        const typingIndicator = messageElement.querySelector('.typing-indicator');
        if (typingIndicator) typingIndicator.remove();
    }
    
    // Add sources if available
    if (message.sources && message.sources.length > 0) {
        const sourcesContainer = document.createElement('div');
        sourcesContainer.className = 'message-sources mt-2';
        
        const sourcesList = document.createElement('div');
        sourcesList.className = 'sources-list';
        
        // Add sources heading
        sourcesList.innerHTML = `<small class="sources-heading">Sources:</small>`;
        
        // Add each source
        message.sources.forEach(source => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            
            // Get icon based on source type
            let icon = 'fa-link';
            if (source.source_type === 'confluence') icon = 'fa-book';
            else if (source.source_type === 'jira') icon = 'fa-ticket-alt';
            else if (source.source_type === 'remedy') icon = 'fa-tools';
            
            // Source display name
            const sourceName = source.source || 'Source';
            
            // Create source link
            sourceItem.innerHTML = `
                <a href="${source.source_link || '#'}" target="_blank" class="source-link">
                    <i class="fas ${icon} me-1"></i> ${sourceName}
                </a>
            `;
            
            sourcesList.appendChild(sourceItem);
        });
        
        sourcesContainer.appendChild(sourcesList);
        messageElement.appendChild(sourcesContainer);
    }
    
    // Scroll to bottom
    scrollToBottom();
}

/**
 * Append a message to the chat display
 */
function appendMessageToChat(message, isTyping = false) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    // Create message container
    const messageElement = document.createElement('div');
    messageElement.className = `message message-${message.role}`;
    if (message.id) messageElement.id = message.id;
    
    // Message avatar/icon
    let avatarIcon = 'fa-user';
    if (message.role === 'assistant') avatarIcon = 'fa-robot';
    else if (message.role === 'system') avatarIcon = 'fa-cog';
    
    // Format timestamp
    const timestamp = new Date(message.timestamp).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit'
    });
    
    // Build message HTML
    messageElement.innerHTML = `
        <div class="message-avatar">
            <i class="fas ${avatarIcon}"></i>
        </div>
        <div class="message-body">
            <div class="message-header">
                <span class="message-sender">${message.role === 'assistant' ? 'RAG Assistant' : 'You'}</span>
                <span class="message-time">${timestamp}</span>
            </div>
            <div class="message-content"></div>
        </div>
    `;
    
    // Add content
    const contentElement = messageElement.querySelector('.message-content');
    
    if (isTyping) {
        // Add typing indicator for assistant
        contentElement.innerHTML = `
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
    } else {
        // Regular content with markdown
        contentElement.innerHTML = message.role === 'assistant' ? 
            marked.parse(message.content) : 
            message.content.replace(/\n/g, '<br>');
        
        // Add sources if available (for assistant)
        if (message.role === 'assistant' && message.sources && message.sources.length > 0) {
            const sourcesContainer = document.createElement('div');
            sourcesContainer.className = 'message-sources mt-2';
            
            const sourcesList = document.createElement('div');
            sourcesList.className = 'sources-list';
            
            // Add sources heading
            sourcesList.innerHTML = `<small class="sources-heading">Sources:</small>`;
            
            // Add each source
            message.sources.forEach(source => {
                const sourceItem = document.createElement('div');
                sourceItem.className = 'source-item';
                
                // Get icon based on source type
                let icon = 'fa-link';
                if (source.source_type === 'confluence') icon = 'fa-book';
                else if (source.source_type === 'jira') icon = 'fa-ticket-alt';
                else if (source.source_type === 'remedy') icon = 'fa-tools';
                
                // Source display name
                const sourceName = source.source || 'Source';
                
                // Create source link
                sourceItem.innerHTML = `
                    <a href="${source.source_link || '#'}" target="_blank" class="source-link">
                        <i class="fas ${icon} me-1"></i> ${sourceName}
                    </a>
                `;
                
                sourcesList.appendChild(sourceItem);
            });
            
            sourcesContainer.appendChild(sourcesList);
            messageElement.querySelector('.message-body').appendChild(sourcesContainer);
        }
    }
    
    // Add message actions for user messages
    if (message.role === 'user') {
        const actionsContainer = document.createElement('div');
        actionsContainer.className = 'message-actions';
        
        // Edit button (currently disabled)
        actionsContainer.innerHTML = `
            <button class="btn btn-sm action-btn edit-message" disabled title="Edit message">
                <i class="fas fa-edit"></i>
            </button>
        `;
        
        messageElement.querySelector('.message-body').appendChild(actionsContainer);
    }
    
    // Add message actions for assistant messages
    if (message.role === 'assistant' && !isTyping) {
        const actionsContainer = document.createElement('div');
        actionsContainer.className = 'message-actions';
        
        // Copy and download buttons
        actionsContainer.innerHTML = `
            <button class="btn btn-sm action-btn copy-message" title="Copy response">
                <i class="fas fa-copy"></i>
            </button>
            <button class="btn btn-sm action-btn download-message" title="Download response">
                <i class="fas fa-download"></i>
            </button>
        `;
        
        messageElement.querySelector('.message-body').appendChild(actionsContainer);
        
        // Add event listeners
        const copyBtn = actionsContainer.querySelector('.copy-message');
        const downloadBtn = actionsContainer.querySelector('.download-message');
        
        copyBtn.addEventListener('click', () => {
            copyToClipboard(message.content);
        });
        
        downloadBtn.addEventListener('click', () => {
            const filename = `rag-response-${new Date().toISOString().replace(/:/g, '-')}.md`;
            downloadAsFile(message.content, filename, 'text/markdown');
        });
    }
    
    // Append to chat
    chatMessages.appendChild(messageElement);
    
    // Scroll to bottom
    scrollToBottom();
}

/**
 * Scroll to the bottom of the chat
 */
function scrollToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

/**
 * Clear the chat history
 */
function clearChat() {
    // Confirm
    if (!confirm('Are you sure you want to clear the chat history? This cannot be undone.')) {
        return;
    }
    
    // Cancel any ongoing streaming response
    if (chatState.currentReader) {
        chatState.currentReader.cancel();
    }
    
    // Reset state
    chatState.messages = [];
    chatState.isProcessing = false;
    
    // Clear chat display
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        chatMessages.innerHTML = '';
    }
    
    // Re-enable input
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.disabled = false;
    }
    
    // Add welcome message again
    const welcomeMessage = {
        role: 'assistant',
        content: "I've cleared our conversation. How else can I help you?",
        timestamp: new Date().toISOString()
    };
    
    appendMessageToChat(welcomeMessage);
    chatState.messages.push(welcomeMessage);
    
    // Clear localStorage
    localStorage.removeItem('chatHistory');
}

/**
 * Save chat history to localStorage
 */
function saveChatHistory() {
    // Limit history to last maxContext messages
    const historyToSave = chatState.messages.slice(-chatState.maxContext * 2);
    localStorage.setItem('chatHistory', JSON.stringify(historyToSave));
}

/**
 * Load chat history from localStorage
 */
function loadChatHistory() {
    const savedHistory = localStorage.getItem('chatHistory');
    if (!savedHistory) return;
    
    try {
        const history = JSON.parse(savedHistory);
        
        // Clear existing messages
        const chatMessages = document.getElementById('chat-messages');
        if (chatMessages) {
            chatMessages.innerHTML = '';
        }
        
        // Add messages to UI and state
        chatState.messages = [];
        history.forEach(message => {
            appendMessageToChat(message);
            chatState.messages.push(message);
        });
        
        // Scroll to bottom
        scrollToBottom();
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

/**
 * Cancel current request if in progress
 */
function cancelRequest() {
    if (!chatState.isProcessing) return;
    
    // Cancel reader if streaming
    if (chatState.currentReader) {
        chatState.currentReader.cancel();
        chatState.currentReader = null;
    }
    
    // Reset state
    chatState.isProcessing = false;
    
    // Enable input
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.disabled = false;
    }
    
    // Add cancellation message
    const cancelMessage = {
        role: 'system',
        content: "Request cancelled.",
        timestamp: new Date().toISOString()
    };
    
    appendMessageToChat(cancelMessage);
    
    // Show notification
    showNotification('Request cancelled', 'info');
}











