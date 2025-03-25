rag-confluence-remedy/
├── .env
├── app.py
├── config.py
├── requirements.txt
├── run.py
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── images/
│       └── logo.png
├── templates/
│   ├── base.html
│   ├── index.html
│   └── chat.html
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   └── helpers.py
└── modules/
    ├── __init__.py
    ├── data_sources/
    │   ├── __init__.py
    │   ├── confluence.py
    │   └── remedy.py
    ├── processing/
    │   ├── __init__.py
    │   ├── chunking.py
    │   ├── embedding.py
    │   └── indexing.py
    ├── retrieval/
    │   ├── __init__.py
    │   ├── vector_search.py
    │   ├── lexical_search.py
    │   └── hybrid_search.py
    ├── llm/
    │   ├── __init__.py
    │   ├── gemini.py
    │   └── prompt_templates.py
    └── api/
        ├── __init__.py
        ├── routes.py
        └── schemas.py














# Flask and related packages
Flask==2.3.3
Flask-Cors==4.0.0
Flask-RESTful==0.3.10
python-dotenv==1.0.0
gunicorn==21.2.0

# Data processing
beautifulsoup4==4.12.2
lxml==4.9.3
html2text==2020.1.16
jsonschema==4.19.0
markdown==3.4.4

# API and HTTP
requests==2.31.0
requests-cache==1.1.0
urllib3==2.0.5
jwt==1.3.1
PyJWT==2.8.0

# Vector database and embeddings
faiss-cpu==1.7.4
sentence-transformers==2.2.2
scikit-learn==1.3.0
numpy==1.25.2
pandas==2.1.0

# Text processing
spacy==3.6.1
nltk==3.8.1
rank-bm25==0.2.2

# Google AI and Gemini
google-cloud-aiplatform==1.36.4
vertexai==0.0.1
google-api-python-client==2.97.0
google-auth==2.23.0
google-auth-oauthlib==1.0.0

# Caching and optimization
redis==5.0.0
cachetools==5.3.1

# Utilities
tqdm==4.66.1
pydantic==2.3.0
tenacity==8.2.3
loguru==0.7.0
python-slugify==8.0.1










# Flask configuration
FLASK_APP=app.py
FLASK_ENV=development

# Confluence settings
CONFLUENCE_URL="https://csagroup-restapi.atlassian.net"
CONFLUENCE_SPACE_ID=""
CONFLUENCE_USER_ID=""
CONFLUENCE_API_TOKEN=""

# Remedy settings
REMEDY_SERVER="csagroup-restapi.onbmc.com"
REMEDY_API_BASE="https://csagroup-restapi.onbmc.com"
REMEDY_USERNAME=""
REMEDY_PASSWORD=""

# Project settings
PROJECT_ID="prj-dv-cpa-4363"
REGION="us-central1"
MODEL_NAME="gemini-2.0-flash-001"

# Logging
LOG_LEVEL="INFO"

















"""
Configuration module for the RAG application.
Loads environment variables and provides configuration settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import tempfile

# Load environment variables from .env file
load_dotenv()

# Base directory of the application
BASE_DIR = Path(__file__).resolve().parent

# Flask configuration
DEBUG = os.getenv("FLASK_ENV") == "development"
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-replace-this")
PORT = int(os.getenv("PORT", 5000))

# Confluence API settings
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_SPACE_ID = os.getenv("CONFLUENCE_SPACE_ID")
CONFLUENCE_USER_ID = os.getenv("CONFLUENCE_USER_ID")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

# Remedy API settings
REMEDY_SERVER = os.getenv("REMEDY_SERVER")
REMEDY_API_BASE = os.getenv("REMEDY_API_BASE")
REMEDY_USERNAME = os.getenv("REMEDY_USERNAME")
REMEDY_PASSWORD = os.getenv("REMEDY_PASSWORD")

# Google Cloud and Gemini settings
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-001")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Cache settings
CACHE_DIR = os.path.join(tempfile.gettempdir(), "rag_cache")
Path(CACHE_DIR).mkdir(exist_ok=True)

# Vector store settings
VECTOR_STORE_PATH = os.path.join(CACHE_DIR, "vector_store")
Path(VECTOR_STORE_PATH).mkdir(exist_ok=True)

# Embedding model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for the embedding model chosen

# RAG settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
NUM_RESULTS = 5
TEMPERATURE = 0.2

# Validate required configuration
def validate_config():
    """Validate critical configuration settings."""
    required_vars = {
        "CONFLUENCE_URL": CONFLUENCE_URL,
        "CONFLUENCE_API_TOKEN": CONFLUENCE_API_TOKEN,
        "REMEDY_API_BASE": REMEDY_API_BASE,
        "PROJECT_ID": PROJECT_ID,
        "MODEL_NAME": MODEL_NAME
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        missing_vars_str = ", ".join(missing_vars)
        logger.warning(f"Missing required environment variables: {missing_vars_str}")
        
    return not missing_vars













"""
Logging utility for the RAG application.
Sets up a standardized logging format and configuration.
"""
import os
import sys
from datetime import datetime
from loguru import logger

import config

# Remove default logger
logger.remove()

# Set log level from configuration
LOG_LEVEL = getattr(config, "LOG_LEVEL", "INFO")

# Create logs directory if it doesn't exist
logs_dir = os.path.join(config.BASE_DIR, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Get current date for log filename
current_date = datetime.now().strftime("%Y-%m-%d")
log_file = os.path.join(logs_dir, f"{current_date}.log")

# Configure logger format
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Add console handler
logger.add(
    sys.stderr,
    format=log_format,
    level=LOG_LEVEL,
    colorize=True,
)

# Add file handler
logger.add(
    log_file,
    format=log_format,
    level=LOG_LEVEL,
    rotation="00:00",  # New file at midnight
    retention="7 days",  # Keep logs for 7 days
    compression="zip",  # Compress old log files
)

# Function to get logger for a specific module
def get_logger(name):
    """
    Get a configured logger instance for the specified module name.
    
    Args:
        name (str): The name of the module or component
        
    Returns:
        loguru.Logger: A configured logger instance
    """
    return logger.bind(name=name)

















"""
Helper functions for the RAG application.
Provides utility functions used across different modules.
"""
import re
import time
import json
import hashlib
import unicodedata
from functools import wraps
from datetime import datetime
from pathlib import Path
import os
import uuid

from loguru import logger
import config


def generate_cache_key(text, prefix=""):
    """
    Generate a unique cache key based on text content.
    
    Args:
        text (str): The text to generate a key for
        prefix (str, optional): A prefix to add to the key
        
    Returns:
        str: A unique key suitable for caching
    """
    # Normalize the text to ensure consistent keys
    normalized_text = unicodedata.normalize("NFKD", text.lower().strip())
    
    # Create a hash of the text
    hash_obj = hashlib.md5(normalized_text.encode())
    hash_str = hash_obj.hexdigest()
    
    # Return the key with an optional prefix
    return f"{prefix}_{hash_str}" if prefix else hash_str


def timer_decorator(func):
    """
    Decorator to measure and log the execution time of a function.
    
    Args:
        func (callable): The function to be timed
        
    Returns:
        callable: The wrapped function with timing functionality
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.debug(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper


def clean_html(html_content):
    """
    Clean HTML content by removing scripts, styles, and unnecessary tags.
    
    Args:
        html_content (str): The HTML content to clean
        
    Returns:
        str: Cleaned text content
    """
    from bs4 import BeautifulSoup
    import html2text
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    
    # Convert to markdown first (preserves some structure)
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_tables = False
    markdown_text = h.handle(str(soup))
    
    # Clean up the markdown text
    return markdown_text.strip()


def extract_code_blocks(text):
    """
    Extract code blocks from markdown text.
    
    Args:
        text (str): Markdown text that may contain code blocks
        
    Returns:
        list: List of dictionaries containing code blocks and their language
    """
    # Pattern to match code blocks with or without language specification
    pattern = r"```(\w*)\n([\s\S]*?)```"
    
    # Find all matches
    matches = re.findall(pattern, text)
    
    code_blocks = []
    for language, code in matches:
        code_blocks.append({
            "language": language.strip() or "text",
            "code": code.strip()
        })
    
    return code_blocks


def extract_tables(html_content):
    """
    Extract tables from HTML content.
    
    Args:
        html_content (str): The HTML content with possible tables
        
    Returns:
        list: List of dictionaries containing table data
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    
    extracted_tables = []
    for table in tables:
        # Extract headers
        headers = []
        header_row = table.find("thead")
        if header_row:
            headers = [th.get_text().strip() for th in header_row.find_all("th")]
        
        # Extract rows
        rows = []
        body = table.find("tbody")
        if body:
            for row in body.find_all("tr"):
                cells = [cell.get_text().strip() for cell in row.find_all(["td", "th"])]
                rows.append(cells)
        else:
            # If no tbody, get all rows directly
            for row in table.find_all("tr"):
                cells = [cell.get_text().strip() for cell in row.find_all(["td", "th"])]
                rows.append(cells)
        
        extracted_tables.append({
            "headers": headers,
            "rows": rows
        })
    
    return extracted_tables


def ensure_directory(directory_path):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        str: The path to the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def save_to_disk(data, directory, filename=None):
    """
    Save data to disk in JSON format.
    
    Args:
        data (dict or list): The data to save
        directory (str): Directory to save to
        filename (str, optional): Filename to use. If None, generates a UUID
        
    Returns:
        str: Path to the saved file
    """
    # Ensure directory exists
    ensure_directory(directory)
    
    # Generate filename if not provided
    if filename is None:
        filename = f"{uuid.uuid4()}.json"
    
    # Ensure filename has .json extension
    if not filename.endswith(".json"):
        filename += ".json"
    
    # Create full path
    file_path = os.path.join(directory, filename)
    
    # Save data
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return file_path


def load_from_disk(file_path):
    """
    Load JSON data from disk.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict or list: The loaded data, or None if file not found
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return None


def is_valid_jwt(token):
    """
    Check if a JWT token is valid (not expired).
    
    Args:
        token (str): The JWT token to check
        
    Returns:
        bool: True if valid, False otherwise
    """
    import jwt
    from jwt.exceptions import PyJWTError
    
    try:
        # Decode without verification (we just want to check expiry)
        decoded = jwt.decode(token, options={"verify_signature": False})
        
        # Check if token is expired
        exp = decoded.get("exp", 0)
        current_time = datetime.utcnow().timestamp()
        
        return exp > current_time
    except PyJWTError:
        return False


def format_chunk_for_prompt(chunk, include_metadata=True):
    """
    Format a chunk for inclusion in an LLM prompt.
    
    Args:
        chunk (dict): The chunk to format
        include_metadata (bool): Whether to include metadata
        
    Returns:
        str: Formatted chunk text
    """
    text = chunk.get("text", "")
    
    if include_metadata and "metadata" in chunk:
        metadata = chunk["metadata"]
        source = metadata.get("source", "Unknown")
        title = metadata.get("title", "Untitled")
        url = metadata.get("url", "")
        
        header = f"Source: {source}\nTitle: {title}\n"
        if url:
            header += f"URL: {url}\n"
            
        return f"{header}\n{text}"
    else:
        return text

















"""
Utilities package for the RAG application.
"""
from .logger import get_logger
from .helpers import (
    generate_cache_key,
    timer_decorator,
    clean_html,
    extract_code_blocks,
    extract_tables,
    ensure_directory,
    save_to_disk,
    load_from_disk,
    is_valid_jwt,
    format_chunk_for_prompt
)

__all__ = [
    'get_logger',
    'generate_cache_key',
    'timer_decorator',
    'clean_html',
    'extract_code_blocks',
    'extract_tables',
    'ensure_directory',
    'save_to_disk',
    'load_from_disk',
    'is_valid_jwt',
    'format_chunk_for_prompt'
]










"""
Confluence API integration for the RAG application.
Provides functionality to fetch and process content from Confluence.
"""
import base64
import json
import time
from urllib.parse import urljoin

import requests
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from utils import get_logger, timer_decorator, clean_html, extract_tables, extract_code_blocks

# Initialize logger
logger = get_logger("confluence")

class ConfluenceClient:
    """Client for interacting with the Confluence API."""
    
    def __init__(self, base_url=None, user_id=None, api_token=None):
        """
        Initialize the Confluence client.
        
        Args:
            base_url (str, optional): Confluence base URL. Defaults to config.CONFLUENCE_URL.
            user_id (str, optional): Confluence user ID. Defaults to config.CONFLUENCE_USER_ID.
            api_token (str, optional): Confluence API token. Defaults to config.CONFLUENCE_API_TOKEN.
        """
        self.base_url = base_url or config.CONFLUENCE_URL
        self.user_id = user_id or config.CONFLUENCE_USER_ID
        self.api_token = api_token or config.CONFLUENCE_API_TOKEN
        
        # Ensure base URL ends with a slash
        if not self.base_url.endswith('/'):
            self.base_url += '/'
            
        # Set up authentication
        if self.user_id and self.api_token:
            auth_str = f"{self.user_id}:{self.api_token}"
            self.auth_header = base64.b64encode(auth_str.encode()).decode()
        else:
            self.auth_header = None
            logger.warning("Confluence API credentials not provided. API calls will fail.")
    
    def _get_headers(self):
        """
        Get headers for API requests.
        
        Returns:
            dict: Headers to use in API requests
        """
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        if self.auth_header:
            headers["Authorization"] = f"Basic {self.auth_header}"
            
        return headers
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _make_request(self, method, endpoint, params=None, data=None):
        """
        Make an HTTP request to the Confluence API.
        
        Args:
            method (str): HTTP method (GET, POST, etc.)
            endpoint (str): API endpoint (relative to base URL)
            params (dict, optional): Query parameters
            data (dict, optional): Body data for POST requests
            
        Returns:
            dict: JSON response
            
        Raises:
            RequestException: If the request fails
        """
        url = urljoin(self.base_url, endpoint)
        headers = self._get_headers()
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data,
                timeout=30  # 30 second timeout
            )
            
            # Raise exception for 4XX/5XX status codes
            response.raise_for_status()
            
            # Parse JSON response
            return response.json()
            
        except RequestException as e:
            logger.error(f"Confluence API request failed: {e}")
            raise
    
    @timer_decorator
    def get_spaces(self, limit=25):
        """
        Get a list of spaces.
        
        Args:
            limit (int, optional): Maximum number of spaces to return.
            
        Returns:
            list: List of space objects
        """
        try:
            result = self._make_request(
                method="GET",
                endpoint="wiki/rest/api/space",
                params={"limit": limit}
            )
            
            return result.get("results", [])
            
        except Exception as e:
            logger.error(f"Failed to get Confluence spaces: {e}")
            return []
    
    @timer_decorator
    def get_space(self, space_key):
        """
        Get details about a specific space.
        
        Args:
            space_key (str): Space key
            
        Returns:
            dict: Space details
        """
        try:
            return self._make_request(
                method="GET",
                endpoint=f"wiki/rest/api/space/{space_key}"
            )
            
        except Exception as e:
            logger.error(f"Failed to get Confluence space {space_key}: {e}")
            return {}
    
    @timer_decorator
    def get_pages(self, space_key, limit=100, start=0, expand=None):
        """
        Get pages in a space.
        
        Args:
            space_key (str): Space key
            limit (int, optional): Maximum number of pages to return
            start (int, optional): Starting index for pagination
            expand (str, optional): Properties to expand
            
        Returns:
            list: List of page objects
        """
        try:
            params = {
                "spaceKey": space_key,
                "limit": limit,
                "start": start
            }
            
            if expand:
                params["expand"] = expand
                
            result = self._make_request(
                method="GET",
                endpoint="wiki/rest/api/content",
                params=params
            )
            
            return result.get("results", [])
            
        except Exception as e:
            logger.error(f"Failed to get pages for space {space_key}: {e}")
            return []
    
    @timer_decorator
    def get_all_pages(self, space_key, expand=None):
        """
        Get all pages in a space with pagination.
        
        Args:
            space_key (str): Space key
            expand (str, optional): Properties to expand
            
        Returns:
            list: List of all page objects
        """
        all_pages = []
        start = 0
        limit = 100
        
        while True:
            pages = self.get_pages(space_key, limit=limit, start=start, expand=expand)
            
            if not pages:
                break
                
            all_pages.extend(pages)
            start += limit
            
            # Simple rate limiting
            time.sleep(0.5)
            
        return all_pages
    
    @timer_decorator
    def get_page_by_id(self, page_id, expand=None):
        """
        Get a specific page by ID.
        
        Args:
            page_id (str): Page ID
            expand (str, optional): Properties to expand
            
        Returns:
            dict: Page object
        """
        try:
            params = {}
            if expand:
                params["expand"] = expand
                
            return self._make_request(
                method="GET",
                endpoint=f"wiki/rest/api/content/{page_id}",
                params=params
            )
            
        except Exception as e:
            logger.error(f"Failed to get page {page_id}: {e}")
            return {}
    
    @timer_decorator
    def get_page_content(self, page_id):
        """
        Get the content of a page in HTML format.
        
        Args:
            page_id (str): Page ID
            
        Returns:
            str: HTML content of the page
        """
        try:
            page = self.get_page_by_id(
                page_id=page_id,
                expand="body.storage"
            )
            
            return page.get("body", {}).get("storage", {}).get("value", "")
            
        except Exception as e:
            logger.error(f"Failed to get content for page {page_id}: {e}")
            return ""
    
    @timer_decorator
    def get_page_attachments(self, page_id):
        """
        Get attachments for a page.
        
        Args:
            page_id (str): Page ID
            
        Returns:
            list: List of attachment objects
        """
        try:
            result = self._make_request(
                method="GET",
                endpoint=f"wiki/rest/api/content/{page_id}/child/attachment"
            )
            
            return result.get("results", [])
            
        except Exception as e:
            logger.error(f"Failed to get attachments for page {page_id}: {e}")
            return []
    
    @timer_decorator
    def get_page_comments(self, page_id):
        """
        Get comments for a page.
        
        Args:
            page_id (str): Page ID
            
        Returns:
            list: List of comment objects
        """
        try:
            result = self._make_request(
                method="GET",
                endpoint=f"wiki/rest/api/content/{page_id}/child/comment"
            )
            
            return result.get("results", [])
            
        except Exception as e:
            logger.error(f"Failed to get comments for page {page_id}: {e}")
            return []
    
    @timer_decorator
    def search(self, cql_query, limit=25, start=0, expand=None):
        """
        Search Confluence using CQL.
        
        Args:
            cql_query (str): CQL query string
            limit (int, optional): Maximum number of results
            start (int, optional): Starting index for pagination
            expand (str, optional): Properties to expand
            
        Returns:
            list: Search results
        """
        try:
            params = {
                "cql": cql_query,
                "limit": limit,
                "start": start
            }
            
            if expand:
                params["expand"] = expand
                
            result = self._make_request(
                method="GET",
                endpoint="wiki/rest/api/search",
                params=params
            )
            
            return result.get("results", [])
            
        except Exception as e:
            logger.error(f"Failed to search Confluence with query {cql_query}: {e}")
            return []


class ConfluenceContentProcessor:
    """Process and extract information from Confluence content."""
    
    def __init__(self, client=None):
        """
        Initialize the content processor.
        
        Args:
            client (ConfluenceClient, optional): Confluence client instance
        """
        self.client = client or ConfluenceClient()
        self.logger = get_logger("confluence_processor")
    
    @timer_decorator
    def process_page(self, page_id, include_comments=False):
        """
        Process a Confluence page and extract its content.
        
        Args:
            page_id (str): Page ID
            include_comments (bool, optional): Whether to include comments
            
        Returns:
            dict: Processed page data
        """
        # Get page details
        page = self.client.get_page_by_id(
            page_id=page_id,
            expand="body.storage,metadata,version,history,ancestors"
        )
        
        if not page:
            self.logger.error(f"Failed to retrieve page {page_id}")
            return None
            
        # Extract page content
        html_content = page.get("body", {}).get("storage", {}).get("value", "")
        
        # Process the content
        clean_content = clean_html(html_content)
        tables = extract_tables(html_content)
        code_blocks = extract_code_blocks(clean_content)
        
        # Build metadata
        metadata = {
            "id": page.get("id"),
            "title": page.get("title"),
            "type": page.get("type"),
            "space_key": page.get("space", {}).get("key"),
            "space_name": page.get("space", {}).get("name"),
            "url": page.get("_links", {}).get("self"),
            "version": page.get("version", {}).get("number"),
            "created_at": page.get("history", {}).get("createdDate"),
            "updated_at": page.get("version", {}).get("when"),
            "created_by": page.get("history", {}).get("createdBy", {}).get("displayName"),
            "last_updated_by": page.get("version", {}).get("by", {}).get("displayName"),
            "ancestors": [a.get("title") for a in page.get("ancestors", [])]
        }
        
        # Get comments if requested
        comments = []
        if include_comments:
            comment_objects = self.client.get_page_comments(page_id)
            for comment in comment_objects:
                comment_content = comment.get("body", {}).get("storage", {}).get("value", "")
                comments.append({
                    "id": comment.get("id"),
                    "author": comment.get("history", {}).get("createdBy", {}).get("displayName"),
                    "created_at": comment.get("history", {}).get("createdDate"),
                    "content": clean_html(comment_content)
                })
        
        # Build the result
        result = {
            "metadata": metadata,
            "content": clean_content,
            "tables": tables,
            "code_blocks": code_blocks
        }
        
        if include_comments:
            result["comments"] = comments
            
        return result
    
    @timer_decorator
    def process_space(self, space_key, limit=None):
        """
        Process all pages in a space.
        
        Args:
            space_key (str): Space key
            limit (int, optional): Maximum number of pages to process
            
        Returns:
            list: List of processed pages
        """
        # Get all pages in the space
        pages = self.client.get_all_pages(space_key)
        
        if limit:
            pages = pages[:limit]
            
        # Process each page
        processed_pages = []
        for page in pages:
            page_id = page.get("id")
            if page_id:
                processed_page = self.process_page(page_id)
                if processed_page:
                    processed_pages.append(processed_page)
                    
                # Simple rate limiting
                time.sleep(0.5)
        
        return processed_pages
    
    @timer_decorator
    def create_document_for_indexing(self, page):
        """
        Create a document suitable for indexing from a processed page.
        
        Args:
            page (dict): Processed page data
            
        Returns:
            dict: Document ready for indexing
        """
        if not page:
            return None
            
        metadata = page.get("metadata", {})
        content = page.get("content", "")
        
        # Create the document
        document = {
            "id": metadata.get("id"),
            "title": metadata.get("title"),
            "content": content,
            "source": "confluence",
            "source_url": f"{config.CONFLUENCE_URL}/wiki/spaces/{metadata.get('space_key')}/pages/{metadata.get('id')}",
            "metadata": {
                "space_key": metadata.get("space_key"),
                "space_name": metadata.get("space_name"),
                "created_at": metadata.get("created_at"),
                "updated_at": metadata.get("updated_at"),
                "created_by": metadata.get("created_by"),
                "ancestors": metadata.get("ancestors")
            }
        }
        
        return document











"""
Remedy API integration for the RAG application.
Provides functionality to fetch and process content from BMC Remedy.
"""
import time
import json
import os
from urllib.parse import urljoin
import base64

import requests
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from utils import get_logger, timer_decorator, clean_html, generate_cache_key, is_valid_jwt

# Initialize logger
logger = get_logger("remedy")

class RemedyClient:
    """Client for interacting with the BMC Remedy API."""
    
    def __init__(self, base_url=None, username=None, password=None):
        """
        Initialize the Remedy client.
        
        Args:
            base_url (str, optional): Remedy API base URL. Defaults to config.REMEDY_API_BASE.
            username (str, optional): Remedy username. Defaults to config.REMEDY_USERNAME.
            password (str, optional): Remedy password. Defaults to config.REMEDY_PASSWORD.
        """
        self.base_url = base_url or config.REMEDY_API_BASE
        self.username = username or config.REMEDY_USERNAME
        self.password = password or config.REMEDY_PASSWORD
        
        # Ensure base URL ends with a slash
        if not self.base_url.endswith('/'):
            self.base_url += '/'
            
        # Token details
        self._token = None
        self._token_expiry = 0
        
        # Cache directory for tokens
        self.token_cache_dir = os.path.join(config.CACHE_DIR, "remedy_tokens")
        os.makedirs(self.token_cache_dir, exist_ok=True)
    
    def _get_headers(self, include_auth=True):
        """
        Get headers for API requests.
        
        Args:
            include_auth (bool): Whether to include authorization token
            
        Returns:
            dict: Headers to use in API requests
        """
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        if include_auth:
            token = self.get_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
                
        return headers
    
    def get_token(self, force_refresh=False):
        """
        Get or refresh the authentication token.
        
        Args:
            force_refresh (bool): Force token refresh even if current one is valid
            
        Returns:
            str: JWT token
        """
        # Check if we have a valid token in memory
        if not force_refresh and self._token and time.time() < self._token_expiry:
            return self._token
            
        # Check if we have a valid token in cache
        cache_file = os.path.join(self.token_cache_dir, generate_cache_key(self.username))
        if not force_refresh and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    token = cached_data.get('token')
                    expiry = cached_data.get('expiry', 0)
                    
                    if token and expiry > time.time():
                        self._token = token
                        self._token_expiry = expiry
                        return self._token
            except Exception as e:
                logger.error(f"Error reading token cache: {e}")
        
        # Get a new token
        try:
            url = urljoin(self.base_url, "api/jwt/login")
            
            # BMC Remedy typically uses basic auth for token requests
            auth_str = f"{self.username}:{self.password}"
            auth_header = base64.b64encode(auth_str.encode()).decode()
            
            headers = {
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            response = requests.post(
                url=url,
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            
            # Extract token and expiry
            token_data = response.json()
            token = token_data.get('token')
            
            if not token:
                logger.error("No token in Remedy API response")
                return None
                
            # Most Remedy API tokens are valid for 1 hour
            # Calculate expiry time (subtract 5 minutes for safety)
            expiry = time.time() + (55 * 60)
            
            # Save token in memory
            self._token = token
            self._token_expiry = expiry
            
            # Save token to cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump({
                        'token': token,
                        'expiry': expiry
                    }, f)
            except Exception as e:
                logger.error(f"Error saving token to cache: {e}")
                
            return token
            
        except Exception as e:
            logger.error(f"Failed to get Remedy API token: {e}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _make_request(self, method, endpoint, params=None, data=None, retry_auth=True):
        """
        Make an HTTP request to the Remedy API.
        
        Args:
            method (str): HTTP method (GET, POST, etc.)
            endpoint (str): API endpoint (relative to base URL)
            params (dict, optional): Query parameters
            data (dict, optional): Body data for POST requests
            retry_auth (bool): Whether to retry with a new token on auth failure
            
        Returns:
            dict: JSON response
            
        Raises:
            RequestException: If the request fails
        """
        url = urljoin(self.base_url, endpoint)
        headers = self._get_headers()
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=data,
                timeout=30  # 30 second timeout
            )
            
            # Handle authentication failure
            if response.status_code in (401, 403) and retry_auth:
                logger.debug("Authentication failed, refreshing token and retrying")
                self.get_token(force_refresh=True)
                return self._make_request(method, endpoint, params, data, retry_auth=False)
                
            # Raise exception for other 4XX/5XX status codes
            response.raise_for_status()
            
            # Parse JSON response
            return response.json()
            
        except RequestException as e:
            logger.error(f"Remedy API request failed: {e}")
            raise
    
    @timer_decorator
    def get_incident(self, incident_id):
        """
        Get details for a specific incident.
        
        Args:
            incident_id (str): Incident ID
            
        Returns:
            dict: Incident details
        """
        try:
            endpoint = f"api/arsys/v1/entry/HPD:IncidentInterface/{incident_id}"
            
            return self._make_request(
                method="GET",
                endpoint=endpoint
            )
            
        except Exception as e:
            logger.error(f"Failed to get incident {incident_id}: {e}")
            return {}
    
    @timer_decorator
    def search_incidents(self, query=None, limit=100, offset=0, fields=None):
        """
        Search for incidents.
        
        Args:
            query (str, optional): Query string in Remedy query format
            limit (int, optional): Maximum number of results
            offset (int, optional): Starting offset for pagination
            fields (list, optional): List of fields to include
            
        Returns:
            list: List of incidents
        """
        try:
            endpoint = "api/arsys/v1/entry/HPD:IncidentInterface"
            
            params = {
                "limit": limit,
                "offset": offset
            }
            
            if query:
                params["q"] = query
                
            if fields:
                params["fields"] = ",".join(fields)
                
            result = self._make_request(
                method="GET",
                endpoint=endpoint,
                params=params
            )
            
            return result.get("entries", [])
            
        except Exception as e:
            logger.error(f"Failed to search incidents: {e}")
            return []
    
    @timer_decorator
    def get_change_request(self, change_id):
        """
        Get details for a specific change request.
        
        Args:
            change_id (str): Change request ID
            
        Returns:
            dict: Change request details
        """
        try:
            endpoint = f"api/arsys/v1/entry/CHG:ChangeInterface/{change_id}"
            
            return self._make_request(
                method="GET",
                endpoint=endpoint
            )
            
        except Exception as e:
            logger.error(f"Failed to get change request {change_id}: {e}")
            return {}
    
    @timer_decorator
    def search_change_requests(self, query=None, limit=100, offset=0, fields=None):
        """
        Search for change requests.
        
        Args:
            query (str, optional): Query string in Remedy query format
            limit (int, optional): Maximum number of results
            offset (int, optional): Starting offset for pagination
            fields (list, optional): List of fields to include
            
        Returns:
            list: List of change requests
        """
        try:
            endpoint = "api/arsys/v1/entry/CHG:ChangeInterface"
            
            params = {
                "limit": limit,
                "offset": offset
            }
            
            if query:
                params["q"] = query
                
            if fields:
                params["fields"] = ",".join(fields)
                
            result = self._make_request(
                method="GET",
                endpoint=endpoint,
                params=params
            )
            
            return result.get("entries", [])
            
        except Exception as e:
            logger.error(f"Failed to search change requests: {e}")
            return []
    
    @timer_decorator
    def get_knowledge_article(self, article_id):
        """
        Get details for a specific knowledge article.
        
        Args:
            article_id (str): Knowledge article ID
            
        Returns:
            dict: Knowledge article details
        """
        try:
            endpoint = f"api/arsys/v1/entry/KBM:KnowledgeArticle/{article_id}"
            
            return self._make_request(
                method="GET",
                endpoint=endpoint
            )
            
        except Exception as e:
            logger.error(f"Failed to get knowledge article {article_id}: {e}")
            return {}
    
    @timer_decorator
    def search_knowledge_articles(self, query=None, limit=100, offset=0, fields=None):
        """
        Search for knowledge articles.
        
        Args:
            query (str, optional): Query string in Remedy query format
            limit (int, optional): Maximum number of results
            offset (int, optional): Starting offset for pagination
            fields (list, optional): List of fields to include
            
        Returns:
            list: List of knowledge articles
        """
        try:
            endpoint = "api/arsys/v1/entry/KBM:KnowledgeArticle"
            
            params = {
                "limit": limit,
                "offset": offset
            }
            
            if query:
                params["q"] = query
                
            if fields:
                params["fields"] = ",".join(fields)
                
            result = self._make_request(
                method="GET",
                endpoint=endpoint,
                params=params
            )
            
            return result.get("entries", [])
            
        except Exception as e:
            logger.error(f"Failed to search knowledge articles: {e}")
            return []


class RemedyContentProcessor:
    """Process and extract information from Remedy content."""
    
    def __init__(self, client=None):
        """
        Initialize the content processor.
        
        Args:
            client (RemedyClient, optional): Remedy client instance
        """
        self.client = client or RemedyClient()
        self.logger = get_logger("remedy_processor")
    
    def _extract_values(self, data):
        """
        Extract values from Remedy API response.
        
        Args:
            data (dict): Remedy API response data
            
        Returns:
            dict: Extracted values
        """
        if not data or "values" not in data:
            return {}
            
        return data.get("values", {})
    
    @timer_decorator
    def process_incident(self, incident_id):
        """
        Process a Remedy incident and extract its content.
        
        Args:
            incident_id (str): Incident ID
            
        Returns:
            dict: Processed incident data
        """
        # Get incident details
        incident = self.client.get_incident(incident_id)
        
        if not incident:
            self.logger.error(f"Failed to retrieve incident {incident_id}")
            return None
            
        # Extract values
        values = self._extract_values(incident)
        
        if not values:
            self.logger.error(f"No values found in incident {incident_id}")
            return None
            
        # Extract common fields (adjust based on actual Remedy field names)
        incident_number = values.get("Incident Number", "")
        summary = values.get("Summary", "")
        status = values.get("Status", "")
        priority = values.get("Priority", "")
        impact = values.get("Impact", "")
        urgency = values.get("Urgency", "")
        assigned_group = values.get("Assigned Group", "")
        assigned_support = values.get("Assigned Support Company", "")
        assignee = values.get("Assignee", "")
        customer = values.get("Customer", "")
        service = values.get("Service", "")
        create_date = values.get("Create Date", "")
        last_modified = values.get("Last Modified Date", "")
        description = values.get("Description", "")
        notes = values.get("Notes", "")
        resolution = values.get("Resolution", "")
        
        # Clean HTML content if present
        if description and "<" in description and ">" in description:
            description = clean_html(description)
            
        if notes and "<" in notes and ">" in notes:
            notes = clean_html(notes)
            
        if resolution and "<" in resolution and ">" in resolution:
            resolution = clean_html(resolution)
        
        # Build metadata
        metadata = {
            "id": incident_id,
            "incident_number": incident_number,
            "status": status,
            "priority": priority,
            "impact": impact,
            "urgency": urgency,
            "assigned_group": assigned_group,
            "assigned_support": assigned_support,
            "assignee": assignee,
            "customer": customer,
            "service": service,
            "created_at": create_date,
            "updated_at": last_modified
        }
        
        # Build full content text
        content_parts = []
        
        if summary:
            content_parts.append(f"Summary: {summary}")
            
        if description:
            content_parts.append(f"Description: {description}")
            
        if notes:
            content_parts.append(f"Notes: {notes}")
            
        if resolution:
            content_parts.append(f"Resolution: {resolution}")
            
        content = "\n\n".join(content_parts)
        
        # Build the result
        result = {
            "metadata": metadata,
            "content": content,
            "raw_values": values
        }
            
        return result
    
    @timer_decorator
    def process_change_request(self, change_id):
        """
        Process a Remedy change request and extract its content.
        
        Args:
            change_id (str): Change request ID
            
        Returns:
            dict: Processed change request data
        """
        # Get change request details
        change = self.client.get_change_request(change_id)
        
        if not change:
            self.logger.error(f"Failed to retrieve change request {change_id}")
            return None
            
        # Extract values
        values = self._extract_values(change)
        
        if not values:
            self.logger.error(f"No values found in change request {change_id}")
            return None
            
        # Extract common fields (adjust based on actual Remedy field names)
        change_number = values.get("Change Request ID", "")
        summary = values.get("Summary", "")
        status = values.get("Status", "")
        priority = values.get("Priority", "")
        impact = values.get("Impact", "")
        risk = values.get("Risk", "")
        assigned_group = values.get("Assigned Group", "")
        assignee = values.get("Assignee", "")
        change_coordinator = values.get("Change Coordinator", "")
        create_date = values.get("Create Date", "")
        last_modified = values.get("Last Modified Date", "")
        description = values.get("Description", "")
        notes = values.get("Notes", "")
        business_justification = values.get("Business Justification", "")
        implementation_plan = values.get("Implementation Plan", "")
        backout_plan = values.get("Backout Plan", "")
        
        # Clean HTML content if present
        for field in ["description", "notes", "business_justification", "implementation_plan", "backout_plan"]:
            value = locals()[field]
            if value and "<" in value and ">" in value:
                locals()[field] = clean_html(value)
        
        # Build metadata
        metadata = {
            "id": change_id,
            "change_number": change_number,
            "status": status,
            "priority": priority,
            "impact": impact,
            "risk": risk,
            "assigned_group": assigned_group,
            "assignee": assignee,
            "change_coordinator": change_coordinator,
            "created_at": create_date,
            "updated_at": last_modified
        }
        
        # Build full content text
        content_parts = []
        
        if summary:
            content_parts.append(f"Summary: {summary}")
            
        if description:
            content_parts.append(f"Description: {description}")
            
        if business_justification:
            content_parts.append(f"Business Justification: {business_justification}")
            
        if implementation_plan:
            content_parts.append(f"Implementation Plan: {implementation_plan}")
            
        if backout_plan:
            content_parts.append(f"Backout Plan: {backout_plan}")
            
        if notes:
            content_parts.append(f"Notes: {notes}")
            
        content = "\n\n".join(content_parts)
        
        # Build the result
        result = {
            "metadata": metadata,
            "content": content,
            "raw_values": values
        }
            
        return result
    
    @timer_decorator
    def process_knowledge_article(self, article_id):
        """
        Process a Remedy knowledge article and extract its content.
        
        Args:
            article_id (str): Knowledge article ID
            
        Returns:
            dict: Processed knowledge article data
        """
        # Get knowledge article details
        article = self.client.get_knowledge_article(article_id)
        
        if not article:
            self.logger.error(f"Failed to retrieve knowledge article {article_id}")
            return None
            
        # Extract values
        values = self._extract_values(article)
        
        if not values:
            self.logger.error(f"No values found in knowledge article {article_id}")
            return None
            
        # Extract common fields (adjust based on actual Remedy field names)
        article_number = values.get("Article Number", "")
        title = values.get("Title", "")
        status = values.get("Status", "")
        article_type = values.get("Article Type", "")
        created_by = values.get("Created By", "")
        create_date = values.get("Create Date", "")
        last_modified = values.get("Last Modified Date", "")
        description = values.get("Description", "")
        content = values.get("Content", "")
        
        # Clean HTML content if present
        if description and "<" in description and ">" in description:
            description = clean_html(description)
            
        if content and "<" in content and ">" in content:
            content = clean_html(content)
        
        # Build metadata
        metadata = {
            "id": article_id,
            "article_number": article_number,
            "title": title,
            "status": status,
            "article_type": article_type,
            "created_by": created_by,
            "created_at": create_date,
            "updated_at": last_modified
        }
        
        # Build full content text
        content_parts = []
        
        if title:
            content_parts.append(f"Title: {title}")
            
        if description:
            content_parts.append(f"Description: {description}")
            
        if content:
            content_parts.append(f"Content: {content}")
            
        full_content = "\n\n".join(content_parts)
        
        # Build the result
        result = {
            "metadata": metadata,
            "content": full_content,
            "raw_values": values
        }
            
        return result
    
    @timer_decorator
    def create_document_for_indexing(self, item, item_type):
        """
        Create a document suitable for indexing from a processed Remedy item.
        
        Args:
            item (dict): Processed Remedy item data
            item_type (str): Type of item ('incident', 'change', 'knowledge')
            
        Returns:
            dict: Document ready for indexing
        """
        if not item:
            return None
            
        metadata = item.get("metadata", {})
        content = item.get("content", "")
        
        # Build basic document
        document = {
            "id": metadata.get("id"),
            "content": content,
            "source": "remedy",
            "source_type": item_type,
            "metadata": metadata
        }
        
        # Add type-specific fields
        if item_type == "incident":
            document["title"] = f"Incident {metadata.get('incident_number')}"
            document["source_url"] = f"{config.REMEDY_SERVER}/incident/{metadata.get('incident_number')}"
            
        elif item_type == "change":
            document["title"] = f"Change Request {metadata.get('change_number')}"
            document["source_url"] = f"{config.REMEDY_SERVER}/change/{metadata.get('change_number')}"
            
        elif item_type == "knowledge":
            document["title"] = metadata.get("title", f"Knowledge Article {metadata.get('article_number')}")
            document["source_url"] = f"{config.REMEDY_SERVER}/knowledge/{metadata.get('article_number')}"
        
        return document











"""
Data sources package for the RAG application.
"""
from .confluence import ConfluenceClient, ConfluenceContentProcessor
from .remedy import RemedyClient, RemedyContentProcessor

__all__ = [
    'ConfluenceClient',
    'ConfluenceContentProcessor',
    'RemedyClient',
    'RemedyContentProcessor'
]

















"""
Chunking module for the RAG application.
Provides functionality to split documents into chunks for embedding and indexing.
"""
import re
import copy
from typing import List, Dict, Any, Tuple, Optional
import nltk
from nltk.tokenize import sent_tokenize

import config
from utils import get_logger, timer_decorator

# Initialize logger
logger = get_logger("chunking")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class ChunkingStrategy:
    """Base class for document chunking strategies."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the chunking strategy.
        
        Args:
            chunk_size (int, optional): Maximum size of chunks. Defaults to config value.
            chunk_overlap (int, optional): Overlap between chunks. Defaults to config value.
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.logger = get_logger(f"chunking_{self.__class__.__name__}")
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks.
        
        Args:
            document (dict): Document to split
            
        Returns:
            list: List of chunks
        """
        raise NotImplementedError("Subclasses must implement this method")


class SimpleChunker(ChunkingStrategy):
    """Simple chunking strategy that splits by character count."""
    
    @timer_decorator
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks based on character count.
        
        Args:
            document (dict): Document to split
            
        Returns:
            list: List of chunks
        """
        content = document.get("content", "")
        if not content:
            self.logger.warning(f"Document {document.get('id', 'unknown')} has no content")
            return []
            
        # Get document metadata
        doc_id = document.get("id", "")
        title = document.get("title", "")
        metadata = document.get("metadata", {})
        source = document.get("source", "")
        source_url = document.get("source_url", "")
        
        # Split content into chunks
        chunks = []
        start = 0
        
        while start < len(content):
            # Extract chunk
            end = min(start + self.chunk_size, len(content))
            chunk_text = content[start:end]
            
            # Create chunk document
            chunk = {
                "id": f"{doc_id}_{len(chunks)}",
                "doc_id": doc_id,
                "text": chunk_text,
                "metadata": {
                    "title": title,
                    "source": source,
                    "source_url": source_url,
                    "chunk_index": len(chunks),
                    **metadata
                }
            }
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            
            # Avoid tiny chunks at the end
            if len(content) - start < self.chunk_size // 4:
                break
        
        self.logger.debug(f"Document {doc_id} split into {len(chunks)} chunks")
        return chunks


class SentenceChunker(ChunkingStrategy):
    """Chunking strategy that respects sentence boundaries."""
    
    @timer_decorator
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks while respecting sentence boundaries.
        
        Args:
            document (dict): Document to split
            
        Returns:
            list: List of chunks
        """
        content = document.get("content", "")
        if not content:
            self.logger.warning(f"Document {document.get('id', 'unknown')} has no content")
            return []
            
        # Get document metadata
        doc_id = document.get("id", "")
        title = document.get("title", "")
        metadata = document.get("metadata", {})
        source = document.get("source", "")
        source_url = document.get("source_url", "")
        
        # Split text into sentences
        sentences = sent_tokenize(content)
        
        # Create chunks from sentences
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, start a new chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk document
                chunk = {
                    "id": f"{doc_id}_{len(chunks)}",
                    "doc_id": doc_id,
                    "text": current_chunk.strip(),
                    "metadata": {
                        "title": title,
                        "source": source,
                        "source_url": source_url,
                        "chunk_index": len(chunks),
                        **metadata
                    }
                }
                
                chunks.append(chunk)
                
                # Start new chunk with overlap (if any)
                if self.chunk_overlap > 0:
                    # Find last sentences that fit in the overlap
                    overlap_text = ""
                    for s in reversed(current_chunk.split(". ")):
                        if len(s) + len(overlap_text) <= self.chunk_overlap:
                            overlap_text = s + ". " + overlap_text
                        else:
                            break
                    
                    current_chunk = overlap_text
                else:
                    current_chunk = ""
            
            # Add the current sentence to the chunk
            current_chunk += " " + sentence
        
        # Add the final chunk if it's not empty
        if current_chunk.strip():
            chunk = {
                "id": f"{doc_id}_{len(chunks)}",
                "doc_id": doc_id,
                "text": current_chunk.strip(),
                "metadata": {
                    "title": title,
                    "source": source,
                    "source_url": source_url,
                    "chunk_index": len(chunks),
                    **metadata
                }
            }
            
            chunks.append(chunk)
        
        self.logger.debug(f"Document {doc_id} split into {len(chunks)} chunks")
        return chunks


class SemanticChunker(ChunkingStrategy):
    """
    Chunking strategy that tries to respect semantic boundaries.
    Uses headings, paragraphs, and other markers to create logical chunks.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the semantic chunker.
        
        Args:
            chunk_size (int, optional): Maximum size of chunks. Defaults to config value.
            chunk_overlap (int, optional): Overlap between chunks. Defaults to config value.
        """
        super().__init__(chunk_size, chunk_overlap)
        
        # Patterns for identifying section boundaries
        self.section_patterns = [
            r"#+\s+(.+)\n",  # Markdown headings
            r"==+\s*\n(.+)\n==+",  # Markdown heading underlined with =
            r"--+\s*\n(.+)\n--+",  # Markdown heading underlined with -
            r"^(.+)\n=+\s*$",  # Alternate heading style with = underneath
            r"^(.+)\n-+\s*$",  # Alternate heading style with - underneath
            r"^\s*\d+\.\s+(.+)",  # Numbered sections
            r"^\s*[A-Z][A-Z\s]+\s*$"  # ALL CAPS sections
        ]
    
    def _identify_sections(self, text: str) -> List[Tuple[int, str]]:
        """
        Identify section boundaries in text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            list: List of (position, heading) tuples
        """
        sections = []
        
        # Find all matches for each pattern
        for pattern in self.section_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                position = match.start()
                # Try to extract heading from group 1, if not, use whole match
                try:
                    heading = match.group(1)
                except IndexError:
                    heading = match.group(0)
                
                sections.append((position, heading.strip()))
        
        # Add document start
        sections.append((0, "Start"))
        
        # Sort by position
        sections.sort()
        
        return sections
    
    @timer_decorator
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks based on semantic boundaries.
        
        Args:
            document (dict): Document to split
            
        Returns:
            list: List of chunks
        """
        content = document.get("content", "")
        if not content:
            self.logger.warning(f"Document {document.get('id', 'unknown')} has no content")
            return []
            
        # Get document metadata
        doc_id = document.get("id", "")
        title = document.get("title", "")
        metadata = document.get("metadata", {})
        source = document.get("source", "")
        source_url = document.get("source_url", "")
        
        # Identify sections
        sections = self._identify_sections(content)
        
        # If no sections found (or only the start), fall back to paragraph splitting
        if len(sections) <= 1:
            # Split by paragraphs (double newline)
            paragraphs = re.split(r"\n\s*\n", content)
            
            # Use sentence chunker for paragraphs
            sentence_chunker = SentenceChunker(self.chunk_size, self.chunk_overlap)
            
            # Create temporary documents for each paragraph
            temp_docs = []
            for i, para in enumerate(paragraphs):
                if para.strip():  # Skip empty paragraphs
                    temp_doc = {
                        "id": f"{doc_id}_para_{i}",
                        "title": title,
                        "content": para,
                        "metadata": metadata,
                        "source": source,
                        "source_url": source_url
                    }
                    temp_docs.append(temp_doc)
            
            # Chunk each paragraph
            all_chunks = []
            for temp_doc in temp_docs:
                chunks = sentence_chunker.chunk_document(temp_doc)
                all_chunks.extend(chunks)
            
            # Renumber chunks
            for i, chunk in enumerate(all_chunks):
                chunk["id"] = f"{doc_id}_{i}"
                chunk["metadata"]["chunk_index"] = i
            
            self.logger.debug(f"Document {doc_id} split into {len(all_chunks)} chunks by paragraphs")
            return all_chunks
        
        # Process sections to create chunks
        chunks = []
        
        # Add document end position
        sections.append((len(content), "End"))
        
        # Create chunks from sections
        for i in range(len(sections) - 1):
            start_pos, heading = sections[i]
            end_pos, _ = sections[i + 1]
            
            section_text = content[start_pos:end_pos].strip()
            if not section_text:
                continue
            
            # If section is smaller than chunk size, use it as a chunk
            if len(section_text) <= self.chunk_size:
                chunk = {
                    "id": f"{doc_id}_{len(chunks)}",
                    "doc_id": doc_id,
                    "text": section_text,
                    "metadata": {
                        "title": title,
                        "section": heading,
                        "source": source,
                        "source_url": source_url,
                        "chunk_index": len(chunks),
                        **metadata
                    }
                }
                
                chunks.append(chunk)
            else:
                # Section is too large, use sentence chunker
                temp_doc = {
                    "id": f"{doc_id}_section_{i}",
                    "title": title,
                    "content": section_text,
                    "metadata": {**metadata, "section": heading},
                    "source": source,
                    "source_url": source_url
                }
                
                sentence_chunker = SentenceChunker(self.chunk_size, self.chunk_overlap)
                section_chunks = sentence_chunker.chunk_document(temp_doc)
                
                # Add section chunks to main chunks list
                for chunk in section_chunks:
                    chunk["id"] = f"{doc_id}_{len(chunks)}"
                    chunk["metadata"]["chunk_index"] = len(chunks)
                    chunks.append(chunk)
        
        self.logger.debug(f"Document {doc_id} split into {len(chunks)} chunks by sections")
        return chunks


class HierarchicalChunker(ChunkingStrategy):
    """
    Hierarchical chunking strategy that creates both document-level and smaller chunks.
    Preserves relationships between chunks of different granularity.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the hierarchical chunker.
        
        Args:
            chunk_size (int, optional): Maximum size of chunks. Defaults to config value.
            chunk_overlap (int, optional): Overlap between chunks. Defaults to config value.
        """
        super().__init__(chunk_size, chunk_overlap)
        
        # Define different chunk sizes for different levels
        self.document_level_size = self.chunk_size * 3
        self.section_level_size = self.chunk_size * 2
        self.paragraph_level_size = self.chunk_size
        
        # Create lower-level chunkers
        self.semantic_chunker = SemanticChunker(self.section_level_size, self.chunk_overlap)
        self.sentence_chunker = SentenceChunker(self.paragraph_level_size, self.chunk_overlap)
    
    @timer_decorator
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create a hierarchical chunking of the document.
        
        Args:
            document (dict): Document to split
            
        Returns:
            list: List of chunks at different granularity levels
        """
        content = document.get("content", "")
        if not content:
            self.logger.warning(f"Document {document.get('id', 'unknown')} has no content")
            return []
            
        # Get document metadata
        doc_id = document.get("id", "")
        title = document.get("title", "")
        metadata = document.get("metadata", {})
        source = document.get("source", "")
        source_url = document.get("source_url", "")
        
        chunks = []
        
        # Create document-level chunk
        if len(content) <= self.document_level_size:
            # Document fits in a single chunk
            doc_chunk = {
                "id": f"{doc_id}_doc",
                "doc_id": doc_id,
                "text": content,
                "metadata": {
                    "title": title,
                    "source": source,
                    "source_url": source_url,
                    "chunk_level": "document",
                    "chunk_index": 0,
                    **metadata
                }
            }
            
            chunks.append(doc_chunk)
        else:
            # Document-level chunks (sampling beginning, middle, end)
            doc_start = content[:self.document_level_size // 3]
            doc_middle_start = max(0, len(content) // 2 - self.document_level_size // 6)
            doc_middle_end = min(len(content), len(content) // 2 + self.document_level_size // 6)
            doc_middle = content[doc_middle_start:doc_middle_end]
            doc_end = content[max(0, len(content) - self.document_level_size // 3):]
            
            doc_summary = f"{doc_start}... {doc_middle}... {doc_end}"
            
            doc_chunk = {
                "id": f"{doc_id}_doc",
                "doc_id": doc_id,
                "text": doc_summary,
                "metadata": {
                    "title": title,
                    "source": source,
                    "source_url": source_url,
                    "chunk_level": "document",
                    "chunk_index": 0,
                    "is_summary": True,
                    **metadata
                }
            }
            
            chunks.append(doc_chunk)
        
        # Create section-level chunks
        section_doc = copy.deepcopy(document)
        section_chunks = self.semantic_chunker.chunk_document(section_doc)
        
        # Update metadata for section chunks
        for i, chunk in enumerate(section_chunks):
            chunk["id"] = f"{doc_id}_section_{i}"
            chunk["metadata"]["chunk_level"] = "section"
            chunk["metadata"]["parent_chunk"] = f"{doc_id}_doc"
            chunks.append(chunk)
        
        # Create paragraph-level chunks
        paragraph_chunks = []
        
        # Create paragraph chunks for each section chunk
        for i, section in enumerate(section_chunks):
            section_text = section.get("text", "")
            
            # Skip if text is too small
            if len(section_text) <= self.paragraph_level_size // 2:
                continue
                
            # Create temp document for the section
            para_doc = {
                "id": f"{doc_id}_section_{i}",
                "title": title,
                "content": section_text,
                "metadata": metadata,
                "source": source,
                "source_url": source_url
            }
            
            # Create paragraph chunks
            para_chunks = self.sentence_chunker.chunk_document(para_doc)
            paragraph_chunks.extend(para_chunks)
        
        # Update metadata for paragraph chunks
        for i, chunk in enumerate(paragraph_chunks):
            chunk["id"] = f"{doc_id}_para_{i}"
            chunk["metadata"]["chunk_level"] = "paragraph"
            
            # Link to parent section
            section_index = chunk["metadata"].get("chunk_index", 0)
            if section_index < len(section_chunks):
                chunk["metadata"]["parent_chunk"] = f"{doc_id}_section_{section_index}"
            else:
                chunk["metadata"]["parent_chunk"] = f"{doc_id}_doc"
                
            chunks.append(chunk)
        
        self.logger.debug(f"Document {doc_id} split into {len(chunks)} hierarchical chunks")
        return chunks


class ChunkerFactory:
    """Factory for creating chunker instances."""
    
    @staticmethod
    def get_chunker(strategy: str = "semantic", chunk_size: int = None, chunk_overlap: int = None) -> ChunkingStrategy:
        """
        Get a chunker instance based on the specified strategy.
        
        Args:
            strategy (str): Chunking strategy ('simple', 'sentence', 'semantic', 'hierarchical')
            chunk_size (int, optional): Maximum size of chunks
            chunk_overlap (int, optional): Overlap between chunks
            
        Returns:
            ChunkingStrategy: Chunker instance
        """
        if strategy == "simple":
            return SimpleChunker(chunk_size, chunk_overlap)
        elif strategy == "sentence":
            return SentenceChunker(chunk_size, chunk_overlap)
        elif strategy == "hierarchical":
            return HierarchicalChunker(chunk_size, chunk_overlap)
        else:
            # Default to semantic chunking
            return SemanticChunker(chunk_size, chunk_overlap)
















"""
Embedding module for the RAG application.
Provides functionality to generate embeddings for document chunks.
"""
from typing import List, Dict, Any, Union, Optional
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
from utils import get_logger, timer_decorator, generate_cache_key, ensure_directory

# Initialize logger
logger = get_logger("embedding")

class EmbeddingModel:
    """Base class for embedding models."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str, optional): Name of the embedding model. Defaults to config value.
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.logger = get_logger(f"embedding_{self.__class__.__name__}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list: Embedding vector
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts (list): List of texts to embed
            
        Returns:
            list: List of embedding vectors
        """
        raise NotImplementedError("Subclasses must implement this method")


class SentenceTransformerEmbedding(EmbeddingModel):
    """Embedding model using SentenceTransformers."""
    
    def __init__(self, model_name: str = None, cache_dir: str = None):
        """
        Initialize the SentenceTransformer embedding model.
        
        Args:
            model_name (str, optional): Name of the embedding model. Defaults to config value.
            cache_dir (str, optional): Directory to cache embeddings. Defaults to config value.
        """
        super().__init__(model_name)
        
        self.cache_dir = cache_dir or os.path.join(config.CACHE_DIR, "embeddings")
        ensure_directory(self.cache_dir)
        
        # Initialize the model
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Initialized embedding model {self.model_name} with dimension {self.embedding_dim}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model {self.model_name}: {e}")
            raise
    
    def _get_cache_path(self, text: str) -> str:
        """
        Get the cache path for an embedding.
        
        Args:
            text (str): Text to generate cache path for
            
        Returns:
            str: Cache file path
        """
        key = generate_cache_key(text, prefix="emb")
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def _load_from_cache(self, text: str) -> Optional[List[float]]:
        """
        Load embedding from cache.
        
        Args:
            text (str): Text to load embedding for
            
        Returns:
            list or None: Embedding vector if found, None otherwise
        """
        cache_path = self._get_cache_path(text)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return data.get("embedding")
            except Exception as e:
                self.logger.error(f"Error loading embedding from cache: {e}")
        
        return None
    
    def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """
        Save embedding to cache.
        
        Args:
            text (str): Text that was embedded
            embedding (list): Embedding vector
        """
        cache_path = self._get_cache_path(text)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "embedding": embedding
                }, f)
        except Exception as e:
            self.logger.error(f"Error saving embedding to cache: {e}")
    
    @timer_decorator
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text with caching.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list: Embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.embedding_dim
        
        # Check cache first
        cached_embedding = self._load_from_cache(text)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate embedding
        try:
            embedding = self.model.encode(text, show_progress_bar=False).tolist()
            
            # Save to cache
            self._save_to_cache(text, embedding)
            
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            # Return zero vector on error
            return [0.0] * self.embedding_dim
    
    @timer_decorator
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with caching.
        
        Args:
            texts (list): List of texts to embed
            
        Returns:
            list: List of embedding vectors
        """
        if not texts:
            return []
        
        # Check which texts need embedding
        to_embed = []
        to_embed_indices = []
        cached_embeddings = {}
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                cached_embeddings[i] = [0.0] * self.embedding_dim
                continue
                
            cached = self._load_from_cache(text)
            if cached is not None:
                cached_embeddings[i] = cached
            else:
                to_embed.append(text)
                to_embed_indices.append(i)
        
        # Generate embeddings for texts not in cache
        new_embeddings = []
        if to_embed:
            try:
                self.logger.info(f"Generating {len(to_embed)} embeddings")
                new_embeddings = self.model.encode(
                    to_embed,
                    batch_size=32,
                    show_progress_bar=True
                ).tolist()
                
                # Save to cache
                for text, embedding in zip(to_embed, new_embeddings):
                    self._save_to_cache(text, embedding)
            except Exception as e:
                self.logger.error(f"Error generating batch embeddings: {e}")
                # Return zero vectors on error
                new_embeddings = [[0.0] * self.embedding_dim for _ in range(len(to_embed))]
        
        # Combine cached and new embeddings
        all_embeddings = [None] * len(texts)
        
        # Add cached embeddings
        for idx, embedding in cached_embeddings.items():
            all_embeddings[idx] = embedding
        
        # Add new embeddings
        for i, idx in enumerate(to_embed_indices):
            all_embeddings[idx] = new_embeddings[i]
        
        return all_embeddings


class EmbeddingProcessor:
    """Process documents and generate embeddings for chunks."""
    
    def __init__(self, embedding_model: EmbeddingModel = None):
        """
        Initialize the embedding processor.
        
        Args:
            embedding_model (EmbeddingModel, optional): Embedding model to use
        """
        self.embedding_model = embedding_model or SentenceTransformerEmbedding()
        self.logger = get_logger("embedding_processor")
    
    @timer_decorator
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process chunks and add embeddings.
        
        Args:
            chunks (list): List of document chunks
            
        Returns:
            list: List of chunks with embeddings
        """
        if not chunks:
            return []
        
        # Extract texts to embed
        texts = [chunk.get("text", "") for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.generate_embeddings(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        self.logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks
    
    @timer_decorator
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query (str): Query text
            
        Returns:
            list: Query embedding vector
        """
        return self.embedding_model.generate_embedding(query)
    
    @timer_decorator
    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple queries.
        
        Args:
            queries (list): List of query texts
            
        Returns:
            list: List of query embedding vectors
        """
        return self.embedding_model.generate_embeddings(queries)














"""
Indexing module for the RAG application.
Provides functionality to build and manage vector indexes for document retrieval.
"""
import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from utils import get_logger, timer_decorator, ensure_directory

# Initialize logger
logger = get_logger("indexing")

class VectorIndex:
    """Base class for vector indexes."""
    
    def __init__(self, index_dir: str = None, dimension: int = None):
        """
        Initialize the vector index.
        
        Args:
            index_dir (str, optional): Directory to store the index. Defaults to config value.
            dimension (int, optional): Dimension of the vectors. Defaults to config value.
        """
        self.index_dir = index_dir or config.VECTOR_STORE_PATH
        self.dimension = dimension or config.EMBEDDING_DIMENSION
        self.logger = get_logger(f"vector_index_{self.__class__.__name__}")
        
        # Ensure index directory exists
        ensure_directory(self.index_dir)
    
    def add(self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]] = None) -> None:
        """
        Add vectors to the index.
        
        Args:
            ids (list): List of document IDs
            vectors (list): List of embedding vectors
            metadatas (list, optional): List of document metadata
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector (list): Query embedding vector
            k (int, optional): Number of results to return
            
        Returns:
            list: List of search results with IDs, scores, and metadata
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self) -> None:
        """Save the index to disk."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def load(self) -> bool:
        """
        Load the index from disk.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def clear(self) -> None:
        """Clear the index."""
        raise NotImplementedError("Subclasses must implement this method")


class FAISSIndex(VectorIndex):
    """Vector index implementation using FAISS."""
    
    def __init__(self, index_dir: str = None, dimension: int = None, index_type: str = "flat"):
        """
        Initialize the FAISS index.
        
        Args:
            index_dir (str, optional): Directory to store the index. Defaults to config value.
            dimension (int, optional): Dimension of the vectors. Defaults to config value.
            index_type (str, optional): Type of FAISS index ('flat', 'ivf', 'hnsw'). Defaults to 'flat'.
        """
        super().__init__(index_dir, dimension)
        
        self.index_type = index_type
        self.index = None
        self.id_map = {}  # Map FAISS indices to document IDs
        self.metadata_map = {}  # Map document IDs to metadata
        
        # Initialize index
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize the FAISS index based on the specified type."""
        try:
            if self.index_type == "ivf":
                # IVF index - faster search, less accurate
                # Requires vectors for clustering
                nlist = max(1, min(2048, int(1000000 / self.dimension)))
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
                # Note: This index requires training before adding vectors
                self.index_needs_training = True
            elif self.index_type == "hnsw":
                # HNSW index - good balance of speed and accuracy
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 connections per node
                self.index_needs_training = False
            else:
                # Flat index - slower search, most accurate
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index_needs_training = False
                
            self.logger.info(f"Initialized FAISS index of type {self.index_type} with dimension {self.dimension}")
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {e}")
            # Fall back to flat index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index_needs_training = False
            self.index_type = "flat"
    
    def _convert_vectors(self, vectors: List[List[float]]) -> np.ndarray:
        """
        Convert list of vectors to numpy array.
        
        Args:
            vectors (list): List of embedding vectors
            
        Returns:
            numpy.ndarray: Array of vectors
        """
        return np.array(vectors).astype('float32')
    
    def _train_if_needed(self, vectors: np.ndarray) -> None:
        """
        Train the index if required.
        
        Args:
            vectors (numpy.ndarray): Vectors to train on
        """
        if self.index_needs_training and not self.index.is_trained:
            if vectors.shape[0] < 100:
                self.logger.warning("Too few vectors for reliable IVF training, using random vectors")
                # Generate random vectors for training
                random_vectors = np.random.random((max(100, vectors.shape[0] * 2), self.dimension)).astype('float32')
                self.index.train(random_vectors)
            else:
                self.logger.info(f"Training IVF index with {vectors.shape[0]} vectors")
                self.index.train(vectors)
    
    @timer_decorator
    def add(self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]] = None) -> None:
        """
        Add vectors to the index.
        
        Args:
            ids (list): List of document IDs
            vectors (list): List of embedding vectors
            metadatas (list, optional): List of document metadata
        """
        if not ids or not vectors:
            return
            
        if len(ids) != len(vectors):
            self.logger.error(f"Number of IDs ({len(ids)}) doesn't match number of vectors ({len(vectors)})")
            return
            
        # Convert to numpy array
        vectors_np = self._convert_vectors(vectors)
        
        # Train index if needed
        self._train_if_needed(vectors_np)
        
        # Get current index size
        current_size = self.index.ntotal
        
        # Update ID map
        for i, doc_id in enumerate(ids):
            self.id_map[current_size + i] = doc_id
            
        # Update metadata map
        if metadatas:
            for doc_id, metadata in zip(ids, metadatas):
                self.metadata_map[doc_id] = metadata
        
        # Add vectors to the index
        self.index.add(vectors_np)
        
        self.logger.info(f"Added {len(ids)} vectors to the index, total: {self.index.ntotal}")
    
    @timer_decorator
    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector (list): Query embedding vector
            k (int, optional): Number of results to return
            
        Returns:
            list: List of search results with IDs, scores, and metadata
        """
        if not self.index or self.index.ntotal == 0:
            self.logger.warning("Empty index, no results returned")
            return []
            
        try:
            # Convert query vector to numpy array
            query_np = self._convert_vectors([query_vector])
            
            # Search the index
            max_results = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_np, max_results)
            
            # Process results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                # Skip invalid indices
                if idx < 0:
                    continue
                    
                # Get document ID
                doc_id = self.id_map.get(idx)
                if not doc_id:
                    continue
                    
                # Get metadata
                metadata = self.metadata_map.get(doc_id, {})
                
                # Convert distance to similarity score (FAISS uses L2 distance)
                # Smaller distance means more similar, so we invert it
                max_distance = 100.0  # Arbitrary max distance for normalization
                similarity = max(0.0, 1.0 - (distance / max_distance))
                
                # Add to results
                results.append({
                    "id": doc_id,
                    "score": similarity,
                    "metadata": metadata
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching index: {e}")
            return []
    
    @timer_decorator
    def save(self) -> None:
        """Save the index and metadata to disk."""
        if not self.index:
            self.logger.warning("No index to save")
            return
            
        try:
            # Save the FAISS index
            index_path = os.path.join(self.index_dir, f"faiss_{self.index_type}.index")
            faiss.write_index(self.index, index_path)
            
            # Save the ID map
            id_map_path = os.path.join(self.index_dir, "id_map.json")
            with open(id_map_path, 'w') as f:
                # Convert integer keys to strings for JSON
                id_map_str = {str(k): v for k, v in self.id_map.items()}
                json.dump(id_map_str, f)
            
            # Save the metadata map
            metadata_path = os.path.join(self.index_dir, "metadata_map.pickle")
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata_map, f)
                
            self.logger.info(f"Saved index with {self.index.ntotal} vectors to {self.index_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
    
    @timer_decorator
    def load(self) -> bool:
        """
        Load the index and metadata from disk.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Check if index file exists
            index_path = os.path.join(self.index_dir, f"faiss_{self.index_type}.index")
            if not os.path.exists(index_path):
                self.logger.warning(f"Index file not found at {index_path}")
                return False
                
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load ID map
            id_map_path = os.path.join(self.index_dir, "id_map.json")
            if os.path.exists(id_map_path):
                with open(id_map_path, 'r') as f:
                    # Convert string keys back to integers
                    id_map_str = json.load(f)
                    self.id_map = {int(k): v for k, v in id_map_str.items()}
            
            # Load metadata map
            metadata_path = os.path.join(self.index_dir, "metadata_map.pickle")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.metadata_map = pickle.load(f)
            
            self.logger.info(f"Loaded index with {self.index.ntotal} vectors from {self.index_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            # Reinitialize index
            self._initialize_index()
            return False
    
    @timer_decorator
    def clear(self) -> None:
        """Clear the index and metadata."""
        # Reset the index
        self._initialize_index()
        
        # Clear metadata
        self.id_map = {}
        self.metadata_map = {}
        
        self.logger.info("Cleared index")


class IndexManager:
    """Manager for building and using vector indexes."""
    
    def __init__(self, index: VectorIndex = None):
        """
        Initialize the index manager.
        
        Args:
            index (VectorIndex, optional): Vector index to use
        """
        self.index = index or FAISSIndex()
        self.logger = get_logger("index_manager")
        
        # Try to load existing index
        self.index.load()
    
    @timer_decorator
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Index document chunks.
        
        Args:
            chunks (list): List of document chunks with embeddings
        """
        if not chunks:
            self.logger.warning("No chunks to index")
            return
            
        # Check if chunks have embeddings
        if "embedding" not in chunks[0]:
            self.logger.error("Chunks must have embeddings to be indexed")
            return
            
        # Extract data for indexing
        ids = []
        vectors = []
        metadatas = []
        
        for chunk in chunks:
            if "embedding" not in chunk:
                continue
                
            # Get chunk ID
            chunk_id = chunk.get("id")
            if not chunk_id:
                continue
                
            # Get chunk embedding
            embedding = chunk.get("embedding")
            if not embedding:
                continue
                
            # Get chunk metadata (everything except the embedding and text)
            metadata = {k: v for k, v in chunk.items() if k not in ("embedding", "text")}
            
            # Add chunk text to metadata (for retrieval)
            metadata["text"] = chunk.get("text", "")
            
            # Add to lists
            ids.append(chunk_id)
            vectors.append(embedding)
            metadatas.append(metadata)
        
        # Add to index
        if ids and vectors:
            self.index.add(ids, vectors, metadatas)
            
        self.logger.info(f"Indexed {len(ids)} chunks")
    
    @timer_decorator
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding (list): Query embedding vector
            k (int, optional): Number of results to return
            
        Returns:
            list: List of search results
        """
        return self.index.search(query_embedding, k)
    
    def save_index(self) -> None:
        """Save the index to disk."""
        self.index.save()
    
    def clear_index(self) -> None:
        """Clear the index."""
        self.index.clear()













"""
Processing package for the RAG application.
"""
from .chunking import (
    ChunkingStrategy,
    SimpleChunker,
    SentenceChunker,
    SemanticChunker,
    HierarchicalChunker,
    ChunkerFactory
)
from .embedding import (
    EmbeddingModel,
    SentenceTransformerEmbedding,
    EmbeddingProcessor
)
from .indexing import (
    VectorIndex,
    FAISSIndex,
    IndexManager
)

__all__ = [
    'ChunkingStrategy',
    'SimpleChunker',
    'SentenceChunker',
    'SemanticChunker',
    'HierarchicalChunker',
    'ChunkerFactory',
    'EmbeddingModel',
    'SentenceTransformerEmbedding',
    'EmbeddingProcessor',
    'VectorIndex',
    'FAISSIndex',
    'IndexManager'
]














"""
Vector search module for the RAG application.
Provides functionality for semantic search using vector embeddings.
"""
from typing import List, Dict, Any, Optional
import time

import config
from utils import get_logger, timer_decorator
from modules.processing import EmbeddingProcessor, IndexManager

# Initialize logger
logger = get_logger("vector_search")

class VectorSearchRetriever:
    """Retriever that uses vector similarity search."""
    
    def __init__(self, embedding_processor: EmbeddingProcessor = None, index_manager: IndexManager = None):
        """
        Initialize the vector search retriever.
        
        Args:
            embedding_processor (EmbeddingProcessor, optional): Embedding processor to use
            index_manager (IndexManager, optional): Index manager to use
        """
        self.embedding_processor = embedding_processor or EmbeddingProcessor()
        self.index_manager = index_manager or IndexManager()
        self.logger = get_logger("vector_search_retriever")
    
    @timer_decorator
    def search(self, query: str, k: int = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query.
        
        Args:
            query (str): Query text
            k (int, optional): Number of results to return. Defaults to config value.
            min_score (float, optional): Minimum similarity score. Defaults to 0.0.
            
        Returns:
            list: List of search results
        """
        # Use default k if not specified
        if k is None:
            k = config.NUM_RESULTS
        
        # Get query embedding
        start_time = time.time()
        query_embedding = self.embedding_processor.embed_query(query)
        embedding_time = time.time() - start_time
        
        # Search the index
        start_time = time.time()
        results = self.index_manager.search(query_embedding, k=k)
        search_time = time.time() - start_time
        
        # Filter by minimum score
        filtered_results = [result for result in results if result.get("score", 0) >= min_score]
        
        # Log search metrics
        self.logger.debug(f"Vector search metrics - Embedding: {embedding_time:.4f}s, Search: {search_time:.4f}s")
        self.logger.info(f"Vector search found {len(filtered_results)} results for query: {query[:50]}...")
        
        return filtered_results
    
    @timer_decorator
    def multi_query_search(self, query: str, k: int = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search with multiple query variations to improve recall.
        
        Args:
            query (str): Original query text
            k (int, optional): Number of results to return per query variant. Defaults to config value.
            min_score (float, optional): Minimum similarity score. Defaults to 0.0.
            
        Returns:
            list: Merged list of search results
        """
        # Use default k if not specified
        if k is None:
            k = config.NUM_RESULTS
        
        # Generate query variations
        query_variations = self._generate_query_variations(query)
        
        # Get embeddings for all variations
        all_embeddings = self.embedding_processor.embed_queries(query_variations)
        
        # Search with each variation
        all_results = []
        for i, (variation, embedding) in enumerate(zip(query_variations, all_embeddings)):
            # Search the index
            results = self.index_manager.search(embedding, k=k)
            
            # Add query variant info to results
            for result in results:
                result["query_variant"] = i
                result["query_text"] = variation
            
            all_results.extend(results)
        
        # Merge results (remove duplicates, keep highest score)
        merged_results = {}
        
        for result in all_results:
            result_id = result.get("id")
            score = result.get("score", 0)
            
            if result_id not in merged_results or score > merged_results[result_id].get("score", 0):
                merged_results[result_id] = result
        
        # Convert to list and sort by score
        results_list = list(merged_results.values())
        results_list.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Apply minimum score filter
        filtered_results = [result for result in results_list if result.get("score", 0) >= min_score]
        
        # Truncate to k results
        final_results = filtered_results[:k]
        
        self.logger.info(f"Multi-query search found {len(final_results)} results for query: {query[:50]}...")
        return final_results
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """
        Generate variations of a query to improve retrieval.
        
        Args:
            query (str): Original query
            
        Returns:
            list: List of query variations
        """
        variations = [query]  # Original query
        
        # Add a "what is" variant if query is short and doesn't have it
        if len(query) < 100 and not query.lower().startswith("what is"):
            variations.append(f"What is {query}?")
        
        # Add a more detailed variant
        if len(query) < 100:
            variations.append(f"Explain in detail about {query}")
        
        # Add a simpler variant (first 8 words or so)
        words = query.split()
        if len(words) > 8:
            variations.append(" ".join(words[:8]))
        
        # If query is a question, add an instruction variant
        if "?" in query:
            variations.append(query.replace("?", ""))
        
        # Remove duplicates
        unique_variations = list(dict.fromkeys(variations))
        
        self.logger.debug(f"Generated {len(unique_variations)} query variations")
        return unique_variations
    
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Index document chunks.
        
        Args:
            chunks (list): List of document chunks
        """
        # Generate embeddings for chunks
        chunks_with_embeddings = self.embedding_processor.process_chunks(chunks)
        
        # Index the chunks
        self.index_manager.index_chunks(chunks_with_embeddings)
        
        # Save the index
        self.index_manager.save_index()
















"""
Lexical search module for the RAG application.
Provides functionality for keyword-based search using BM25 algorithm.
"""
from typing import List, Dict, Any, Set, Optional
import os
import json
import pickle
import time
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

import config
from utils import get_logger, timer_decorator, ensure_directory

# Initialize logger
logger = get_logger("lexical_search")

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class LexicalSearchRetriever:
    """Retriever that uses BM25 for keyword-based search."""
    
    def __init__(self, index_dir: str = None):
        """
        Initialize the lexical search retriever.
        
        Args:
            index_dir (str, optional): Directory to store the index. Defaults to config value.
        """
        self.index_dir = index_dir or os.path.join(config.CACHE_DIR, "lexical_index")
        ensure_directory(self.index_dir)
        
        self.logger = get_logger("lexical_search_retriever")
        
        # Initialize tokenizer components
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize index
        self.bm25_index = None
        self.doc_tokens = []
        self.doc_mapping = {}  # Maps internal indices to document IDs
        self.doc_metadata = {}  # Maps document IDs to metadata
        
        # Load existing index if available
        self.load_index()
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for indexing or searching.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            list: List of preprocessed tokens
        """
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers (keep letters and spaces)
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and stem
        preprocessed_tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]
        
        return preprocessed_tokens
    
    @timer_decorator
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Index document chunks.
        
        Args:
            chunks (list): List of document chunks
        """
        if not chunks:
            self.logger.warning("No chunks to index")
            return
        
        # Clear existing index if any
        self.clear_index()
        
        # Process each chunk
        doc_tokens = []
        doc_mapping = {}
        doc_metadata = {}
        
        for i, chunk in enumerate(chunks):
            # Get chunk text
            text = chunk.get("text", "")
            if not text:
                continue
            
            # Get chunk ID
            chunk_id = chunk.get("id")
            if not chunk_id:
                continue
            
            # Preprocess text
            tokens = self._preprocess_text(text)
            if not tokens:
                continue
            
            # Store tokens
            doc_tokens.append(tokens)
            
            # Map internal index to chunk ID
            doc_mapping[i] = chunk_id
            
            # Store metadata
            metadata = {k: v for k, v in chunk.items() if k != "text"}
            metadata["text"] = text  # Include original text for retrieval
            doc_metadata[chunk_id] = metadata
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(doc_tokens)
        self.doc_tokens = doc_tokens
        self.doc_mapping = doc_mapping
        self.doc_metadata = doc_metadata
        
        # Save the index
        self.save_index()
        
        self.logger.info(f"Indexed {len(doc_tokens)} chunks for lexical search")
    
    @timer_decorator
    def search(self, query: str, k: int = None, min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for chunks matching the query.
        
        Args:
            query (str): Query text
            k (int, optional): Number of results to return. Defaults to config value.
            min_score (float, optional): Minimum BM25 score. Defaults to 0.1.
            
        Returns:
            list: List of search results
        """
        # Use default k if not specified
        if k is None:
            k = config.NUM_RESULTS
        
        # Check if index exists
        if not self.bm25_index or not self.doc_tokens:
            self.logger.warning("Lexical index is empty, no results returned")
            return []
        
        # Preprocess query
        start_time = time.time()
        query_tokens = self._preprocess_text(query)
        preprocessing_time = time.time() - start_time
        
        # Handle empty query tokens
        if not query_tokens:
            self.logger.warning("Query contains no meaningful terms after preprocessing")
            return []
        
        # Search using BM25
        start_time = time.time()
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        search_time = time.time() - start_time
        
        # Get top K results with scores above threshold
        results = []
        for i, score in enumerate(bm25_scores):
            if score >= min_score:
                # Get document ID from internal index
                doc_id = self.doc_mapping.get(i)
                if not doc_id:
                    continue
                
                # Get metadata
                metadata = self.doc_metadata.get(doc_id, {})
                
                # Normalize score (BM25 scores are not bounded)
                # This is a simple normalization; adjust as needed
                normalized_score = min(score / 10.0, 1.0)
                
                # Add to results
                results.append({
                    "id": doc_id,
                    "score": normalized_score,
                    "bm25_score": score,
                    "metadata": metadata
                })
        
        # Sort by score and limit to k results
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = results[:k]
        
        # Log search metrics
        self.logger.debug(f"Lexical search metrics - Preprocessing: {preprocessing_time:.4f}s, Search: {search_time:.4f}s")
        self.logger.info(f"Lexical search found {len(results)} results for query: {query[:50]}...")
        
        return results
    
    @timer_decorator
    def save_index(self) -> None:
        """Save the index to disk."""
        try:
            # Save BM25 index
            index_path = os.path.join(self.index_dir, "bm25_index.pickle")
            with open(index_path, 'wb') as f:
                pickle.dump(self.bm25_index, f)
            
            # Save document tokens
            tokens_path = os.path.join(self.index_dir, "doc_tokens.pickle")
            with open(tokens_path, 'wb') as f:
                pickle.dump(self.doc_tokens, f)
            
            # Save document mapping
            mapping_path = os.path.join(self.index_dir, "doc_mapping.json")
            with open(mapping_path, 'w') as f:
                # Convert integer keys to strings for JSON
                mapping_str = {str(k): v for k, v in self.doc_mapping.items()}
                json.dump(mapping_str, f)
            
            # Save document metadata
            metadata_path = os.path.join(self.index_dir, "doc_metadata.pickle")
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.doc_metadata, f)
                
            self.logger.info(f"Saved lexical index with {len(self.doc_tokens)} documents")
            
        except Exception as e:
            self.logger.error(f"Error saving lexical index: {e}")
    
    @timer_decorator
    def load_index(self) -> bool:
        """
        Load the index from disk.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Check if index files exist
            index_path = os.path.join(self.index_dir, "bm25_index.pickle")
            tokens_path = os.path.join(self.index_dir, "doc_tokens.pickle")
            mapping_path = os.path.join(self.index_dir, "doc_mapping.json")
            metadata_path = os.path.join(self.index_dir, "doc_metadata.pickle")
            
            if not all(os.path.exists(p) for p in [index_path, tokens_path, mapping_path, metadata_path]):
                self.logger.warning("One or more lexical index files not found")
                return False
                
            # Load BM25 index
            with open(index_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
            
            # Load document tokens
            with open(tokens_path, 'rb') as f:
                self.doc_tokens = pickle.load(f)
            
            # Load document mapping
            with open(mapping_path, 'r') as f:
                # Convert string keys back to integers
                mapping_str = json.load(f)
                self.doc_mapping = {int(k): v for k, v in mapping_str.items()}
            
            # Load document metadata
            with open(metadata_path, 'rb') as f:
                self.doc_metadata = pickle.load(f)
            
            self.logger.info(f"Loaded lexical index with {len(self.doc_tokens)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading lexical index: {e}")
            self.clear_index()
            return False
    
    def clear_index(self) -> None:
        """Clear the in-memory index."""
        self.bm25_index = None
        self.doc_tokens = []
        self.doc_mapping = {}
        self.doc_metadata = {}










"""
Hybrid search module for the RAG application.
Combines vector and lexical search for improved retrieval.
"""
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import time

import config
from utils import get_logger, timer_decorator
from modules.retrieval.vector_search import VectorSearchRetriever
from modules.retrieval.lexical_search import LexicalSearchRetriever

# Initialize logger
logger = get_logger("hybrid_search")

class HybridSearchRetriever:
    """Retriever that combines vector and lexical search."""
    
    def __init__(
        self,
        vector_retriever: VectorSearchRetriever = None,
        lexical_retriever: LexicalSearchRetriever = None,
        vector_weight: float = 0.7,
        lexical_weight: float = 0.3
    ):
        """
        Initialize the hybrid search retriever.
        
        Args:
            vector_retriever (VectorSearchRetriever, optional): Vector search retriever
            lexical_retriever (LexicalSearchRetriever, optional): Lexical search retriever
            vector_weight (float, optional): Weight for vector search scores. Defaults to 0.7.
            lexical_weight (float, optional): Weight for lexical search scores. Defaults to 0.3.
        """
        self.vector_retriever = vector_retriever or VectorSearchRetriever()
        self.lexical_retriever = lexical_retriever or LexicalSearchRetriever()
        
        # Ensure weights sum to 1.0
        total_weight = vector_weight + lexical_weight
        self.vector_weight = vector_weight / total_weight
        self.lexical_weight = lexical_weight / total_weight
        
        self.logger = get_logger("hybrid_search_retriever")
    
    @timer_decorator
    def search(self, query: str, k: int = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for chunks using both vector and lexical search.
        
        Args:
            query (str): Query text
            k (int, optional): Number of results to return. Defaults to config value.
            min_score (float, optional): Minimum combined score. Defaults to 0.0.
            
        Returns:
            list: List of search results
        """
        # Use default k if not specified
        if k is None:
            k = config.NUM_RESULTS
        
        # We'll retrieve more results from each search to get better candidates
        retrieval_k = max(k * 2, 10)
        
        # Start vector search
        start_time = time.time()
        vector_results = self.vector_retriever.search(query, k=retrieval_k)
        vector_time = time.time() - start_time
        
        # Start lexical search
        start_time = time.time()
        lexical_results = self.lexical_retriever.search(query, k=retrieval_k)
        lexical_time = time.time() - start_time
        
        # Merge results with weighted scores
        start_time = time.time()
        merged_results = self._merge_results(vector_results, lexical_results)
        merge_time = time.time() - start_time
        
        # Filter by minimum score
        filtered_results = [result for result in merged_results if result.get("score", 0) >= min_score]
        
        # Sort by score and limit to k results
        filtered_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        final_results = filtered_results[:k]
        
        # Log search metrics
        self.logger.debug(
            f"Hybrid search metrics - Vector: {vector_time:.4f}s, "
            f"Lexical: {lexical_time:.4f}s, Merge: {merge_time:.4f}s"
        )
        self.logger.info(
            f"Hybrid search found {len(final_results)} results "
            f"(from {len(vector_results)} vector, {len(lexical_results)} lexical) "
            f"for query: {query[:50]}..."
        )
        
        return final_results
    
    @timer_decorator
    def multi_query_search(self, query: str, k: int = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search with multiple query variations using both vector and lexical search.
        
        Args:
            query (str): Original query text
            k (int, optional): Number of results to return. Defaults to config value.
            min_score (float, optional): Minimum combined score. Defaults to 0.0.
            
        Returns:
            list: Merged list of search results
        """
        # Use default k if not specified
        if k is None:
            k = config.NUM_RESULTS
        
        # We'll retrieve more results to get better candidates
        retrieval_k = max(k * 2, 10)
        
        # Vector search with multiple queries
        start_time = time.time()
        vector_results = self.vector_retriever.multi_query_search(query, k=retrieval_k)
        vector_time = time.time() - start_time
        
        # Regular lexical search (already handles variations to some extent)
        start_time = time.time()
        lexical_results = self.lexical_retriever.search(query, k=retrieval_k)
        lexical_time = time.time() - start_time
        
        # Merge results with weighted scores
        start_time = time.time()
        merged_results = self._merge_results(vector_results, lexical_results)
        merge_time = time.time() - start_time
        
        # Filter by minimum score
        filtered_results = [result for result in merged_results if result.get("score", 0) >= min_score]
        
        # Sort by score and limit to k results
        filtered_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        final_results = filtered_results[:k]
        
        # Log search metrics
        self.logger.debug(
            f"Hybrid multi-query search metrics - Vector: {vector_time:.4f}s, "
            f"Lexical: {lexical_time:.4f}s, Merge: {merge_time:.4f}s"
        )
        self.logger.info(
            f"Hybrid multi-query search found {len(final_results)} results "
            f"for query: {query[:50]}..."
        )
        
        return final_results
    
    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        lexical_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge results from vector and lexical search with weighted scoring.
        
        Args:
            vector_results (list): Results from vector search
            lexical_results (list): Results from lexical search
            
        Returns:
            list: Merged results with combined scores
        """
        # Create a map of document IDs to scores
        result_map = {}
        
        # Process vector results
        for result in vector_results:
            doc_id = result.get("id")
            if not doc_id:
                continue
                
            score = result.get("score", 0)
            metadata = result.get("metadata", {})
            
            result_map[doc_id] = {
                "id": doc_id,
                "vector_score": score,
                "lexical_score": 0,
                "metadata": metadata,
                "sources": ["vector"]
            }
        
        # Process lexical results
        for result in lexical_results:
            doc_id = result.get("id")
            if not doc_id:
                continue
                
            score = result.get("score", 0)
            metadata = result.get("metadata", {})
            
            if doc_id in result_map:
                # Update existing entry
                result_map[doc_id]["lexical_score"] = score
                result_map[doc_id]["sources"].append("lexical")
                
                # Use metadata from lexical result if vector doesn't have it
                if not result_map[doc_id]["metadata"] and metadata:
                    result_map[doc_id]["metadata"] = metadata
            else:
                # Create new entry
                result_map[doc_id] = {
                    "id": doc_id,
                    "vector_score": 0,
                    "lexical_score": score,
                    "metadata": metadata,
                    "sources": ["lexical"]
                }
        
        # Calculate combined scores
        results = []
        for doc_id, data in result_map.items():
            vector_score = data.get("vector_score", 0)
            lexical_score = data.get("lexical_score", 0)
            
            # Combine scores
            combined_score = (vector_score * self.vector_weight) + (lexical_score * self.lexical_weight)
            
            # Create final result item
            result = {
                "id": doc_id,
                "score": combined_score,
                "vector_score": vector_score,
                "lexical_score": lexical_score,
                "metadata": data.get("metadata", {}),
                "sources": data.get("sources", [])
            }
            
            results.append(result)
        
        return results
    
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Index document chunks for both vector and lexical search.
        
        Args:
            chunks (list): List of document chunks
        """
        # First, index for vector search
        self.vector_retriever.index_chunks(chunks)
        
        # Then, index for lexical search
        self.lexical_retriever.index_chunks(chunks)
        
        self.logger.info(f"Indexed {len(chunks)} chunks for hybrid search")


class ReRanker:
    """Re-rank search results for improved relevance."""
    
    def __init__(self, vector_retriever: VectorSearchRetriever = None):
        """
        Initialize the re-ranker.
        
        Args:
            vector_retriever (VectorSearchRetriever, optional): Vector search retriever for embeddings
        """
        self.vector_retriever = vector_retriever or VectorSearchRetriever()
        self.logger = get_logger("reranker")
    
    @timer_decorator
    def rerank(self, query: str, results: List[Dict[str, Any]], k: int = None) -> List[Dict[str, Any]]:
        """
        Re-rank search results for improved relevance.
        
        Args:
            query (str): Query text
            results (list): Initial search results
            k (int, optional): Number of results to return. Defaults to length of results.
            
        Returns:
            list: Re-ranked results
        """
        if not results:
            return []
            
        if k is None:
            k = len(results)
        
        # Extract texts from results
        texts = []
        for result in results:
            # Get text from metadata
            metadata = result.get("metadata", {})
            text = metadata.get("text", "")
            
            # If no text in metadata, try to get it from other fields
            if not text and "text" in result:
                text = result.get("text", "")
            
            texts.append(text)
        
        # Generate query and document embeddings
        query_embedding = self.vector_retriever.embedding_processor.embed_query(query)
        doc_embeddings = self.vector_retriever.embedding_processor.embed_queries(texts)
        
        # Calculate more precise similarity scores
        new_scores = []
        for i, doc_embedding in enumerate(doc_embeddings):
            # Compute cosine similarity (dot product of normalized vectors)
            similarity = self._compute_similarity(query_embedding, doc_embedding)
            new_scores.append(similarity)
        
        # Add new scores to results
        for i, (result, new_score) in enumerate(zip(results, new_scores)):
            result["rerank_score"] = new_score
            
            # Optionally adjust the original score
            # This gives more weight to the re-ranking score
            result["original_score"] = result.get("score", 0)
            result["score"] = (result.get("score", 0) * 0.4) + (new_score * 0.6)
        
        # Sort by new scores
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Return top k results
        return results[:k]
    
    def _compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1 (list): First vector
            vec2 (list): Second vector
            
        Returns:
            float: Cosine similarity (between -1 and 1)
        """
        import numpy as np
        
        # Convert to numpy arrays
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # Compute dot product
        dot_product = np.dot(v1, v2)
        
        # Compute magnitudes
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        
        # Avoid division by zero
        if mag1 * mag2 == 0:
            return 0
        
        # Compute cosine similarity
        cos_sim = dot_product / (mag1 * mag2)
        
        # Ensure result is in valid range
        return max(-1.0, min(1.0, cos_sim))










"""
Retrieval package for the RAG application.
"""
from .vector_search import VectorSearchRetriever
from .lexical_search import LexicalSearchRetriever
from .hybrid_search import HybridSearchRetriever, ReRanker

__all__ = [
    'VectorSearchRetriever',
    'LexicalSearchRetriever',
    'HybridSearchRetriever',
    'ReRanker'
]

















"""
Google Gemini integration for the RAG application.
"""
from typing import List, Dict, Any, Optional, Union, Generator
import os
import time

from google.api_core.exceptions import GoogleAPIError
import vertexai
from vertexai.generative_models import GenerativeModel, Content, Part
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config
from utils import get_logger, timer_decorator, format_chunk_for_prompt

# Initialize logger
logger = get_logger("gemini")

class GeminiClient:
    """Client for interacting with the Google Gemini API."""
    
    def __init__(self, model_name: str = None, temperature: float = None):
        """
        Initialize the Gemini client.
        
        Args:
            model_name (str, optional): Gemini model name. Defaults to config value.
            temperature (float, optional): Temperature for generation. Defaults to config value.
        """
        self.model_name = model_name or config.MODEL_NAME
        self.temperature = temperature or config.TEMPERATURE
        self.logger = get_logger("gemini_client")
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=config.PROJECT_ID, location=config.REGION)
            self.logger.info(f"Initialized Vertex AI for project {config.PROJECT_ID} in {config.REGION}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
        
        # Initialize model
        try:
            self.model = GenerativeModel(self.model_name)
            self.logger.info(f"Initialized Gemini model {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model {self.model_name}: {e}")
            raise
    
    @retry(
        retry=retry_if_exception_type(GoogleAPIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    @timer_decorator
    def generate_text(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Union[str, Generator]:
        """
        Generate text response from Gemini.
        
        Args:
            prompt (str): Prompt text
            temperature (float, optional): Temperature for generation. Defaults to instance value.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 1024.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            
        Returns:
            str or Generator: Generated text or response stream
        """
        if temperature is None:
            temperature = self.temperature
        
        try:
            # Create content
            content = [Content.from_str(prompt)]
            
            # Generate response
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.9,
                "top_k": 40
            }
            
            if stream:
                response = self.model.generate_content(
                    content,
                    generation_config=generation_config,
                    stream=True
                )
                
                # Return the streaming response for the caller to process
                return response
            else:
                response = self.model.generate_content(
                    content,
                    generation_config=generation_config
                )
                
                # Extract text from response
                return response.text
                
        except GoogleAPIError as e:
            self.logger.error(f"Gemini API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return f"Error generating response: {str(e)}"
    
    @retry(
        retry=retry_if_exception_type(GoogleAPIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    @timer_decorator
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Union[str, Generator]:
        """
        Generate response in a chat format from Gemini.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            temperature (float, optional): Temperature for generation. Defaults to instance value.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 1024.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            
        Returns:
            str or Generator: Generated text or response stream
        """
        if temperature is None:
            temperature = self.temperature
        
        try:
            # Convert messages to Gemini format
            content_parts = []
            
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "user":
                    content_parts.append(Content(role="user", parts=[Part.from_text(content)]))
                elif role in ["assistant", "model"]:
                    content_parts.append(Content(role="model", parts=[Part.from_text(content)]))
                elif role == "system":
                    # For system messages, we'll add them as user messages with a special prefix
                    content_parts.append(Content(role="user", parts=[Part.from_text(f"[SYSTEM] {content}")]))
            
            # Generate response
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.9,
                "top_k": 40
            }
            
            if stream:
                response = self.model.generate_content(
                    content_parts,
                    generation_config=generation_config,
                    stream=True
                )
                
                # Return the streaming response for the caller to process
                return response
            else:
                response = self.model.generate_content(
                    content_parts,
                    generation_config=generation_config
                )
                
                # Extract text from response
                return response.text
                
        except GoogleAPIError as e:
            self.logger.error(f"Gemini API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error generating chat response: {e}")
            return f"Error generating response: {str(e)}"


class RAGGenerator:
    """Generate responses using the RAG approach with Gemini."""
    
    def __init__(self, gemini_client: GeminiClient = None):
        """
        Initialize the RAG generator.
        
        Args:
            gemini_client (GeminiClient, optional): Gemini client to use
        """
        self.gemini_client = gemini_client or GeminiClient()
        self.logger = get_logger("rag_generator")
    
    @timer_decorator
    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Union[str, Generator]:
        """
        Generate a response using RAG.
        
        Args:
            query (str): User query
            chunks (list): Retrieved document chunks
            system_prompt (str, optional): System prompt for the LLM
            temperature (float, optional): Temperature for generation
            max_tokens (int, optional): Maximum tokens to generate
            stream (bool, optional): Whether to stream the response
            
        Returns:
            str or Generator: Generated text or response stream
        """
        # Format chunks
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            text = metadata.get("text", "")
            title = metadata.get("title", "")
            source = metadata.get("source", "Unknown")
            url = metadata.get("source_url", "")
            
            # Format the chunk for inclusion in the prompt
            formatted_chunk = f"[Document {i+1}] {title}\n"
            formatted_chunk += f"Source: {source}\n"
            if url:
                formatted_chunk += f"URL: {url}\n"
            formatted_chunk += f"Content: {text}\n"
            
            formatted_chunks.append(formatted_chunk)
        
        # Default system prompt if not provided
        if not system_prompt:
            system_prompt = """
            You are an intelligent assistant that provides helpful, accurate, 
            and thoughtful responses to queries. Base your answers on the 
            provided context documents. If the documents don't provide enough 
            information to answer completely, acknowledge the limitations of your 
            knowledge. Always cite the sources of information in your response.
            """.strip()
        
        # Build the full prompt
        prompt = f"""
        {system_prompt}
        
        Here are the reference documents:
        
        {"\n\n".join(formatted_chunks)}
        
        User Query: {query}
        
        Please provide a comprehensive answer based on the provided documents.
        If the documents don't provide enough information, say so clearly.
        Include citations to the relevant document numbers in your response.
        """.strip()
        
        # Generate response
        start_time = time.time()
        response = self.gemini_client.generate_text(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if not stream:
            generation_time = time.time() - start_time
            response_length = len(response) if isinstance(response, str) else "streaming"
            self.logger.info(f"Generated RAG response ({response_length} chars) in {generation_time:.2f} seconds")
        
        return response
    
    @timer_decorator
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        chunks: List[Dict[str, Any]],
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Union[str, Generator]:
        """
        Generate a chat response using RAG.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            chunks (list): Retrieved document chunks
            system_prompt (str, optional): System prompt for the LLM
            temperature (float, optional): Temperature for generation
            max_tokens (int, optional): Maximum tokens to generate
            stream (bool, optional): Whether to stream the response
            
        Returns:
            str or Generator: Generated text or response stream
        """
        # Format chunks
        context_str = ""
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            text = metadata.get("text", "")
            title = metadata.get("title", "")
            source = metadata.get("source", "Unknown")
            url = metadata.get("source_url", "")
            
            # Format the chunk for inclusion in the prompt
            chunk_str = f"[Document {i+1}] {title}\n"
            chunk_str += f"Source: {source}\n"
            if url:
                chunk_str += f"URL: {url}\n"
            chunk_str += f"Content: {text}\n\n"
            
            context_str += chunk_str
        
        # Default system prompt if not provided
        if not system_prompt:
            system_prompt = """
            You are an intelligent assistant that provides helpful, accurate, 
            and thoughtful responses to queries. Base your answers on the 
            provided context documents. If the documents don't provide enough 
            information to answer completely, acknowledge the limitations of your 
            knowledge. Always cite the sources of information in your response.
            """.strip()
        
        # Create a new message list with context and system prompt
        new_messages = []
        
        # Add system prompt and context as the first message
        system_with_context = f"""
        {system_prompt}
        
        Here are the reference documents:
        
        {context_str}
        
        Please provide answers based on the provided documents.
        If the documents don't provide enough information, say so clearly.
        Include citations to the relevant document numbers in your response.
        """.strip()
        
        new_messages.append({"role": "system", "content": system_with_context})
        
        # Add user messages
        for message in messages:
            new_messages.append(message)
        
        # Generate response
        start_time = time.time()
        response = self.gemini_client.chat(
            messages=new_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if not stream:
            generation_time = time.time() - start_time
            response_length = len(response) if isinstance(response, str) else "streaming"
            self.logger.info(f"Generated RAG chat response ({response_length} chars) in {generation_time:.2f} seconds")
        
        return response












"""
Prompt templates for the RAG application.
"""
from typing import List, Dict, Any
from string import Template

class PromptTemplate:
    """Base class for prompt templates."""
    
    def __init__(self, template: str):
        """
        Initialize the prompt template.
        
        Args:
            template (str): Template string with placeholders
        """
        self.template = Template(template)
    
    def format(self, **kwargs) -> str:
        """
        Format the template with provided values.
        
        Args:
            **kwargs: Values for template placeholders
            
        Returns:
            str: Formatted prompt
        """
        return self.template.safe_substitute(**kwargs)


class RAGPromptTemplate(PromptTemplate):
    """Prompt template for RAG responses."""
    
    def __init__(self):
        """Initialize the RAG prompt template."""
        template = """
You are an intelligent assistant that provides helpful, accurate, 
and thoughtful responses to queries. Base your answers on the 
provided context documents. If the documents don't provide enough 
information to answer completely, acknowledge the limitations of your 
knowledge. Always cite the sources of information in your response.

Here are the reference documents:

$context

User Query: $query

Please provide a comprehensive answer based on the provided documents.
If the documents don't provide enough information, say so clearly.
Include citations to the relevant document numbers in your response.
        """.strip()
        
        super().__init__(template)
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format document chunks for inclusion in the prompt.
        
        Args:
            chunks (list): List of document chunks
            
        Returns:
            str: Formatted context string
        """
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            text = metadata.get("text", "")
            title = metadata.get("title", "")
            source = metadata.get("source", "Unknown")
            url = metadata.get("source_url", "")
            
            # Format the chunk
            formatted_chunk = f"[Document {i+1}] {title}\n"
            formatted_chunk += f"Source: {source}\n"
            if url:
                formatted_chunk += f"URL: {url}\n"
            formatted_chunk += f"Content: {text}\n"
            
            formatted_chunks.append(formatted_chunk)
        
        return "\n\n".join(formatted_chunks)
    
    def format_with_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Format the template with query and chunks.
        
        Args:
            query (str): User query
            chunks (list): List of document chunks
            
        Returns:
            str: Formatted prompt
        """
        context = self.format_context(chunks)
        return self.format(query=query, context=context)


class RAGChatPromptTemplate(PromptTemplate):
    """Prompt template for RAG chat responses."""
    
    def __init__(self):
        """Initialize the RAG chat prompt template."""
        template = """
You are an intelligent assistant that provides helpful, accurate, 
and thoughtful responses to queries. Base your answers on the 
provided context documents. If the documents don't provide enough 
information to answer completely, acknowledge the limitations of your 
knowledge. Always cite the sources of information in your response.

Here are the reference documents:

$context

Please provide answers based on the provided documents.
If the documents don't provide enough information, say so clearly.
Include citations to the relevant document numbers in your response.
        """.strip()
        
        super().__init__(template)
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format document chunks for inclusion in the prompt.
        
        Args:
            chunks (list): List of document chunks
            
        Returns:
            str: Formatted context string
        """
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            text = metadata.get("text", "")
            title = metadata.get("title", "")
            source = metadata.get("source", "Unknown")
            url = metadata.get("source_url", "")
            
            # Format the chunk
            formatted_chunk = f"[Document {i+1}] {title}\n"
            formatted_chunk += f"Source: {source}\n"
            if url:
                formatted_chunk += f"URL: {url}\n"
            formatted_chunk += f"Content: {text}\n"
            
            formatted_chunks.append(formatted_chunk)
        
        return "\n\n".join(formatted_chunks)
    
    def format_with_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format the template with chunks.
        
        Args:
            chunks (list): List of document chunks
            
        Returns:
            str: Formatted system prompt with context
        """
        context = self.format_context(chunks)
        return self.format(context=context)


# Predefined system prompts
SYSTEM_PROMPTS = {
    "general": """
You are an intelligent assistant that provides helpful, accurate, 
and thoughtful responses to queries. Base your answers on the 
provided context documents. If the documents don't provide enough 
information to answer completely, acknowledge the limitations of your 
knowledge. Always cite the sources of information in your response.
    """.strip(),
    
    "technical": """
You are a technical expert that provides accurate, detailed, and 
technically precise responses. Base your answers on the provided 
context documents. Use technical terminology appropriately and 
explain complex concepts clearly. If the documents don't provide 
enough information, acknowledge the limitations of your knowledge.
Always cite the sources of information in your response.
    """.strip(),
    
    "concise": """
You are a concise assistant that provides brief, to-the-point 
responses. Base your answers solely on the provided context 
documents. Focus on the most important information and avoid 
unnecessary details. If the documents don't provide enough 
information, acknowledge this briefly. Always cite the sources 
of information in your response.
    """.strip(),
    
    "customer_support": """
You are a helpful customer support assistant that provides 
friendly, supportive, and clear responses. Base your answers on 
the provided context documents. Use a conversational and 
empathetic tone. Focus on providing solutions and addressing 
the user's concerns. If the documents don't provide enough 
information, acknowledge this and suggest next steps.
Always cite the sources of information in your response.
    """.strip()
}














"""
LLM package for the RAG application.
"""
from .gemini import GeminiClient, RAGGenerator
from .prompt_templates import (
    PromptTemplate,
    RAGPromptTemplate,
    RAGChatPromptTemplate,
    SYSTEM_PROMPTS
)

__all__ = [
    'GeminiClient',
    'RAGGenerator',
    'PromptTemplate',
    'RAGPromptTemplate',
    'RAGChatPromptTemplate',
    'SYSTEM_PROMPTS'
]











"""
API schemas for the RAG application.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Schema for query request."""
    
    query: str = Field(..., description="The query text")
    sources: List[str] = Field(
        default=["confluence", "remedy"],
        description="Sources to search (confluence, remedy)"
    )
    top_k: int = Field(
        default=5,
        description="Number of results to retrieve",
        ge=1,
        le=20
    )
    mode: str = Field(
        default="hybrid",
        description="Search mode (vector, lexical, hybrid)"
    )
    multi_query: bool = Field(
        default=False,
        description="Whether to use multi-query retrieval"
    )
    rerank: bool = Field(
        default=True,
        description="Whether to rerank results"
    )
    prompt_template: Optional[str] = Field(
        default=None,
        description="Prompt template to use (general, technical, concise, customer_support)"
    )
    temperature: float = Field(
        default=0.2,
        description="Temperature for generation",
        ge=0.0,
        le=1.0
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum tokens to generate",
        ge=1,
        le=8192
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )


class QueryResponse(BaseModel):
    """Schema for query response."""
    
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources used for the answer"
    )
    query: str = Field(..., description="Original query")
    timing: Dict[str, float] = Field(
        default_factory=dict,
        description="Timing information"
    )


class ChatMessage(BaseModel):
    """Schema for chat message."""
    
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Schema for chat request."""
    
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    sources: List[str] = Field(
        default=["confluence", "remedy"],
        description="Sources to search (confluence, remedy)"
    )
    top_k: int = Field(
        default=5,
        description="Number of results to retrieve",
        ge=1,
        le=20
    )
    mode: str = Field(
        default="hybrid",
        description="Search mode (vector, lexical, hybrid)"
    )
    multi_query: bool = Field(
        default=False,
        description="Whether to use multi-query retrieval"
    )
    rerank: bool = Field(
        default=True,
        description="Whether to rerank results"
    )
    prompt_template: Optional[str] = Field(
        default=None,
        description="Prompt template to use (general, technical, concise, customer_support)"
    )
    temperature: float = Field(
        default=0.2,
        description="Temperature for generation",
        ge=0.0,
        le=1.0
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum tokens to generate",
        ge=1,
        le=8192
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )


class ChatResponse(BaseModel):
    """Schema for chat response."""
    
    message: ChatMessage = Field(..., description="Assistant message")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources used for the answer"
    )
    timing: Dict[str, float] = Field(
        default_factory=dict,
        description="Timing information"
    )


class IndexRequest(BaseModel):
    """Schema for index request."""
    
    sources: List[str] = Field(
        default=["confluence", "remedy"],
        description="Sources to index (confluence, remedy)"
    )
    confluence_space_key: Optional[str] = Field(
        default=None,
        description="Confluence space key to index"
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of documents to index"
    )
    chunking_strategy: str = Field(
        default="semantic",
        description="Chunking strategy (simple, sentence, semantic, hierarchical)"
    )
    force_reindex: bool = Field(
        default=False,
        description="Whether to force reindexing"
    )


class IndexResponse(BaseModel):
    """Schema for index response."""
    
    status: str = Field(..., description="Status of the indexing operation")
    message: str = Field(..., description="Status message")
    counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Document counts by source"
    )
    timing: Dict[str, float] = Field(
        default_factory=dict,
        description="Timing information"
    )


class HealthResponse(BaseModel):
    """Schema for health check response."""
    
    status: str = Field(..., description="Status of the application")
    version: str = Field(..., description="Application version")
    endpoints: List[str] = Field(
        default_factory=list,
        description="Available API endpoints"
    )
    sources: Dict[str, bool] = Field(
        default_factory=dict,
        description="Status of data sources"
    )
    index_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Index statistics"
    )














"""
API routes for the RAG application.
"""
from typing import List, Dict, Any, Optional, Union, Generator
import time
import json
from flask import Blueprint, request, jsonify, Response, stream_with_context

import config
from utils import get_logger, timer_decorator
from modules.data_sources import ConfluenceClient, ConfluenceContentProcessor, RemedyClient, RemedyContentProcessor
from modules.processing import ChunkerFactory, EmbeddingProcessor, IndexManager
from modules.retrieval import VectorSearchRetriever, LexicalSearchRetriever, HybridSearchRetriever, ReRanker
from modules.llm import GeminiClient, RAGGenerator, SYSTEM_PROMPTS
from .schemas import (
    QueryRequest, QueryResponse,
    ChatMessage, ChatRequest, ChatResponse,
    IndexRequest, IndexResponse,
    HealthResponse
)

# Initialize logger
logger = get_logger("api_routes")

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize components
confluence_client = ConfluenceClient()
confluence_processor = ConfluenceContentProcessor(confluence_client)
remedy_client = RemedyClient()
remedy_processor = RemedyContentProcessor(remedy_client)

embedding_processor = EmbeddingProcessor()
index_manager = IndexManager()

vector_retriever = VectorSearchRetriever(embedding_processor, index_manager)
lexical_retriever = LexicalSearchRetriever()
hybrid_retriever = HybridSearchRetriever(vector_retriever, lexical_retriever)
reranker = ReRanker(vector_retriever)

gemini_client = GeminiClient()
rag_generator = RAGGenerator(gemini_client)


@api_bp.route('/health', methods=['GET'])
@timer_decorator
def health_check():
    """Health check endpoint."""
    # Check data source connections
    confluence_status = False
    remedy_status = False
    
    try:
        spaces = confluence_client.get_spaces(limit=1)
        confluence_status = len(spaces) > 0
    except Exception as e:
        logger.error(f"Confluence health check failed: {e}")
    
    try:
        token = remedy_client.get_token()
        remedy_status = token is not None
    except Exception as e:
        logger.error(f"Remedy health check failed: {e}")
    
    # Get index stats
    index_stats = {}
    try:
        # Add vector index stats
        vector_size = getattr(index_manager.index, "ntotal", 0)
        index_stats["vector_index_size"] = vector_size
        
        # Add lexical index stats
        lexical_size = len(getattr(lexical_retriever, "doc_tokens", []))
        index_stats["lexical_index_size"] = lexical_size
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
    
    # Build response
    response = HealthResponse(
        status="ok",
        version="1.0.0",
        endpoints=[
            "/api/health",
            "/api/query",
            "/api/chat",
            "/api/index"
        ],
        sources={
            "confluence": confluence_status,
            "remedy": remedy_status
        },
        index_stats=index_stats
    )
    
    return jsonify(response.dict())


@api_bp.route('/query', methods=['POST'])
@timer_decorator
def query():
    """Query endpoint for RAG."""
    try:
        # Parse request
        req_data = request.get_json()
        query_request = QueryRequest(**req_data)
        
        # Start timing
        overall_start = time.time()
        timings = {}
        
        # Get query parameters
        query_text = query_request.query
        sources = query_request.sources
        top_k = query_request.top_k
        search_mode = query_request.mode
        multi_query = query_request.multi_query
        rerank_results = query_request.rerank
        prompt_template = query_request.prompt_template
        temperature = query_request.temperature
        max_tokens = query_request.max_tokens
        stream_response = query_request.stream
        
        logger.info(f"Query received: {query_text[:100]}...")
        
        # Select retriever based on mode
        retriever = None
        if search_mode == "vector":
            retriever = vector_retriever
        elif search_mode == "lexical":
            retriever = lexical_retriever
        else:  # hybrid is default
            retriever = hybrid_retriever
        
        # Retrieve documents
        retrieval_start = time.time()
        
        search_func = retriever.multi_query_search if multi_query else retriever.search
        results = search_func(query_text, k=top_k)
        
        retrieval_time = time.time() - retrieval_start
        timings["retrieval"] = retrieval_time
        
        # Rerank if requested
        if rerank_results and results and search_mode != "lexical":
            rerank_start = time.time()
            results = reranker.rerank(query_text, results, k=top_k)
            timings["reranking"] = time.time() - rerank_start
        
        # Generate response
        if results:
            # Get system prompt based on template
            system_prompt = None
            if prompt_template and prompt_template in SYSTEM_PROMPTS:
                system_prompt = SYSTEM_PROMPTS[prompt_template]
            
            # Generate answer
            generation_start = time.time()
            
            if stream_response:
                # Stream the response
                def generate():
                    # First, send a JSON object with metadata
                    source_info = []
                    for result in results:
                        metadata = result.get("metadata", {})
                        source_info.append({
                            "title": metadata.get("title", ""),
                            "source": metadata.get("source", "Unknown"),
                            "url": metadata.get("source_url", ""),
                            "score": result.get("score", 0)
                        })
                    
                    metadata = {
                        "query": query_text,
                        "sources": source_info,
                        "timing": {
                            "retrieval": retrieval_time
                        }
                    }
                    
                    yield json.dumps({"metadata": metadata}) + "\n"
                    
                    # Stream the answer
                    response_stream = rag_generator.generate(
                        query=query_text,
                        chunks=results,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )
                    
                    for chunk in response_stream:
                        if hasattr(chunk, "text") and chunk.text:
                            yield json.dumps({"content": chunk.text}) + "\n"
                
                return Response(stream_with_context(generate()), mimetype='application/x-ndjson')
            else:
                # Generate a complete response
                answer = rag_generator.generate(
                    query=query_text,
                    chunks=results,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                generation_time = time.time() - generation_start
                timings["generation"] = generation_time
                
                # Prepare sources for response
                sources_info = []
                for result in results:
                    metadata = result.get("metadata", {})
                    sources_info.append({
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", "Unknown"),
                        "url": metadata.get("source_url", ""),
                        "score": result.get("score", 0)
                    })
                
                # Calculate total time
                total_time = time.time() - overall_start
                timings["total"] = total_time
                
                # Build response
                response = QueryResponse(
                    answer=answer,
                    sources=sources_info,
                    query=query_text,
                    timing=timings
                )
                
                return jsonify(response.dict())
        else:
            # No results found
            if stream_response:
                def generate():
                    metadata = {
                        "query": query_text,
                        "sources": [],
                        "timing": {
                            "retrieval": retrieval_time
                        }
                    }
                    
                    yield json.dumps({"metadata": metadata}) + "\n"
                    yield json.dumps({"content": "I couldn't find any relevant information to answer your question."}) + "\n"
                
                return Response(stream_with_context(generate()), mimetype='application/x-ndjson')
            else:
                answer = "I couldn't find any relevant information to answer your question."
                
                # Calculate total time
                total_time = time.time() - overall_start
                timings["total"] = total_time
                
                # Build response
                response = QueryResponse(
                    answer=answer,
                    sources=[],
                    query=query_text,
                    timing=timings
                )
                
                return jsonify(response.dict())
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/chat', methods=['POST'])
@timer_decorator
def chat():
    """Chat endpoint for RAG."""
    try:
        # Parse request
        req_data = request.get_json()
        chat_request = ChatRequest(**req_data)
        
        # Start timing
        overall_start = time.time()
        timings = {}
        
        # Get chat parameters
        messages = chat_request.messages
        sources = chat_request.sources
        top_k = chat_request.top_k
        search_mode = chat_request.mode
        multi_query = chat_request.multi_query
        rerank_results = chat_request.rerank
        prompt_template = chat_request.prompt_template
        temperature = chat_request.temperature
        max_tokens = chat_request.max_tokens
        stream_response = chat_request.stream
        
        # Extract query from last user message
        query_text = ""
        for message in reversed(messages):
            if message.role == "user":
                query_text = message.content
                break
        
        if not query_text:
            return jsonify({"error": "No user message found"}), 400
        
        logger.info(f"Chat query received: {query_text[:100]}...")
        
        # Select retriever based on mode
        retriever = None
        if search_mode == "vector":
            retriever = vector_retriever
        elif search_mode == "lexical":
            retriever = lexical_retriever
        else:  # hybrid is default
            retriever = hybrid_retriever
        
        # Retrieve documents
        retrieval_start = time.time()
        
        search_func = retriever.multi_query_search if multi_query else retriever.search
        results = search_func(query_text, k=top_k)
        
        retrieval_time = time.time() - retrieval_start
        timings["retrieval"] = retrieval_time
        
        # Rerank if requested
        if rerank_results and results and search_mode != "lexical":
            rerank_start = time.time()
            results = reranker.rerank(query_text, results, k=top_k)
            timings["reranking"] = time.time() - rerank_start
        
        # Generate response
        if results:
            # Get system prompt based on template
            system_prompt = None
            if prompt_template and prompt_template in SYSTEM_PROMPTS:
                system_prompt = SYSTEM_PROMPTS[prompt_template]
            
            # Convert messages to format expected by generator
            formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            # Generate answer
            generation_start = time.time()
            
            if stream_response:
                # Stream the response
                def generate():
                    # First, send a JSON object with metadata
                    source_info = []
                    for result in results:
                        metadata = result.get("metadata", {})
                        source_info.append({
                            "title": metadata.get("title", ""),
                            "source": metadata.get("source", "Unknown"),
                            "url": metadata.get("source_url", ""),
                            "score": result.get("score", 0)
                        })
                    
                    metadata = {
                        "query": query_text,
                        "sources": source_info,
                        "timing": {
                            "retrieval": retrieval_time
                        }
                    }
                    
                    yield json.dumps({"metadata": metadata}) + "\n"
                    
                    # Stream the answer
                    response_stream = rag_generator.generate_chat(
                        messages=formatted_messages,
                        chunks=results,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )
                    
                    for chunk in response_stream:
                        if hasattr(chunk, "text") and chunk.text:
                            yield json.dumps({"content": chunk.text}) + "\n"
                
                return Response(stream_with_context(generate()), mimetype='application/x-ndjson')
            else:
                # Generate a complete response
                answer = rag_generator.generate_chat(
                    messages=formatted_messages,
                    chunks=results,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                generation_time = time.time() - generation_start
                timings["generation"] = generation_time
                
                # Prepare sources for response
                sources_info = []
                for result in results:
                    metadata = result.get("metadata", {})
                    sources_info.append({
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", "Unknown"),
                        "url": metadata.get("source_url", ""),
                        "score": result.get("score", 0)
                    })
                
                # Calculate total time
                total_time = time.time() - overall_start
                timings["total"] = total_time
                
                # Build response
                response = ChatResponse(
                    message=ChatMessage(role="assistant", content=answer),
                    sources=sources_info,
                    timing=timings
                )
                
                return jsonify(response.dict())
        else:
            # No results found
            if stream_response:
                def generate():
                    metadata = {
                        "query": query_text,
                        "sources": [],
                        "timing": {
                            "retrieval": retrieval_time
                        }
                    }
                    
                    yield json.dumps({"metadata": metadata}) + "\n"
                    yield json.dumps({"content": "I couldn't find any relevant information to answer your question."}) + "\n"
                
                return Response(stream_with_context(generate()), mimetype='application/x-ndjson')
            else:
                answer = "I couldn't find any relevant information to answer your question."
                
                # Calculate total time
                total_time = time.time() - overall_start
                timings["total"] = total_time
                
                # Build response
                response = ChatResponse(
                    message=ChatMessage(role="assistant", content=answer),
                    sources=[],
                    timing=timings
                )
                
                return jsonify(response.dict())
    
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/index', methods=['POST'])
@timer_decorator
def index():
    """Index data from sources."""
    try:
        # Parse request
        req_data = request.get_json()
        index_request = IndexRequest(**req_data)
        
        # Start timing
        overall_start = time.time()
        timings = {}
        
        # Get index parameters
        sources = index_request.sources
        confluence_space_key = index_request.confluence_space_key
        limit = index_request.limit
        chunking_strategy = index_request.chunking_strategy
        force_reindex = index_request.force_reindex
        
        # Create chunker
        chunker = ChunkerFactory.get_chunker(
            strategy=chunking_strategy,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        # Clear indexes if requested
        if force_reindex:
            index_manager.clear_index()
            lexical_retriever.clear_index()
        
        # Initialize document counts
        document_counts = {}
        all_chunks = []
        
        # Process Confluence data
        if "confluence" in sources:
            confluence_start = time.time()
            
            # Get Confluence pages
            if confluence_space_key:
                # Get pages from specific space
                pages = confluence_processor.process_space(
                    space_key=confluence_space_key,
                    limit=limit
                )
            else:
                # Get pages from configured space
                pages = confluence_processor.process_space(
                    space_key=config.CONFLUENCE_SPACE_ID,
                    limit=limit
                )
            
            # Create documents for indexing
            confluence_docs = []
            for page in pages:
                doc = confluence_processor.create_document_for_indexing(page)
                if doc:
                    confluence_docs.append(doc)
            
            # Chunk documents
            confluence_chunks = []
            for doc in confluence_docs:
                chunks = chunker.chunk_document(doc)
                confluence_chunks.extend(chunks)
            
            # Add to all chunks
            all_chunks.extend(confluence_chunks)
            document_counts["confluence"] = len(confluence_docs)
            
            timings["confluence"] = time.time() - confluence_start
        
        # Process Remedy data
        if "remedy" in sources:
            remedy_start = time.time()
            
            # Get Remedy data (incidents, changes, knowledge articles)
            remedy_docs = []
            
            # Get incidents
            incidents = remedy_client.search_incidents(limit=limit or 100)
            for incident in incidents:
                incident_id = incident.get("id")
                if incident_id:
                    processed = remedy_processor.process_incident(incident_id)
                    if processed:
                        doc = remedy_processor.create_document_for_indexing(processed, "incident")
                        if doc:
                            remedy_docs.append(doc)
            
            # Get change requests
            changes = remedy_client.search_change_requests(limit=limit or 100)
            for change in changes:
                change_id = change.get("id")
                if change_id:
                    processed = remedy_processor.process_change_request(change_id)
                    if processed:
                        doc = remedy_processor.create_document_for_indexing(processed, "change")
                        if doc:
                            remedy_docs.append(doc)
            
            # Get knowledge articles
            articles = remedy_client.search_knowledge_articles(limit=limit or 100)
            for article in articles:
                article_id = article.get("id")
                if article_id:
                    processed = remedy_processor.process_knowledge_article(article_id)
                    if processed:
                        doc = remedy_processor.create_document_for_indexing(processed, "knowledge")
                        if doc:
                            remedy_docs.append(doc)
            
            # Chunk documents
            remedy_chunks = []
            for doc in remedy_docs:
                chunks = chunker.chunk_document(doc)
                remedy_chunks.extend(chunks)
            
            # Add to all chunks
            all_chunks.extend(remedy_chunks)
            document_counts["remedy"] = len(remedy_docs)
            
            timings["remedy"] = time.time() - remedy_start
        
        # Index all chunks
        if all_chunks:
            indexing_start = time.time()
            
            # Index for vector search
            vector_retriever.index_chunks(all_chunks)
            
            # Index for lexical search
            lexical_retriever.index_chunks(all_chunks)
            
            timings["indexing"] = time.time() - indexing_start
        
        # Calculate total time
        total_time = time.time() - overall_start
        timings["total"] = total_time
        
        # Build response
        document_counts["total"] = sum(document_counts.values())
        document_counts["chunks"] = len(all_chunks)
        
        response = IndexResponse(
            status="success",
            message=f"Indexed {document_counts['total']} documents with {len(all_chunks)} chunks",
            counts=document_counts,
            timing=timings
        )
        
        return jsonify(response.dict())
    
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return jsonify({"error": str(e)}), 500














"""
API package for the RAG application.
"""
from .routes import api_bp
from .schemas import (
    QueryRequest, QueryResponse,
    ChatMessage, ChatRequest, ChatResponse,
    IndexRequest, IndexResponse,
    HealthResponse
)

__all__ = [
    'api_bp',
    'QueryRequest',
    'QueryResponse',
    'ChatMessage',
    'ChatRequest',
    'ChatResponse',
    'IndexRequest',
    'IndexResponse',
    'HealthResponse'
]


















"""
Modules package for the RAG application.
"""

















"""
Main application module for the RAG application.
"""
import os
from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

import config
from utils import get_logger
from modules.api import api_bp

# Initialize logger
logger = get_logger("app")

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, static_folder='static', template_folder='templates')
    
    # Configure app
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['DEBUG'] = config.DEBUG
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Add routes
    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html')
    
    @app.route('/chat')
    def chat():
        """Render the chat page."""
        return render_template('chat.html')
    
    @app.route('/favicon.ico')
    def favicon():
        """Serve favicon."""
        return send_from_directory(os.path.join(app.root_path, 'static', 'images'),
                                  'favicon.ico', mimetype='image/vnd.microsoft.icon')
    
    # Add error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors."""
        return render_template('error.html', error_code=404, error_message="Page not found"), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors."""
        logger.error(f"Server error: {str(e)}")
        return render_template('error.html', error_code=500, error_message="Server error"), 500
    
    # Validate configuration
    if not config.validate_config():
        logger.warning("Application configuration is incomplete. Some features may not work.")
    
    logger.info(f"Application initialized in {config.DEBUG and 'DEBUG' or 'PRODUCTION'} mode")
    return app

















"""
Run script for the RAG application.
"""
from app import create_app
import config
from utils import get_logger

# Initialize logger
logger = get_logger("run")

if __name__ == '__main__':
    app = create_app()
    
    logger.info(f"Starting application on port {config.PORT}")
    app.run(host='0.0.0.0', port=config.PORT, debug=config.DEBUG)















<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}RAG Confluence & Remedy{% endblock %}</title>
    <link rel="shortcut icon" href="{{ url_for('static', filename='images/favicon.ico') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    {% block extra_css %}{% endblock %}
</head>
<body>
    <header>
        <div class="header-container">
            <div class="logo">
                <a href="/">
                    <img src="{{ url_for('static', filename='images/logo.png') }}" alt="Logo">
                    <span>RAG Assistant</span>
                </a>
            </div>
            <nav>
                <ul>
                    <li><a href="/" {% if request.path == '/' %}class="active"{% endif %}>Home</a></li>
                    <li><a href="/chat" {% if request.path == '/chat' %}class="active"{% endif %}>Chat</a></li>
                </ul>
            </nav>
        </div>
    </header>
    
    <main>
        <div class="container">
            {% block content %}{% endblock %}
        </div>
    </main>
    
    <footer>
        <div class="container">
            <p>&copy; {% now year %} RAG Confluence & Remedy Assistant</p>
        </div>
    </footer>
    
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
















{% extends "base.html" %}

{% block title %}RAG Assistant - Home{% endblock %}

{% block content %}
<div class="home-container">
    <div class="hero">
        <h1>Retrieval-Augmented Generation</h1>
        <p class="subtitle">Confluence & Remedy Knowledge Assistant</p>
        <p class="description">
            Ask questions about your Confluence documentation and Remedy tickets.
            Get accurate answers powered by Google's Gemini AI with context from your enterprise data.
        </p>
        <div class="cta-buttons">
            <a href="/chat" class="btn primary">Start Chatting</a>
            <button id="indexBtn" class="btn secondary">Index Content</button>
        </div>
    </div>
    
    <div class="features">
        <div class="feature-card">
            <div class="feature-icon confluence-icon"></div>
            <h3>Confluence Integration</h3>
            <p>Search and retrieve information from your Confluence workspace, including pages, attachments, and comments.</p>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon remedy-icon"></div>
            <h3>Remedy Integration</h3>
            <p>Access data from BMC Remedy, including incidents, change requests, and knowledge articles.</p>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon gemini-icon"></div>
            <h3>Gemini AI</h3>
            <p>Powered by Google's advanced Gemini language model for intelligent, context-aware responses.</p>
        </div>
    </div>
    
    <div class="info-section">
        <h2>How It Works</h2>
        <div class="steps">
            <div class="step">
                <div class="step-number">1</div>
                <h4>Index Your Content</h4>
                <p>Content from Confluence and Remedy is indexed for fast retrieval.</p>
            </div>
            
            <div class="step">
                <div class="step-number">2</div>
                <h4>Ask a Question</h4>
                <p>Enter your question about documentation or tickets in natural language.</p>
            </div>
            
            <div class="step">
                <div class="step-number">3</div>
                <h4>Get Accurate Answers</h4>
                <p>Receive detailed answers based on your enterprise knowledge with source references.</p>
            </div>
        </div>
    </div>
</div>

<!-- Indexing Modal -->
<div id="indexModal" class="modal">
    <div class="modal-content">
        <span class="close">&times;</span>
        <h2>Index Content</h2>
        <p>Index content from Confluence and Remedy for retrieval.</p>
        
        <form id="indexForm">
            <div class="form-group">
                <label>
                    <input type="checkbox" name="sources" value="confluence" checked>
                    Confluence
                </label>
            </div>
            
            <div class="form-group">
                <label>
                    <input type="checkbox" name="sources" value="remedy" checked>
                    Remedy
                </label>
            </div>
            
            <div class="form-group">
                <label for="confluenceSpace">Confluence Space Key (optional):</label>
                <input type="text" id="confluenceSpace" name="confluenceSpace" placeholder="Leave empty for default space">
            </div>
            
            <div class="form-group">
                <label for="chunkingStrategy">Chunking Strategy:</label>
                <select id="chunkingStrategy" name="chunkingStrategy">
                    <option value="semantic">Semantic (Default)</option>
                    <option value="hierarchical">Hierarchical</option>
                    <option value="sentence">Sentence-Based</option>
                    <option value="simple">Simple</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>
                    <input type="checkbox" name="forceReindex" value="true">
                    Force Reindexing (clear existing index)
                </label>
            </div>
            
            <div class="form-actions">
                <button type="submit" class="btn primary">Start Indexing</button>
                <button type="button" class="btn secondary cancel-btn">Cancel</button>
            </div>
        </form>
        
        <div id="indexingProgress" style="display: none;">
            <h3>Indexing in Progress</h3>
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
            <p id="indexingStatus">Initializing...</p>
        </div>
        
        <div id="indexingResults" style="display: none;">
            <h3>Indexing Complete</h3>
            <div id="indexingStats"></div>
            <button class="btn primary close-results-btn">Close</button>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script>
    document.addEventListener('DOMContentLoaded', function() {
        const indexBtn = document.getElementById('indexBtn');
        const indexModal = document.getElementById('indexModal');
        const closeBtn = indexModal.querySelector('.close');
        const cancelBtn = indexModal.querySelector('.cancel-btn');
        const closeResultsBtn = indexModal.querySelector('.close-results-btn');
        
        const indexForm = document.getElementById('indexForm');
        const indexingProgress = document.getElementById('indexingProgress');
        const indexingResults = document.getElementById('indexingResults');
        const indexingStatus = document.getElementById('indexingStatus');
        const indexingStats = document.getElementById('indexingStats');
        
        // Open modal
        indexBtn.addEventListener('click', function() {
            indexModal.style.display = 'block';
            indexForm.style.display = 'block';
            indexingProgress.style.display = 'none';
            indexingResults.style.display = 'none';
        });
        
        // Close modal functions
        const closeModal = function() {
            indexModal.style.display = 'none';
        };
        
        closeBtn.addEventListener('click', closeModal);
        cancelBtn.addEventListener('click', closeModal);
        closeResultsBtn.addEventListener('click', closeModal);
        
        // Close modal if clicked outside
        window.addEventListener('click', function(event) {
            if (event.target == indexModal) {
                closeModal();
            }
        });
        
        // Handle form submission
        indexForm.addEventListener('submit', function(event) {
            event.preventDefault();
            
            // Get form data
            const formData = new FormData(indexForm);
            const sources = Array.from(formData.getAll('sources'));
            const confluenceSpace = formData.get('confluenceSpace');
            const chunkingStrategy = formData.get('chunkingStrategy');
            const forceReindex = formData.has('forceReindex');
            
            // Show progress UI
            indexForm.style.display = 'none';
            indexingProgress.style.display = 'block';
            indexingStatus.textContent = 'Starting indexing...';
            
            // Prepare request data
            const requestData = {
                sources: sources,
                chunking_strategy: chunkingStrategy,
                force_reindex: forceReindex
            };
            
            if (confluenceSpace) {
                requestData.confluence_space_key = confluenceSpace;
            }
            
            // Send indexing request
            fetch('/api/index', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Indexing failed');
                }
                return response.json();
            })
            .then(data => {
                // Show results
                indexingProgress.style.display = 'none';
                indexingResults.style.display = 'block';
                
                // Display stats
                let statsHtml = '<ul>';
                statsHtml += `<li><strong>Status:</strong> ${data.status}</li>`;
                statsHtml += `<li><strong>Message:</strong> ${data.message}</li>`;
                
                if (data.counts) {
                    statsHtml += '<li><strong>Document Counts:</strong><ul>';
                    for (const [source, count] of Object.entries(data.counts)) {
                        statsHtml += `<li>${source}: ${count}</li>`;
                    }
                    statsHtml += '</ul></li>';
                }
                
                if (data.timing) {
                    statsHtml += '<li><strong>Timing:</strong><ul>';
                    for (const [operation, time] of Object.entries(data.timing)) {
                        statsHtml += `<li>${operation}: ${time.toFixed(2)}s</li>`;
                    }
                    statsHtml += '</ul></li>';
                }
                
                statsHtml += '</ul>';
                indexingStats.innerHTML = statsHtml;
            })
            .catch(error => {
                // Show error
                indexingProgress.style.display = 'none';
                indexingResults.style.display = 'block';
                indexingStats.innerHTML = `<div class="error">${error.message}</div>`;
            });
        });
    });
</script>
{% endblock %}











{% extends "base.html" %}

{% block title %}RAG Assistant - Chat{% endblock %}

{% block extra_css %}
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
{% endblock %}

{% block content %}
<div class="chat-container">
    <div class="sidebar">
        <div class="search-settings">
            <h3>Search Settings</h3>
            <div class="form-group">
                <label>Data Sources:</label>
                <div class="checkbox-group">
                    <label>
                        <input type="checkbox" id="sourceConfluence" name="sources" value="confluence" checked>
                        Confluence
                    </label>
                    <label>
                        <input type="checkbox" id="sourceRemedy" name="sources" value="remedy" checked>
                        Remedy
                    </label>
                </div>
            </div>
            
            <div class="form-group">
                <label for="searchMode">Search Mode:</label>
                <select id="searchMode" name="searchMode">
                    <option value="hybrid" selected>Hybrid (Default)</option>
                    <option value="vector">Vector Search</option>
                    <option value="lexical">Lexical Search</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="promptTemplate">Response Style:</label>
                <select id="promptTemplate" name="promptTemplate">
                    <option value="general" selected>General (Default)</option>
                    <option value="technical">Technical</option>
                    <option value="concise">Concise</option>
                    <option value="customer_support">Customer Support</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Advanced Options:</label>
                <div class="checkbox-group">
                    <label>
                        <input type="checkbox" id="multiQuery" name="multiQuery">
                        Multi-query retrieval
                    </label>
                    <label>
                        <input type="checkbox" id="rerank" name="rerank" checked>
                        Re-rank results
                    </label>
                </div>
            </div>
            
            <div class="advanced-settings-toggle">
                <button id="advancedSettingsBtn">Advanced Settings <i class="fa fa-chevron-down"></i></button>
                <div id="advancedSettings" class="advanced-settings">
                    <div class="form-group">
                        <label for="temperature">Temperature:</label>
                        <input type="range" id="temperature" name="temperature" min="0" max="1" step="0.1" value="0.2">
                        <span id="temperatureValue">0.2</span>
                    </div>
                    
                    <div class="form-group">
                        <label for="topK">Result Count:</label>
                        <input type="number" id="topK" name="topK" min="1" max="20" value="5">
                    </div>
                    
                    <div class="form-group">
                        <label for="maxTokens">Max Tokens:</label>
                        <input type="number" id="maxTokens" name="maxTokens" min="100" max="8192" value="1024">
                    </div>
                </div>
            </div>
        </div>
        
        <div class="conversation-history">
            <h3>Conversation History</h3>
            <div id="historyList" class="history-list">
                <!-- History items will be added here dynamically -->
                <div class="empty-history">No conversation history yet.</div>
            </div>
            
            <div class="history-actions">
                <button id="clearHistoryBtn" class="btn secondary">Clear History</button>
            </div>
        </div>
    </div>
    
    <div class="chat-main">
        <div class="chat-messages" id="chatMessages">
            <div class="message system">
                <div class="message-content">
                    <p>Hello! I'm your RAG assistant for Confluence and Remedy. How can I help you today?</p>
                </div>
            </div>
            <!-- Messages will be added here dynamically -->
        </div>
        
        <div class="chat-input">
            <form id="chatForm">
                <textarea id="userInput" placeholder="Type your message here..." rows="1"></textarea>
                <button type="submit" id="sendBtn">
                    <i class="fa fa-paper-plane"></i>
                </button>
            </form>
        </div>
    </div>
    
    <div class="sources-panel" id="sourcesPanel">
        <div class="sources-header">
            <h3>Sources</h3>
            <button id="closeSourcesBtn" class="close-btn">
                <i class="fa fa-times"></i>
            </button>
        </div>
        
        <div class="sources-content" id="sourcesContent">
            <!-- Sources will be added here dynamically -->
            <div class="no-sources">No sources available for this response.</div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js"></script>
<script>
    document.addEventListener('DOMContentLoaded', function() {
        // Elements
        const chatForm = document.getElementById('chatForm');
        const userInput = document.getElementById('userInput');
        const chatMessages = document.getElementById('chatMessages');
        const sendBtn = document.getElementById('sendBtn');
        const advancedSettingsBtn = document.getElementById('advancedSettingsBtn');
        const advancedSettings = document.getElementById('advancedSettings');
        const temperatureSlider = document.getElementById('temperature');
        const temperatureValue = document.getElementById('temperatureValue');
        const sourcesPanel = document.getElementById('sourcesPanel');
        const closeSourcesBtn = document.getElementById('closeSourcesBtn');
        const sourcesContent = document.getElementById('sourcesContent');
        const historyList = document.getElementById('historyList');
        const clearHistoryBtn = document.getElementById('clearHistoryBtn');
        
        // Chat state
        let chatHistory = [];
        let activeSources = [];
        let isWaitingForResponse = false;
        
        // Load conversation history from local storage
        loadChatHistory();
        
        // Toggle advanced settings
        advancedSettingsBtn.addEventListener('click', function() {
            advancedSettings.classList.toggle('show');
            const icon = advancedSettingsBtn.querySelector('i');
            icon.classList.toggle('fa-chevron-down');
            icon.classList.toggle('fa-chevron-up');
        });
        
        // Update temperature value display
        temperatureSlider.addEventListener('input', function() {
            temperatureValue.textContent = this.value;
        });
        
        // Close sources panel
        closeSourcesBtn.addEventListener('click', function() {
            sourcesPanel.classList.remove('show');
        });
        
        // Auto-resize textarea
        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        
        // Clear history
        clearHistoryBtn.addEventListener('click', function() {
            if (confirm('Are you sure you want to clear the conversation history?')) {
                chatHistory = [];
                localStorage.removeItem('chatHistory');
                updateHistoryList();
                
                // Clear chat messages except the first system message
                while (chatMessages.children.length > 1) {
                    chatMessages.removeChild(chatMessages.lastChild);
                }
            }
        });
        
        // Handle form submission
        chatForm.addEventListener('submit', function(event) {
            event.preventDefault();
            
            const message = userInput.value.trim();
            if (!message || isWaitingForResponse) return;
            
            // Add user message to UI
            addMessageToUI('user', message);
            
            // Clear input and reset height
            userInput.value = '';
            userInput.style.height = 'auto';
            
            // Disable input while waiting for response
            isWaitingForResponse = true;
            userInput.disabled = true;
            sendBtn.disabled = true;
            
            // Add loading message
            const loadingId = 'loading-' + Date.now();
            addLoadingMessage(loadingId);
            
            // Get settings
            const settings = getSettings();
            
            // Add message to history
            chatHistory.push({ role: 'user', content: message });
            
            // Prepare request
            const requestData = {
                messages: chatHistory,
                sources: settings.sources,
                top_k: settings.topK,
                mode: settings.searchMode,
                multi_query: settings.multiQuery,
                rerank: settings.rerank,
                prompt_template: settings.promptTemplate,
                temperature: settings.temperature,
                max_tokens: settings.maxTokens,
                stream: settings.stream
            };
            
            // Send request
            if (settings.stream) {
                // Streaming response
                fetchChatStreamResponse(requestData, loadingId);
            } else {
                // Regular response
                fetchChatResponse(requestData, loadingId);
            }
        });
        
        // Fetch streaming chat response
        function fetchChatStreamResponse(requestData, loadingId) {
            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                
                // Create a reader for the stream
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let responseText = '';
                let responseSources = [];
                let metadataReceived = false;
                
                // Remove loading message
                const loadingMessage = document.getElementById(loadingId);
                if (loadingMessage) {
                    chatMessages.removeChild(loadingMessage);
                }
                
                // Add assistant message
                const messageElement = addMessageToUI('assistant', '');
                
                // Process the stream
                function processStream() {
                    return reader.read().then(({ done, value }) => {
                        if (done) {
                            // End of stream
                            if (responseText) {
                                // Update chat history
                                chatHistory.push({ role: 'assistant', content: responseText });
                                saveChatHistory();
                                updateHistoryList();
                            }
                            
                            // Update sources panel
                            if (responseSources.length > 0) {
                                updateSourcesPanel(responseSources);
                                sourcesPanel.classList.add('show');
                            }
                            
                            // Re-enable input
                            isWaitingForResponse = false;
                            userInput.disabled = false;
                            sendBtn.disabled = false;
                            userInput.focus();
                            
                            return;
                        }
                        
                        // Decode the chunk
                        const chunk = decoder.decode(value, { stream: true });
                        
                        // Process each line (each line is a JSON object)
                        const lines = chunk.split('\n').filter(line => line.trim());
                        for (const line of lines) {
                            try {
                                const data = JSON.parse(line);
                                
                                if (data.metadata && !metadataReceived) {
                                    // First chunk contains metadata
                                    metadataReceived = true;
                                    responseSources = data.metadata.sources || [];
                                } else if (data.content) {
                                    // Content chunk
                                    responseText += data.content;
                                    updateMessageContent(messageElement, responseText);
                                }
                            } catch (e) {
                                console.error('Error parsing JSON:', e);
                            }
                        }
                        
                        // Continue reading
                        return processStream();
                    });
                }
                
                return processStream();
            })
            .catch(error => {
                console.error('Error:', error);
                
                // Remove loading message
                const loadingMessage = document.getElementById(loadingId);
                if (loadingMessage) {
                    chatMessages.removeChild(loadingMessage);
                }
                
                // Add error message
                addMessageToUI('system', 'Sorry, an error occurred while processing your request.');
                
                // Re-enable input
                isWaitingForResponse = false;
                userInput.disabled = false;
                sendBtn.disabled = false;
                userInput.focus();
            });
        }
        
        // Fetch regular chat response
        function fetchChatResponse(requestData, loadingId) {
            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Remove loading message
                const loadingMessage = document.getElementById(loadingId);
                if (loadingMessage) {
                    chatMessages.removeChild(loadingMessage);
                }
                
                // Add assistant message
                const message = data.message.content;
                addMessageToUI('assistant', message);
                
                // Add to history
                chatHistory.push({ role: 'assistant', content: message });
                saveChatHistory();
                updateHistoryList();
                
                // Update sources panel
                if (data.sources && data.sources.length > 0) {
                    updateSourcesPanel(data.sources);
                    sourcesPanel.classList.add('show');
                }
                
                // Re-enable input
                isWaitingForResponse = false;
                userInput.disabled = false;
                sendBtn.disabled = false;
                userInput.focus();
            })
            .catch(error => {
                console.error('Error:', error);
                
                // Remove loading message
                const loadingMessage = document.getElementById(loadingId);
                if (loadingMessage) {
                    chatMessages.removeChild(loadingMessage);
                }
                
                // Add error message
                addMessageToUI('system', 'Sorry, an error occurred while processing your request.');
                
                // Re-enable input
                isWaitingForResponse = false;
                userInput.disabled = false;
                sendBtn.disabled = false;
                userInput.focus();
            });
        }
        
        // Get current settings
        function getSettings() {
            return {
                sources: Array.from(document.querySelectorAll('input[name="sources"]:checked')).map(el => el.value),
                searchMode: document.getElementById('searchMode').value,
                promptTemplate: document.getElementById('promptTemplate').value,
                multiQuery: document.getElementById('multiQuery').checked,
                rerank: document.getElementById('rerank').checked,
                temperature: parseFloat(document.getElementById('temperature').value),
                topK: parseInt(document.getElementById('topK').value),
                maxTokens: parseInt(document.getElementById('maxTokens').value),
                stream: true // Always use streaming for better UX
            };
        }
        
        // Add message to UI
        function addMessageToUI(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            updateMessageContent(contentDiv, content);
            
            messageDiv.appendChild(contentDiv);
            chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            return contentDiv;
        }
        
        // Update message content with markdown rendering
        function updateMessageContent(element, content) {
            // Convert markdown to HTML
            const html = marked.parse(content);
            element.innerHTML = html;
            
            // Apply syntax highlighting to code blocks
            element.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightBlock(block);
            });
        }
        
        // Add loading message
        function addLoadingMessage(id) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant loading';
            messageDiv.id = id;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'loading-indicator';
            loadingDiv.innerHTML = '<div></div><div></div><div></div>';
            
            contentDiv.appendChild(loadingDiv);
            messageDiv.appendChild(contentDiv);
            chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Update sources panel
        function updateSourcesPanel(sources) {
            // Clear current sources
            sourcesContent.innerHTML = '';
            
            if (!sources || sources.length === 0) {
                sourcesContent.innerHTML = '<div class="no-sources">No sources available for this response.</div>';
                return;
            }
            
            // Create sources list
            const sourcesList = document.createElement('ul');
            sourcesList.className = 'sources-list';
            
            sources.forEach((source, index) => {
                const sourceItem = document.createElement('li');
                sourceItem.className = 'source-item';
                
                const sourceHeader = document.createElement('div');
                sourceHeader.className = 'source-header';
                
                const sourceTitle = document.createElement('h4');
                sourceTitle.textContent = source.title || `Source ${index + 1}`;
                
                const sourceScore = document.createElement('span');
                sourceScore.className = 'source-score';
                sourceScore.textContent = `Score: ${(source.score * 100).toFixed(1)}%`;
                
                sourceHeader.appendChild(sourceTitle);
                sourceHeader.appendChild(sourceScore);
                
                const sourceDetails = document.createElement('div');
                sourceDetails.className = 'source-details';
                
                const sourceType = document.createElement('p');
                sourceType.className = 'source-type';
                sourceType.textContent = `Source: ${source.source || 'Unknown'}`;
                
                sourceDetails.appendChild(sourceType);
                
                if (source.url) {
                    const sourceLink = document.createElement('a');
                    sourceLink.className = 'source-link';
                    sourceLink.href = source.url;
                    sourceLink.target = '_blank';
                    sourceLink.textContent = 'View Source';
                    sourceDetails.appendChild(sourceLink);
                }
                
                sourceItem.appendChild(sourceHeader);
                sourceItem.appendChild(sourceDetails);
                sourcesList.appendChild(sourceItem);
            });
            
            sourcesContent.appendChild(sourcesList);
        }
        
        // Load chat history from localStorage
        function loadChatHistory() {
            const savedHistory = localStorage.getItem('chatHistory');
            if (savedHistory) {
                try {
                    chatHistory = JSON.parse(savedHistory);
                    
                    // Update history list
                    updateHistoryList();
                    
                    // Add messages to UI
                    for (const message of chatHistory) {
                        if (message.role !== 'system') {
                            addMessageToUI(message.role, message.content);
                        }
                    }
                } catch (e) {
                    console.error('Error loading chat history:', e);
                    chatHistory = [];
                }
            }
        }
        
        // Save chat history to localStorage
        function saveChatHistory() {
            // Only save the last 50 messages to prevent localStorage overflow
            const historyToSave = chatHistory.slice(-50);
            localStorage.setItem('chatHistory', JSON.stringify(historyToSave));
        }
        
        // Update conversation history list
        function updateHistoryList() {
            historyList.innerHTML = '';
            
            if (chatHistory.length === 0) {
                const emptyDiv = document.createElement('div');
                emptyDiv.className = 'empty-history';
                emptyDiv.textContent = 'No conversation history yet.';
                historyList.appendChild(emptyDiv);
                return;
            }
            
            // Group messages by conversation
            const conversations = [];
            let currentConversation = [];
            
            for (const message of chatHistory) {
                currentConversation.push(message);
                
                // End of conversation when assistant responds
                if (message.role === 'assistant') {
                    conversations.push([...currentConversation]);
                    currentConversation = [];
                }
            }
            
            // Add incomplete conversation if any
            if (currentConversation.length > 0) {
                conversations.push(currentConversation);
            }
            
            // Create history items for recent conversations
            for (let i = conversations.length - 1; i >= Math.max(0, conversations.length - 10); i--) {
                const conv = conversations[i];
                
                // Find user message
                const userMessage = conv.find(m => m.role === 'user');
                if (!userMessage) continue;
                
                const historyItem = document.createElement('div');
                historyItem.className = 'history-item';
                
                const itemContent = document.createElement('div');
                itemContent.className = 'history-item-content';
                
                const itemText = document.createElement('p');
                itemText.textContent = userMessage.content.substring(0, 50) + (userMessage.content.length > 50 ? '...' : '');
                
                itemContent.appendChild(itemText);
                historyItem.appendChild(itemContent);
                
                // Add click handler to jump to this conversation
                historyItem.addEventListener('click', function() {
                    // Find the index of this conversation in the chat history
                    const index = chatHistory.indexOf(conv[0]);
                    if (index >= 0) {
                        // Clear chat messages except the first system message
                        while (chatMessages.children.length > 1) {
                            chatMessages.removeChild(chatMessages.lastChild);
                        }
                        
                        // Add messages from this conversation and all later ones
                        for (let j = index; j < chatHistory.length; j++) {
                            if (chatHistory[j].role !== 'system') {
                                addMessageToUI(chatHistory[j].role, chatHistory[j].content);
                            }
                        }
                    }
                });
                
                historyList.appendChild(historyItem);
            }
        }
    });
</script>
{% endblock %}










{% extends "base.html" %}

{% block title %}Error {{ error_code }} - RAG Assistant{% endblock %}

{% block content %}
<div class="error-container">
    <div class="error-code">{{ error_code }}</div>
    <h1 class="error-title">{{ error_message }}</h1>
    <p class="error-description">
        {% if error_code == 404 %}
        The page you were looking for could not be found.
        {% elif error_code == 500 %}
        An internal server error occurred. Please try again later.
        {% else %}
        An error occurred. Please try again later.
        {% endif %}
    </p>
    <a href="/" class="btn primary">Go Back Home</a>
</div>
{% endblock %}










/* Base styles */
:root {
    --primary-color: #4c6ef5;
    --primary-light: #e7ebff;
    --primary-dark: #364fc7;
    --secondary-color: #868e96;
    --text-color: #212529;
    --text-light: #495057;
    --bg-color: #ffffff;
    --bg-light: #f8f9fa;
    --bg-dark: #e9ecef;
    --border-color: #dee2e6;
    --success-color: #40c057;
    --warning-color: #fab005;
    --danger-color: #fa5252;
    --font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    --border-radius: 4px;
    --box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: var(--font-family);
    color: var(--text-color);
    background-color: var(--bg-light);
    line-height: 1.6;
}

a {
    color: var(--primary-color);
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

h1, h2, h3, h4, h5, h6 {
    margin-bottom: 0.75rem;
    font-weight: 600;
    line-height: 1.2;
}

p {
    margin-bottom: 1rem;
}

ul, ol {
    margin-bottom: 1rem;
    padding-left: 1.5rem;
}

.container {
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
}

.btn {
    display: inline-block;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    font-weight: 500;
    text-align: center;
    border: 1px solid transparent;
    border-radius: var(--border-radius);
    cursor: pointer;
    transition: all 0.2s ease-in-out;
}

.btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.btn.primary {
    background-color: var(--primary-color);
    color: white;
}

.btn.primary:hover {
    background-color: var(--primary-dark);
    text-decoration: none;
}

.btn.secondary {
    background-color: var(--bg-dark);
    color: var(--text-color);
    border-color: var(--border-color);
}

.btn.secondary:hover {
    background-color: var(--border-color);
    text-decoration: none;
}

.form-group {
    margin-bottom: 1rem;
}

.form-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
}

.form-group input[type="text"],
.form-group input[type="number"],
.form-group select,
.form-group textarea {
    width: 100%;
    padding: 0.5rem;
    font-size: 1rem;
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    background-color: var(--bg-color);
}

.form-group select {
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23212529' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 0.5rem center;
    padding-right: 2rem;
}

.checkbox-group {
    margin-top: 0.25rem;
}

.checkbox-group label {
    display: flex;
    align-items: center;
    margin-bottom: 0.25rem;
    font-weight: normal;
}

.checkbox-group input[type="checkbox"] {
    margin-right: 0.5rem;
}

/* Header and Footer */
header {
    background-color: var(--bg-color);
    box-shadow: var(--box-shadow);
    padding: 1rem 0;
    margin-bottom: 2rem;
}

.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo a {
    display: flex;
    align-items: center;
    color: var(--text-color);
    font-weight: bold;
    font-size: 1.25rem;
    text-decoration: none;
}

.logo img {
    height: 40px;
    margin-right: 0.5rem;
}

nav ul {
    display: flex;
    list-style: none;
    margin: 0;
    padding: 0;
}

nav li {
    margin-left: 1.5rem;
}

nav a {
    color: var(--text-color);
    text-decoration: none;
    transition: color 0.2s ease-in-out;
}

nav a:hover, nav a.active {
    color: var(--primary-color);
}

footer {
    background-color: var(--bg-color);
    border-top: 1px solid var(--border-color);
    padding: 1.5rem 0;
    margin-top: 2rem;
    text-align: center;
    color: var(--text-light);
}

/* Home page styles */
.home-container {
    max-width: 1000px;
    margin: 0 auto;
    padding: 2rem 1rem;
}

.hero {
    text-align: center;
    margin-bottom: 4rem;
}

.hero h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    color: var(--primary-color);
}

.hero .subtitle {
    font-size: 1.5rem;
    margin-bottom: 1.5rem;
    color: var(--text-light);
}

.hero .description {
    max-width: 700px;
    margin: 0 auto 2rem;
    font-size: 1.1rem;
}

.cta-buttons {
    display: flex;
    justify-content: center;
    gap: 1rem;
}

.features {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-bottom: 4rem;
}

.feature-card {
    background-color: var(--bg-color);
    border-radius: var(--border-radius);
    padding: 2rem;
    box-shadow: var(--box-shadow);
    text-align: center;
}

.feature-icon {
    width: 60px;
    height: 60px;
    margin: 0 auto 1.5rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.5rem;
}

.confluence-icon {
    background-color: #0052CC;
}

.remedy-icon {
    background-color: #17A2B8;
}

.gemini-icon {
    background-color: #FF5722;
}

.feature-card h3 {
    margin-bottom: 1rem;
}

.info-section {
    text-align: center;
    margin-bottom: 3rem;
}

.info-section h2 {
    margin-bottom: 2rem;
}

.steps {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
}

.step {
    position: relative;
    padding: 0 1rem;
}

.step-number {
    width: 40px;
    height: 40px;
    background-color: var(--primary-color);
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    margin: 0 auto 1rem;
}

/* Modal styles */
.modal {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: 1000;
    overflow: auto;
}

.modal-content {
    background-color: var(--bg-color);
    margin: 2rem auto;
    padding: 2rem;
    border-radius: var(--border-radius);
    max-width: 600px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    position: relative;
}

.modal .close {
    position: absolute;
    top: 1rem;
    right: 1rem;
    font-size: 1.5rem;
    cursor: pointer;
}

.form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.75rem;
    margin-top: 2rem;
}

.progress-bar {
    height: 8px;
    background-color: var(--bg-dark);
    border-radius: var(--border-radius);
    margin: 1rem 0;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background-color: var(--primary-color);
    width: 100%;
    animation: progress 2s infinite ease-in-out;
}

@keyframes progress {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* Chat page styles */
.chat-container {
    display: grid;
    grid-template-columns: 300px 1fr 0fr;
    height: calc(100vh - 140px);
    background-color: var(--bg-color);
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    overflow: hidden;
}

.sidebar {
    background-color: var(--bg-light);
    border-right: 1px solid var(--border-color);
    padding: 1.5rem;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
}

.search-settings {
    margin-bottom: 2rem;
}

.search-settings h3 {
    font-size: 1.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1rem;
}

.advanced-settings-toggle {
    margin-top: 1.5rem;
}

.advanced-settings-toggle button {
    width: 100%;
    background: none;
    border: none;
    text-align: left;
    font-size: 1rem;
    color: var(--primary-color);
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.advanced-settings {
    margin-top: 1rem;
    display: none;
}

.advanced-settings.show {
    display: block;
}

.conversation-history {
    flex: 1;
}

.conversation-history h3 {
    font-size: 1.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1rem;
}

.history-list {
    margin-bottom: 1.5rem;
    max-height: 300px;
    overflow-y: auto;
}

.empty-history {
    color: var(--text-light);
    font-style: italic;
}

.history-item {
    padding: 0.75rem;
    border-radius: var(--border-radius);
    margin-bottom: 0.5rem;
    background-color: var(--bg-color);
    cursor: pointer;
    transition: background-color 0.2s ease;
}

.history-item:hover {
    background-color: var(--primary-light);
}

.history-item p {
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.history-actions {
    text-align: center;
}

.chat-main {
    display: flex;
    flex-direction: column;
    height: 100%;
}

.chat-messages {
    flex: 1;
    padding: 1.5rem;
    overflow-y: auto;
}

.message {
    margin-bottom: 1.5rem;
    display: flex;
    flex-direction: column;
}

.message:last-child {
    margin-bottom: 0;
}

.message.user .message-content {
    background-color: var(--primary-light);
    border-radius: var(--border-radius);
    padding: 1rem;
    max-width: 80%;
    align-self: flex-end;
}

.message.assistant .message-content {
    background-color: var(--bg-light);
    border-radius: var(--border-radius);
    padding: 1rem;
    max-width: 80%;
    align-self: flex-start;
}

.message.system .message-content {
    background-color: var(--bg-dark);
    border-radius: var(--border-radius);
    padding: 1rem;
    max-width: 80%;
    align-self: center;
    text-align: center;
    font-style: italic;
}

.message.loading .message-content {
    padding: 1rem;
    display: flex;
    justify-content: center;
}

.loading-indicator {
    display: flex;
    gap: 0.25rem;
}

.loading-indicator div {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background-color: var(--primary-color);
    animation: bounce 1.4s infinite ease-in-out;
    animation-fill-mode: both;
}

.loading-indicator div:nth-child(1) {
    animation-delay: -0.32s;
}

.loading-indicator div:nth-child(2) {
    animation-delay: -0.16s;
}

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}

.message-content p:last-child {
    margin-bottom: 0;
}

.message-content pre {
    background-color: #f6f8fa;
    border-radius: var(--border-radius);
    padding: 0.75rem;
    overflow-x: auto;
    margin: 0.75rem 0;
}

.message-content code {
    font-family: Consolas, Monaco, 'Andale Mono', monospace;
    font-size: 0.9rem;
}

.message-content blockquote {
    border-left: 4px solid var(--border-color);
    padding-left: 1rem;
    margin-left: 0;
    color: var(--text-light);
}

.chat-input {
    background-color: var(--bg-color);
    border-top: 1px solid var(--border-color);
    padding: 1rem 1.5rem;
}

.chat-input form {
    display: flex;
    align-items: center;
}

.chat-input textarea {
    flex: 1;
    resize: none;
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    padding: 0.75rem;
    font-family: var(--font-family);
    font-size: 1rem;
    max-height: 150px;
    overflow-y: auto;
}

.chat-input button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    border-radius: var(--border-radius);
    width: 40px;
    height: 40px;
    margin-left: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: background-color 0.2s ease;
}

.chat-input button:hover {
    background-color: var(--primary-dark);
}

.sources-panel {
    width: 0;
    background-color: var(--bg-light);
    border-left: 1px solid var(--border-color);
    transition: width 0.3s ease;
    overflow: hidden;
}

.sources-panel.show {
    width: 350px;
}

.sources-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem;
    border-bottom: 1px solid var(--border-color);
}

.sources-header h3 {
    margin: 0;
    font-size: 1.2rem;
}

.close-btn {
    background: none;
    border: none;
    cursor: pointer;
    font-size: 1.25rem;
    color: var(--text-light);
}

.sources-content {
    padding: 1.5rem;
    overflow-y: auto;
    height: calc(100% - 70px);
}

.no-sources {
    color: var(--text-light);
    font-style: italic;
    text-align: center;
    margin-top: 2rem;
}

.sources-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.source-item {
    background-color: var(--bg-color);
    border-radius: var(--border-radius);
    padding: 1rem;
    margin-bottom: 1rem;
}

.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

.source-header h4 {
    margin: 0;
    font-size: 1rem;
}

.source-score {
    font-size: 0.85rem;
    color: var(--text-light);
}

.source-details p {
    margin: 0 0 0.5rem;
    font-size: 0.9rem;
    color: var(--text-light);
}

.source-link {
    font-size: 0.9rem;
}

/* Error page styles */
.error-container {
    max-width: 600px;
    margin: 3rem auto;
    text-align: center;
    padding: 2rem;
    background-color: var(--bg-color);
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
}

.error-code {
    font-size: 5rem;
    font-weight: 700;
    color: var(--primary-color);
    margin-bottom: 1rem;
}

.error-title {
    margin-bottom: 1.5rem;
}

.error-description {
    margin-bottom: 2rem;
    color: var(--text-light);
}

/* Responsive styles */
@media (max-width: 992px) {
    .chat-container {
        grid-template-columns: 1fr;
        grid-template-rows: 1fr auto;
    }
    
    .sidebar {
        display: none;
    }
    
    .sources-panel.show {
        position: fixed;
        top: 0;
        right: 0;
        bottom: 0;
        width: 300px;
        z-index: 900;
    }
}

@media (max-width: 768px) {
    .features {
        grid-template-columns: 1fr;
    }
    
    .hero h1 {
        font-size: 2rem;
    }
    
    .hero .subtitle {
        font-size: 1.25rem;
    }
    
    .cta-buttons {
        flex-direction: column;
        gap: 0.75rem;
    }
    
    .sources-panel.show {
        width: 260px;
    }
}

@media (max-width: 576px) {
    .header-container {
        flex-direction: column;
        gap: 1rem;
    }
    
    nav ul {
        justify-content: center;
    }
    
    nav li {
        margin: 0 0.75rem;
    }
    
    .steps {
        grid-template-columns: 1fr;
    }
    
    .modal-content {
        margin: 1rem;
        padding: 1.5rem;
    }
}















/**
 * Main JavaScript for RAG application
 */

document.addEventListener('DOMContentLoaded', function() {
    // Auto-resize textareas
    const autoResizeTextareas = function() {
        document.querySelectorAll('textarea').forEach(textarea => {
            textarea.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = (this.scrollHeight) + 'px';
            });
            
            // Initial resize
            textarea.style.height = 'auto';
            textarea.style.height = (textarea.scrollHeight) + 'px';
        });
    };
    
    // Highlight code blocks
    const highlightCode = function() {
        if (window.hljs) {
            document.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightBlock(block);
            });
        }
    };
    
    // Initialize range input displays
    const initRangeInputs = function() {
        document.querySelectorAll('input[type="range"]').forEach(input => {
            const valueDisplay = document.getElementById(`${input.id}Value`);
            if (valueDisplay) {
                valueDisplay.textContent = input.value;
                
                input.addEventListener('input', function() {
                    valueDisplay.textContent = this.value;
                });
            }
        });
    };
    
    // Initialize tooltips
    const initTooltips = function() {
        const tooltips = document.querySelectorAll('[data-tooltip]');
        
        tooltips.forEach(tooltip => {
            tooltip.addEventListener('mouseenter', function() {
                const text = this.getAttribute('data-tooltip');
                
                const tooltipElement = document.createElement('div');
                tooltipElement.className = 'tooltip';
                tooltipElement.textContent = text;
                
                document.body.appendChild(tooltipElement);
                
                const rect = this.getBoundingClientRect();
                tooltipElement.style.top = `${rect.top - tooltipElement.offsetHeight - 10}px`;
                tooltipElement.style.left = `${rect.left + (rect.width / 2) - (tooltipElement.offsetWidth / 2)}px`;
                
                tooltipElement.classList.add('show');
                
                this.addEventListener('mouseleave', function() {
                    tooltipElement.remove();
                }, { once: true });
            });
        });
    };
    
    // Check if element is in viewport
    const isInViewport = function(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    };
    
    // Scroll to element if not in viewport
    const scrollToElement = function(element) {
        if (!isInViewport(element)) {
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    };
    
    // Format timestamps
    const formatTimestamp = function(date) {
        const now = new Date();
        const yesterday = new Date(now);
        yesterday.setDate(yesterday.getDate() - 1);
        
        const messageDate = new Date(date);
        
        // If same day, return time
        if (messageDate.toDateString() === now.toDateString()) {
            return messageDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }
        
        // If yesterday, return 'Yesterday'
        if (messageDate.toDateString() === yesterday.toDateString()) {
            return 'Yesterday';
        }
        
        // Otherwise, return date
        return messageDate.toLocaleDateString();
    };
    
    // Call initialization functions
    autoResizeTextareas();
    highlightCode();
    initRangeInputs();
    initTooltips();
    
    // Make these functions available globally
    window.appUtils = {
        autoResizeTextareas,
        highlightCode,
        initRangeInputs,
        initTooltips,
        isInViewport,
        scrollToElement,
        formatTimestamp
    };
});
