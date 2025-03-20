requirements.txt - # API Integration
requests==2.31.0
atlassian-python-api==3.41.7
urllib3==2.0.4
python-dotenv==1.0.0

# Core NLP & Data Processing
nltk==3.8.1
spacy==3.7.2
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.0.3
beautifulsoup4==4.12.2
lxml==4.9.3
pdfminer.six==20221105
python-docx==0.8.11
Pillow==10.0.1
pytesseract==0.3.10
rank_bm25==0.2.2

# Vector Databases & Embeddings
faiss-cpu==1.7.4
sentence-transformers==2.2.2

# Web Application
flask==2.3.3
flask-cors==4.0.0
gunicorn==21.2.0

# Utilities
tqdm==4.66.1
python-dateutil==2.8.2







app/config.py - 
"""
Configuration settings for the Enterprise Knowledge Hub.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Application settings
APP_NAME = "Enterprise Knowledge Hub"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
PORT = int(os.getenv("PORT", "5000"))
HOST = os.getenv("HOST", "0.0.0.0")

# API Keys and endpoints
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", "")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME", "")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN", "")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY", "")

# Remedy API settings
REMEDY_SERVER = os.getenv("REMEDY_SERVER", "")
REMEDY_USERNAME = os.getenv("REMEDY_USERNAME", "")
REMEDY_PASSWORD = os.getenv("REMEDY_PASSWORD", "")

# Data processing settings
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cache"))
INDICES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "indices"))
MAX_CACHE_AGE_DAYS = int(os.getenv("MAX_CACHE_AGE_DAYS", "7"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# NLP settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))

# Make sure required directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICES_DIR, exist_ok=True)

def validate_config():
    """Validate the configuration settings."""
    required_vars = [
        "CONFLUENCE_URL",
        "CONFLUENCE_USERNAME",
        "CONFLUENCE_API_TOKEN",
        "CONFLUENCE_SPACE_KEY",
        "REMEDY_SERVER",
        "REMEDY_USERNAME",
        "REMEDY_PASSWORD"
    ]

    missing_vars = [var for var in required_vars if not globals()[var]]
    
    if missing_vars:
        print(f"Warning: Missing configuration: {', '.join(missing_vars)}")
        print("Some features may not work correctly.")
    
    return len(missing_vars) == 0

# Check configuration validity on import
CONFIG_VALID = validate_config()





app/api/confluence.py - 
"""
Confluence API integration module for retrieving and processing Confluence content.
"""
import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import requests
from requests.auth import HTTPBasicAuth
from atlassian import Confluence

from app.config import (
    CONFLUENCE_URL, 
    CONFLUENCE_USERNAME, 
    CONFLUENCE_API_TOKEN,
    CONFLUENCE_SPACE_KEY,
    CACHE_DIR
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfluenceClient:
    """Client for interacting with Confluence API."""
    
    def __init__(self):
        """Initialize the Confluence client."""
        self.url = CONFLUENCE_URL
        self.username = CONFLUENCE_USERNAME
        self.api_token = CONFLUENCE_API_TOKEN
        self.space_key = CONFLUENCE_SPACE_KEY
        self.cache_dir = os.path.join(CACHE_DIR, "confluence")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize the Confluence client
        self.client = None
        if self.url and self.username and self.api_token:
            try:
                self.client = Confluence(
                    url=self.url,
                    username=self.username,
                    password=self.api_token,
                    cloud=True  # Set to False for server installations
                )
                logger.info("Confluence client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Confluence client: {str(e)}")
        else:
            logger.warning("Confluence credentials not provided. Client not initialized.")

    def is_connected(self) -> bool:
        """Check if the Confluence client is connected."""
        if not self.client:
            return False
        
        try:
            # Try to get spaces to verify connection
            spaces = self.client.get_all_spaces(start=0, limit=1)
            return True
        except Exception as e:
            logger.error(f"Connection check failed: {str(e)}")
            return False

    def get_space_information(self) -> Dict[str, Any]:
        """Get information about the configured Confluence space."""
        if not self.client or not self.space_key:
            logger.error("Confluence client not initialized or space key not provided")
            return {}
        
        try:
            space_info = self.client.get_space(self.space_key)
            return space_info
        except Exception as e:
            logger.error(f"Failed to get space information: {str(e)}")
            return {}

    def get_all_pages(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all pages from the configured Confluence space.
        
        Args:
            limit: Maximum number of pages to retrieve
            
        Returns:
            List of page dictionaries with metadata
        """
        if not self.client or not self.space_key:
            logger.error("Confluence client not initialized or space key not provided")
            return []
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"{self.space_key}_pages.json")
        if os.path.exists(cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(days=1):  # Cache for 1 day
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cached pages: {str(e)}")
        
        try:
            # Get all pages from the space
            pages = []
            start = 0
            
            while True:
                results = self.client.get_all_pages_from_space(
                    space=self.space_key, 
                    start=start, 
                    limit=50,
                    expand="version"
                )
                
                if not results:
                    break
                    
                pages.extend(results)
                start += 50
                
                if len(pages) >= limit:
                    pages = pages[:limit]
                    break
                    
                # Add a small delay to avoid rate limiting
                time.sleep(0.1)
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(pages, f)
                
            return pages
            
        except Exception as e:
            logger.error(f"Failed to get pages from space: {str(e)}")
            return []

    def get_page_content(self, page_id: str, expand: str = "body.storage") -> Dict[str, Any]:
        """
        Get the content of a specific Confluence page.
        
        Args:
            page_id: ID of the page to retrieve
            expand: Content expansion parameters
            
        Returns:
            Dictionary with page content and metadata
        """
        if not self.client:
            logger.error("Confluence client not initialized")
            return {}
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"page_{page_id}.json")
        if os.path.exists(cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(days=1):  # Cache for 1 day
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cached page: {str(e)}")
        
        try:
            content = self.client.get_page_by_id(
                page_id=page_id,
                expand=expand
            )
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(content, f)
                
            return content
            
        except Exception as e:
            logger.error(f"Failed to get page content: {str(e)}")
            return {}

    def search_content(self, query: str, cql: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for content in Confluence.
        
        Args:
            query: Search query string
            cql: Optional Confluence Query Language string for advanced filtering
            limit: Maximum number of results to return
            
        Returns:
            List of search result dictionaries
        """
        if not self.client:
            logger.error("Confluence client not initialized")
            return []
        
        try:
            # Build CQL query with space restriction if no custom CQL provided
            if not cql and self.space_key:
                cql = f"space = \"{self.space_key}\" AND text ~ \"{query}\""
            elif not cql:
                cql = f"text ~ \"{query}\""
            
            results = self.client.cql(cql, limit=limit)
            
            if 'results' in results:
                return results['results']
            return []
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def get_attachments(self, page_id: str) -> List[Dict[str, Any]]:
        """
        Get all attachments for a specific page.
        
        Args:
            page_id: ID of the page
            
        Returns:
            List of attachment dictionaries
        """
        if not self.client:
            logger.error("Confluence client not initialized")
            return []
        
        try:
            attachments = self.client.get_attachments_from_content(page_id)
            return attachments.get('results', [])
        except Exception as e:
            logger.error(f"Failed to get attachments: {str(e)}")
            return []

    def download_attachment(self, attachment_id: str, filename: str) -> str:
        """
        Download a specific attachment.
        
        Args:
            attachment_id: ID of the attachment
            filename: Name of the attachment
            
        Returns:
            Path to the downloaded file or empty string on failure
        """
        if not self.client:
            logger.error("Confluence client not initialized")
            return ""
        
        try:
            # Create attachments directory if it doesn't exist
            attachments_dir = os.path.join(self.cache_dir, "attachments")
            os.makedirs(attachments_dir, exist_ok=True)
            
            # Download the attachment
            filename_clean = os.path.basename(filename)
            file_path = os.path.join(attachments_dir, f"{attachment_id}_{filename_clean}")
            
            # Check if already downloaded
            if os.path.exists(file_path):
                return file_path
                
            content = self.client.download_attachment(attachment_id)
            
            with open(file_path, 'wb') as f:
                f.write(content)
                
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download attachment: {str(e)}")
            return ""

    def get_page_history(self, page_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the version history of a page.
        
        Args:
            page_id: ID of the page
            limit: Maximum number of versions to retrieve
            
        Returns:
            List of version dictionaries
        """
        if not self.client:
            logger.error("Confluence client not initialized")
            return []
        
        try:
            history = self.client.get_content_history(page_id, limit=limit)
            return history.get('latest', [])
        except Exception as e:
            logger.error(f"Failed to get page history: {str(e)}")
            return []

# Create a singleton instance
confluence_client = ConfluenceClient()








app/api/remedy.py - 
"""
Remedy API integration module for retrieving and processing ticket information.
"""
import os
import json
import logging
import time
import base64
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import requests
from requests.exceptions import RequestException

from app.config import (
    REMEDY_SERVER,
    REMEDY_USERNAME,
    REMEDY_PASSWORD,
    CACHE_DIR
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RemedyClient:
    """Client for interacting with Remedy API."""
    
    def __init__(self):
        """Initialize the Remedy client."""
        self.server = REMEDY_SERVER
        self.username = REMEDY_USERNAME
        self.password = REMEDY_PASSWORD
        self.token = None
        self.token_expiry = None
        self.cache_dir = os.path.join(CACHE_DIR, "remedy")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Endpoint URLs
        self.login_url = f"https://{self.server}/api/jwt/login"
        self.logout_url = f"https://{self.server}/api/jwt/logout"
        self.incident_url = f"https://{self.server}/api/arsys/v1/entry/HPD:Help Desk"
        self.kb_url = f"https://{self.server}/api/arsys/v1/entry/KNW:Knowledge"
        
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication for API requests."""
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        
        if self.token:
            headers["Authorization"] = f"AR-JWT {self.token}"
            
        return headers
    
    def login(self) -> bool:
        """
        Login to Remedy and get authentication token.
        
        Returns:
            Boolean indicating success or failure
        """
        if not self.server or not self.username or not self.password:
            logger.error("Remedy credentials not provided")
            return False
        
        # Check if token is still valid
        if self.token and self.token_expiry and datetime.now() < self.token_expiry:
            return True
        
        try:
            payload = {
                "username": self.username,
                "password": self.password
            }
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            response = requests.post(
                self.login_url,
                data=payload,
                headers=headers,
                verify=True
            )
            
            if response.status_code == 200:
                self.token = response.text
                # Set token expiry to 8 hours from now (typical Remedy JWT expiry)
                self.token_expiry = datetime.now() + timedelta(hours=8)
                logger.info("Successfully logged in to Remedy")
                return True
            else:
                logger.error(f"Failed to login to Remedy: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error during Remedy login: {str(e)}")
            return False
    
    def logout(self) -> bool:
        """
        Logout from Remedy and invalidate the token.
        
        Returns:
            Boolean indicating success or failure
        """
        if not self.token:
            return True
            
        try:
            headers = self._get_headers()
            
            response = requests.post(
                self.logout_url,
                headers=headers,
                verify=True
            )
            
            if response.status_code == 204:
                self.token = None
                self.token_expiry = None
                logger.info("Successfully logged out from Remedy")
                return True
            else:
                logger.error(f"Failed to logout from Remedy: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error during Remedy logout: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """Check if the client is connected to Remedy."""
        if not self.server or not self.username or not self.password:
            return False
            
        return self.login()
    
    def get_incident(self, incident_id: str) -> Dict[str, Any]:
        """
        Get a specific incident by ID.
        
        Args:
            incident_id: The incident ID to retrieve
            
        Returns:
            Dictionary containing incident details or empty dict on failure
        """
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"incident_{incident_id}.json")
        if os.path.exists(cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(hours=1):  # Cache for 1 hour
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cached incident: {str(e)}")
        
        if not self.login():
            logger.error("Not logged in to Remedy")
            return {}
            
        try:
            # Ensure the incident ID is in the correct format (e.g., INC000001234567)
            if not re.match(r'^INC\d+$', incident_id):
                incident_id = f"INC{incident_id.zfill(12)}" if incident_id.isdigit() else incident_id
            
            url = f"{self.incident_url}/{incident_id}"
            headers = self._get_headers()
            
            response = requests.get(
                url,
                headers=headers,
                verify=True
            )
            
            if response.status_code == 200:
                incident_data = response.json()
                
                # Save to cache
                with open(cache_file, 'w') as f:
                    json.dump(incident_data, f)
                    
                return incident_data
            else:
                logger.error(f"Failed to get incident {incident_id}: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting incident {incident_id}: {str(e)}")
            return {}
    
    def search_incidents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for incidents based on query.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of incident dictionaries
        """
        if not self.login():
            logger.error("Not logged in to Remedy")
            return []
            
        try:
            # Build a qualification string for the API
            qualification = f"'Incident Number' LIKE \"%{query}%\" OR 'Description' LIKE \"%{query}%\" OR 'Detailed Description' LIKE \"%{query}%\""
            
            url = f"{self.incident_url}?limit={limit}&q={qualification}"
            headers = self._get_headers()
            
            response = requests.get(
                url,
                headers=headers,
                verify=True
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('entries', [])
            else:
                logger.error(f"Failed to search incidents: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching incidents: {str(e)}")
            return []
    
    def get_all_active_incidents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all active incidents.
        
        Args:
            limit: Maximum number of incidents to retrieve
            
        Returns:
            List of incident dictionaries
        """
        # Check cache first
        cache_file = os.path.join(self.cache_dir, "active_incidents.json")
        if os.path.exists(cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(hours=1):  # Cache for 1 hour
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cached incidents: {str(e)}")
        
        if not self.login():
            logger.error("Not logged in to Remedy")
            return []
            
        try:
            # Get active incidents (Status not Closed, Cancelled, or Resolved)
            qualification = "'Status' != \"Closed\" AND 'Status' != \"Cancelled\" AND 'Status' != \"Resolved\""
            
            url = f"{self.incident_url}?limit={limit}&q={qualification}"
            headers = self._get_headers()
            
            response = requests.get(
                url,
                headers=headers,
                verify=True
            )
            
            if response.status_code == 200:
                data = response.json()
                incidents = data.get('entries', [])
                
                # Save to cache
                with open(cache_file, 'w') as f:
                    json.dump(incidents, f)
                    
                return incidents
            else:
                logger.error(f"Failed to get active incidents: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting active incidents: {str(e)}")
            return []
    
    def get_knowledge_article(self, article_id: str) -> Dict[str, Any]:
        """
        Get a knowledge article by ID.
        
        Args:
            article_id: ID of the knowledge article to retrieve
            
        Returns:
            Dictionary containing the article or empty dict on failure
        """
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"kb_{article_id}.json")
        if os.path.exists(cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(days=1):  # Cache for 1 day
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cached KB article: {str(e)}")
        
        if not self.login():
            logger.error("Not logged in to Remedy")
            return {}
            
        try:
            url = f"{self.kb_url}/{article_id}"
            headers = self._get_headers()
            
            response = requests.get(
                url,
                headers=headers,
                verify=True
            )
            
            if response.status_code == 200:
                article_data = response.json()
                
                # Save to cache
                with open(cache_file, 'w') as f:
                    json.dump(article_data, f)
                    
                return article_data
            else:
                logger.error(f"Failed to get KB article {article_id}: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting KB article: {str(e)}")
            return {}
    
    def search_knowledge_base(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for articles matching the query.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of knowledge article dictionaries
        """
        if not self.login():
            logger.error("Not logged in to Remedy")
            return []
            
        try:
            # Build a qualification string for the API
            qualification = f"'Article Title' LIKE \"%{query}%\" OR 'Article Text' LIKE \"%{query}%\""
            
            url = f"{self.kb_url}?limit={limit}&q={qualification}"
            headers = self._get_headers()
            
            response = requests.get(
                url,
                headers=headers,
                verify=True
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('entries', [])
            else:
                logger.error(f"Failed to search KB: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching KB: {str(e)}")
            return []

# Create a singleton instance
remedy_client = RemedyClient()






app/core/document_processor.py
"""
Document processor module for extracting and processing text from various sources.
"""
import os
import re
import logging
import json
import hashlib
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import tempfile

import nltk
from bs4 import BeautifulSoup
import pandas as pd
import spacy
from PIL import Image
import pytesseract
import docx
from pdfminer.high_level import extract_text

from app.api.confluence import confluence_client
from app.api.remedy import remedy_client
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, CACHE_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("SpaCy model not found. Downloading en_core_web_sm...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class Document:
    """Class representing a document with text content and metadata."""
    
    def __init__(
        self, 
        content: str, 
        source: str, 
        doc_id: str, 
        title: str = "", 
        url: str = "", 
        created_at: Optional[datetime] = None,
        last_updated: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a Document object.
        
        Args:
            content: The text content of the document
            source: Source system (e.g., 'confluence', 'remedy')
            doc_id: Unique identifier in the source system
            title: Document title
            url: URL to the document
            created_at: Creation timestamp
            last_updated: Last update timestamp
            metadata: Additional metadata
        """
        self.content = content
        self.source = source
        self.doc_id = doc_id
        self.title = title
        self.url = url
        self.created_at = created_at
        self.last_updated = last_updated
        self.metadata = metadata or {}
        self.chunks = []
        
    def __repr__(self) -> str:
        return f"Document(source={self.source}, id={self.doc_id}, title={self.title})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "source": self.source,
            "doc_id": self.doc_id,
            "title": self.title,
            "url": self.url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "metadata": self.metadata,
            "chunk_count": len(self.chunks)
        }

class DocumentChunk:
    """Class representing a chunk of a document for retrieval."""
    
    def __init__(
        self,
        text: str,
        doc_id: str,
        chunk_id: int,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a DocumentChunk.
        
        Args:
            text: The text content of the chunk
            doc_id: ID of the source document
            chunk_id: ID of the chunk within the document
            source: Source system (e.g., 'confluence', 'remedy')
            metadata: Additional metadata
        """
        self.text = text
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.source = source
        self.metadata = metadata or {}
        
    def __repr__(self) -> str:
        return f"DocumentChunk(source={self.source}, doc_id={self.doc_id}, chunk_id={self.chunk_id})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary representation."""
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "metadata": self.metadata
        }

class DocumentProcessor:
    """Processor for extracting text from various document types and sources."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.documents = {}  # Cache of processed documents
        self.chunks = []  # All document chunks
        self.processed_sources = set()  # Track processed sources
        self.cache_dir = os.path.join(CACHE_DIR, "processed")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def process_all_sources(self, force_reload: bool = False) -> int:
        """
        Process all available data sources.
        
        Args:
            force_reload: Whether to force reprocessing of already processed sources
            
        Returns:
            Total number of documents processed
        """
        total_docs = 0
        
        # Process Confluence pages
        if confluence_client.is_connected() and (force_reload or "confluence" not in self.processed_sources):
            logger.info("Processing Confluence pages...")
            confluence_docs = self.process_confluence_content()
            total_docs += len(confluence_docs)
            self.processed_sources.add("confluence")
            
        # Process Remedy tickets
        if remedy_client.is_connected() and (force_reload or "remedy" not in self.processed_sources):
            logger.info("Processing Remedy tickets...")
            remedy_docs = self.process_remedy_content()
            total_docs += len(remedy_docs)
            self.processed_sources.add("remedy")
            
        logger.info(f"Processed {total_docs} documents from all sources")
        return total_docs
    
    def process_confluence_content(self) -> List[Document]:
        """
        Process content from Confluence.
        
        Returns:
            List of processed Document objects
        """
        processed_docs = []
        
        # Get all pages from configured space
        pages = confluence_client.get_all_pages()
        
        for page in pages:
            try:
                page_id = page.get('id')
                if not page_id:
                    continue
                    
                # Check if already processed and cached
                doc_id = f"confluence_{page_id}"
                if doc_id in self.documents:
                    processed_docs.append(self.documents[doc_id])
                    continue
                    
                # Get detailed page content
                page_content = confluence_client.get_page_content(page_id)
                
                if not page_content:
                    continue
                    
                title = page_content.get('title', '')
                
                # Get HTML content
                body = page_content.get('body', {})
                storage = body.get('storage', {})
                html_content = storage.get('value', '')
                
                if not html_content:
                    continue
                    
                # Extract text from HTML
                text_content = self._extract_text_from_html(html_content)
                
                # Create document object
                space = page_content.get('space', {})
                space_key = space.get('key', '')
                
                # Build URL
                url = f"{confluence_client.url}/display/{space_key}/{page_id}"
                
                # Get timestamps
                version = page_content.get('version', {})
                when = version.get('when', '')
                created = page_content.get('history', {}).get('createdDate', '')
                
                created_at = datetime.fromisoformat(created.replace('Z', '+00:00')) if created else None
                updated_at = datetime.fromisoformat(when.replace('Z', '+00:00')) if when else None
                
                # Additional metadata
                metadata = {
                    'space_key': space_key,
                    'content_type': 'page',
                    'version': version.get('number', 1),
                    'has_attachments': False
                }
                
                # Check for attachments
                attachments = confluence_client.get_attachments(page_id)
                if attachments:
                    metadata['has_attachments'] = True
                    metadata['attachment_count'] = len(attachments)
                    
                    # Process attachments
                    attachment_texts = []
                    for attachment in attachments:
                        attachment_id = attachment.get('id')
                        filename = attachment.get('title', '')
                        
                        if attachment_id and filename:
                            # Process certain file types
                            file_ext = os.path.splitext(filename)[1].lower()
                            if file_ext in ['.pdf', '.docx', '.txt', '.jpg', '.jpeg', '.png']:
                                attachment_text = self._process_attachment(attachment_id, filename)
                                if attachment_text:
                                    attachment_texts.append(f"ATTACHMENT [{filename}]: {attachment_text}")
                    
                    # Add attachment text to content
                    if attachment_texts:
                        text_content += "\n\nATTACHMENTS:\n" + "\n\n".join(attachment_texts)
                
                # Create the document
                doc = Document(
                    content=text_content,
                    source='confluence',
                    doc_id=page_id,
                    title=title,
                    url=url,
                    created_at=created_at,
                    last_updated=updated_at,
                    metadata=metadata
                )
                
                # Create chunks
                self._create_chunks(doc)
                
                # Add to documents dictionary
                self.documents[doc_id] = doc
                processed_docs.append(doc)
                
                logger.info(f"Processed Confluence page: {title} ({page_id})")
                
            except Exception as e:
                logger.error(f"Error processing Confluence page {page.get('id', 'unknown')}: {str(e)}")
        
        logger.info(f"Processed {len(processed_docs)} Confluence pages")
        return processed_docs
    
    def process_remedy_content(self) -> List[Document]:
        """
        Process content from Remedy.
        
        Returns:
            List of processed Document objects
        """
        processed_docs = []
        
        # Get active incidents
        incidents = remedy_client.get_all_active_incidents()
        
        for incident in incidents:
            try:
                values = incident.get('values', {})
                incident_id = values.get('Incident Number', '')
                
                if not incident_id:
                    continue
                    
                # Check if already processed and cached
                doc_id = f"remedy_{incident_id}"
                if doc_id in self.documents:
                    processed_docs.append(self.documents[doc_id])
                    continue
                
                # Extract fields
                title = values.get('Summary', '')
                description = values.get('Description', '')
                detailed_description = values.get('Detailed Description', '')
                status = values.get('Status', '')
                urgency = values.get('Urgency', '')
                impact = values.get('Impact', '')
                assigned_group = values.get('Assigned Group', '')
                assignee = values.get('Assignee', '')
                service = values.get('Service', '')
                reported_date = values.get('Reported Date', '')
                resolved_date = values.get('Resolved Date', '')
                
                # Combine content
                content_parts = [
                    f"Incident: {incident_id}",
                    f"Summary: {title}",
                    f"Status: {status}",
                    f"Urgency: {urgency}",
                    f"Impact: {impact}",
                    f"Assigned Group: {assigned_group}",
                    f"Assignee: {assignee}",
                    f"Service: {service}",
                    f"Description: {description}",
                    f"Detailed Description: {detailed_description}"
                ]
                
                content = "\n\n".join([p for p in content_parts if p])
                
                # Parse dates
                created_at = None
                if reported_date:
                    try:
                        created_at = datetime.strptime(reported_date, "%Y-%m-%d %H:%M:%S GMT")
                    except:
                        pass
                        
                resolved_at = None
                if resolved_date:
                    try:
                        resolved_at = datetime.strptime(resolved_date, "%Y-%m-%d %H:%M:%S GMT")
                    except:
                        pass
                
                # URL (if applicable)
                url = f"{remedy_client.server}/arsys/forms/onbmc-s/HPD:Help+Desk/Incident+Console/?mode=search&F304255500='{incident_id}'"
                
                # Additional metadata
                metadata = {
                    'status': status,
                    'urgency': urgency,
                    'impact': impact,
                    'assigned_group': assigned_group,
                    'assignee': assignee,
                    'service': service,
                    'resolved_date': resolved_at.isoformat() if resolved_at else None
                }
                
                # Create document
                doc = Document(
                    content=content,
                    source='remedy',
                    doc_id=incident_id,
                    title=title,
                    url=url,
                    created_at=created_at,
                    last_updated=created_at,  # Use created_at as last_updated since we don't have a modified date
                    metadata=metadata
                )
                
                # Create chunks
                self._create_chunks(doc)
                
                # Add to documents dictionary
                self.documents[doc_id] = doc
                processed_docs.append(doc)
                
                logger.info(f"Processed Remedy incident: {incident_id}")
                
            except Exception as e:
                incident_id = incident.get('values', {}).get('Incident Number', 'unknown')
                logger.error(f"Error processing Remedy incident {incident_id}: {str(e)}")
        
        # Process Knowledge Base articles as well
        # This is a placeholder - you'd need to implement KB article retrieval in your Remedy client
        
        logger.info(f"Processed {len(processed_docs)} Remedy incidents")
        return processed_docs
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """
        Extract clean text from HTML content.
        
        Args:
            html_content: HTML content to process
            
        Returns:
            Extracted text
        """
        if not html_content:
            return ""
            
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Process tables
        tables = []
        for table in soup.find_all('table'):
            try:
                rows = []
                for tr in table.find_all('tr'):
                    cols = [td.text.strip() for td in tr.find_all(['td', 'th'])]
                    if cols:
                        rows.append(' | '.join(cols))
                
                if rows:
                    table_text = "TABLE:\n" + '\n'.join(rows)
                    tables.append(table_text)
                    # Replace table with a marker
                    table.replace_with(BeautifulSoup(f"[TABLE_{len(tables)}]", 'html.parser'))
            except Exception as e:
                logger.warning(f"Error processing table: {str(e)}")
        
        # Process code blocks
        code_blocks = []
        for code in soup.find_all(['code', 'pre']):
            try:
                code_text = "CODE:\n" + code.text.strip()
                code_blocks.append(code_text)
                # Replace code with a marker
                code.replace_with(BeautifulSoup(f"[CODE_{len(code_blocks)}]", 'html.parser'))
            except Exception as e:
                logger.warning(f"Error processing code block: {str(e)}")
        
        # Process lists
        lists = []
        for list_elem in soup.find_all(['ul', 'ol']):
            try:
                list_type = 'UNORDERED LIST' if list_elem.name == 'ul' else 'ORDERED LIST'
                list_items = [li.text.strip() for li in list_elem.find_all('li')]
                if list_items:
                    list_text = f"{list_type}:\n" + '\n'.join([f"- {item}" for item in list_items])
                    lists.append(list_text)
                    # Replace list with a marker
                    list_elem.replace_with(BeautifulSoup(f"[LIST_{len(lists)}]", 'html.parser'))
            except Exception as e:
                logger.warning(f"Error processing list: {str(e)}")
        
        # Extract text, preserving some structure
        for element in soup(['script', 'style', 'head', 'title', 'meta', '[document]']):
            element.extract()
        
        # Get main text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up the text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        # Replace markers with actual content
        for i, table_text in enumerate(tables, 1):
            text = text.replace(f"[TABLE_{i}]", f"\n\n{table_text}\n\n")
            
        for i, code_text in enumerate(code_blocks, 1):
            text = text.replace(f"[CODE_{i}]", f"\n\n{code_text}\n\n")
            
        for i, list_text in enumerate(lists, 1):
            text = text.replace(f"[LIST_{i}]", f"\n\n{list_text}\n\n")
        
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _process_attachment(self, attachment_id: str, filename: str) -> str:
        """
        Process a Confluence attachment.
        
        Args:
            attachment_id: ID of the attachment
            filename: Name of the attachment file
            
        Returns:
            Extracted text from the attachment
        """
        try:
            # Download the attachment
            file_path = confluence_client.download_attachment(attachment_id, filename)
            
            if not file_path or not os.path.exists(file_path):
                return ""
                
            # Process based on file type
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext == '.pdf':
                return self._extract_text_from_pdf(file_path)
                
            elif file_ext == '.docx':
                return self._extract_text_from_docx(file_path)
                
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
                    
            elif file_ext in ['.jpg', '.jpeg', '.png']:
                return self._extract_text_from_image(file_path)
                
            return ""
            
        except Exception as e:
            logger.error(f"Error processing attachment {filename}: {str(e)}")
            return ""
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            text = extract_text(file_path)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text
        """
        try:
            doc = docx.Document(file_path)
            text = "\n\n".join([para.text for para in doc.paragraphs if para.text])
            
            # Process tables if present
            for table in doc.tables:
                table_text = []
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        row_text.append(cell.text.strip())
                    if row_text:
                        table_text.append(" | ".join(row_text))
                
                if table_text:
                    text += "\n\nTABLE:\n" + "\n".join(table_text)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            return ""
    
    def _extract_text_from_image(self, file_path: str) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Extracted text
        """
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return ""
    
    def _create_chunks(self, doc: Document) -> None:
        """
        Split a document into chunks for better retrieval.
        
        Args:
            doc: Document to chunk
        """
        if not doc.content:
            return
            
        content = doc.content
        chunk_size = CHUNK_SIZE
        chunk_overlap = CHUNK_OVERLAP
        
        # Split into sentences
        sentences = nltk.sent_tokenize(content)
        
        # Create chunks with overlap
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence.split())
            
            # If adding this sentence would exceed chunk size,
            # save the current chunk and start a new one
            if current_size + sentence_len > chunk_size and current_chunk:
                # Join sentences into a chunk
                chunk_text = " ".join(current_chunk)
                
                # Create a document chunk
                chunk = DocumentChunk(
                    text=chunk_text,
                    doc_id=doc.doc_id,
                    chunk_id=len(chunks),
                    source=doc.source,
                    metadata={
                        "title": doc.title,
                        "url": doc.url,
                        "doc_type": doc.source
                    }
                )
                
                chunks.append(chunk)
                
                # Start a new chunk with overlap
                overlap_size = 0
                overlap_chunk = []
                
                # Add sentences from the end of the previous chunk for overlap
                for s in reversed(current_chunk):
                    s_len = len(s.split())
                    if overlap_size + s_len <= chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += s_len
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_size += sentence_len
        
        # Add the last chunk if it has content
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            
            chunk = DocumentChunk(
                text=chunk_text,
                doc_id=doc.doc_id,
                chunk_id=len(chunks),
                source=doc.source,
                metadata={
                    "title": doc.title,
                    "url": doc.url,
                    "doc_type": doc.source
                }
            )
            
            chunks.append(chunk)
        
        # Store chunks in the document
        doc.chunks = chunks
        
        # Add chunks to global chunk list
        self.chunks.extend(chunks)
        
        logger.debug(f"Created {len(chunks)} chunks for document {doc.doc_id}")
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document object or None if not found
        """
        return self.documents.get(doc_id)
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        """Get all document chunks."""
        return self.chunks
    
    def get_chunks_by_source(self, source: str) -> List[DocumentChunk]:
        """
        Get document chunks from a specific source.
        
        Args:
            source: Source system (e.g., 'confluence', 'remedy')
            
        Returns:
            List of document chunks
        """
        return [chunk for chunk in self.chunks if chunk.source == source]
    
    def save_state(self) -> None:
        """Save processor state to cache."""
        try:
            # Save document metadata (not full content)
            docs_meta = {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()}
            docs_file = os.path.join(self.cache_dir, "documents_meta.json")
            with open(docs_file, 'w') as f:
                json.dump(docs_meta, f)
            
            # Save chunks
            chunks_data = [chunk.to_dict() for chunk in self.chunks]
            chunks_file = os.path.join(self.cache_dir, "chunks.json")
            with open(chunks_file, 'w') as f:
                json.dump(chunks_data, f)
                
            # Save processed sources
            sources_file = os.path.join(self.cache_dir, "processed_sources.json")
            with open(sources_file, 'w') as f:
                json.dump(list(self.processed_sources), f)
                
            logger.info("Saved document processor state to cache")
                
        except Exception as e:
            logger.error(f"Error saving document processor state: {str(e)}")
    
    def load_state(self) -> bool:
        """
        Load processor state from cache.
        
        Returns:
            Boolean indicating success or failure
        """
        try:
            # Load processed sources
            sources_file = os.path.join(self.cache_dir, "processed_sources.json")
            if os.path.exists(sources_file):
                with open(sources_file, 'r') as f:
                    self.processed_sources = set(json.load(f))
            
            # Load chunks
            chunks_file = os.path.join(self.cache_dir, "chunks.json")
            if os.path.exists(chunks_file):
                with open(chunks_file, 'r') as f:
                    chunks_data = json.load(f)
                    self.chunks = [
                        DocumentChunk(
                            text=c['text'],
                            doc_id=c['doc_id'],
                            chunk_id=c['chunk_id'],
                            source=c['source'],
                            metadata=c['metadata']
                        )
                        for c in chunks_data
                    ]
            
            logger.info(f"Loaded {len(self.chunks)} chunks from cache")
            return True
            
        except Exception as e:
            logger.error(f"Error loading document processor state: {str(e)}")
            return False

# Create a singleton instance
document_processor = DocumentProcessor()










app/core/query_processor.py
"""
Query processor module for analyzing and understanding user queries.
"""
import re
import logging
from typing import Dict, List, Any, Tuple, Set, Optional
import string

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("SpaCy model not found. Downloading en_core_web_sm...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class QueryProcessor:
    """Processor for analyzing and enhancing user queries."""
    
    def __init__(self):
        """Initialize the query processor."""
        self.source_keywords = {
            'confluence': {'confluence', 'wiki', 'kb', 'knowledge base', 'documentation', 'docs', 'article', 'page'},
            'remedy': {'remedy', 'ticket', 'incident', 'service desk', 'request', 'helpdesk', 'help desk', 'issue'}
        }
        
        self.intent_patterns = {
            'search': [r'find', r'search', r'look for', r'get', r'retrieve', r'where'],
            'explain': [r'explain', r'how', r'why', r'what is', r'what are', r'describe', r'details'],
            'compare': [r'compare', r'difference', r'versus', r'vs', r'similarities'],
            'troubleshoot': [r'troubleshoot', r'solve', r'fix', r'resolve', r'issue', r'problem'],
            'summary': [r'summarize', r'summary', r'brief', r'overview']
        }
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query to understand intent and extract key information.
        
        Args:
            query: The user's query string
            
        Returns:
            Dictionary with query analysis information
        """
        if not query or not query.strip():
            return {
                'original_query': query,
                'processed_query': '',
                'sources': [],
                'keywords': [],
                'intent': 'unknown',
                'entities': {}
            }
            
        # Clean and normalize query
        processed_query = self._clean_query(query)
        
        # Determine relevant sources
        sources = self._detect_sources(processed_query)
        
        # Extract keywords
        keywords = self._extract_keywords(processed_query)
        
        # Determine query intent
        intent = self._determine_intent(processed_query)
        
        # Extract entities
        entities = self._extract_entities(processed_query)
        
        # Expand query
        expanded_query = self._expand_query(processed_query, keywords)
        
        return {
            'original_query': query,
            'processed_query': processed_query,
            'expanded_query': expanded_query,
            'sources': sources,
            'keywords': keywords,
            'intent': intent,
            'entities': entities
        }
    
    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize a query string.
        
        Args:
            query: The query to clean
            
        Returns:
            Cleaned query string
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove punctuation
        query = query.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        return query
    
    def _detect_sources(self, query: str) -> List[str]:
        """
        Detect which sources the query is targeting.
        
        Args:
            query: The processed query
            
        Returns:
            List of source identifiers (confluence, remedy, or both)
        """
        sources = []
        query_tokens = set(query.split())
        
        # Check for explicit source mentions
        for source, keywords in self.source_keywords.items():
            if any(keyword in query for keyword in keywords):
                sources.append(source)
                
        # If no explicit source, use both
        if not sources:
            sources = ['confluence', 'remedy']
            
        return sources
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from the query.
        
        Args:
            query: The processed query
            
        Returns:
            List of extracted keywords
        """
        # Tokenize
        tokens = nltk.word_tokenize(query)
        
        # Remove stopwords
        tokens = [token for token in tokens if token.lower() not in stop_words]
        
        # Lemmatize
        keywords = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _determine_intent(self, query: str) -> str:
        """
        Determine the intent of the query.
        
        Args:
            query: The processed query
            
        Returns:
            Intent string
        """
        # Check for each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(r'\b' + pattern + r'\b', query, re.IGNORECASE):
                    return intent
        
        # Default to search intent
        return 'search'
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract named entities from the query.
        
        Args:
            query: The processed query
            
        Returns:
            Dictionary of entity types and values
        """
        # Parse with SpaCy
        doc = nlp(query)
        
        # Extract entities
        entities = {}
        for ent in doc.ents:
            entity_type = ent.label_
            entity_text = ent.text
            
            if entity_type not in entities:
                entities[entity_type] = []
                
            entities[entity_type].append(entity_text)
        
        return entities
    
    def _expand_query(self, query: str, keywords: List[str]) -> str:
        """
        Expand the query with synonyms or related terms for better retrieval.
        
        Args:
            query: The processed query
            keywords: Extracted keywords
            
        Returns:
            Expanded query
        """
        # For a simple implementation, just return the original query
        # In a more sophisticated system, this would add synonyms/related terms
        return query
    
    def parse_source_preference(self, query: str) -> Tuple[str, str]:
        """
        Parse query to extract explicitly requested source and actual query.
        
        Args:
            query: The original query
            
        Returns:
            Tuple of (source preference, cleaned query)
        """
        query_lower = query.lower()
        
        # Check for explicit source specifications
        confluence_patterns = [
            r'(^|\s)(?:in|from|on|search)\s+confluence:?\s*(.+)$',
            r'(^|\s)(?:in|from|on|search)\s+wiki:?\s*(.+)$',
            r'(^|\s)(?:in|from|on|search)\s+documentation:?\s*(.+)$',
            r'(^|\s)(?:in|from|on|search)\s+kb:?\s*(.+)$',
        ]
        
        remedy_patterns = [
            r'(^|\s)(?:in|from|on|search)\s+remedy:?\s*(.+)$',
            r'(^|\s)(?:in|from|on|search)\s+tickets?:?\s*(.+)$',
            r'(^|\s)(?:in|from|on|search)\s+incidents?:?\s*(.+)$',
            r'(^|\s)(?:in|from|on|search)\s+helpdesk:?\s*(.+)$',
        ]
        
        # Check for Confluence patterns
        for pattern in confluence_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return "confluence", match.group(2).strip()
        
        # Check for Remedy patterns
        for pattern in remedy_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return "remedy", match.group(2).strip()
        
        # No explicit preference
        return "both", query
    
    def extract_search_query(self, query: str) -> str:
        """
        Extract the actual search query, removing any prefixes or system indicators.
        
        Args:
            query: The original query
            
        Returns:
            Cleaned search query
        """
        # Extract the search part from common query formats
        source_pref, clean_query = self.parse_source_preference(query)
        
        # Remove common prefixes
        prefixes = [
            "find", "search for", "look for", "get", "retrieve",
            "tell me about", "what is", "show me", "i want to know about",
            "can you find", "can you get", "can you tell me about"
        ]
        
        for prefix in prefixes:
            pattern = f"^{prefix}\\s+"
            clean_query = re.sub(pattern, "", clean_query, flags=re.IGNORECASE)
        
        # Remove trailing punctuation
        clean_query = clean_query.rstrip("?.!,;:")
        
        return clean_query

# Create a singleton instance
query_processor = QueryProcessor()






app/core/retriever.py
"""
Retriever module for finding relevant information based on user queries.
"""
import os
import logging
import json
import pickle
import re
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from app.core.document_processor import document_processor, DocumentChunk
from app.config import INDICES_DIR, EMBEDDING_MODEL, TOP_K_RESULTS, SIMILARITY_THRESHOLD

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    
stop_words = set(stopwords.words('english'))

class SearchResult:
    """Class representing a search result with relevant information."""
    
    def __init__(
        self,
        chunk: DocumentChunk,
        score: float,
        source: str = "",
        position: int = 0
    ):
        """
        Initialize a search result.
        
        Args:
            chunk: The document chunk
            score: Relevance score
            source: Source system
            position: Position in results list
        """
        self.chunk = chunk
        self.score = score
        self.source = source or chunk.source
        self.position = position
        self.doc_id = chunk.doc_id
        self.metadata = chunk.metadata
    
    def __repr__(self) -> str:
        return f"SearchResult(doc_id={self.doc_id}, score={self.score:.4f}, source={self.source})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.chunk.text,
            "score": self.score,
            "source": self.source,
            "position": self.position,
            "doc_id": self.doc_id,
            "metadata": self.metadata
        }

class Retriever:
    """Retriever for finding relevant document chunks based on queries."""
    
    def __init__(self):
        """Initialize the retriever with various retrieval methods."""
        self.indices_dir = INDICES_DIR
        os.makedirs(self.indices_dir, exist_ok=True)
        
        # Initialize embedding model if available
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                logger.info(f"Loaded sentence transformer model: {EMBEDDING_MODEL}")
            except Exception as e:
                logger.error(f"Error loading sentence transformer model: {str(e)}")
        
        # TF-IDF vectorizer (fallback if embeddings not available)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # BM25 for lexical search
        self.bm25_index = None
        self.bm25_corpus = []
        
        # FAISS index for vector search
        self.faiss_index = None
        self.faiss_id_mapping = []
        
        # Track if indices are built
        self.indices_built = False
        
        # Default parameters
        self.top_k = TOP_K_RESULTS
        self.similarity_threshold = SIMILARITY_THRESHOLD
    
    def build_indices(self, force_rebuild: bool = False) -> bool:
        """
        Build search indices for efficient retrieval.
        
        Args:
            force_rebuild: Whether to force rebuilding even if indices exist
            
        Returns:
            Boolean indicating success
        """
        # Skip if already built and not forcing rebuild
        if self.indices_built and not force_rebuild:
            return True
            
        # Get all document chunks
        chunks = document_processor.get_all_chunks()
        
        if not chunks:
            logger.warning("No document chunks available to build indices")
            return False
            
        try:
            # Build TF-IDF index
            self._build_tfidf_index(chunks)
            
            # Build BM25 index
            self._build_bm25_index(chunks)
            
            # Build embedding index if available
            if SENTENCE_TRANSFORMERS_AVAILABLE and self.embedding_model:
                self._build_embedding_index(chunks)
                
            self.indices_built = True
            logger.info("Successfully built all search indices")
            return True
            
        except Exception as e:
            logger.error(f"Error building indices: {str(e)}")
            return False
    
    def _build_tfidf_index(self, chunks: List[DocumentChunk]) -> None:
        """
        Build TF-IDF vectorizer and matrix for text search.
        
        Args:
            chunks: List of document chunks
        """
        # Extract text from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Initialize and fit vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )
        
        # Create the TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Save the vectorizer and matrix
        tfidf_dir = os.path.join(self.indices_dir, "tfidf")
        os.makedirs(tfidf_dir, exist_ok=True)
        
        with open(os.path.join(tfidf_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)
            
        with open(os.path.join(tfidf_dir, "matrix.npz"), "wb") as f:
            np.savez_compressed(f, data=self.tfidf_matrix.data, indices=self.tfidf_matrix.indices, 
                         indptr=self.tfidf_matrix.indptr, shape=self.tfidf_matrix.shape)
        
        logger.info(f"Built TF-IDF index with {len(texts)} documents")
    
    def _build_bm25_index(self, chunks: List[DocumentChunk]) -> None:
        """
        Build BM25 index for lexical search.
        
        Args:
            chunks: List of document chunks
        """
        # Prepare corpus by tokenizing document texts
        self.bm25_corpus = []
        
        for chunk in chunks:
            # Tokenize text
            tokens = word_tokenize(chunk.text.lower())
            # Remove stopwords
            tokens = [token for token in tokens if token not in stop_words]
            self.bm25_corpus.append(tokens)
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        
        # Save the index
        bm25_dir = os.path.join(self.indices_dir, "bm25")
        os.makedirs(bm25_dir, exist_ok=True)
        
        with open(os.path.join(bm25_dir, "index.pkl"), "wb") as f:
            pickle.dump(self.bm25_index, f)
            
        with open(os.path.join(bm25_dir, "corpus.pkl"), "wb") as f:
            pickle.dump(self.bm25_corpus, f)
        
        logger.info(f"Built BM25 index with {len(self.bm25_corpus)} documents")
    
    def _build_embedding_index(self, chunks: List[DocumentChunk]) -> None:
        """
        Build vector index for semantic search using embeddings.
        
        Args:
            chunks: List of document chunks
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not self.embedding_model:
            logger.warning("Sentence transformers not available, skipping embedding index")
            return
            
        # Extract text from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index if available
        if FAISS_AVAILABLE:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Get embedding dimension
            dimension = embeddings.shape[1]
            
            # Create index
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
            self.faiss_index.add(embeddings)
            
            # Save mapping from FAISS IDs to document chunks
            self.faiss_id_mapping = chunks
            
            # Save the index
            faiss_dir = os.path.join(self.indices_dir, "faiss")
            os.makedirs(faiss_dir, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.faiss_index, os.path.join(faiss_dir, "index.bin"))
            
            # Save ID mapping (just save the chunk IDs, not the full chunks)
            mapping_data = [(i, chunk.doc_id, chunk.chunk_id, chunk.source) for i, chunk in enumerate(chunks)]
            with open(os.path.join(faiss_dir, "id_mapping.json"), "w") as f:
                json.dump(mapping_data, f)
                
            logger.info(f"Built FAISS embedding index with {len(texts)} documents")
        else:
            # Store raw embeddings if FAISS is not available
            self.embeddings = embeddings
            self.embedding_chunks = chunks
            
            # Save the embeddings
            embed_dir = os.path.join(self.indices_dir, "embeddings")
            os.makedirs(embed_dir, exist_ok=True)
            
            np.save(os.path.join(embed_dir, "embeddings.npy"), embeddings)
            
            # Save mapping from embedding IDs to document chunks
            mapping_data = [(i, chunk.doc_id, chunk.chunk_id, chunk.source) for i, chunk in enumerate(chunks)]
            with open(os.path.join(embed_dir, "id_mapping.json"), "w") as f:
                json.dump(mapping_data, f)
                
            logger.info(f"Built basic embedding index with {len(texts)} documents")
    
    def load_indices(self) -> bool:
        """
        Load search indices from disk.
        
        Returns:
            Boolean indicating success
        """
        try:
            # Load TF-IDF index
            tfidf_dir = os.path.join(self.indices_dir, "tfidf")
            vectorizer_path = os.path.join(tfidf_dir, "vectorizer.pkl")
            matrix_path = os.path.join(tfidf_dir, "matrix.npz")
            
            if os.path.exists(vectorizer_path) and os.path.exists(matrix_path):
                with open(vectorizer_path, "rb") as f:
                    self.tfidf_vectorizer = pickle.load(f)
                
                with open(matrix_path, "rb") as f:
                    loader = np.load(f)
                    self.tfidf_matrix = scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
                
                logger.info("Loaded TF-IDF index")
            
            # Load BM25 index
            bm25_dir = os.path.join(self.indices_dir, "bm25")
            bm25_path = os.path.join(bm25_dir, "index.pkl")
            corpus_path = os.path.join(bm25_dir, "corpus.pkl")
            
            if os.path.exists(bm25_path) and os.path.exists(corpus_path):
                with open(bm25_path, "rb") as f:
                    self.bm25_index = pickle.load(f)
                
                with open(corpus_path, "rb") as f:
                    self.bm25_corpus = pickle.load(f)
                
                logger.info("Loaded BM25 index")
            
            # Load embedding index if available
            if SENTENCE_TRANSFORMERS_AVAILABLE and self.embedding_model:
                if FAISS_AVAILABLE:
                    # Load FAISS index
                    faiss_dir = os.path.join(self.indices_dir, "faiss")
                    index_path = os.path.join(faiss_dir, "index.bin")
                    mapping_path = os.path.join(faiss_dir, "id_mapping.json")
                    
                    if os.path.exists(index_path) and os.path.exists(mapping_path):
                        self.faiss_index = faiss.read_index(index_path)
                        
                        # Load mapping data
                        with open(mapping_path, "r") as f:
                            mapping_data = json.load(f)
                        
                        # Reconstruct chunk objects from the mapping
                        self.faiss_id_mapping = []
                        chunks = document_processor.get_all_chunks()
                        chunk_dict = {(chunk.doc_id, chunk.chunk_id): chunk for chunk in chunks}
                        
                        for _, doc_id, chunk_id, _ in mapping_data:
                            if (doc_id, chunk_id) in chunk_dict:
                                self.faiss_id_mapping.append(chunk_dict[(doc_id, chunk_id)])
                        
                        logger.info("Loaded FAISS embedding index")
                else:
                    # Load raw embeddings
                    embed_dir = os.path.join(self.indices_dir, "embeddings")
                    embed_path = os.path.join(embed_dir, "embeddings.npy")
                    mapping_path = os.path.join(embed_dir, "id_mapping.json")
                    
                    if os.path.exists(embed_path) and os.path.exists(mapping_path):
                        self.embeddings = np.load(embed_path)
                        
                        # Load mapping data
                        with open(mapping_path, "r") as f:
                            mapping_data = json.load(f)
                        
                        # Reconstruct chunk objects from the mapping
                        self.embedding_chunks = []
                        chunks = document_processor.get_all_chunks()
                        chunk_dict = {(chunk.doc_id, chunk.chunk_id): chunk for chunk in chunks}
                        
                        for _, doc_id, chunk_id, _ in mapping_data:
                            if (doc_id, chunk_id) in chunk_dict:
                                self.embedding_chunks.append(chunk_dict[(doc_id, chunk_id)])
                        
                        logger.info("Loaded basic embedding index")
            
            self.indices_built = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading indices: {str(e)}")
            return False
    
    def search(self, query: str, 
              sources: List[str] = None, 
              top_k: int = None, 
              threshold: float = None) -> List[SearchResult]:
        """
        Search for relevant document chunks based on a query.
        
        Args:
            query: Search query
            sources: List of sources to search (default: all)
            top_k: Number of results to return (default: from config)
            threshold: Similarity threshold (default: from config)
            
        Returns:
            List of SearchResult objects
        """
        if not query:
            return []
            
        # Use default parameters if not specified
        top_k = top_k or self.top_k
        threshold = threshold or self.similarity_threshold
        
        # Build indices if needed
        if not self.indices_built:
            if not self.load_indices():
                self.build_indices()
        
        # Get all chunks
        all_chunks = document_processor.get_all_chunks()
        
        if not all_chunks:
            logger.warning("No document chunks available for search")
            return []
            
        # Filter by source if specified
        if sources:
            chunks = [chunk for chunk in all_chunks if chunk.source in sources]
        else:
            chunks = all_chunks
            
        if not chunks:
            logger.warning(f"No chunks available for specified sources: {sources}")
            return []
        
        # Perform hybrid search
        results = self._hybrid_search(query, chunks, top_k, threshold)
        
        return results
    
    def _hybrid_search(self, query: str, chunks: List[DocumentChunk], 
                     top_k: int, threshold: float) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and lexical methods.
        
        Args:
            query: Search query
            chunks: Document chunks to search
            top_k: Number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of SearchResult objects
        """
        # Perform searches with different methods
        semantic_results = self._semantic_search(query, chunks, top_k * 2)
        bm25_results = self._bm25_search(query, chunks, top_k * 2)
        
        # Combine results
        combined_results = self._combine_search_results(semantic_results, bm25_results, top_k)
        
        # Filter by threshold
        filtered_results = [r for r in combined_results if r.score >= threshold]
        
        return filtered_results
    
    def _semantic_search(self, query: str, chunks: List[DocumentChunk], top_k: int) -> List[SearchResult]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            chunks: Document chunks to search
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        # Use embeddings if available
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.embedding_model:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            if FAISS_AVAILABLE and self.faiss_index:
                # Normalize query embedding
                query_embedding_normalized = query_embedding.copy()
                faiss.normalize_L2(np.array([query_embedding_normalized]))
                
                # Search in FAISS index
                scores, indices = self.faiss_index.search(np.array([query_embedding_normalized]), top_k)
                
                # Convert to search results
                results = []
                for i in range(len(indices[0])):
                    idx = indices[0][i]
                    score = scores[0][i]
                    
                    if idx < 0 or idx >= len(self.faiss_id_mapping):
                        continue
                        
                    chunk = self.faiss_id_mapping[idx]
                    results.append(SearchResult(chunk, float(score), chunk.source, i))
                
                return results
            elif hasattr(self, 'embeddings') and hasattr(self, 'embedding_chunks'):
                # Calculate similarities
                similarities = np.dot(self.embeddings, query_embedding) / (
                    np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
                
                # Get top-k indices
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                # Convert to search results
                results = []
                for i, idx in enumerate(top_indices):
                    chunk = self.embedding_chunks[idx]
                    score = similarities[idx]
                    results.append(SearchResult(chunk, float(score), chunk.source, i))
                
                return results
        
        # Fall back to TF-IDF if embeddings not available
        return self._tfidf_search(query, chunks, top_k)
    
    def _tfidf_search(self, query: str, chunks: List[DocumentChunk], top_k: int) -> List[SearchResult]:
        """
        Perform TF-IDF based search.
        
        Args:
            query: Search query
            chunks: Document chunks to search
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        # Initialize vectorizer if not already done
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            # Extract text from chunks
            texts = [chunk.text for chunk in chunks]
            
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                max_features=10000,
                ngram_range=(1, 2)
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Vectorize query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Convert to search results
        results = []
        for i, idx in enumerate(top_indices):
            chunk = chunks[idx]
            score = similarities[idx]
            results.append(SearchResult(chunk, float(score), chunk.source, i))
        
        return results
    
    def _bm25_search(self, query: str, chunks: List[DocumentChunk], top_k: int) -> List[SearchResult]:
        """
        Perform BM25 lexical search.
        
        Args:
            query: Search query
            chunks: Document chunks to search
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        # Initialize BM25 index if not already done
        if self.bm25_index is None:
            # Prepare corpus
            self.bm25_corpus = []
            for chunk in chunks:
                tokens = word_tokenize(chunk.text.lower())
                tokens = [token for token in tokens if token not in stop_words]
                self.bm25_corpus.append(tokens)
            
            # Create BM25 index
            self.bm25_index = BM25Okapi(self.bm25_corpus)
        
        # Tokenize query
        query_tokens = word_tokenize(query.lower())
        query_tokens = [token for token in query_tokens if token not in stop_words]
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Convert to search results
        results = []
        for i, idx in enumerate(top_indices):
            chunk = chunks[idx]
            score = float(scores[idx]) / 10.0  # Normalize BM25 scores to 0-1 range (approximately)
            if score > 1.0:  # Ensure score is in 0-1 range
                score = 1.0
            results.append(SearchResult(chunk, score, chunk.source, i))
        
        return results
    
    def _combine_search_results(self, semantic_results: List[SearchResult], 
                              lexical_results: List[SearchResult], 
                              top_k: int) -> List[SearchResult]:
        """
        Combine results from different search methods.
        
        Args:
            semantic_results: Results from semantic search
            lexical_results: Results from lexical search
            top_k: Number of results to return
            
        Returns:
            Combined and ranked search results
        """
        # Create a map for quick access
        combined_map = {}
        
        # Add semantic results with a weight of 0.7
        for i, result in enumerate(semantic_results):
            key = (result.chunk.doc_id, result.chunk.chunk_id)
            combined_map[key] = {
                'chunk': result.chunk,
                'semantic_score': result.score * 0.7,
                'lexical_score': 0.0,
                'position': min(i, 1000),  # Lower position is better
                'source': result.source
            }
        
        # Add lexical results with a weight of 0.3
        for i, result in enumerate(lexical_results):
            key = (result.chunk.doc_id, result.chunk.chunk_id)
            if key in combined_map:
                combined_map[key]['lexical_score'] = result.score * 0.3
                combined_map[key]['position'] = min(combined_map[key]['position'], i)
            else:
                combined_map[key] = {
                    'chunk': result.chunk,
                    'semantic_score': 0.0,
                    'lexical_score': result.score * 0.3,
                    'position': min(i, 1000),
                    'source': result.source
                }
        
        # Calculate combined scores
        combined_results = []
        for key, data in combined_map.items():
            # Combine scores (semantic + lexical)
            combined_score = data['semantic_score'] + data['lexical_score']
            
            # Apply position boost (earlier positions get a small boost)
            position_boost = max(0.0, (1000 - data['position']) / 10000)
            final_score = min(1.0, combined_score + position_boost)
            
            combined_results.append(
                SearchResult(
                    chunk=data['chunk'],
                    score=final_score,
                    source=data['source'],
                    position=data['position']
                )
            )
        
        # Sort by score (descending)
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top-k results
        return combined_results[:top_k]
    
    def search_by_source(self, query: str, source: str, top_k: int = None) -> List[SearchResult]:
        """
        Search within a specific source.
        
        Args:
            query: Search query
            source: Source to search (confluence or remedy)
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        return self.search(query, sources=[source], top_k=top_k)
    
    def suggest_related_queries(self, query: str, results: List[SearchResult], max_suggestions: int = 3) -> List[str]:
        """
        Generate related query suggestions based on search results.
        
        Args:
            query: Original query
            results: Search results
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of suggested queries
        """
        if not results:
            return []
            
        # Extract key phrases from result text
        all_text = " ".join([result.chunk.text for result in results[:5]])
        
        # Simple approach: extract noun phrases using regex
        noun_phrase_pattern = r'\b(?:[A-Z][a-z]+\s)+[A-Z][a-z]+\b|\b[A-Z][a-z]+\b'
        noun_phrases = re.findall(noun_phrase_pattern, all_text)
        
        # Filter out phrases that are already in the query
        query_lower = query.lower()
        filtered_phrases = [phrase for phrase in noun_phrases 
                           if phrase.lower() not in query_lower and len(phrase) > 3]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_phrases = [phrase for phrase in filtered_phrases 
                         if not (phrase.lower() in seen or seen.add(phrase.lower()))]
        
        # Limit number of suggestions
        suggestions = unique_phrases[:max_suggestions]
        
        # Create suggestions by combining with original query
        query_suggestions = [f"{query} {suggestion}" for suggestion in suggestions]
        
        return query_suggestions

# Create a singleton instance
retriever = Retriever()





app/core/summarizer.py

"""
Summarizer module for generating concise summaries from retrieved content.
"""
import re
import logging
import string
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.core.retriever import SearchResult

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

class SummaryResult:
    """Class representing a generated summary with metadata."""
    
    def __init__(
        self,
        text: str,
        source: str,
        confidence: float,
        search_results: List[SearchResult]
    ):
        """
        Initialize a summary result.
        
        Args:
            text: Summary text
            source: Source system (confluence, remedy, or combined)
            confidence: Confidence score
            search_results: Search results used to generate the summary
        """
        self.text = text
        self.source = source
        self.confidence = confidence
        self.search_results = search_results
    
    def __repr__(self) -> str:
        return f"SummaryResult(source={self.source}, confidence={self.confidence:.2f}, text={self.text[:50]}...)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "source": self.source,
            "confidence": self.confidence,
            "results": [result.to_dict() for result in self.search_results]
        }

class Summarizer:
    """Summarizer for generating concise responses from retrieved content."""
    
    def __init__(self):
        """Initialize the summarizer."""
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def generate_summary(self, query: str, search_results: List[SearchResult], 
                        max_length: int = 500, source: str = "combined") -> SummaryResult:
        """
        Generate a concise summary from search results.
        
        Args:
            query: Original search query
            search_results: Retrieved search results
            max_length: Maximum length of summary in characters
            source: Source of the search results (confluence, remedy, or combined)
            
        Returns:
            SummaryResult object with the generated summary
        """
        if not search_results:
            return SummaryResult(
                text="No relevant information found.",
                source=source,
                confidence=0.0,
                search_results=[]
            )
        
        # Extract text from search results
        docs = [result.chunk.text for result in search_results]
        
        # Generate summary based on number of results
        if len(docs) >= 3:
            # Use extractive summarization for multiple documents
            summary_text = self._extractive_summarization(query, docs, max_length)
            confidence = self._calculate_confidence(search_results)
        else:
            # For 1-2 results, use the top result with light processing
            summary_text = self._process_single_result(docs[0], max_length)
            confidence = search_results[0].score if search_results else 0.5
        
        # Apply post-processing
        summary_text = self._post_process_summary(summary_text, query)
        
        return SummaryResult(
            text=summary_text,
            source=source,
            confidence=confidence,
            search_results=search_results
        )
    
    def generate_source_summaries(self, query: str, 
                                confluence_results: List[SearchResult],
                                remedy_results: List[SearchResult],
                                max_length: int = 300) -> Dict[str, SummaryResult]:
        """
        Generate separate summaries for each source.
        
        Args:
            query: Original search query
            confluence_results: Search results from Confluence
            remedy_results: Search results from Remedy
            max_length: Maximum length of each summary
            
        Returns:
            Dictionary with source as key and SummaryResult as value
        """
        summaries = {}
        
        # Generate summary for Confluence results
        if confluence_results:
            confluence_summary = self.generate_summary(
                query, 
                confluence_results, 
                max_length, 
                source="confluence"
            )
            summaries["confluence"] = confluence_summary
        else:
            summaries["confluence"] = SummaryResult(
                text="No relevant information found in Confluence.",
                source="confluence",
                confidence=0.0,
                search_results=[]
            )
        
        # Generate summary for Remedy results
        if remedy_results:
            remedy_summary = self.generate_summary(
                query, 
                remedy_results, 
                max_length, 
                source="remedy"
            )
            summaries["remedy"] = remedy_summary
        else:
            summaries["remedy"] = SummaryResult(
                text="No relevant information found in Remedy.",
                source="remedy",
                confidence=0.0,
                search_results=[]
            )
        
        # Generate combined summary if both sources have results
        if confluence_results and remedy_results:
            # Combine results, sorting by score
            combined_results = confluence_results + remedy_results
            combined_results.sort(key=lambda x: x.score, reverse=True)
            
            combined_summary = self.generate_summary(
                query, 
                combined_results[:5],  # Use top 5 results from both sources
                max_length, 
                source="combined"
            )
            summaries["combined"] = combined_summary
        
        return summaries
    
    def _extractive_summarization(self, query: str, docs: List[str], max_length: int) -> str:
        """
        Generate an extractive summary from multiple documents.
        
        Args:
            query: Original search query
            docs: List of document texts
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        # Split documents into sentences
        all_sentences = []
        for doc in docs:
            sentences = sent_tokenize(doc)
            all_sentences.extend(sentences)
        
        # Remove duplicates and very short sentences
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in all_sentences:
            sentence_cleaned = re.sub(r'\s+', ' ', sentence).strip().lower()
            
            # Skip very short sentences
            if len(sentence_cleaned) < 20 or len(sentence.split()) < 5:
                continue
                
            # Skip duplicate sentences (after cleaning)
            if sentence_cleaned in seen_sentences:
                continue
                
            seen_sentences.add(sentence_cleaned)
            unique_sentences.append(sentence)
        
        if not unique_sentences:
            return "No relevant information could be extracted."
        
        # Score sentences using multiple methods
        sentence_scores = {}
        
        # 1. TF-IDF similarity to query
        tfidf_scores = self._tfidf_similarity(query, unique_sentences)
        
        # 2. Position-based score (earlier sentences are more important)
        position_scores = {sentence: max(0, 1.0 - 0.05 * i) 
                         for i, sentence in enumerate(unique_sentences)}
        
        # 3. Length-based score (prefer medium-length sentences)
        length_scores = {}
        for sentence in unique_sentences:
            words = len(sentence.split())
            if words < 5:
                length_scores[sentence] = 0.3
            elif words > 30:
                length_scores[sentence] = 0.4
            else:
                length_scores[sentence] = 0.8
        
        # Combine scores: TF-IDF (60%) + Position (20%) + Length (20%)
        for i, sentence in enumerate(unique_sentences):
            sentence_scores[sentence] = (
                tfidf_scores.get(sentence, 0) * 0.6 +
                position_scores.get(sentence, 0) * 0.2 +
                length_scores.get(sentence, 0) * 0.2
            )
        
        # Rank sentences by score
        ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top sentences up to max length
        selected_sentences = []
        current_length = 0
        
        for sentence, _ in ranked_sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= max_length:
                selected_sentences.append(sentence)
                current_length += sentence_length
            else:
                break
        
        # Restore original order to improve coherence
        ordered_sentences = [s for s in unique_sentences if s in selected_sentences]
        
        if not ordered_sentences:
            # Fallback if no sentences were selected
            return ranked_sentences[0][0] if ranked_sentences else "No relevant information could be extracted."
        
        # Join sentences into a coherent summary
        summary = ' '.join(ordered_sentences)
        
        # Truncate to max_length if needed
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(' ', 1)[0] + '...'
        
        return summary
    
    def _process_single_result(self, text: str, max_length: int) -> str:
        """
        Process a single result for summarization.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Processed summary
        """
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Remove very short sentences and duplicates
        filtered_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence_clean = re.sub(r'\s+', ' ', sentence).strip().lower()
            
            if len(sentence_clean) < 10 or sentence_clean in seen:
                continue
                
            seen.add(sentence_clean)
            filtered_sentences.append(sentence)
        
        # Take sentences from the beginning up to max_length
        summary = ""
        for sentence in filtered_sentences:
            if len(summary) + len(sentence) + 1 <= max_length:
                summary += sentence + " "
            else:
                break
        
        return summary.strip()
    
    def _tfidf_similarity(self, query: str, sentences: List[str]) -> Dict[str, float]:
        """
        Calculate TF-IDF similarity between query and sentences.
        
        Args:
            query: Query string
            sentences: List of sentences
            
        Returns:
            Dictionary mapping sentences to similarity scores
        """
        if not sentences:
            return {}
            
        # Prepare documents (query + sentences)
        documents = [query] + sentences
        
        try:
            # Compute TF-IDF matrix
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            
            # Calculate similarity between query and each sentence
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            
            # Create a dictionary of sentence to similarity score
            sentence_similarities = {
                sentence: float(similarity)
                for sentence, similarity in zip(sentences, similarities[0])
            }
            
            return sentence_similarities
            
        except Exception as e:
            logger.error(f"Error calculating TF-IDF similarity: {str(e)}")
            # Fallback to simple keyword matching
            return self._keyword_similarity(query, sentences)
    
    def _keyword_similarity(self, query: str, sentences: List[str]) -> Dict[str, float]:
        """
        Calculate keyword-based similarity as a fallback method.
        
        Args:
            query: Query string
            sentences: List of sentences
            
        Returns:
            Dictionary mapping sentences to similarity scores
        """
        # Tokenize and clean query
        query_tokens = set(word.lower() for word in word_tokenize(query) 
                        if word.lower() not in stop_words and word.lower() not in string.punctuation)
        
        similarities = {}
        
        for sentence in sentences:
            # Tokenize and clean sentence
            sentence_tokens = set(word.lower() for word in word_tokenize(sentence) 
                               if word.lower() not in stop_words and word.lower() not in string.punctuation)
            
            # Calculate Jaccard similarity
            intersection = query_tokens.intersection(sentence_tokens)
            union = query_tokens.union(sentence_tokens)
            
            if union:
                similarity = len(intersection) / len(union)
            else:
                similarity = 0.0
                
            similarities[sentence] = similarity
        
        return similarities
    
    def _calculate_confidence(self, search_results: List[SearchResult]) -> float:
        """
        Calculate overall confidence score for the summary.
        
        Args:
            search_results: List of search results
            
        Returns:
            Confidence score (0-1)
        """
        if not search_results:
            return 0.0
            
        # Use weighted average of top 3 result scores
        top_results = search_results[:min(3, len(search_results))]
        weights = [0.6, 0.3, 0.1][:len(top_results)]
        
        confidence = sum(result.score * weight for result, weight in zip(top_results, weights))
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, confidence))
    
    def _post_process_summary(self, summary: str, query: str) -> str:
        """
        Apply post-processing to improve summary quality.
        
        Args:
            summary: Generated summary text
            query: Original query
            
        Returns:
            Post-processed summary
        """
        if not summary:
            return "No relevant information found."
            
        # Clean up whitespace
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        # Add query-focused introduction if summary is longer than 100 characters
        if len(summary) > 100:
            # Check if query is a question
            is_question = query.endswith('?') or query.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who', 'can', 'does', 'do'))
            
            if is_question:
                # Try to extract main topic from query
                query_clean = re.sub(r'[^\w\s]', '', query.lower())
                query_words = [w for w in query_clean.split() if w not in stop_words]
                
                if query_words:
                    topic = query_words[0].capitalize()  # Just use first non-stopword
                    summary = f"Regarding your question about {topic}: {summary}"
            else:
                summary = f"Here's information about your query: {summary}"
        
        return summary
    
    def format_dual_summary(self, confluence_summary: SummaryResult, remedy_summary: SummaryResult) -> str:
        """
        Format a combined response with separate summaries from both sources.
        
        Args:
            confluence_summary: Summary from Confluence
            remedy_summary: Summary from Remedy
            
        Returns:
            Formatted combined summary text
        """
        combined = []
        
        # Add Confluence section if there's content
        if confluence_summary and confluence_summary.text and confluence_summary.text != "No relevant information found in Confluence.":
            combined.append("## Confluence Knowledge Base\n\n" + confluence_summary.text)
        
        # Add Remedy section if there's content
        if remedy_summary and remedy_summary.text and remedy_summary.text != "No relevant information found in Remedy.":
            combined.append("## Remedy Ticket System\n\n" + remedy_summary.text)
        
        # If both are empty, return a not found message
        if not combined:
            return "No relevant information found in either system."
            
        return "\n\n".join(combined)
    
    def estimate_accuracy(self, summary: SummaryResult) -> float:
        """
        Estimate the factual accuracy of a generated summary.
        
        Args:
            summary: Generated summary
            
        Returns:
            Estimated accuracy score (0-1)
        """
        # For now, use the summary confidence as a proxy for accuracy
        # In a real system, you might implement more sophisticated methods
        return summary.confidence

# Create a singleton instance
summarizer = Summarizer()







app/utils/text_utils.py
"""
Utility functions for text processing.
"""
import re
import string
import logging
from typing import List, Dict, Set, Tuple, Optional
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Initialize resources
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Replace newlines and tabs with spaces
    text = re.sub(r'[\n\t\r]+', ' ', text)
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract important keywords from text.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of extracted keywords
    """
    if not text:
        return []
        
    # Tokenize and clean
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords, punctuation, and short words
    tokens = [token for token in tokens if token not in STOPWORDS
              and token not in string.punctuation
              and len(token) > 2]
    
    # Count token frequencies
    token_counts = {}
    for token in tokens:
        lemma = LEMMATIZER.lemmatize(token)
        token_counts[lemma] = token_counts.get(lemma, 0) + 1
    
    # Sort by frequency (descending)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Extract top keywords
    keywords = [token for token, _ in sorted_tokens[:max_keywords]]
    
    return keywords

def extract_sentences(text: str, max_sentences: int = None) -> List[str]:
    """
    Extract sentences from text.
    
    Args:
        text: Input text
        max_sentences: Maximum number of sentences to extract
        
    Returns:
        List of extracted sentences
    """
    if not text:
        return []
        
    # Split into sentences
    sentences = sent_tokenize(text)
    
    # Clean sentences
    sentences = [re.sub(r'\s+', ' ', s).strip() for s in sentences]
    
    # Filter out very short sentences
    sentences = [s for s in sentences if len(s) > 10]
    
    # Limit number of sentences if specified
    if max_sentences and len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    
    return sentences

def detect_language(text: str) -> str:
    """
    Detect the language of a text (simple implementation).
    
    Args:
        text: Input text
        
    Returns:
        ISO language code (defaults to 'en')
    """
    # This is a simplified implementation that only checks for English
    # In a real system, you would use a proper language detection library like langdetect
    return 'en'

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text (simple implementation).
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of entity types and values
    """
    entities = {
        'ORG': [],    # Organizations
        'PERSON': [], # People
        'LOC': []     # Locations
    }
    
    # In a real system, you would use a proper NER library like SpaCy
    # For now, use basic pattern matching for demonstration
    
    # Match potential organizations
    org_pattern = r'\b([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*?(?:[ ](?:Inc|Corp|Ltd|LLC|GmbH|Co|Company)))\b'
    matches = re.findall(org_pattern, text)
    entities['ORG'].extend(matches)
    
    # Match potential people (simplified)
    person_pattern = r'\b([A-Z][a-z]+(?:[ ][A-Z][a-z]+)+)\b'
    matches = re.findall(person_pattern, text)
    entities['PERSON'].extend(matches)
    
    # Remove duplicates while preserving order
    for entity_type in entities:
        unique_entities = []
        seen = set()
        for entity in entities[entity_type]:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        entities[entity_type] = unique_entities
    
    return entities

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate the similarity between two texts (simple implementation).
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    if not text1 or not text2:
        return 0.0
        
    # Convert to sets of tokens for Jaccard similarity
    tokens1 = set(word_tokenize(text1.lower()))
    tokens2 = set(word_tokenize(text2.lower()))
    
    # Remove stopwords and punctuation
    tokens1 = {token for token in tokens1 if token not in STOPWORDS and token not in string.punctuation}
    tokens2 = {token for token in tokens2 if token not in STOPWORDS and token not in string.punctuation}
    
    # Calculate Jaccard similarity
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    if not union:
        return 0.0
        
    return len(intersection) / len(union)

def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Input text
        max_length: Maximum length in characters
        add_ellipsis: Whether to add ellipsis to truncated text
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
        
    # Truncate at word boundary
    truncated = text[:max_length].rsplit(' ', 1)[0]
    
    if add_ellipsis:
        truncated += '...'
        
    return truncated

def extract_bullet_points(text: str) -> List[str]:
    """
    Extract bullet points from text.
    
    Args:
        text: Input text
        
    Returns:
        List of bullet points
    """
    if not text:
        return []
        
    # Look for common bullet point patterns
    patterns = [
        r'(?m)^[ \t]*•[ \t]*(.*?)(?=\n|$)',              # • bullet
        r'(?m)^[ \t]*\*[ \t]*(.*?)(?=\n|$)',             # * bullet
        r'(?m)^[ \t]*-[ \t]*(.*?)(?=\n|$)',              # - bullet
        r'(?m)^[ \t]*\d+\.[ \t]*(.*?)(?=\n|$)',          # 1. numbered
        r'(?m)^[ \t]*\(?\d+\)?[ \t]*(.*?)(?=\n|$)'       # (1) numbered
    ]
    
    bullet_points = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        bullet_points.extend([match.strip() for match in matches if match.strip()])
    
    return bullet_points

def format_as_html(text: str) -> str:
    """
    Format plain text as HTML with basic formatting.
    
    Args:
        text: Input text
        
    Returns:
        HTML-formatted text
    """
    if not text:
        return ""
        
    # Escape HTML special characters
    html = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # Convert newlines to <br>
    html = html.replace("\n", "<br>")
    
    # Bold text between asterisks
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
    
    # Simple header detection
    html = re.sub(r'(?m)^## (.*?)$', r'<h2>\1</h2>', html)
    html = re.sub(r'(?m)^# (.*?)$', r'<h1>\1</h1>', html)
    
    return html

def has_actionable_content(text: str) -> bool:
    """
    Check if text contains actionable content (instructions, steps, etc.).
    
    Args:
        text: Input text
        
    Returns:
        Boolean indicating presence of actionable content
    """
    # Look for patterns indicating actionable content
    action_patterns = [
        r'\b(?:step|steps|procedure|instructions)\b',
        r'\b(?:follow|click|press|select|choose|enter)\b',
        r'\b(?:must|should|need to|required)\b',
        r'(?:\d+\.\s+\w+)',  # Numbered steps
        r'(?:•|\*|-)\s+\w+'  # Bullet points
    ]
    
    for pattern in action_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
            
    return False

def is_question(text: str) -> bool:
    """
    Check if text is a question.
    
    Args:
        text: Input text
        
    Returns:
        Boolean indicating if text is a question
    """
    # Strip whitespace
    text = text.strip()
    
    # Check if text ends with question mark
    if text.endswith('?'):
        return True
        
    # Check for question words at beginning
    question_starters = [
        r'^(?:what|where|when|who|whom|whose|which|why|how)\b',
        r'^(?:are|is|was|were|will|do|does|did|have|has|had|can|could|should|would|may|might)\b'
    ]
    
    for pattern in question_starters:
        if re.search(pattern, text, re.IGNORECASE):
            return True
            
    return False

def extract_code_blocks(text: str) -> List[str]:
    """
    Extract code blocks from text (markdown-style).
    
    Args:
        text: Input text
        
    Returns:
        List of extracted code blocks
    """
    # Match markdown code blocks
    pattern = r'```(?:\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Match code blocks with indentation
    indent_pattern = r'(?m)^( {4}|\t)(.+)$'
    indent_code = []
    
    current_block = []
    in_block = False
    
    for line in text.split('\n'):
        indent_match = re.match(indent_pattern, line)
        if indent_match:
            current_block.append(indent_match.group(2))
            in_block = True
        elif in_block and line.strip() == '':
            # Empty line might still be part of the block
            current_block.append('')
        elif in_block:
            # End of block
            if current_block:
                indent_code.append('\n'.join(current_block))
                current_block = []
                in_block = False
    
    # Add the last block if exists
    if current_block:
        indent_code.append('\n'.join(current_block))
    
    # Combine all code blocks
    code_blocks = matches + indent_code
    
    return code_blocks









    app/utils/html_utils.py
    """
Utility functions for HTML processing.
"""
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import unicodedata
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_html(html: str) -> str:
    """
    Extract clean text from HTML content.
    
    Args:
        html: HTML content
        
    Returns:
        Extracted text
    """
    if not html:
        return ""
        
    # Parse HTML
    soup = BeautifulSoup(html, 'lxml')
    
    # Remove script, style, head tags
    for element in soup(['script', 'style', 'head', 'title', 'meta', '[document]']):
        element.extract()
    
    # Get text
    text = soup.get_text(separator=' ', strip=True)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_tables_from_html(html: str) -> List[pd.DataFrame]:
    """
    Extract tables from HTML content.
    
    Args:
        html: HTML content
        
    Returns:
        List of pandas DataFrames
    """
    if not html:
        return []
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'lxml')
        
        # Find all tables
        tables = []
        
        for table_elem in soup.find_all('table'):
            # Extract rows
            rows = []
            headers = []
            
            # Check for header row first
            thead = table_elem.find('thead')
            if thead:
                header_cells = [cell.get_text(strip=True) for cell in thead.find_all(['th', 'td'])]
                if header_cells:
                    headers = header_cells
            
            # If no headers found in thead, check first row
            if not headers:
                first_row = table_elem.find('tr')
                if first_row:
                    header_cells = [cell.get_text(strip=True) for cell in first_row.find_all(['th', 'td'])]
                    if all(cell != '' for cell in header_cells):
                        headers = header_cells
            
            # Extract data rows
            for row in table_elem.find_all('tr'):
                # Skip if this is the header row we already processed
                if headers and row == table_elem.find('tr') and not thead:
                    continue
                    
                # Get cell data
                cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                
                if cells:
                    rows.append(cells)
            
            # Create DataFrame
            if rows:
                if headers and len(headers) == len(rows[0]):
                    df = pd.DataFrame(rows, columns=headers)
                else:
                    df = pd.DataFrame(rows)
                    
                tables.append(df)
        
        return tables
        
    except Exception as e:
        logger.error(f"Error extracting tables from HTML: {str(e)}")
        return []

def extract_lists_from_html(html: str) -> List[Dict[str, Any]]:
    """
    Extract lists from HTML content.
    
    Args:
        html: HTML content
        
    Returns:
        List of extracted lists with metadata
    """
    if not html:
        return []
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'lxml')
        
        # Find all lists
        lists = []
        
        # Extract unordered lists
        for ul in soup.find_all('ul'):
            items = [li.get_text(strip=True) for li in ul.find_all('li')]
            if items:
                lists.append({
                    'type': 'unordered',
                    'items': items
                })
        
        # Extract ordered lists
        for ol in soup.find_all('ol'):
            items = [li.get_text(strip=True) for li in ol.find_all('li')]
            if items:
                lists.append({
                    'type': 'ordered',
                    'items': items
                })
        
        # Extract definition lists
        for dl in soup.find_all('dl'):
            items = []
            for dt, dd in zip(dl.find_all('dt'), dl.find_all('dd')):
                items.append({
                    'term': dt.get_text(strip=True),
                    'definition': dd.get_text(strip=True)
                })
            if items:
                lists.append({
                    'type': 'definition',
                    'items': items
                })
        
        return lists
        
    except Exception as e:
        logger.error(f"Error extracting lists from HTML: {str(e)}")
        return []

def extract_links_from_html(html: str, base_url: str = "") -> List[Dict[str, str]]:
    """
    Extract links from HTML content.
    
    Args:
        html: HTML content
        base_url: Base URL for resolving relative links
        
    Returns:
        List of dictionaries with link information
    """
    if not html:
        return []
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'lxml')
        
        # Find all links
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            
            # Skip empty links
            if not href:
                continue
                
            # Resolve relative URLs if base_url is provided
            if base_url and not bool(urlparse(href).netloc):
                href = urljoin(base_url, href)
            
            links.append({
                'url': href,
                'text': text,
                'title': link.get('title', '')
            })
        
        return links
        
    except Exception as e:
        logger.error(f"Error extracting links from HTML: {str(e)}")
        return []

def extract_headings_from_html(html: str) -> List[Dict[str, Any]]:
    """
    Extract headings from HTML content.
    
    Args:
        html: HTML content
        
    Returns:
        List of dictionaries with heading information
    """
    if not html:
        return []
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'lxml')
        
        # Find all headings
        headings = []
        
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            for heading in soup.find_all(tag):
                headings.append({
                    'level': int(tag[1]),
                    'text': heading.get_text(strip=True),
                    'id': heading.get('id', '')
                })
        
        return headings
        
    except Exception as e:
        logger.error(f"Error extracting headings from HTML: {str(e)}")
        return []

def extract_images_from_html(html: str, base_url: str = "") -> List[Dict[str, str]]:
    """
    Extract images from HTML content.
    
    Args:
        html: HTML content
        base_url: Base URL for resolving relative URLs
        
    Returns:
        List of dictionaries with image information
    """
    if not html:
        return []
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'lxml')
        
        # Find all images
        images = []
        
        for img in soup.find_all('img'):
            src = img.get('src', '')
            
            # Skip images without source
            if not src:
                continue
                
            # Resolve relative URLs if base_url is provided
            if base_url and not bool(urlparse(src).netloc):
                src = urljoin(base_url, src)
            
            images.append({
                'src': src,
                'alt': img.get('alt', ''),
                'title': img.get('title', ''),
                'width': img.get('width', ''),
                'height': img.get('height', '')
            })
        
        return images
        
    except Exception as e:
        logger.error(f"Error extracting images from HTML: {str(e)}")
        return []

def extract_code_blocks_from_html(html: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from HTML content.
    
    Args:
        html: HTML content
        
    Returns:
        List of dictionaries with code information
    """
    if not html:
        return []
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'lxml')
        
        # Find all code blocks
        code_blocks = []
        
        # Look for <pre><code> combinations
        for pre in soup.find_all('pre'):
            code = pre.find('code')
            if code:
                language = ''
                
                # Check for language class
                for class_name in code.get('class', []):
                    if class_name.startswith('language-'):
                        language = class_name.replace('language-', '')
                        break
                
                code_blocks.append({
                    'code': code.get_text(strip=False),
                    'language': language
                })
            else:
                # Pre without code tag
                code_blocks.append({
                    'code': pre.get_text(strip=False),
                    'language': ''
                })
        
        # Also look for code tags without pre (inline code)
        for code in soup.find_all('code'):
            # Skip if parent is pre (already handled)
            if code.parent.name == 'pre':
                continue
                
            code_blocks.append({
                'code': code.get_text(strip=True),
                'language': '',
                'inline': True
            })
        
        return code_blocks
        
    except Exception as e:
        logger.error(f"Error extracting code blocks from HTML: {str(e)}")
        return []

def clean_html(html: str) -> str:
    """
    Clean and sanitize HTML content.
    
    Args:
        html: HTML content
        
    Returns:
        Cleaned HTML
    """
    if not html:
        return ""
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove script and style tags
        for tag in soup(['script', 'style']):
            tag.extract()
        
        # Convert to string
        clean_html = str(soup)
        
        return clean_html
        
    except Exception as e:
        logger.error(f"Error cleaning HTML: {str(e)}")
        return html

def html_to_plain_text(html: str) -> str:
    """
    Convert HTML to plain text while preserving some structure.
    
    Args:
        html: HTML content
        
    Returns:
        Plain text version
    """
    if not html:
        return ""
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'lxml')
        
        # Replace some elements with their plain text equivalent
        for tag_name, replacement in [
            ('br', '\n'),
            ('p', '\n\n'),
            ('div', '\n'),
            ('h1', '\n\n'),
            ('h2', '\n\n'),
            ('h3', '\n\n'),
            ('h4', '\n\n'),
            ('h5', '\n\n'),
            ('h6', '\n\n'),
            ('li', '\n- ')
        ]:
            for tag in soup.find_all(tag_name):
                tag.insert_before(soup.new_string(replacement))
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error converting HTML to plain text: {str(e)}")
        return extract_text_from_html(html)

def convert_html_tables_to_markdown(html: str) -> str:
    """
    Convert HTML tables to markdown format.
    
    Args:
        html: HTML content
        
    Returns:
        HTML with tables replaced by markdown
    """
    if not html:
        return ""
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'lxml')
        
        # Find all tables
        for table in soup.find_all('table'):
            rows = table.find_all('tr')
            
            if not rows:
                continue
                
            # Extract headers
            headers = []
            header_row = rows[0]
            for th in header_row.find_all(['th', 'td']):
                headers.append(th.get_text(strip=True))
            
            # Create markdown table
            markdown_table = []
            
            # Add header row
            header_line = '| ' + ' | '.join(headers) + ' |'
            markdown_table.append(header_line)
            
            # Add separator row
            separator = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
            markdown_table.append(separator)
            
            # Add data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                cell_texts = [cell.get_text(strip=True) for cell in cells]
                row_line = '| ' + ' | '.join(cell_texts) + ' |'
                markdown_table.append(row_line)
            
            # Replace table with markdown
            table_markdown = '\n'.join(markdown_table)
            table.replace_with(soup.new_string(table_markdown))
        
        return str(soup)
        
    except Exception as e:
        logger.error(f"Error converting HTML tables to markdown: {str(e)}")
        return html

def extract_structured_data(html: str) -> Dict[str, Any]:
    """
    Extract structured data from HTML (e.g., microdata, JSON-LD).
    
    Args:
        html: HTML content
        
    Returns:
        Dictionary of extracted structured data
    """
    if not html:
        return {}
        
    structured_data = {
        'json_ld': [],
        'microdata': {}
    }
    
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'lxml')
        
        # Extract JSON-LD
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                import json
                data = json.loads(script.string)
                structured_data['json_ld'].append(data)
            except:
                pass
        
        # Extract microdata
        # This is a simplified implementation
        elements = soup.find_all(itemscope=True)
        for element in elements:
            item_type = element.get('itemtype', '')
            if not item_type:
                continue
                
            item_props = {}
            for prop in element.find_all(itemprop=True):
                prop_name = prop.get('itemprop', '')
                
                # Extract property value
                if prop.name == 'meta':
                    prop_value = prop.get('content', '')
                elif prop.name == 'link':
                    prop_value = prop.get('href', '')
                elif prop.name == 'img':
                    prop_value = prop.get('src', '')
                elif prop.name == 'time':
                    prop_value = prop.get('datetime', prop.get_text(strip=True))
                else:
                    prop_value = prop.get_text(strip=True)
                
                item_props[prop_name] = prop_value
            
            structured_data['microdata'][item_type] = item_props
        
        return structured_data
        
    except Exception as e:
        logger.error(f"Error extracting structured data from HTML: {str(e)}")
        return {}






app/utils/image_utils.py

"""
Utility functions for image processing.
"""
import os
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple
import base64
import io
from urllib.request import urlopen
from urllib.parse import urlparse

try:
    from PIL import Image, ExifTags
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_image_file(filename: str) -> bool:
    """
    Check if a file is an image based on extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        Boolean indicating if file is an image
    """
    if not filename:
        return False
        
    # Get file extension
    ext = os.path.splitext(filename.lower())[1]
    
    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    return ext in image_extensions

def get_image_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extract metadata from an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary of image metadata
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, cannot extract image metadata")
        return {}
        
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return {}
        
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Basic metadata
        metadata = {
            'format': img.format,
            'mode': img.mode,
            'width': img.width,
            'height': img.height,
            'size': os.path.getsize(image_path)
        }
        
        # Extract EXIF data if available
        if hasattr(img, '_getexif') and img._getexif():
            exif = {
                ExifTags.TAGS.get(tag, tag): value
                for tag, value in img._getexif().items()
                if tag in ExifTags.TAGS
            }
            
            # Extract key EXIF fields
            if 'DateTimeOriginal' in exif:
                metadata['date_taken'] = exif['DateTimeOriginal']
            if 'Make' in exif:
                metadata['camera_make'] = exif['Make']
            if 'Model' in exif:
                metadata['camera_model'] = exif['Model']
            if 'GPSInfo' in exif:
                # Process GPS info if available
                gps_info = {}
                for key, value in exif['GPSInfo'].items():
                    gps_info[ExifTags.GPSTAGS.get(key, key)] = value
                metadata['gps_info'] = gps_info
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting image metadata: {str(e)}")
        return {}

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text
    """
    if not OCR_AVAILABLE:
        logger.warning("Pytesseract not available, cannot perform OCR")
        return ""
        
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return ""
        
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Perform OCR
        text = pytesseract.image_to_string(img)
        
        return text
        
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        return ""

def resize_image(image_path: str, width: int = None, height: int = None, 
                output_path: str = None) -> Optional[str]:
    """
    Resize an image.
    
    Args:
        image_path: Path to the image file
        width: Target width (if None, calculated to maintain aspect ratio)
        height: Target height (if None, calculated to maintain aspect ratio)
        output_path: Path to save the resized image (if None, creates a temporary file)
        
    Returns:
        Path to the resized image or None on failure
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, cannot resize image")
        return None
        
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None
        
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Calculate new dimensions
        original_width, original_height = img.size
        
        if width and height:
            new_size = (width, height)
        elif width:
            # Calculate height to maintain aspect ratio
            new_height = int(original_height * (width / original_width))
            new_size = (width, new_height)
        elif height:
            # Calculate width to maintain aspect ratio
            new_width = int(original_width * (height / original_height))
            new_size = (new_width, height)
        else:
            # No resizing needed
            return image_path
        
        # Resize the image
        resized_img = img.resize(new_size, Image.LANCZOS)
        
        # Save the resized image
        if output_path:
            resized_img.save(output_path)
            return output_path
        else:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                resized_img.save(temp_path)
                return temp_path
                
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return None

def convert_image_format(image_path: str, target_format: str, 
                       output_path: str = None) -> Optional[str]:
    """
    Convert an image to a different format.
    
    Args:
        image_path: Path to the image file
        target_format: Target format (e.g., 'JPEG', 'PNG')
        output_path: Path to save the converted image (if None, creates a temporary file)
        
    Returns:
        Path to the converted image or None on failure
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, cannot convert image format")
        return None
        
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None
        
    try:
        # Open the image
        img = Image.open(image_path)
        
        # If no output path specified, create a temporary file
        if not output_path:
            ext = target_format.lower()
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                output_path = tmp.name
        
        # Convert and save
        if target_format.upper() == 'JPEG' and img.mode == 'RGBA':
            # JPEG doesn't support alpha channel, convert to RGB
            img = img.convert('RGB')
            
        img.save(output_path, format=target_format)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error converting image format: {str(e)}")
        return None

def encode_image_base64(image_path: str) -> Optional[str]:
    """
    Encode an image as a base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64-encoded image string or None on failure
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None
        
    try:
        # Get mime type based on extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(ext, 'application/octet-stream')
        
        # Read the file
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            
        # Encode as base64
        base64_data = base64.b64encode(img_data).decode('utf-8')
        
        # Format as data URL
        return f"data:{mime_type};base64,{base64_data}"
        
    except Exception as e:
        logger.error(f"Error encoding image as base64: {str(e)}")
        return None

def download_image(url: str, output_path: str = None) -> Optional[str]:
    """
    Download an image from a URL.
    
    Args:
        url: URL of the image
        output_path: Path to save the downloaded image (if None, creates a temporary file)
        
    Returns:
        Path to the downloaded image or None on failure
    """
    try:
        # Parse URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.error(f"Invalid URL: {url}")
            return None
            
        # Get image data
        with urlopen(url) as response:
            img_data = response.read()
            
        # Determine file extension from Content-Type if available
        content_type = response.info().get('Content-Type', '')
        ext = '.jpg'  # Default
        
        if content_type == 'image/jpeg':
            ext = '.jpg'
        elif content_type == 'image/png':
            ext = '.png'
        elif content_type == 'image/gif':
            ext = '.gif'
        elif content_type == 'image/bmp':
            ext = '.bmp'
        elif content_type == 'image/webp':
            ext = '.webp'
            
        # If no output path specified, create a temporary file
        if not output_path:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                output_path = tmp.name
                
        # Save the image
        with open(output_path, 'wb') as f:
            f.write(img_data)
            
        return output_path
        
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        return None

def crop_image(image_path: str, left: int, top: int, right: int, bottom: int, 
             output_path: str = None) -> Optional[str]:
    """
    Crop an image.
    
    Args:
        image_path: Path to the image file
        left: Left coordinate
        top: Top coordinate
        right: Right coordinate
        bottom: Bottom coordinate
        output_path: Path to save the cropped image (if None, creates a temporary file)
        
    Returns:
        Path to the cropped image or None on failure
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, cannot crop image")
        return None
        
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None
        
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))
        
        # Save the cropped image
        if output_path:
            cropped_img.save(output_path)
            return output_path
        else:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                cropped_img.save(temp_path)
                return temp_path
                
    except Exception as e:
        logger.error(f"Error cropping image: {str(e)}")
        return None

def get_image_colors(image_path: str, num_colors: int = 5) -> List[Tuple[int, int, int]]:
    """
    Extract dominant colors from an image.
    
    Args:
        image_path: Path to the image file
        num_colors: Number of dominant colors to extract
        
    Returns:
        List of RGB color tuples
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, cannot extract image colors")
        return []
        
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return []
        
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Resize for faster processing
        img = img.resize((100, 100))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Get pixels
        pixels = list(img.getdata())
        
        # Count color frequencies
        color_counts = {}
        for pixel in pixels:
            if pixel in color_counts:
                color_counts[pixel] += 1
            else:
                color_counts[pixel] = 1
                
        # Sort by frequency
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return the top N colors
        return [color for color, _ in sorted_colors[:num_colors]]
        
    except Exception as e:
        logger.error(f"Error extracting image colors: {str(e)}")
        return []

def create_image_thumbnail(image_path: str, size: Tuple[int, int] = (128, 128), 
                         output_path: str = None) -> Optional[str]:
    """
    Create a thumbnail of an image.
    
    Args:
        image_path: Path to the image file
        size: Thumbnail size as (width, height)
        output_path: Path to save the thumbnail (if None, creates a temporary file)
        
    Returns:
        Path to the thumbnail or None on failure
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, cannot create thumbnail")
        return None
        
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None
        
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Create a copy to avoid modifying the original
        img = img.copy()
        
        # Generate thumbnail (preserves aspect ratio)
        img.thumbnail(size)
        
        # Save the thumbnail
        if output_path:
            img.save(output_path)
            return output_path
        else:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                img.save(temp_path)
                return temp_path
                
    except Exception as e:
        logger.error(f"Error creating thumbnail: {str(e)}")
        return None




app/main.py
"""
Main application module for the Enterprise Knowledge Hub.
"""
import os
import sys
import logging
import json
from typing import Dict, List, Any, Tuple, Optional
import time
import threading

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from app.config import (
    APP_NAME,
    APP_VERSION,
    DEBUG,
    PORT,
    HOST,
    CONFLUENCE_URL,
    CONFLUENCE_USERNAME,
    CONFLUENCE_API_TOKEN,
    CONFLUENCE_SPACE_KEY,
    REMEDY_SERVER,
    REMEDY_USERNAME,
    REMEDY_PASSWORD,
    CONFIG_VALID
)

from app.api.confluence import confluence_client
from app.api.remedy import remedy_client
from app.core.document_processor import document_processor
from app.core.query_processor import query_processor
from app.core.retriever import retriever
from app.core.summarizer import summarizer

# Set up logging
logging.basicConfig(
    level=logging.INFO if not DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder='web/static', template_folder='web/templates')
CORS(app)

# Configuration
app.config['JSON_SORT_KEYS'] = False
app.config['DEBUG'] = DEBUG

# Connection status
connection_status = {
    "confluence": False,
    "remedy": False,
    "indices_built": False,
    "last_check": None,
    "documents_loaded": 0
}

def check_connections() -> Dict[str, Any]:
    """Check connections to data sources."""
    global connection_status
    
    # Check Confluence connection
    confluence_status = confluence_client.is_connected()
    
    # Check Remedy connection
    remedy_status = remedy_client.is_connected()
    
    # Update status
    connection_status["confluence"] = confluence_status
    connection_status["remedy"] = remedy_status
    connection_status["last_check"] = time.time()
    
    return connection_status

def initialize_app() -> None:
    """Initialize the application components."""
    global connection_status
    
    logger.info(f"Initializing {APP_NAME} v{APP_VERSION}")
    
    # Check connections
    check_connections()
    
    # Load document processor state or start processing
    try:
        if document_processor.load_state():
            logger.info("Document processor state loaded from cache")
            connection_status["documents_loaded"] = len(document_processor.get_all_chunks())
        else:
            logger.info("Starting document processing")
            document_processor.process_all_sources()
            document_processor.save_state()
            connection_status["documents_loaded"] = len(document_processor.get_all_chunks())
            
        # Initialize retriever
        if retriever.load_indices():
            logger.info("Search indices loaded from cache")
            connection_status["indices_built"] = True
        else:
            logger.info("Building search indices")
            retriever.build_indices()
            connection_status["indices_built"] = True
            
        logger.info(f"Initialization complete. {connection_status['documents_loaded']} documents loaded.")
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")

# Define background job for periodic tasks
def background_update_job() -> None:
    """Background job for periodic tasks."""
    while True:
        try:
            # Check connections every hour
            check_connections()
            
            # Refresh document cache once a day (86400 seconds)
            # This is simplified - a real implementation might use a more sophisticated scheduler
            document_processor.process_all_sources(force_reload=True)
            document_processor.save_state()
            
            # Rebuild indices
            retriever.build_indices(force_rebuild=True)
            
            connection_status["documents_loaded"] = len(document_processor.get_all_chunks())
            connection_status["indices_built"] = True
            
            logger.info("Background update completed successfully")
            
        except Exception as e:
            logger.error(f"Error in background update job: {str(e)}")
            
        # Sleep for a day
        time.sleep(86400)

def process_query(query: str, sources: List[str] = None) -> Dict[str, Any]:
    """
    Process a user query and generate a response.
    
    Args:
        query: User query
        sources: List of sources to search (default: both)
        
    Returns:
        Dictionary with the processed query and response
    """
    start_time = time.time()
    
    try:
        # Process query to understand intent
        query_info = query_processor.process_query(query)
        
        # Extract source preference from query
        preferred_source, clean_query = query_processor.parse_source_preference(query)
        
        # If sources not specified, use preferred source from query or default to all
        if sources is None:
            if preferred_source != "both":
                sources = [preferred_source]
            else:
                sources = ["confluence", "remedy"]
        
        # Retrieve relevant information
        search_results = retriever.search(clean_query, sources=sources)
        
        # Separate results by source
        confluence_results = [r for r in search_results if r.source == "confluence"]
        remedy_results = [r for r in search_results if r.source == "remedy"]
        
        # Generate summaries based on retrieved information
        summaries = summarizer.generate_source_summaries(
            clean_query, 
            confluence_results, 
            remedy_results
        )
        
        # Format response based on sources
        if "confluence" in sources and "remedy" in sources:
            # Generate a dual summary if both sources are requested
            formatted_response = summarizer.format_dual_summary(
                summaries.get("confluence"), 
                summaries.get("remedy")
            )
        elif "confluence" in sources:
            # Only Confluence summary
            formatted_response = summaries.get("confluence").text
        elif "remedy" in sources:
            # Only Remedy summary
            formatted_response = summaries.get("remedy").text
        else:
            formatted_response = "No information sources were specified."
        
        # Calculate timing
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Format response
        response = {
            "query": query,
            "processed_query": clean_query,
            "sources": sources,
            "results": {
                "confluence": {
                    "count": len(confluence_results),
                    "summary": summaries.get("confluence").to_dict() if "confluence" in summaries else None
                },
                "remedy": {
                    "count": len(remedy_results),
                    "summary": summaries.get("remedy").to_dict() if "remedy" in summaries else None
                }
            },
            "response": formatted_response,
            "processing_time": processing_time
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        
        # Return error response
        return {
            "query": query,
            "error": True,
            "message": f"Error processing query: {str(e)}",
            "response": "I encountered an error while processing your query. Please try again or contact support if the issue persists."
        }

# Define Flask routes (API endpoints)
@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html', app_name=APP_NAME, app_version=APP_VERSION)

@app.route('/api/query', methods=['POST'])
def api_query():
    """API endpoint for processing queries."""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({"error": "Missing query parameter"}), 400
        
    query = data['query']
    sources = data.get('sources', None)
    
    # Process the query
    response = process_query(query, sources)
    
    return jsonify(response)

@app.route('/api/status', methods=['GET'])
def api_status():
    """API endpoint for checking system status."""
    # Check connections
    status = check_connections()
    
    # Add version info
    status["app_name"] = APP_NAME
    status["app_version"] = APP_VERSION
    status["documents_loaded"] = connection_status["documents_loaded"]
    status["indices_built"] = connection_status["indices_built"]
    
    return jsonify(status)

@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    """API endpoint for refreshing the document cache."""
    try:
        # Process documents
        document_processor.process_all_sources(force_reload=True)
        document_processor.save_state()
        
        # Rebuild indices
        retriever.build_indices(force_rebuild=True)
        
        # Update status
        connection_status["documents_loaded"] = len(document_processor.get_all_chunks())
        connection_status["indices_built"] = True
        
        return jsonify({
            "success": True,
            "message": f"Refreshed {connection_status['documents_loaded']} documents",
            "documents_loaded": connection_status["documents_loaded"]
        })
        
    except Exception as e:
        logger.error(f"Error refreshing cache: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error refreshing cache: {str(e)}"
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """API endpoint for health checking."""
    healthy = connection_status["last_check"] is not None
    status_code = 200 if healthy else 503
    
    return jsonify({
        "status": "healthy" if healthy else "unhealthy",
        "version": APP_VERSION
    }), status_code

# Error handlers
@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    if request.path.startswith('/api/'):
        return jsonify({"error": "Endpoint not found"}), 404
    return render_template('error.html', error_code=404, error_message="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(e)}")
    if request.path.startswith('/api/'):
        return jsonify({"error": "Internal server error"}), 500
    return render_template('error.html', error_code=500, error_message="Server error"), 500

def main():
    """Main entry point for the application."""
    # Initialize app
    initialize_app()
    
    # Start background job in a separate thread
    bg_thread = threading.Thread(target=background_update_job, daemon=True)
    bg_thread.start()
    
    # Check if config is valid before starting
    if not CONFIG_VALID:
        logger.warning("Running with incomplete configuration. Some features may not work correctly.")
    
    # Run the Flask app
    app.run(host=HOST, port=PORT, debug=DEBUG)

if __name__ == '__main__':
    main()








app/web/routes.py
"""
Web routes module for handling HTTP requests.
"""
import os
import logging
from typing import Dict, List, Any, Optional
import time

from flask import Blueprint, request, jsonify, render_template, send_from_directory, current_app

from app.config import APP_NAME, APP_VERSION
from app.core.document_processor import document_processor
from app.core.query_processor import query_processor
from app.core.retriever import retriever
from app.core.summarizer import summarizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create blueprint
routes = Blueprint('routes', __name__)

@routes.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html', app_name=APP_NAME, app_version=APP_VERSION)

@routes.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html', app_name=APP_NAME, app_version=APP_VERSION)

@routes.route('/api/query', methods=['POST'])
def api_query():
    """API endpoint for processing queries."""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({"error": "Missing query parameter"}), 400
        
    query = data['query']
    sources = data.get('sources', None)
    
    # Process the query
    return jsonify(process_query(query, sources))

@routes.route('/api/status', methods=['GET'])
def api_status():
    """API endpoint for checking system status."""
    # Check document count
    total_chunks = len(document_processor.get_all_chunks())
    
    # Count by source
    confluence_chunks = len(document_processor.get_chunks_by_source("confluence"))
    remedy_chunks = len(document_processor.get_chunks_by_source("remedy"))
    
    status = {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "documents_loaded": {
            "total": total_chunks,
            "confluence": confluence_chunks,
            "remedy": remedy_chunks
        },
        "indices_built": retriever.indices_built,
        "sources": {
            "confluence": {
                "connected": current_app.config.get("CONFLUENCE_CONNECTED", False),
                "space_key": current_app.config.get("CONFLUENCE_SPACE_KEY", "")
            },
            "remedy": {
                "connected": current_app.config.get("REMEDY_CONNECTED", False),
                "server": current_app.config.get("REMEDY_SERVER", "")
            }
        }
    }
    
    return jsonify(status)

@routes.route('/api/refresh', methods=['POST'])
def api_refresh():
    """API endpoint for refreshing the document cache."""
    try:
        # Get source parameter (optional)
        source = request.json.get('source') if request.json else None
        force_reload = request.json.get('force_reload', False) if request.json else False
        
        # Process documents
        if source == "confluence":
            documents = document_processor.process_confluence_content()
            message = f"Refreshed {len(documents)} Confluence documents"
        elif source == "remedy":
            documents = document_processor.process_remedy_content()
            message = f"Refreshed {len(documents)} Remedy documents"
        else:
            # Process both sources
            document_processor.process_all_sources(force_reload=force_reload)
            message = f"Refreshed all documents"
            
        # Save state
        document_processor.save_state()
        
        # Rebuild indices
        retriever.build_indices(force_rebuild=force_reload)
        
        # Get updated counts
        total_chunks = len(document_processor.get_all_chunks())
        confluence_chunks = len(document_processor.get_chunks_by_source("confluence"))
        remedy_chunks = len(document_processor.get_chunks_by_source("remedy"))
        
        return jsonify({
            "success": True,
            "message": message,
            "documents_loaded": {
                "total": total_chunks,
                "confluence": confluence_chunks,
                "remedy": remedy_chunks
            }
        })
        
    except Exception as e:
        logger.error(f"Error refreshing cache: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error refreshing cache: {str(e)}"
        }), 500

@routes.route('/api/search', methods=['GET'])
def api_search():
    """API endpoint for direct search (without summarization)."""
    query = request.args.get('q')
    source = request.args.get('source')
    limit = request.args.get('limit', 10, type=int)
    
    if not query:
        return jsonify({"error": "Missing query parameter"}), 400
        
    # Set source filter
    sources = None
    if source == "confluence":
        sources = ["confluence"]
    elif source == "remedy":
        sources = ["remedy"]
    
    try:
        # Search for relevant chunks
        search_results = retriever.search(query, sources=sources, top_k=limit)
        
        # Format results
        results = [result.to_dict() for result in search_results]
        
        return jsonify({
            "query": query,
            "source": source,
            "count": len(results),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error searching: {str(e)}")
        return jsonify({
            "error": True,
            "message": f"Error searching: {str(e)}"
        }), 500

@routes.route('/api/sources', methods=['GET'])
def api_sources():
    """API endpoint for getting available sources."""
    # Count chunks by source
    confluence_chunks = len(document_processor.get_chunks_by_source("confluence"))
    remedy_chunks = len(document_processor.get_chunks_by_source("remedy"))
    
    sources = [
        {
            "id": "confluence",
            "name": "Confluence",
            "description": "Knowledge base documentation and articles",
            "document_count": confluence_chunks,
            "available": confluence_chunks > 0
        },
        {
            "id": "remedy",
            "name": "Remedy",
            "description": "Service desk tickets and issues",
            "document_count": remedy_chunks,
            "available": remedy_chunks > 0
        }
    ]
    
    return jsonify(sources)

@routes.route('/api/suggest', methods=['GET'])
def api_suggest():
    """API endpoint for query suggestions."""
    query = request.args.get('q')
    
    if not query or len(query) < 3:
        return jsonify([])
        
    try:
        # Get related queries
        # For demonstration, we'll just provide some hardcoded suggestions
        # In a real system, you might use query logs or more sophisticated methods
        suggestions = [
            f"{query} confluence",
            f"{query} documentation",
            f"{query} remedy ticket",
            f"{query} troubleshooting",
            f"{query} guide"
        ]
        
        return jsonify(suggestions[:3])
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        return jsonify([])

def process_query(query: str, sources: List[str] = None) -> Dict[str, Any]:
    """
    Process a user query and generate a response.
    
    Args:
        query: User query
        sources: List of sources to search (default: both)
        
    Returns:
        Dictionary with the processed query and response
    """
    start_time = time.time()
    
    try:
        # Process query to understand intent
        query_info = query_processor.process_query(query)
        
        # Extract source preference from query
        preferred_source, clean_query = query_processor.parse_source_preference(query)
        
        # If sources not specified, use preferred source from query or default to all
        if sources is None:
            if preferred_source != "both":
                sources = [preferred_source]
            else:
                sources = ["confluence", "remedy"]
        
        # Retrieve relevant information
        search_results = retriever.search(clean_query, sources=sources)
        
        # Separate results by source
        confluence_results = [r for r in search_results if r.source == "confluence"]
        remedy_results = [r for r in search_results if r.source == "remedy"]
        
        # Generate summaries based on retrieved information
        summaries = summarizer.generate_source_summaries(
            clean_query, 
            confluence_results, 
            remedy_results
        )
        
        # Format response based on sources
        if "confluence" in sources and "remedy" in sources:
            # Generate a dual summary if both sources are requested
            formatted_response = summarizer.format_dual_summary(
                summaries.get("confluence"), 
                summaries.get("remedy")
            )
        elif "confluence" in sources:
            # Only Confluence summary
            formatted_response = summaries.get("confluence").text
        elif "remedy" in sources:
            # Only Remedy summary
            formatted_response = summaries.get("remedy").text
        else:
            formatted_response = "No information sources were specified."
        
        # Calculate timing
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Format response
        response = {
            "query": query,
            "processed_query": clean_query,
            "sources": sources,
            "results": {
                "confluence": {
                    "count": len(confluence_results),
                    "summary": summaries.get("confluence").to_dict() if "confluence" in summaries else None
                },
                "remedy": {
                    "count": len(remedy_results),
                    "summary": summaries.get("remedy").to_dict() if "remedy" in summaries else None
                }
            },
            "response": formatted_response,
            "processing_time": processing_time
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        
        # Return error response
        return {
            "query": query,
            "error": True,
            "message": f"Error processing query: {str(e)}",
            "response": "I encountered an error while processing your query. Please try again or contact support if the issue persists."
        }




app/__init__.py

"""
Application initialization module.
"""
import os
import logging
from flask import Flask
from flask_cors import CORS

from app.config import (
    APP_NAME,
    APP_VERSION,
    DEBUG,
    CONFLUENCE_URL,
    CONFLUENCE_USERNAME,
    CONFLUENCE_API_TOKEN,
    CONFLUENCE_SPACE_KEY,
    REMEDY_SERVER,
    REMEDY_USERNAME,
    REMEDY_PASSWORD
)

from app.api.confluence import confluence_client
from app.api.remedy import remedy_client
from app.web.routes import routes

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app(testing=False):
    """
    Create and configure the Flask application.
    
    Args:
        testing: Whether to create the app in testing mode
        
    Returns:
        Configured Flask app
    """
    # Create Flask app
    app = Flask(__name__, 
               static_folder=os.path.join(os.path.dirname(__file__), 'web', 'static'),
               template_folder=os.path.join(os.path.dirname(__file__), 'web', 'templates'))
    
    # Enable CORS
    CORS(app)
    
    # Configure app
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
        DEBUG=DEBUG,
        TESTING=testing,
        APP_NAME=APP_NAME,
        APP_VERSION=APP_VERSION,
        
        # API endpoints
        CONFLUENCE_URL=CONFLUENCE_URL,
        CONFLUENCE_USERNAME=CONFLUENCE_USERNAME,
        CONFLUENCE_API_TOKEN=CONFLUENCE_API_TOKEN,
        CONFLUENCE_SPACE_KEY=CONFLUENCE_SPACE_KEY,
        REMEDY_SERVER=REMEDY_SERVER,
        REMEDY_USERNAME=REMEDY_USERNAME,
        REMEDY_PASSWORD=REMEDY_PASSWORD,
        
        # Connection status (updated by main.py)
        CONFLUENCE_CONNECTED=False,
        REMEDY_CONNECTED=False
    )
    
    # Check connections on startup
    if not testing:
        # Check Confluence connection
        confluence_connected = confluence_client.is_connected()
        app.config['CONFLUENCE_CONNECTED'] = confluence_connected
        
        if confluence_connected:
            logger.info(f"Connected to Confluence: {CONFLUENCE_URL}")
        else:
            logger.warning(f"Could not connect to Confluence: {CONFLUENCE_URL}")
        
        # Check Remedy connection
        remedy_connected = remedy_client.is_connected()
        app.config['REMEDY_CONNECTED'] = remedy_connected
        
        if remedy_connected:
            logger.info(f"Connected to Remedy: {REMEDY_SERVER}")
        else:
            logger.warning(f"Could not connect to Remedy: {REMEDY_SERVER}")
    
    # Register blueprints
    app.register_blueprint(routes)
    
    # Configure error handlers
    @app.errorhandler(404)
    def not_found(e):
        """Handle 404 errors."""
        if request.path.startswith('/api/'):
            return jsonify({"error": "Endpoint not found"}), 404
        return render_template('error.html', error_code=404, error_message="Page not found"), 404

    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors."""
        logger.error(f"Server error: {str(e)}")
        if request.path.startswith('/api/'):
            return jsonify({"error": "Internal server error"}), 500
        return render_template('error.html', error_code=500, error_message="Server error"), 500
    
    # Return app instance
    return app



app/web/templates/index.html

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ app_name }} - Enterprise Knowledge Hub</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="icon" type="image/png" href="{{ url_for('static', filename='img/favicon.png') }}">
    <meta name="description" content="Enterprise Knowledge Hub - Search and retrieve information from Confluence and Remedy systems">
</head>
<body>
    <div class="app-container">
        <!-- Header -->
        <header class="app-header">
            <div class="logo">
                <img src="{{ url_for('static', filename='img/logo.svg') }}" alt="{{ app_name }}">
                <h1>{{ app_name }}</h1>
            </div>
            <div class="header-actions">
                <button id="refresh-button" class="icon-button" title="Refresh Data">
                    <i class="fas fa-sync-alt"></i>
                </button>
                <button id="theme-toggle" class="icon-button" title="Toggle Theme">
                    <i class="fas fa-moon"></i>
                </button>
                <span class="version-info">v{{ app_version }}</span>
            </div>
        </header>

        <!-- Main Content -->
        <main class="app-main">
            <section class="search-section">
                <div class="search-container">
                    <h2>Ask Your Enterprise Knowledge Hub</h2>
                    <p class="subtitle">Get answers from your organization's knowledge base and ticket system</p>
                    
                    <div class="search-box">
                        <input type="text" id="search-input" placeholder="Ask a question (e.g., 'How to reset my password?')">
                        <button id="search-button">
                            <i class="fas fa-search"></i>
                        </button>
                    </div>
                    
                    <div class="search-options">
                        <label class="source-option">
                            <input type="checkbox" id="confluence-source" checked>
                            <span class="checkmark"></span>
                            <span class="source-name">
                                <i class="fas fa-book"></i> Confluence
                            </span>
                        </label>
                        <label class="source-option">
                            <input type="checkbox" id="remedy-source" checked>
                            <span class="checkmark"></span>
                            <span class="source-name">
                                <i class="fas fa-ticket-alt"></i> Remedy
                            </span>
                        </label>
                    </div>
                </div>
            </section>

            <section class="results-section" id="results-section">
                <div class="results-container">
                    <div class="loading-indicator" id="loading-indicator">
                        <div class="spinner"></div>
                        <p>Searching knowledge base...</p>
                    </div>
                    
                    <div class="results-content" id="results-content">
                        <!-- Results will be displayed here -->
                    </div>
                </div>
            </section>
        </main>

        <!-- Footer -->
        <footer class="app-footer">
            <div class="connection-status">
                <div class="status-indicator" id="confluence-status">
                    <span class="status-dot"></span>
                    <span class="status-name">Confluence</span>
                </div>
                <div class="status-indicator" id="remedy-status">
                    <span class="status-dot"></span>
                    <span class="status-name">Remedy</span>
                </div>
            </div>
            <div class="copyright">
                &copy; 2023-2025 Enterprise Knowledge Hub
            </div>
        </footer>
    </div>

    <!-- Modals -->
    <div class="modal" id="refresh-modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3>Refresh Knowledge Base</h3>
                <span class="close-modal">&times;</span>
            </div>
            <div class="modal-body">
                <p>Select which knowledge sources to refresh:</p>
                <div class="refresh-options">
                    <label class="source-option">
                        <input type="checkbox" id="refresh-confluence" checked>
                        <span class="checkmark"></span>
                        <span class="source-name">Confluence</span>
                    </label>
                    <label class="source-option">
                        <input type="checkbox" id="refresh-remedy" checked>
                        <span class="checkmark"></span>
                        <span class="source-name">Remedy</span>
                    </label>
                </div>
                <div class="modal-options">
                    <label class="source-option">
                        <input type="checkbox" id="force-reload">
                        <span class="checkmark"></span>
                        <span class="source-name">Force full reload</span>
                    </label>
                </div>
            </div>
            <div class="modal-footer">
                <button id="cancel-refresh" class="secondary-button">Cancel</button>
                <button id="confirm-refresh" class="primary-button">Refresh</button>
            </div>
        </div>
    </div>

    <!-- Templates -->
    <template id="result-template">
        <div class="result-card">
            <div class="result-header">
                <div class="query-info">
                    <span class="query-text"></span>
                    <span class="processing-time"><i class="fas fa-clock"></i> <span class="time-value"></span> seconds</span>
                </div>
            </div>
            <div class="result-body">
                <div class="tabs">
                    <button class="tab-button active" data-tab="combined">Combined</button>
                    <button class="tab-button" data-tab="confluence">Confluence</button>
                    <button class="tab-button" data-tab="remedy">Remedy</button>
                </div>
                <div class="tab-content">
                    <div class="tab-pane active" id="combined-tab">
                        <div class="markdown-content"></div>
                    </div>
                    <div class="tab-pane" id="confluence-tab">
                        <div class="confluence-content"></div>
                        <div class="source-info">
                            <span class="result-count"><span class="count-value"></span> results from Confluence</span>
                            <span class="confidence">Confidence: <span class="confidence-value"></span></span>
                        </div>
                    </div>
                    <div class="tab-pane" id="remedy-tab">
                        <div class="remedy-content"></div>
                        <div class="source-info">
                            <span class="result-count"><span class="count-value"></span> results from Remedy</span>
                            <span class="confidence">Confidence: <span class="confidence-value"></span></span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </template>

    <!-- Scripts -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js"></script>
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>




app/web/static/css/style.css


:root {
    /* Light theme colors */
    --primary-color: #4a6cf7;
    --primary-color-dark: #3a5bd8;
    --secondary-color: #6c757d;
    --accent-color: #29cc97;
    --background-color: #f8fafc;
    --card-color: #ffffff;
    --text-color: #334155;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --shadow-color: rgba(0, 0, 0, 0.05);
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
    --info-color: #3b82f6;
    
    /* Font sizes */
    --font-size-xs: 0.75rem;
    --font-size-sm: 0.875rem;
    --font-size-md: 1rem;
    --font-size-lg: 1.125rem;
    --font-size-xl: 1.25rem;
    --font-size-2xl: 1.5rem;
    --font-size-3xl: 1.875rem;
    
    /* Spacing */
    --spacing-xs: 0.25rem;
    --spacing-sm: 0.5rem;
    --spacing-md: 1rem;
    --spacing-lg: 1.5rem;
    --spacing-xl: 2rem;
    --spacing-2xl: 3rem;
    
    /* Border radius */
    --border-radius-sm: 0.25rem;
    --border-radius-md: 0.5rem;
    --border-radius-lg: 1rem;
    --border-radius-full: 9999px;
    
    /* Transitions */
    --transition-fast: 150ms ease;
    --transition-normal: 300ms ease;
    --transition-slow: 500ms ease;
}

/* Dark Theme */
.dark-theme {
    --primary-color: #5a7bf9;
    --primary-color-dark: #4a6cf7;
    --background-color: #0f172a;
    --card-color: #1e293b;
    --text-color: #e2e8f0;
    --text-secondary: #94a3b8;
    --border-color: #334155;
    --shadow-color: rgba(0, 0, 0, 0.2);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Roboto', sans-serif;
    font-size: var(--font-size-md);
    color: var(--text-color);
    background-color: var(--background-color);
    line-height: 1.6;
    transition: background-color var(--transition-normal);
}

.app-container {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

/* Header styles */
.app-header {
    background-color: var(--card-color);
    box-shadow: 0 2px 10px var(--shadow-color);
    padding: var(--spacing-md) var(--spacing-xl);
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 10;
}

.logo {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
}

.logo img {
    height: 2.5rem;
}

.logo h1 {
    font-size: var(--font-size-xl);
    font-weight: 500;
    color: var(--primary-color);
}

.header-actions {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
}

.icon-button {
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: var(--font-size-lg);
    padding: var(--spacing-sm);
    border-radius: var(--border-radius-full);
    transition: all var(--transition-fast);
    display: flex;
    align-items: center;
    justify-content: center;
}

.icon-button:hover {
    color: var(--primary-color);
    background-color: rgba(74, 108, 247, 0.1);
}

.version-info {
    font-size: var(--font-size-xs);
    color: var(--text-secondary);
    padding: var(--spacing-xs) var(--spacing-sm);
    background-color: var(--background-color);
    border-radius: var(--border-radius-full);
}

/* Main content styles */
.app-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: var(--spacing-xl);
    gap: var(--spacing-xl);
}

.search-section {
    max-width: 800px;
    margin: 0 auto;
    width: 100%;
}

.search-container {
    background-color: var(--card-color);
    padding: var(--spacing-xl);
    border-radius: var(--border-radius-lg);
    box-shadow: 0 4px 20px var(--shadow-color);
    text-align: center;
}

.search-container h2 {
    font-size: var(--font-size-2xl);
    margin-bottom: var(--spacing-sm);
    color: var(--primary-color);
}

.subtitle {
    color: var(--text-secondary);
    margin-bottom: var(--spacing-lg);
}

.search-box {
    display: flex;
    margin-bottom: var(--spacing-md);
    box-shadow: 0 2px 10px var(--shadow-color);
    border-radius: var(--border-radius-full);
    overflow: hidden;
}

.search-box input {
    flex: 1;
    padding: var(--spacing-md) var(--spacing-lg);
    border: none;
    font-size: var(--font-size-md);
    color: var(--text-color);
    background-color: var(--card-color);
    outline: none;
}

.search-box button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    padding: var(--spacing-md) var(--spacing-lg);
    cursor: pointer;
    transition: background-color var(--transition-fast);
}

.search-box button:hover {
    background-color: var(--primary-color-dark);
}

.search-options {
    display: flex;
    justify-content: center;
    gap: var(--spacing-lg);
    margin-top: var(--spacing-md);
}

.source-option {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    cursor: pointer;
    user-select: none;
}

.source-option input {
    position: absolute;
    opacity: 0;
    cursor: pointer;
    height: 0;
    width: 0;
}

.checkmark {
    position: relative;
    display: inline-block;
    height: 18px;
    width: 18px;
    background-color: #eee;
    border-radius: var(--border-radius-sm);
    transition: all var(--transition-fast);
}

.source-option:hover .checkmark {
    background-color: #ccc;
}

.source-option input:checked ~ .checkmark {
    background-color: var(--primary-color);
}

.checkmark:after {
    content: "";
    position: absolute;
    display: none;
}

.source-option input:checked ~ .checkmark:after {
    display: block;
}

.source-option .checkmark:after {
    left: 6px;
    top: 2px;
    width: 5px;
    height: 10px;
    border: solid white;
    border-width: 0 2px 2px 0;
    transform: rotate(45deg);
}

.source-name {
    color: var(--text-secondary);
    font-size: var(--font-size-sm);
}

.source-option input:checked ~ .source-name {
    color: var(--text-color);
}

/* Results section */
.results-section {
    max-width: 800px;
    margin: 0 auto;
    width: 100%;
    display: none;
}

.results-container {
    background-color: var(--card-color);
    border-radius: var(--border-radius-lg);
    box-shadow: 0 4px 20px var(--shadow-color);
    overflow: hidden;
}

.loading-indicator {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: var(--spacing-xl);
    display: none;
}

.spinner {
    width: 40px;
    height: 40px;
    border: 4px solid rgba(0, 0, 0, 0.1);
    border-radius: 50%;
    border-top-color: var(--primary-color);
    animation: spin 1s ease-in-out infinite;
    margin-bottom: var(--spacing-md);
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.loading-indicator p {
    color: var(--text-secondary);
}

.result-card {
    border-radius: var(--border-radius-lg);
}

.result-header {
    padding: var(--spacing-md) var(--spacing-lg);
    border-bottom: 1px solid var(--border-color);
}

.query-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.query-text {
    font-weight: 500;
    color: var(--text-color);
}

.processing-time {
    font-size: var(--font-size-xs);
    color: var(--text-secondary);
}

.result-body {
    padding: var(--spacing-md);
}

.tabs {
    display: flex;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: var(--spacing-md);
}

.tab-button {
    padding: var(--spacing-sm) var(--spacing-md);
    border: none;
    background: none;
    cursor: pointer;
    color: var(--text-secondary);
    font-size: var(--font-size-sm);
    transition: color var(--transition-fast);
    border-bottom: 2px solid transparent;
}

.tab-button:hover {
    color: var(--primary-color);
}

.tab-button.active {
    color: var(--primary-color);
    border-bottom: 2px solid var(--primary-color);
}

.tab-content {
    padding: var(--spacing-md) 0;
}

.tab-pane {
    display: none;
}

.tab-pane.active {
    display: block;
}

.markdown-content,
.confluence-content,
.remedy-content {
    line-height: 1.6;
    color: var(--text-color);
}

.markdown-content h1,
.markdown-content h2,
.markdown-content h3,
.markdown-content h4,
.markdown-content h5,
.markdown-content h6,
.confluence-content h1,
.confluence-content h2,
.confluence-content h3,
.confluence-content h4,
.confluence-content h5,
.confluence-content h6,
.remedy-content h1,
.remedy-content h2,
.remedy-content h3,
.remedy-content h4,
.remedy-content h5,
.remedy-content h6 {
    margin: var(--spacing-md) 0 var(--spacing-sm);
    color: var(--text-color);
}

.markdown-content h2,
.confluence-content h2,
.remedy-content h2 {
    font-size: var(--font-size-xl);
    padding-bottom: var(--spacing-sm);
    border-bottom: 1px solid var(--border-color);
}

.markdown-content p,
.confluence-content p,
.remedy-content p {
    margin-bottom: var(--spacing-md);
}

.markdown-content ul,
.confluence-content ul,
.remedy-content ul,
.markdown-content ol,
.confluence-content ol,
.remedy-content ol {
    margin-bottom: var(--spacing-md);
    padding-left: var(--spacing-lg);
}

.markdown-content li,
.confluence-content li,
.remedy-content li {
    margin-bottom: var(--spacing-xs);
}

.markdown-content pre,
.confluence-content pre,
.remedy-content pre {
    background-color: var(--background-color);
    padding: var(--spacing-md);
    border-radius: var(--border-radius-md);
    overflow-x: auto;
    margin-bottom: var(--spacing-md);
}

.markdown-content code,
.confluence-content code,
.remedy-content code {
    font-family: monospace;
    background-color: var(--background-color);
    padding: 2px 4px;
    border-radius: var(--border-radius-sm);
    font-size: 0.9em;
}

.markdown-content blockquote,
.confluence-content blockquote,
.remedy-content blockquote {
    border-left: 4px solid var(--primary-color);
    padding-left: var(--spacing-md);
    color: var(--text-secondary);
    margin-bottom: var(--spacing-md);
}

.source-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: var(--spacing-md);
    padding-top: var(--spacing-md);
    border-top: 1px solid var(--border-color);
    font-size: var(--font-size-xs);
    color: var(--text-secondary);
}

/* Footer styles */
.app-footer {
    background-color: var(--card-color);
    padding: var(--spacing-md) var(--spacing-xl);
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 -2px 10px var(--shadow-color);
}

.connection-status {
    display: flex;
    gap: var(--spacing-md);
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    font-size: var(--font-size-sm);
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background-color: var(--danger-color);
}

.status-dot.connected {
    background-color: var(--success-color);
}

.status-name {
    color: var(--text-secondary);
}

.copyright {
    font-size: var(--font-size-xs);
    color: var(--text-secondary);
}

/* Modal styles */
.modal {
    display: none;
    position: fixed;
    z-index: 100;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(5px);
}

.modal-content {
    background-color: var(--card-color);
    margin: 10% auto;
    border-radius: var(--border-radius-lg);
    box-shadow: 0 4px 20px var(--shadow-color);
    max-width: 500px;
    width: 90%;
    animation: modalFadeIn var(--transition-normal);
}

@keyframes modalFadeIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

.modal-header {
    padding: var(--spacing-md) var(--spacing-lg);
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.modal-header h3 {
    color: var(--text-color);
    font-size: var(--font-size-lg);
}

.close-modal {
    cursor: pointer;
    font-size: var(--font-size-xl);
    color: var(--text-secondary);
    transition: color var(--transition-fast);
}

.close-modal:hover {
    color: var(--danger-color);
}

.modal-body {
    padding: var(--spacing-lg);
}

.refresh-options {
    display: flex;
    justify-content: space-between;
    margin: var(--spacing-md) 0;
}

.modal-options {
    margin-top: var(--spacing-lg);
}

.modal-footer {
    padding: var(--spacing-md) var(--spacing-lg);
    border-top: 1px solid var(--border-color);
    display: flex;
    justify-content: flex-end;
    gap: var(--spacing-md);
}

.primary-button,
.secondary-button {
    padding: var(--spacing-sm) var(--spacing-lg);
    border-radius: var(--border-radius-md);
    border: none;
    font-size: var(--font-size-sm);
    cursor: pointer;
    transition: all var(--transition-fast);
}

.primary-button {
    background-color: var(--primary-color);
    color: white;
}

.primary-button:hover {
    background-color: var(--primary-color-dark);
}

.secondary-button {
    background-color: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
}

.secondary-button:hover {
    color: var(--text-color);
    border-color: var(--text-color);
}

/* Responsive styles */
@media (max-width: 768px) {
    .app-header {
        padding: var(--spacing-md);
    }
    
    .logo img {
        height: 2rem;
    }
    
    .logo h1 {
        font-size: var(--font-size-lg);
    }
    
    .app-main {
        padding: var(--spacing-md);
    }
    
    .search-container {
        padding: var(--spacing-md);
    }
    
    .search-container h2 {
        font-size: var(--font-size-xl);
    }
    
    .search-options {
        flex-direction: column;
        gap: var(--spacing-sm);
        align-items: flex-start;
    }
    
    .app-footer {
        flex-direction: column;
        gap: var(--spacing-md);
        text-align: center;
    }
    
    .connection-status {
        justify-content: center;
    }
}







app/web/static/js/main.js


// Enterprise Knowledge Hub - Frontend JavaScript

// Global state
const state = {
    darkMode: false,
    lastQuery: null,
    connectionStatus: {
        confluence: false,
        remedy: false
    }
};

// DOM elements
const elements = {
    // Search elements
    searchInput: document.getElementById('search-input'),
    searchButton: document.getElementById('search-button'),
    confluenceSource: document.getElementById('confluence-source'),
    remedySource: document.getElementById('remedy-source'),
    resultsSection: document.getElementById('results-section'),
    resultsContent: document.getElementById('results-content'),
    loadingIndicator: document.getElementById('loading-indicator'),
    
    // Header elements
    refreshButton: document.getElementById('refresh-button'),
    themeToggle: document.getElementById('theme-toggle'),
    
    // Status elements
    confluenceStatus: document.getElementById('confluence-status'),
    remedyStatus: document.getElementById('remedy-status'),
    
    // Modal elements
    refreshModal: document.getElementById('refresh-modal'),
    refreshConfluence: document.getElementById('refresh-confluence'),
    refreshRemedy: document.getElementById('refresh-remedy'),
    forceReload: document.getElementById('force-reload'),
    confirmRefresh: document.getElementById('confirm-refresh'),
    cancelRefresh: document.getElementById('cancel-refresh'),
    closeModal: document.querySelector('.close-modal'),
    
    // Templates
    resultTemplate: document.getElementById('result-template')
};

// Event listeners
function setupEventListeners() {
    // Search functionality
    elements.searchButton.addEventListener('click', handleSearch);
    elements.searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });
    
    // Theme toggle
    elements.themeToggle.addEventListener('click', toggleDarkMode);
    
    // Refresh modal
    elements.refreshButton.addEventListener('click', openRefreshModal);
    elements.closeModal.addEventListener('click', closeRefreshModal);
    elements.cancelRefresh.addEventListener('click', closeRefreshModal);
    elements.confirmRefresh.addEventListener('click', handleRefresh);
    
    // Close modal when clicking outside
    window.addEventListener('click', (e) => {
        if (e.target === elements.refreshModal) {
            closeRefreshModal();
        }
    });
    
    // Check system status when page loads
    window.addEventListener('load', checkSystemStatus);
}

// Handle search functionality
async function handleSearch() {
    const query = elements.searchInput.value.trim();
    
    if (!query) {
        // Show error if query is empty
        elements.searchInput.classList.add('error');
        setTimeout(() => {
            elements.searchInput.classList.remove('error');
        }, 1000);
        return;
    }
    
    // Get selected sources
    const sources = [];
    if (elements.confluenceSource.checked) sources.push('confluence');
    if (elements.remedySource.checked) sources.push('remedy');
    
    if (sources.length === 0) {
        // At least one source should be selected
        alert('Please select at least one knowledge source');
        return;
    }
    
    // Show loading and results section
    elements.resultsSection.style.display = 'block';
    elements.loadingIndicator.style.display = 'flex';
    elements.resultsContent.style.display = 'none';
    
    try {
        // Save query to state
        state.lastQuery = query;
        
        // Execute search query
        const response = await executeQuery(query, sources);
        
        // Display results
        displayResults(response);
    } catch (error) {
        console.error('Search error:', error);
        displayError('An error occurred while processing your query. Please try again.');
    } finally {
        // Hide loading indicator
        elements.loadingIndicator.style.display = 'none';
        elements.resultsContent.style.display = 'block';
    }
}

// Execute search query
async function executeQuery(query, sources) {
    const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query: query,
            sources: sources
        })
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
    }
    
    return await response.json();
}

// Display search results
function displayResults(data) {
    // Clear previous results
    elements.resultsContent.innerHTML = '';
    
    if (data.error) {
        displayError(data.message || 'An error occurred while processing your query');
        return;
    }
    
    // Clone result template
    const template = elements.resultTemplate.content.cloneNode(true);
    
    // Fill in query info
    template.querySelector('.query-text').textContent = `"${data.query}"`;
    template.querySelector('.time-value').textContent = data.processing_time.toFixed(2);
    
    // Fill in tab content
    const combinedTab = template.querySelector('#combined-tab .markdown-content');
    combinedTab.innerHTML = marked.parse(data.response);
    
    // Confluence tab
    const confluenceTab = template.querySelector('#confluence-tab');
    const confluenceContent = confluenceTab.querySelector('.confluence-content');
    const confluenceCount = confluenceTab.querySelector('.count-value');
    const confluenceConfidence = confluenceTab.querySelector('.confidence-value');
    
    if (data.results.confluence && data.results.confluence.summary) {
        confluenceContent.innerHTML = marked.parse(data.results.confluence.summary.text);
        confluenceCount.textContent = data.results.confluence.count;
        confluenceConfidence.textContent = (data.results.confluence.summary.confidence * 100).toFixed(0) + '%';
    } else {
        confluenceContent.innerHTML = '<p>No information available from Confluence.</p>';
        confluenceCount.textContent = '0';
        confluenceConfidence.textContent = 'N/A';
    }
    
    // Remedy tab
    const remedyTab = template.querySelector('#remedy-tab');
    const remedyContent = remedyTab.querySelector('.remedy-content');
    const remedyCount = remedyTab.querySelector('.count-value');
    const remedyConfidence = remedyTab.querySelector('.confidence-value');
    
    if (data.results.remedy && data.results.remedy.summary) {
        remedyContent.innerHTML = marked.parse(data.results.remedy.summary.text);
        remedyCount.textContent = data.results.remedy.count;
        remedyConfidence.textContent = (data.results.remedy.summary.confidence * 100).toFixed(0) + '%';
    } else {
        remedyContent.innerHTML = '<p>No information available from Remedy.</p>';
        remedyCount.textContent = '0';
        remedyConfidence.textContent = 'N/A';
    }
    
    // Set up tab switching
    const tabButtons = template.querySelectorAll('.tab-button');
    const tabPanes = template.querySelectorAll('.tab-pane');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons and panes
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Get target tab and activate it
            const targetTab = this.getAttribute('data-tab');
            template.querySelector(`#${targetTab}-tab`).classList.add('active');
        });
    });
    
    // Add result to the page
    elements.resultsContent.appendChild(template);
}

// Display error message
function displayError(message) {
    elements.resultsContent.innerHTML = `
        <div class="error-message">
            <i class="fas fa-exclamation-circle"></i>
            <p>${message}</p>
        </div>
    `;
}

// Toggle dark mode
function toggleDarkMode() {
    const body = document.body;
    const themeIcon = elements.themeToggle.querySelector('i');
    
    state.darkMode = !state.darkMode;
    
    if (state.darkMode) {
        body.classList.add('dark-theme');
        themeIcon.classList.remove('fa-moon');
        themeIcon.classList.add('fa-sun');
    } else {
        body.classList.remove('dark-theme');
        themeIcon.classList.remove('fa-sun');
        themeIcon.classList.add('fa-moon');
    }
    
    // Save preference to localStorage
    localStorage.setItem('darkMode', state.darkMode);
}

// Open refresh modal
function openRefreshModal() {
    elements.refreshModal.style.display = 'block';
}

// Close refresh modal
function closeRefreshModal() {
    elements.refreshModal.style.display = 'none';
}

// Handle refresh
async function handleRefresh() {
    // Get selected sources
    const sources = [];
    if (elements.refreshConfluence.checked) sources.push('confluence');
    if (elements.refreshRemedy.checked) sources.push('remedy');
    
    if (sources.length === 0) {
        alert('Please select at least one source to refresh');
        return;
    }
    
    // Show loading in modal
    elements.confirmRefresh.textContent = 'Refreshing...';
    elements.confirmRefresh.disabled = true;
    elements.cancelRefresh.disabled = true;
    
    try {
        // For each selected source, send refresh request
        const refreshPromises = sources.map(source => {
            return fetch('/api/refresh', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    source: source,
                    force_reload: elements.forceReload.checked
                })
            }).then(res => res.json());
        });
        
        // Wait for all refreshes to complete
        const results = await Promise.all(refreshPromises);
        
        // Display result
        let successCount = results.filter(r => r.success).length;
        
        if (successCount === sources.length) {
            alert(`Successfully refreshed ${sources.join(' and ')} data.`);
        } else {
            alert('Some refresh operations failed. Please check the console for details.');
            console.error('Refresh results:', results);
        }
        
        // Check system status again
        await checkSystemStatus();
        
    } catch (error) {
        console.error('Refresh error:', error);
        alert('An error occurred during refresh. Please try again.');
    } finally {
        // Reset modal
        elements.confirmRefresh.textContent = 'Refresh';
        elements.confirmRefresh.disabled = false;
        elements.cancelRefresh.disabled = false;
        closeRefreshModal();
    }
}

// Check system status
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/status');
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update connection status
        updateConnectionStatus('confluence', data.sources.confluence.connected);
        updateConnectionStatus('remedy', data.sources.remedy.connected);
        
        // Save to state
        state.connectionStatus.confluence = data.sources.confluence.connected;
        state.connectionStatus.remedy = data.sources.remedy.connected;
        
    } catch (error) {
        console.error('Status check error:', error);
        // Set all as disconnected
        updateConnectionStatus('confluence', false);
        updateConnectionStatus('remedy', false);
    }
}

// Update connection status indicator
function updateConnectionStatus(source, connected) {
    const element = source === 'confluence' ? elements.confluenceStatus : elements.remedyStatus;
    const dot = element.querySelector('.status-dot');
    
    if (connected) {
        dot.classList.add('connected');
    } else {
        dot.classList.remove('connected');
    }
}

// Load saved dark mode preference
function loadSavedPreferences() {
    // Check for saved dark mode preference
    const savedDarkMode = localStorage.getItem('darkMode');
    
    if (savedDarkMode === 'true') {
        // Apply dark mode
        state.darkMode = true;
        document.body.classList.add('dark-theme');
        const themeIcon = elements.themeToggle.querySelector('i');
        themeIcon.classList.remove('fa-moon');
        themeIcon.classList.add('fa-sun');
    }
}

// Initialize the app
function init() {
    setupEventListeners();
    loadSavedPreferences();
    checkSystemStatus();
    console.log('Enterprise Knowledge Hub initialized');
}

// Start the application
init();




app/web/templates/error.html

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Enterprise Knowledge Hub</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="icon" type="image/png" href="{{ url_for('static', filename='img/favicon.png') }}">
    <style>
        .error-container {
            max-width: 600px;
            margin: 0 auto;
            text-align: center;
            padding: var(--spacing-xl);
            background-color: var(--card-color);
            border-radius: var(--border-radius-lg);
            box-shadow: 0 4px 20px var(--shadow-color);
            margin-top: 50px;
        }
        
        .error-icon {
            font-size: 5rem;
            color: var(--danger-color);
            margin-bottom: var(--spacing-lg);
        }
        
        .error-code {
            font-size: 3rem;
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: var(--spacing-md);
        }
        
        .error-message {
            font-size: 1.2rem;
            color: var(--text-secondary);
            margin-bottom: var(--spacing-xl);
        }
        
        .back-button {
            display: inline-block;
            padding: var(--spacing-sm) var(--spacing-lg);
            background-color: var(--primary-color);
            color: white;
            border-radius: var(--border-radius-md);
            text-decoration: none;
            font-weight: 500;
            transition: background-color var(--transition-fast);
        }
        
        .back-button:hover {
            background-color: var(--primary-color-dark);
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Header -->
        <header class="app-header">
            <div class="logo">
                <img src="{{ url_for('static', filename='img/logo.svg') }}" alt="Enterprise Knowledge Hub">
                <h1>Enterprise Knowledge Hub</h1>
            </div>
            <div class="header-actions">
                <button id="theme-toggle" class="icon-button" title="Toggle Theme">
                    <i class="fas fa-moon"></i>
                </button>
            </div>
        </header>

        <!-- Error Content -->
        <main class="app-main">
            <div class="error-container">
                <div class="error-icon">
                    <i class="fas fa-exclamation-circle"></i>
                </div>
                <div class="error-code">{{ error_code }}</div>
                <div class="error-message">{{ error_message }}</div>
                <a href="/" class="back-button">
                    <i class="fas fa-home"></i> Back to Home
                </a>
            </div>
        </main>

        <!-- Footer -->
        <footer class="app-footer">
            <div class="copyright">
                &copy; 2023-2025 Enterprise Knowledge Hub
            </div>
        </footer>
    </div>

    <script>
        // Theme toggle functionality
        const themeToggle = document.getElementById('theme-toggle');
        const themeIcon = themeToggle.querySelector('i');
        
        // Load saved preference
        const savedDarkMode = localStorage.getItem('darkMode');
        
        if (savedDarkMode === 'true') {
            document.body.classList.add('dark-theme');
            themeIcon.classList.remove('fa-moon');
            themeIcon.classList.add('fa-sun');
        }
        
        // Toggle theme
        themeToggle.addEventListener('click', () => {
            const isDarkMode = document.body.classList.toggle('dark-theme');
            
            if (isDarkMode) {
                themeIcon.classList.remove('fa-moon');
                themeIcon.classList.add('fa-sun');
            } else {
                themeIcon.classList.remove('fa-sun');
                themeIcon.classList.add('fa-moon');
            }
            
            localStorage.setItem('darkMode', isDarkMode);
        });
    </script>
</body>
</html>



app/web/templates/about.html

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - {{ app_name }}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="icon" type="image/png" href="{{ url_for('static', filename='img/favicon.png') }}">
    <style>
        .about-container {
            max-width: 800px;
            margin: 0 auto;
            padding: var(--spacing-xl);
            background-color: var(--card-color);
            border-radius: var(--border-radius-lg);
            box-shadow: 0 4px 20px var(--shadow-color);
        }
        
        .section {
            margin-bottom: var(--spacing-xl);
        }
        
        .section h2 {
            font-size: var(--font-size-xl);
            color: var(--primary-color);
            margin-bottom: var(--spacing-md);
            padding-bottom: var(--spacing-sm);
            border-bottom: 1px solid var(--border-color);
        }
        
        .section p {
            margin-bottom: var(--spacing-md);
            line-height: 1.6;
        }
        
        .feature-list {
            list-style-type: none;
            padding: 0;
        }
        
        .feature-item {
            display: flex;
            align-items: flex-start;
            margin-bottom: var(--spacing-md);
        }
        
        .feature-icon {
            color: var(--primary-color);
            font-size: var(--font-size-lg);
            margin-right: var(--spacing-md);
            margin-top: 3px;
        }
        
        .feature-details {
            flex: 1;
        }
        
        .feature-title {
            font-weight: 500;
            margin-bottom: var(--spacing-xs);
        }
        
        .system-info {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: var(--spacing-lg);
            margin-top: var(--spacing-md);
        }
        
        .info-card {
            background-color: var(--background-color);
            padding: var(--spacing-md);
            border-radius: var(--border-radius-md);
            text-align: center;
        }
        
        .info-icon {
            font-size: var(--font-size-2xl);
            color: var(--primary-color);
            margin-bottom: var(--spacing-sm);
        }
        
        .info-value {
            font-size: var(--font-size-xl);
            font-weight: 700;
            margin-bottom: var(--spacing-xs);
        }
        
        .info-label {
            color: var(--text-secondary);
            font-size: var(--font-size-sm);
        }
        
        .action-buttons {
            display: flex;
            justify-content: center;
            gap: var(--spacing-md);
            margin-top: var(--spacing-xl);
        }
        
        .action-button {
            padding: var(--spacing-sm) var(--spacing-lg);
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: var(--border-radius-md);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
            transition: background-color var(--transition-fast);
        }
        
        .action-button:hover {
            background-color: var(--primary-color-dark);
        }
        
        .secondary-action {
            background-color: transparent;
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
        }
        
        .secondary-action:hover {
            background-color: var(--background-color);
            color: var(--text-color);
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Header -->
        <header class="app-header">
            <div class="logo">
                <img src="{{ url_for('static', filename='img/logo.svg') }}" alt="{{ app_name }}">
                <h1>{{ app_name }}</h1>
            </div>
            <div class="header-actions">
                <a href="/" class="icon-button" title="Home">
                    <i class="fas fa-home"></i>
                </a>
                <button id="theme-toggle" class="icon-button" title="Toggle Theme">
                    <i class="fas fa-moon"></i>
                </button>
                <span class="version-info">v{{ app_version }}</span>
            </div>
        </header>

        <!-- Main Content -->
        <main class="app-main">
            <div class="about-container">
                <div class="section">
                    <h2>About Enterprise Knowledge Hub</h2>
                    <p>
                        The Enterprise Knowledge Hub is a comprehensive system that connects to your organization's Confluence and Remedy instances, allowing you to search and retrieve information from both platforms in one place.
                    </p>
                    <p>
                        Using advanced natural language processing and retrieval techniques, the system analyzes your query and finds the most relevant information, generating concise summaries that help you quickly find the answers you need.
                    </p>
                </div>
                
                <div class="section">
                    <h2>Key Features</h2>
                    <ul class="feature-list">
                        <li class="feature-item">
                            <div class="feature-icon">
                                <i class="fas fa-search"></i>
                            </div>
                            <div class="feature-details">
                                <div class="feature-title">Unified Search</div>
                                <div class="feature-description">
                                    Search across both Confluence documentation and Remedy tickets with a single query.
                                </div>
                            </div>
                        </li>
                        <li class="feature-item">
                            <div class="feature-icon">
                                <i class="fas fa-brain"></i>
                            </div>
                            <div class="feature-details">
                                <div class="feature-title">Smart Summaries</div>
                                <div class="feature-description">
                                    Get concise, relevant summaries instead of having to read through multiple documents.
                                </div>
                            </div>
                        </li>
                        <li class="feature-item">
                            <div class="feature-icon">
                                <i class="fas fa-file-alt"></i>
                            </div>
                            <div class="feature-details">
                                <div class="feature-title">Rich Content Support</div>
                                <div class="feature-description">
                                    The system processes text, tables, images, and formatted content for comprehensive results.
                                </div>
                            </div>
                        </li>
                        <li class="feature-item">
                            <div class="feature-icon">
                                <i class="fas fa-language"></i>
                            </div>
                            <div class="feature-details">
                                <div class="feature-title">Natural Language Understanding</div>
                                <div class="feature-description">
                                    Ask questions in plain English and get relevant answers without complex search syntax.
                                </div>
                            </div>
                        </li>
                        <li class="feature-item">
                            <div class="feature-icon">
                                <i class="fas fa-sync-alt"></i>
                            </div>
                            <div class="feature-details">
                                <div class="feature-title">Real-time Updates</div>
                                <div class="feature-description">
                                    Refresh knowledge sources to ensure you always have the latest information.
                                </div>
                            </div>
                        </li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>System Information</h2>
                    <div class="system-info" id="system-info">
                        <!-- System info will be filled by JavaScript -->
                        <div class="info-card">
                            <div class="info-icon">
                                <i class="fas fa-spinner fa-spin"></i>
                            </div>
                            <div class="info-value">Loading...</div>
                            <div class="info-label">Please wait</div>
                        </div>
                    </div>
                </div>
                
                <div class="action-buttons">
                    <a href="/" class="action-button">
                        <i class="fas fa-search"></i> Start Searching
                    </a>
                    <button id="refresh-data" class="action-button secondary-action">
                        <i class="fas fa-sync-alt"></i> Refresh Data
                    </button>
                </div>
            </div>
        </main>

        <!-- Footer -->
        <footer class="app-footer">
            <div class="connection-status">
                <div class="status-indicator" id="confluence-status">
                    <span class="status-dot"></span>
                    <span class="status-name">Confluence</span>
                </div>
                <div class="status-indicator" id="remedy-status">
                    <span class="status-dot"></span>
                    <span class="status-name">Remedy</span>
                </div>
            </div>
            <div class="copyright">
                &copy; 2023-2025 Enterprise Knowledge Hub
            </div>
        </footer>
    </div>

    <!-- Modals -->
    <div class="modal" id="refresh-modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3>Refresh Knowledge Base</h3>
                <span class="close-modal">&times;</span>
            </div>
            <div class="modal-body">
                <p>Select which knowledge sources to refresh:</p>
                <div class="refresh-options">
                    <label class="source-option">
                        <input type="checkbox" id="refresh-confluence" checked>
                        <span class="checkmark"></span>
                        <span class="source-name">Confluence</span>
                    </label>
                    <label class="source-option">
                        <input type="checkbox" id="refresh-remedy" checked>
                        <span class="checkmark"></span>
                        <span class="source-name">Remedy</span>
                    </label>
                </div>
                <div class="modal-options">
                    <label class="source-option">
                        <input type="checkbox" id="force-reload">
                        <span class="checkmark"></span>
                        <span class="source-name">Force full reload</span>
                    </label>
                </div>
            </div>
            <div class="modal-footer">
                <button id="cancel-refresh" class="secondary-button">Cancel</button>
                <button id="confirm-refresh" class="primary-button">Refresh</button>
            </div>
        </div>
    </div>

    <script>
        // DOM elements
        const elements = {
            themeToggle: document.getElementById('theme-toggle'),
            confluenceStatus: document.getElementById('confluence-status'),
            remedyStatus: document.getElementById('remedy-status'),
            systemInfo: document.getElementById('system-info'),
            refreshButton: document.getElementById('refresh-data'),
            refreshModal: document.getElementById('refresh-modal'),
            closeModal: document.querySelector('.close-modal'),
            cancelRefresh: document.getElementById('cancel-refresh'),
            confirmRefresh: document.getElementById('confirm-refresh'),
            refreshConfluence: document.getElementById('refresh-confluence'),
            refreshRemedy: document.getElementById('refresh-remedy'),
            forceReload: document.getElementById('force-reload')
        };
        
        // Theme toggle functionality
        function initThemeToggle() {
            const themeIcon = elements.themeToggle.querySelector('i');
            
            // Load saved preference
            const savedDarkMode = localStorage.getItem('darkMode');
            
            if (savedDarkMode === 'true') {
                document.body.classList.add('dark-theme');
                themeIcon.classList.remove('fa-moon');
                themeIcon.classList.add('fa-sun');
            }
            
            // Toggle theme
            elements.themeToggle.addEventListener('click', () => {
                const isDarkMode = document.body.classList.toggle('dark-theme');
                
                if (isDarkMode) {
                    themeIcon.classList.remove('fa-moon');
                    themeIcon.classList.add('fa-sun');
                } else {
                    themeIcon.classList.remove('fa-sun');
                    themeIcon.classList.add('fa-moon');
                }
                
                localStorage.setItem('darkMode', isDarkMode);
            });
        }
        
        // Update connection status
        function updateConnectionStatus(source, connected) {
            const element = source === 'confluence' ? elements.confluenceStatus : elements.remedyStatus;
            const dot = element.querySelector('.status-dot');
            
            if (connected) {
                dot.classList.add('connected');
            } else {
                dot.classList.remove('connected');
            }
        }
        
        // Load system information
        async function loadSystemInfo() {
            try {
                const response = await fetch('/api/status');
                
                if (!response.ok) {
                    throw new Error(`HTTP error ${response.status}`);
                }
                
                const data = await response.json();
                
                // Update connection status
                updateConnectionStatus('confluence', data.sources.confluence.connected);
                updateConnectionStatus('remedy', data.sources.remedy.connected);
                
                // Clear current info
                elements.systemInfo.innerHTML = '';
                
                // Add version info
                const versionCard = createInfoCard('fa-code-branch', data.app_version, 'Version');
                elements.systemInfo.appendChild(versionCard);
                
                // Add total documents
                const totalDocs = data.documents_loaded.total;
                const docsCard = createInfoCard('fa-file-alt', totalDocs, 'Total Documents');
                elements.systemInfo.appendChild(docsCard);
                
                // Add Confluence documents
                const confluenceDocs = data.documents_loaded.confluence;
                const confluenceCard = createInfoCard('fa-book', confluenceDocs, 'Confluence Documents');
                elements.systemInfo.appendChild(confluenceCard);
                
                // Add Remedy documents
                const remedyDocs = data.documents_loaded.remedy;
                const remedyCard = createInfoCard('fa-ticket-alt', remedyDocs, 'Remedy Documents');
                elements.systemInfo.appendChild(remedyCard);
                
            } catch (error) {
                console.error('Error loading system info:', error);
                elements.systemInfo.innerHTML = '<div class="error-message">Error loading system information</div>';
            }
        }
        
        // Create an info card
        function createInfoCard(iconClass, value, label) {
            const card = document.createElement('div');
            card.className = 'info-card';
            
            const icon = document.createElement('div');
            icon.className = 'info-icon';
            icon.innerHTML = `<i class="fas ${iconClass}"></i>`;
            
            const valueElement = document.createElement('div');
            valueElement.className = 'info-value';
            valueElement.textContent = value;
            
            const labelElement = document.createElement('div');
            labelElement.className = 'info-label';
            labelElement.textContent = label;
            
            card.appendChild(icon);
            card.appendChild(valueElement);
            card.appendChild(labelElement);
            
            return card;
        }
        
        // Handle refresh functionality
        function initRefreshFunctionality() {
            // Open modal
            elements.refreshButton.addEventListener('click', () => {
                elements.refreshModal.style.display = 'block';
            });
            
            // Close modal
            elements.closeModal.addEventListener('click', () => {
                elements.refreshModal.style.display = 'none';
            });
            
            elements.cancelRefresh.addEventListener('click', () => {
                elements.refreshModal.style.display = 'none';
            });
            
            // Close modal when clicking outside
            window.addEventListener('click', (e) => {
                if (e.target === elements.refreshModal) {
                    elements.refreshModal.style.display = 'none';
                }
            });
            
            // Handle refresh confirmation
            elements.confirmRefresh.addEventListener('click', handleRefresh);
        }
        
        // Handle refresh action
        async function handleRefresh() {
            // Get selected sources
            const sources = [];
            if (elements.refreshConfluence.checked) sources.push('confluence');
            if (elements.refreshRemedy.checked) sources.push('remedy');
            
            if (sources.length === 0) {
                alert('Please select at least one source to refresh');
                return;
            }
            
            // Show loading in button
            elements.confirmRefresh.textContent = 'Refreshing...';
            elements.confirmRefresh.disabled = true;
            elements.cancelRefresh.disabled = true;
            
            try {
                // For each selected source, send refresh request
                const refreshPromises = sources.map(source => {
                    return fetch('/api/refresh', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            source: source,
                            force_reload: elements.forceReload.checked
                        })
                    }).then(res => res.json());
                });
                
                // Wait for all refreshes to complete
                const results = await Promise.all(refreshPromises);
                
                // Display result
                let successCount = results.filter(r => r.success).length;
                
                if (successCount === sources.length) {
                    alert(`Successfully refreshed ${sources.join(' and ')} data.`);
                } else {
                    alert('Some refresh operations failed. Please check the console for details.');
                    console.error('Refresh results:', results);
                }
                
                // Reload system info
                await loadSystemInfo();
                
            } catch (error) {
                console.error('Refresh error:', error);
                alert('An error occurred during refresh. Please try again.');
            } finally {
                // Reset modal
                elements.confirmRefresh.textContent = 'Refresh';
                elements.confirmRefresh.disabled = false;
                elements.cancelRefresh.disabled = false;
                elements.refreshModal.style.display = 'none';
            }
        }
        
        // Initialize the page
        function initPage() {
            initThemeToggle();
            loadSystemInfo();
            initRefreshFunctionality();
        }
        
        // Start when DOM is loaded
        document.addEventListener('DOMContentLoaded', initPage);
    </script>
</body>
</html>


app/web/static/img/logo.svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <!-- Background Circle -->
  <circle cx="100" cy="100" r="90" fill="#4a6cf7" opacity="0.1"/>
  
  <!-- Main Components -->
  <g fill="#4a6cf7">
    <!-- Data Nodes -->
    <circle cx="90" cy="60" r="12"/>
    <circle cx="140" cy="90" r="12"/>
    <circle cx="110" cy="140" r="12"/>
    <circle cx="60" cy="110" r="12"/>
    
    <!-- Center Node -->
    <circle cx="100" cy="100" r="20"/>
    
    <!-- Connection Lines -->
    <line x1="90" y1="60" x2="100" y2="100" stroke="#4a6cf7" stroke-width="4"/>
    <line x1="140" y1="90" x2="100" y2="100" stroke="#4a6cf7" stroke-width="4"/>
    <line x1="110" y1="140" x2="100" y2="100" stroke="#4a6cf7" stroke-width="4"/>
    <line x1="60" y1="110" x2="100" y2="100" stroke="#4a6cf7" stroke-width="4"/>
  </g>
  
  <!-- Knowledge Symbols -->
  <g fill="white">
    <!-- Document Icon in Node 1 -->
    <path d="M90,56 L84,56 L84,64 L96,64 L96,56 L90,56 Z M94,62 L86,62 L86,58 L94,58 Z"/>
    
    <!-- Database Icon in Node 2 -->
    <path d="M140,86 C136,86 136,94 140,94 C144,94 144,86 140,86 Z M140,88 C142,88 142,92 140,92 C138,92 138,88 140,88 Z"/>
    
    <!-- Search Icon in Center -->
    <path d="M105,95 L103,97 L107,101 L109,99 Z M100,95 C98,95 96,97 96,100 C96,103 98,105 100,105 C102,105 104,103 104,100 C104,97 102,95 100,95 Z M100,97 C101,97 102,98 102,100 C102,102 101,103 100,103 C99,103 98,102 98,100 C98,98 99,97 100,97 Z"/>
    
    <!-- Ticket Icon in Node 3 -->
    <path d="M110,136 L106,136 L106,144 L114,144 L114,136 L110,136 Z M112,142 L108,142 L108,138 L112,138 Z"/>
    
    <!-- Book Icon in Node 4 -->
    <path d="M60,106 L56,106 L56,114 L64,114 L64,106 L60,106 Z M62,112 L58,112 L58,108 L62,108 Z"/>
  </g>
</svg>








app/web/static/img/favicon.svg


<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
  <!-- Background -->
  <circle cx="16" cy="16" r="16" fill="#4a6cf7"/>
  
  <!-- Knowledge Hub Icon -->
  <g fill="white">
    <!-- Center Node -->
    <circle cx="16" cy="16" r="4"/>
    
    <!-- Satellite Nodes -->
    <circle cx="16" cy="7" r="2"/>
    <circle cx="25" cy="16" r="2"/>
    <circle cx="16" cy="25" r="2"/>
    <circle cx="7" cy="16" r="2"/>
    
    <!-- Connection Lines -->
    <line x1="16" y1="7" x2="16" y2="12" stroke="white" stroke-width="1.5"/>
    <line x1="25" y1="16" x2="20" y2="16" stroke="white" stroke-width="1.5"/>
    <line x1="16" y1="25" x2="16" y2="20" stroke="white" stroke-width="1.5"/>
    <line x1="7" y1="16" x2="12" y2="16" stroke="white" stroke-width="1.5"/>
  </g>
</svg>





.env.example


# Application settings
DEBUG=False
PORT=5000
HOST=0.0.0.0

# Confluence API settings
CONFLUENCE_URL=https://your-company.atlassian.net
CONFLUENCE_USERNAME=your-email@company.com
CONFLUENCE_API_TOKEN=your-api-token
CONFLUENCE_SPACE_KEY=TEAM

# Remedy API settings
REMEDY_SERVER=cmegroup-restapi.onbmc.com
REMEDY_USERNAME=your-username
REMEDY_PASSWORD=your-password

# Processing settings
MAX_CACHE_AGE_DAYS=7
CHUNK_SIZE=300
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.6

# NLP settings
EMBEDDING_MODEL=all-MiniLM-L6-v2



run.py 

#!/usr/bin/env python
"""
Run script for the Enterprise Knowledge Hub.
"""
import os
import sys
import logging
from app.main import main

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )
    
    # Run the application
    main()
