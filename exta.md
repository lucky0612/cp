jira-confluence-chatbot/
├── backend/
│   ├── app.py                   # Flask application server
│   ├── config.py                # Configuration settings
│   ├── confluence_client.py     # Adapted from provided script
│   ├── jira_client.py           # Adapted from provided script
│   ├── gemini_client.py         # Shared Gemini AI integration
│   ├── utils.py                 # Utility functions
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── index.html               # Main HTML file
│   ├── css/
│   │   └── styles.css           # Main CSS styles
│   ├── js/
│   │   ├── main.js              # Main JavaScript file
│   │   ├── chat.js              # Chat functionality
│   │   └── api.js               # API communication
│   └── assets/
│       ├── images/              # Images for the UI
│       └── favicon.ico          # Website favicon
└── README.md                    # Project documentation















# Backend Configuration

```python
import os
from pathlib import Path

# Base configuration
DEBUG = os.environ.get("DEBUG", "True").lower() == "true"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000))

# Cache settings
CACHE_DIR = os.environ.get("CACHE_DIR", "./cache")
CACHE_EXPIRY_DAYS = int(os.environ.get("CACHE_EXPIRY_DAYS", "7"))

# Google Gemini AI settings
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")

# Confluence settings
CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL", "https://your-company.atlassian.net")
CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME", "your.email@company.com")
CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN", "your-api-token")

# Jira settings
JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://jira.yourcompany.com")
JIRA_USERNAME = os.environ.get("JIRA_USERNAME", "your_username")
JIRA_TOKEN = os.environ.get("JIRA_TOKEN", "your_api_token")

# Search settings
MAX_RESULTS_PER_QUERY = int(os.environ.get("MAX_RESULTS_PER_QUERY", "50"))

# Create cache directory
os.makedirs(Path(CACHE_DIR), exist_ok=True)
```















# Shared Gemini AI Client

```python
import logging
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.api_core.exceptions import GoogleAPICallError
from config import PROJECT_ID, REGION, MODEL_NAME

logger = logging.getLogger("GeminiClient")

class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
        logger.info(f"Initialized Gemini client with model {MODEL_NAME}")
    
    def generate_response(self, prompt, system_prompt=None, temperature=0.7, max_tokens=8192, stream=True):
        """
        Generate a response from Gemini.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated response text or generator if streaming
        """
        try:
            logger.info(f"Generating response from Gemini with temperature {temperature}")
            logger.info(f"System prompt length: {len(system_prompt) if system_prompt else 0}")
            logger.info(f"User prompt length: {len(prompt)}")
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=max_tokens,
            )
            
            # Build the full prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            if stream:
                # Return a generator for streaming
                return self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                    stream=True,
                )
            else:
                # Generate response without streaming
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                )
                
                if response.candidates and response.candidates[0].text:
                    return response.candidates[0].text
                return ""
                
        except GoogleAPICallError as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"
    
    def build_system_prompt(self, source_type="combined", context_docs=None):
        """
        Build a comprehensive system prompt for Gemini.
        
        Args:
            source_type: Type of data source ("jira", "confluence", or "combined")
            context_docs: List of documents to include as context
            
        Returns:
            System prompt string
        """
        if source_type == "jira":
            system_prompt = """You are a helpful AI assistant specializing in providing information from Jira. Your responses should be:

1. PRECISE AND ACCURATE: Always base your answers strictly on the provided Jira data. If the information isn't in the provided context, clearly state that you don't have that specific information.

2. PROFESSIONAL BUT FRIENDLY: Maintain a professional tone that would be appropriate in a corporate environment, while still being approachable and helpful.

3. WELL-STRUCTURED: For complex responses, use appropriate formatting with headings, bullet points, or numbered lists to improve readability.

4. CONTEXTUALLY AWARE: If you need clarification to provide a better answer, politely ask follow-up questions to better understand the user's needs.

5. SOURCE-TRANSPARENT: Always include references to the specific Jira tickets you used to answer the question.

When responding to ticket questions:
- Be precise with technical terminology
- Include code snippets when relevant
- Explain complex concepts clearly

If you don't have enough information to fully answer the question:
- Clearly state what you do know based on the provided context
- Explain what additional information would be needed
"""
        elif source_type == "confluence":
            system_prompt = """You are a highly knowledgeable and professional AI assistant that specializes in providing accurate information from a company's Confluence knowledge base. Your responses should be:

1. PRECISE AND ACCURATE: Always base your answers strictly on the provided Confluence documents. If the information isn't in the provided context, clearly state that you don't have that specific information.

2. PROFESSIONAL BUT FRIENDLY: Maintain a professional tone that would be appropriate in a corporate environment, while still being approachable and helpful.

3. WELL-STRUCTURED: For complex responses, use appropriate formatting with headings, bullet points, or numbered lists to improve readability.

4. CONTEXTUALLY AWARE: If you need clarification to provide a better answer, politely ask follow-up questions to better understand the user's needs.

5. SOURCE-TRANSPARENT: Always include references to the specific Confluence pages you used to answer the question. Include the exact page titles and URLs at the end of your response.

When data or tables are mentioned in the context:
- Present numerical data clearly, using tables if appropriate
- Explain what the data means in business terms

If you don't have enough information to fully answer the question:
- Clearly state what you do know based on the provided context
- Explain what additional information would be needed
- Suggest which Confluence spaces might contain the relevant information
"""
        else:  # combined
            system_prompt = """You are an intelligent AI assistant with access to both Jira tickets and Confluence documentation. Your role is to provide comprehensive and accurate information by combining data from both sources. Your responses should be:

1. PRECISE AND ACCURATE: Always base your answers strictly on the provided Jira and Confluence data. If the information isn't in the provided context, clearly state that limitation.

2. PROFESSIONAL BUT FRIENDLY: Maintain a professional tone that would be appropriate in a corporate environment, while still being approachable and helpful.

3. WELL-STRUCTURED: For complex responses, use appropriate formatting with headings, bullet points, or numbered lists to improve readability.

4. CONTEXTUALLY AWARE: Connect information between Jira tickets and Confluence pages when relevant. Look for solutions in Confluence that might address issues in Jira tickets.

5. SOURCE-TRANSPARENT: Clearly cite when information comes from Jira tickets versus Confluence pages. Include specific references to both sources.

Special capabilities:
- When asked about an issue, you can provide both the issue details from Jira and potential solutions from Confluence
- You can suggest Confluence documentation that might help resolve specific Jira tickets
- You can explain complex technical concepts documented in Confluence to help with Jira tickets

If you don't have enough information:
- Clearly state what you do know based on the provided context
- Explain what additional information would be helpful
- Suggest where that information might be found in either system
"""
        
        # Add context documents if provided
        if context_docs and len(context_docs) > 0:
            context_text = "\n\n### CONTEXT DOCUMENTS ###\n\n"
            
            for i, doc in enumerate(context_docs):
                doc_type = doc.get("source_type", "document")
                title = doc.get("metadata", {}).get("title", f"Document {i+1}")
                url = doc.get("metadata", {}).get("url", "")
                content = doc.get("content", "")
                
                context_text += f"[{doc_type.upper()} {i+1}]: {title}\n"
                context_text += f"URL: {url}\n"
                context_text += f"CONTENT: {content}\n\n"
            
            system_prompt += context_text
        
        return system_prompt

    def build_query_for_solution(self, jira_ticket, user_query=None):
        """
        Build a query to find solutions in Confluence based on a Jira ticket.
        
        Args:
            jira_ticket: Jira ticket data
            user_query: Optional user query for additional context
            
        Returns:
            Generated query for finding solutions
        """
        # Extract key information from the ticket
        metadata = jira_ticket.get("metadata", {})
        ticket_key = metadata.get("key", "")
        summary = metadata.get("summary", "")
        issue_type = metadata.get("issuetype", "")
        content = jira_ticket.get("content", "")
        
        # Build a prompt for Gemini to generate a search query
        prompt = f"""Based on the following Jira ticket, create a concise search query (3-6 keywords) to find relevant solutions or documentation in Confluence:

Ticket Key: {ticket_key}
Summary: {summary}
Issue Type: {issue_type}
Content:
{content[:1000]}  # Limit content to avoid overwhelming the model

Additional user query: {user_query if user_query else ""}

Output only the search query terms without any explanation or formatting. For example: "database connection timeout error" or "Jenkins pipeline deployment failure".
"""
        
        # Configure for a concise response
        generation_config = GenerationConfig(
            temperature=0.3,
            top_p=0.95,
            max_output_tokens=50,
        )
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            if response.candidates and response.candidates[0].text:
                search_query = response.candidates[0].text.strip()
                logger.info(f"Generated search query: {search_query}")
                return search_query
            return f"{summary} {issue_type}"
        except Exception as e:
            logger.error(f"Error generating search query: {str(e)}")
            # Fallback to using summary and issue type
            return f"{summary} {issue_type}"
```














# Utility Functions

```python
import json
import logging
import pickle
import hashlib
import os
import time
import re
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logger = logging.getLogger("Utils")

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
        self.text = re.sub(r'<[^>]*>', ' ', data)
        # Replace multiple whitespace with single space
        self.text = re.sub(r'\s+', ' ', self.text).strip()


class Cache:
    """A simple disk-based cache system."""
    
    def __init__(self, cache_dir, expiry_days=7):
        """
        Initialize the cache system.
        
        Args:
            cache_dir: Directory to store cache files
            expiry_days: Number of days after which cache expires
        """
        self.cache_dir = Path(cache_dir)
        self.expiry_seconds = expiry_days * 24 * 60 * 60
        self.memory_cache = {}
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized cache in {self.cache_dir} with {expiry_days} day expiry")
    
    def get_cache_path(self, key):
        """Get the file path for a cache item."""
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed_key}.pickle"
    
    def is_cache_valid(self, key):
        """Check if a cache item exists and is still valid (not expired)."""
        cache_path = self.get_cache_path(key)
        
        if key in self.memory_cache:
            return True
        
        if cache_path.exists():
            # Check if the cache file is recent enough
            mtime = cache_path.stat().st_mtime
            if time.time() - mtime <= self.expiry_seconds:
                return True
        
        return False
    
    def get(self, key):
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if not self.is_cache_valid(key):
            return None
        
        # Try memory cache first
        if key in self.memory_cache:
            logger.info(f"Cache hit (memory): {key}")
            return self.memory_cache[key]
        
        # Then try file cache
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
                # Store in memory cache for faster future access
                self.memory_cache[key] = value
                logger.info(f"Cache hit (file): {key}")
                return value
        except Exception as e:
            logger.error(f"Error reading cache for {key}: {str(e)}")
            return None
    
    def set(self, key, value):
        """
        Store a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Store in memory cache
        self.memory_cache[key] = value
        
        # Store in file cache
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logger.info(f"Cached: {key}")
        except Exception as e:
            logger.error(f"Error writing cache for {key}: {str(e)}")
    
    def clear(self, key=None):
        """
        Clear cache items.
        
        Args:
            key: Specific key to clear, or None to clear all
        """
        if key:
            # Clear specific key
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            cache_path = self.get_cache_path(key)
            if cache_path.exists():
                os.remove(cache_path)
            logger.info(f"Cleared cache for {key}")
        else:
            # Clear all cache
            self.memory_cache = {}
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pickle'):
                    os.remove(os.path.join(self.cache_dir, file))
            logger.info("Cleared all cache")


class TextProcessor:
    """Class for text processing and analysis."""
    
    def __init__(self):
        """Initialize text processing components."""
        # Common stop words
        self.stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                          'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'against', 
                          'between', 'into', 'through', 'during', 'before', 'after', 'above',
                          'below', 'from', 'up', 'down', 'of', 'off', 'over', 'under', 'again',
                          'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                          'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                          'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                          'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
        
        logger.info("Initialized text processor")
    
    def tokenize(self, text):
        """Simple tokenization by splitting on whitespace and removing punctuation."""
        if not text or not isinstance(text, str):
            return []
        
        # Remove punctuation and lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split on whitespace
        tokens = text.split()
        return tokens
    
    def remove_stop_words(self, tokens):
        """Remove stop words from a list of tokens."""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_word(self, word):
        """Very simple stemming (just removes common endings)."""
        if len(word) < 4:
            return word
            
        suffixes = ['ing', 'ed', 's', 'es', 'ly', 'ment', 'ness', 'ity', 'tion']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    def preprocess_text(self, text):
        """
        Preprocess text by tokenizing, removing stop words, and stemming.
        
        Args:
            text: Input text to process
            
        Returns:
            Dict containing original, tokens, filtered_tokens, stemmed_tokens
        """
        if not text or not isinstance(text, str):
            return {
                "original": "",
                "tokens": [],
                "filtered_tokens": [],
                "stemmed_tokens": [],
                "segments": []
            }
        
        # Segment text into sentences (simple split on period followed by space)
        segments = re.split(r'\.[\s\n]+', text)
        segments = [s.strip() for s in segments if s.strip()]
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stop words
        filtered_tokens = self.remove_stop_words(tokens)
        
        # Stem tokens
        stemmed_tokens = [self.stem_word(token) for token in filtered_tokens]
        
        return {
            "original": text,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "stemmed_tokens": stemmed_tokens,
            "segments": segments
        }
    
    def extract_keywords(self, text, top_n=10):
        """
        Extract the most important keywords from text.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        processed = self.preprocess_text(text)
        
        # Count word frequencies
        word_freq = {}
        for token in processed["stemmed_tokens"]:
            if token in word_freq:
                word_freq[token] += 1
            else:
                word_freq[token] = 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N keywords
        return [word for word, freq in sorted_words[:top_n]]
```
















# Adapted Confluence Client

```python
import requests
import logging
import json
import re
from urllib.parse import quote
from config import CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN, MAX_RESULTS_PER_QUERY
from utils import HTMLFilter, Cache, TextProcessor

# Configure logging
logger = logging.getLogger("ConfluenceClient")

class ConfluenceClient:
    """Client for Confluence REST API operations with comprehensive error handling."""
    
    def __init__(self, base_url=CONFLUENCE_URL, username=CONFLUENCE_USERNAME, api_token=CONFLUENCE_API_TOKEN, cache=None):
        """
        Initialize the Confluence client with authentication details.
        
        Args:
            base_url: The base URL of the Confluence instance (e.g., https://company.atlassian.net)
            username: The username for authentication
            api_token: The API token for authentication
            cache: Cache object for storing data
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.session = requests.Session()
        self.cache = cache or Cache("./confluence_cache")
        self.text_processor = TextProcessor()
        logger.info(f"Initialized Confluence client for {self.base_url}")
    
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            response = self.session.get(
                f"{self.base_url}/rest/api/content",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                verify=False  # Setting SSL verify to False as requested
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response content (first 500 chars): {raw_content[:500]}...")
            
            # Handle empty response
            if not raw_content.strip():
                logger.warning("Empty response received during connection test")
                return True  # Still consider it a success if status code is OK
            
            try:
                response.json()
                logger.info("Connection successful!")
                return True
            except json.JSONDecodeError as e:
                logger.error(f"Response content: {raw_content}")
                return False
        except json.JSONDecodeError as e:
            logger.error(f"Connection successful but received invalid JSON: {str(e)}")
            logger.error(f"Response content: {raw_content}")
            return False
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return False
    
    def get_content_by_id(self, content_id, expand=None):
        """
        Get content by ID with optional expansion parameters.
        
        Args:
            content_id: The ID of the content to retrieve
            expand: Comma-separated list of properties to expand (e.g., "body.storage,version,space")
        """
        try:
            # Check cache first
            cache_key = f"confluence_content_{content_id}_{expand}"
            cached_content = self.cache.get(cache_key)
            if cached_content:
                return cached_content
            
            params = {}
            if expand:
                params["expand"] = expand
            
            logger.info(f"Fetching content with ID: {content_id}")
            response = self.session.get(
                f"{self.base_url}/rest/api/content/{content_id}",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            
            # Handle empty response
            if not response.text.strip():
                logger.warning("Empty response received when retrieving content")
                return None
            
            content = response.json()
            logger.info(f"Successfully retrieved content: {content.get('title', 'Unknown title')}")
            
            # Cache the result
            self.cache.set(cache_key, content)
            
            return content
        except Exception as e:
            logger.error(f"Failed to get content by ID {content_id}: {str(e)}")
            return None
    
    def get_page_content(self, page_id):
        """
        Get the content of a page in a suitable format for NLP.
        This extracts and processes the content to be more suitable for embeddings.
        
        Args:
            page_id: The ID of the page
        """
        try:
            # Check cache first
            cache_key = f"confluence_page_content_{page_id}"
            cached_content = self.cache.get(cache_key)
            if cached_content:
                return cached_content
            
            page = self.get_content_by_id(page_id, expand="body.storage,metadata.labels")
            if not page:
                return None
            
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "url": f"{self.base_url}/wiki/spaces/{page.get('_expandable', {}).get('space', '').split('/')[-1]}/pages/{page.get('id')}",
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
            }
            
            # Get raw content
            content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process HTML content to plain text
            html_filter = HTMLFilter()
            html_filter.feed(content)
            plain_text = html_filter.text
            
            result = {
                "source_type": "confluence",
                "metadata": metadata,
                "content": plain_text,
                "raw_html": content
            }
            
            # Cache the result
            self.cache.set(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None
    
    def search_content(self, query, content_type="page", limit=MAX_RESULTS_PER_QUERY):
        """
        Search for content using CQL or specific parameters.
        
        Args:
            query: Search query
            content_type: Type of content to search for (default: page)
            limit: Maximum number of results to return
            
        Returns:
            List of content items matching the query
        """
        try:
            # Check cache first
            cache_key = f"confluence_search_{content_type}_{query}_{limit}"
            cached_results = self.cache.get(cache_key)
            if cached_results:
                return cached_results
            
            # First try an exact match search
            logger.info(f"Searching Confluence for: {query}")
            
            # Extract keywords for better search
            keywords = self.text_processor.extract_keywords(query)
            
            # Create search terms
            search_terms = []
            
            # Add the exact query if it's short enough
            if len(query.split()) <= 6:
                safe_query = query.replace('"', '\\"')
                search_terms.append(f'text ~ "{safe_query}"')
            
            # Add keyword-based search
            if keywords:
                keyword_query = " AND ".join([f'text ~ "{keyword}"' for keyword in keywords[:5]])
                search_terms.append(f"({keyword_query})")
            
            # Final CQL
            cql = " OR ".join(search_terms)
            if content_type:
                cql = f"type={content_type} AND ({cql})"
                
            params = {
                "cql": cql,
                "limit": limit,
                "expand": "space"
            }
            
            logger.info(f"Searching with CQL: {cql}")
            response = self.session.get(
                f"{self.base_url}/rest/api/content/search",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            
            results = response.json()
            logger.info(f"Search returned {len(results.get('results', []))} results")
            
            # Process results to get full content
            processed_results = []
            for result in results.get("results", []):
                content_id = result.get("id")
                
                if not content_id:
                    continue
                    
                processed = self.get_page_content(content_id)
                if processed:
                    processed_results.append(processed)
            
            # Cache the results
            self.cache.set(cache_key, processed_results)
            
            return processed_results
        except Exception as e:
            logger.error(f"Failed to search content: {str(e)}")
            return []
```
















# Adapted Jira Client

```python
import requests
import logging
import json
import re
import tempfile
import os
from urllib.parse import quote
from config import JIRA_BASE_URL, JIRA_USERNAME, JIRA_TOKEN, MAX_RESULTS_PER_QUERY
from utils import Cache, TextProcessor

# Configure logging
logger = logging.getLogger("JiraClient")

class JiraClient:
    """Client for interacting with Jira API."""
    
    def __init__(self, base_url=JIRA_BASE_URL, username=JIRA_USERNAME, token=JIRA_TOKEN, cache=None):
        """
        Initialize the Jira client.
        
        Args:
            base_url: Base URL of the Jira instance
            username: Jira username or email
            token: Jira API token
            cache: Cache object for storing Jira data
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, token)
        self.cache = cache or Cache("./jira_cache")
        self.text_processor = TextProcessor()
        self.session = requests.Session()
        self.session.auth = self.auth
        # Disable SSL verification as requested
        self.session.verify = False
        logger.info(f"Jira client initialized for {base_url}")
        
    def _make_request(self, method, endpoint, params=None, data=None, headers=None, use_cache=True):
        """
        Make a request to the Jira API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            params: Query parameters
            data: Request body data
            headers: HTTP headers
            use_cache: Whether to use cache for GET requests
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        all_headers = {**default_headers, **(headers or {})}
        
        # Try to get from cache for GET requests
        if method.upper() == 'GET' and use_cache:
            cache_key = f"jira_{method}_{url}_{str(params)}"
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        try:
            logger.info(f"Making {method} request to {url}")
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=all_headers,
                verify=False  # Disable SSL verification
            )
            response.raise_for_status()
            
            if response.status_code == 204:  # No content
                result = {}
            else:
                result = response.json()
                
            # Cache the result for GET requests
            if method.upper() == 'GET' and use_cache:
                cache_key = f"jira_{method}_{url}_{str(params)}"
                self.cache.set(cache_key, result)
                
            return result
        except requests.RequestException as e:
            logger.error(f"Error making request to Jira: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response text: {e.response.text}")
            raise
            
    def test_connection(self):
        """Test the connection to Jira.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Use the /serverInfo endpoint to test connection
            info = self._make_request('GET', '/rest/api/2/serverInfo', use_cache=False)
            logger.info(f"Successfully connected to Jira server version: {info.get('version', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Jira: {str(e)}")
            return False
            
    def get_issue(self, issue_key, fields=None):
        """Get a specific issue by its key.
        
        Args:
            issue_key: The Jira issue key (e.g., PROJ-1)
            fields: Comma-separated list of fields to include
            
        Returns:
            Issue details as dictionary
        """
        params = {}
        if fields:
            params['fields'] = fields
            
        return self._make_request('GET', f'/rest/api/2/issue/{issue_key}', params=params)
        
    def search_issues(self, jql, start_at=0, max_results=50, fields=None, expand=None):
        """Search for issues using JQL.
        
        Args:
            jql: JQL search string
            start_at: Index of the first result to return
            max_results: Maximum number of results to return
            fields: Comma-separated list of fields to include
            expand: Additional information to expand
            
        Returns:
            Search results as dictionary
        """
        params = {
            'jql': jql,
            'startAt': start_at,
            'maxResults': max_results
        }
        
        if fields:
            params['fields'] = fields
            
        if expand:
            params['expand'] = expand
            
        return self._make_request('GET', '/rest/api/2/search', params=params)
        
    def get_all_issues(self, jql, fields=None, max_results=1000):
        """Get all issues matching a JQL query, handling pagination.
        
        Args:
            jql: JQL search string
            fields: Comma-separated list of fields to include
            max_results: Maximum total number of results to return
            
        Returns:
            List of all matching issues
        """
        # Check cache first
        cache_key = f"jira_all_issues_{jql}_{fields}_{max_results}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
            
        issues = []
        page_size = 100  # Jira recommends 100 for optimal performance
        start_at = 0
        
        logger.info(f"Retrieving all issues matching JQL: {jql}")
        
        while True:
            page = self.search_issues(
                jql=jql,
                start_at=start_at,
                max_results=min(page_size, max_results - len(issues)),
                fields=fields
            )
            
            if not page.get('issues'):
                break
                
            issues.extend(page['issues'])
            logger.info(f"Retrieved {len(issues)} of {page['total']} issues")
            
            # Check if we've reached the total or our max limit
            if len(issues) >= page['total'] or len(issues) >= max_results:
                break
                
            # Get next page
            start_at += len(page['issues'])
            
            # If no issues were returned, we're done
            if not page['issues']:
                break
                
        logger.info(f"Retrieved a total of {len(issues)} issues")
        
        # Cache the results
        self.cache.set(cache_key, issues)
        
        return issues
        
    def get_issue_types(self):
        """Get all issue types defined in the Jira instance."""
        return self._make_request('GET', '/rest/api/2/issuetype')
        
    def get_projects(self):
        """Get all projects visible to the authenticated user."""
        return self._make_request('GET', '/rest/api/2/project')
        
    def _extract_text_from_adf(self, adf_doc):
        """Extract plain text from Atlassian Document Format (ADF) object.
        
        This is a simplified parser that doesn't handle all ADF features.
        
        Args:
            adf_doc: ADF document object
            
        Returns:
            Extracted plain text
        """
        if not adf_doc or not isinstance(adf_doc, dict):
            return ""
            
        text_parts = []
        
        def extract_from_content(content_list):
            parts = []
            if not content_list or not isinstance(content_list, list):
                return ""
                
            for item in content_list:
                if not isinstance(item, dict):
                    continue
                    
                # Extract text nodes
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                    
                # Extract from content recursively
                if "content" in item and isinstance(item["content"], list):
                    parts.append(extract_from_content(item["content"]))
                    
            return " ".join(parts)
            
        # Extract text from the main content array
        if "content" in adf_doc and isinstance(adf_doc["content"], list):
            text_parts.append(extract_from_content(adf_doc["content"]))
            
        # Extract text from the version array if it exists
        if "version" in adf_doc:
            text_parts.append(f"Document version: {adf_doc['version']}")
            
        return " ".join(text_parts)
    
    def get_issue_content(self, issue_key):
        """Get the full content of an issue in a format suitable for the chatbot.
        
        This method enriches the issue data with additional information like
        comments, attachments, and other related data.
        
        Args:
            issue_key: The Jira issue key (e.g., PROJ-1)
            
        Returns:
            Enriched issue data as dictionary
        """
        # Check cache first
        cache_key = f"jira_issue_content_{issue_key}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Get issue with all fields and expansion
        issue = self.get_issue(issue_key=issue_key, fields="*all")
        
        # Extract key metadata
        metadata = {
            "key": issue.get("key"),
            "summary": issue.get("fields", {}).get("summary"),
            "issuetype": issue.get("fields", {}).get("issuetype", {}).get("name"),
            "status": issue.get("fields", {}).get("status", {}).get("name"),
            "created": issue.get("fields", {}).get("created"),
            "updated": issue.get("fields", {}).get("updated"),
        }
        
        # Extract people
        if "assignee" in issue.get("fields", {}):
            metadata["assignee"] = issue.get("fields", {}).get("assignee", {}).get("displayName", "Unassigned")
            
        if "reporter" in issue.get("fields", {}):
            metadata["reporter"] = issue.get("fields", {}).get("reporter", {}).get("displayName", "Unknown")
            
        # Extract content fields
        content_parts = []
        
        # Add summary
        summary = issue.get("fields", {}).get("summary", "")
        if summary:
            content_parts.append(f"Summary: {summary}")
            
        # Add description
        # Note: This is simplified and assumes the description is in plain text
        # In reality, Jira descriptions can be in various formats
        description = issue.get("fields", {}).get("description", "")
        if description:
            # Try to extract text from Atlassian Document Format (ADF)
            if isinstance(description, dict):
                # This is a simplified ADF parser, might need enhancement
                description_text = self._extract_text_from_adf(description)
                content_parts.append(f"Description: {description_text}")
            else:
                content_parts.append(f"Description: {description}")
                
        # Add comments
        comments = issue.get("fields", {}).get("comment", {}).get("comments", [])
        for comment in comments:
            author = comment.get("author", {}).get("displayName", "Unknown")
            created = comment.get("created", "")
            # Similar to description, handle ADF format
            comment_body = comment.get("body", "")
            if isinstance(comment_body, dict):
                comment_text = self._extract_text_from_adf(comment_body)
                content_parts.append(f"Comment by {author} on {created}: {comment_text}")
            else:
                content_parts.append(f"Comment by {author} on {created}: {comment_body}")
                
        # Combine all content
        full_content = "\n\n".join(content_parts)
        
        result = {
            "source_type": "jira",
            "metadata": metadata,
            "content": full_content,
            "url": f"{self.base_url}/browse/{issue_key}"
        }
        
        # Cache the result
        self.cache.set(cache_key, result)
        
        return result
    
    def search_issues_by_query(self, query, max_results=MAX_RESULTS_PER_QUERY):
        """
        Search for issues based on a natural language query.
        
        Args:
            query: Natural language query
            max_results: Maximum number of results to return
            
        Returns:
            List of matching issues with full content
        """
        # Extract keywords from the query for better search
        keywords = self.text_processor.extract_keywords(query)
        
        if not keywords:
            logger.warning("No keywords extracted from query")
            return []
        
        # Build a JQL query from the keywords
        keyword_terms = []
        for keyword in keywords[:5]:  # Use top 5 keywords
            # Escape special JQL characters
            safe_keyword = keyword.replace('"', '\\"').replace("'", "\\'")
            if len(safe_keyword) > 2:  # Skip very short keywords
                keyword_terms.append(f'text ~ "{safe_keyword}"')
        
        if not keyword_terms:
            # Fallback to a more general search if no usable keywords
            jql = 'order by updated DESC'
        else:
            jql = ' AND '.join(keyword_terms)
        
        logger.info(f"Searching with JQL: {jql}")
        
        # Get matching issues
        issues = self.get_all_issues(jql, max_results=max_results)
        
        # Process issues to get full content
        processed_issues = []
        for issue in issues:
            issue_key = issue.get("key")
            if issue_key:
                processed = self.get_issue_content(issue_key)
                processed_issues.append(processed)
        
        return processed_issues
```































# Flask App Server

```python
import os
import json
import logging
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

from config import HOST, PORT, DEBUG, CACHE_DIR
from confluence_client import ConfluenceClient
from jira_client import JiraClient
from gemini_client import GeminiClient
from utils import Cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jira_confluence_chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AppServer")

# Initialize app
app = Flask(__name__, static_folder="../frontend")
CORS(app)  # Enable CORS for all routes

# Initialize clients
cache = Cache(CACHE_DIR)
confluence_client = ConfluenceClient(cache=cache)
jira_client = JiraClient(cache=cache)
gemini_client = GeminiClient()

# Configure thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=5)

# Flag to track if clients are initialized
clients_ready = {
    "confluence": False,
    "jira": False
}

def test_connections():
    """Test connections to Jira and Confluence in background."""
    global clients_ready
    
    try:
        # Test Confluence connection
        logger.info("Testing Confluence connection...")
        confluence_result = confluence_client.test_connection()
        clients_ready["confluence"] = confluence_result
        logger.info(f"Confluence connection test result: {confluence_result}")
        
        # Test Jira connection
        logger.info("Testing Jira connection...")
        jira_result = jira_client.test_connection()
        clients_ready["jira"] = jira_result
        logger.info(f"Jira connection test result: {jira_result}")
    except Exception as e:
        logger.error(f"Error testing connections: {str(e)}")

# Start connection tests in background
threading.Thread(target=test_connections, daemon=True).start()

# Routes
@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/status")
def get_status():
    """Get the status of the application."""
    return jsonify({
        "status": "online",
        "connections": clients_ready
    })

@app.route("/api/search", methods=["POST"])
def search():
    """Search for content in Confluence, Jira, or both."""
    try:
        data = request.json
        query = data.get("query", "")
        source = data.get("source", "combined")  # confluence, jira, or combined
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        results = []
        
        # Search in Confluence
        if source in ["confluence", "combined"]:
            if clients_ready["confluence"]:
                confluence_results = confluence_client.search_content(query)
                results.extend(confluence_results)
            else:
                logger.warning("Confluence client not ready, skipping search")
                if source == "confluence":
                    return jsonify({"error": "Confluence connection not available"}), 503
        
        # Search in Jira
        if source in ["jira", "combined"]:
            if clients_ready["jira"]:
                jira_results = jira_client.search_issues_by_query(query)
                results.extend(jira_results)
            else:
                logger.warning("Jira client not ready, skipping search")
                if source == "jira":
                    return jsonify({"error": "Jira connection not available"}), 503
        
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Error searching: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    """Generate a response from Gemini based on provided context."""
    try:
        data = request.json
        query = data.get("query", "")
        source = data.get("source", "combined")
        context_ids = data.get("context_ids", [])
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Get the context documents
        context_docs = []
        
        for context_id in context_ids:
            parts = context_id.split(":")
            if len(parts) != 2:
                continue
                
            source_type, item_id = parts
            
            if source_type == "confluence" and clients_ready["confluence"]:
                doc = confluence_client.get_page_content(item_id)
                if doc:
                    context_docs.append(doc)
                    
            elif source_type == "jira" and clients_ready["jira"]:
                doc = jira_client.get_issue_content(item_id)
                if doc:
                    context_docs.append(doc)
        
        # Build system prompt based on source and context
        system_prompt = gemini_client.build_system_prompt(source, context_docs)
        
        # Generate response
        response = gemini_client.generate_response(
            prompt=query,
            system_prompt=system_prompt,
            temperature=0.3,
            stream=False
        )
        
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/jira/ticket/<ticket_id>")
def get_jira_ticket(ticket_id):
    """Get details for a specific Jira ticket."""
    try:
        if not clients_ready["jira"]:
            return jsonify({"error": "Jira connection not available"}), 503
            
        ticket = jira_client.get_issue_content(ticket_id)
        if not ticket:
            return jsonify({"error": f"Ticket {ticket_id} not found"}), 404
            
        return jsonify({"ticket": ticket})
    except Exception as e:
        logger.error(f"Error getting Jira ticket: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/confluence/page/<page_id>")
def get_confluence_page(page_id):
    """Get details for a specific Confluence page."""
    try:
        if not clients_ready["confluence"]:
            return jsonify({"error": "Confluence connection not available"}), 503
            
        page = confluence_client.get_page_content(page_id)
        if not page:
            return jsonify({"error": f"Page {page_id} not found"}), 404
            
        return jsonify({"page": page})
    except Exception as e:
        logger.error(f"Error getting Confluence page: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/find-solution", methods=["POST"])
def find_solution():
    """Find solutions in Confluence for a Jira ticket."""
    try:
        data = request.json
        ticket_id = data.get("ticket_id", "")
        query = data.get("query", "")
        
        if not ticket_id:
            return jsonify({"error": "No ticket ID provided"}), 400
            
        if not clients_ready["jira"] or not clients_ready["confluence"]:
            return jsonify({"error": "Jira or Confluence connection not available"}), 503
            
        # Get the ticket details
        ticket = jira_client.get_issue_content(ticket_id)
        if not ticket:
            return jsonify({"error": f"Ticket {ticket_id} not found"}), 404
            
        # Generate search query for Confluence
        search_query = gemini_client.build_query_for_solution(ticket, query)
        
        # Search Confluence for solutions
        solutions = confluence_client.search_content(search_query)
        
        # Create context with both ticket and solutions
        context_docs = [ticket] + solutions[:3]  # Limit to top 3 solutions
        
        # Build system prompt for combined sources
        system_prompt = gemini_client.build_system_prompt("combined", context_docs)
        
        # Generate response
        prompt = f"Based on the provided Jira ticket {ticket_id} and Confluence documentation, suggest a solution for this issue."
        if query:
            prompt += f" The user specifically wants to know: {query}"
            
        response = gemini_client.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            stream=False
        )
        
        return jsonify({
            "ticket": ticket,
            "solutions": solutions,
            "response": response
        })
    except Exception as e:
        logger.error(f"Error finding solution: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/clear-cache", methods=["POST"])
def clear_cache():
    """Clear the cache."""
    try:
        cache.clear()
        return jsonify({"message": "Cache cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
```











# Python Dependencies

```
flask==2.3.3
flask-cors==4.0.0
requests==2.31.0
vertexai==1.45.0
google-cloud-aiplatform==1.45.0
python-dotenv==1.0.0
```


















<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jira-Confluence Chatbot</title>
    <link rel="stylesheet" href="css/styles.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="app-logo">
                <i class="fa-solid fa-robot"></i>
                <h1>JC Chatbot</h1>
            </div>
            <div class="source-selector">
                <h3>Data Sources</h3>
                <div class="source-options">
                    <label>
                        <input type="radio" name="source" value="combined" checked>
                        <span>Both</span>
                    </label>
                    <label>
                        <input type="radio" name="source" value="jira">
                        <span>Jira</span>
                    </label>
                    <label>
                        <input type="radio" name="source" value="confluence">
                        <span>Confluence</span>
                    </label>
                </div>
            </div>
            <div class="connection-status">
                <h3>System Status</h3>
                <div class="status-indicator">
                    <div class="status-item">
                        <span>Jira:</span>
                        <span id="jira-status" class="status pending">Connecting...</span>
                    </div>
                    <div class="status-item">
                        <span>Confluence:</span>
                        <span id="confluence-status" class="status pending">Connecting...</span>
                    </div>
                </div>
            </div>
            <div class="settings">
                <button id="clear-cache" class="btn-secondary">
                    <i class="fa-solid fa-broom"></i> Clear Cache
                </button>
                <button id="clear-chat" class="btn-secondary">
                    <i class="fa-solid fa-trash"></i> Clear Chat
                </button>
            </div>
        </div>

        <!-- Main Chat Area -->
        <div class="main-content">
            <!-- Chat Header -->
            <div class="chat-header">
                <h2>Jira-Confluence Assistant</h2>
                <div class="action-buttons">
                    <button id="help-button" class="btn-icon">
                        <i class="fa-solid fa-circle-question"></i>
                    </button>
                </div>
            </div>

            <!-- Chat Messages -->
            <div class="chat-messages" id="chat-messages">
                <div class="message system">
                    <div class="message-content">
                        <p>Hello! I'm your Jira-Confluence assistant. I can help you find information from your Jira tickets and Confluence pages, or even find solutions from Confluence for your Jira issues.</p>
                        <p>To get started, simply ask me a question about a Jira ticket, a Confluence page, or any topic that might be discussed in either system.</p>
                    </div>
                </div>
            </div>

            <!-- Search Results Panel (hidden by default) -->
            <div class="search-results-panel" id="search-results-panel">
                <div class="panel-header">
                    <h3>Search Results</h3>
                    <button class="btn-icon close-panel">
                        <i class="fa-solid fa-times"></i>
                    </button>
                </div>
                <div class="results-list" id="search-results-list">
                    <!-- Results will be inserted here -->
                </div>
            </div>

            <!-- Selected Context Panel (hidden by default) -->
            <div class="context-panel" id="context-panel">
                <div class="panel-header">
                    <h3>Selected Context</h3>
                    <button class="btn-icon close-panel">
                        <i class="fa-solid fa-times"></i>
                    </button>
                </div>
                <div class="context-list" id="context-list">
                    <!-- Selected context items will be inserted here -->
                </div>
                <div class="panel-footer">
                    <button id="use-context" class="btn-primary">Use Selected Context</button>
                </div>
            </div>

            <!-- Chat Input Area -->
            <div class="chat-input-area">
                <textarea id="chat-input" placeholder="Type your message here..." rows="1"></textarea>
                <button id="send-button" class="btn-primary">
                    <i class="fa-solid fa-paper-plane"></i>
                </button>
            </div>
        </div>
    </div>

    <!-- Details Modal -->
    <div class="modal" id="details-modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modal-title">Item Details</h3>
                <button class="btn-icon close-modal">
                    <i class="fa-solid fa-times"></i>
                </button>
            </div>
            <div class="modal-body" id="modal-body">
                <!-- Details will be inserted here -->
            </div>
            <div class="modal-footer">
                <button id="find-solution-btn" class="btn-primary">Find Solution in Confluence</button>
                <button id="add-to-context-btn" class="btn-secondary">Add to Context</button>
                <button class="btn-secondary close-modal">Close</button>
            </div>
        </div>
    </div>

    <!-- Help Modal -->
    <div class="modal" id="help-modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3>How to Use This Chatbot</h3>
                <button class="btn-icon close-modal">
                    <i class="fa-solid fa-times"></i>
                </button>
            </div>
            <div class="modal-body">
                <h4>Basic Questions</h4>
                <p>Simply type your question about Jira tickets or Confluence pages in the chat input, and I'll try to find relevant information.</p>
                
                <h4>Data Sources</h4>
                <p>You can choose which systems to search:</p>
                <ul>
                    <li><strong>Both:</strong> Search both Jira and Confluence (default)</li>
                    <li><strong>Jira:</strong> Only search Jira tickets</li>
                    <li><strong>Confluence:</strong> Only search Confluence pages</li>
                </ul>
                
                <h4>Working with Search Results</h4>
                <p>After you ask a question, I'll search for relevant information and may show you search results:</p>
                <ul>
                    <li>Click on a result to view more details</li>
                    <li>You can add items to your context to include them in future questions</li>
                    <li>For Jira tickets, you can use the "Find Solution" button to search Confluence for potential solutions</li>
                </ul>
                
                <h4>Advanced Features</h4>
                <ul>
                    <li><strong>Context Panel:</strong> Build a collection of related items to provide more context for complex questions</li>
                    <li><strong>Find Solutions:</strong> When viewing a Jira ticket, click "Find Solution" to search Confluence for relevant documentation</li>
                    <li><strong>Clear Cache:</strong> If data seems outdated, clear the cache to fetch fresh information</li>
                </ul>
            </div>
            <div class="modal-footer">
                <button class="btn-primary close-modal">Got it!</button>
            </div>
        </div>
    </div>

    <!-- Scripts -->
    <script src="js/utils.js"></script>
    <script src="js/api.js"></script>
    <script src="js/chat.js"></script>
    <script src="js/main.js"></script>
</body>
</html>
















/* Global Styles */
:root {
    --primary-color: #0052cc;
    --primary-light: #4c9aff;
    --primary-dark: #0747a6;
    --secondary-color: #36b37e;
    --secondary-light: #57d9a3;
    --secondary-dark: #00875a;
    --warning-color: #ffab00;
    --error-color: #ff5630;
    --success-color: #36b37e;
    --text-primary: #172b4d;
    --text-secondary: #6b778c;
    --text-inverted: #ffffff;
    --bg-light: #f4f5f7;
    --bg-white: #ffffff;
    --bg-dark: #091e42;
    --border-color: #dfe1e6;
    --shadow-default: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);
    --shadow-elevated: 0 10px 20px rgba(0, 0, 0, 0.19), 0 6px 6px rgba(0, 0, 0, 0.23);
    --radius-sm: 3px;
    --radius-md: 6px;
    --radius-lg: 12px;
    --font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    --transition: all 0.3s ease;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: var(--font-family);
    color: var(--text-primary);
    background-color: var(--bg-light);
    line-height: 1.5;
    height: 100vh;
    overflow: hidden;
}

/* Layout */
.app-container {
    display: flex;
    height: 100vh;
    overflow: hidden;
}

.sidebar {
    width: 260px;
    background-color: var(--bg-dark);
    color: var(--text-inverted);
    padding: 16px;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
}

.main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
}

/* Sidebar Components */
.app-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
}

.app-logo i {
    font-size: 24px;
    color: var(--primary-light);
}

.app-logo h1 {
    font-size: 20px;
    font-weight: 500;
}

.source-selector, 
.connection-status {
    margin-bottom: 24px;
}

.source-selector h3, 
.connection-status h3 {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    opacity: 0.8;
}

.source-options {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.source-options label {
    display: flex;
    align-items: center;
    cursor: pointer;
    padding: 8px 12px;
    border-radius: var(--radius-sm);
    transition: var(--transition);
}

.source-options label:hover {
    background-color: rgba(255, 255, 255, 0.1);
}

.source-options input {
    margin-right: 8px;
}

.status-indicator {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.status-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.status {
    padding: 4px 8px;
    border-radius: var(--radius-sm);
    font-size: 12px;
    font-weight: 500;
}

.status.pending {
    background-color: var(--warning-color);
    color: var(--bg-dark);
}

.status.connected {
    background-color: var(--success-color);
    color: var(--text-inverted);
}

.status.disconnected {
    background-color: var(--error-color);
    color: var(--text-inverted);
}

.settings {
    margin-top: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

/* Chat Components */
.chat-header {
    padding: 16px 24px;
    background-color: var(--bg-white);
    box-shadow: var(--shadow-default);
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 10;
}

.chat-header h2 {
    font-size: 18px;
    font-weight: 500;
}

.action-buttons {
    display: flex;
    gap: 8px;
}

.chat-messages {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
    background-color: var(--bg-light);
}

.message {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: var(--radius-md);
    position: relative;
    animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.message.user {
    align-self: flex-end;
    background-color: var(--primary-color);
    color: var(--text-inverted);
}

.message.bot {
    align-self: flex-start;
    background-color: var(--bg-white);
    box-shadow: var(--shadow-default);
}

.message.system {
    align-self: center;
    background-color: var(--bg-white);
    border: 1px dashed var(--border-color);
    max-width: 90%;
    text-align: center;
}

.message.thinking {
    align-self: flex-start;
    background-color: var(--bg-white);
    opacity: 0.7;
}

.message-content {
    word-break: break-word;
}

.message-content p {
    margin-bottom: 8px;
}

.message-content p:last-child {
    margin-bottom: 0;
}

.chat-input-area {
    padding: 16px 24px;
    background-color: var(--bg-white);
    box-shadow: 0 -1px 3px rgba(0, 0, 0, 0.1);
    display: flex;
    gap: 12px;
    align-items: center;
}

.chat-input-area textarea {
    flex: 1;
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 12px 16px;
    font-family: var(--font-family);
    font-size: 14px;
    outline: none;
    resize: none;
    transition: var(--transition);
}

.chat-input-area textarea:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(0, 82, 204, 0.2);
}

/* Panels */
.search-results-panel,
.context-panel {
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    width: 350px;
    background-color: var(--bg-white);
    box-shadow: var(--shadow-elevated);
    display: flex;
    flex-direction: column;
    z-index: 100;
    transform: translateX(100%);
    transition: transform 0.3s ease;
}

.search-results-panel.visible,
.context-panel.visible {
    transform: translateX(0);
}

.panel-header {
    padding: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-color);
}

.panel-header h3 {
    font-size: 16px;
    font-weight: 500;
}

.results-list,
.context-list {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
}

.result-item,
.context-item {
    padding: 12px;
    border-radius: var(--radius-sm);
    margin-bottom: 8px;
    cursor: pointer;
    transition: var(--transition);
    background-color: var(--bg-light);
    border-left: 4px solid transparent;
}

.result-item:hover,
.context-item:hover {
    background-color: #e8f0fe;
}

.result-item.jira,
.context-item.jira {
    border-left-color: var(--primary-color);
}

.result-item.confluence,
.context-item.confluence {
    border-left-color: var(--secondary-color);
}

.result-item h4,
.context-item h4 {
    font-size: 14px;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.result-item h4 i,
.context-item h4 i {
    font-size: 12px;
}

.result-item p,
.context-item p {
    font-size: 12px;
    color: var(--text-secondary);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.panel-footer {
    padding: 16px;
    border-top: 1px solid var(--border-color);
    display: flex;
    justify-content: flex-end;
}

/* Modals */
.modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    visibility: hidden;
    opacity: 0;
    transition: all 0.3s ease;
}

.modal.visible {
    visibility: visible;
    opacity: 1;
}

.modal-content {
    background-color: var(--bg-white);
    border-radius: var(--radius-md);
    width: 600px;
    max-width: 90%;
    max-height: 90vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    animation: modalSlide 0.3s ease;
}

@keyframes modalSlide {
    from { transform: translateY(30px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

.modal-header {
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.modal-header h3 {
    font-size: 18px;
    font-weight: 500;
}

.modal-body {
    padding: 16px;
    overflow-y: auto;
    max-height: 60vh;
}

.modal-body h4 {
    margin-top: 16px;
    margin-bottom: 8px;
    font-size: 16px;
}

.modal-body ul {
    padding-left: 20px;
    margin-bottom: 16px;
}

.modal-body p {
    margin-bottom: 16px;
}

.modal-footer {
    padding: 16px;
    border-top: 1px solid var(--border-color);
    display: flex;
    justify-content: flex-end;
    gap: 12px;
}

/* Buttons */
.btn-primary, 
.btn-secondary, 
.btn-icon {
    padding: 8px 16px;
    border-radius: var(--radius-sm);
    font-family: var(--font-family);
    font-size: 14px;
    cursor: pointer;
    transition: var(--transition);
    border: none;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

.btn-primary {
    background-color: var(--primary-color);
    color: var(--text-inverted);
}

.btn-primary:hover {
    background-color: var(--primary-dark);
}

.btn-secondary {
    background-color: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
}

.btn-secondary:hover {
    background-color: var(--bg-light);
    color: var(--text-primary);
}

.btn-icon {
    background-color: transparent;
    color: var(--text-secondary);
    padding: 6px;
    font-size: 16px;
}

.btn-icon:hover {
    color: var(--primary-color);
    background-color: var(--bg-light);
}

/* Detail View */
.detail-view {
    padding: 16px;
}

.detail-header {
    margin-bottom: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color);
}

.detail-header h3 {
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.detail-header h3 i.jira {
    color: var(--primary-color);
}

.detail-header h3 i.confluence {
    color: var(--secondary-color);
}

.detail-header .meta {
    font-size: 12px;
    color: var(--text-secondary);
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
}

.detail-header .meta span {
    display: flex;
    align-items: center;
    gap: 4px;
}

.detail-content {
    margin-bottom: 16px;
    line-height: 1.6;
}

.detail-content h4 {
    margin-top: 16px;
    margin-bottom: 8px;
}

.tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 16px;
}

.tag {
    background-color: var(--bg-light);
    color: var(--text-secondary);
    padding: 4px 8px;
    border-radius: var(--radius-sm);
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 4px;
}

.url-link {
    display: inline-block;
    margin-top: 16px;
    color: var(--primary-color);
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 14px;
}

.url-link:hover {
    text-decoration: underline;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .app-container {
        flex-direction: column;
    }
    
    .sidebar {
        width: 100%;
        height: auto;
        padding: 12px;
    }
    
    .app-logo {
        margin-bottom: 12px;
        padding-bottom: 12px;
    }
    
    .source-selector, 
    .connection-status,
    .settings {
        margin-bottom: 16px;
    }
    
    .source-options {
        flex-direction: row;
        flex-wrap: wrap;
    }
    
    .search-results-panel,
    .context-panel {
        width: 100%;
    }
    
    .message {
        max-width: 90%;
    }
}

















/**
 * API Client for communicating with the backend
 */
class ApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl || '';
    }

    /**
     * Make an API request
     * @param {string} endpoint - API endpoint
     * @param {string} method - HTTP method (GET, POST, etc.)
     * @param {Object} data - Request data for POST requests
     * @returns {Promise} - Promise resolving to response data
     */
    async request(endpoint, method = 'GET', data = null) {
        const url = `${this.baseUrl}/api/${endpoint}`;
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (data && (method === 'POST' || method === 'PUT')) {
            options.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(url, options);
            
            // Check if the request was successful
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({
                    error: `HTTP error ${response.status}`,
                }));
                throw new Error(errorData.error || `HTTP error ${response.status}`);
            }
            
            // Parse and return the response data
            return await response.json();
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            throw error;
        }
    }

    /**
     * Get application status
     * @returns {Promise} - Promise resolving to status data
     */
    async getStatus() {
        return this.request('status');
    }

    /**
     * Search for content using the provided query
     * @param {string} query - Search query
     * @param {string} source - Source to search (jira, confluence, or combined)
     * @returns {Promise} - Promise resolving to search results
     */
    async search(query, source = 'combined') {
        return this.request('search', 'POST', { query, source });
    }

    /**
     * Get chat response based on query and context
     * @param {string} query - User query
     * @param {string} source - Source to use for context (jira, confluence, or combined)
     * @param {Array} contextIds - IDs of context documents to include
     * @returns {Promise} - Promise resolving to chat response
     */
    async chat(query, source = 'combined', contextIds = []) {
        return this.request('chat', 'POST', { query, source, context_ids: contextIds });
    }

    /**
     * Get details for a specific Jira ticket
     * @param {string} ticketId - Jira ticket ID
     * @returns {Promise} - Promise resolving to ticket details
     */
    async getJiraTicket(ticketId) {
        return this.request(`jira/ticket/${ticketId}`, 'GET');
    }

    /**
     * Get details for a specific Confluence page
     * @param {string} pageId - Confluence page ID
     * @returns {Promise} - Promise resolving to page details
     */
    async getConfluencePage(pageId) {
        return this.request(`confluence/page/${pageId}`, 'GET');
    }

    /**
     * Find solutions in Confluence for a Jira ticket
     * @param {string} ticketId - Jira ticket ID
     * @param {string} query - Optional additional query to focus the search
     * @returns {Promise} - Promise resolving to solutions data
     */
    async findSolution(ticketId, query = '') {
        return this.request('find-solution', 'POST', { ticket_id: ticketId, query });
    }

    /**
     * Clear the server cache
     * @returns {Promise} - Promise resolving to confirmation message
     */
    async clearCache() {
        return this.request('clear-cache', 'POST');
    }
}

// Create global API client instance
const api = new ApiClient();














/**
 * Utility functions for the application
 */

/**
 * Format a timestamp for display
 * @param {string} timestamp - ISO timestamp string
 * @returns {string} - Formatted date string
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return 'Unknown date';
    
    const date = new Date(timestamp);
    return date.toLocaleString();
}

/**
 * Truncate text to a specified length with ellipsis
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length of the text
 * @returns {string} - Truncated text
 */
function truncateText(text, maxLength = 100) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    
    return text.substring(0, maxLength) + '...';
}

/**
 * Create an HTML element with specified attributes
 * @param {string} tag - HTML tag name
 * @param {Object} attributes - Element attributes
 * @param {string|Node} content - Element content (string or DOM node)
 * @returns {HTMLElement} - Created HTML element
 */
function createElement(tag, attributes = {}, content = '') {
    const element = document.createElement(tag);
    
    // Set attributes
    for (const [key, value] of Object.entries(attributes)) {
        if (key === 'classList' && Array.isArray(value)) {
            element.classList.add(...value);
        } else {
            element[key] = value;
        }
    }
    
    // Set content
    if (content) {
        if (typeof content === 'string') {
            element.innerHTML = content;
        } else {
            element.appendChild(content);
        }
    }
    
    return element;
}

/**
 * Show a notification message to the user
 * @param {string} message - Message to display
 * @param {string} type - Message type (success, error, warning)
 * @param {number} duration - Duration in milliseconds
 */
function showNotification(message, type = 'info', duration = 3000) {
    // Create notification element if it doesn't exist
    let notificationContainer = document.querySelector('.notification-container');
    
    if (!notificationContainer) {
        notificationContainer = createElement('div', {
            classList: ['notification-container']
        });
        document.body.appendChild(notificationContainer);
        
        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            .notification-container {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            .notification {
                padding: 12px 16px;
                border-radius: 4px;
                background-color: #ffffff;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                display: flex;
                align-items: center;
                gap: 10px;
                animation: slideIn 0.3s ease, fadeOut 0.3s ease forwards;
                animation-delay: 0s, ${(duration - 300) / 1000}s;
                max-width: 350px;
            }
            
            .notification i {
                font-size: 18px;
            }
            
            .notification.info i {
                color: #2684FF;
            }
            
            .notification.success i {
                color: #36B37E;
            }
            
            .notification.warning i {
                color: #FFAB00;
            }
            
            .notification.error i {
                color: #FF5630;
            }
            
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            @keyframes fadeOut {
                from { opacity: 1; }
                to { opacity: 0; transform: translateY(-10px); }
            }
        `;
        document.head.appendChild(style);
    }
    
    // Get icon based on type
    let icon;
    switch (type) {
        case 'success':
            icon = 'fa-solid fa-check-circle';
            break;
        case 'warning':
            icon = 'fa-solid fa-exclamation-triangle';
            break;
        case 'error':
            icon = 'fa-solid fa-times-circle';
            break;
        default:
            icon = 'fa-solid fa-info-circle';
    }
    
    // Create notification
    const notification = createElement('div', {
        classList: ['notification', type]
    }, `<i class="${icon}"></i> ${message}`);
    
    // Add to container
    notificationContainer.appendChild(notification);
    
    // Remove after duration
    setTimeout(() => {
        notification.remove();
    }, duration);
}

/**
 * Sanitize HTML to prevent XSS
 * @param {string} html - HTML to sanitize
 * @returns {string} - Sanitized HTML
 */
function sanitizeHtml(html) {
    if (!html) return '';
    
    const temp = document.createElement('div');
    temp.textContent = html;
    return temp.innerHTML;
}

/**
 * Get the source icon based on the source type
 * @param {string} sourceType - Source type (jira or confluence)
 * @returns {string} - HTML for the icon
 */
function getSourceIcon(sourceType) {
    if (sourceType === 'jira') {
        return '<i class="fa-solid fa-ticket-alt" style="color: var(--primary-color);"></i>';
    } else if (sourceType === 'confluence') {
        return '<i class="fa-solid fa-book" style="color: var(--secondary-color);"></i>';
    }
    return '';
}

/**
 * Format content for display
 * @param {string} content - Content to format
 * @returns {string} - Formatted content
 */
function formatContent(content) {
    if (!content) return '';
    
    // Replace newlines with <br>
    return content.replace(/\n/g, '<br>');
}

/**
 * Detect if the string is a Jira ticket key
 * @param {string} str - String to check
 * @returns {boolean} - True if the string matches a Jira ticket key pattern
 */
function isJiraTicketKey(str) {
    return /^[A-Z]+-\d+$/.test(str);
}

/**
 * Extract Jira ticket keys from text
 * @param {string} text - Text to extract ticket keys from
 * @returns {Array} - Array of ticket keys
 */
function extractJiraTicketKeys(text) {
    const regex = /\b([A-Z]+-\d+)\b/g;
    return text.match(regex) || [];
}

/**
 * Enable auto-resizing for textareas
 * @param {HTMLTextAreaElement} textarea - Textarea element to enable auto-resize for
 */
function enableTextareaAutoResize(textarea) {
    function resize() {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    }
    
    // Call resize on input
    textarea.addEventListener('input', resize);
    
    // Initial resize
    resize();
}












/**
 * Chat functionality management
 */
class ChatManager {
    constructor() {
        // DOM elements
        this.messageContainer = document.getElementById('chat-messages');
        this.inputField = document.getElementById('chat-input');
        this.sendButton = document.getElementById('send-button');
        this.sourceOptions = document.getElementsByName('source');
        this.searchResultsPanel = document.getElementById('search-results-panel');
        this.searchResultsList = document.getElementById('search-results-list');
        this.contextPanel = document.getElementById('context-panel');
        this.contextList = document.getElementById('context-list');
        this.useContextButton = document.getElementById('use-context');
        
        // State
        this.isProcessing = false;
        this.chatHistory = [];
        this.selectedContextItems = [];
        this.lastQuery = '';
        this.detailsModalItem = null;
        
        // Initialize
        this.initialize();
    }
    
    /**
     * Initialize the chat manager
     */
    initialize() {
        // Set up event listeners
        this.inputField.addEventListener('keydown', this.handleInputKeydown.bind(this));
        this.sendButton.addEventListener('click', this.handleSendClick.bind(this));
        this.useContextButton.addEventListener('click', this.handleUseContextClick.bind(this));
        
        // Set up textarea auto-resize
        enableTextareaAutoResize(this.inputField);
        
        // Close panels when clicking close buttons
        document.querySelectorAll('.close-panel').forEach(button => {
            button.addEventListener('click', this.closeAllPanels.bind(this));
        });
    }
    
    /**
     * Handle keydown event in the input field
     * @param {KeyboardEvent} event - Keyboard event
     */
    handleInputKeydown(event) {
        // Send message on Enter (without Shift)
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }
    
    /**
     * Handle click on the send button
     */
    handleSendClick() {
        this.sendMessage();
    }
    
    /**
     * Close all panels
     */
    closeAllPanels() {
        this.searchResultsPanel.classList.remove('visible');
        this.contextPanel.classList.remove('visible');
    }
    
    /**
     * Get the currently selected source
     * @returns {string} - Selected source (jira, confluence, or combined)
     */
    getSelectedSource() {
        for (const option of this.sourceOptions) {
            if (option.checked) {
                return option.value;
            }
        }
        return 'combined'; // Default
    }
    
    /**
     * Send a message
     */
    async sendMessage() {
        const message = this.inputField.value.trim();
        
        if (!message || this.isProcessing) {
            return;
        }
        
        // Clear input field
        this.inputField.value = '';
        this.inputField.style.height = 'auto';
        
        // Show user message
        this.addUserMessage(message);
        
        // Process the message
        this.isProcessing = true;
        this.lastQuery = message;
        
        try {
            await this.processMessage(message);
        } catch (error) {
            console.error('Error processing message:', error);
            this.addBotMessage(`I encountered an error while processing your request: ${error.message || 'Unknown error'}`);
        } finally {
            this.isProcessing = false;
        }
    }
    
    /**
     * Process a user message
     * @param {string} message - User message
     */
    async processMessage(message) {
        // Add thinking indicator
        this.addThinkingIndicator();
        
        // Check if this is a special command
        if (this.processSpecialCommands(message)) {
            this.removeThinkingIndicator();
            return;
        }
        
        // Get selected source
        const source = this.getSelectedSource();
        
        // Check if we have selected context items
        if (this.selectedContextItems.length > 0) {
            // Use context items for chat
            await this.processChatWithContext(message, source);
        } else {
            // Perform search first
            await this.processSearch(message, source);
        }
        
        // Remove thinking indicator
        this.removeThinkingIndicator();
    }
    
    /**
     * Process special commands
     * @param {string} message - User message
     * @returns {boolean} - True if the message was a special command
     */
    processSpecialCommands(message) {
        const lowerMessage = message.toLowerCase();
        
        // Check for help command
        if (lowerMessage === 'help' || lowerMessage === '/help') {
            document.getElementById('help-modal').classList.add('visible');
            return true;
        }
        
        // Check for clear command
        if (lowerMessage === 'clear' || lowerMessage === '/clear') {
            this.clearChat();
            this.addSystemMessage('Chat history cleared.');
            return true;
        }
        
        // Check if message is a Jira ticket key
        if (isJiraTicketKey(message)) {
            this.processJiraTicket(message);
            return true;
        }
        
        return false;
    }
    
    /**
     * Process a search query
     * @param {string} query - Search query
     * @param {string} source - Source to search
     */
    async processSearch(query, source) {
        try {
            const response = await api.search(query, source);
            
            if (!response.results || response.results.length === 0) {
                // No results found, generate a response without context
                this.addBotMessage(`I couldn't find any specific information about that in our knowledge base. Let me try to help with what I know generally.`);
                
                const chatResponse = await api.chat(query, source, []);
                this.addBotMessage(chatResponse.response);
                return;
            }
            
            // Display the search results
            this.displaySearchResults(response.results);
            
            // Use the top result as context for an initial response
            const topResultId = `${response.results[0].source_type}:${response.results[0].metadata.id}`;
            const chatResponse = await api.chat(query, source, [topResultId]);
            
            this.addBotMessage(chatResponse.response);
        } catch (error) {
            console.error('Search error:', error);
            this.addBotMessage(`I had trouble searching for that information. Please try again or rephrase your question.`);
        }
    }
    
    /**
     * Process chat with context
     * @param {string} message - User message
     * @param {string} source - Source to use
     */
    async processChatWithContext(message, source) {
        try {
            // Get context IDs
            const contextIds = this.selectedContextItems.map(item => 
                `${item.sourceType}:${item.id}`
            );
            
            // Get chat response
            const response = await api.chat(message, source, contextIds);
            
            // Display response
            this.addBotMessage(response.response);
        } catch (error) {
            console.error('Chat error:', error);
            this.addBotMessage(`I had trouble processing your request with the selected context. Please try again.`);
        }
    }
    
    /**
     * Process a Jira ticket key
     * @param {string} ticketKey - Jira ticket key
     */
    async processJiraTicket(ticketKey) {
        try {
            // Get ticket details
            const response = await api.getJiraTicket(ticketKey);
            
            if (!response.ticket) {
                this.addBotMessage(`I couldn't find the ticket ${ticketKey}. Please check if the ticket key is correct.`);
                return;
            }
            
            // Display the ticket
            this.addBotMessage(`I found the ticket <strong>${ticketKey}</strong>: ${response.ticket.metadata.summary}`);
            
            // Show ticket details in modal
            this.showItemDetails(response.ticket);
        } catch (error) {
            console.error('Jira ticket error:', error);
            this.addBotMessage(`I had trouble retrieving information for ticket ${ticketKey}. Please try again.`);
        }
    }
    
    /**
     * Display search results
     * @param {Array} results - Search results
     */
    displaySearchResults(results) {
        // Clear previous results
        this.searchResultsList.innerHTML = '';
        
        // Add each result
        results.forEach(result => {
            const sourceType = result.source_type;
            const metadata = result.metadata;
            
            const resultItem = createElement('div', {
                classList: ['result-item', sourceType],
                dataset: {
                    id: metadata.id,
                    sourceType: sourceType
                }
            });
            
            // Create header with icon
            const header = createElement('h4', {}, 
                `${getSourceIcon(sourceType)} ${metadata.title || (sourceType === 'jira' ? metadata.key : 'Untitled')}`
            );
            resultItem.appendChild(header);
            
            // Add snippet
            const snippet = createElement('p', {}, truncateText(result.content, 150));
            resultItem.appendChild(snippet);
            
            // Add click event
            resultItem.addEventListener('click', () => {
                this.showItemDetails(result);
            });
            
            // Add to results list
            this.searchResultsList.appendChild(resultItem);
        });
        
        // Show results panel
        this.searchResultsPanel.classList.add('visible');
        this.contextPanel.classList.remove('visible');
        
        // Add system message
        this.addSystemMessage(`Found ${results.length} results. Click on a result to view details.`);
    }
    
    /**
     * Show item details in modal
     * @param {Object} item - Item to show details for
     */
    showItemDetails(item) {
        // Store current item
        this.detailsModalItem = item;
        
        // Set modal title
        const modalTitle = document.getElementById('modal-title');
        const sourceType = item.source_type;
        
        if (sourceType === 'jira') {
            modalTitle.innerHTML = `${getSourceIcon(sourceType)} Jira Ticket: ${item.metadata.key}`;
        } else {
            modalTitle.innerHTML = `${getSourceIcon(sourceType)} Confluence Page: ${item.metadata.title}`;
        }
        
        // Set modal body
        const modalBody = document.getElementById('modal-body');
        modalBody.innerHTML = '';
        
        // Create detail view
        const detailView = createElement('div', { classList: ['detail-view'] });
        
        // Header
        const header = createElement('div', { classList: ['detail-header'] });
        
        // Title
        const title = createElement('h3', {}, 
            sourceType === 'jira' ? item.metadata.summary : item.metadata.title
        );
        header.appendChild(title);
        
        // Metadata
        const meta = createElement('div', { classList: ['meta'] });
        
        if (sourceType === 'jira') {
            meta.innerHTML = `
                <span><i class="fa-solid fa-tag"></i> ${item.metadata.issuetype || 'Unknown'}</span>
                <span><i class="fa-solid fa-circle-check"></i> ${item.metadata.status || 'Unknown'}</span>
                <span><i class="fa-solid fa-user"></i> ${item.metadata.assignee || 'Unassigned'}</span>
            `;
        } else {
            // Confluence metadata
        }
        
        header.appendChild(meta);
        detailView.appendChild(header);
        
        // Content
        const content = createElement('div', { classList: ['detail-content'] });
        content.innerHTML = formatContent(item.content);
        detailView.appendChild(content);
        
        // URL
        if (item.url) {
            const urlLink = createElement('a', {
                classList: ['url-link'],
                href: item.url,
                target: '_blank'
            }, `<i class="fa-solid fa-external-link-alt"></i> Open in ${sourceType === 'jira' ? 'Jira' : 'Confluence'}`);
            
            detailView.appendChild(urlLink);
        }
        
        // Add to modal body
        modalBody.appendChild(detailView);
        
        // Set "Find Solution" button visibility
        const findSolutionBtn = document.getElementById('find-solution-btn');
        findSolutionBtn.style.display = sourceType === 'jira' ? 'block' : 'none';
        
        // Show modal
        document.getElementById('details-modal').classList.add('visible');
    }
    
    /**
     * Add item to selected context
     * @param {Object} item - Item to add to context
     */
    addItemToContext(item) {
        // Check if item is already in context
        const existingIndex = this.selectedContextItems.findIndex(
            contextItem => contextItem.id === item.metadata.id && contextItem.sourceType === item.source_type
        );
        
        if (existingIndex !== -1) {
            showNotification('This item is already in your context', 'info');
            return;
        }
        
        // Add to context
        this.selectedContextItems.push({
            id: item.metadata.id,
            title: item.metadata.title || (item.source_type === 'jira' ? item.metadata.key : 'Untitled'),
            sourceType: item.source_type,
            content: truncateText(item.content, 100)
        });
        
        // Update context list
        this.updateContextList();
        
        // Show notification
        showNotification('Added to context', 'success');
        
        // Show context panel
        this.contextPanel.classList.add('visible');
        this.searchResultsPanel.classList.remove('visible');
    }
    
    /**
     * Update the context list display
     */
    updateContextList() {
        // Clear current list
        this.contextList.innerHTML = '';
        
        // Add each context item
        this.selectedContextItems.forEach((item, index) => {
            const contextItem = createElement('div', {
                classList: ['context-item', item.sourceType]
            });
            
            // Header with remove button
            const header = createElement('div', { classList: ['context-item-header'] });
            
            const title = createElement('h4', {}, 
                `${getSourceIcon(item.sourceType)} ${item.title}`
            );
            
            const removeBtn = createElement('button', {
                classList: ['btn-icon'],
                title: 'Remove from context'
            }, '<i class="fa-solid fa-times"></i>');
            
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.removeContextItem(index);
            });
            
            header.appendChild(title);
            header.appendChild(removeBtn);
            contextItem.appendChild(header);
            
            // Snippet
            const snippet = createElement('p', {}, item.content);
            contextItem.appendChild(snippet);
            
            // Add to list
            this.contextList.appendChild(contextItem);
        });
        
        // Update button state
        this.useContextButton.disabled = this.selectedContextItems.length === 0;
    }
    
    /**
     * Remove an item from context
     * @param {number} index - Index of item to remove
     */
    removeContextItem(index) {
        this.selectedContextItems.splice(index, 1);
        this.updateContextList();
        
        // Show notification
        showNotification('Removed from context', 'info');
        
        // Hide panel if empty
        if (this.selectedContextItems.length === 0) {
            this.contextPanel.classList.remove('visible');
        }
    }
    
    /**
     * Handle click on "Use Context" button
     */
    handleUseContextClick() {
        // Close panels
        this.closeAllPanels();
        
        // Add system message
        this.addSystemMessage(`Using ${this.selectedContextItems.length} item(s) as context for the next question.`);
        
        // Focus input
        this.inputField.focus();
    }
    
    /**
     * Find solution for current item
     * @param {string} additionalQuery - Additional query to focus the search
     */
    async findSolution(additionalQuery = '') {
        if (!this.detailsModalItem || this.detailsModalItem.source_type !== 'jira') {
            showNotification('No Jira ticket selected', 'error');
            return;
        }
        
        try {
            // Add thinking message
            this.addThinkingIndicator();
            
            // Close modal
            document.getElementById('details-modal').classList.remove('visible');
            
            // Get ticket key
            const ticketKey = this.detailsModalItem.metadata.key;
            
            // Show searching message
            this.addBotMessage(`Searching for solutions to ticket <strong>${ticketKey}</strong>...`);
            
            // Find solutions
            const response = await api.findSolution(ticketKey, additionalQuery);
            
            // Display solutions if available
            if (response.solutions && response.solutions.length > 0) {
                this.displaySearchResults(response.solutions);
            }
            
            // Display response
            this.addBotMessage(response.response);
            
        } catch (error) {
            console.error('Find solution error:', error);
            this.addBotMessage(`I had trouble finding solutions for this ticket. Please try again.`);
        } finally {
            this.removeThinkingIndicator();
        }
    }
    
    /**
     * Add a user message to the chat
     * @param {string} message - Message text
     */
    addUserMessage(message) {
        const messageElement = createElement('div', {
            classList: ['message', 'user']
        });
        
        const contentElement = createElement('div', {
            classList: ['message-content']
        }, sanitizeHtml(message));
        
        messageElement.appendChild(contentElement);
        this.messageContainer.appendChild(messageElement);
        
        // Scroll to bottom
        this.scrollToBottom();
        
        // Add to history
        this.chatHistory.push({
            role: 'user',
            content: message
        });
    }
    
    /**
     * Add a bot message to the chat
     * @param {string} message - Message text
     */
    addBotMessage(message) {
        const messageElement = createElement('div', {
            classList: ['message', 'bot']
        });
        
        const contentElement = createElement('div', {
            classList: ['message-content']
        }, message);
        
        messageElement.appendChild(contentElement);
        this.messageContainer.appendChild(messageElement);
        
        // Scroll to bottom
        this.scrollToBottom();
        
        // Add to history
        this.chatHistory.push({
            role: 'bot',
            content: message
        });
    }
    
    /**
     * Add a system message to the chat
     * @param {string} message - Message text
     */
    addSystemMessage(message) {
        const messageElement = createElement('div', {
            classList: ['message', 'system']
        });
        
        const contentElement = createElement('div', {
            classList: ['message-content']
        }, message);
        
        messageElement.appendChild(contentElement);
        this.messageContainer.appendChild(messageElement);
        
        // Scroll to bottom
        this.scrollToBottom();
    }
    
    /**
     * Add thinking indicator
     */
    addThinkingIndicator() {
        // Remove existing indicator if any
        this.removeThinkingIndicator();
        
        // Create new indicator
        const messageElement = createElement('div', {
            classList: ['message', 'bot', 'thinking'],
            id: 'thinking-indicator'
        });
        
        const contentElement = createElement('div', {
            classList: ['message-content']
        }, '<i class="fa-solid fa-spinner fa-spin"></i> Thinking...');
        
        messageElement.appendChild(contentElement);
        this.messageContainer.appendChild(messageElement);
        
        // Scroll to bottom
        this.scrollToBottom();
    }
    
    /**
     * Remove thinking indicator
     */
    removeThinkingIndicator() {
        const indicator = document.getElementById('thinking-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    /**
     * Scroll chat to bottom
     */
    scrollToBottom() {
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }
    
    /**
     * Clear chat history
     */
    clearChat() {
        this.messageContainer.innerHTML = '';
        this.chatHistory = [];
        this.selectedContextItems = [];
        this.updateContextList();
        this.closeAllPanels();
    }
}














/**
 * Main application initialization
 */
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the chat manager
    const chatManager = new ChatManager();
    window.chatManager = chatManager;  // Make available globally for debugging
    
    // Set up event listeners
    setupEventListeners(chatManager);
    
    // Check server status
    checkServerStatus();
    
    // Auto-focus the input field
    document.getElementById('chat-input').focus();
});

/**
 * Set up event listeners for UI elements
 * @param {ChatManager} chatManager - Chat manager instance
 */
function setupEventListeners(chatManager) {
    // Clear cache button
    document.getElementById('clear-cache').addEventListener('click', handleClearCache);
    
    // Clear chat button
    document.getElementById('clear-chat').addEventListener('click', () => {
        chatManager.clearChat();
        chatManager.addSystemMessage('Chat history cleared.');
    });
    
    // Help button
    document.getElementById('help-button').addEventListener('click', () => {
        document.getElementById('help-modal').classList.add('visible');
    });
    
    // Close modal buttons
    document.querySelectorAll('.close-modal').forEach(button => {
        button.addEventListener('click', () => {
            document.querySelectorAll('.modal').forEach(modal => {
                modal.classList.remove('visible');
            });
        });
    });
    
    // Find solution button
    document.getElementById('find-solution-btn').addEventListener('click', () => {
        chatManager.findSolution();
    });
    
    // Add to context button
    document.getElementById('add-to-context-btn').addEventListener('click', () => {
        if (chatManager.detailsModalItem) {
            chatManager.addItemToContext(chatManager.detailsModalItem);
            document.getElementById('details-modal').classList.remove('visible');
        }
    });
    
    // Close modal when clicking outside
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', function(event) {
            if (event.target === this) {
                this.classList.remove('visible');
            }
        });
    });
}

/**
 * Check server status and update UI
 */
async function checkServerStatus() {
    try {
        // Update status indicators
        const jiraStatus = document.getElementById('jira-status');
        const confluenceStatus = document.getElementById('confluence-status');
        
        // Initially set to pending
        jiraStatus.className = 'status pending';
        jiraStatus.textContent = 'Connecting...';
        confluenceStatus.className = 'status pending';
        confluenceStatus.textContent = 'Connecting...';
        
        // Get status from server
        const response = await api.getStatus();
        
        // Update Jira status
        if (response.connections.jira) {
            jiraStatus.className = 'status connected';
            jiraStatus.textContent = 'Connected';
        } else {
            jiraStatus.className = 'status disconnected';
            jiraStatus.textContent = 'Disconnected';
        }
        
        // Update Confluence status
        if (response.connections.confluence) {
            confluenceStatus.className = 'status connected';
            confluenceStatus.textContent = 'Connected';
        } else {
            confluenceStatus.className = 'status disconnected';
            confluenceStatus.textContent = 'Disconnected';
        }
        
        // Check again after 30 seconds
        setTimeout(checkServerStatus, 30000);
    } catch (error) {
        console.error('Error checking status:', error);
        
        // Set all to disconnected
        document.getElementById('jira-status').className = 'status disconnected';
        document.getElementById('jira-status').textContent = 'Disconnected';
        document.getElementById('confluence-status').className = 'status disconnected';
        document.getElementById('confluence-status').textContent = 'Disconnected';
        
        // Try again after 5 seconds
        setTimeout(checkServerStatus, 5000);
    }
}

/**
 * Handle clear cache button click
 */
async function handleClearCache() {
    try {
        await api.clearCache();
        showNotification('Cache cleared successfully', 'success');
    } catch (error) {
        console.error('Error clearing cache:', error);
        showNotification('Failed to clear cache', 'error');
    }
}


















#!/usr/bin/env python3
import requests
import logging
import os
import sys
import json
import re
import time
import hashlib
import pickle
import threading
from datetime import datetime, timedelta
from urllib.parse import quote
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import Counter, defaultdict
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.api_core.exceptions import GoogleAPICallError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("confluence_gemini.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ConfluenceGemini")

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
CACHE_DIR = os.environ.get("CACHE_DIR", "./confluence_cache")
CACHE_EXPIRY_DAYS = int(os.environ.get("CACHE_EXPIRY_DAYS", "7"))
MAX_RESULTS_PER_QUERY = int(os.environ.get("MAX_RESULTS_PER_QUERY", "50"))

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class HTMLFilter:
    def __init__(self):
        self.text = ""
    
    def feed(self, data):
        self.text = re.sub(r'<[^>]*>', ' ', data)
        self.text = re.sub(r'\s+', ' ', self.text).strip()

class ConfluenceClient:
    """Client for Confluence REST API operations with comprehensive error handling."""
    
    def __init__(self, base_url, username, api_token):
        """
        Initialize the Confluence client with authentication details.
        
        Args:
            base_url: The base URL of the Confluence instance (e.g., https://company.atlassian.net)
            username: The username for authentication
            api_token: The API token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.session = requests.Session()
        logger.info(f"Initialized Confluence client for {self.base_url}")
    
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            response = self.session.get(
                f"{self.base_url}/rest/api/content",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                verify=False  # Setting SSL verify to False as requested
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response content (first 500 chars): {raw_content[:500]}...")
            
            # Handle empty response
            if not raw_content.strip():
                logger.warning("Empty response received during connection test")
                return True  # Still consider it a success if status code is OK
            
            try:
                response.json()
                logger.info("Connection successful!")
                return True
            except json.JSONDecodeError as e:
                logger.error(f"Response content: {raw_content}")
                return False
        except json.JSONDecodeError as e:
            logger.error(f"Connection successful but received invalid JSON: {str(e)}")
            logger.error(f"Response content: {raw_content}")
            return False
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return False
    
    def get_content_by_id(self, content_id, expand=None):
        """
        Get content by ID with optional expansion parameters.
        
        Args:
            content_id: The ID of the content to retrieve
            expand: Comma-separated list of properties to expand (e.g., "body.storage,version,space")
        """
        try:
            params = {}
            if expand:
                params["expand"] = expand
            
            logger.info(f"Fetching content with ID: {content_id}")
            response = self.session.get(
                f"{self.base_url}/rest/api/content/{content_id}",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response content (content by ID): {raw_content[:500]}...")
            
            # Handle empty response
            if not raw_content.strip():
                logger.warning("Empty response received when retrieving content")
                return None
            
            try:
                content = response.json()
                logger.info(f"Successfully retrieved content: {content.get('title', 'Unknown title')}")
                return content
            except json.JSONDecodeError as e:
                logger.error(f"Response content: {raw_content}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for content ID {content_id}: {str(e)}")
            logger.error(f"Response content: {raw_content}")
            return None
        except Exception as e:
            logger.error(f"Connection successful but received invalid JSON: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def get_page_content(self, page_id):
        """
        Get the content of a page in a suitable format for NLP.
        This extracts and processes the content to be more suitable for embeddings.
        
        Args:
            page_id: The ID of the page
        """
        try:
            page = self.get_content_by_id(page_id, expand="body.storage,metadata.labels")
            if not page:
                return None
            
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "url": f"{self.base_url}/wiki/spaces/{page.get('_expandable', {}).get('space', '').split('/')[-1]}/pages/{page.get('id')}",
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
            }
            
            # Get raw content
            content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process HTML content to plain text
            html_filter = HTMLFilter()
            html_filter.feed(content)
            plain_text = html_filter.text
            
            return {
                "metadata": metadata,
                "content": plain_text,
                "raw_html": content
            }
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None
    
    def get_all_content(self, content_type="page", limit=100, expand=None):
        """
        Retrieve all content of specified type with pagination handling.
        
        Args:
            content_type: Type of content to retrieve (default: page)
            limit: Maximum number of results per request
            expand: Properties to expand in results
        """
        all_content = []
        start = 0
        limit = 25  # Confluence API commonly uses 25 as default
        
        logger.info(f"Retrieving all {content_type} content")
        
        while True:
            try:
                params = {
                    "type": content_type,
                    "limit": limit,
                    "start": start
                }
                if expand:
                    params["expand"] = expand
                
                response = self.session.get(
                    f"{self.base_url}/rest/api/content",
                    auth=self.auth,
                    headers=self.headers,
                    params=params,
                    verify=False
                )
                response.raise_for_status()
                
                # Print raw response for debugging
                raw_content = response.text
                logger.info(f"Raw response content (all content): {raw_content[:500]}...")
                
                # Handle empty response
                if not raw_content.strip():
                    logger.warning("Empty response received when retrieving all content")
                    break
                
                try:
                    data = response.json()
                    results = data.get("results", [])
                    if not results:
                        break
                    
                    all_content.extend(results)
                    logger.info(f"Retrieved {len(results)} {content_type} content (total: {len(all_content)})")
                    
                    # Check if there are more pages
                    if len(results) < limit:
                        break
                    
                    start += limit
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON for {content_type}: {str(e)}")
                    logger.error(f"Response content: {raw_content}")
                    break
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON when retrieving content: {str(e)}")
                logger.error(f"Response content: {raw_content}")
                break
            except Exception as e:
                logger.error(f"Error retrieving content: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_details = e.response.json()
                        logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                    except:
                        logger.error(f"Response content: {e.response.text}")
                break
        
        logger.info(f"Retrieved a total of {len(all_content)} {content_type}")
        return all_content
    
    def get_spaces(self, limit=25, start=0):
        """
        Get all spaces the user has access to.
        
        Args:
            limit: Maximum number of results per request
            start: Starting index for pagination
        """
        try:
            params = {
                "limit": limit,
                "start": start
            }
            
            logger.info("Fetching spaces...")
            response = self.session.get(
                f"{self.base_url}/rest/api/space",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response content (spaces): {raw_content[:500]}...")
            
            # Handle empty response
            if not raw_content.strip():
                logger.warning("Empty response received when fetching spaces")
                return {"results": []}
            
            try:
                spaces = response.json()
                logger.info(f"Successfully retrieved {len(spaces.get('results', []))} spaces")
                return spaces
            except json.JSONDecodeError as e:
                logger.error(f"Response content: {raw_content}")
                return {"results": []}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for spaces: {str(e)}")
            logger.error(f"Response content: {raw_content}")
            return {"results": []}
        except Exception as e:
            logger.error(f"Failed to get spaces: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return {"results": []}
    
    def get_all_spaces(self):
        """Retrieve all spaces with pagination handling."""
        all_spaces = []
        start = 0
        limit = 25  # Confluence API commonly uses 25 as default
        
        logger.info("Retrieving all spaces")
        
        while True:
            spaces = self.get_spaces(limit=limit, start=start)
            results = spaces.get("results", [])
            if not results:
                break
            
            all_spaces.extend(results)
            logger.info(f"Retrieved {len(results)} spaces (total: {len(all_spaces)})")
            
            # Check if there are more spaces
            if len(results) < limit:
                break
            
            start += limit
        
        logger.info(f"Retrieved a total of {len(all_spaces)} spaces")
        return all_spaces
    
    def search_content(self, cql=None, title=None, content_type="page", expand=None, limit=10, start=0):
        """
        Search for content using CQL or specific parameters.
        
        Args:
            cql: Confluence Query Language string
            title: Title to search for
            content_type: Type of content to search for (default: page)
            expand: Properties to expand in results
            limit: Maximum number of results to return
            start: Starting index for pagination
        """
        try:
            params = {}
            
            # Build CQL if not provided
            query_parts = []
            if content_type:
                query_parts.append(f"type={content_type}")
            if title:
                # Fix for special characters in title
                safe_title = title.replace('"', '\\"')
                query_parts.append(f'title~"{safe_title}"')
            
            if query_parts:
                params["cql"] = " AND ".join(query_parts)
            
            # Override with explicit CQL if provided
            if cql:
                params["cql"] = cql
            
            if expand:
                params["expand"] = expand
            
            params["limit"] = limit
            params["start"] = start
            
            logger.info(f"Searching for content with params: {params}")
            response = self.session.get(
                f"{self.base_url}/rest/api/content/search",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_content = response.text
            logger.info(f"Raw response content (search): {raw_content[:500]}...")
            
            # Handle empty response
            if not raw_content.strip():
                logger.warning("Empty response received for search query")
                return {"results": []}
            
            try:
                results = response.json()
                logger.info(f"Search returned {len(results.get('results', []))} results")
                return results
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for search: {str(e)}")
                logger.error(f"Response content: {raw_content}")
                return {"results": []}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for search: {str(e)}")
            logger.error(f"Response content: {raw_content}")
            return {"results": []}
        except Exception as e:
            logger.error(f"Failed to search content: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            return {"results": []}

class TextProcessor:
    """Class for advanced text processing without relying on spaCy."""
    
    def __init__(self):
        """Initialize text processing components."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Add domain-specific stop words if needed
        self.domain_stop_words = {'the', 'is', 'in', 'to', 'and', 'of', 'a', 'for', 'on', 'with'}
        self.stop_words.update(self.domain_stop_words)
        
        logger.info("Initialized text processor")
    
    def preprocess_text(self, text):
        """
        Preprocess text by tokenizing, removing stop words, and lemmatizing.
        
        Args:
            text: Input text to process
            
        Returns:
            Dict containing original, tokens, filtered_tokens, lemmatized_tokens, and segments
        """
        if not text or not isinstance(text, str):
            return {
                "original": "",
                "tokens": [],
                "filtered_tokens": [],
                "lemmatized_tokens": [],
                "segments": []
            }
        
        # Segment text into sentences
        segments = sent_tokenize(text)
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stop words and non-alphanumeric tokens
        filtered_tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        
        # Lemmatize tokens
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        
        return {
            "original": text,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "lemmatized_tokens": lemmatized_tokens,
            "segments": segments
        }
    
    def extract_keywords(self, text, top_n=10):
        """
        Extract the most important keywords from text.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        processed = self.preprocess_text(text)
        word_freq = Counter(processed["lemmatized_tokens"])
        return [word for word, freq in word_freq.most_common(top_n)]
    
    def segment_text(self, text, max_segment_length=500):
        """
        Segment text into smaller chunks while respecting sentence boundaries.
        
        Args:
            text: Input text to segment
            max_segment_length: Maximum character length for each segment
            
        Returns:
            List of text segments
        """
        sentences = sent_tokenize(text)
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            if len(current_segment) + len(sentence) <= max_segment_length:
                current_segment += " " + sentence if current_segment else sentence
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence
        
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments
    
    def preprocess_question(self, question):
        """
        Preprocess a question to extract key components and intent.
        
        Args:
            question: User question
            
        Returns:
            Dict with processed question information
        """
        processed = self.preprocess_text(question)
        
        # Detect question type based on first word
        question_words = {"what", "who", "where", "when", "why", "how", "which", "can", "do", "is", "are", "will"}
        tokens = processed["tokens"]
        
        question_type = "unknown"
        if tokens and tokens[0].lower() in question_words:
            question_type = tokens[0].lower()
        
        # Extract keywords
        keywords = self.extract_keywords(question, top_n=5)
        
        # Detect if it's a multi-part question by looking for specific patterns
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
            "is_multi_part": is_multi_part
        }
    
    def split_multi_part_question(self, question):
        """
        Split a multi-part question into individual questions.
        
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

class ConfluenceCache:
    """Handles caching of Confluence content to avoid repeated API calls."""
    
    def __init__(self, cache_dir=CACHE_DIR, expiry_days=CACHE_EXPIRY_DAYS):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            expiry_days: Number of days after which cache should be considered stale
        """
        self.cache_dir = cache_dir
        self.expiry_days = expiry_days
        self.memory_cache = {}
        self.content_index = defaultdict(list)
        self.metadata = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing index if available
        self.load_index()
        
        logger.info(f"Initialized Confluence cache in {self.cache_dir} with {expiry_days} day expiry")
    
    def get_cache_path(self, key):
        """Get the file path for a cache item."""
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.pickle")
    
    def is_cache_valid(self, key):
        """Check if a cache item exists and is still valid (not expired)."""
        cache_path = self.get_cache_path(key)
        
        if key in self.memory_cache:
            return True
        
        if os.path.exists(cache_path):
            # Check if the cache file is recent enough
            modification_time = os.path.getmtime(cache_path)
            modification_date = datetime.fromtimestamp(modification_time)
            expiry_date = datetime.now() - timedelta(days=self.expiry_days)
            
            return modification_date > expiry_date
        
        return False
    
    def get(self, key):
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if not self.is_cache_valid(key):
            return None
        
        # Try memory cache first
        if key in self.memory_cache:
            logger.info(f"Cache hit (memory): {key}")
            return self.memory_cache[key]
        
        # Then try file cache
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
                # Store in memory cache for faster future access
                self.memory_cache[key] = value
                logger.info(f"Cache hit (file): {key}")
                return value
        except Exception as e:
            logger.error(f"Error reading cache for {key}: {str(e)}")
            return None
    
    def set(self, key, value):
        """
        Store a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Store in memory cache
        self.memory_cache[key] = value
        
        # Store in file cache
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logger.info(f"Cached: {key}")
        except Exception as e:
            logger.error(f"Error writing cache for {key}: {str(e)}")
    
    def clear(self, key=None):
        """
        Clear cache items.
        
        Args:
            key: Specific key to clear, or None to clear all
        """
        if key:
            # Clear specific key
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            cache_path = self.get_cache_path(key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
            logger.info(f"Cleared cache for {key}")
        else:
            # Clear all cache
            self.memory_cache = {}
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pickle'):
                    os.remove(os.path.join(self.cache_dir, file))
            logger.info("Cleared all cache")
    
    def build_index(self, content_list):
        """
        Build a searchable index from content.
        
        Args:
            content_list: List of content items to index
        """
        text_processor = TextProcessor()
        
        for content in content_list:
            if not content or not isinstance(content, dict):
                continue
            
            content_id = content.get("metadata", {}).get("id")
            if not content_id:
                continue
            
            # Store metadata
            self.metadata[content_id] = content.get("metadata", {})
            
            # Process text
            text = content.get("content", "")
            processed = text_processor.preprocess_text(text)
            
            # Index by lemmatized tokens
            for token in processed["lemmatized_tokens"]:
                self.content_index[token].append(content_id)
        
        logger.info(f"Built index with {len(self.content_index)} terms and {len(self.metadata)} documents")
        
        # Save the index
        self.save_index()
    
    def search_index(self, query, max_results=10):
        """
        Search the index for relevant content.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of content IDs sorted by relevance
        """
        if not self.content_index:
            logger.warning("Index is empty, search returned no results")
            return []
        
        text_processor = TextProcessor()
        processed_query = text_processor.preprocess_text(query)
        
        # Get scores for each document
        scores = defaultdict(int)
        
        # Score based on lemmatized tokens
        for token in processed_query["lemmatized_tokens"]:
            for content_id in self.content_index.get(token, []):
                scores[content_id] += 1
        
        # Sort by score (descending)
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top results
        top_results = [content_id for content_id, score in sorted_results[:max_results]]
        
        logger.info(f"Search for '{query}' returned {len(top_results)} results")
        
        return top_results
    
    def save_index(self):
        """Save the index to disk."""
        try:
            index_path = os.path.join(self.cache_dir, "index.pickle")
            metadata_path = os.path.join(self.cache_dir, "metadata.pickle")
            
            with open(index_path, 'wb') as f:
                pickle.dump(dict(self.content_index), f)
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            logger.info(f"Saved index with {len(self.content_index)} terms and {len(self.metadata)} documents")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def load_index(self):
        """Load the index from disk."""
        try:
            index_path = os.path.join(self.cache_dir, "index.pickle")
            metadata_path = os.path.join(self.cache_dir, "metadata.pickle")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                with open(index_path, 'rb') as f:
                    self.content_index = defaultdict(list, pickle.load(f))
                
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                logger.info(f"Loaded index with {len(self.content_index)} terms and {len(self.metadata)} documents")
            else:
                logger.info("No existing index found")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            # Initialize empty index and metadata
            self.content_index = defaultdict(list)
            self.metadata = {}

class GeminiClient:
    """Wrapper for interacting with Google's Gemini API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
        logger.info(f"Initialized Gemini client with model {MODEL_NAME}")
    
    def generate_response(self, prompt, system_prompt=None, temperature=0.7, max_tokens=8192):
        """
        Generate a response from Gemini.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            logger.info(f"Generating response from Gemini with temperature {temperature}")
            logger.info(f"System prompt length: {len(system_prompt) if system_prompt else 0}")
            logger.info(f"User prompt length: {len(prompt)}")
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=max_tokens,
            )
            
            # Build the full prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Generate response with streaming
            response_text = ""
            for chunk in self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True,
            ):
                if chunk.candidates and chunk.candidates[0].text:
                    response_text += chunk.candidates[0].text
                    print(".", end="", flush=True)  # Show progress
            
            print()  # New line after progress dots
            
            logger.info(f"Response length: {len(response_text)} characters")
            return response_text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def build_system_prompt(self, context_docs=None):
        """
        Build a comprehensive system prompt for Gemini.
        
        Args:
            context_docs: List of documents to include as context
            
        Returns:
            System prompt string
        """
        system_prompt = """You are a highly knowledgeable and professional AI assistant that specializes in providing accurate information from a company's Confluence knowledge base. Your responses should be:

1. PRECISE AND ACCURATE: Always base your answers strictly on the provided Confluence documents. If the information isn't in the provided context, clearly state that you don't have that specific information.

2. PROFESSIONAL BUT FRIENDLY: Maintain a professional tone that would be appropriate in a corporate environment, while still being approachable and helpful.

3. WELL-STRUCTURED: For complex responses, use appropriate formatting with headings, bullet points, or numbered lists to improve readability.

4. CONTEXTUALLY AWARE: If you need clarification to provide a better answer, politely ask follow-up questions to better understand the user's needs.

5. SOURCE-TRANSPARENT: Always include references to the specific Confluence pages you used to answer the question. Include the exact page titles and URLs at the end of your response.

When responding to technical questions:
- Be precise with technical terminology
- Include code snippets when relevant
- Explain complex concepts clearly

When data or tables are mentioned in the context:
- Present numerical data clearly, using tables if appropriate
- Explain what the data means in business terms

When images are referenced in the context:
- Clearly describe what information the image contains
- Explain the relevance of the image to the question

If the question seems ambiguous or could have multiple interpretations:
- Consider the most likely interpretation based on the available context
- If necessary, provide answers to multiple interpretations
- Politely ask for clarification

If you don't have enough information to fully answer the question:
- Clearly state what you do know based on the provided context
- Explain what additional information would be needed
- Suggest which Confluence spaces might contain the relevant information

FORMAT YOUR RESPONSES:
1. Start with a direct answer to the question
2. Follow with supporting details, explanations, or elaborations
3. For complex topics, use appropriate headings and structured formatting
4. End with source references listing the Confluence pages you used

Remember that you are assisting with company-internal information. Your responses should be helpful for employees trying to find and understand information in their company's knowledge base.
"""
        
        # Add context documents if provided
        if context_docs and len(context_docs) > 0:
            context_text = "\n\n### CONTEXT DOCUMENTS ###\n\n"
            
            for i, doc in enumerate(context_docs):
                metadata = doc.get("metadata", {})
                title = metadata.get("title", "Untitled Document")
                url = metadata.get("url", "")
                content = doc.get("content", "")
                
                context_text += f"[DOCUMENT {i+1}]: {title}\n"
                context_text += f"URL: {url}\n"
                context_text += f"CONTENT: {content}\n\n"
            
            system_prompt += context_text
        
        return system_prompt

class ConfluenceGeminiBot:
    """Main class that integrates Confluence content with Gemini for answering questions."""
    
    def __init__(self, confluence_url, username, api_token, cache_dir=CACHE_DIR):
        """
        Initialize the bot with Confluence credentials.
        
        Args:
            confluence_url: Base URL of Confluence instance
            username: Confluence username
            api_token: Confluence API token
            cache_dir: Directory to store cache
        """
        self.confluence = ConfluenceClient(confluence_url, username, api_token)
        self.cache = ConfluenceCache(cache_dir)
        self.text_processor = TextProcessor()
        self.gemini = GeminiClient()
        
        # Test connection
        self.confluence.test_connection()
        
        logger.info("Initialized ConfluenceGeminiBot")
    
    def index_all_content(self, force_refresh=False):
        """
        Index all Confluence content for faster searching.
        
        Args:
            force_refresh: Force refreshing content even if cached
        """
        # Check if we already have an index
        if not force_refresh and self.cache.content_index and self.cache.metadata:
            logger.info(f"Using existing index with {len(self.cache.content_index)} terms and {len(self.cache.metadata)} documents")
            return
        
        logger.info("Indexing all Confluence content, this may take a while...")
        
        # Get all content
        all_content = self.confluence.get_all_content(expand="body.storage")
        logger.info(f"Retrieved {len(all_content)} content items")
        
        # Process and cache each content item
        processed_content = []
        
        for content in all_content:
            content_id = content.get("id")
            
            # Skip if already cached and not forcing refresh
            if not force_refresh and self.cache.is_cache_valid(f"content_{content_id}"):
                cached_content = self.cache.get(f"content_{content_id}")
                if cached_content:
                    processed_content.append(cached_content)
                    continue
            
            # Process content
            processed = self.confluence.get_page_content(content_id)
            if processed:
                self.cache.set(f"content_{content_id}", processed)
                processed_content.append(processed)
        
        # Build index
        self.cache.build_index(processed_content)
        
        logger.info(f"Indexed {len(processed_content)} content items")
    
    def search_confluence(self, query, max_results=MAX_RESULTS_PER_QUERY):
        """
        Search Confluence for content related to the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of content items matching the query
        """
        logger.info(f"Searching Confluence for: {query}")
        
        # First try the cached index search
        if self.cache.content_index and self.cache.metadata:
            content_ids = self.cache.search_index(query, max_results=max_results)
            
            if content_ids:
                # Get full content for each ID
                results = []
                for content_id in content_ids:
                    cached_content = self.cache.get(f"content_{content_id}")
                    if cached_content:
                        results.append(cached_content)
                
                logger.info(f"Found {len(results)} results from index search")
                return results
        
        # If index search failed or returned no results, try direct Confluence search
        logger.info("Falling back to direct Confluence search API")
        
        # First, try an exact match search
        search_results = self.confluence.search_content(
            cql=f'text ~ "{query}"',  # Exact phrase match
            content_type="page",
            limit=max_results
        )
        
        # If no results, try a more relaxed search
        if not search_results.get("results"):
            # Use keywords for broader search
            keywords = self.text_processor.extract_keywords(query, top_n=3)
            keyword_query = " OR ".join(keywords)
            
            search_results = self.confluence.search_content(
                cql=f'text ~ "({keyword_query})"',
                content_type="page",
                limit=max_results
            )
        
        # Process search results
        results = []
        for result in search_results.get("results", []):
            content_id = result.get("id")
            
            # Try to get from cache first
            cached_content = self.cache.get(f"content_{content_id}")
            if cached_content:
                results.append(cached_content)
                continue
            
            # If not in cache, get content and cache it
            processed = self.confluence.get_page_content(content_id)
            if processed:
                self.cache.set(f"content_{content_id}", processed)
                results.append(processed)
        
        logger.info(f"Found {len(results)} results from direct Confluence search")
        return results
    
    def answer_question(self, question, temperature=0.7):
        """
        Answer a question using Confluence content and Gemini.
        
        Args:
            question: User's question
            temperature: Temperature for Gemini response generation
            
        Returns:
            Answer from Gemini
        """
        logger.info(f"Answering question: {question}")
        
        # Process the question
        processed_question = self.text_processor.preprocess_question(question)
        
        # Handle multi-part questions
        if processed_question["is_multi_part"]:
            logger.info("Detected multi-part question, splitting into parts")
            question_parts = self.text_processor.split_multi_part_question(question)
            
            if len(question_parts) > 1:
                logger.info(f"Split into {len(question_parts)} parts: {question_parts}")
                
                # Process each part separately
                responses = []
                for i, part in enumerate(question_parts):
                    logger.info(f"Processing question part {i+1}: {part}")
                    part_response = self._process_single_question(part, temperature)
                    responses.append(f"Part {i+1}: {part}\n\n{part_response}")
                
                # Combine responses
                combined_response = "\n\n".join(responses)
                return combined_response
        
        # For single questions or if splitting failed
        return self._process_single_question(question, temperature)
    
    def _process_single_question(self, question, temperature=0.7):
        """
        Process a single question.
        
        Args:
            question: Question to answer
            temperature: Temperature for response generation
            
        Returns:
            Answer to the question
        """
        # Extract keywords for better search
        keywords = self.text_processor.extract_keywords(question)
        logger.info(f"Extracted keywords: {keywords}")
        
        # Search for relevant content
        search_query = " ".join(keywords[:3]) if keywords else question
        relevant_docs = self.search_confluence(search_query)
        
        if not relevant_docs:
            logger.warning(f"No relevant documents found for question: {question}")
            return "I couldn't find any information in the Confluence knowledge base that answers your question. Could you rephrase your question or provide more details?"
        
        # Prepare context for Gemini
        logger.info(f"Found {len(relevant_docs)} relevant documents for the question")
        
        # Build system prompt with context
        system_prompt = self.gemini.build_system_prompt(context_docs=relevant_docs)
        
        # Generate response
        response = self.gemini.generate_response(
            prompt=question,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        return response
    
    def interactive_session(self):
        """Start an interactive Q&A session in the console."""
        print("\n=== Confluence Gemini Bot ===")
        print("Type 'exit' or 'quit' to end the session.")
        
        while True:
            try:
                question = input("\nYour question: ")
                
                if question.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                # Process the question
                print("\nThinking...")
                answer = self.answer_question(question)
                
                print("\nAnswer:")
                print("=" * 80)
                print(answer)
                print("=" * 80)
            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {str(e)}")
                print(f"An error occurred: {str(e)}")

# Main function to run the bot
def main():
    """Main function to run the application."""
    # Configuration
    confluence_url = os.environ.get("CONFLUENCE_URL", "https://your-company.atlassian.net")
    username = os.environ.get("CONFLUENCE_USERNAME", "your.email@company.com")
    api_token = os.environ.get("CONFLUENCE_API_TOKEN", "your-api-token")
    
    # Initialize the bot
    try:
        bot = ConfluenceGeminiBot(confluence_url, username, api_token)
        
        # Index content in background thread to speed up startup
        threading.Thread(target=bot.index_all_content).start()
        
        # Start interactive session
        bot.interactive_session()
    except Exception as e:
        logger.error(f"Error initializing bot: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()






















#!/usr/bin/env python3
import logging
import os
import sys
import json
import re
import pickle
import hashlib
import tempfile
import base64
import urllib3
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings - as requested
urllib3.disable_warnings(InsecureRequestWarning)

# For Google Gemini setup
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.api_core.exceptions import GoogleAPICallError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jira_gemini_chatbot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("JiraGeminiChatbot")

# Configuration (Environment Variables or Config File)
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://jira.yourcompany.com")
JIRA_USERNAME = os.environ.get("JIRA_USERNAME", "your_username")
JIRA_TOKEN = os.environ.get("JIRA_TOKEN", "your_api_token")

# Cache configuration
CACHE_DIR = os.environ.get("CACHE_DIR", str(Path.home() / ".jira_gemini_cache"))
CACHE_EXPIRY = int(os.environ.get("CACHE_EXPIRY", 60 * 60 * 24))  # 24 hours in seconds

class Cache:
    """A local caching system for Jira data."""
    
    def __init__(self, cache_dir: str = CACHE_DIR, expiry: int = CACHE_EXPIRY):
        """Initialize the cache system.
        
        Args:
            cache_dir: Directory to store cache files
            expiry: Cache expiry time in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.expiry = expiry
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache initialized at {self.cache_dir}")
        
    def _get_cache_key(self, key: str) -> str:
        """Generate a unique cache key based on the input."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, key: str) -> Any:
        """Retrieve data from cache if it exists and is not expired."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
            
        try:
            # Check if cache is expired
            mtime = cache_path.stat().st_mtime
            if time.time() - mtime > self.expiry:
                logger.info(f"Cache expired for {key}")
                return None
                
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                logger.info(f"Cache hit for {key}")
                return cached_data
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None
    
    def set(self, key: str, data: Any) -> None:
        """Store data in cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Data cached for {key}")
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")
            
    def invalidate(self, key: str) -> None:
        """Remove a specific item from cache."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Cache invalidated for {key}")
            
    def clear_all(self) -> None:
        """Clear the entire cache."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("All cache cleared")


class JiraClient:
    """Client for interacting with Jira API."""
    
    def __init__(self, base_url: str, username: str, token: str, cache: Optional[Cache] = None):
        """Initialize the Jira client.
        
        Args:
            base_url: Base URL of the Jira instance
            username: Jira username or email
            token: Jira API token
            cache: Cache object for storing Jira data
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, token)
        self.cache = cache or Cache()
        self.session = requests.Session()
        self.session.auth = self.auth
        # Disable SSL verification as requested
        self.session.verify = False
        logger.info(f"Jira client initialized for {base_url}")
        
    def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                     data: Dict = None, headers: Dict = None, 
                     use_cache: bool = True) -> Dict:
        """Make a request to the Jira API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            params: Query parameters
            data: Request body data
            headers: HTTP headers
            use_cache: Whether to use cache for GET requests
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        all_headers = {**default_headers, **(headers or {})}
        
        # Try to get from cache for GET requests
        if method.upper() == 'GET' and use_cache:
            cache_key = f"{method}:{url}:{str(params)}"
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        try:
            logger.info(f"Making {method} request to {url}")
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=all_headers,
                verify=False  # Disable SSL verification
            )
            response.raise_for_status()
            
            if response.status_code == 204:  # No content
                result = {}
            else:
                result = response.json()
                
            # Cache the result for GET requests
            if method.upper() == 'GET' and use_cache:
                cache_key = f"{method}:{url}:{str(params)}"
                self.cache.set(cache_key, result)
                
            return result
        except requests.RequestException as e:
            logger.error(f"Error making request to Jira: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    logger.error(f"Response text: {e.response.text}")
            raise
            
    def test_connection(self) -> bool:
        """Test the connection to Jira.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Use the /serverInfo endpoint to test connection
            info = self._make_request('GET', '/rest/api/2/serverInfo', use_cache=False)
            logger.info(f"Successfully connected to Jira server version: {info.get('version', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Jira: {str(e)}")
            return False
            
    def get_issue(self, issue_key: str, fields: str = None) -> Dict:
        """Get a specific issue by its key.
        
        Args:
            issue_key: The Jira issue key (e.g., PROJ-1)
            fields: Comma-separated list of fields to include
            
        Returns:
            Issue details as dictionary
        """
        params = {}
        if fields:
            params['fields'] = fields
            
        return self._make_request('GET', f'/rest/api/2/issue/{issue_key}', params=params)
        
    def search_issues(self, jql: str, start_at: int = 0, max_results: int = 50, 
                     fields: str = None, expand: str = None) -> Dict:
        """Search for issues using JQL.
        
        Args:
            jql: JQL search string
            start_at: Index of the first result to return
            max_results: Maximum number of results to return
            fields: Comma-separated list of fields to include
            expand: Additional information to expand
            
        Returns:
            Search results as dictionary
        """
        params = {
            'jql': jql,
            'startAt': start_at,
            'maxResults': max_results
        }
        
        if fields:
            params['fields'] = fields
            
        if expand:
            params['expand'] = expand
            
        return self._make_request('GET', '/rest/api/2/search', params=params)
        
    def get_all_issues(self, jql: str, fields: str = None, 
                      max_results: int = 1000) -> List[Dict]:
        """Get all issues matching a JQL query, handling pagination.
        
        Args:
            jql: JQL search string
            fields: Comma-separated list of fields to include
            max_results: Maximum total number of results to return
            
        Returns:
            List of all matching issues
        """
        issues = []
        page_size = 100  # Jira recommends 100 for optimal performance
        start_at = 0
        
        logger.info(f"Retrieving all issues matching JQL: {jql}")
        
        while True:
            page = self.search_issues(
                jql=jql,
                start_at=start_at,
                max_results=min(page_size, max_results - len(issues)),
                fields=fields
            )
            
            if not page.get('issues'):
                break
                
            issues.extend(page['issues'])
            logger.info(f"Retrieved {len(issues)} of {page['total']} issues")
            
            # Check if we've reached the total or our max limit
            if len(issues) >= page['total'] or len(issues) >= max_results:
                break
                
            # Get next page
            start_at += len(page['issues'])
            
            # If no issues were returned, we're done
            if not page['issues']:
                break
                
        logger.info(f"Retrieved a total of {len(issues)} issues")
        return issues
        
    def get_issue_types(self) -> List[Dict]:
        """Get all issue types defined in the Jira instance."""
        return self._make_request('GET', '/rest/api/2/issuetype')
        
    def get_projects(self) -> List[Dict]:
        """Get all projects visible to the authenticated user."""
        return self._make_request('GET', '/rest/api/2/project')
        
    def get_issue_content(self, issue_key: str) -> Dict:
        """Get the full content of an issue in a format suitable for the chatbot.
        
        This method enriches the issue data with additional information like
        comments, attachments, and other related data.
        
        Args:
            issue_key: The Jira issue key (e.g., PROJ-1)
            
        Returns:
            Enriched issue data as dictionary
        """
        # Get issue with all fields and expansion
        issue = self.get_issue(
            issue_key=issue_key, 
            fields="*all"
        )
        
        # Extract key metadata
        metadata = {
            "key": issue.get("key"),
            "summary": issue.get("fields", {}).get("summary"),
            "issuetype": issue.get("fields", {}).get("issuetype", {}).get("name"),
            "status": issue.get("fields", {}).get("status", {}).get("name"),
            "created": issue.get("fields", {}).get("created"),
            "updated": issue.get("fields", {}).get("updated"),
        }
        
        # Extract people
        if "assignee" in issue.get("fields", {}):
            metadata["assignee"] = issue.get("fields", {}).get("assignee", {}).get("displayName", "Unassigned")
            
        if "reporter" in issue.get("fields", {}):
            metadata["reporter"] = issue.get("fields", {}).get("reporter", {}).get("displayName", "Unknown")
            
        # Extract content fields
        content_parts = []
        
        # Add summary
        summary = issue.get("fields", {}).get("summary", "")
        if summary:
            content_parts.append(f"Summary: {summary}")
            
        # Add description
        # Note: This is simplified and assumes the description is in plain text
        # In reality, Jira descriptions can be in various formats
        description = issue.get("fields", {}).get("description", "")
        if description:
            # Try to extract text from Atlassian Document Format (ADF)
            if isinstance(description, dict):
                # This is a simplified ADF parser, might need enhancement
                description_text = self._extract_text_from_adf(description)
                content_parts.append(f"Description: {description_text}")
            else:
                content_parts.append(f"Description: {description}")
                
        # Add comments
        comments = issue.get("fields", {}).get("comment", {}).get("comments", [])
        for comment in comments:
            author = comment.get("author", {}).get("displayName", "Unknown")
            created = comment.get("created", "")
            # Similar to description, handle ADF format
            comment_body = comment.get("body", "")
            if isinstance(comment_body, dict):
                comment_text = self._extract_text_from_adf(comment_body)
                content_parts.append(f"Comment by {author} on {created}: {comment_text}")
            else:
                content_parts.append(f"Comment by {author} on {created}: {comment_body}")
                
        # Combine all content
        full_content = "\n\n".join(content_parts)
        
        return {
            "metadata": metadata,
            "content": full_content,
            "url": f"{self.base_url}/browse/{issue_key}"
        }
        
    def _extract_text_from_adf(self, adf_doc: Dict) -> str:
        """Extract plain text from Atlassian Document Format (ADF) object.
        
        This is a simplified parser that doesn't handle all ADF features.
        
        Args:
            adf_doc: ADF document object
            
        Returns:
            Extracted plain text
        """
        if not adf_doc or not isinstance(adf_doc, dict):
            return ""
            
        text_parts = []
        
        def extract_from_content(content_list):
            parts = []
            if not content_list or not isinstance(content_list, list):
                return ""
                
            for item in content_list:
                if not isinstance(item, dict):
                    continue
                    
                # Extract text nodes
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                    
                # Extract from content recursively
                if "content" in item and isinstance(item["content"], list):
                    parts.append(extract_from_content(item["content"]))
                    
            return " ".join(parts)
            
        # Extract text from the main content array
        if "content" in adf_doc and isinstance(adf_doc["content"], list):
            text_parts.append(extract_from_content(adf_doc["content"]))
            
        # Extract text from the version array if it exists
        if "version" in adf_doc:
            text_parts.append(f"Document version: {adf_doc['version']}")
            
        return " ".join(text_parts)
        
    def process_attachments(self, issue_key: str, temp_dir: str = None) -> List[Dict]:
        """Process and analyze attachments from an issue.
        
        Args:
            issue_key: The Jira issue key (e.g., PROJ-1)
            temp_dir: Temporary directory to save attachments
            
        Returns:
            List of attachment info dictionaries with metadata and content
        """
        # Get issue with attachment fields
        issue = self.get_issue(issue_key, fields="attachment")
        attachments = issue.get("fields", {}).get("attachment", [])
        
        if not attachments:
            logger.info(f"No attachments found for issue {issue_key}")
            return []
            
        # Create temp dir if not provided
        if not temp_dir:
            temp_dir = tempfile.mkdtemp(prefix="jira_attachments_")
        else:
            os.makedirs(temp_dir, exist_ok=True)
            
        logger.info(f"Processing {len(attachments)} attachments for issue {issue_key}")
        
        processed_attachments = []
        for attachment in attachments:
            attachment_id = attachment.get("id")
            filename = attachment.get("filename")
            content_type = attachment.get("mimeType")
            
            if not attachment_id or not filename:
                continue
                
            try:
                # Construct the attachment URL
                attachment_url = f"{self.base_url}/secure/attachment/{attachment_id}/{filename}"
                
                # Download the attachment
                response = self.session.get(attachment_url, verify=False)
                response.raise_for_status()
                
                # Save the attachment to a temporary file
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(response.content)
                    
                # Process based on content type
                attachment_info = {
                    "id": attachment_id,
                    "filename": filename,
                    "content_type": content_type,
                    "file_path": file_path,
                    "size": len(response.content),
                    "download_url": attachment_url
                }
                
                # Extract text for text-based attachments
                if content_type in ["text/plain", "text/csv", "application/json"]:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        attachment_info["text_content"] = f.read()
                        
                # For images, encode as base64 for potential use with image models
                elif content_type.startswith("image/"):
                    attachment_info["is_image"] = True
                    attachment_info["base64_content"] = base64.b64encode(response.content).decode("utf-8")
                    
                processed_attachments.append(attachment_info)
                logger.info(f"Processed attachment: {filename}")
                
            except Exception as e:
                logger.error(f"Error processing attachment {filename}: {str(e)}")
                
        return processed_attachments


class GeminiAI:
    """Class for interacting with Gemini models via Vertex AI."""
    
    def __init__(self, project_id: str, location: str, model_name: str):
        """Initialize the Gemini AI interface.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
            model_name: Gemini model name
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.conversation_history = []
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
        logger.info(f"Gemini AI initialized with model {model_name}")
        
    def create_enhanced_system_prompt(self) -> str:
        """Create an enhanced system prompt for Gemini.
        
        Returns:
            Detailed system prompt string
        """
        return """
        You are JiraGPT, an advanced AI assistant specialized in providing information from Jira tickets and answering questions in a professional and friendly manner. Follow these guidelines:

        1. RESPONSE STYLE:
           - Use a professional yet conversational tone.
           - Be precise, accurate, and concise in your answers.
           - Respond with confidence when you have information.
           - Readily acknowledge when you're uncertain or need more details.
           - Adjust response formats based on the question (use tables, bullet points, or paragraphs as appropriate).
           - Be proactive in offering assistance beyond the immediate question when relevant.

        2. JIRA EXPERTISE:
           - You have access to Jira tickets, their fields, comments, attachments, and related information.
           - When discussing tickets, always include the ticket ID/key and direct links when available.
           - When referencing information from Jira, specify which ticket and field it came from.
           - For complex Jira data, organize information logically with headers and structured formats.
           - Understand Jira terminology (epics, stories, subtasks, components, fix versions, etc.).

        3. HANDLING IMAGES AND TABLES:
           - Describe the content of images in Jira tickets when relevant.
           - Interpret and explain table data clearly, with summaries when tables are complex.
           - Format tabular data appropriately in your responses when applicable.

        4. CONVERSATIONAL CAPABILITY:
           - Ask clarifying questions when the request is ambiguous to provide more accurate answers.
           - Maintain context throughout the conversation and reference previous exchanges when relevant.
           - Anticipate follow-up questions and provide information that might be helpful.
           - Break down complex answers into digestible parts.

        5. SOURCE ATTRIBUTION:
           - Always provide source links for information extracted from Jira.
           - Clearly indicate when information comes from attachments or comments.
           - Distinguish between factual information from Jira and your own analysis or suggestions.

        6. HANDLING DIFFERENT QUESTION TYPES:
           - For specific ticket questions: Provide detailed, focused information on that ticket.
           - For general questions: Give broader context and helpful information.
           - For irrelevant questions: Politely redirect to Jira-related topics or provide general help.
           - For complex analysis: Break down information methodically and highlight key insights.

        7. ERROR HANDLING:
           - If data is missing, acknowledge the gap and suggest alternative approaches.
           - If you can't answer a question, explain why and offer to help in other ways.
           - If a request is too broad, ask for specification or provide the most relevant information.

        8. PRIVACY AND SECURITY:
           - Never share sensitive information marked as confidential.
           - Do not speculate about security issues without clear data.
           - Do not make up information if it's not available in the Jira data.

        Remember that you are here to make users' interaction with Jira data easier, more efficient, and more productive. Always prioritize being helpful, accurate, and respectful of time.

        When you don't know the answer or don't have the requested information, say so clearly rather than making up a response.
        """
        
    def generate_response(self, 
                         prompt: str, 
                         jira_context: Dict = None, 
                         attachments: List[Dict] = None,
                         conversation_context: bool = True,
                         temperature: float = 0.4) -> str:
        """Generate a response using Gemini.
        
        Args:
            prompt: User query or prompt
            jira_context: Context from Jira data
            attachments: List of processed attachments
            conversation_context: Whether to use conversation history
            temperature: Temperature for generation (higher = more creative)
            
        Returns:
            Generated response text
        """
        # Build the context with Jira data if available
        context_parts = []
        
        if jira_context:
            # Format ticket metadata as a table
            metadata = jira_context.get("metadata", {})
            if metadata:
                metadata_text = "JIRA TICKET INFORMATION:\n"
                for key, value in metadata.items():
                    metadata_text += f"{key}: {value}\n"
                context_parts.append(metadata_text)
                
            # Add ticket content
            content = jira_context.get("content", "")
            if content:
                context_parts.append(f"TICKET CONTENT:\n{content}")
                
            # Add ticket URL
            url = jira_context.get("url", "")
            if url:
                context_parts.append(f"TICKET URL: {url}")
                
        # Add attachment information
        if attachments:
            attachment_text = f"ATTACHMENTS ({len(attachments)}):\n"
            for i, attachment in enumerate(attachments, 1):
                attachment_text += f"{i}. {attachment.get('filename')} ({attachment.get('content_type')}, {attachment.get('size')} bytes)\n"
                
                # Add text content for text attachments
                if "text_content" in attachment:
                    attachment_text += f"Content preview: {attachment.get('text_content')[:500]}...\n"
                elif attachment.get("is_image", False):
                    attachment_text += f"[Image attachment]\n"
                    
            context_parts.append(attachment_text)
            
        # Build the full prompt
        system_prompt = self.create_enhanced_system_prompt()
        context_text = "\n\n".join(context_parts) if context_parts else ""
        
        # Add the conversation history if enabled
        if conversation_context and self.conversation_history:
            history_text = "\n\n".join([
                f"User: {exchange[0]}\nAssistant: {exchange[1]}"
                for exchange in self.conversation_history
            ])
            full_prompt = f"{system_prompt}\n\nCONVERSATION HISTORY:\n{history_text}\n\nJIRA CONTEXT:\n{context_text}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = f"{system_prompt}\n\nJIRA CONTEXT:\n{context_text}\n\nUser: {prompt}\nAssistant:"
            
        # Configure generation parameters
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
        )
        
        # Generate the response
        try:
            logger.info("Generating response from Gemini...")
            response_text = ""
            
            # Stream the response to show progress
            for chunk in self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True,
            ):
                if chunk.candidates and chunk.candidates[0].text:
                    chunk_text = chunk.candidates[0].text
                    response_text += chunk_text
                    print(chunk_text, end="", flush=True)
                    
            print()  # New line after streaming
            
            # Update conversation history
            self.conversation_history.append((prompt, response_text))
            
            # Keep only the last 10 exchanges to prevent context overflow
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
                
            return response_text
            
        except GoogleAPICallError as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return f"I encountered a technical issue while generating a response. Error: {str(e)}"
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return "I encountered an unexpected error. Please try again or rephrase your question."
            
    def detect_follow_up_questions(self, user_prompt: str) -> List[str]:
        """Detect if follow-up questions are needed for a user prompt.
        
        Args:
            user_prompt: The user's prompt/question
            
        Returns:
            List of follow-up questions if needed, empty list otherwise
        """
        # Short prompts often need clarification
        if len(user_prompt.split()) < 5:
            # Generate a compact prompt for the clarification model
            clarification_prompt = f"""
            The user has asked: "{user_prompt}"
            This query seems vague or ambiguous. Generate 1-2 brief clarifying questions to better understand what the user is looking for.
            Only generate questions if truly necessary. If the query is clear enough, respond with "CLEAR".
            Keep questions concise and focused.
            """
            
            # Configure for quick, minimal response
            generation_config = GenerationConfig(
                temperature=0.3,
                top_p=0.95,
                max_output_tokens=500,
            )
            
            try:
                response = self.model.generate_content(
                    clarification_prompt,
                    generation_config=generation_config,
                )
                
                if response.candidates and response.candidates[0].text:
                    result = response.candidates[0].text.strip()
                    
                    # If model thinks query is clear, return empty list
                    if "CLEAR" in result:
                        return []
                        
                    # Extract questions, assuming one per line
                    questions = [q.strip() for q in result.split('\n') if '?' in q]
                    
                    # Limit to at most 2 questions
                    return questions[:2]
            except Exception as e:
                logger.error(f"Error detecting follow-up questions: {str(e)}")
                
        return []
        
    def analyze_jira_data(self, jira_data: List[Dict], query: str) -> str:
        """Analyze a collection of Jira data to answer a specific query.
        
        Args:
            jira_data: List of Jira ticket data
            query: The user's query to answer
            
        Returns:
            Analysis response
        """
        if not jira_data:
            return "No Jira data available for analysis."
            
        # Prepare the data for analysis
        data_summary = []
        for i, ticket in enumerate(jira_data[:20]):  # Limit to first 20 for context size
            metadata = ticket.get("metadata", {})
            summary = f"Ticket #{i+1}: {metadata.get('key', 'Unknown')} - {metadata.get('summary', 'No summary')}"
            summary += f"\nType: {metadata.get('issuetype', 'Unknown')}, Status: {metadata.get('status', 'Unknown')}"
            summary += f"\nURL: {ticket.get('url', 'No URL')}"
            data_summary.append(summary)
            
        data_context = "\n\n".join(data_summary)
        
        # Create an analysis prompt
        analysis_prompt = f"""
        JIRA DATA FOR ANALYSIS:
        {data_context}
        
        USER QUERY: {query}
        
        Please analyze the above Jira data to answer the user's query. 
        Be thorough but concise, focus on the most relevant information, and format your response clearly.
        Include ticket IDs and links where appropriate.
        If the data doesn't contain information needed to answer the query, acknowledge this limitation.
        """
        
        # Configure generation parameters for analysis
        generation_config = GenerationConfig(
            temperature=0.3,
            top_p=0.95,
            max_output_tokens=8192,
        )
        
        try:
            logger.info("Generating Jira data analysis...")
            response_text = ""
            
            # Stream the response
            for chunk in self.model.generate_content(
                analysis_prompt,
                generation_config=generation_config,
                stream=True,
            ):
                if chunk.candidates and chunk.candidates[0].text:
                    chunk_text = chunk.candidates[0].text
                    response_text += chunk_text
                    print(chunk_text, end="", flush=True)
                    
            print()  # New line after streaming
            return response_text
            
        except Exception as e:
            error_msg = f"Error analyzing Jira data: {str(e)}"
            logger.error(error_msg)
            return f"I encountered an error while analyzing the Jira data: {str(e)}"
            

class JiraGeminiChatbot:
    """Main chatbot class that combines Jira and Gemini capabilities."""
    
    def __init__(self):
        """Initialize the chatbot."""
        # Initialize cache
        self.cache = Cache()
        
        # Initialize Jira client
        self.jira = JiraClient(
            base_url=JIRA_BASE_URL,
            username=JIRA_USERNAME,
            token=JIRA_TOKEN,
            cache=self.cache
        )
        
        # Initialize Gemini AI
        self.gemini = GeminiAI(
            project_id=PROJECT_ID,
            location=REGION,
            model_name=MODEL_NAME
        )
        
        # Pre-fetch data as needed
        self.projects = []
        self.issue_types = []
        
        logger.info("JiraGeminiChatbot initialized")
        
    def start(self):
        """Start the chatbot."""
        print("\n=== Jira Gemini Chatbot ===")
        print("Welcome! I'm your Jira assistant powered by Gemini AI.")
        print("I can help you find and understand information from your Jira instance.")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'clear cache' to clear the cached data.")
        print("Type 'test connection' to test the connection to Jira.")
        print("Type 'help' for more commands.")
        print("============================\n")
        
        # Test Jira connection
        print("Testing connection to Jira...")
        if self.jira.test_connection():
            print("✅ Connected to Jira successfully!")
            
            # Pre-fetch some data for faster responses
            try:
                print("Fetching Jira projects...")
                self.projects = self.jira.get_projects()
                print(f"✅ Fetched {len(self.projects)} projects")
                
                print("Fetching issue types...")
                self.issue_types = self.jira.get_issue_types()
                print(f"✅ Fetched {len(self.issue_types)} issue types")
            except Exception as e:
                print(f"⚠️ Error pre-fetching data: {str(e)}")
        else:
            print("❌ Failed to connect to Jira. Check your credentials and try again.")
            
        self.chat_loop()
        
    def chat_loop(self):
        """Main chat loop for the chatbot."""
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Check for exit command
                if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                    print("\nThank you for using JiraGeminiChatbot. Goodbye!")
                    break
                    
                # Check for clear cache command
                elif user_input.lower() == "clear cache":
                    self.cache.clear_all()
                    print("✅ Cache cleared successfully")
                    continue
                    
                # Check for test connection command
                elif user_input.lower() == "test connection":
                    if self.jira.test_connection():
                        print("✅ Connected to Jira successfully!")
                    else:
                        print("❌ Failed to connect to Jira")
                    continue
                    
                # Check for help command
                elif user_input.lower() == "help":
                    self.show_help()
                    continue
                    
                # Skip empty input
                if not user_input:
                    continue
                    
                # Process the query
                self.process_query(user_input)
                
            except KeyboardInterrupt:
                print("\n\nOperation interrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                logger.error(f"Error in chat loop: {str(e)}", exc_info=True)
                
    def process_query(self, query: str):
        """Process a user query.
        
        Args:
            query: The user's query/prompt
        """
        print("\nProcessing your query...")
        
        # Check if clarification is needed
        follow_ups = self.gemini.detect_follow_up_questions(query)
        
        if follow_ups:
            print("\nTo better assist you, I need a bit more information:")
            for i, question in enumerate(follow_ups, 1):
                print(f"{i}. {question}")
                
            # Get clarification from user
            clarification = input("\nYour clarification: ").strip()
            if clarification:
                # Combine original query with clarification
                query = f"{query} {clarification}"
                print(f"\nProcessing your clarified query: {query}")
                
        # Check if query is about a specific Jira ticket
        ticket_pattern = r'\b([A-Z]+-\d+)\b'
        ticket_matches = re.findall(ticket_pattern, query)
        
        jira_context = None
        attachments = None
        
        if ticket_matches:
            # Query is about specific tickets
            print(f"Found references to Jira tickets: {', '.join(ticket_matches)}")
            
            try:
                # Get the first ticket mentioned for context
                primary_ticket = ticket_matches[0]
                jira_context = self.jira.get_issue_content(primary_ticket)
                print(f"Retrieved data for ticket {primary_ticket}")
                
                # Process attachments if needed
                attachments = self.jira.process_attachments(primary_ticket)
                if attachments:
                    print(f"Processed {len(attachments)} attachments")
                    
            except Exception as e:
                print(f"⚠️ Error retrieving ticket data: {str(e)}")
                logger.error(f"Error retrieving ticket data: {str(e)}")
                
        elif any(keyword in query.lower() for keyword in ["search", "find", "list", "show me", "get"]):
            # This might be a search query
            print("This looks like a search query. Searching Jira...")
            
            try:
                # Extract possible search terms
                search_terms = [term for term in query.split() if len(term) > 3]
                
                if search_terms:
                    # Construct a simple JQL query
                    jql = f'text ~ "{" OR ".join(search_terms)}"'
                    print(f"Searching with JQL: {jql}")
                    
                    # Search for issues
                    issues = self.jira.get_all_issues(jql, max_results=10)
                    
                    if issues:
                        print(f"Found {len(issues)} matching issues")
                        
                        # Process issues to get full content
                        processed_issues = []
                        for issue in issues:
                            issue_key = issue.get("key")
                            if issue_key:
                                issue_data = self.jira.get_issue_content(issue_key)
                                processed_issues.append(issue_data)
                                
                        # Use analysis function to summarize results
                        analysis_response = self.gemini.analyze_jira_data(processed_issues, query)
                        print("\nJiraGPT: ", end="")
                        print(analysis_response)
                        
                        # Skip regular processing since we've handled it here
                        return
                    else:
                        print("No matching issues found")
                        
            except Exception as e:
                print(f"⚠️ Error searching Jira: {str(e)}")
                logger.error(f"Error searching Jira: {str(e)}")
                
        # Generate response
        print("\nJiraGPT: ", end="")
        response = self.gemini.generate_response(
            prompt=query,
            jira_context=jira_context,
            attachments=attachments
        )
        
    def show_help(self):
        """Show help information."""
        print("\n=== JiraGeminiChatbot Help ===")
        print("Available commands:")
        print("  - 'exit', 'quit', 'bye', 'goodbye': Exit the chatbot")
        print("  - 'clear cache': Clear the cached Jira data")
        print("  - 'test connection': Test the connection to Jira")
        print("  - 'help': Show this help message")
        print("\nYou can:")
        print("  - Ask about specific Jira tickets (e.g., 'Tell me about PROJECT-123')")
        print("  - Search for issues (e.g., 'Find all open bugs in the Project X')")
        print("  - Ask general questions about Jira data")
        print("  - Request analysis of multiple tickets")
        print("  - Ask for clarification on Jira terminology")
        print("\nTips:")
        print("  - Be specific when asking about tickets")
        print("  - Include ticket IDs when you know them")
        print("  - For complex questions, provide context")
        print("  - If my response isn't helpful, try rephrasing your question")
        print("===========================")


if __name__ == "__main__":
    # Create and start the chatbot
    try:
        chatbot = JiraGeminiChatbot()
        chatbot.start()
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}", exc_info=True)
        print(f"\n❌ Critical error: {str(e)}")
        print("Check the log file for details: jira_gemini_chatbot.log")
        sys.exit(1)















  












# Enhanced Text Processing Utilities

```python
import re
import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import Counter, defaultdict
from pathlib import Path
import pickle
import os
import hashlib
import time
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger("TextProcessor")

# Download NLTK resources if needed
def download_nltk_resources():
    """Download required NLTK resources."""
    resources = [
        'punkt',  # for tokenization
        'stopwords',  # for stopword removal
        'wordnet',  # for lemmatization
        'averaged_perceptron_tagger'  # for POS tagging
    ]
    
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            logger.info(f"NLTK resource '{resource}' is already downloaded")
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)


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
    """Class for advanced text processing and analysis."""
    
    def __init__(self):
        """Initialize text processing components."""
        # Ensure NLTK resources are downloaded
        download_nltk_resources()
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Add domain-specific stop words
        self.domain_stop_words = {
            'the', 'is', 'in', 'to', 'and', 'of', 'a', 'for', 'on', 'with',
            'jira', 'confluence', 'page', 'issue', 'ticket', 'project', 'user',
            'atlassian', 'key', 'id', 'com', 'org', 'http', 'https', 'www'
        }
        self.stop_words.update(self.domain_stop_words)
        
        logger.info("Initialized text processor with NLTK components")
    
    def preprocess_text(self, text):
        """
        Preprocess text by tokenizing, removing stop words, and lemmatizing.
        
        Args:
            text: Input text to process
            
        Returns:
            Dict containing original, tokens, filtered_tokens, lemmatized_tokens, and segments
        """
        if not text or not isinstance(text, str):
            return {
                "original": "",
                "tokens": [],
                "filtered_tokens": [],
                "lemmatized_tokens": [],
                "stemmed_tokens": [],
                "segments": [],
                "pos_tags": []
            }
        
        # Segment text into sentences
        segments = sent_tokenize(text)
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # POS tagging
        pos_tags = nltk.pos_tag(tokens)
        
        # Remove stop words and non-alphanumeric tokens
        filtered_tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        
        # Lemmatize tokens
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        
        # Stem tokens
        stemmed_tokens = [self.stemmer.stem(token) for token in filtered_tokens]
        
        return {
            "original": text,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "lemmatized_tokens": lemmatized_tokens,
            "stemmed_tokens": stemmed_tokens,
            "segments": segments,
            "pos_tags": pos_tags
        }
    
    def extract_keywords(self, text, top_n=10):
        """
        Extract the most important keywords from text.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        processed = self.preprocess_text(text)
        word_freq = Counter(processed["lemmatized_tokens"])
        return [word for word, freq in word_freq.most_common(top_n)]
    
    def segment_text(self, text, max_segment_length=500, overlap=50):
        """
        Segment text into smaller chunks while respecting sentence boundaries.
        
        Args:
            text: Input text to segment
            max_segment_length: Maximum character length for each segment
            overlap: Number of characters to overlap between segments
            
        Returns:
            List of text segments
        """
        if not text:
            return []
            
        # First split by sentences
        sentences = sent_tokenize(text)
        
        # Group sentences into segments
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max length, save current segment
            if len(current_segment) + len(sentence) > max_segment_length and current_segment:
                segments.append(current_segment.strip())
                # Start new segment with overlap from previous one
                if len(current_segment) > overlap:
                    # Find last complete sentence in the overlap region
                    overlap_sentences = sent_tokenize(current_segment[-overlap:])
                    if overlap_sentences:
                        current_segment = overlap_sentences[-1]
                    else:
                        current_segment = ""
                else:
                    current_segment = ""
            
            # Add the current sentence
            current_segment += " " + sentence if current_segment else sentence
        
        # Add the last segment if not empty
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments
    
    def preprocess_question(self, question):
        """
        Preprocess a question to extract key components and intent.
        
        Args:
            question: User question
            
        Returns:
            Dict with processed question information
        """
        processed = self.preprocess_text(question)
        
        # Detect question type based on first word
        question_words = {"what", "who", "where", "when", "why", "how", "which", "can", "do", "is", "are", "will"}
        tokens = processed["tokens"]
        
        question_type = "unknown"
        if tokens and tokens[0].lower() in question_words:
            question_type = tokens[0].lower()
        
        # Extract keywords
        keywords = self.extract_keywords(question, top_n=5)
        
        # Detect if it's a multi-part question by looking for specific patterns
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
        
        # Extract entities (based on POS tags)
        entities = []
        for word, tag in processed["pos_tags"]:
            # Proper nouns are often entities
            if tag.startswith('NNP'):
                entities.append(word)
        
        return {
            "processed": processed,
            "type": question_type,
            "keywords": keywords,
            "is_multi_part": is_multi_part,
            "entities": entities
        }
    
    def split_multi_part_question(self, question):
        """
        Split a multi-part question into individual questions.
        
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


class EnhancedCache:
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
```











# Enhanced Gemini AI Integration

```python
import logging
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel, Tool, FunctionDeclaration, Content
from vertexai.preview.generative_models import ResponseValidationError
from google.api_core.exceptions import GoogleAPICallError
import json
import time
import re
from typing import Dict, List, Optional, Any

# Configure logging
logger = logging.getLogger("GeminiClient")

class GeminiClient:
    """Advanced client for interacting with Google's Gemini AI."""
    
    def __init__(self, project_id, region, model_name):
        """
        Initialize the Gemini client.
        
        Args:
            project_id: Google Cloud project ID
            region: Google Cloud region
            model_name: Gemini model name
        """
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        self.conversation_history = []
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        self.model = GenerativeModel(model_name)
        logger.info(f"Initialized Gemini client with model {model_name}")
        
        # Define function calling tools
        self.tools = self._define_tools()
    
    def _define_tools(self):
        """Define function calling tools for Gemini."""
        search_jira = FunctionDeclaration(
            name="search_jira",
            description="Search for Jira tickets based on criteria",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or keywords"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project key to filter by"
                    },
                    "status": {
                        "type": "string",
                        "description": "Optional status to filter by (e.g., 'Open', 'In Progress')"
                    },
                    "issueType": {
                        "type": "string",
                        "description": "Optional issue type to filter by (e.g., 'Bug', 'Story')"
                    }
                },
                "required": ["query"]
            }
        )
        
        search_confluence = FunctionDeclaration(
            name="search_confluence",
            description="Search for Confluence pages based on criteria",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or keywords"
                    },
                    "space": {
                        "type": "string",
                        "description": "Optional space key to filter by"
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional labels to filter by"
                    }
                },
                "required": ["query"]
            }
        )
        
        find_solution = FunctionDeclaration(
            name="find_solution",
            description="Find solutions in Confluence for a Jira ticket",
            parameters={
                "type": "object",
                "properties": {
                    "ticketKey": {
                        "type": "string",
                        "description": "Jira ticket key (e.g., PROJ-123)"
                    },
                    "additionalContext": {
                        "type": "string",
                        "description": "Optional additional context or specific aspect to focus on"
                    }
                },
                "required": ["ticketKey"]
            }
        )
        
        # Create the tools list
        return [
            Tool(function_declarations=[search_jira, search_confluence, find_solution])
        ]
    
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: Optional[str] = None,
                         context_docs: Optional[List[Dict]] = None,
                         conversation_history: Optional[List[Dict]] = None,
                         temperature: float = 0.4,
                         max_tokens: int = 8192,
                         stream: bool = False,
                         enable_tools: bool = False) -> Any:
        """
        Generate a response using Gemini.
        
        Args:
            prompt: User prompt/query
            system_prompt: System instructions
            context_docs: List of context documents
            conversation_history: List of conversation messages
            temperature: Temperature for generation (0-1)
            max_tokens: Maximum output tokens
            stream: Whether to stream the response
            enable_tools: Whether to enable function calling
            
        Returns:
            Generated response or generator
        """
        try:
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=max_tokens,
            )
            
            # Build the content parts
            content_parts = []
            
            # Add system prompt
            if system_prompt:
                content_parts.append(
                    Content(
                        role="user",
                        parts=[f"<system>\n{system_prompt}\n</system>"]
                    )
                )
            
            # Add conversation history
            if conversation_history:
                for message in conversation_history:
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    
                    if role in ["user", "model"] and content:
                        content_parts.append(
                            Content(
                                role="user" if role == "user" else "model",
                                parts=[content]
                            )
                        )
            
            # Add context documents
            if context_docs:
                context_text = ""
                for i, doc in enumerate(context_docs):
                    source_type = doc.get("source_type", "document")
                    title = doc.get("metadata", {}).get("title", f"Document {i+1}")
                    doc_id = doc.get("metadata", {}).get("id", "")
                    url = doc.get("metadata", {}).get("url", "")
                    content = doc.get("content", "")
                    
                    context_text += f"\n\n===== {source_type.upper()} {i+1} =====\n"
                    context_text += f"Title: {title}\n"
                    context_text += f"ID: {doc_id}\n"
                    context_text += f"URL: {url}\n"
                    context_text += f"Content:\n{content}\n"
                
                if context_text:
                    content_parts.append(
                        Content(
                            role="user",
                            parts=[f"<context>\n{context_text}\n</context>"]
                        )
                    )
            
            # Add the user's prompt
            content_parts.append(
                Content(
                    role="user",
                    parts=[prompt]
                )
            )
            
            # Generate response
            logger.info(f"Generating response with temperature {temperature}")
            if stream:
                # Streaming response
                response_stream = self.model.generate_content(
                    content_parts,
                    generation_config=generation_config,
                    tools=self.tools if enable_tools else None,
                    stream=True
                )
                return response_stream
            else:
                # Single response
                response = self.model.generate_content(
                    content_parts,
                    generation_config=generation_config,
                    tools=self.tools if enable_tools else None
                )
                
                # Check for function calls
                if response.candidates and response.candidates[0].content.parts:
                    text = response.candidates[0].content.parts[0].text
                    
                    # Process function calls if present
                    function_calls = self._extract_function_calls(text)
                    if function_calls and enable_tools:
                        return {
                            "text": text,
                            "function_calls": function_calls
                        }
                    
                    return text
                
                return ""
                
        except GoogleAPICallError as e:
            logger.error(f"Google API error: {str(e)}")
            return f"I encountered a technical issue: {str(e)}"
        except ResponseValidationError as e:
            logger.error(f"Response validation error: {str(e)}")
            return "I couldn't generate a valid response. Please try a different question."
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return "I encountered an unexpected error. Please try again."
    
    def _extract_function_calls(self, text):
        """Extract function calls from response text."""
        try:
            # Simple regex to find function call blocks
            function_call_pattern = r"```json\s*\{\s*\"function\":\s*\"([^\"]+)\",\s*\"parameters\":\s*(\{[^}]+\})\s*\}\s*```"
            
            matches = re.findall(function_call_pattern, text)
            function_calls = []
            
            for match in matches:
                function_name, parameters_str = match
                try:
                    parameters = json.loads(parameters_str)
                    function_calls.append({
                        "function": function_name,
                        "parameters": parameters
                    })
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse function parameters: {parameters_str}")
            
            return function_calls
        except Exception as e:
            logger.error(f"Error extracting function calls: {str(e)}")
            return []
    
    def create_enhanced_system_prompt(self, source_type="combined"):
        """
        Create an enhanced system prompt for Gemini.
        
        Args:
            source_type: Type of data source ("jira", "confluence", or "combined")
            
        Returns:
            System prompt string
        """
        base_prompt = """
You are an AI assistant specialized in helping users find and understand information from their company's knowledge base. Your primary data sources are Jira tickets and Confluence documentation.

You have the following capabilities:
1. Searching and analyzing Jira tickets
2. Searching and understanding Confluence documentation
3. Finding relevant Confluence solutions for Jira issues
4. Maintaining conversation context across multiple questions
5. Using function calling to retrieve additional information when needed

When answering questions, follow these guidelines:

1. PRECISE AND ACCURATE
- Always base your answers on the provided context
- Cite specific sources when referencing information
- If information is not in the context, acknowledge the limitation
- Never make up information or assume details not present in the data

2. PROFESSIONAL AND HELPFUL
- Use a professional, conversational tone
- Organize complex information clearly with headings and structure
- Anticipate follow-up needs and offer additional assistance
- Be concise but thorough, focusing on what's most relevant to the user

3. KNOWLEDGE INTEGRATION
- Connect related information across Jira tickets and Confluence pages
- Highlight relationships between issues and their potential solutions
- Synthesize information from multiple sources when applicable
- Provide meaningful analysis, not just raw data regurgitation

4. CONTEXTUAL AWARENESS
- Consider the full conversation history when appropriate
- Remember previously established context from earlier exchanges
- Ask clarifying questions when needed to better address the user's needs
- Respect the selected data sources (Jira, Confluence, or both)

For technical content:
- Explain complex concepts clearly without oversimplifying
- Include code snippets or commands when relevant
- Maintain proper formatting for technical information
- Preserve important technical details when summarizing

For problem-solving:
- Focus on practical solutions
- Prioritize solutions from Confluence documentation when available
- Suggest potential approaches even when exact solutions aren't documented
- Identify similar known issues when exact matches aren't found

Remember: Your goal is to help users efficiently find and understand information from their knowledge base, connecting questions to the most relevant and helpful resources available.
"""
        
        # Add source-specific instructions
        if source_type == "jira":
            base_prompt += """
JIRA-SPECIFIC GUIDELINES:
- Focus exclusively on Jira ticket information
- Pay attention to ticket status, assignee, and other metadata
- Highlight key information from ticket descriptions and comments
- Format issue keys in the standard way (e.g., PROJECT-123)
- Include ticket URLs when referencing specific tickets
"""
        elif source_type == "confluence":
            base_prompt += """
CONFLUENCE-SPECIFIC GUIDELINES:
- Focus exclusively on Confluence documentation
- Maintain proper document structure in your answers
- Preserve formatting for technical content when important
- Include page titles and URLs when referencing specific pages
- Consider the recency of documentation when providing information
"""
        else:  # combined
            base_prompt += """
INTEGRATED KNOWLEDGE GUIDELINES:
- Seamlessly combine information from both Jira and Confluence
- Clearly indicate which source information comes from
- Prioritize finding Confluence documentation that solves Jira issues
- Draw connections between related tickets and documentation
- Provide a comprehensive view by leveraging both sources
"""
        
        return base_prompt
    
    def build_query_for_solution(self, jira_ticket, user_query=None):
        """
        Build a sophisticated query to find solutions in Confluence based on a Jira ticket.
        
        Args:
            jira_ticket: Jira ticket data
            user_query: Optional user query for additional context
            
        Returns:
            Generated query for finding solutions
        """
        # Extract key information from the ticket
        metadata = jira_ticket.get("metadata", {})
        ticket_key = metadata.get("key", "")
        summary = metadata.get("summary", "")
        issue_type = metadata.get("issuetype", "")
        status = metadata.get("status", "")
        content = jira_ticket.get("content", "")
        
        # Build a prompt for Gemini to generate a search query
        prompt = f"""
Given this Jira ticket information, create an optimal search query to find relevant solutions in Confluence.
Extract the main technical problem, error messages, and key concepts to create a targeted search query.

TICKET DETAILS:
Key: {ticket_key}
Summary: {summary}
Type: {issue_type}
Status: {status}

DESCRIPTION:
{content[:1500]}  # Limit content to avoid overwhelming the model

ADDITIONAL CONTEXT FROM USER:
{user_query if user_query else "None provided"}

Your task:
1. Identify the core technical problem or concept
2. Extract any error messages, codes, or specific technical terms
3. Identify key technologies or systems mentioned
4. Create a focused search query using 3-8 keywords or phrases
5. Format your response ONLY as a search query, without any explanation

EXAMPLES:
Bad query: "How to fix the error in the application"
Good query: "NullPointerException ConfigurationService startup initialization"

Output only the search query:
"""
        
        # Configure generation parameters for a concise, focused response
        generation_config = GenerationConfig(
            temperature=0.2,  # Low temperature for deterministic output
            top_p=0.95,
            max_output_tokens=100,  # Short response
        )
        
        try:
            # Generate the search query
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response.candidates and response.candidates[0].content.parts:
                search_query = response.candidates[0].content.parts[0].text.strip()
                logger.info(f"Generated search query: {search_query}")
                return search_query
            return f"{summary} {issue_type}"
        except Exception as e:
            logger.error(f"Error generating search query: {str(e)}")
            # Fallback to using summary and issue type
            return f"{summary} {issue_type}"
    
    def analyze_query_type(self, query):
        """
        Analyze the query to determine the best approach for handling it.
        
        Args:
            query: User query string
            
        Returns:
            Dict with analysis results
        """
        analysis_prompt = f"""
Analyze this user query to determine the best way to handle it:

USER QUERY: {query}

Categorize this query and extract key information. Respond in JSON format with these fields:
- query_type: One of ["direct_question", "jira_specific", "confluence_specific", "solution_seeking", "comparison", "multi_part"]
- requires_search: Boolean indicating if this needs to search for information
- entity_extraction: List of specific entities (jira tickets, confluence pages, etc.)
- is_technical: Boolean indicating if this is a technical question
- specific_technology: Any specific technology mentioned (or null)
- time_sensitivity: One of ["current", "historical", "any"]
- complexity: One of ["simple", "moderate", "complex"]

Output only valid JSON.
"""
        
        try:
            # Configure for analytical response
            generation_config = GenerationConfig(
                temperature=0.1,  # Very low temperature for consistent output
                top_p=0.95,
                max_output_tokens=500,
            )
            
            # Generate the analysis
            response = self.model.generate_content(
                analysis_prompt,
                generation_config=generation_config
            )
            
            if response.candidates and response.candidates[0].content.parts:
                analysis_text = response.candidates[0].content.parts[0].text
                
                # Extract JSON from response
                try:
                    # Find JSON pattern in the response
                    json_match = re.search(r'(\{.*\})', analysis_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        analysis = json.loads(json_str)
                        return analysis
                    else:
                        # Try parsing the whole text as JSON
                        analysis = json.loads(analysis_text)
                        return analysis
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse query analysis: {analysis_text}")
            
            # Default analysis if parsing fails
            return {
                "query_type": "direct_question",
                "requires_search": True,
                "entity_extraction": [],
                "is_technical": False,
                "specific_technology": None,
                "time_sensitivity": "any",
                "complexity": "moderate"
            }
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            # Return default analysis on error
            return {
                "query_type": "direct_question",
                "requires_search": True,
                "entity_extraction": [],
                "is_technical": False,
                "specific_technology": None,
                "time_sensitivity": "any",
                "complexity": "moderate"
            }
    
    def generate_multi_step_reasoning(self, query, context_docs, conversation_history=None):
        """
        Generate a response with multi-step reasoning for complex questions.
        
        Args:
            query: User query
            context_docs: Context documents
            conversation_history: Conversation history
            
        Returns:
            Dict with reasoning steps and final answer
        """
        # Build the system prompt for reasoning
        reasoning_prompt = """
You are an expert problem-solver with access to information from Jira and Confluence. For this complex question, use a multi-step reasoning approach:

1. ANALYZE THE QUESTION
- Break down what information is being asked for
- Identify key concepts, entities, and relationships
- Determine what specific details you need to find

2. GATHER RELEVANT INFORMATION
- Carefully extract relevant facts from the provided context
- Note where information might be missing or unclear
- Connect related pieces of information from different sources

3. REASONING STEPS
- Think step by step through the problem
- Consider multiple interpretations if the question is ambiguous
- Evaluate evidence for different possible answers
- Identify logical connections between facts

4. FINAL ANSWER
- Synthesize a complete, accurate answer based on your reasoning
- Cite specific sources for key information
- Acknowledge any limitations or uncertainties
- Format the answer clearly and concisely

When you respond, explicitly show your work by including these labeled sections:
<analysis>Your analysis of the question</analysis>
<information_gathering>Key facts extracted from context</information_gathering>
<reasoning>Your step-by-step reasoning process</reasoning>
<answer>Your final comprehensive answer</answer>
"""
        
        # Generate the reasoned response
        response = self.generate_response(
            prompt=query,
            system_prompt=reasoning_prompt,
            context_docs=context_docs,
            conversation_history=conversation_history,
            temperature=0.2,  # Lower temperature for more focused reasoning
            max_tokens=8192,
            stream=False
        )
        
        # Extract the reasoning steps from the response
        try:
            analysis = re.search(r'<analysis>(.*?)</analysis>', response, re.DOTALL)
            info_gathering = re.search(r'<information_gathering>(.*?)</information_gathering>', response, re.DOTALL)
            reasoning = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
            answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            
            return {
                "analysis": analysis.group(1).strip() if analysis else "",
                "information_gathering": info_gathering.group(1).strip() if info_gathering else "",
                "reasoning": reasoning.group(1).strip() if reasoning else "",
                "answer": answer.group(1).strip() if answer else response,
                "full_response": response
            }
        except Exception as e:
            logger.error(f"Error extracting reasoning steps: {str(e)}")
            return {
                "analysis": "",
                "information_gathering": "",
                "reasoning": "",
                "answer": response,
                "full_response": response
            }
```












# Enhanced Jira-Confluence Integration

```python
import logging
import re
import json
import time
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

from utils import TextProcessor, ContentIndex, EnhancedCache, HTMLFilter
from jira_client import JiraClient
from confluence_client import ConfluenceClient
from gemini_client import GeminiClient

# Configure logging
logger = logging.getLogger("IntegrationManager")

class IntegrationManager:
    """Manager for integrating Jira and Confluence data."""
    
    def __init__(self, jira_client, confluence_client, gemini_client, cache_dir=None):
        """
        Initialize the integration manager.
        
        Args:
            jira_client: JiraClient instance
            confluence_client: ConfluenceClient instance
            gemini_client: GeminiClient instance
            cache_dir: Directory for caching
        """
        self.jira_client = jira_client
        self.confluence_client = confluence_client
        self.gemini_client = gemini_client
        
        # Set up caching
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./integration_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = EnhancedCache(str(self.cache_dir / "general"))
        
        # Set up content indexing
        self.jira_index = ContentIndex(str(self.cache_dir / "jira_index"))
        self.confluence_index = ContentIndex(str(self.cache_dir / "confluence_index"))
        
        # Text processor for analysis
        self.text_processor = TextProcessor()
        
        # Thread pool for background tasks
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Knowledge graph (simple implementation)
        self.knowledge_graph = self._load_knowledge_graph()
        
        logger.info("Initialized Integration Manager")
    
    def _load_knowledge_graph(self):
        """Load the knowledge graph from disk or initialize a new one."""
        graph_path = self.cache_dir / "knowledge_graph.json"
        
        if graph_path.exists():
            try:
                with open(graph_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading knowledge graph: {str(e)}")
        
        # Initialize new graph
        return {
            "nodes": {
                "jira": {},
                "confluence": {}
            },
            "relationships": []
        }
    
    def _save_knowledge_graph(self):
        """Save the knowledge graph to disk."""
        graph_path = self.cache_dir / "knowledge_graph.json"
        
        try:
            with open(graph_path, 'w') as f:
                json.dump(self.knowledge_graph, f)
            logger.info("Saved knowledge graph")
            return True
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {str(e)}")
            return False
    
    def add_to_knowledge_graph(self, item, item_type):
        """
        Add an item to the knowledge graph.
        
        Args:
            item: Item to add (Jira ticket or Confluence page)
            item_type: Type of item ("jira" or "confluence")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if item_type not in ["jira", "confluence"]:
                logger.error(f"Invalid item type: {item_type}")
                return False
            
            # Extract item ID
            metadata = item.get("metadata", {})
            item_id = metadata.get("id")
            
            if not item_id:
                logger.error("Item missing ID")
                return False
            
            # Add to nodes
            self.knowledge_graph["nodes"][item_type][item_id] = {
                "id": item_id,
                "title": metadata.get("title", metadata.get("key", "Untitled")),
                "url": metadata.get("url", ""),
                "keywords": self.text_processor.extract_keywords(item.get("content", ""), top_n=10)
            }
            
            # Save graph
            self._save_knowledge_graph()
            
            # Schedule relationship discovery in background
            self.executor.submit(self._discover_relationships, item_id, item_type)
            
            return True
        except Exception as e:
            logger.error(f"Error adding to knowledge graph: {str(e)}")
            return False
    
    def _discover_relationships(self, item_id, item_type):
        """
        Discover relationships between items in the knowledge graph.
        
        Args:
            item_id: ID of the item to discover relationships for
            item_type: Type of item ("jira" or "confluence")
        """
        try:
            # Get the item
            item_data = self.knowledge_graph["nodes"][item_type].get(item_id)
            
            if not item_data:
                return
            
            # Find related items based on keyword overlap
            item_keywords = set(item_data.get("keywords", []))
            
            if not item_keywords:
                return
            
            # Define the other type to search
            other_type = "confluence" if item_type == "jira" else "jira"
            
            # Look for relationships with items of the other type
            for other_id, other_data in self.knowledge_graph["nodes"][other_type].items():
                other_keywords = set(other_data.get("keywords", []))
                
                # Calculate similarity based on keyword overlap
                if not other_keywords:
                    continue
                
                overlap = item_keywords.intersection(other_keywords)
                similarity = len(overlap) / max(len(item_keywords), len(other_keywords))
                
                # If similarity above threshold, add relationship
                if similarity >= 0.3:  # 30% keyword overlap
                    relationship = {
                        "source_type": item_type,
                        "source_id": item_id,
                        "target_type": other_type,
                        "target_id": other_id,
                        "similarity": similarity,
                        "shared_keywords": list(overlap)
                    }
                    
                    # Check if relationship already exists
                    exists = False
                    for rel in self.knowledge_graph["relationships"]:
                        if (rel["source_type"] == item_type and rel["source_id"] == item_id and
                            rel["target_type"] == other_type and rel["target_id"] == other_id):
                            exists = True
                            break
                    
                    if not exists:
                        self.knowledge_graph["relationships"].append(relationship)
            
            # Save graph
            self._save_knowledge_graph()
            
        except Exception as e:
            logger.error(f"Error discovering relationships: {str(e)}")
    
    def find_related_items(self, item_id, item_type, min_similarity=0.2):
        """
        Find items related to the given item.
        
        Args:
            item_id: ID of the item
            item_type: Type of item ("jira" or "confluence")
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of related items
        """
        related = []
        
        try:
            # Look for direct relationships in the graph
            for rel in self.knowledge_graph["relationships"]:
                if rel["source_type"] == item_type and rel["source_id"] == item_id and rel["similarity"] >= min_similarity:
                    related.append({
                        "id": rel["target_id"],
                        "type": rel["target_type"],
                        "similarity": rel["similarity"],
                        "shared_keywords": rel["shared_keywords"]
                    })
                elif rel["target_type"] == item_type and rel["target_id"] == item_id and rel["similarity"] >= min_similarity:
                    related.append({
                        "id": rel["source_id"],
                        "type": rel["source_type"],
                        "similarity": rel["similarity"],
                        "shared_keywords": rel["shared_keywords"]
                    })
        except Exception as e:
            logger.error(f"Error finding related items: {str(e)}")
        
        return related
    
    def extract_jira_keys(self, text):
        """
        Extract Jira ticket keys from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            List of Jira ticket keys
        """
        if not text:
            return []
        
        # Pattern for Jira keys (PROJECT-123)
        pattern = r'\b[A-Z]+-\d+\b'
        return re.findall(pattern, text)
    
    def find_solutions_for_ticket(self, ticket_key, additional_query=None):
        """
        Find solutions in Confluence for a Jira ticket.
        
        Args:
            ticket_key: Jira ticket key
            additional_query: Optional additional query
            
        Returns:
            Dict with ticket, solutions, and response
        """
        try:
            # Check cache first
            cache_key = f"solutions_{ticket_key}_{additional_query}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Using cached solutions for {ticket_key}")
                return cached_result
            
            # Get the ticket details
            ticket = self.jira_client.get_issue_content(ticket_key)
            if not ticket:
                logger.error(f"Ticket not found: {ticket_key}")
                return {"error": f"Ticket {ticket_key} not found"}
            
            # Add to knowledge graph
            self.add_to_knowledge_graph(ticket, "jira")
            
            # Find related items from knowledge graph
            related_items = self.find_related_items(ticket["metadata"]["id"], "jira")
            confluence_solutions = []
            
            # If we have related items, fetch their details
            if related_items:
                for item in related_items:
                    if item["type"] == "confluence":
                        page = self.confluence_client.get_page_content(item["id"])
                        if page:
                            confluence_solutions.append(page)
            
            # If not enough solutions from graph, search Confluence
            if len(confluence_solutions) < 3:
                # Generate search query based on ticket details
                search_query = self.gemini_client.build_query_for_solution(ticket, additional_query)
                
                # Search Confluence
                search_results = self.confluence_client.search_content(search_query)
                
                # Add to solutions list (avoiding duplicates)
                existing_ids = [sol["metadata"]["id"] for sol in confluence_solutions]
                for result in search_results:
                    if result["metadata"]["id"] not in existing_ids:
                        confluence_solutions.append(result)
                        # Add to knowledge graph
                        self.add_to_knowledge_graph(result, "confluence")
            
            # Limit to top 5 solutions
            confluence_solutions = confluence_solutions[:5]
            
            # Create context with both ticket and solutions
            context_docs = [ticket] + confluence_solutions
            
            # Build system prompt for combined sources
            system_prompt = self.gemini_client.create_enhanced_system_prompt("combined")
            
            # Generate response
            prompt = f"Based on the provided Jira ticket {ticket_key} and Confluence documentation, suggest a detailed solution for this issue."
            if additional_query:
                prompt += f" Specifically address: {additional_query}"
                
            response = self.gemini_client.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                context_docs=context_docs,
                temperature=0.3,
                stream=False
            )
            
            result = {
                "ticket": ticket,
                "solutions": confluence_solutions,
                "response": response
            }
            
            # Cache the result
            self.cache.set(cache_key, result, expiry=86400)  # 24 hours
            
            return result
        except Exception as e:
            logger.error(f"Error finding solutions for ticket: {str(e)}")
            return {"error": str(e)}
    
    def extract_technical_details_from_ticket(self, ticket):
        """
        Extract key technical details from a Jira ticket.
        
        Args:
            ticket: Jira ticket data
            
        Returns:
            Dict with extracted technical details
        """
        if not ticket:
            return {}
        
        try:
            content = ticket.get("content", "")
            
            # Extract error messages
            error_messages = []
            error_patterns = [
                r'(?:error|exception|failure):\s*(.*?)(?:\n|$)',
                r'(?:error|exception|failure)[:\s]\s*(.*?)(?:\n|$)',
                r'(?:failed with):\s*(.*?)(?:\n|$)',
                r'(?:stack trace):\s*((?:.*\n)+?.*?)(?:\n\s*\n|\n\s*\w|\Z)',
            ]
            
            for pattern in error_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    error_msg = match.group(1).strip()
                    if error_msg and len(error_msg) < 200:  # Avoid huge traces
                        error_messages.append(error_msg)
            
            # Extract versions/components
            versions = []
            version_patterns = [
                r'(?:version):\s*([\w.-]+)',
                r'(?:v|ver|version)\s*([\d.]+)',
            ]
            
            for pattern in version_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    version = match.group(1).strip()
                    if version:
                        versions.append(version)
            
            # Extract technologies mentioned
            # Simple approach - look for known technology terms
            tech_terms = [
                "java", "python", "javascript", "nodejs", "react", "angular", "vue", 
                "spring", "django", "flask", "express", "mongodb", "mysql", "postgres",
                "oracle", "sql server", "docker", "kubernetes", "aws", "azure", "gcp",
                "linux", "windows", "macos", "ios", "android", "api", "rest", "graphql",
                "soap", "http", "https", "tcp", "udp", "git", "svn", "jenkins", "circleci",
                "travis", "oauth", "jwt", "saml", "ldap", "hadoop", "spark", "kafka"
            ]
            
            found_tech = []
            for tech in tech_terms:
                if re.search(r'\b' + re.escape(tech) + r'\b', content, re.IGNORECASE):
                    found_tech.append(tech)
            
            return {
                "error_messages": error_messages,
                "versions": versions,
                "technologies": found_tech,
                "keywords": self.text_processor.extract_keywords(content, top_n=10)
            }
        except Exception as e:
            logger.error(f"Error extracting technical details: {str(e)}")
            return {}
    
    def analyze_solutions_quality(self, ticket, solutions):
        """
        Analyze the quality of solutions for a ticket.
        
        Args:
            ticket: Jira ticket data
            solutions: List of solution pages
            
        Returns:
            Dict with quality analysis
        """
        if not ticket or not solutions:
            return {"relevance_scores": []}
        
        try:
            # Extract technical details from ticket
            ticket_details = self.extract_technical_details_from_ticket(ticket)
            ticket_keywords = set(ticket_details.get("keywords", []))
            ticket_tech = set(ticket_details.get("technologies", []))
            
            # Analyze each solution
            scores = []
            
            for solution in solutions:
                solution_content = solution.get("content", "")
                solution_keywords = set(self.text_processor.extract_keywords(solution_content, top_n=10))
                
                # Calculate keyword overlap
                keyword_overlap = ticket_keywords.intersection(solution_keywords)
                keyword_score = len(keyword_overlap) / max(len(ticket_keywords), 1) if ticket_keywords else 0
                
                # Check for technical detail matches
                tech_score = 0
                if ticket_tech:
                    solution_tech = []
                    for tech in ticket_tech:
                        if re.search(r'\b' + re.escape(tech) + r'\b', solution_content, re.IGNORECASE):
                            solution_tech.append(tech)
                    
                    tech_score = len(solution_tech) / len(ticket_tech)
                
                # Check for error message matches
                error_score = 0
                for error_msg in ticket_details.get("error_messages", []):
                    if error_msg and len(error_msg) > 10:
                        # Clean up error message for regex
                        clean_error = re.escape(error_msg[:50])  # Use first 50 chars
                        if re.search(clean_error, solution_content, re.IGNORECASE):
                            error_score = 1.0
                            break
                
                # Calculate overall relevance score
                relevance = (keyword_score * 0.5) + (tech_score * 0.3) + (error_score * 0.2)
                
                scores.append({
                    "solution_id": solution["metadata"]["id"],
                    "relevance_score": relevance,
                    "keyword_score": keyword_score,
                    "tech_score": tech_score,
                    "error_score": error_score,
                    "shared_keywords": list(keyword_overlap)
                })
            
            # Sort by relevance
            scores.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return {
                "relevance_scores": scores,
                "ticket_details": ticket_details
            }
        except Exception as e:
            logger.error(f"Error analyzing solutions quality: {str(e)}")
            return {"relevance_scores": []}
    
    def get_references_between_jira_confluence(self):
        """
        Get explicit references between Jira tickets and Confluence pages.
        
        Returns:
            Dict with references
        """
        try:
            # Check cache
            cache_key = "jira_confluence_references"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            references = {
                "jira_to_confluence": [],
                "confluence_to_jira": []
            }
            
            # Sample Jira tickets to check for Confluence links
            # In a real implementation, you would page through all tickets
            jql = "updated >= -30d order by updated desc"  # Last 30 days
            jira_issues = self.jira_client.get_all_issues(jql, max_results=100)
            
            # Process each issue
            for issue in jira_issues:
                issue_key = issue.get("key")
                if not issue_key:
                    continue
                
                # Get full content
                issue_content = self.jira_client.get_issue_content(issue_key)
                if not issue_content:
                    continue
                
                content = issue_content.get("content", "")
                
                # Look for Confluence links
                confluence_links = re.findall(r'confluence\..*?/pages/(\d+)', content)
                
                if confluence_links:
                    references["jira_to_confluence"].append({
                        "jira_key": issue_key,
                        "jira_id": issue.get("id", ""),
                        "confluence_ids": list(set(confluence_links))
                    })
            
            # Sample Confluence pages to check for Jira links
            # In a real implementation, you would page through many pages
            confluence_pages = self.confluence_client.search_content("created >= now(-30d)", limit=100)
            
            # Process each page
            for page in confluence_pages:
                page_id = page["metadata"].get("id")
                if not page_id:
                    continue
                
                content = page.get("content", "")
                
                # Extract Jira keys
                jira_keys = self.extract_jira_keys(content)
                
                if jira_keys:
                    references["confluence_to_jira"].append({
                        "confluence_id": page_id,
                        "confluence_title": page["metadata"].get("title", ""),
                        "jira_keys": list(set(jira_keys))
                    })
            
            # Cache the result
            self.cache.set(cache_key, references, expiry=86400*7)  # 7 days
            
            return references
        except Exception as e:
            logger.error(f"Error getting references: {str(e)}")
            return {"jira_to_confluence": [], "confluence_to_jira": []}
```











<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jira-Confluence Knowledge Assistant</title>
    <link rel="stylesheet" href="css/styles.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism.min.css" rel="stylesheet" />
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="app-logo">
                <i class="fa-solid fa-robot"></i>
                <h1>Knowledge Assistant</h1>
            </div>

            <div class="source-selector">
                <h3>Data Sources</h3>
                <div class="source-options">
                    <label class="source-option-label">
                        <input type="radio" name="source" value="combined" checked>
                        <span class="source-option-text">
                            <i class="fa-solid fa-database"></i>
                            <span>Both Sources</span>
                        </span>
                    </label>
                    <label class="source-option-label">
                        <input type="radio" name="source" value="jira">
                        <span class="source-option-text">
                            <i class="fa-solid fa-ticket-alt"></i>
                            <span>Jira Only</span>
                        </span>
                    </label>
                    <label class="source-option-label">
                        <input type="radio" name="source" value="confluence">
                        <span class="source-option-text">
                            <i class="fa-solid fa-book"></i>
                            <span>Confluence Only</span>
                        </span>
                    </label>
                </div>
            </div>

            <div class="sidebar-section">
                <h3>My Context</h3>
                <div class="context-summary" id="context-summary">
                    <div class="empty-state">
                        <i class="fa-solid fa-layer-group"></i>
                        <p>No context items added yet</p>
                    </div>
                </div>
                <button id="manage-context-btn" class="btn-secondary btn-sm" disabled>
                    <i class="fa-solid fa-cog"></i> Manage Context
                </button>
            </div>

            <div class="sidebar-section">
                <h3>Recent Searches</h3>
                <div class="recent-searches" id="recent-searches">
                    <div class="empty-state">
                        <i class="fa-solid fa-clock-rotate-left"></i>
                        <p>No recent searches</p>
                    </div>
                </div>
            </div>

            <div class="connection-status">
                <h3>System Status</h3>
                <div class="status-items">
                    <div class="status-item">
                        <span class="status-label">Jira:</span>
                        <span id="jira-status" class="status-badge pending">Connecting...</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Confluence:</span>
                        <span id="confluence-status" class="status-badge pending">Connecting...</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">AI:</span>
                        <span id="ai-status" class="status-badge connected">Ready</span>
                    </div>
                </div>
            </div>

            <div class="sidebar-actions">
                <button id="clear-cache-btn" class="btn-secondary btn-sm">
                    <i class="fa-solid fa-broom"></i> Clear Cache
                </button>
                <button id="clear-chat-btn" class="btn-secondary btn-sm">
                    <i class="fa-solid fa-trash"></i> Clear Chat
                </button>
            </div>
        </div>

        <!-- Main Content -->
        <div class="main-content">
            <!-- Chat Header -->
            <div class="chat-header">
                <div class="chat-header-title">
                    <h2>Jira-Confluence Knowledge Assistant</h2>
                    <span class="version-badge">v1.0</span>
                </div>
                <div class="chat-header-actions">
                    <button id="view-knowledge-graph-btn" class="btn-secondary btn-sm" title="View Knowledge Graph">
                        <i class="fa-solid fa-diagram-project"></i>
                    </button>
                    <button id="help-btn" class="btn-secondary btn-sm" title="Help">
                        <i class="fa-solid fa-circle-question"></i>
                    </button>
                </div>
            </div>

            <!-- Chat Messages -->
            <div class="chat-container">
                <div class="chat-messages" id="chat-messages">
                    <div class="message system">
                        <div class="message-content">
                            <h3><i class="fa-solid fa-robot"></i> Knowledge Assistant</h3>
                            <p>Hello! I'm your Jira-Confluence Knowledge Assistant. I can help you find information from your Jira tickets and Confluence pages, or even find solutions from Confluence for your Jira issues.</p>
                            <div class="suggestion-chips">
                                <div class="suggestion-chip" data-query="Find open bugs in the authentication system">Find open bugs in the authentication system</div>
                                <div class="suggestion-chip" data-query="Show me documentation about our API">Show me documentation about our API</div>
                                <div class="suggestion-chip" data-query="Help me troubleshoot SSO login failures">Help me troubleshoot SSO login failures</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Chat Input Area -->
                <div class="chat-input-area">
                    <div class="input-wrapper">
                        <textarea id="chat-input" placeholder="Type your question or Jira ticket key (e.g., PROJ-123)..." rows="1"></textarea>
                        <div class="input-actions">
                            <button id="upload-btn" class="btn-icon" title="Upload file">
                                <i class="fa-solid fa-paperclip"></i>
                            </button>
                            <button id="clear-input-btn" class="btn-icon" title="Clear input">
                                <i class="fa-solid fa-times"></i>
                            </button>
                        </div>
                    </div>
                    <button id="send-btn" class="btn-primary" disabled>
                        <i class="fa-solid fa-paper-plane"></i>
                    </button>
                </div>
            </div>

            <!-- Right Panel (for search results or context) -->
            <div class="right-panel" id="right-panel">
                <div class="panel-header">
                    <h3 id="panel-title">Search Results</h3>
                    <button id="close-panel-btn" class="btn-icon">
                        <i class="fa-solid fa-times"></i>
                    </button>
                </div>
                <div class="panel-content" id="panel-content">
                    <!-- Panel content will be inserted here -->
                </div>
                <div class="panel-footer" id="panel-footer">
                    <!-- Panel footer actions will be inserted here -->
                </div>
            </div>
        </div>
    </div>

    <!-- Modals -->
    <!-- Item Details Modal -->
    <div class="modal" id="item-details-modal">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 id="item-details-title">Item Details</h3>
                    <button class="btn-icon close-modal">
                        <i class="fa-solid fa-times"></i>
                    </button>
                </div>
                <div class="modal-body" id="item-details-body">
                    <!-- Item details will be inserted here -->
                </div>
                <div class="modal-footer" id="item-details-footer">
                    <button id="find-solutions-btn" class="btn-primary">Find Solutions in Confluence</button>
                    <button id="add-to-context-btn" class="btn-secondary">Add to Context</button>
                    <button class="btn-secondary close-modal">Close</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Knowledge Graph Modal -->
    <div class="modal" id="knowledge-graph-modal">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Knowledge Graph</h3>
                    <button class="btn-icon close-modal">
                        <i class="fa-solid fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="knowledge-graph-controls">
                        <div class="filter-controls">
                            <label class="checkbox-label">
                                <input type="checkbox" id="show-jira" checked>
                                <span class="checkbox-text">Jira Tickets</span>
                            </label>
                            <label class="checkbox-label">
                                <input type="checkbox" id="show-confluence" checked>
                                <span class="checkbox-text">Confluence Pages</span>
                            </label>
                            <label class="checkbox-label">
                                <input type="checkbox" id="show-relationships" checked>
                                <span class="checkbox-text">Relationships</span>
                            </label>
                        </div>
                        <div class="search-controls">
                            <input type="text" id="graph-search" placeholder="Search the graph...">
                        </div>
                    </div>
                    <div class="knowledge-graph-container" id="knowledge-graph-container">
                        <!-- Graph will be rendered here -->
                    </div>
                    <div class="knowledge-graph-legend">
                        <div class="legend-item">
                            <span class="legend-icon jira-icon"></span>
                            <span class="legend-text">Jira Ticket</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-icon confluence-icon"></span>
                            <span class="legend-text">Confluence Page</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-icon relationship-line"></span>
                            <span class="legend-text">Relationship</span>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn-secondary close-modal">Close</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Help Modal -->
    <div class="modal" id="help-modal">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>How to Use This Assistant</h3>
                    <button class="btn-icon close-modal">
                        <i class="fa-solid fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="help-section">
                        <h4><i class="fa-solid fa-search"></i> Finding Information</h4>
                        <p>Simply type your question in the chat input. You can ask about:</p>
                        <ul>
                            <li>Specific Jira tickets (e.g., "Tell me about PROJ-123")</li>
                            <li>Confluence documentation (e.g., "Find pages about API authentication")</li>
                            <li>Technical issues (e.g., "How do I fix the database connection error?")</li>
                        </ul>
                    </div>

                    <div class="help-section">
                        <h4><i class="fa-solid fa-sliders"></i> Data Sources</h4>
                        <p>You can control which systems are searched:</p>
                        <ul>
                            <li><strong>Both Sources:</strong> Search both Jira and Confluence (default)</li>
                            <li><strong>Jira Only:</strong> Only search Jira tickets</li>
                            <li><strong>Confluence Only:</strong> Only search Confluence pages</li>
                        </ul>
                    </div>

                    <div class="help-section">
                        <h4><i class="fa-solid fa-layer-group"></i> Using Context</h4>
                        <p>The assistant can use multiple items for context:</p>
                        <ol>
                            <li>When viewing a search result, click "Add to Context"</li>
                            <li>Add multiple related items to build comprehensive context</li>
                            <li>Ask follow-up questions with this context in mind</li>
                        </ol>
                    </div>

                    <div class="help-section">
                        <h4><i class="fa-solid fa-wand-magic-sparkles"></i> Special Features</h4>
                        <ul>
                            <li><strong>Find Solutions:</strong> When viewing a Jira ticket, click "Find Solutions" to search for relevant Confluence documentation</li>
                            <li><strong>Knowledge Graph:</strong> View relationships between Jira tickets and Confluence pages</li>
                            <li><strong>Direct Ticket Access:</strong> Type a Jira ticket key (e.g., "PROJ-123") to directly access that ticket</li>
                        </ul>
                    </div>

                    <div class="help-section">
                        <h4><i class="fa-solid fa-keyboard"></i> Keyboard Shortcuts</h4>
                        <div class="shortcuts-grid">
                            <div class="shortcut">
                                <span class="key">Enter</span>
                                <span class="description">Send message</span>
                            </div>
                            <div class="shortcut">
                                <span class="key">Shift + Enter</span>
                                <span class="description">New line</span>
                            </div>
                            <div class="shortcut">
                                <span class="key">Esc</span>
                                <span class="description">Close panels/modals</span>
                            </div>
                            <div class="shortcut">
                                <span class="key">?</span>
                                <span class="description">Open help</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn-primary close-modal">Got it!</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Context Management Modal -->
    <div class="modal" id="context-modal">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Manage Context</h3>
                    <button class="btn-icon close-modal">
                        <i class="fa-solid fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="context-info">
                        <p>Items in your context will be used as reference when answering your questions.</p>
                    </div>
                    <div class="context-items" id="context-items-list">
                        <!-- Context items will be inserted here -->
                    </div>
                </div>
                <div class="modal-footer">
                    <button id="clear-context-btn" class="btn-secondary">Clear All</button>
                    <button class="btn-primary close-modal">Done</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Reasoning Steps Modal -->
    <div class="modal" id="reasoning-modal">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>AI Reasoning Process</h3>
                    <button class="btn-icon close-modal">
                        <i class="fa-solid fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="reasoning-steps">
                        <div class="reasoning-step">
                            <div class="step-header" data-target="analysis-step">
                                <h4><i class="fa-solid fa-magnifying-glass"></i> Question Analysis</h4>
                                <i class="fa-solid fa-chevron-down"></i>
                            </div>
                            <div class="step-content" id="analysis-step">
                                <!-- Analysis content will be inserted here -->
                            </div>
                        </div>
                        
                        <div class="reasoning-step">
                            <div class="step-header" data-target="information-step">
                                <h4><i class="fa-solid fa-file-alt"></i> Information Gathering</h4>
                                <i class="fa-solid fa-chevron-down"></i>
                            </div>
                            <div class="step-content" id="information-step">
                                <!-- Information content will be inserted here -->
                            </div>
                        </div>
                        
                        <div class="reasoning-step">
                            <div class="step-header" data-target="reasoning-step">
                                <h4><i class="fa-solid fa-brain"></i> Reasoning Process</h4>
                                <i class="fa-solid fa-chevron-down"></i>
                            </div>
                            <div class="step-content" id="reasoning-step">
                                <!-- Reasoning content will be inserted here -->
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn-primary close-modal">Close</button>
                </div>
            </div>
        </div>
    </div>

    <!-- File Upload Modal -->
    <div class="modal" id="upload-modal">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Upload File</h3>
                    <button class="btn-icon close-modal">
                        <i class="fa-solid fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="upload-area" id="upload-area">
                        <i class="fa-solid fa-cloud-upload-alt"></i>
                        <p>Drag and drop file here or click to browse</p>
                        <p class="upload-info">Supported files: PDF, Word, Excel, Images, Text (Max 10MB)</p>
                        <input type="file" id="file-input" hidden>
                    </div>
                    <div class="upload-preview" id="upload-preview">
                        <!-- Upload preview will be inserted here -->
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn-secondary close-modal">Cancel</button>
                    <button id="upload-file-btn" class="btn-primary" disabled>Upload & Analyze</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked@4.0.0/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-java.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-javascript.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-sql.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-bash.min.js"></script>
    <script src="js/utils.js"></script>
    <script src="js/api.js"></script>
    <script src="js/chat.js"></script>
    <script src="js/ui.js"></script>
    <script src="js/graph.js"></script>
    <script src="js/main.js"></script>
</body>
</html>
























/* Variables */
:root {
    /* Colors */
    --primary: #0052CC;
    --primary-dark: #0747A6;
    --primary-light: #4C9AFF;
    --secondary: #57D9A3;
    --secondary-dark: #00875A;
    --secondary-light: #79F2C0;
    
    --neutral-dark: #172B4D;
    --neutral: #505F79;
    --neutral-light: #97A0AF;
    --neutral-lighter: #DFE1E6;
    
    --background: #F4F5F7;
    --background-light: #FAFBFC;
    --white: #FFFFFF;
    
    --red: #FF5630;
    --yellow: #FFAB00;
    --green: #36B37E;
    --blue: #0065FF;
    --purple: #6554C0;
    
    /* Shadows */
    --shadow-small: 0 1px 2px rgba(0, 0, 0, 0.1);
    --shadow-medium: 0 3px 6px rgba(0, 0, 0, 0.15);
    --shadow-large: 0 8px 16px rgba(0, 0, 0, 0.15);
    
    /* Spacing */
    --spacing-xs: 4px;
    --spacing-sm: 8px; 
    --spacing-md: 16px;
    --spacing-lg: 24px;
    --spacing-xl: 32px;
    
    /* Border radius */
    --radius-sm: 3px;
    --radius-md: 6px;
    --radius-lg: 12px;
    --radius-round: 50%;
    
    /* Font */
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    --font-size-xs: 12px;
    --font-size-sm: 14px;
    --font-size-md: 16px;
    --font-size-lg: 18px;
    --font-size-xl: 24px;
    
    /* Transitions */
    --transition-fast: 0.15s ease;
    --transition-normal: 0.25s ease;
    --transition-slow: 0.4s ease;
}

/* Reset and base styles */
*, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html, body {
    height: 100%;
    font-family: var(--font-family);
    font-size: var(--font-size-md);
    color: var(--neutral-dark);
    line-height: 1.5;
    background-color: var(--background);
}

a {
    color: var(--primary);
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

h1, h2, h3, h4, h5, h6 {
    font-weight: 600;
    line-height: 1.3;
}

/* Layout */
.app-container {
    display: flex;
    height: 100vh;
    overflow: hidden;
}

.sidebar {
    width: 280px;
    background-color: var(--neutral-dark);
    color: var(--white);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding: var(--spacing-lg);
    gap: var(--spacing-lg);
    transition: width var(--transition-normal);
    flex-shrink: 0;
}

.main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
}

/* App Logo */
.app-logo {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
    padding-bottom: var(--spacing-lg);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: var(--spacing-md);
}

.app-logo i {
    font-size: var(--font-size-xl);
    color: var(--primary-light);
}

.app-logo h1 {
    font-size: var(--font-size-lg);
    font-weight: 600;
}

/* Sidebar sections */
.sidebar-section {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
}

.sidebar h3 {
    font-size: var(--font-size-sm);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--neutral-lighter);
    margin-bottom: var(--spacing-xs);
}

/* Source selector */
.source-options {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
}

.source-option-label {
    display: flex;
    align-items: center;
    cursor: pointer;
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--radius-sm);
    transition: background-color var(--transition-fast);
}

.source-option-label:hover {
    background-color: rgba(255, 255, 255, 0.1);
}

.source-option-label input {
    position: absolute;
    opacity: 0;
    height: 0;
    width: 0;
}

.source-option-text {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
    color: var(--neutral-lighter);
}

.source-option-label input:checked + .source-option-text {
    color: var(--white);
    font-weight: 500;
}

.source-option-label input:checked + .source-option-text i {
    color: var(--primary-light);
}

/* Context summary */
.context-summary, .recent-searches {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: var(--radius-md);
    padding: var(--spacing-md);
    min-height: 100px;
}

.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--spacing-sm);
    height: 100%;
    color: var(--neutral-light);
    text-align: center;
    padding: var(--spacing-lg) 0;
}

.empty-state i {
    font-size: 24px;
    opacity: 0.5;
}

.empty-state p {
    font-size: var(--font-size-sm);
}

/* Connection status */
.connection-status {
    margin-top: auto;
}

.status-items {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
}

.status-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.status-label {
    font-size: var(--font-size-sm);
}

.status-badge {
    padding: 2px 8px;
    border-radius: 12px;
    font-size: var(--font-size-xs);
    font-weight: 500;
}

.status-badge.pending {
    background-color: var(--yellow);
    color: var(--neutral-dark);
}

.status-badge.connected {
    background-color: var(--green);
    color: var(--white);
}

.status-badge.disconnected {
    background-color: var(--red);
    color: var(--white);
}

/* Sidebar actions */
.sidebar-actions {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
    margin-top: var(--spacing-md);
}

/* Chat header */
.chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--spacing-md) var(--spacing-lg);
    background-color: var(--white);
    border-bottom: 1px solid var(--neutral-lighter);
    z-index: 10;
}

.chat-header-title {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.chat-header-title h2 {
    font-size: var(--font-size-lg);
    font-weight: 600;
}

.version-badge {
    font-size: var(--font-size-xs);
    background-color: var(--neutral-lighter);
    padding: 2px 6px;
    border-radius: 10px;
    color: var(--neutral);
}

.chat-header-actions {
    display: flex;
    gap: var(--spacing-sm);
}

/* Chat container */
.chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background-color: var(--background-light);
}

.chat-messages {
    flex: 1;
    padding: var(--spacing-lg);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: var(--spacing-lg);
}

/* Messages */
.message {
    max-width: 85%;
    display: flex;
    flex-direction: column;
    animation: fadeInUp 0.3s ease;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.message.user {
    align-self: flex-end;
}

.message.bot, .message.system, .message.thinking {
    align-self: flex-start;
}

.message-content {
    padding: var(--spacing-md);
    border-radius: var(--radius-md);
    position: relative;
}

.message.user .message-content {
    background-color: var(--primary);
    color: var(--white);
    border-radius: var(--radius-md) var(--radius-md) 0 var(--radius-md);
}

.message.bot .message-content {
    background-color: var(--white);
    box-shadow: var(--shadow-small);
    border-radius: var(--radius-md) var(--radius-md) var(--radius-md) 0;
}

.message.system .message-content {
    background-color: var(--background-light);
    border: 1px solid var(--neutral-lighter);
    border-radius: var(--radius-md);
    max-width: 750px;
    margin: 0 auto;
    text-align: center;
}

.message.thinking .message-content {
    background-color: var(--white);
    opacity: 0.7;
    border-radius: var(--radius-md) var(--radius-md) var(--radius-md) 0;
}

.message-content h3 {
    margin-bottom: var(--spacing-sm);
    font-size: var(--font-size-md);
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.message-content p {
    margin-bottom: var(--spacing-md);
}

.message-content p:last-child {
    margin-bottom: 0;
}

.message-content code {
    background-color: rgba(0, 0, 0, 0.05);
    padding: 2px 4px;
    border-radius: var(--radius-sm);
    font-family: monospace;
    font-size: 0.9em;
}

.message.user .message-content code {
    background-color: rgba(255, 255, 255, 0.2);
}

.message-content pre {
    background-color: rgba(0, 0, 0, 0.05);
    padding: var(--spacing-md);
    border-radius: var(--radius-sm);
    overflow-x: auto;
    margin-bottom: var(--spacing-md);
}

.message-content ul, .message-content ol {
    margin-bottom: var(--spacing-md);
    padding-left: var(--spacing-lg);
}

.message-actions {
    display: flex;
    justify-content: flex-end;
    gap: var(--spacing-sm);
    margin-top: var(--spacing-sm);
    opacity: 0;
    transition: opacity var(--transition-fast);
}

.message:hover .message-actions {
    opacity: 1;
}

/* Suggestion chips */
.suggestion-chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-sm);
    margin-top: var(--spacing-md);
}

.suggestion-chip {
    background-color: var(--white);
    border: 1px solid var(--neutral-lighter);
    border-radius: 16px;
    padding: 6px 12px;
    font-size: var(--font-size-sm);
    cursor: pointer;
    transition: all var(--transition-fast);
    white-space: nowrap;
}

.suggestion-chip:hover {
    background-color: var(--primary-light);
    color: var(--white);
    border-color: var(--primary-light);
}

/* Chat input area */
.chat-input-area {
    padding: var(--spacing-md) var(--spacing-lg);
    background-color: var(--white);
    border-top: 1px solid var(--neutral-lighter);
    display: flex;
    gap: var(--spacing-md);
    align-items: flex-end;
}

.input-wrapper {
    flex: 1;
    position: relative;
    background-color: var(--background-light);
    border: 1px solid var(--neutral-lighter);
    border-radius: var(--radius-md);
    transition: border-color var(--transition-fast);
}

.input-wrapper:focus-within {
    border-color: var(--primary);
}

.chat-input-area textarea {
    width: 100%;
    border: none;
    background: transparent;
    padding: var(--spacing-md);
    padding-right: 70px;
    font-family: var(--font-family);
    font-size: var(--font-size-md);
    resize: none;
    max-height: 200px;
    outline: none;
}

.input-actions {
    position: absolute;
    right: 8px;
    bottom: 8px;
    display: flex;
    gap: 4px;
}

/* Right panel */
.right-panel {
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: 380px;
    background-color: var(--white);
    box-shadow: var(--shadow-large);
    display: flex;
    flex-direction: column;
    z-index: 20;
    transform: translateX(100%);
    transition: transform var(--transition-normal);
}

.right-panel.visible {
    transform: translateX(0);
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--spacing-md) var(--spacing-lg);
    border-bottom: 1px solid var(--neutral-lighter);
}

.panel-header h3 {
    font-size: var(--font-size-md);
}

.panel-content {
    flex: 1;
    overflow-y: auto;
    padding: var(--spacing-md);
}

.panel-footer {
    padding: var(--spacing-md) var(--spacing-lg);
    border-top: 1px solid var(--neutral-lighter);
    display: flex;
    justify-content: flex-end;
    gap: var(--spacing-md);
}

/* Search result items */
.search-result-list {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
}

.search-result-item {
    background-color: var(--background-light);
    border-radius: var(--radius-md);
    overflow: hidden;
    transition: transform var(--transition-fast), box-shadow var(--transition-fast);
    cursor: pointer;
    position: relative;
}

.search-result-item:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-medium);
}

.search-result-item.jira {
    border-left: 4px solid var(--blue);
}

.search-result-item.confluence {
    border-left: 4px solid var(--purple);
}

.search-result-header {
    padding: var(--spacing-md);
    border-bottom: 1px solid var(--neutral-lighter);
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.search-result-header i {
    color: var(--neutral);
}

.search-result-header.jira i {
    color: var(--blue);
}

.search-result-header.confluence i {
    color: var(--purple);
}

.search-result-header h4 {
    font-size: var(--font-size-md);
    flex: 1;
}

.search-result-body {
    padding: var(--spacing-md);
}

.search-result-snippet {
    font-size: var(--font-size-sm);
    color: var(--neutral);
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.search-result-footer {
    padding: var(--spacing-sm) var(--spacing-md);
    background-color: rgba(0, 0, 0, 0.02);
    border-top: 1px solid var(--neutral-lighter);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: var(--font-size-xs);
    color: var(--neutral);
}

.search-result-meta {
    display: flex;
    gap: var(--spacing-md);
}

.search-result-match {
    padding: 2px 6px;
    background-color: rgba(0, 82, 204, 0.1);
    border-radius: var(--radius-sm);
    color: var(--primary);
    font-weight: 500;
}

/* Context items */
.context-item {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
    padding: var(--spacing-sm);
    background-color: var(--background-light);
    border-radius: var(--radius-sm);
    margin-bottom: var(--spacing-sm);
}

.context-item-icon {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 32px;
    height: 32px;
    border-radius: var(--radius-round);
    background-color: var(--neutral-lighter);
}

.context-item-icon.jira {
    background-color: rgba(0, 101, 255, 0.1);
    color: var(--blue);
}

.context-item-icon.confluence {
    background-color: rgba(101, 84, 192, 0.1);
    color: var(--purple);
}

.context-item-content {
    flex: 1;
    min-width: 0;
}

.context-item-title {
    font-size: var(--font-size-sm);
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.context-item-subtitle {
    font-size: var(--font-size-xs);
    color: var(--neutral);
}

.context-item-actions {
    display: flex;
    gap: var(--spacing-xs);
}

/* Knowledge graph components */
.knowledge-graph-controls {
    display: flex;
    justify-content: space-between;
    margin-bottom: var(--spacing-md);
    gap: var(--spacing-md);
}

.filter-controls {
    display: flex;
    gap: var(--spacing-md);
}

.checkbox-label {
    display: flex;
    align-items: center;
    cursor: pointer;
    gap: var(--spacing-sm);
}

.checkbox-text {
    font-size: var(--font-size-sm);
}

.knowledge-graph-container {
    height: 400px;
    border: 1px solid var(--neutral-lighter);
    border-radius: var(--radius-md);
    overflow: hidden;
    background-color: var(--background-light);
}

.knowledge-graph-legend {
    display: flex;
    justify-content: center;
    gap: var(--spacing-lg);
    margin-top: var(--spacing-md);
}

.legend-item {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    font-size: var(--font-size-sm);
}

.legend-icon {
    width: 16px;
    height: 16px;
    border-radius: var(--radius-round);
}

.jira-icon {
    background-color: var(--blue);
}

.confluence-icon {
    background-color: var(--purple);
}

.relationship-line {
    width: 16px;
    height: 2px;
    background-color: var(--neutral);
}

/* Modals */
.modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(9, 30, 66, 0.54);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    opacity: 0;
    visibility: hidden;
    transition: opacity var(--transition-normal), visibility var(--transition-normal);
}

.modal.visible {
    opacity: 1;
    visibility: visible;
}

.modal-dialog {
    width: 600px;
    max-width: 90vw;
    max-height: 90vh;
    background-color: var(--white);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-large);
    display: flex;
    flex-direction: column;
    transform: translateY(20px);
    opacity: 0;
    transition: transform var(--transition-normal), opacity var(--transition-normal);
}

.modal-dialog.modal-lg {
    width: 800px;
}

.modal.visible .modal-dialog {
    transform: translateY(0);
    opacity: 1;
}

.modal-content {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    height: 100%;
}

.modal-header {
    padding: var(--spacing-md) var(--spacing-lg);
    border-bottom: 1px solid var(--neutral-lighter);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.modal-header h3 {
    font-size: var(--font-size-lg);
}

.modal-body {
    flex: 1;
    overflow-y: auto;
    padding: var(--spacing-lg);
}

.modal-footer {
    padding: var(--spacing-md) var(--spacing-lg);
    border-top: 1px solid var(--neutral-lighter);
    display: flex;
    justify-content: flex-end;
    gap: var(--spacing-md);
}

/* Help modal styles */
.help-section {
    margin-bottom: var(--spacing-lg);
}

.help-section h4 {
    margin-bottom: var(--spacing-md);
    font-size: var(--font-size-md);
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.help-section p {
    margin-bottom: var(--spacing-md);
}

.help-section ul, .help-section ol {
    margin-bottom: var(--spacing-md);
    padding-left: var(--spacing-lg);
}

.shortcuts-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: var(--spacing-md);
}

.shortcut {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
}

.key {
    background-color: var(--neutral-lighter);
    padding: 2px 8px;
    border-radius: var(--radius-sm);
    font-family: monospace;
    font-size: var(--font-size-sm);
}

/* Reasoning steps modal */
.reasoning-steps {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
}

.reasoning-step {
    border: 1px solid var(--neutral-lighter);
    border-radius: var(--radius-md);
    overflow: hidden;
}

.step-header {
    padding: var(--spacing-md);
    background-color: var(--background-light);
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
}

.step-header h4 {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    font-size: var(--font-size-md);
}

.step-content {
    padding: var(--spacing-md);
    border-top: 1px solid var(--neutral-lighter);
    display: none;
}

.step-content.visible {
    display: block;
}

/* Upload modal */
.upload-area {
    border: 2px dashed var(--neutral-lighter);
    border-radius: var(--radius-md);
    padding: var(--spacing-xl);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--spacing-md);
    cursor: pointer;
    transition: background-color var(--transition-fast);
}

.upload-area:hover {
    background-color: var(--background-light);
}

.upload-area i {
    font-size: 48px;
    color: var(--neutral-light);
}

.upload-info {
    font-size: var(--font-size-xs);
    color: var(--neutral);
}

.upload-preview {
    margin-top: var(--spacing-lg);
    display: none;
}

.upload-preview.visible {
    display: block;
}

.file-preview {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
    padding: var(--spacing-md);
    background-color: var(--background-light);
    border-radius: var(--radius-md);
}

.file-icon {
    font-size: 24px;
    color: var(--neutral);
}

.file-icon.pdf {
    color: var(--red);
}

.file-icon.doc {
    color: var(--blue);
}

.file-icon.xls {
    color: var(--green);
}

.file-icon.img {
    color: var(--purple);
}

.file-info {
    flex: 1;
}

.file-name {
    font-weight: 500;
}

.file-size {
    font-size: var(--font-size-xs);
    color: var(--neutral);
}

.file-actions {
    display: flex;
    gap: var(--spacing-sm);
}

/* Buttons */
.btn-primary, 
.btn-secondary, 
.btn-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--spacing-sm);
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--radius-sm);
    font-size: var(--font-size-sm);
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-fast);
    border: none;
    outline: none;
}

.btn-sm {
    padding: 4px 8px;
    font-size: var(--font-size-xs);
}

.btn-primary {
    background-color: var(--primary);
    color: var(--white);
}

.btn-primary:hover {
    background-color: var(--primary-dark);
}

.btn-primary:disabled {
    background-color: var(--neutral-lighter);
    color: var(--neutral);
    cursor: not-allowed;
}

.btn-secondary {
    background-color: var(--white);
    color: var(--neutral-dark);
    border: 1px solid var(--neutral-lighter);
}

.btn-secondary:hover {
    background-color: var(--background-light);
}

.btn-secondary:disabled {
    color: var(--neutral-light);
    cursor: not-allowed;
}

.btn-icon {
    background-color: transparent;
    color: var(--neutral);
    padding: var(--spacing-xs);
}

.btn-icon:hover {
    background-color: var(--background-light);
    color: var(--primary);
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .app-container {
        flex-direction: column;
    }

    .sidebar {
        width: 100%;
        height: auto;
        max-height: 50vh;
    }

    .right-panel {
        width: 100%;
    }
}

















/**
 * Enhanced API Client for Jira-Confluence chatbot
 */
class ApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl || '';
        this.pendingRequests = {};
    }

    /**
     * Generic request method with caching and abort support
     * @param {string} endpoint - API endpoint
     * @param {string} method - HTTP method (GET, POST, etc.)
     * @param {Object} data - Request data (for POST, PUT, etc.)
     * @param {Object} options - Additional options (abortKey, useCache, etc.)
     * @returns {Promise} - Request promise
     */
    async request(endpoint, method = 'GET', data = null, options = {}) {
        const url = `${this.baseUrl}/api/${endpoint}`;
        const { abortKey, useCache = true, cacheTTL = 300000 } = options; // cacheTTL: 5 minutes by default
        
        // Generate cache key (if caching enabled)
        const cacheKey = useCache ? `${method}:${url}:${JSON.stringify(data)}` : null;
        
        // Check cache first (for GET requests)
        if (method === 'GET' && useCache) {
            const cachedResponse = this._getCachedResponse(cacheKey);
            if (cachedResponse) {
                return cachedResponse;
            }
        }
        
        // Create abort controller for this request
        const controller = new AbortController();
        
        // Store abort controller if abortKey is provided
        if (abortKey) {
            // Abort previous request with the same key
            if (this.pendingRequests[abortKey]) {
                this.pendingRequests[abortKey].abort();
            }
            this.pendingRequests[abortKey] = controller;
        }
        
        // Prepare request options
        const requestOptions = {
            method,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            signal: controller.signal
        };
        
        // Add body for non-GET requests
        if (data && method !== 'GET') {
            requestOptions.body = JSON.stringify(data);
        }
        
        try {
            // Make the request
            const response = await fetch(url, requestOptions);
            
            // Check for HTTP errors
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({
                    error: `HTTP error ${response.status}`
                }));
                throw new Error(errorData.error || `HTTP error ${response.status}`);
            }
            
            // Parse response as JSON
            const result = await response.json();
            
            // Cache the response (for GET requests)
            if (method === 'GET' && useCache) {
                this._cacheResponse(cacheKey, result, cacheTTL);
            }
            
            // Clean up pending request
            if (abortKey) {
                delete this.pendingRequests[abortKey];
            }
            
            return result;
        } catch (error) {
            // Ignore abort errors
            if (error.name === 'AbortError') {
                console.log(`Request aborted: ${endpoint}`);
                return null;
            }
            
            // Log and rethrow other errors
            console.error(`API Error (${endpoint}):`, error);
            throw error;
        }
    }
    
    /**
     * Get a cached response
     * @param {string} key - Cache key
     * @returns {Object|null} - Cached response or null
     */
    _getCachedResponse(key) {
        try {
            const cachedItem = localStorage.getItem(`api_cache:${key}`);
            if (!cachedItem) return null;
            
            const { data, expiry } = JSON.parse(cachedItem);
            
            // Check if cache has expired
            if (Date.now() > expiry) {
                localStorage.removeItem(`api_cache:${key}`);
                return null;
            }
            
            return data;
        } catch (error) {
            console.warn('Error retrieving from cache:', error);
            return null;
        }
    }
    
    /**
     * Cache a response
     * @param {string} key - Cache key
     * @param {Object} data - Data to cache
     * @param {number} ttl - Time to live (ms)
     */
    _cacheResponse(key, data, ttl) {
        try {
            const cacheItem = {
                data,
                expiry: Date.now() + ttl
            };
            localStorage.setItem(`api_cache:${key}`, JSON.stringify(cacheItem));
        } catch (error) {
            console.warn('Error storing in cache:', error);
        }
    }
    
    /**
     * Clear all cached responses
     */
    clearCache() {
        try {
            const keys = [];
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key.startsWith('api_cache:')) {
                    keys.push(key);
                }
            }
            
            keys.forEach(key => localStorage.removeItem(key));
            console.log(`Cleared ${keys.length} cached responses`);
        } catch (error) {
            console.warn('Error clearing cache:', error);
        }
    }
    
    /**
     * Abort all pending requests
     */
    abortAll() {
        Object.values(this.pendingRequests).forEach(controller => {
            try {
                controller.abort();
            } catch (error) {
                console.warn('Error aborting request:', error);
            }
        });
        this.pendingRequests = {};
    }
    
    /**
     * Get system status
     * @returns {Promise} - Promise resolving to status data
     */
    async getStatus() {
        return this.request('status');
    }
    
    /**
     * Search for content
     * @param {string} query - Search query
     * @param {string} source - Source to search (jira, confluence, combined)
     * @returns {Promise} - Promise resolving to search results
     */
    async search(query, source = 'combined') {
        // Use abortKey to cancel previous search requests
        return this.request('search', 'POST', { query, source }, { abortKey: 'search' });
    }
    
    /**
     * Get chat response
     * @param {string} query - User query
     * @param {string} source - Source to use (jira, confluence, combined)
     * @param {Array} contextIds - IDs of context documents
     * @param {boolean} needsReasoning - Whether to use detailed reasoning
     * @returns {Promise} - Promise resolving to chat response
     */
    async chat(query, source = 'combined', contextIds = [], needsReasoning = false) {
        return this.request('chat', 'POST', { 
            query, 
            source, 
            context_ids: contextIds,
            needs_reasoning: needsReasoning 
        }, { abortKey: 'chat' });
    }
    
    /**
     * Get a specific Jira ticket
     * @param {string} ticketId - Jira ticket ID
     * @returns {Promise} - Promise resolving to ticket data
     */
    async getJiraTicket(ticketId) {
        return this.request(`jira/ticket/${ticketId}`, 'GET');
    }
    
    /**
     * Get a specific Confluence page
     * @param {string} pageId - Confluence page ID
     * @returns {Promise} - Promise resolving to page data
     */
    async getConfluencePage(pageId) {
        return this.request(`confluence/page/${pageId}`, 'GET');
    }
    
    /**
     * Find solutions in Confluence for a Jira ticket
     * @param {string} ticketId - Jira ticket ID
     * @param {string} query - Optional additional query to focus the search
     * @returns {Promise} - Promise resolving to solutions data
     */
    async findSolution(ticketId, query = '') {
        return this.request('find-solution', 'POST', { 
            ticket_id: ticketId, 
            query 
        }, { abortKey: 'find-solution' });
    }
    
    /**
     * Get knowledge graph data
     * @returns {Promise} - Promise resolving to knowledge graph data
     */
    async getKnowledgeGraph() {
        return this.request('knowledge-graph', 'GET');
    }
    
    /**
     * Get relationships between Jira tickets and Confluence pages
     * @returns {Promise} - Promise resolving to relationships data
     */
    async getRelationships() {
        return this.request('relationships', 'GET');
    }
    
    /**
     * Analyze file content
     * @param {File} file - File to analyze
     * @returns {Promise} - Promise resolving to analysis data
     */
    async analyzeFile(file) {
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Custom request for file upload
        const url = `${this.baseUrl}/api/analyze-file`;
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({
                    error: `HTTP error ${response.status}`
                }));
                throw new Error(errorData.error || `HTTP error ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('File analysis error:', error);
            throw error;
        }
    }
    
    /**
     * Clear server cache
     * @returns {Promise} - Promise resolving to success message
     */
    async clearServerCache() {
        return this.request('clear-cache', 'POST', {}, { useCache: false });
    }
}

// Create a global instance
const api = new ApiClient();














/**
 * Knowledge Graph visualization with D3.js
 */
class KnowledgeGraph {
    constructor(containerId, data = null) {
        this.container = document.getElementById(containerId);
        this.data = data || { nodes: [], links: [] };
        this.simulation = null;
        this.svg = null;
        this.width = 0;
        this.height = 0;
        this.nodeElements = null;
        this.linkElements = null;
        this.textElements = null;
        this.dragHandler = null;
        this.zoomHandler = null;
        this.tooltip = null;
        this.filters = {
            showJira: true,
            showConfluence: true,
            showRelationships: true,
            searchTerm: ''
        };
        
        // Initialize if data is provided
        if (data) {
            this.initialize();
        }
    }
    
    /**
     * Set the data for the graph
     * @param {Object} data - Graph data (nodes and links)
     */
    setData(data) {
        this.data = data;
        
        // Initialize or update the graph
        if (this.svg) {
            this.updateGraph();
        } else {
            this.initialize();
        }
    }
    
    /**
     * Set filters for the graph
     * @param {Object} filters - Filter settings
     */
    setFilters(filters) {
        this.filters = { ...this.filters, ...filters };
        this.updateGraph();
    }
    
    /**
     * Initialize the graph
     */
    initialize() {
        if (!this.container) {
            console.error('Container element not found');
            return;
        }
        
        // Clear any existing content
        this.container.innerHTML = '';
        
        // Get container dimensions
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        
        // Create SVG element
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${this.width} ${this.height}`)
            .attr('class', 'knowledge-graph-svg');
        
        // Create tooltip
        this.tooltip = d3.select(this.container)
            .append('div')
            .attr('class', 'graph-tooltip')
            .style('position', 'absolute')
            .style('opacity', 0)
            .style('background-color', 'white')
            .style('border', '1px solid #ccc')
            .style('border-radius', '4px')
            .style('padding', '8px')
            .style('box-shadow', '0 2px 4px rgba(0,0,0,0.2)')
            .style('pointer-events', 'none')
            .style('z-index', 1000);
        
        // Create groups for links and nodes
        const container = this.svg.append('g').attr('class', 'graph-container');
        
        // Add zoom behavior
        this.zoomHandler = d3.zoom()
            .scaleExtent([0.25, 4])
            .on('zoom', (event) => {
                container.attr('transform', event.transform);
            });
        
        this.svg.call(this.zoomHandler);
        
        // Create drag behavior
        this.dragHandler = d3.drag()
            .on('start', this._dragStarted.bind(this))
            .on('drag', this._dragged.bind(this))
            .on('end', this._dragEnded.bind(this));
        
        // Create link elements
        this.linkElements = container.append('g')
            .attr('class', 'links')
            .selectAll('line');
        
        // Create node elements
        this.nodeElements = container.append('g')
            .attr('class', 'nodes')
            .selectAll('circle');
        
        // Create text elements
        this.textElements = container.append('g')
            .attr('class', 'texts')
            .selectAll('text');
        
        // Apply force simulation
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(50));
        
        // Initial update
        this.updateGraph();
    }
    
    /**
     * Update the graph based on data and filters
     */
    updateGraph() {
        if (!this.svg || !this.data) return;
        
        // Apply filters to nodes
        const filteredNodes = this.data.nodes.filter(node => {
            // Filter by type
            if (node.type === 'jira' && !this.filters.showJira) return false;
            if (node.type === 'confluence' && !this.filters.showConfluence) return false;
            
            // Filter by search term
            if (this.filters.searchTerm && this.filters.searchTerm.trim() !== '') {
                const term = this.filters.searchTerm.toLowerCase();
                const title = (node.title || '').toLowerCase();
                const id = (node.id || '').toString().toLowerCase();
                
                if (!title.includes(term) && !id.includes(term)) {
                    return false;
                }
            }
            
            return true;
        });
        
        // Get node IDs for link filtering
        const nodeIds = new Set(filteredNodes.map(n => n.id));
        
        // Apply filters to links
        const filteredLinks = this.data.links.filter(link => {
            // Filter by relationship visibility
            if (!this.filters.showRelationships) return false;
            
            // Only include links where both source and target nodes are visible
            return nodeIds.has(link.source) && nodeIds.has(link.target);
        });
        
        // Update links
        this.linkElements = this.linkElements
            .data(filteredLinks, d => `${d.source}-${d.target}`)
            .join(
                enter => enter.append('line')
                    .attr('stroke-width', d => Math.max(1, d.weight || 1) * 2)
                    .attr('stroke', '#999')
                    .attr('stroke-opacity', 0.6)
                    .attr('class', 'graph-link')
                    .on('mouseover', (event, d) => {
                        const sourceNode = filteredNodes.find(n => n.id === d.source);
                        const targetNode = filteredNodes.find(n => n.id === d.target);
                        
                        if (sourceNode && targetNode) {
                            this.tooltip
                                .html(`<strong>Relationship</strong><br>
                                       <strong>${sourceNode.title || sourceNode.id}</strong> → 
                                       <strong>${targetNode.title || targetNode.id}</strong><br>
                                       ${d.relationship || ''}`)
                                .style('left', (event.pageX + 10) + 'px')
                                .style('top', (event.pageY - 20) + 'px')
                                .style('opacity', 0.9);
                        }
                    })
                    .on('mouseout', () => {
                        this.tooltip.style('opacity', 0);
                    }),
                update => update
                    .attr('stroke-width', d => Math.max(1, d.weight || 1) * 2),
                exit => exit.remove()
            );
        
        // Update nodes
        this.nodeElements = this.nodeElements
            .data(filteredNodes, d => d.id)
            .join(
                enter => enter.append('circle')
                    .attr('r', 10)
                    .attr('fill', d => this._getNodeColor(d))
                    .attr('class', 'graph-node')
                    .call(this.dragHandler)
                    .on('click', (event, d) => {
                        this._nodeClick(event, d);
                    })
                    .on('mouseover', (event, d) => {
                        // Show tooltip
                        this.tooltip
                            .html(`<strong>${d.title || d.id}</strong><br>
                                  <span style="color: #666;">${d.type}</span>`)
                            .style('left', (event.pageX + 10) + 'px')
                            .style('top', (event.pageY - 20) + 'px')
                            .style('opacity', 0.9);
                        
                        // Highlight connected nodes
                        this._highlightConnections(d);
                    })
                    .on('mouseout', () => {
                        this.tooltip.style('opacity', 0);
                        this._resetHighlighting();
                    }),
                update => update
                    .attr('fill', d => this._getNodeColor(d)),
                exit => exit.remove()
            );
        
        // Update text labels
        this.textElements = this.textElements
            .data(filteredNodes, d => d.id)
            .join(
                enter => enter.append('text')
                    .text(d => this._getNodeLabel(d))
                    .attr('font-size', 10)
                    .attr('text-anchor', 'middle')
                    .attr('dy', 20)
                    .style('pointer-events', 'none')
                    .attr('class', 'graph-text'),
                update => update
                    .text(d => this._getNodeLabel(d)),
                exit => exit.remove()
            );
        
        // Update simulation
        this.simulation
            .nodes(filteredNodes)
            .on('tick', () => {
                this.linkElements
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                this.nodeElements
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
                
                this.textElements
                    .attr('x', d => d.x)
                    .attr('y', d => d.y);
            });
        
        // Update link force
        this.simulation.force('link')
            .links(filteredLinks);
        
        // Restart simulation
        this.simulation.alpha(1).restart();
    }
    
    /**
     * Get color for a node based on its type
     * @param {Object} node - Node data
     * @returns {string} - Color code
     */
    _getNodeColor(node) {
        switch (node.type) {
            case 'jira':
                return '#0065FF';
            case 'confluence':
                return '#6554C0';
            default:
                return '#97A0AF';
        }
    }
    
    /**
     * Get label for a node
     * @param {Object} node - Node data
     * @returns {string} - Node label
     */
    _getNodeLabel(node) {
        if (node.title && node.title.length > 15) {
            return node.title.substring(0, 12) + '...';
        }
        return node.title || node.id;
    }
    
    /**
     * Highlight node connections
     * @param {Object} node - Node to highlight connections for
     */
    _highlightConnections(node) {
        // Dim all nodes and links
        this.nodeElements.attr('opacity', 0.3);
        this.linkElements.attr('opacity', 0.3);
        this.textElements.attr('opacity', 0.3);
        
        // Find connected node IDs
        const connectedNodeIds = new Set();
        connectedNodeIds.add(node.id);
        
        this.linkElements.each(d => {
            if (d.source.id === node.id || d.source === node.id) {
                connectedNodeIds.add(d.target.id || d.target);
            } else if (d.target.id === node.id || d.target === node.id) {
                connectedNodeIds.add(d.source.id || d.source);
            }
        });
        
        // Highlight connected nodes and links
        this.nodeElements
            .filter(d => connectedNodeIds.has(d.id))
            .attr('opacity', 1);
        
        this.textElements
            .filter(d => connectedNodeIds.has(d.id))
            .attr('opacity', 1);
        
        this.linkElements
            .filter(d => {
                const sourceId = d.source.id || d.source;
                const targetId = d.target.id || d.target;
                return sourceId === node.id || targetId === node.id;
            })
            .attr('opacity', 1)
            .attr('stroke', '#333')
            .attr('stroke-width', d => (d.weight || 1) * 3);
    }
    
    /**
     * Reset highlighting
     */
    _resetHighlighting() {
        this.nodeElements.attr('opacity', 1);
        this.linkElements
            .attr('opacity', 0.6)
            .attr('stroke', '#999')
            .attr('stroke-width', d => Math.max(1, d.weight || 1) * 2);
        this.textElements.attr('opacity', 1);
    }
    
    /**
     * Handle node click
     * @param {Event} event - Click event
     * @param {Object} node - Node data
     */
    _nodeClick(event, node) {
        // Dispatch custom event for node click
        const clickEvent = new CustomEvent('node:click', {
            detail: { node }
        });
        this.container.dispatchEvent(clickEvent);
    }
    
    /**
     * Drag started handler
     */
    _dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    /**
     * Dragged handler
     */
    _dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    /**
     * Drag ended handler
     */
    _dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    
    /**
     * Center the graph and fit to view
     */
    centerGraph() {
        const svg = d3.select(this.container).select('svg');
        const container = svg.select('.graph-container');
        
        svg.call(this.zoomHandler.transform, d3.zoomIdentity);
    }
    
    /**
     * Convert Jira-Confluence data to graph format
     * @param {Object} data - Raw data from API
     * @returns {Object} - Formatted graph data
     */
    static formatGraphData(data) {
        if (!data) {
            return { nodes: [], links: [] };
        }
        
        // Process nodes
        const nodes = [];
        const nodeMap = {};
        
        // Add Jira nodes
        if (data.jira_tickets) {
            data.jira_tickets.forEach(ticket => {
                const node = {
                    id: ticket.id,
                    type: 'jira',
                    title: ticket.key,
                    key: ticket.key,
                    summary: ticket.summary,
                    status: ticket.status,
                    url: ticket.url
                };
                
                nodes.push(node);
                nodeMap[ticket.id] = node;
            });
        }
        
        // Add Confluence nodes
        if (data.confluence_pages) {
            data.confluence_pages.forEach(page => {
                const node = {
                    id: page.id,
                    type: 'confluence',
                    title: page.title,
                    space: page.space,
                    url: page.url
                };
                
                nodes.push(node);
                nodeMap[page.id] = node;
            });
        }
        
        // Process links
        const links = [];
        
        if (data.relationships) {
            data.relationships.forEach(rel => {
                // Only add link if both nodes exist
                if (nodeMap[rel.source_id] && nodeMap[rel.target_id]) {
                    links.push({
                        source: rel.source_id,
                        target: rel.target_id,
                        relationship: rel.type,
                        weight: rel.weight || 1
                    });
                }
            });
        }
        
        return { nodes, links };
    }
}

















/**
 * UI Components and Utilities
 */
class UIManager {
    constructor() {
        // DOM elements
        this.rightPanel = document.getElementById('right-panel');
        this.panelTitle = document.getElementById('panel-title');
        this.panelContent = document.getElementById('panel-content');
        this.panelFooter = document.getElementById('panel-footer');
        
        // Modals
        this.itemDetailsModal = document.getElementById('item-details-modal');
        this.knowledgeGraphModal = document.getElementById('knowledge-graph-modal');
        this.helpModal = document.getElementById('help-modal');
        this.contextModal = document.getElementById('context-modal');
        this.reasoningModal = document.getElementById('reasoning-modal');
        this.uploadModal = document.getElementById('upload-modal');
        
        // Item details elements
        this.itemDetailsTitle = document.getElementById('item-details-title');
        this.itemDetailsBody = document.getElementById('item-details-body');
        this.itemDetailsFooter = document.getElementById('item-details-footer');
        this.findSolutionsBtn = document.getElementById('find-solutions-btn');
        
        // Context panel
        this.contextSummary = document.getElementById('context-summary');
        this.contextItemsList = document.getElementById('context-items-list');
        this.manageContextBtn = document.getElementById('manage-context-btn');
        
        // Knowledge graph elements
        this.graphContainer = document.getElementById('knowledge-graph-container');
        this.showJiraCheckbox = document.getElementById('show-jira');
        this.showConfluenceCheckbox = document.getElementById('show-confluence');
        this.showRelationshipsCheckbox = document.getElementById('show-relationships');
        this.graphSearch = document.getElementById('graph-search');
        
        // Upload elements
        this.uploadArea = document.getElementById('upload-area');
        this.fileInput = document.getElementById('file-input');
        this.uploadPreview = document.getElementById('upload-preview');
        this.uploadFileBtn = document.getElementById('upload-file-btn');
        
        // Initialize
        this.initialize();
    }
    
    /**
     * Initialize UI manager
     */
    initialize() {
        // Initialize modals
        this._initializeModals();
        
        // Initialize right panel
        this._initializeRightPanel();
        
        // Initialize knowledge graph
        this._initializeKnowledgeGraph();
        
        // Initialize upload functionality
        this._initializeUpload();
        
        // Initialize reasoning steps
        this._initializeReasoningSteps();
    }
    
    /**
     * Initialize modals
     */
    _initializeModals() {
        // Close modals when clicking outside or on close button
        document.querySelectorAll('.modal').forEach(modal => {
            // Close when clicking outside content
            modal.addEventListener('click', (event) => {
                if (event.target === modal) {
                    this.hideModal(modal);
                }
            });
            
            // Close when clicking close button
            modal.querySelectorAll('.close-modal').forEach(closeBtn => {
                closeBtn.addEventListener('click', () => {
                    this.hideModal(modal);
                });
            });
        });
        
        // Close modals with Escape key
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                document.querySelectorAll('.modal.visible').forEach(modal => {
                    this.hideModal(modal);
                });
            }
        });
        
        // Set up context management
        document.getElementById('clear-context-btn').addEventListener('click', () => {
            // Dispatch event to clear context
            document.dispatchEvent(new CustomEvent('context:clear'));
        });
    }
    
    /**
     * Initialize right panel
     */
    _initializeRightPanel() {
        // Close panel button
        document.getElementById('close-panel-btn').addEventListener('click', () => {
            this.hideRightPanel();
        });
    }
    
    /**
     * Initialize knowledge graph
     */
    _initializeKnowledgeGraph() {
        // Create graph instance
        this.graph = new KnowledgeGraph('knowledge-graph-container');
        
        // Graph filter events
        if (this.showJiraCheckbox) {
            this.showJiraCheckbox.addEventListener('change', () => {
                this._updateGraphFilters();
            });
        }
        
        if (this.showConfluenceCheckbox) {
            this.showConfluenceCheckbox.addEventListener('change', () => {
                this._updateGraphFilters();
            });
        }
        
        if (this.showRelationshipsCheckbox) {
            this.showRelationshipsCheckbox.addEventListener('change', () => {
                this._updateGraphFilters();
            });
        }
        
        if (this.graphSearch) {
            this.graphSearch.addEventListener('input', () => {
                this._updateGraphFilters();
            });
        }
        
        // Graph node click event
        this.graphContainer.addEventListener('node:click', (event) => {
            const node = event.detail.node;
            
            // Fetch and show node details
            if (node.type === 'jira') {
                this._fetchAndShowJiraTicket(node.key || node.id);
            } else if (node.type === 'confluence') {
                this._fetchAndShowConfluencePage(node.id);
            }
        });
    }
    
    /**
     * Update graph filters
     */
    _updateGraphFilters() {
        if (!this.graph) return;
        
        const filters = {
            showJira: this.showJiraCheckbox ? this.showJiraCheckbox.checked : true,
            showConfluence: this.showConfluenceCheckbox ? this.showConfluenceCheckbox.checked : true,
            showRelationships: this.showRelationshipsCheckbox ? this.showRelationshipsCheckbox.checked : true,
            searchTerm: this.graphSearch ? this.graphSearch.value : ''
        };
        
        this.graph.setFilters(filters);
    }
    
    /**
     * Initialize file upload
     */
    _initializeUpload() {
        if (!this.uploadArea || !this.fileInput) return;
        
        // Open file dialog when clicking upload area
        this.uploadArea.addEventListener('click', () => {
            this.fileInput.click();
        });
        
        // Handle file selection
        this.fileInput.addEventListener('change', (event) => {
            if (event.target.files.length > 0) {
                this._handleFileSelection(event.target.files[0]);
            }
        });
        
        // Handle drag and drop
        this.uploadArea.addEventListener('dragover', (event) => {
            event.preventDefault();
            this.uploadArea.classList.add('dragover');
        });
        
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });
        
        this.uploadArea.addEventListener('drop', (event) => {
            event.preventDefault();
            this.uploadArea.classList.remove('dragover');
            
            if (event.dataTransfer.files.length > 0) {
                this._handleFileSelection(event.dataTransfer.files[0]);
            }
        });
        
        // Handle upload button
        if (this.uploadFileBtn) {
            this.uploadFileBtn.addEventListener('click', () => {
                this._uploadFile();
            });
        }
    }
    
    /**
     * Handle file selection
     * @param {File} file - Selected file
     */
    _handleFileSelection(file) {
        // Check file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            showNotification('File is too large. Maximum size is 10MB.', 'error');
            return;
        }
        
        // Store file for upload
        this.selectedFile = file;
        
        // Update preview
        if (this.uploadPreview) {
            let fileIconClass = 'fa-file';
            
            // Determine file type icon
            if (file.type.startsWith('image/')) {
                fileIconClass = 'fa-file-image';
            } else if (file.type === 'application/pdf') {
                fileIconClass = 'fa-file-pdf';
            } else if (file.type.includes('word')) {
                fileIconClass = 'fa-file-word';
            } else if (file.type.includes('excel') || file.type.includes('spreadsheet')) {
                fileIconClass = 'fa-file-excel';
            } else if (file.type === 'text/plain') {
                fileIconClass = 'fa-file-alt';
            }
            
            // Format file size
            const size = this._formatFileSize(file.size);
            
            // Update preview
            this.uploadPreview.innerHTML = `
                <div class="file-preview">
                    <i class="fa-solid ${fileIconClass} file-icon"></i>
                    <div class="file-info">
                        <div class="file-name">${file.name}</div>
                        <div class="file-size">${size}</div>
                    </div>
                    <div class="file-actions">
                        <button class="btn-icon" id="remove-file-btn">
                            <i class="fa-solid fa-times"></i>
                        </button>
                    </div>
                </div>
            `;
            
            // Show preview
            this.uploadPreview.classList.add('visible');
            
            // Enable upload button
            if (this.uploadFileBtn) {
                this.uploadFileBtn.disabled = false;
            }
            
            // Remove file button
            const removeFileBtn = document.getElementById('remove-file-btn');
            if (removeFileBtn) {
                removeFileBtn.addEventListener('click', (event) => {
                    event.stopPropagation();
                    this._clearFileSelection();
                });
            }
        }
    }
    
    /**
     * Clear file selection
     */
    _clearFileSelection() {
        this.selectedFile = null;
        
        if (this.fileInput) {
            this.fileInput.value = '';
        }
        
        if (this.uploadPreview) {
            this.uploadPreview.innerHTML = '';
            this.uploadPreview.classList.remove('visible');
        }
        
        if (this.uploadFileBtn) {
            this.uploadFileBtn.disabled = true;
        }
    }
    
    /**
     * Upload selected file
     */
    _uploadFile() {
        if (!this.selectedFile) return;
        
        // Dispatch file upload event
        const uploadEvent = new CustomEvent('file:upload', {
            detail: { file: this.selectedFile }
        });
        
        document.dispatchEvent(uploadEvent);
        
        // Hide modal
        this.hideModal(this.uploadModal);
        
        // Clear selection
        this._clearFileSelection();
    }
    
    /**
     * Format file size
     * @param {number} bytes - File size in bytes
     * @returns {string} - Formatted file size
     */
    _formatFileSize(bytes) {
        if (bytes < 1024) {
            return bytes + ' B';
        } else if (bytes < 1024 * 1024) {
            return (bytes / 1024).toFixed(1) + ' KB';
        } else {
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }
    }
    
    /**
     * Initialize reasoning steps
     */
    _initializeReasoningSteps() {
        // Toggle step content visibility
        document.querySelectorAll('.step-header').forEach(header => {
            const targetId = header.getAttribute('data-target');
            if (!targetId) return;
            
            const targetContent = document.getElementById(targetId);
            if (!targetContent) return;
            
            header.addEventListener('click', () => {
                targetContent.classList.toggle('visible');
                
                // Toggle chevron icon
                const chevron = header.querySelector('.fa-chevron-down, .fa-chevron-up');
                if (chevron) {
                    chevron.classList.toggle('fa-chevron-down');
                    chevron.classList.toggle('fa-chevron-up');
                }
            });
        });
    }
    
    /**
     * Fetch and show Jira ticket details
     * @param {string} ticketId - Jira ticket ID
     */
    _fetchAndShowJiraTicket(ticketId) {
        api.getJiraTicket(ticketId)
            .then(response => {
                if (response && response.ticket) {
                    this.showItemDetails(response.ticket);
                } else {
                    showNotification(`Could not retrieve ticket ${ticketId}`, 'error');
                }
            })
            .catch(error => {
                console.error('Error fetching Jira ticket:', error);
                showNotification(`Error retrieving ticket: ${error.message}`, 'error');
            });
    }
    
    /**
     * Fetch and show Confluence page details
     * @param {string} pageId - Confluence page ID
     */
    _fetchAndShowConfluencePage(pageId) {
        api.getConfluencePage(pageId)
            .then(response => {
                if (response && response.page) {
                    this.showItemDetails(response.page);
                } else {
                    showNotification(`Could not retrieve page ${pageId}`, 'error');
                }
            })
            .catch(error => {
                console.error('Error fetching Confluence page:', error);
                showNotification(`Error retrieving page: ${error.message}`, 'error');
            });
    }
    
    /**
     * Show search results in right panel
     * @param {Array} results - Search results
     */
    showSearchResults(results) {
        if (!results || results.length === 0) {
            this.hideRightPanel();
            return;
        }
        
        // Set panel title
        this.panelTitle.textContent = `Search Results (${results.length})`;
        
        // Create results HTML
        let resultsHtml = '<div class="search-result-list">';
        
        results.forEach(result => {
            const sourceType = result.source_type || 'unknown';
            const metadata = result.metadata || {};
            const title = metadata.key || metadata.title || 'Untitled';
            const snippet = truncateText(result.content, 200);
            
            resultsHtml += `
                <div class="search-result-item ${sourceType}" data-id="${metadata.id}" data-source-type="${sourceType}">
                    <div class="search-result-header ${sourceType}">
                        <i class="fa-solid ${sourceType === 'jira' ? 'fa-ticket-alt' : 'fa-book'}"></i>
                        <h4>${escapeHtml(title)}</h4>
                    </div>
                    <div class="search-result-body">
                        <div class="search-result-snippet">${escapeHtml(snippet)}</div>
                    </div>
                    <div class="search-result-footer">
                        <div class="search-result-meta">
                            <span>${sourceType === 'jira' ? metadata.status || 'Unknown' : (metadata.space || 'Unknown')}</span>
                        </div>
                        <div class="search-result-match">Relevant Match</div>
                    </div>
                </div>
            `;
        });
        
        resultsHtml += '</div>';
        
        // Set panel content
        this.panelContent.innerHTML = resultsHtml;
        
        // Set panel footer
        this.panelFooter.innerHTML = `
            <button id="add-all-context-btn" class="btn-secondary">
                <i class="fa-solid fa-layer-group"></i> Add All to Context
            </button>
            <button id="close-results-btn" class="btn-primary">
                <i class="fa-solid fa-times"></i> Close
            </button>
        `;
        
        // Add event listeners
        document.querySelectorAll('.search-result-item').forEach(item => {
            item.addEventListener('click', () => {
                const itemId = item.getAttribute('data-id');
                const sourceType = item.getAttribute('data-source-type');
                
                // Find the corresponding result
                const result = results.find(r => 
                    r.metadata && r.metadata.id === itemId && r.source_type === sourceType
                );
                
                if (result) {
                    this.showItemDetails(result);
                }
            });
        });
        
        // Add all to context button
        const addAllBtn = document.getElementById('add-all-context-btn');
        if (addAllBtn) {
            addAllBtn.addEventListener('click', () => {
                // Dispatch event to add all to context
                document.dispatchEvent(new CustomEvent('context:add-all', {
                    detail: { items: results }
                }));
                
                // Show notification
                showNotification(`Added ${results.length} items to context`, 'success');
            });
        }
        
        // Close button
        const closeBtn = document.getElementById('close-results-btn');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                this.hideRightPanel();
            });
        }
        
        // Show panel
        this.showRightPanel();
    }
    
    /**
     * Show item details in modal
     * @param {Object} item - Item data
     */
    showItemDetails(item) {
        if (!item) return;
        
        const sourceType = item.source_type || 'unknown';
        const metadata = item.metadata || {};
        
        // Set modal title
        if (sourceType === 'jira') {
            this.itemDetailsTitle.innerHTML = `
                <i class="fa-solid fa-ticket-alt" style="color: var(--blue);"></i>
                Jira Ticket: ${escapeHtml(metadata.key || 'Unknown')}
            `;
        } else {
            this.itemDetailsTitle.innerHTML = `
                <i class="fa-solid fa-book" style="color: var(--purple);"></i>
                Confluence Page: ${escapeHtml(metadata.title || 'Unknown')}
            `;
        }
        
        // Set modal body
        let bodyHtml = '<div class="item-details">';
        
        // Header info
        bodyHtml += '<div class="item-header">';
        
        if (sourceType === 'jira') {
            bodyHtml += `
                <h3>${escapeHtml(metadata.summary || 'No summary')}</h3>
                <div class="item-meta">
                    <span><i class="fa-solid fa-tag"></i> ${escapeHtml(metadata.issuetype || 'Unknown')}</span>
                    <span><i class="fa-solid fa-circle-check"></i> ${escapeHtml(metadata.status || 'Unknown')}</span>
                    <span><i class="fa-solid fa-user"></i> ${escapeHtml(metadata.assignee || 'Unassigned')}</span>
                </div>
            `;
        } else {
            bodyHtml += `
                <h3>${escapeHtml(metadata.title || 'Untitled')}</h3>
                <div class="item-meta">
                    <span><i class="fa-solid fa-folder"></i> ${escapeHtml(metadata.space || 'Unknown')}</span>
                </div>
            `;
        }
        
        bodyHtml += '</div>'; // End header
        
        // Content
        bodyHtml += '<div class="item-content">';
        
        // Process content with Markdown if available
        if (item.content) {
            try {
                // Use marked library for Markdown rendering if available
                if (typeof marked !== 'undefined') {
                    bodyHtml += marked.parse(item.content);
                } else {
                    // Simple formatting
                    bodyHtml += formatText(item.content);
                }
            } catch (error) {
                console.error('Error formatting content:', error);
                bodyHtml += `<p>${escapeHtml(item.content)}</p>`;
            }
        } else {
            bodyHtml += '<p>No content available</p>';
        }
        
        bodyHtml += '</div>'; // End content
        
        // Add link to original
        if (item.url || metadata.url) {
            bodyHtml += `
                <div class="item-actions">
                    <a href="${item.url || metadata.url}" target="_blank" class="btn-secondary">
                        <i class="fa-solid fa-external-link-alt"></i> Open in ${sourceType === 'jira' ? 'Jira' : 'Confluence'}
                    </a>
                </div>
            `;
        }
        
        bodyHtml += '</div>'; // End item-details
        
        this.itemDetailsBody.innerHTML = bodyHtml;
        
        // Configure footer buttons
        if (sourceType === 'jira') {
            this.findSolutionsBtn.style.display = 'block';
            this.findSolutionsBtn.onclick = () => {
                // Hide modal
                this.hideModal(this.itemDetailsModal);
                
                // Dispatch find solutions event
                document.dispatchEvent(new CustomEvent('solutions:find', {
                    detail: { ticket: item }
                }));
            };
        } else {
            this.findSolutionsBtn.style.display = 'none';
        }
        
        // Add to context button
        const addToContextBtn = document.getElementById('add-to-context-btn');
        if (addToContextBtn) {
            addToContextBtn.onclick = () => {
                // Dispatch add to context event
                document.dispatchEvent(new CustomEvent('context:add', {
                    detail: { item }
                }));
                
                // Hide modal
                this.hideModal(this.itemDetailsModal);
                
                // Show notification
                showNotification('Added to context', 'success');
            };
        }
        
        // Show modal
        this.showModal(this.itemDetailsModal);
        
        // Store current item
        this.currentItem = item;
    }
    
    /**
     * Show reasoning steps
     * @param {Object} reasoning - Reasoning data
     */
    showReasoningSteps(reasoning) {
        if (!reasoning) return;
        
        // Set content for each step
        const analysisStep = document.getElementById('analysis-step');
        const informationStep = document.getElementById('information-step');
        const reasoningStep = document.getElementById('reasoning-step');
        
        if (analysisStep) {
            analysisStep.innerHTML = reasoning.analysis ? marked.parse(reasoning.analysis) : '<p>No analysis available</p>';
        }
        
        if (informationStep) {
            informationStep.innerHTML = reasoning.information_gathering ? 
                marked.parse(reasoning.information_gathering) : 
                '<p>No information gathering details available</p>';
        }
        
        if (reasoningStep) {
            reasoningStep.innerHTML = reasoning.reasoning ? 
                marked.parse(reasoning.reasoning) : 
                '<p>No reasoning details available</p>';
        }
        
        // Show first step by default
        if (analysisStep) {
            analysisStep.classList.add('visible');
            const header = document.querySelector(`[data-target="analysis-step"] i.fa-chevron-down`);
            if (header) {
                header.classList.remove('fa-chevron-down');
                header.classList.add('fa-chevron-up');
            }
        }
        
        // Show modal
        this.showModal(this.reasoningModal);
    }
    
    /**
     * Update context UI
     * @param {Array} contextItems - Context items
     */
    updateContextUI(contextItems) {
        // Update context summary
        if (this.contextSummary) {
            if (contextItems.length === 0) {
                this.contextSummary.innerHTML = `
                    <div class="empty-state">
                        <i class="fa-solid fa-layer-group"></i>
                        <p>No context items added yet</p>
                    </div>
                `;
                // Disable manage context button
                if (this.manageContextBtn) {
                    this.manageContextBtn.disabled = true;
                }
            } else {
                let html = '';
                
                // Show summary of first 3 items
                contextItems.slice(0, 3).forEach(item => {
                    const icon = item.sourceType === 'jira' ? 'fa-ticket-alt' : 'fa-book';
                    html += `
                        <div class="context-item">
                            <div class="context-item-icon ${item.sourceType}">
                                <i class="fa-solid ${icon}"></i>
                            </div>
                            <div class="context-item-content">
                                <div class="context-item-title">${escapeHtml(item.title)}</div>
                                <div class="context-item-subtitle">${item.sourceType}</div>
                            </div>
                        </div>
                    `;
                });
                
                // Add count if more items
                if (contextItems.length > 3) {
                    html += `<div class="context-count">+${contextItems.length - 3} more</div>`;
                }
                
                this.contextSummary.innerHTML = html;
                
                // Enable manage context button
                if (this.manageContextBtn) {
                    this.manageContextBtn.disabled = false;
                }
            }
        }
        
        // Update context items list (for modal)
        if (this.contextItemsList) {
            if (contextItems.length === 0) {
                this.contextItemsList.innerHTML = `
                    <div class="empty-state">
                        <i class="fa-solid fa-layer-group"></i>
                        <p>No context items added yet</p>
                    </div>
                `;
            } else {
                let html = '';
                
                contextItems.forEach((item, index) => {
                    const icon = item.sourceType === 'jira' ? 'fa-ticket-alt' : 'fa-book';
                    html += `
                        <div class="context-item">
                            <div class="context-item-icon ${item.sourceType}">
                                <i class="fa-solid ${icon}"></i>
                            </div>
                            <div class="context-item-content">
                                <div class="context-item-title">${escapeHtml(item.title)}</div>
                                <div class="context-item-subtitle">${item.sourceType}</div>
                            </div>
                            <div class="context-item-actions">
                                <button class="btn-icon remove-context-item" data-index="${index}">
                                    <i class="fa-solid fa-times"></i>
                                </button>
                            </div>
                        </div>
                    `;
                });
                
                this.contextItemsList.innerHTML = html;
                
                // Add remove handlers
                document.querySelectorAll('.remove-context-item').forEach(btn => {
                    btn.addEventListener('click', (event) => {
                        const index = parseInt(event.currentTarget.getAttribute('data-index'), 10);
                        
                        // Dispatch remove event
                        document.dispatchEvent(new CustomEvent('context:remove', {
                            detail: { index }
                        }));
                    });
                });
            }
        }
    }
    
    /**
     * Show right panel
     */
    showRightPanel() {
        if (this.rightPanel) {
            this.rightPanel.classList.add('visible');
        }
    }
    
    /**
     * Hide right panel
     */
    hideRightPanel() {
        if (this.rightPanel) {
            this.rightPanel.classList.remove('visible');
        }
    }
    
    /**
     * Show modal
     * @param {HTMLElement} modal - Modal element
     */
    showModal(modal) {
        if (modal) {
            modal.classList.add('visible');
        }
    }
    
    /**
     * Hide modal
     * @param {HTMLElement} modal - Modal element
     */
    hideModal(modal) {
        if (modal) {
            modal.classList.remove('visible');
        }
    }
    
    /**
     * Set app connection status
     * @param {Object} status - Connection status data
     */
    setConnectionStatus(status) {
        // Jira status
        const jiraStatus = document.getElementById('jira-status');
        if (jiraStatus) {
            jiraStatus.className = 'status-badge';
            if (status.connections.jira) {
                jiraStatus.classList.add('connected');
                jiraStatus.textContent = 'Connected';
            } else {
                jiraStatus.classList.add('disconnected');
                jiraStatus.textContent = 'Disconnected';
            }
        }
        
        // Confluence status
        const confluenceStatus = document.getElementById('confluence-status');
        if (confluenceStatus) {
            confluenceStatus.className = 'status-badge';
            if (status.connections.confluence) {
                confluenceStatus.classList.add('connected');
                confluenceStatus.textContent = 'Connected';
            } else {
                confluenceStatus.classList.add('disconnected');
                confluenceStatus.textContent = 'Disconnected';
            }
        }
    }
}

/**
 * Format text for display
 * @param {string} text - Input text
 * @returns {string} - Formatted HTML
 */
function formatText(text) {
    if (!text) return '';
    
    // Escape HTML
    const escaped = escapeHtml(text);
    
    // Convert line breaks to <br>
    const withLineBreaks = escaped.replace(/\n/g, '<br>');
    
    // Simple formatting for headers, code, etc.
    // This is a very basic implementation
    return withLineBreaks;
}

/**
 * Escape HTML special characters
 * @param {string} text - Input text
 * @returns {string} - Escaped text
 */
function escapeHtml(text) {
    if (!text) return '';
    
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Truncate text with ellipsis
 * @param {string} text - Input text
 * @param {number} maxLength - Maximum length
 * @returns {string} - Truncated text
 */
function truncateText(text, maxLength) {
    if (!text || text.length <= maxLength) {
        return text || '';
    }
    
    return text.substring(0, maxLength) + '...';
}

/**
 * Show notification
 * @param {string} message - Notification message
 * @param {string} type - Notification type (success, error, warning, info)
 * @param {number} duration - Duration in milliseconds
 */
function showNotification(message, type = 'info', duration = 3000) {
    // Check if notification container exists
    let container = document.querySelector('.notification-container');
    
    if (!container) {
        // Create container
        container = document.createElement('div');
        container.className = 'notification-container';
        container.style.position = 'fixed';
        container.style.top = '20px';
        container.style.right = '20px';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }
    
    // Create notification
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    // Icon based on type
    let iconClass = 'fa-info-circle';
    switch (type) {
        case 'success':
            iconClass = 'fa-check-circle';
            break;
        case 'error':
            iconClass = 'fa-times-circle';
            break;
        case 'warning':
            iconClass = 'fa-exclamation-triangle';
            break;
    }
    
    // Set content
    notification.innerHTML = `
        <i class="fa-solid ${iconClass}"></i>
        <div class="notification-message">${message}</div>
    `;
    
    // Add to container
    container.appendChild(notification);
    
    // Add animation
    notification.style.animation = 'fadeInRight 0.3s ease forwards';
    
    // Auto remove after duration
    setTimeout(() => {
        notification.style.animation = 'fadeOutRight 0.3s ease forwards';
        setTimeout(() => {
            container.removeChild(notification);
        }, 300);
    }, duration);
}

// Create keyframe animations for notifications
(function() {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes fadeInRight {
            from {
                opacity: 0;
                transform: translateX(20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes fadeOutRight {
            from {
                opacity: 1;
                transform: translateX(0);
            }
            to {
                opacity: 0;
                transform: translateX(20px);
            }
        }
        
        .notification-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .notification {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 16px;
            border-radius: 4px;
            background-color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            min-width: 280px;
            max-width: 400px;
        }
        
        .notification.info i {
            color: var(--blue);
        }
        
        .notification.success i {
            color: var(--green);
        }
        
        .notification.warning i {
            color: var(--yellow);
        }
        
        .notification.error i {
            color: var(--red);
        }
    `;
    
    document.head.appendChild(style);
})();

















/**
 * Enhanced Chat functionality for Jira-Confluence chatbot
 */
class ChatManager {
    constructor() {
        // DOM elements
        this.chatMessages = document.getElementById('chat-messages');
        this.chatInput = document.getElementById('chat-input');
        this.sendButton = document.getElementById('send-btn');
        this.clearInputButton = document.getElementById('clear-input-btn');
        this.sourceRadios = document.getElementsByName('source');
        
        // State
        this.isProcessing = false;
        this.contextItems = [];
        this.chatHistory = [];
        this.recentSearches = [];
        
        // UI manager
        this.ui = new UIManager();
        
        // Initialize
        this.initialize();
    }
    
    /**
     * Initialize chat manager
     */
    initialize() {
        // Set up event listeners
        this._setupEventListeners();
        
        // Auto-resize input
        this._setupInputAutoResize();
        
        // Check server status
        this._checkServerStatus();
        
        // Initialize buttons state
        this._updateButtonsState();
    }
    
    /**
     * Set up event listeners
     */
    _setupEventListeners() {
        // Send button
        if (this.sendButton) {
            this.sendButton.addEventListener('click', () => {
                this.sendMessage();
            });
        }
        
        // Input events
        if (this.chatInput) {
            // Send on Enter (but not with Shift)
            this.chatInput.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    this.sendMessage();
                }
            });
            
            // Update buttons state on input
            this.chatInput.addEventListener('input', () => {
                this._updateButtonsState();
            });
        }
        
        // Clear input button
        if (this.clearInputButton) {
            this.clearInputButton.addEventListener('click', () => {
                this.chatInput.value = '';
                this._updateButtonsState();
                this.chatInput.focus();
            });
        }
        
        // Suggestion chips
        document.querySelectorAll('.suggestion-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const query = chip.getAttribute('data-query');
                if (query) {
                    this.chatInput.value = query;
                    this._updateButtonsState();
                    this.sendMessage();
                }
            });
        });
        
        // Clear chat button
        document.getElementById('clear-chat-btn').addEventListener('click', () => {
            this.clearChat();
        });
        
        // Clear cache button
        document.getElementById('clear-cache-btn').addEventListener('click', () => {
            this.clearCache();
        });
        
        // Help button
        document.getElementById('help-btn').addEventListener('click', () => {
            this.ui.showModal(document.getElementById('help-modal'));
        });
        
        // View knowledge graph button
        document.getElementById('view-knowledge-graph-btn').addEventListener('click', () => {
            this._showKnowledgeGraph();
        });
        
        // Upload button
        document.getElementById('upload-btn').addEventListener('click', () => {
            this.ui.showModal(document.getElementById('upload-modal'));
        });
        
        // Manage context button
        document.getElementById('manage-context-btn').addEventListener('click', () => {
            this.ui.showModal(document.getElementById('context-modal'));
        });
        
        // Context events
        document.addEventListener('context:add', (event) => {
            this.addToContext(event.detail.item);
        });
        
        document.addEventListener('context:add-all', (event) => {
            this.addAllToContext(event.detail.items);
        });
        
        document.addEventListener('context:remove', (event) => {
            this.removeFromContext(event.detail.index);
        });
        
        document.addEventListener('context:clear', () => {
            this.clearContext();
        });
        
        // Find solutions event
        document.addEventListener('solutions:find', (event) => {
            this.findSolutions(event.detail.ticket);
        });
        
        // File upload event
        document.addEventListener('file:upload', (event) => {
            this.analyzeFile(event.detail.file);
        });
    }
    
    /**
     * Set up input auto-resize
     */
    _setupInputAutoResize() {
        if (!this.chatInput) return;
        
        const resizeInput = () => {
            this.chatInput.style.height = 'auto';
            const newHeight = Math.min(Math.max(this.chatInput.scrollHeight, 40), 200);
            this.chatInput.style.height = newHeight + 'px';
        };
        
        this.chatInput.addEventListener('input', resizeInput);
        
        // Initial resize
        setTimeout(resizeInput, 0);
    }
    
    /**
     * Update buttons state based on input
     */
    _updateButtonsState() {
        if (!this.chatInput || !this.sendButton) return;
        
        const isEmpty = this.chatInput.value.trim() === '';
        
        // Send button
        this.sendButton.disabled = isEmpty;
        
        // Clear input button
        if (this.clearInputButton) {
            this.clearInputButton.style.display = isEmpty ? 'none' : 'flex';
        }
    }
    
    /**
     * Check server status
     */
    _checkServerStatus() {
        api.getStatus()
            .then(status => {
                this.ui.setConnectionStatus(status);
                
                // Check again after 30 seconds
                setTimeout(() => this._checkServerStatus(), 30000);
            })
            .catch(error => {
                console.error('Status check failed:', error);
                
                // Show error for all connections
                this.ui.setConnectionStatus({
                    connections: {
                        jira: false,
                        confluence: false
                    }
                });
                
                // Try again sooner
                setTimeout(() => this._checkServerStatus(), 5000);
            });
    }
    
    /**
     * Send message
     */
    sendMessage() {
        if (this.isProcessing) return;
        
        const message = this.chatInput.value.trim();
        if (!message) return;
        
        // Clear input
        this.chatInput.value = '';
        this._updateButtonsState();
        
        // Add user message
        this.addUserMessage(message);
        
        // Process message
        this.processMessage(message);
    }
    
    /**
     * Process message
     * @param {string} message - User message
     */
    async processMessage(message) {
        this.isProcessing = true;
        
        // Add thinking indicator
        this.addThinkingIndicator();
        
        try {
            // Check if it's a special command
            if (this._handleSpecialCommand(message)) {
                this.removeThinkingIndicator();
                this.isProcessing = false;
                return;
            }
            
            // Get selected source
            const source = this._getSelectedSource();
            
            // Check if we need to analyze query complexity
            const isComplex = this._isComplexQuery(message);
            
            // Add to recent searches
            this._addToRecentSearches(message);
            
            // If we have context items, use them
            if (this.contextItems.length > 0) {
                await this._processWithContext(message, source, isComplex);
            } else {
                // Otherwise, do a regular search
                await this._processRegularSearch(message, source, isComplex);
            }
        } catch (error) {
            console.error('Error processing message:', error);
            this.addBotMessage('Sorry, I encountered an error while processing your request. Please try again.');
        } finally {
            // Remove thinking indicator
            this.removeThinkingIndicator();
            this.isProcessing = false;
        }
    }
    
    /**
     * Handle special commands
     * @param {string} message - User message
     * @returns {boolean} - True if handled as special command
     */
    _handleSpecialCommand(message) {
        // Check for direct Jira ticket reference
        const ticketMatch = message.match(/^([A-Z]+-\d+)$/);
        if (ticketMatch) {
            const ticketKey = ticketMatch[1];
            this._fetchAndShowTicket(ticketKey);
            return true;
        }
        
        // Check for clear command
        if (/^\/?(clear|reset)$/i.test(message)) {
            this.clearChat();
            this.addSystemMessage('Chat cleared.');
            return true;
        }
        
        // Check for help command
        if (/^\/?(help|\?)$/i.test(message)) {
            this.ui.showModal(document.getElementById('help-modal'));
            return true;
        }
        
        // Not a special command
        return false;
    }
    
    /**
     * Get selected source
     * @returns {string} - Selected source (jira, confluence, combined)
     */
    _getSelectedSource() {
        for (const radio of this.sourceRadios) {
            if (radio.checked) {
                return radio.value;
            }
        }
        return 'combined'; // Default
    }
    
    /**
     * Determine if query is complex
     * @param {string} query - User query
     * @returns {boolean} - True if query is complex
     */
    _isComplexQuery(query) {
        // Check query length
        if (query.length > 100) return true;
        
        // Check for complex query indicators
        const complexIndicators = [
            'compare', 'relationship', 'relate', 'difference', 'between', 
            'analyze', 'explain', 'summarize', 'find all', 'show me all',
            'deep dive', 'in detail', 'comprehensively', 'thoroughly'
        ];
        
        for (const indicator of complexIndicators) {
            if (query.toLowerCase().includes(indicator)) {
                return true;
            }
        }
        
        // Count the number of question marks
        const questionMarkCount = (query.match(/\?/g) || []).length;
        if (questionMarkCount > 1) return true;
        
        return false;
    }
    
    /**
     * Process query with context
     * @param {string} message - User message
     * @param {string} source - Selected source
     * @param {boolean} isComplex - Whether query is complex
     */
    async _processWithContext(message, source, isComplex) {
        // Get context IDs
        const contextIds = this.contextItems.map(item => 
            `${item.sourceType}:${item.id}`
        );
        
        // Generate response with context
        try {
            const response = await api.chat(message, source, contextIds, isComplex);
            
            // Show reasoning if complex query
            if (isComplex && response.reasoning) {
                this.ui.showReasoningSteps(response.reasoning);
            }
            
            // Add response to chat
            this.addBotMessage(response.response || response);
        } catch (error) {
            console.error('Error generating response with context:', error);
            this.addBotMessage('Sorry, I had trouble processing your question with the current context. Please try again or modify your question.');
        }
    }
    
    /**
     * Process regular search query
     * @param {string} message - User message
     * @param {string} source - Selected source
     * @param {boolean} isComplex - Whether query is complex
     */
    async _processRegularSearch(message, source, isComplex) {
        try {
            // Search for relevant content
            const searchResults = await api.search(message, source);
            
            if (!searchResults.results || searchResults.results.length === 0) {
                // No results, generate a response without context
                const response = await api.chat(message, source, [], isComplex);
                
                // Show reasoning if complex query
                if (isComplex && response.reasoning) {
                    this.ui.showReasoningSteps(response.reasoning);
                }
                
                this.addBotMessage(response.response || response);
                return;
            }
            
            // Show search results
            this.ui.showSearchResults(searchResults.results);
            
            // Use the top result to generate an initial response
            const topResult = searchResults.results[0];
            const contextId = `${topResult.source_type}:${topResult.metadata.id}`;
            
            const response = await api.chat(message, source, [contextId], isComplex);
            
            // Show reasoning if complex query
            if (isComplex && response.reasoning) {
                this.ui.showReasoningSteps(response.reasoning);
            }
            
            // Add response to chat
            this.addBotMessage(response.response || response);
        } catch (error) {
            console.error('Error processing search query:', error);
            this.addBotMessage('Sorry, I had trouble searching for information related to your question. Please try again or rephrase your question.');
        }
    }
    
    /**
     * Fetch and show Jira ticket
     * @param {string} ticketKey - Jira ticket key
     */
    async _fetchAndShowTicket(ticketKey) {
        try {
            const response = await api.getJiraTicket(ticketKey);
            
            if (response && response.ticket) {
                // Add confirmation message
                this.addBotMessage(`Found Jira ticket <strong>${ticketKey}</strong>: ${response.ticket.metadata.summary}`);
                
                // Show ticket details
                this.ui.showItemDetails(response.ticket);
            } else {
                this.addBotMessage(`I couldn't find the Jira ticket ${ticketKey}. Please check if the ticket key is correct.`);
            }
        } catch (error) {
            console.error('Error fetching Jira ticket:', error);
            this.addBotMessage(`Sorry, I encountered an error while trying to retrieve the ticket ${ticketKey}. Please try again.`);
        }
    }
    
    /**
     * Find solutions for a Jira ticket
     * @param {Object} ticket - Jira ticket
     */
    async findSolutions(ticket) {
        if (!ticket || ticket.source_type !== 'jira') {
            showNotification('Invalid ticket data', 'error');
            return;
        }
        
        try {
            // Add message about searching
            const ticketKey = ticket.metadata.key;
            this.addBotMessage(`Searching for solutions to Jira ticket <strong>${ticketKey}</strong>...`);
            
            // Find solutions
            const response = await api.findSolution(ticketKey);
            
            if (!response) {
                this.addBotMessage(`I couldn't find any solutions for ticket ${ticketKey}.`);
                return;
            }
            
            // Show solutions in search results panel
            if (response.solutions && response.solutions.length > 0) {
                this.ui.showSearchResults(response.solutions);
                
                // Add response
                this.addBotMessage(response.response);
            } else {
                this.addBotMessage(response.response || `I found the ticket but couldn't find any specific solutions in Confluence.`);
            }
        } catch (error) {
            console.error('Error finding solutions:', error);
            this.addBotMessage('Sorry, I encountered an error while searching for solutions. Please try again.');
        }
    }
    
    /**
     * Analyze uploaded file
     * @param {File} file - Uploaded file
     */
    async analyzeFile(file) {
        if (!file) return;
        
        try {
            // Add message about analyzing file
            this.addSystemMessage(`Analyzing file: ${file.name} (${this._formatFileSize(file.size)})`);
            
            // Add thinking indicator
            this.addThinkingIndicator();
            
            // Analyze file
            const response = await api.analyzeFile(file);
            
            if (!response) {
                this.addBotMessage(`I couldn't analyze the file ${file.name}.`);
                return;
            }
            
            // Add response
            this.addBotMessage(response.content || `I've analyzed the file ${file.name}.`);
            
            // Show file content in search results if available
            if (response.extracted_content) {
                // Format as result
                const result = {
                    metadata: {
                        id: `file-${Date.now()}`,
                        title: file.name,
                        type: 'file'
                    },
                    content: response.extracted_content,
                    source_type: 'file'
                };
                
                this.ui.showSearchResults([result]);
            }
        } catch (error) {
            console.error('Error analyzing file:', error);
            this.addBotMessage(`Sorry, I encountered an error while analyzing the file ${file.name}. Please try again or try a different file.`);
        } finally {
            // Remove thinking indicator
            this.removeThinkingIndicator();
        }
    }
    
    /**
     * Format file size
     * @param {number} bytes - File size in bytes
     * @returns {string} - Formatted file size
     */
    _formatFileSize(bytes) {
        if (bytes < 1024) {
            return bytes + ' B';
        } else if (bytes < 1024 * 1024) {
            return (bytes / 1024).toFixed(1) + ' KB';
        } else {
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }
    }
    
    /**
     * Add to recent searches
     * @param {string} query - Search query
     */
    _addToRecentSearches(query) {
        // Don't add if it's the same as the most recent
        if (this.recentSearches.length > 0 && this.recentSearches[0].query === query) {
            return;
        }
        
        // Add to recent searches
        this.recentSearches.unshift({
            query,
            timestamp: new Date().toISOString()
        });
        
        // Limit to 5 recent searches
        this.recentSearches = this.recentSearches.slice(0, 5);
        
        // Update UI
        this._updateRecentSearchesUI();
    }
    
    /**
     * Update recent searches UI
     */
    _updateRecentSearchesUI() {
        const container = document.getElementById('recent-searches');
        if (!container) return;
        
        if (this.recentSearches.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fa-solid fa-clock-rotate-left"></i>
                    <p>No recent searches</p>
                </div>
            `;
            return;
        }
        
        let html = '';
        
        this.recentSearches.forEach(item => {
            const query = item.query;
            const truncatedQuery = query.length > 30 ? query.substring(0, 27) + '...' : query;
            
            html += `
                <div class="recent-search-item" data-query="${escapeHtml(query)}">
                    <i class="fa-solid fa-search"></i>
                    <span class="recent-search-text">${escapeHtml(truncatedQuery)}</span>
                </div>
            `;
        });
        
        container.innerHTML = html;
        
        // Add click handlers
        document.querySelectorAll('.recent-search-item').forEach(item => {
            item.addEventListener('click', () => {
                const query = item.getAttribute('data-query');
                if (query) {
                    this.chatInput.value = query;
                    this._updateButtonsState();
                    this.chatInput.focus();
                }
            });
        });
    }
    
    /**
     * Show knowledge graph
     */
    async _showKnowledgeGraph() {
        try {
            // Show modal with loading indicator
            const graphContainer = document.getElementById('knowledge-graph-container');
            if (graphContainer) {
                graphContainer.innerHTML = `
                    <div class="loading-indicator">
                        <i class="fa-solid fa-spinner fa-spin"></i>
                        <p>Loading knowledge graph...</p>
                    </div>
                `;
            }
            
            this.ui.showModal(document.getElementById('knowledge-graph-modal'));
            
            // Fetch graph data
            const graphData = await api.getKnowledgeGraph();
            
            // Format data for visualization
            const formattedData = KnowledgeGraph.formatGraphData(graphData);
            
            // Initialize or update graph
            if (!this.knowledgeGraph) {
                this.knowledgeGraph = new KnowledgeGraph('knowledge-graph-container', formattedData);
            } else {
                this.knowledgeGraph.setData(formattedData);
            }
        } catch (error) {
            console.error('Error loading knowledge graph:', error);
            
            // Show error in container
            const graphContainer = document.getElementById('knowledge-graph-container');
            if (graphContainer) {
                graphContainer.innerHTML = `
                    <div class="error-state">
                        <i class="fa-solid fa-exclamation-triangle"></i>
                        <p>Error loading knowledge graph: ${error.message}</p>
                    </div>
                `;
            }
        }
    }
    
    /**
     * Add user message to chat
     * @param {string} message - Message text
     */
    addUserMessage(message) {
        const html = `
            <div class="message user">
                <div class="message-content">
                    <p>${escapeHtml(message)}</p>
                </div>
            </div>
        `;
        
        this._addMessageToChat(html);
        
        // Add to history
        this.chatHistory.push({
            role: 'user',
            content: message
        });
    }
    
    /**
     * Add bot message to chat
     * @param {string} message - Message text or HTML
     */
    addBotMessage(message) {
        // Process message for markdown if marked library is available
        let processedMessage = message;
        
        if (typeof marked !== 'undefined') {
            try {
                processedMessage = marked.parse(message);
            } catch (error) {
                console.error('Error parsing markdown:', error);
                processedMessage = message;
            }
        } else {
            // Basic processing for code blocks and links
            processedMessage = message
                .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                .replace(/\n/g, '<br>');
        }
        
        const html = `
            <div class="message bot">
                <div class="message-content">
                    ${processedMessage}
                </div>
                <div class="message-actions">
                    <button class="btn-icon" title="View reasoning" data-action="view-reasoning">
                        <i class="fa-solid fa-brain"></i>
                    </button>
                    <button class="btn-icon" title="Copy to clipboard" data-action="copy">
                        <i class="fa-solid fa-copy"></i>
                    </button>
                </div>
            </div>
        `;
        
        this._addMessageToChat(html);
        
        // Add event listeners to action buttons
        const messageElement = this.chatMessages.lastElementChild;
        
        if (messageElement) {
            const copyButton = messageElement.querySelector('[data-action="copy"]');
            if (copyButton) {
                copyButton.addEventListener('click', () => {
                    // Get text content
                    const content = messageElement.querySelector('.message-content').textContent;
                    
                    // Copy to clipboard
                    navigator.clipboard.writeText(content).then(() => {
                        showNotification('Copied to clipboard', 'success');
                    }).catch(err => {
                        console.error('Could not copy text: ', err);
                    });
                });
            }
            
            const reasoningButton = messageElement.querySelector('[data-action="view-reasoning"]');
            if (reasoningButton) {
                reasoningButton.addEventListener('click', () => {
                    // Show reasoning modal
                    this.ui.showReasoningSteps({
                        analysis: "I'll explain my thought process for this response.",
                        information_gathering: "I gathered information from various sources to form my response.",
                        reasoning: "Based on the gathered information, here's how I arrived at my answer."
                    });
                });
            }
        }
        
        // Add to history
        this.chatHistory.push({
            role: 'bot',
            content: message
        });
        
        // Apply syntax highlighting if Prism is available
        if (typeof Prism !== 'undefined') {
            setTimeout(() => {
                Prism.highlightAllUnder(messageElement);
            }, 0);
        }
    }
    
    /**
     * Add system message to chat
     * @param {string} message - Message text
     */
    addSystemMessage(message) {
        const html = `
            <div class="message system">
                <div class="message-content">
                    <p>${escapeHtml(message)}</p>
                </div>
            </div>
        `;
        
        this._addMessageToChat(html);
    }
    
    /**
     * Add thinking indicator
     */
    addThinkingIndicator() {
        const html = `
            <div class="message thinking" id="thinking-indicator">
                <div class="message-content">
                    <div class="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        `;
        
        this._addMessageToChat(html);
        
        // Add typing indicator style if not already added
        if (!document.getElementById('typing-indicator-style')) {
            const style = document.createElement('style');
            style.id = 'typing-indicator-style';
            style.textContent = `
                .typing-indicator {
                    display: flex;
                    align-items: center;
                    gap: 4px;
                }
                
                .typing-indicator span {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background-color: var(--neutral);
                    display: inline-block;
                    animation: typing-bounce 1.4s infinite ease-in-out both;
                }
                
                .typing-indicator span:nth-child(1) {
                    animation-delay: -0.32s;
                }
                
                .typing-indicator span:nth-child(2) {
                    animation-delay: -0.16s;
                }
                
                @keyframes typing-bounce {
                    0%, 80%, 100% { transform: scale(0.6); }
                    40% { transform: scale(1); }
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    /**
     * Remove thinking indicator
     */
    removeThinkingIndicator() {
        const indicator = document.getElementById('thinking-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    /**
     * Add message to chat
     * @param {string} html - Message HTML
     */
    _addMessageToChat(html) {
        if (!this.chatMessages) return;
        
        // Create temporary div to hold the message
        const temp = document.createElement('div');
        temp.innerHTML = html;
        const messageElement = temp.firstChild;
        
        // Append to chat
        this.chatMessages.appendChild(messageElement);
        
        // Scroll to bottom
        this._scrollToBottom();
    }
    
    /**
     * Scroll chat to bottom
     */
    _scrollToBottom() {
        if (this.chatMessages) {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }
    }
    
    /**
     * Add item to context
     * @param {Object} item - Item to add to context
     */
    addToContext(item) {
        if (!item) return;
        
        // Check if item is already in context
        const existingIndex = this.contextItems.findIndex(
            contextItem => 
                contextItem.id === item.metadata.id && 
                contextItem.sourceType === item.source_type
        );
        
        if (existingIndex !== -1) {
            showNotification('This item is already in your context', 'info');
            return;
        }
        
        // Add to context
        this.contextItems.push({
            id: item.metadata.id,
            title: item.metadata.title || (item.source_type === 'jira' ? item.metadata.key : 'Untitled'),
            sourceType: item.source_type,
            content: truncateText(item.content, 100)
        });
        
        // Update UI
        this.ui.updateContextUI(this.contextItems);
        
        // Add system message
        this.addSystemMessage(`Added ${item.source_type === 'jira' ? 'Jira ticket' : 'Confluence page'} "${item.metadata.title || item.metadata.key || 'Untitled'}" to context.`);
    }
    
    /**
     * Add all items to context
     * @param {Array} items - Items to add to context
     */
    addAllToContext(items) {
        if (!items || !items.length) return;
        
        // Get existing IDs to avoid duplicates
        const existingIds = this.contextItems.map(item => item.id);
        
        // Add each item that doesn't already exist
        let addedCount = 0;
        
        items.forEach(item => {
            if (!item.metadata || existingIds.includes(item.metadata.id)) return;
            
            this.contextItems.push({
                id: item.metadata.id,
                title: item.metadata.title || (item.source_type === 'jira' ? item.metadata.key : 'Untitled'),
                sourceType: item.source_type,
                content: truncateText(item.content, 100)
            });
            
            addedCount++;
            
            // Add ID to existing IDs to avoid duplicates in this batch
            existingIds.push(item.metadata.id);
        });
        
        // Update UI
        this.ui.updateContextUI(this.contextItems);
        
        // Add system message if items were added
        if (addedCount > 0) {
            this.addSystemMessage(`Added ${addedCount} items to your context.`);
        }
    }
    
    /**
     * Remove item from context
     * @param {number} index - Index of item to remove
     */
    removeFromContext(index) {
        if (index < 0 || index >= this.contextItems.length) return;
        
        // Store item for message
        const removedItem = this.contextItems[index];
        
        // Remove from context
        this.contextItems.splice(index, 1);
        
        // Update UI
        this.ui.updateContextUI(this.contextItems);
        
        // Add system message
        this.addSystemMessage(`Removed ${removedItem.sourceType === 'jira' ? 'Jira ticket' : 'Confluence page'} "${removedItem.title}" from context.`);
    }
    
    /**
     * Clear context
     */
    clearContext() {
        if (this.contextItems.length === 0) return;
        
        // Clear context
        this.contextItems = [];
        
        // Update UI
        this.ui.updateContextUI(this.contextItems);
        
        // Add system message
        this.addSystemMessage('Cleared all items from context.');
    }
    
    /**
     * Clear chat
     */
    clearChat() {
        if (!this.chatMessages) return;
        
        // Clear chat messages
        this.chatMessages.innerHTML = '';
        
        // Clear chat history
        this.chatHistory = [];
        
        // Add welcome message
        this.chatMessages.innerHTML = `
            <div class="message system">
                <div class="message-content">
                    <h3><i class="fa-solid fa-robot"></i> Knowledge Assistant</h3>
                    <p>Hello! I'm your Jira-Confluence Knowledge Assistant. I can help you find information from your Jira tickets and Confluence pages, or even find solutions from Confluence for your Jira issues.</p>
                    <div class="suggestion-chips">
                        <div class="suggestion-chip" data-query="Find open bugs in the authentication system">Find open bugs in the authentication system</div>
                        <div class="suggestion-chip" data-query="Show me documentation about our API">Show me documentation about our API</div>
                        <div class="suggestion-chip" data-query="Help me troubleshoot SSO login failures">Help me troubleshoot SSO login failures</div>
                    </div>
                </div>
            </div>
        `;
        
        // Add event listeners to suggestion chips
        document.querySelectorAll('.suggestion-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const query = chip.getAttribute('data-query');
                if (query) {
                    this.chatInput.value = query;
                    this._updateButtonsState();
                    this.sendMessage();
                }
            });
        });
    }
    
    /**
     * Clear cache
     */
    async clearCache() {
        try {
            // Clear client-side cache
            api.clearCache();
            
            // Clear server-side cache
            await api.clearServerCache();
            
            // Show notification
            showNotification('Cache cleared successfully', 'success');
            
            // Add system message
            this.addSystemMessage('Cache cleared. Results will be fetched fresh from Jira and Confluence.');
        } catch (error) {
            console.error('Error clearing cache:', error);
            showNotification('Error clearing cache', 'error');
        }
    }
}



















/**
 * Main application entry point
 */
document.addEventListener('DOMContentLoaded', () => {
    // Initialize chat manager
    const chatManager = new ChatManager();
    
    // Initialize event listeners
    initializeEventListeners();
    
    // Enable keyboard shortcuts
    enableKeyboardShortcuts();
    
    // Initialize session storage for page persistence
    initializeSessionStorage();
    
    // Global access for debugging
    window.app = {
        chatManager,
        api
    };
});

/**
 * Initialize global event listeners
 */
function initializeEventListeners() {
    // Close panels with Escape key
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            // Close right panel if open
            const rightPanel = document.getElementById('right-panel');
            if (rightPanel && rightPanel.classList.contains('visible')) {
                rightPanel.classList.remove('visible');
                return;
            }
            
            // Close modals if any are open
            const visibleModals = document.querySelectorAll('.modal.visible');
            if (visibleModals.length > 0) {
                visibleModals.forEach(modal => {
                    modal.classList.remove('visible');
                });
                return;
            }
        }
    });
    
    // Window resize event for responsive adjustments
    window.addEventListener('resize', debounce(() => {
        // Adjust graph size if visible
        const graphContainer = document.getElementById('knowledge-graph-container');
        const graphModal = document.getElementById('knowledge-graph-modal');
        
        if (graphContainer && graphModal && graphModal.classList.contains('visible')) {
            const graph = window.app?.chatManager?.knowledgeGraph;
            if (graph) {
                graph.centerGraph();
            }
        }
    }, 250));
    
    // Handle beforeunload to save session data
    window.addEventListener('beforeunload', () => {
        saveSessionData();
    });
}

/**
 * Enable keyboard shortcuts
 */
function enableKeyboardShortcuts() {
    document.addEventListener('keydown', (event) => {
        // Help - ?
        if (event.key === '?' && !isInputFocused()) {
            const helpModal = document.getElementById('help-modal');
            if (helpModal) {
                helpModal.classList.add('visible');
            }
        }
        
        // Focus chat input - /
        if (event.key === '/' && !isInputFocused()) {
            event.preventDefault();
            const chatInput = document.getElementById('chat-input');
            if (chatInput) {
                chatInput.focus();
            }
        }
        
        // Clear chat - Ctrl+Shift+C
        if (event.ctrlKey && event.shiftKey && event.key === 'C') {
            event.preventDefault();
            window.app?.chatManager?.clearChat();
        }
    });
}

/**
 * Check if any input is focused
 * @returns {boolean} - True if an input is focused
 */
function isInputFocused() {
    const activeElement = document.activeElement;
    return activeElement.tagName === 'INPUT' || 
           activeElement.tagName === 'TEXTAREA' || 
           activeElement.isContentEditable;
}

/**
 * Initialize session storage for page persistence
 */
function initializeSessionStorage() {
    try {
        // Load chat history
        const savedHistory = sessionStorage.getItem('chatHistory');
        if (savedHistory) {
            const history = JSON.parse(savedHistory);
            restoreChatHistory(history);
        }
        
        // Load context items
        const savedContext = sessionStorage.getItem('contextItems');
        if (savedContext) {
            const contextItems = JSON.parse(savedContext);
            restoreContextItems(contextItems);
        }
    } catch (error) {
        console.error('Error initializing from session storage:', error);
    }
}

/**
 * Restore chat history from saved data
 * @param {Array} history - Saved chat history
 */
function restoreChatHistory(history) {
    if (!history || !Array.isArray(history) || history.length === 0) return;
    
    // Clear current chat
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        chatMessages.innerHTML = '';
    }
    
    // Add messages in order
    for (const message of history) {
        if (message.role === 'user') {
            window.app?.chatManager?.addUserMessage(message.content);
        } else if (message.role === 'bot') {
            window.app?.chatManager?.addBotMessage(message.content);
        } else if (message.role === 'system') {
            window.app?.chatManager?.addSystemMessage(message.content);
        }
    }
}

/**
 * Restore context items from saved data
 * @param {Array} contextItems - Saved context items
 */
function restoreContextItems(contextItems) {
    if (!contextItems || !Array.isArray(contextItems) || contextItems.length === 0) return;
    
    // Set context items
    if (window.app?.chatManager) {
        window.app.chatManager.contextItems = contextItems;
        window.app.chatManager.ui.updateContextUI(contextItems);
    }
}

/**
 * Save current session data
 */
function saveSessionData() {
    try {
        // Save chat history
        if (window.app?.chatManager?.chatHistory) {
            sessionStorage.setItem('chatHistory', 
                JSON.stringify(window.app.chatManager.chatHistory));
        }
        
        // Save context items
        if (window.app?.chatManager?.contextItems) {
            sessionStorage.setItem('contextItems', 
                JSON.stringify(window.app.chatManager.contextItems));
        }
    } catch (error) {
        console.error('Error saving session data:', error);
    }
}

/**
 * Debounce function to limit frequent calls
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} - Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}



















# Knowledge Graph API Endpoints

Now let's add the backend endpoints for the knowledge graph functionality that we implemented in the frontend. Add these endpoints to your `app.py` file:

```python
# Import necessary modules
import networkx as nx
from pathlib import Path
import pickle
import os
import json

# Knowledge graph endpoints
@app.route("/api/knowledge-graph", methods=["GET"])
def get_knowledge_graph():
    """Get the knowledge graph data."""
    try:
        # Get filter parameters
        max_nodes = request.args.get("max_nodes", 100, type=int)
        min_similarity = request.args.get("min_similarity", 0.2, type=float)
        
        # Build knowledge graph from cached data
        graph_data = build_knowledge_graph(max_nodes, min_similarity)
        
        return jsonify(graph_data)
    except Exception as e:
        logger.error(f"Error getting knowledge graph: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/relationships", methods=["GET"])
def get_relationships():
    """Get relationships between Jira and Confluence data."""
    try:
        # Get the integration manager
        integration_manager = get_integration_manager()
        
        # Get relationships
        relationships = integration_manager.get_references_between_jira_confluence()
        
        return jsonify(relationships)
    except Exception as e:
        logger.error(f"Error getting relationships: {str(e)}")
        return jsonify({"error": str(e)}), 500

# File analysis endpoint
@app.route("/api/analyze-file", methods=["POST"])
def analyze_file():
    """Analyze uploaded file content."""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        
        # Check if file has a name
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Check file size (max 10MB)
        if file.content_length and file.content_length > 10 * 1024 * 1024:
            return jsonify({"error": "File too large, maximum size is 10MB"}), 400
            
        # Save file to temporary location
        temp_dir = Path(CACHE_DIR) / "temp_files"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        file_path = temp_dir / file.filename
        file.save(file_path)
        
        # Extract file content based on type
        content, metadata = extract_file_content(file_path)
        
        # Generate response using Gemini
        gemini_client = GeminiClient()
        
        analysis_prompt = f"""
        Analyze this file content and provide a summary. The file is named "{file.filename}" 
        and has the following content:

        {content[:5000]}  # Limiting content size for prompt

        Provide a concise summary of what this file contains, its key information, and any insights.
        If relevant, mention how this relates to Jira tickets or Confluence documentation.
        """
        
        response = gemini_client.generate_response(
            prompt=analysis_prompt,
            temperature=0.3,
            stream=False
        )
        
        # Clean up temporary file
        os.remove(file_path)
        
        return jsonify({
            "filename": file.filename,
            "content": response,
            "extracted_content": content,
            "metadata": metadata
        })
    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Helper functions

def build_knowledge_graph(max_nodes=100, min_similarity=0.2):
    """
    Build a knowledge graph from Jira and Confluence data.
    
    Args:
        max_nodes: Maximum number of nodes in the graph
        min_similarity: Minimum similarity for relationships
        
    Returns:
        Graph data in format suitable for visualization
    """
    try:
        # Get integration manager
        integration_manager = get_integration_manager()
        
        # Extract knowledge graph from integration manager
        kg = integration_manager.knowledge_graph
        
        # Convert to NetworkX graph for processing
        G = nx.Graph()
        
        # Add nodes from Jira
        jira_nodes = []
        for jira_id, node_data in kg.get("nodes", {}).get("jira", {}).items():
            if len(jira_nodes) >= max_nodes // 2:
                break
                
            G.add_node(jira_id, type="jira", **node_data)
            jira_nodes.append({
                "id": jira_id,
                "type": "jira",
                "title": node_data.get("title", ""),
                "url": node_data.get("url", ""),
                "key": node_data.get("title", ""),  # Jira ticket key is in title
                "keywords": node_data.get("keywords", [])
            })
            
        # Add nodes from Confluence
        confluence_nodes = []
        for conf_id, node_data in kg.get("nodes", {}).get("confluence", {}).items():
            if len(confluence_nodes) >= max_nodes // 2:
                break
                
            G.add_node(conf_id, type="confluence", **node_data)
            confluence_nodes.append({
                "id": conf_id,
                "type": "confluence",
                "title": node_data.get("title", ""),
                "url": node_data.get("url", ""),
                "space": node_data.get("space", ""),
                "keywords": node_data.get("keywords", [])
            })
            
        # Add relationships
        relationships = []
        for rel in kg.get("relationships", []):
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            similarity = rel.get("similarity", 0)
            
            if similarity >= min_similarity and source_id in G and target_id in G:
                G.add_edge(source_id, target_id, weight=similarity)
                relationships.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_type": rel.get("source_type"),
                    "target_type": rel.get("target_type"),
                    "type": rel.get("relationship", "related"),
                    "weight": similarity
                })
                
        # If no relationships exist in the knowledge graph, add some based on keyword similarity
        if not relationships:
            # Calculate similarity between Jira and Confluence nodes
            for jira_node in jira_nodes:
                jira_keywords = set(jira_node.get("keywords", []))
                
                if not jira_keywords:
                    continue
                    
                for conf_node in confluence_nodes:
                    conf_keywords = set(conf_node.get("keywords", []))
                    
                    if not conf_keywords:
                        continue
                        
                    # Calculate overlap
                    overlap = jira_keywords.intersection(conf_keywords)
                    similarity = len(overlap) / max(len(jira_keywords), len(conf_keywords))
                    
                    if similarity >= min_similarity:
                        G.add_edge(jira_node["id"], conf_node["id"], weight=similarity)
                        relationships.append({
                            "source_id": jira_node["id"],
                            "target_id": conf_node["id"],
                            "source_type": "jira",
                            "target_type": "confluence",
                            "type": "related",
                            "weight": similarity
                        })
        
        # Return the graph data
        return {
            "jira_tickets": jira_nodes,
            "confluence_pages": confluence_nodes,
            "relationships": relationships
        }
    except Exception as e:
        logger.error(f"Error building knowledge graph: {str(e)}")
        return {
            "jira_tickets": [],
            "confluence_pages": [],
            "relationships": []
        }

def get_integration_manager():
    """Get or create an integration manager instance."""
    # Check if integration manager exists in app context
    if hasattr(app, 'integration_manager'):
        return app.integration_manager
        
    # Create a new integration manager
    cache_dir = Path(CACHE_DIR)
    jira_client = JiraClient(cache=Cache(str(cache_dir / "jira")))
    confluence_client = ConfluenceClient(cache=Cache(str(cache_dir / "confluence")))
    gemini_client = GeminiClient()
    
    app.integration_manager = IntegrationManager(
        jira_client=jira_client,
        confluence_client=confluence_client,
        gemini_client=gemini_client,
        cache_dir=str(cache_dir / "integration")
    )
    
    return app.integration_manager

def extract_file_content(file_path):
    """
    Extract text content from a file based on its type.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (extracted_content, metadata)
    """
    file_path = Path(file_path)
    file_type = file_path.suffix.lower()
    
    # Default content and metadata
    content = ""
    metadata = {
        "filename": file_path.name,
        "size": file_path.stat().st_size,
        "type": file_type
    }
    
    # Text file
    if file_type in ['.txt', '.md', '.csv', '.json', '.xml', '.html', '.js', '.py', '.java']:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading text file: {str(e)}")
            
    # PDF file
    elif file_type == '.pdf':
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n\n"
                    
                metadata["pages"] = len(pdf_reader.pages)
        except ImportError:
            content = "PDF extraction requires PyPDF2 library"
        except Exception as e:
            logger.error(f"Error extracting PDF content: {str(e)}")
            
    # Office documents
    elif file_type in ['.docx', '.xlsx', '.pptx']:
        try:
            if file_type == '.docx':
                import docx
                doc = docx.Document(file_path)
                content = "\n\n".join([para.text for para in doc.paragraphs])
                
            elif file_type == '.xlsx':
                import pandas as pd
                df = pd.read_excel(file_path)
                content = df.to_string()
                
            elif file_type == '.pptx':
                import pptx
                presentation = pptx.Presentation(file_path)
                content = ""
                for slide in presentation.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, 'text'):
                            content += shape.text + "\n\n"
        except ImportError:
            content = f"Extraction of {file_type} files requires additional libraries"
        except Exception as e:
            logger.error(f"Error extracting Office document content: {str(e)}")
            
    # Image files - we don't extract content but note that it's an image
    elif file_type in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        content = f"[Image file: {file_path.name}]"
        metadata["is_image"] = True
        
    # Unknown file type
    else:
        content = f"Unknown file type: {file_type}"
        
    return content, metadata
```

Now, you'll need to add NetworkX to your requirements.txt:

```
# Add to requirements.txt
networkx==3.1
```

To handle different file types, you'll also need to install these optional dependencies:

```
# Optional dependencies for file extraction
PyPDF2==3.0.1
python-docx==0.8.11
pandas==2.0.3
openpyxl==3.1.2
python-pptx==0.6.21
```


















