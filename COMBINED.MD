#!/usr/bin/env python3
"""
Jira-Gemini Chatbot

A comprehensive chatbot that integrates Jira with Google's Gemini AI to answer questions
about Jira tickets, Confluence pages, and related data.

Features:
- Uses Atlassian Search API for powerful direct querying
- Local caching of Jira/Confluence data for performance
- SSL verification disabled by default
- Context-aware responses using Gemini AI
- Understanding of images and tables in Jira tickets
- Ability to provide references to source pages
- Interactive follow-up questions for clarification
"""

import os
import sys
import json
import time
import logging
import pickle
import base64
import re
import html
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

# Third-party imports
import urllib3
import requests
from requests.auth import HTTPBasicAuth
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel, Part, Content
from PIL import Image
import io

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

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
class Config:
    # Jira Configuration
    JIRA_URL = os.environ.get("JIRA_URL", "https://your-jira-instance.atlassian.net")
    JIRA_EMAIL = os.environ.get("JIRA_EMAIL", "your-email@example.com")
    JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "your-api-token")
    
    # Confluence Configuration
    CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL", JIRA_URL)  # Default to same as JIRA
    
    # Cache Settings
    CACHE_ENABLED = True
    CACHE_DIR = os.environ.get("CACHE_DIR", "cache")
    CACHE_TTL = int(os.environ.get("CACHE_TTL", 86400))  # 24 hours in seconds
    
    # Google Vertex AI / Gemini Configuration
    PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
    REGION = os.environ.get("REGION", "us-central1")
    MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-pro-vision")
    
    # SSL Verification
    SSL_VERIFY = False  # Set to False as requested
    
    # Response Settings
    MAX_RESULTS = 100  # Max number of results to fetch from Jira
    
    # Chat settings
    MAX_CONVERSATION_HISTORY = 10  # Number of previous messages to maintain for context

# Cache Manager for efficient data storage
class CacheManager:
    def __init__(self, cache_dir=Config.CACHE_DIR, ttl=Config.CACHE_TTL):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Cache initialized at {self.cache_dir}")
    
    def _get_cache_path(self, key):
        # Create a filename-safe key
        safe_key = base64.urlsafe_b64encode(key.encode()).decode()
        return self.cache_dir / f"{safe_key}.cache"
    
    def get(self, key):
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if cache has expired
            if datetime.now().timestamp() > cache_data.get('expiry', 0):
                logger.debug(f"Cache expired for key: {key}")
                return None
            
            logger.debug(f"Cache hit for key: {key}")
            return cache_data.get('data')
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
            return None
    
    def set(self, key, data):
        cache_path = self._get_cache_path(key)
        
        try:
            expiry = datetime.now().timestamp() + self.ttl
            cache_data = {
                'data': data,
                'expiry': expiry
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.debug(f"Cached data for key: {key}")
            return True
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")
            return False
    
    def clear(self):
        """Clear all cached data"""
        count = 0
        for cache_file in self.cache_dir.glob('*.cache'):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Error removing cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {count} cache items")
        return count

# Atlassian API Client
class AtlassianClient:
    def __init__(self, jira_url=Config.JIRA_URL, confluence_url=Config.CONFLUENCE_URL, 
                 email=Config.JIRA_EMAIL, api_token=Config.JIRA_API_TOKEN, 
                 verify_ssl=Config.SSL_VERIFY):
        self.jira_url = jira_url
        self.confluence_url = confluence_url
        self.auth = HTTPBasicAuth(email, api_token)
        self.verify_ssl = verify_ssl
        self.cache_manager = CacheManager()
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        logger.info(f"Atlassian client initialized for Jira: {jira_url}, Confluence: {confluence_url}")
    
    def _make_request(self, method, url, data=None, params=None, use_cache=True, 
                      headers=None, files=None):
        """Make a request to the Atlassian API with caching support"""
        if headers is None:
            headers = self.headers.copy()
        
        # Create cache key (only for cacheable requests)
        cacheable = method.upper() == 'GET' and use_cache and Config.CACHE_ENABLED and not files
        cache_key = None
        
        if cacheable:
            cache_key = f"{method}:{url}:{json.dumps(params if params else {})}:{json.dumps(data if data else {})}"
            cached_response = self.cache_manager.get(cache_key)
            if cached_response:
                return cached_response
        
        try:
            kwargs = {
                'headers': headers,
                'auth': self.auth,
                'verify': self.verify_ssl
            }
            
            if params:
                kwargs['params'] = params
            
            if data and method.upper() != 'GET':
                kwargs['json'] = data
            
            if files:
                kwargs['files'] = files
                # Remove content-type header when sending files
                if 'Content-Type' in kwargs['headers']:
                    del kwargs['headers']['Content-Type']
            
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            
            # Parse response
            if response.text and response.headers.get('Content-Type', '').startswith('application/json'):
                result = response.json()
            else:
                result = {'status_code': response.status_code, 'text': response.text}
            
            # Cache the response if applicable
            if cacheable and cache_key:
                self.cache_manager.set(cache_key, result)
            
            return result
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise
    
    # Jira API Methods
    def get_all_projects(self):
        """Retrieve all accessible Jira projects"""
        return self._make_request('GET', f"{self.jira_url}/rest/api/3/project")
    
    def search_issues(self, jql, max_results=Config.MAX_RESULTS, fields=None, expand=None, 
                      start_at=0):
        """Search for Jira issues using JQL"""
        params = {
            'jql': jql,
            'maxResults': max_results,
            'startAt': start_at
        }
        
        if fields:
            if isinstance(fields, list):
                params['fields'] = ','.join(fields)
            else:
                params['fields'] = fields
        
        if expand:
            if isinstance(expand, list):
                params['expand'] = ','.join(expand)
            else:
                params['expand'] = expand
        
        return self._make_request('GET', f"{self.jira_url}/rest/api/3/search", params=params)
    
    def get_issue(self, issue_key, fields=None, expand=None):
        """Get a specific Jira issue by key with option to expand fields"""
        params = {}
        
        if fields:
            if isinstance(fields, list):
                params['fields'] = ','.join(fields)
            else:
                params['fields'] = fields
        
        if expand:
            if isinstance(expand, list):
                params['expand'] = ','.join(expand)
            else:
                params['expand'] = expand
        
        return self._make_request(
            'GET', 
            f"{self.jira_url}/rest/api/3/issue/{issue_key}", 
            params=params
        )
    
    def get_issue_attachments(self, issue_key):
        """Get all attachments for an issue"""
        issue_data = self.get_issue(issue_key, fields='attachment')
        return issue_data.get('fields', {}).get('attachment', [])
    
    def download_attachment(self, attachment_url, use_cache=True):
        """Download an attachment from Jira"""
        # Create a cache key based on the URL
        cache_key = f"attachment:{attachment_url}"
        
        if use_cache and Config.CACHE_ENABLED:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
        
        try:
            response = requests.get(
                attachment_url, 
                auth=self.auth, 
                headers={"Accept": "*/*"}, 
                verify=self.verify_ssl,
                stream=True
            )
            response.raise_for_status()
            
            attachment_data = response.content
            
            # Cache the attachment
            if Config.CACHE_ENABLED:
                self.cache_manager.set(cache_key, attachment_data)
            
            return attachment_data
        except Exception as e:
            logger.error(f"Error downloading attachment: {e}")
            raise
    
    # Confluence API Methods
    def get_content(self, content_id, expand=None):
        """Get Confluence content by ID"""
        params = {}
        
        if expand:
            if isinstance(expand, list):
                params['expand'] = ','.join(expand)
            else:
                params['expand'] = expand
        
        return self._make_request(
            'GET', 
            f"{self.confluence_url}/rest/api/content/{content_id}", 
            params=params
        )
    
    def search_content(self, cql, max_results=Config.MAX_RESULTS, start=0, expand=None):
        """Search for Confluence content using CQL"""
        params = {
            'cql': cql,
            'limit': max_results,
            'start': start
        }
        
        if expand:
            if isinstance(expand, list):
                params['expand'] = ','.join(expand)
            else:
                params['expand'] = expand
        
        return self._make_request(
            'GET', 
            f"{self.confluence_url}/rest/api/content/search", 
            params=params
        )
    
    # Direct Atlassian Search API - Unified search across Jira and Confluence
    def search_atlassian(self, query, product="jira,confluence", 
                        max_results=Config.MAX_RESULTS, start=0):
        """Search across Jira and Confluence using the Atlassian Search API"""
        params = {
            'query': query,
            'product': product,
            'limit': max_results,
            'start': start
        }
        
        return self._make_request(
            'GET',
            f"{self.jira_url}/rest/api/3/search/generic",
            params=params
        )
    
    # Additional search method specifically for finding content with context
    def search_with_context(self, query, product="jira,confluence", max_results=20):
        """
        Perform a comprehensive search and extract useful context for AI processing
        Returns list of context entries with metadata and links
        """
        context_entries = []
        
        # First try direct Atlassian search API 
        try:
            search_results = self.search_atlassian(
                query=query, 
                product=product, 
                max_results=max_results
            )
            
            # Process results into context entries
            for item in search_results.get('results', []):
                content_type = item.get('contentType', '')
                entry = {
                    'title': item.get('title', 'Untitled'),
                    'type': content_type,
                    'url': item.get('url', ''),
                    'container': item.get('container', {}).get('title', ''),
                    'excerpt': item.get('excerpt', '')
                }
                
                # Add ID based on content type
                if content_type == 'jira.issue':
                    entry['id'] = item.get('key', '')
                    # Get full issue if it's a Jira ticket
                    try:
                        issue_details = self.get_issue(
                            entry['id'], 
                            fields='summary,description,status,created,updated,issuetype,priority,assignee,reporter,comment'
                        )
                        fields = issue_details.get('fields', {})
                        entry['summary'] = fields.get('summary', '')
                        entry['description'] = self._extract_text_from_adf(fields.get('description', {}))
                        entry['status'] = fields.get('status', {}).get('name', '')
                        entry['issue_type'] = fields.get('issuetype', {}).get('name', '')
                        
                        # Get comments
                        comments = []
                        for comment in fields.get('comment', {}).get('comments', [])[:5]:  # Limit to 5 comments
                            comments.append({
                                'author': comment.get('author', {}).get('displayName', ''),
                                'text': self._extract_text_from_adf(comment.get('body', {})),
                                'created': comment.get('created', '')
                            })
                        entry['comments'] = comments
                        
                        # Check for attachments
                        attachments = self.get_issue_attachments(entry['id'])
                        if attachments:
                            entry['has_attachments'] = True
                            entry['attachment_count'] = len(attachments)
                            entry['attachments'] = [
                                {
                                    'filename': att.get('filename', ''),
                                    'content_type': att.get('mimeType', ''),
                                    'url': att.get('content', '')
                                }
                                for att in attachments[:5]  # Limit to 5 attachments
                            ]
                    except Exception as e:
                        logger.warning(f"Error getting detailed info for issue {entry['id']}: {e}")
                
                elif content_type == 'confluence.page':
                    entry['id'] = item.get('id', '')
                    # Get full page details if it's a Confluence page
                    try:
                        page_details = self.get_content(
                            entry['id'], 
                            expand='body.storage,version'
                        )
                        entry['version'] = page_details.get('version', {}).get('number', '1')
                        body_content = page_details.get('body', {}).get('storage', {}).get('value', '')
                        # Convert HTML to plain text (simplified)
                        entry['content'] = self._clean_html(body_content)
                    except Exception as e:
                        logger.warning(f"Error getting detailed info for page {entry['id']}: {e}")
                
                context_entries.append(entry)
        
        except Exception as e:
            logger.error(f"Error during Atlassian search: {e}")
            
            # Fallback to separate searches if unified search fails
            try:
                # Search Jira
                if 'jira' in product:
                    jql = f'text ~ "{query}" ORDER BY updated DESC'
                    jira_results = self.search_issues(jql, max_results=max_results//2)
                    
                    for issue in jira_results.get('issues', []):
                        fields = issue.get('fields', {})
                        entry = {
                            'id': issue.get('key', ''),
                            'title': fields.get('summary', 'Untitled Issue'),
                            'type': 'jira.issue',
                            'url': f"{self.jira_url}/browse/{issue.get('key', '')}",
                            'summary': fields.get('summary', ''),
                            'description': self._extract_text_from_adf(fields.get('description', {})),
                            'status': fields.get('status', {}).get('name', '') if fields.get('status') else '',
                            'issue_type': fields.get('issuetype', {}).get('name', '') if fields.get('issuetype') else ''
                        }
                        context_entries.append(entry)
                
                # Search Confluence
                if 'confluence' in product:
                    cql = f'type=page AND text ~ "{query}" ORDER BY lastmodified DESC'
                    confluence_results = self.search_content(cql, max_results=max_results//2)
                    
                    for page in confluence_results.get('results', []):
                        entry = {
                            'id': page.get('id', ''),
                            'title': page.get('title', 'Untitled Page'),
                            'type': 'confluence.page',
                            'url': page.get('_links', {}).get('webui', ''),
                            'space': page.get('space', {}).get('name', '') if page.get('space') else '',
                            'excerpt': page.get('excerpt', '')
                        }
                        context_entries.append(entry)
            
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
        
        return context_entries
    
    def _extract_text_from_adf(self, adf_data):
        """Extract plain text from Atlassian Document Format (ADF) JSON"""
        if not adf_data:
            return ""
        
        # Check if it's already a string
        if isinstance(adf_data, str):
            return adf_data
        
        text_parts = []
        
        # Process content based on ADF structure
        def extract_text(node):
            if isinstance(node, dict):
                # Text node with direct text content
                if node.get('type') == 'text':
                    return node.get('text', '')
                
                # Process special nodes
                if node.get('type') in ['heading', 'paragraph', 'bulletList', 'orderedList']:
                    content = []
                    for child in node.get('content', []):
                        content.append(extract_text(child))
                    return '\n'.join(filter(None, content))
                
                # List item
                if node.get('type') == 'listItem':
                    content = []
                    for child in node.get('content', []):
                        content.append(extract_text(child))
                    return '- ' + '\n'.join(filter(None, content))
                
                # Table handling
                if node.get('type') == 'table':
                    table_content = []
                    for row in node.get('content', []):
                        row_content = []
                        for cell in row.get('content', []):
                            row_content.append(extract_text(cell))
                        table_content.append(' | '.join(row_content))
                    return '\n'.join(table_content)
                
                # Code block
                if node.get('type') == 'codeBlock':
                    code_content = []
                    for child in node.get('content', []):
                        code_content.append(extract_text(child))
                    return '```\n' + '\n'.join(code_content) + '\n```'
                
                # Process content recursively for other node types
                result = []
                for child in node.get('content', []):
                    result.append(extract_text(child))
                return ' '.join(filter(None, result))
            
            elif isinstance(node, list):
                result = []
                for item in node:
                    result.append(extract_text(item))
                return '\n'.join(filter(None, result))
            
            return ""
        
        # Extract text from the document
        try:
            if 'content' in adf_data:
                for item in adf_data.get('content', []):
                    text = extract_text(item)
                    if text:
                        text_parts.append(text)
            else:
                # Try to process as a single node
                text = extract_text(adf_data)
                if text:
                    text_parts.append(text)
        except Exception as e:
            logger.error(f"Error extracting text from ADF: {e}")
            return str(adf_data)
        
        return '\n\n'.join(text_parts)
    
    def _clean_html(self, html_content):
        """Basic HTML to plain text conversion"""
        if not html_content:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

# Gemini AI Integration
class GeminiAI:
    def __init__(self, project_id=Config.PROJECT_ID, location=Config.REGION, 
                 model_name=Config.MODEL_NAME):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        # Create model instance
        self.model = GenerativeModel(model_name)
        
        logger.info(f"Gemini AI initialized with model: {model_name}")
    
    def generate_response(self, query, context=None, images=None, chat_history=None, 
                          temperature=0.7, max_output_tokens=2048):
        """
        Generate an AI response using Gemini
        
        Args:
            query: User's question or request
            context: Context data for better responses (e.g., Jira ticket info)
            images: List of image data (bytes) for multimodal queries
            chat_history: Previous conversation history for context
            temperature: Creativity level (0.0-1.0)
            max_output_tokens: Maximum token length for response
            
        Returns:
            Generated response text
        """
        try:
            # Create generation config
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=max_output_tokens,
                candidate_count=1
            )
            
            # Prepare context for the model
            system_instruction = """
            You are an AI assistant specialized in answering questions about Jira tickets 
            and Confluence pages. Use the provided context to give specific, accurate, and helpful 
            answers. If you don't know the answer, say so rather than making something up.
            
            When giving information from Jira tickets or Confluence pages, provide the relevant 
            links and references. If images or tables were provided, include the information 
            from those in your response.
            
            Your responses should be professional but friendly. For technical questions, 
            provide detailed technical answers. For general questions, be helpful and concise.
            
            If the user's question is not related to the provided Jira or Confluence 
            information, answer to the best of your ability or politely explain that the 
            question is outside the scope of the available information.
            """
            
            # Format context for better reasoning
            formatted_context = ""
            if context:
                formatted_context = "--- CONTEXT INFORMATION ---\n\n"
                
                # Add general context summary
                formatted_context += f"Found {len(context)} relevant items:\n"
                
                # Process each context item
                for i, item in enumerate(context, 1):
                    item_type = item.get('type', 'unknown')
                    formatted_context += f"\n{i}. {item_type.upper()}: {item.get('title', 'Untitled')}\n"
                    
                    if item_type == 'jira.issue':
                        formatted_context += f"   ID: {item.get('id', '')}\n"
                        formatted_context += f"   Status: {item.get('status', '')}\n"
                        formatted_context += f"   Type: {item.get('issue_type', '')}\n"
                        formatted_context += f"   URL: {item.get('url', '')}\n\n"
                        
                        formatted_context += "   Summary: " + item.get('summary', '') + "\n\n"
                        
                        if 'description' in item and item['description']:
                            formatted_context += "   Description:\n"
                            formatted_context += "   " + item['description'].replace('\n', '\n   ') + "\n\n"
                        
                        if 'comments' in item and item['comments']:
                            formatted_context += "   Recent Comments:\n"
                            for comment in item['comments']:
                                formatted_context += f"   - {comment.get('author', '')}: {comment.get('text', '')}\n"
                            formatted_context += "\n"
                        
                        if 'has_attachments' in item and item['has_attachments']:
                            formatted_context += f"   This issue has {item.get('attachment_count', 0)} attachments.\n"
                            if 'attachments' in item:
                                formatted_context += "   Attachments:\n"
                                for att in item['attachments']:
                                    formatted_context += f"   - {att.get('filename', '')}\n"
                            formatted_context += "\n"
                    
                    elif item_type == 'confluence.page':
                        formatted_context += f"   ID: {item.get('id', '')}\n"
                        formatted_context += f"   Space: {item.get('space', '')}\n"
                        formatted_context += f"   URL: {item.get('url', '')}\n\n"
                        
                        if 'content' in item and item['content']:
                            # Truncate very long content
                            content = item['content']
                            if len(content) > 5000:
                                content = content[:5000] + "... [content truncated]"
                            
                            formatted_context += "   Content:\n"
                            formatted_context += "   " + content.replace('\n', '\n   ') + "\n\n"
                        elif 'excerpt' in item and item['excerpt']:
                            formatted_context += "   Excerpt: " + item['excerpt'] + "\n\n"
                
                formatted_context += "--- END CONTEXT INFORMATION ---\n\n"
            
            # Prepare content parts for the model
            content_parts = []
            
            # Add system instruction
            content_parts.append(Part.from_text(system_instruction))
            
            # Add context information
            if formatted_context:
                content_parts.append(Part.from_text(formatted_context))
            
            # Add chat history for context
            if chat_history and len(chat_history) > 0:
                history_text = "--- PREVIOUS CONVERSATION ---\n\n"
                for entry in chat_history:
                    if entry['role'] == 'user':
                        history_text += f"User: {entry['content']}\n\n"
                    else:
                        history_text += f"Assistant: {entry['content']}\n\n"
                history_text += "--- END PREVIOUS CONVERSATION ---\n\n"
                content_parts.append(Part.from_text(history_text))
            
            # Add images if any
            if images and len(images) > 0:
                for img_data in images:
                    try:
                        # Convert bytes to image and add to content
                        image_part = Part.from_image(Image.open(io.BytesIO(img_data)))
                        content_parts.append(image_part)
                    except Exception as img_error:
                        logger.error(f"Error processing image: {img_error}")
            
            # Add user query
            content_parts.append(Part.from_text(f"User Question: {query}"))
            
            # Generate the response
            logger.info(f"Generating response for query: {query}")
            response = self.model.generate_content(
                content_parts,
                generation_config=generation_config,
                stream=False
            )
            
            # Extract and return the text response
            response_text = response.text
            logger.info(f"Generated response of length: {len(response_text)}")
            
            return response_text
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error while generating a response: {str(e)}"

# Main Chatbot Class
class JiraGeminiChatbot:
    def __init__(self):
        self.atlassian_client = AtlassianClient()
        self.gemini_ai = GeminiAI()
        self.conversation_history = []
        logger.info("JiraGeminiChatbot initialized")
    
    def ask(self, query, include_attachments=True):
        """
        Process a user query and generate a response
        
        Args:
            query: User's question
            include_attachments: Whether to process attachments in Jira tickets
            
        Returns:
            AI-generated response
        """
        try:
            logger.info(f"Processing query: {query}")
            start_time = time.time()
            
            # Check if query is about clearing cache
            if query.lower() in ["clear cache", "clear the cache", "refresh cache"]:
                count = self.atlassian_client.cache_manager.clear()
                return f"Cache cleared. Removed {count} cached items."
            
            # Search for relevant context
            logger.info("Searching for relevant context...")
            context_entries = self.atlassian_client.search_with_context(query)
            logger.info(f"Found {len(context_entries)} relevant context entries")
            
            # Extract attachments if needed
            images_data = []
            if include_attachments:
                for entry in context_entries:
                    if entry.get('type') == 'jira.issue' and entry.get('has_attachments', False):
                        for attachment in entry.get('attachments', []):
                            if self._is_image_attachment(attachment.get('filename', '')):
                                try:
                                    img_data = self.atlassian_client.download_attachment(attachment.get('url', ''))
                                    if img_data:
                                        images_data.append(img_data)
                                        logger.info(f"Downloaded attachment: {attachment.get('filename', '')}")
                                except Exception as e:
                                    logger.warning(f"Error downloading attachment: {e}")
            
            # Keep conversation history limited
            if len(self.conversation_history) > Config.MAX_CONVERSATION_HISTORY:
                self.conversation_history = self.conversation_history[-Config.MAX_CONVERSATION_HISTORY:]
            
            # Generate response using Gemini
            logger.info("Generating AI response...")
            response = self.gemini_ai.generate_response(
                query=query,
                context=context_entries,
                images=images_data[:5],  # Limit to 5 images to avoid token limits
                chat_history=self.conversation_history
            )
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            end_time = time.time()
            logger.info(f"Request completed in {end_time - start_time:.2f} seconds")
            
            return response
        
        except Exception as e:
            logger.error(f"Error in ask method: {e}")
            return f"I'm sorry, but an error occurred while processing your question: {str(e)}"
    
    def _is_image_attachment(self, filename):
        """Check if a file is an image based on its extension"""
        if not filename:
            return False
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        _, ext = os.path.splitext(filename.lower())
        return ext in image_extensions

def main():
    """
    Main function to run the chatbot interactively
    """
    print("="*50)
    print("Jira-Gemini Chatbot")
    print("="*50)
    print("Type your questions about Jira tickets and Confluence pages.")
    print("Type 'quit', 'exit', or 'bye' to end the session.")
    print("Type 'clear cache' to refresh the cache.")
    print("-"*50)
    
    # Initialize the chatbot
    chatbot = JiraGeminiChatbot()
    
    while True:
        try:
            # Get user input
            query = input("\nYou: ").strip()
            
            # Check for exit commands
            if query.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break
            
            # Skip empty queries
            if not query:
                continue
            
            # Process the query
            print("\nThinking...")
            response = chatbot.ask(query)
            
            # Display the response
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\nSession terminated by user.")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
