#!/usr/bin/env python3
import os
import json
import logging
import re
import time
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Optional, Any
from functools import lru_cache
import threading

# Web framework
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# HTTP requests
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup

# AI/ML
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Enhanced logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UnifiedAssistant")

# Configuration with better defaults
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")

# Confluence Configuration
CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL", "https://cmegroup.atlassian.net")
CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME", "lakshya.vijay@cmegroup.com")
CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN", "")
CONFLUENCE_SPACE = os.environ.get("CONFLUENCE_SPACE", "RE")

# Jira Configuration
JIRA_URL = os.environ.get("JIRA_URL", "https://cmegroup.atlassian.net/")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL", "lakshya.vijay@cmegroup.com")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "")

# Performance settings
MAX_WORKERS = 3
CACHE_SIZE = 100
REQUEST_TIMEOUT = 15

class ContentExtractor:
    """Extract and process content from Confluence HTML."""
    
    def __init__(self, base_url, page_id=None, page_title=None):
        self.base_url = base_url
        self.page_id = page_id
        self.page_title = page_title
    
    def extract_content_from_html(self, html_content):
        """Extract text, tables from HTML content - simplified for speed."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract all text content quickly
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Basic cleanup
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            cleaned_text = '\n'.join(lines)
            
            return {
                "metadata": {
                    "id": self.page_id,
                    "title": self.page_title,
                    "url": f"{self.base_url}/pages/viewpage.action?pageId={self.page_id}" if self.page_id else None
                },
                "content": cleaned_text[:5000]  # Limit content size
            }
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return {
                "metadata": {"title": self.page_title or "Unknown"},
                "content": ""
            }

class ConfluenceClient:
    """Simplified Confluence client."""
    
    def __init__(self, base_url, username, api_token):
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "CME-Knowledge-Assistant/1.0"
        })
        self.cache = {}
        self.space_key = CONFLUENCE_SPACE
        logger.info(f"Confluence client initialized for {self.base_url}")
    
    def test_connection(self):
        """Test connection to Confluence."""
        try:
            logger.info("Testing Confluence connection...")
            response = self.session.get(
                f"{self.base_url}/rest/api/content",
                params={"limit": 1},
                timeout=REQUEST_TIMEOUT,
                verify=False
            )
            connected = response.status_code == 200
            logger.info(f"Confluence connection: {'SUCCESS' if connected else 'FAILED'}")
            return connected
        except Exception as e:
            logger.error(f"Confluence connection test failed: {e}")
            return False
    
    def search_content(self, query, limit=5):
        """Search Confluence content with fallback methods."""
        try:
            logger.info(f"Searching Confluence for: '{query}'")
            
            # Method 1: Try CQL search
            try:
                search_params = {
                    "cql": f'space = "{self.space_key}" AND text ~ "{query}"',
                    "limit": limit,
                    "expand": "body.storage"
                }
                
                response = self.session.get(
                    f"{self.base_url}/rest/api/content/search",
                    params=search_params,
                    timeout=REQUEST_TIMEOUT,
                    verify=False
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    if results:
                        logger.info(f"CQL search found {len(results)} results")
                        return self._process_search_results(results)
                else:
                    logger.warning(f"CQL search failed: {response.status_code}")
            except Exception as e:
                logger.warning(f"CQL search failed: {e}")
            
            # Method 2: Try simple content API with space filter
            try:
                logger.info("Falling back to content API search...")
                search_params = {
                    "spaceKey": self.space_key,
                    "expand": "body.storage",
                    "limit": 50  # Get more to filter locally
                }
                
                response = self.session.get(
                    f"{self.base_url}/rest/api/content",
                    params=search_params,
                    timeout=REQUEST_TIMEOUT,
                    verify=False
                )
                
                if response.status_code == 200:
                    data = response.json()
                    all_results = data.get("results", [])
                    
                    # Filter results locally
                    query_words = query.lower().split()
                    filtered_results = []
                    
                    for result in all_results:
                        title = result.get("title", "").lower()
                        if any(word in title for word in query_words):
                            filtered_results.append(result)
                            if len(filtered_results) >= limit:
                                break
                    
                    if filtered_results:
                        logger.info(f"Content API found {len(filtered_results)} matching results")
                        return self._process_search_results(filtered_results)
                    else:
                        logger.info("No matching results found, returning recent pages")
                        return self._process_search_results(all_results[:limit])
                        
            except Exception as e:
                logger.error(f"Content API search failed: {e}")
            
            return []
            
        except Exception as e:
            logger.error(f"Confluence search error: {e}")
            return []
    
    def _process_search_results(self, results):
        """Process search results into extracted content."""
        extracted_results = []
        
        for result in results:
            try:
                page_id = result.get("id")
                title = result.get("title", "Unknown")
                
                # Get content
                body = result.get("body", {})
                storage = body.get("storage", {})
                html_content = storage.get("value", "")
                
                if not html_content:
                    # Try to fetch content separately
                    try:
                        content_response = self.session.get(
                            f"{self.base_url}/rest/api/content/{page_id}",
                            params={"expand": "body.storage"},
                            timeout=REQUEST_TIMEOUT,
                            verify=False
                        )
                        if content_response.status_code == 200:
                            content_data = content_response.json()
                            html_content = content_data.get("body", {}).get("storage", {}).get("value", "")
                    except:
                        pass
                
                if html_content:
                    extractor = ContentExtractor(self.base_url, page_id, title)
                    extracted = extractor.extract_content_from_html(html_content)
                    if extracted["content"]:
                        extracted_results.append(extracted)
                        
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                continue
        
        logger.info(f"Successfully processed {len(extracted_results)} Confluence pages")
        return extracted_results

class JiraClient:
    """Simplified Jira client."""
    
    def __init__(self):
        self.base_url = JIRA_URL.rstrip('/')
        self.auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        logger.info(f"Jira client initialized for {self.base_url}")
    
    def test_connection(self):
        """Test connection to Jira."""
        try:
            logger.info("Testing Jira connection...")
            response = requests.get(
                f"{self.base_url}/rest/api/3/myself",
                headers=self.headers,
                auth=self.auth,
                verify=False,
                timeout=REQUEST_TIMEOUT
            )
            connected = response.status_code == 200
            logger.info(f"Jira connection: {'SUCCESS' if connected else 'FAILED'}")
            return connected
        except Exception as e:
            logger.error(f"Jira connection test failed: {e}")
            return False
    
    def search_issues(self, query, max_results=10):
        """Search Jira issues with simplified JQL."""
        try:
            logger.info(f"Searching Jira for: '{query}'")
            
            # Build simple JQL queries
            jql_queries = self._build_jql_queries(query)
            
            all_issues = []
            for jql in jql_queries:
                logger.info(f"Executing JQL: {jql}")
                
                params = {
                    "jql": jql,
                    "maxResults": max_results,
                    "fields": "summary,status,priority,assignee,created,updated,description"
                }
                
                response = requests.get(
                    f"{self.base_url}/rest/api/3/search",
                    headers=self.headers,
                    params=params,
                    auth=self.auth,
                    verify=False,
                    timeout=REQUEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    data = response.json()
                    issues = data.get("issues", [])
                    logger.info(f"JQL returned {len(issues)} issues")
                    all_issues.extend(issues)
                    
                    if len(all_issues) >= max_results:
                        break
                else:
                    logger.warning(f"JQL query failed: {response.status_code}")
            
            # Remove duplicates
            unique_issues = []
            seen_keys = set()
            for issue in all_issues:
                key = issue.get("key")
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    unique_issues.append(issue)
            
            logger.info(f"Found {len(unique_issues)} unique Jira issues")
            return unique_issues[:max_results]
            
        except Exception as e:
            logger.error(f"Jira search error: {e}")
            return []
    
    def _build_jql_queries(self, query):
        """Build simple JQL queries based on the user query."""
        words = [w.strip() for w in query.lower().split() if len(w) > 2]
        
        jql_queries = []
        
        if words:
            # Text search (most comprehensive)
            main_terms = ' '.join(words[:3])
            jql_queries.append(f'text ~ "{main_terms}" ORDER BY updated DESC')
            
            # Summary search
            jql_queries.append(f'summary ~ "{words[0]}" ORDER BY updated DESC')
            
            # Description search
            jql_queries.append(f'description ~ "{words[0]}" ORDER BY updated DESC')
        
        # Fallback: recent issues
        jql_queries.append('updated >= -30d ORDER BY updated DESC')
        
        return jql_queries

class AIAssistant:
    """Simplified AI assistant."""
    
    def __init__(self):
        try:
            vertexai.init(project=PROJECT_ID, location=REGION)
            self.model = GenerativeModel(MODEL_NAME)
            logger.info(f"AI Assistant initialized with {MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize AI: {e}")
            self.model = None
    
    def generate_response(self, user_query, confluence_data, jira_data, selected_sources):
        """Generate unified response."""
        if not self.model:
            return "AI service is not available. Please check the configuration."
        
        try:
            logger.info("Generating AI response...")
            
            # Format data for prompt
            confluence_text = self._format_confluence_data(confluence_data)
            jira_text = self._format_jira_data(jira_data)
            
            prompt = f"""You are a helpful CME knowledge assistant. Answer the user's question based on the available data.

USER QUESTION: "{user_query}"
SELECTED SOURCES: {selected_sources}

CONFLUENCE DATA:
{confluence_text}

JIRA DATA:
{jira_text}

Provide a clear, helpful response that:
1. Directly answers the question
2. Uses information from the available sources
3. Is well-formatted with bullet points and sections
4. Includes relevant ticket links: [TICKET-KEY](https://cmegroup.atlassian.net/browse/TICKET-KEY)
5. Is professional but conversational

Response:"""

            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2000
                )
            )
            
            result = response.text if hasattr(response, 'text') else response.candidates[0].text
            logger.info("AI response generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            return f"I found some information but had trouble formatting the response: {str(e)}"
    
    def _format_confluence_data(self, data):
        """Format Confluence data for AI prompt."""
        if not data:
            return "No Confluence data available."
        
        formatted = "CONFLUENCE DOCUMENTATION:\n"
        for i, item in enumerate(data[:3], 1):  # Limit to 3 items
            title = item.get("metadata", {}).get("title", "Unknown")
            content = item.get("content", "")[:1000]  # Limit content
            formatted += f"{i}. {title}\n{content}\n\n"
        
        return formatted
    
    def _format_jira_data(self, data):
        """Format Jira data for AI prompt."""
        if not data:
            return "No Jira data available."
        
        formatted = f"JIRA ISSUES ({len(data)} found):\n"
        for i, issue in enumerate(data[:5], 1):  # Limit to 5 issues
            key = issue.get("key", "Unknown")
            fields = issue.get("fields", {})
            summary = fields.get("summary", "No summary")
            status = (fields.get("status") or {}).get("name", "Unknown")
            
            formatted += f"{i}. [{key}] {summary} (Status: {status})\n"
        
        return formatted

class UnifiedKnowledgeAssistant:
    """Main unified assistant."""
    
    def __init__(self):
        logger.info("Initializing Unified Knowledge Assistant...")
        
        self.confluence = None
        self.jira = None
        self.ai = AIAssistant()
        
        # Initialize clients
        self._initialize_clients()
        
        logger.info("Unified Knowledge Assistant ready!")
    
    def _initialize_clients(self):
        """Initialize Confluence and Jira clients."""
        # Initialize Confluence
        if CONFLUENCE_API_TOKEN:
            try:
                self.confluence = ConfluenceClient(CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN)
            except Exception as e:
                logger.error(f"Failed to initialize Confluence: {e}")
        else:
            logger.warning("Confluence API token not provided")
        
        # Initialize Jira
        if JIRA_API_TOKEN:
            try:
                self.jira = JiraClient()
            except Exception as e:
                logger.error(f"Failed to initialize Jira: {e}")
        else:
            logger.warning("Jira API token not provided")
    
    def process_query(self, user_query, selected_sources):
        """Process user query and return response."""
        start_time = time.time()
        logger.info(f"=" * 50)
        logger.info(f"PROCESSING QUERY: '{user_query}'")
        logger.info(f"SELECTED SOURCES: {selected_sources}")
        logger.info(f"=" * 50)
        
        try:
            confluence_data = []
            jira_data = []
            
            # Search Confluence if selected and available
            if ("confluence" in selected_sources or "both" in selected_sources) and self.confluence:
                logger.info("🔍 Searching Confluence...")
                confluence_data = self.confluence.search_content(user_query, limit=3)
                logger.info(f"✅ Confluence search completed: {len(confluence_data)} results")
            else:
                logger.info("⏭️  Skipping Confluence (not selected or unavailable)")
            
            # Search Jira if selected and available
            if ("jira" in selected_sources or "both" in selected_sources) and self.jira:
                logger.info("🎫 Searching Jira...")
                jira_data = self.jira.search_issues(user_query, max_results=5)
                logger.info(f"✅ Jira search completed: {len(jira_data)} results")
            else:
                logger.info("⏭️  Skipping Jira (not selected or unavailable)")
            
            # Generate response
            logger.info("🤖 Generating AI response...")
            response = self.ai.generate_response(user_query, confluence_data, jira_data, selected_sources)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Query completed in {elapsed:.2f} seconds")
            logger.info(f"📝 Response length: {len(response)} characters")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}", exc_info=True)
            return f"I encountered an error while processing your query: {str(e)}. Please try again or contact support."
    
    def get_status(self):
        """Get system status with page count."""
        logger.info("Checking system status...")
        
        confluence_status = False
        jira_status = False
        confluence_pages = 0
        
        if self.confluence:
            confluence_status = self.confluence.test_connection()
            if confluence_status:
                try:
                    # Try to get page count
                    response = requests.get(
                        f"{self.confluence.base_url}/rest/api/content",
                        params={"spaceKey": CONFLUENCE_SPACE, "limit": 1},
                        auth=self.confluence.auth,
                        timeout=5,
                        verify=False
                    )
                    if response.status_code == 200:
                        data = response.json()
                        confluence_pages = data.get("size", 0)
                except:
                    pass
        
        if self.jira:
            jira_status = self.jira.test_connection()
        
        status = {
            'confluence': confluence_status,
            'jira': jira_status,
            'confluence_pages': confluence_pages,
            'status': 'healthy' if confluence_status or jira_status else 'degraded'
        }
        
        logger.info(f"System status: {status}")
        return status

# Flask Application
app = Flask(__name__)
CORS(app)

# Initialize the assistant
logger.info("🚀 Starting CME Unified Knowledge Assistant...")
assistant = UnifiedKnowledgeAssistant()

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests with detailed logging."""
    request_start = time.time()
    logger.info("📨 Received chat request")
    
    try:
        data = request.get_json()
        if not data:
            logger.warning("❌ No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
        
        user_query = data.get('message', '').strip()
        selected_sources = data.get('sources', [])
        
        logger.info(f"📝 User Query: '{user_query}'")
        logger.info(f"📊 Sources: {selected_sources}")
        
        if not user_query:
            logger.warning("❌ Empty message")
            return jsonify({'error': 'No message provided'}), 400
        
        if not selected_sources:
            logger.warning("❌ No sources selected")
            return jsonify({'error': 'No sources selected'}), 400
        
        # Handle basic greetings quickly
        query_lower = user_query.lower()
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            response = """👋 **Hello! I'm your CME Unified Knowledge Assistant!**

🔍 **I can help you with:**
• **Confluence:** Documentation, procedures, how-to guides
• **Jira:** Tickets, issues, bugs, project tracking  
• **Both:** Combined insights across systems

💡 **Just select your data sources and ask naturally!**

🚀 **Try asking:**
• "Show me rollout restart tickets and how to solve them"
• "BAMPS project documentation"
• "Recent high priority issues"
• "How to handle system maintenance"

What would you like to explore? 🤔"""
            
            logger.info("✅ Responded with greeting")
            return jsonify({'response': response})
        
        if any(word in query_lower for word in ['bye', 'goodbye', 'thanks']):
            response = "👋 **Goodbye!** Feel free to come back anytime for insights across CME's knowledge systems! 🚀"
            logger.info("✅ Responded with goodbye")
            return jsonify({'response': response})
        
        # Process the actual query
        logger.info("🔄 Processing query with assistant...")
        response = assistant.process_query(user_query, selected_sources)
        
        request_time = time.time() - request_start
        logger.info(f"✅ Chat request completed in {request_time:.2f} seconds")
        
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/status')
def status():
    """Check system status."""
    logger.info("📊 Status check requested")
    try:
        status_data = assistant.get_status()
        return jsonify(status_data)
    except Exception as e:
        logger.error(f"❌ Error in status endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting CME Unified Knowledge Assistant...")
    print(f"🔗 Confluence: {CONFLUENCE_URL}")
    print(f"🎫 Jira: {JIRA_URL}")
    print(f"🤖 AI Model: {MODEL_NAME}")
    print(f"📊 Confluence Token: {'✅ SET' if CONFLUENCE_API_TOKEN else '❌ MISSING'}")
    print(f"📊 Jira Token: {'✅ SET' if JIRA_API_TOKEN else '❌ MISSING'}")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)





























/* Reset and Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    /* CME Blue Gradient Colors */
    --primary-blue: #1e3a8a;
    --secondary-blue: #3b82f6;
    --light-blue: #60a5fa;
    --accent-blue: #93c5fd;
    --soft-blue: #dbeafe;
    --white: #ffffff;
    --off-white: #f8fafc;
    --gray-100: #f1f5f9;
    --gray-200: #e2e8f0;
    --gray-300: #cbd5e1;
    --gray-400: #94a3b8;
    --gray-500: #64748b;
    --gray-600: #475569;
    --gray-700: #334155;
    --gray-800: #1e293b;
    --gray-900: #0f172a;
    
    /* Gradients */
    --primary-gradient: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
    --secondary-gradient: linear-gradient(135deg, #60a5fa 0%, #93c5fd 50%, #dbeafe 100%);
    --background-gradient: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 25%, #dbeafe 50%, #ffffff 100%);
    --header-gradient: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    --card-gradient: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.9) 100%);
    
    /* Shadows */
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    --shadow-blue: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    
    /* Transitions */
    --transition-fast: 0.15s ease-out;
    --transition-normal: 0.3s ease-out;
    --transition-slow: 0.5s ease-out;
    
    /* Fonts */
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

body {
    font-family: var(--font-family);
    background: var(--background-gradient);
    min-height: 100vh;
    overflow-x: hidden;
    color: var(--gray-800);
    line-height: 1.6;
}

/* Background Pattern */
.background-pattern {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -2;
    opacity: 0.03;
}

.cme-logo-pattern {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(circle at 20% 80%, var(--primary-blue) 2px, transparent 2px),
        radial-gradient(circle at 80% 20%, var(--secondary-blue) 2px, transparent 2px),
        radial-gradient(circle at 40% 40%, var(--light-blue) 1px, transparent 1px);
    background-size: 100px 100px, 80px 80px, 60px 60px;
    background-repeat: repeat;
}

.gradient-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: var(--background-gradient);
    z-index: -1;
}

/* Status Indicator */
.status-indicator {
    position: fixed;
    top: 20px;
    right: 20px;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    padding: 8px 16px;
    border-radius: 25px;
    box-shadow: var(--shadow-md);
    display: flex;
    align-items: center;
    gap: 8px;
    z-index: 1000;
    font-size: 14px;
    font-weight: 500;
    transition: var(--transition-normal);
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--gray-400);
    transition: var(--transition-normal);
}

.status-indicator.connected .status-dot {
    background: #10b981;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
}

.status-indicator.degraded .status-dot {
    background: #f59e0b;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.2);
}

.status-indicator.error .status-dot {
    background: #ef4444;
    box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2);
}

/* Main Container */
.main-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Header */
.header {
    background: var(--header-gradient);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 30px;
    color: white;
    box-shadow: var(--shadow-blue);
    position: relative;
    overflow: hidden;
}

.header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
    opacity: 0.5;
}

.header-content {
    position: relative;
    z-index: 2;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 20px;
}

.logo-section {
    display: flex;
    align-items: center;
    gap: 20px;
}

.cme-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 24px;
    font-weight: 700;
    background: rgba(255, 255, 255, 0.2);
    padding: 15px 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

.cme-logo i {
    font-size: 28px;
}

.title-section h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 5px;
    background: linear-gradient(45deg, #ffffff, #dbeafe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
    font-weight: 400;
}

/* Source Selector */
.source-selector {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.selector-label {
    font-size: 14px;
    font-weight: 600;
    opacity: 0.9;
}

.source-options {
    display: flex;
    gap: 15px;
}

.source-option {
    position: relative;
}

.source-option input[type="checkbox"] {
    position: absolute;
    opacity: 0;
    pointer-events: none;
}

.source-option label {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 18px;
    background: rgba(255, 255, 255, 0.2);
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 12px;
    cursor: pointer;
    transition: var(--transition-normal);
    font-weight: 500;
    user-select: none;
    backdrop-filter: blur(10px);
}

.source-option input[type="checkbox"]:checked + label {
    background: rgba(255, 255, 255, 0.3);
    border-color: rgba(255, 255, 255, 0.8);
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.source-option label:hover {
    background: rgba(255, 255, 255, 0.25);
    transform: translateY(-1px);
}

/* Chat Container */
.chat-container {
    flex: 1;
    background: var(--card-gradient);
    border-radius: 20px;
    box-shadow: var(--shadow-xl);
    margin-bottom: 20px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.chat-header {
    background: rgba(255, 255, 255, 0.8);
    padding: 20px 30px;
    border-bottom: 1px solid var(--gray-200);
    display: flex;
    justify-content: space-between;
    align-items: center;
    backdrop-filter: blur(10px);
}

.chat-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--gray-800);
}

.chat-title i {
    color: var(--secondary-blue);
    font-size: 1.3rem;
}

.clear-chat {
    background: none;
    border: 2px solid var(--gray-300);
    color: var(--gray-600);
    padding: 8px 16px;
    border-radius: 10px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 500;
    transition: var(--transition-normal);
}

.clear-chat:hover {
    border-color: var(--secondary-blue);
    color: var(--secondary-blue);
    transform: translateY(-1px);
}

/* Welcome Section */
.welcome-section {
    padding: 60px 30px;
    text-align: center;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(147, 197, 253, 0.05) 100%);
}

.welcome-content {
    max-width: 600px;
    margin: 0 auto;
}

.welcome-icon {
    width: 80px;
    height: 80px;
    background: var(--primary-gradient);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 30px;
    box-shadow: var(--shadow-blue);
}

.welcome-icon i {
    font-size: 36px;
    color: white;
}

.welcome-content h2 {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--gray-800);
    margin-bottom: 15px;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.welcome-content p {
    font-size: 1.1rem;
    color: var(--gray-600);
    margin-bottom: 40px;
}

.example-queries h3 {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--gray-800);
    margin-bottom: 20px;
}

.query-examples {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 15px;
    margin-top: 20px;
}

.example-query {
    background: rgba(255, 255, 255, 0.7);
    border: 2px solid var(--gray-200);
    border-radius: 15px;
    padding: 20px;
    cursor: pointer;
    transition: var(--transition-normal);
    display: flex;
    align-items: center;
    gap: 12px;
    text-align: left;
    font-weight: 500;
    color: var(--gray-700);
    backdrop-filter: blur(10px);
}

.example-query:hover {
    border-color: var(--secondary-blue);
    background: rgba(255, 255, 255, 0.9);
    transform: translateY(-3px);
    box-shadow: var(--shadow-lg);
}

.example-query i {
    color: var(--secondary-blue);
    font-size: 18px;
    flex-shrink: 0;
}

/* Chat Messages */
.chat-messages {
    flex: 1;
    padding: 20px 30px;
    overflow-y: auto;
    max-height: 500px;
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.message {
    display: flex;
    gap: 15px;
    margin-bottom: 20px;
    opacity: 1;
    transform: translateY(0);
    transition: opacity 0.5s ease, transform 0.5s ease;
}

.message.user {
    flex-direction: row-reverse;
}

.message-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 18px;
    color: white;
    font-weight: 600;
}

.message.user .message-avatar {
    background: var(--primary-gradient);
}

.message.assistant .message-avatar {
    background: linear-gradient(135deg, #10b981, #34d399);
}

.message-content {
    flex: 1;
    max-width: 70%;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

.message.user .message-content {
    text-align: right;
}

.message-bubble {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid var(--gray-200);
    border-radius: 18px;
    padding: 16px 20px;
    backdrop-filter: blur(10px);
    box-shadow: var(--shadow-sm);
    word-wrap: break-word;
    overflow-wrap: break-word;
    line-height: 1.6;
}

.message.user .message-bubble {
    background: var(--secondary-gradient);
    color: white;
    border-color: transparent;
}

.message-time {
    font-size: 12px;
    color: var(--gray-500);
    margin-top: 5px;
}

.message.user .message-time {
    color: rgba(255, 255, 255, 0.8);
}

/* Typing Indicator */
.typing-indicator {
    display: none;
    align-items: center;
    gap: 15px;
    padding: 15px 30px;
    background: rgba(59, 130, 246, 0.05);
    border-top: 1px solid var(--gray-200);
}

.typing-indicator.show {
    display: flex;
}

.typing-dots {
    display: flex;
    gap: 4px;
}

.typing-dots span {
    width: 8px;
    height: 8px;
    background: var(--secondary-blue);
    border-radius: 50%;
    animation: typingBounce 1.4s infinite ease-in-out;
}

.typing-dots span:nth-child(1) { animation-delay: -0.32s; }
.typing-dots span:nth-child(2) { animation-delay: -0.16s; }

.typing-text {
    color: var(--gray-600);
    font-style: italic;
}

/* Input Section */
.input-section {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    box-shadow: var(--shadow-xl);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.input-container {
    padding: 25px 30px;
}

.input-wrapper {
    display: flex;
    align-items: flex-end;
    gap: 15px;
    background: rgba(255, 255, 255, 0.8);
    border: 2px solid var(--gray-200);
    border-radius: 15px;
    padding: 15px 20px;
    transition: var(--transition-normal);
    backdrop-filter: blur(10px);
}

.input-wrapper:focus-within {
    border-color: var(--secondary-blue);
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

#messageInput {
    flex: 1;
    border: none;
    outline: none;
    background: none;
    font-size: 16px;
    font-family: var(--font-family);
    color: var(--gray-800);
    resize: none;
    max-height: 120px;
    min-height: 24px;
    line-height: 1.5;
}

#messageInput::placeholder {
    color: var(--gray-500);
}

.input-actions {
    display: flex;
    align-items: center;
    gap: 10px;
}

.character-count {
    font-size: 12px;
    color: var(--gray-500);
    font-weight: 500;
}

.send-button {
    width: 44px;
    height: 44px;
    background: var(--primary-gradient);
    border: none;
    border-radius: 12px;
    color: white;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    transition: var(--transition-normal);
    box-shadow: var(--shadow-md);
}

.send-button:disabled {
    background: var(--gray-300);
    cursor: not-allowed;
    box-shadow: none;
}

.send-button:not(:disabled):hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-blue);
}

.input-hints {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 15px;
    font-size: 14px;
    color: var(--gray-600);
}

.hint-item, .selected-sources {
    display: flex;
    align-items: center;
    gap: 8px;
}

.hint-item i, .selected-sources i {
    color: var(--secondary-blue);
}

#sourcesList {
    font-weight: 600;
    color: var(--secondary-blue);
}

/* Loading Overlay */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 9999;
}

.loading-overlay.show {
    display: flex;
}

.loading-content {
    text-align: center;
    background: rgba(255, 255, 255, 0.9);
    padding: 40px;
    border-radius: 20px;
    box-shadow: var(--shadow-xl);
    backdrop-filter: blur(20px);
}

.loading-spinner {
    width: 60px;
    height: 60px;
    border: 4px solid var(--gray-200);
    border-top: 4px solid var(--secondary-blue);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 20px;
}

.loading-text h3 {
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--gray-800);
    margin-bottom: 10px;
}

.loading-text p {
    color: var(--gray-600);
}

/* Toast Notifications */
.toast-container {
    position: fixed;
    top: 80px;
    right: 20px;
    z-index: 10000;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.toast {
    background: rgba(255, 255, 255, 0.95);
    border-left: 4px solid var(--secondary-blue);
    border-radius: 10px;
    padding: 15px 20px;
    box-shadow: var(--shadow-lg);
    backdrop-filter: blur(10px);
    transform: translateX(400px);
    animation: slideInRight 0.3s ease-out forwards;
    max-width: 350px;
}

.toast.error {
    border-left-color: #ef4444;
}

.toast.success {
    border-left-color: #10b981;
}

.toast.warning {
    border-left-color: #f59e0b;
}

/* Floating Help */
.floating-help {
    position: fixed;
    bottom: 30px;
    left: 30px;
    width: 50px;
    height: 50px;
    background: var(--primary-gradient);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    cursor: pointer;
    box-shadow: var(--shadow-lg);
    transition: var(--transition-normal);
    z-index: 1000;
}

.floating-help:hover {
    transform: scale(1.1);
    box-shadow: var(--shadow-blue);
}

.help-tooltip {
    position: absolute;
    bottom: 60px;
    left: 0;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    box-shadow: var(--shadow-xl);
    backdrop-filter: blur(20px);
    border: 1px solid var(--gray-200);
    width: 300px;
    opacity: 0;
    visibility: hidden;
    transform: translateY(10px);
    transition: var(--transition-normal);
}

.floating-help:hover .help-tooltip {
    opacity: 1;
    visibility: visible;
    transform: translateY(0);
}

.help-content h4 {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--gray-800);
    margin-bottom: 15px;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.help-content ul {
    list-style: none;
}

.help-content li {
    padding: 8px 0;
    color: var(--gray-700);
    font-size: 14px;
    line-height: 1.5;
}

.help-content li strong {
    color: var(--secondary-blue);
    font-weight: 600;
}

/* Particles Background */
.particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: -1;
}

.particle {
    position: absolute;
    background: var(--secondary-blue);
    border-radius: 50%;
    opacity: 0.1;
    animation: float 6s ease-in-out infinite;
}

/* Message Content Styling */
.message-content h1, .message-content h2, .message-content h3 {
    color: var(--gray-800);
    margin: 15px 0 10px 0;
    font-weight: 600;
}

.message-content h1 { font-size: 1.5rem; }
.message-content h2 { font-size: 1.3rem; }
.message-content h3 { font-size: 1.1rem; }

.message-content p {
    margin: 10px 0;
    line-height: 1.6;
}

.message-content ul, .message-content ol {
    margin: 10px 0;
    padding-left: 20px;
}

.message-content li {
    margin: 5px 0;
    line-height: 1.5;
}

.message-content code {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 6px;
    padding: 2px 6px;
    font-family: 'Courier New', monospace;
    font-size: 0.9em;
    color: var(--primary-blue);
}

.message-content pre {
    background: var(--gray-100);
    border: 1px solid var(--gray-200);
    border-radius: 10px;
    padding: 15px;
    overflow-x: auto;
    margin: 15px 0;
}

.message-content pre code {
    background: none;
    border: none;
    padding: 0;
    color: var(--gray-800);
}

.message-content a {
    color: var(--secondary-blue);
    text-decoration: none;
    font-weight: 500;
    border-bottom: 1px solid transparent;
    transition: var(--transition-fast);
}

.message-content a:hover {
    border-bottom-color: var(--secondary-blue);
}

.message-content blockquote {
    border-left: 4px solid var(--secondary-blue);
    padding-left: 15px;
    margin: 15px 0;
    color: var(--gray-600);
    font-style: italic;
    background: rgba(59, 130, 246, 0.05);
    padding: 15px;
    border-radius: 0 10px 10px 0;
}

.message-content table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    background: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

.message-content th,
.message-content td {
    padding: 12px 15px;
    text-align: left;
    border-bottom: 1px solid var(--gray-200);
}

.message-content th {
    background: var(--secondary-gradient);
    color: white;
    font-weight: 600;
}

.message-content tr:hover {
    background: rgba(59, 130, 246, 0.05);
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-container {
        padding: 10px;
    }
    
    .header {
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .header-content {
        flex-direction: column;
        align-items: flex-start;
        gap: 15px;
    }
    
    .logo-section {
        flex-direction: column;
        align-items: flex-start;
        gap: 15px;
    }
    
    .title-section h1 {
        font-size: 1.8rem;
    }
    
    .subtitle {
        font-size: 1rem;
    }
    
    .source-options {
        flex-direction: column;
        gap: 10px;
    }
    
    .chat-header {
        padding: 15px 20px;
        flex-direction: column;
        gap: 15px;
        align-items: flex-start;
    }
    
    .chat-messages {
        padding: 15px 20px;
    }
    
    .message-content {
        max-width: 85%;
    }
    
    .welcome-section {
        padding: 40px 20px;
    }
    
    .welcome-content h2 {
        font-size: 1.8rem;
    }
    
    .query-examples {
        grid-template-columns: 1fr;
    }
    
    .input-container {
        padding: 20px;
    }
    
    .input-hints {
        flex-direction: column;
        gap: 10px;
        align-items: flex-start;
    }
    
    .floating-help {
        bottom: 20px;
        left: 20px;
        width: 45px;
        height: 45px;
    }
    
    .help-tooltip {
        width: 280px;
    }
    
    .status-indicator {
        top: 10px;
        right: 10px;
        font-size: 12px;
        padding: 6px 12px;
    }
}

@media (max-width: 480px) {
    .header {
        border-radius: 15px;
        padding: 15px;
    }
    
    .title-section h1 {
        font-size: 1.5rem;
    }
    
    .chat-container {
        border-radius: 15px;
    }
    
    .input-section {
        border-radius: 15px;
    }
    
    .input-wrapper {
        padding: 12px 15px;
        flex-direction: column;
        gap: 10px;
    }
    
    .input-actions {
        align-self: flex-end;
    }
    
    .message-content {
        max-width: 90%;
    }
    
    .help-tooltip {
        width: 250px;
    }
}

/* Animations */
@keyframes fadeInUp {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes typingBounce {
    0%, 80%, 100% {
        transform: scale(0);
    }
    40% {
        transform: scale(1);
    }
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

@keyframes slideInRight {
    to {
        transform: translateX(0);
    }
}

@keyframes float {
    0%, 100% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-20px);
    }
}

/* Utility Classes */
.hidden {
    display: none !important;
}

.fade-in {
    animation: fadeInUp 0.5s ease-out forwards;
}

.blur-background {
    backdrop-filter: blur(10px);
}

/* Scrollbar Styling */
.chat-messages::-webkit-scrollbar {
    width: 6px;
}

.chat-messages::-webkit-scrollbar-track {
    background: var(--gray-100);
    border-radius: 3px;
}

.chat-messages::-webkit-scrollbar-thumb {
    background: var(--gray-300);
    border-radius: 3px;
    transition: var(--transition-normal);
}

.chat-messages::-webkit-scrollbar-thumb:hover {
    background: var(--gray-400);
}

/* Focus Styles */
button:focus-visible,
input:focus-visible,
textarea:focus-visible {
    outline: 2px solid var(--secondary-blue);
    outline-offset: 2px;
}

/* Selection Styles */
::selection {
    background: rgba(59, 130, 246, 0.2);
    color: var(--gray-800);
}

/* Print Styles */
@media print {
    .background-pattern,
    .status-indicator,
    .floating-help,
    .loading-overlay,
    .toast-container,
    .particles {
        display: none !important;
    }
    
    .main-container {
        max-width: none;
        padding: 0;
    }
    
    .header,
    .chat-container,
    .input-section {
        box-shadow: none;
        border: 1px solid var(--gray-300);
    }
}

/* High Contrast Mode */
@media (prefers-contrast: high) {
    :root {
        --gray-100: #f0f0f0;
        --gray-200: #e0e0e0;
        --gray-300: #c0c0c0;
        --gray-400: #a0a0a0;
        --gray-500: #808080;
        --gray-600: #606060;
        --gray-700: #404040;
        --gray-800: #202020;
        --gray-900: #000000;
    }
    
    .message-bubble {
        border-width: 2px;
    }
    
    .source-option label {
        border-width: 3px;
    }
}

/* Reduced Motion */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
    
    .particles {
        display: none;
    }
}

/* Dark Mode Support (for future enhancement) */
@media (prefers-color-scheme: dark) {
    /* Dark mode styles can be added here in the future */
}

























// CME Unified Knowledge Assistant - Complete Frontend JavaScript
class UnifiedAssistant {
    constructor() {
        this.apiBase = '/api';
        this.isTyping = false;
        this.messageHistory = [];
        this.particleCount = 0;
        this.maxParticles = 20;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeParticles();
        this.checkSystemStatus();
        this.setupSourceTracking();
        this.initializeAnimations();
        this.setupAccessibility();
        
        console.log('🚀 CME Unified Knowledge Assistant initialized');
    }

    setupEventListeners() {
        // Message input handling
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const clearChat = document.getElementById('clearChat');

        if (messageInput) {
            messageInput.addEventListener('input', this.handleInputChange.bind(this));
            messageInput.addEventListener('keydown', this.handleKeyDown.bind(this));
            messageInput.addEventListener('paste', this.handlePaste.bind(this));
        }

        if (sendButton) {
            sendButton.addEventListener('click', this.sendMessage.bind(this));
        }

        if (clearChat) {
            clearChat.addEventListener('click', this.clearChatHistory.bind(this));
        }

        // Source selector handling
        const sourceCheckboxes = document.querySelectorAll('.source-option input[type="checkbox"]');
        sourceCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', this.updateSelectedSources.bind(this));
        });

        // Example query handling
        const exampleQueries = document.querySelectorAll('.example-query');
        exampleQueries.forEach(query => {
            query.addEventListener('click', this.handleExampleQuery.bind(this));
        });

        // Floating help
        const floatingHelp = document.getElementById('floatingHelp');
        if (floatingHelp) {
            floatingHelp.addEventListener('click', this.toggleHelp.bind(this));
        }

        // Window events
        window.addEventListener('resize', this.handleResize.bind(this));
        window.addEventListener('scroll', this.handleScroll.bind(this));
        
        // Parallax effect
        document.addEventListener('mousemove', this.handleMouseMove.bind(this));
    }

    handleInputChange(event) {
        const input = event.target;
        const charCount = document.getElementById('charCount');
        const sendButton = document.getElementById('sendButton');

        // Update character count
        if (charCount) {
            charCount.textContent = input.value.length;
        }

        // Auto-resize textarea
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';

        // Enable/disable send button
        if (sendButton) {
            sendButton.disabled = !input.value.trim();
        }

        // Update placeholder based on sources
        this.updateInputPlaceholder();
    }

    handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            if (!this.isTyping && event.target.value.trim()) {
                this.sendMessage();
            }
        }
    }

    handlePaste(event) {
        // Handle paste events
        setTimeout(() => {
            this.handleInputChange(event);
        }, 0);
    }

    async sendMessage() {
        console.log('🚀 Send message triggered');
        
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();

        if (!message || this.isTyping) {
            console.log('❌ Message empty or currently typing, aborting');
            return;
        }

        // Get selected sources
        const selectedSources = this.getSelectedSources();
        console.log('📊 Selected sources:', selectedSources);
        
        if (selectedSources.length === 0) {
            this.showToast('Please select at least one data source', 'warning');
            return;
        }

        console.log('📝 Sending message:', message);

        // Clear input and hide welcome
        messageInput.value = '';
        this.handleInputChange({ target: messageInput });
        this.hideWelcomeSection();

        // Add user message immediately
        this.addMessage(message, 'user');
        
        // Show typing indicator
        this.showTypingIndicator();

        try {
            console.log('🌐 Making API request...');
            
            // Send to backend
            const response = await fetch(`${this.apiBase}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    sources: selectedSources
                })
            });

            console.log('📡 API response status:', response.status);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('❌ API error response:', errorText);
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('✅ API response received:', data);
            
            // Hide typing indicator
            this.hideTypingIndicator();

            // Add assistant response
            if (data.response) {
                this.addMessage(data.response, 'assistant');
                console.log('🤖 Assistant message added successfully');
            } else {
                console.error('❌ No response in API data:', data);
                this.addMessage('I received an empty response. Please try again.', 'assistant');
            }
            
            // Store in history
            this.messageHistory.push({
                user: message,
                assistant: data.response || 'Empty response',
                sources: selectedSources,
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            console.error('❌ Error sending message:', error);
            this.hideTypingIndicator();
            
            const errorMessage = `I apologize, but I'm having trouble connecting to the knowledge systems right now. Please try again in a moment.\n\n**Error Details:** ${error.message}`;
            this.addMessage(errorMessage, 'assistant');
            this.showToast('Connection error - please try again', 'error');
        }
    }

    addMessage(content, sender) {
        console.log(`📨 Adding ${sender} message:`, content.substring(0, 100) + '...');
        
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) {
            console.error('❌ Chat messages container not found');
            return;
        }

        // Create message container
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        // Create avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = sender === 'user' ? 'U' : '🤖';

        // Create message content container
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        // Create message bubble
        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';
        
        // Format and set message content
        messageBubble.innerHTML = this.formatMessage(content);

        // Create timestamp
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = new Date().toLocaleTimeString();

        // Assemble the message
        messageContent.appendChild(messageBubble);
        messageContent.appendChild(messageTime);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);

        // Add to chat with animation
        chatMessages.appendChild(messageDiv);

        // Animate in
        requestAnimationFrame(() => {
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        });

        // Scroll to bottom
        this.scrollToBottom();
        
        console.log(`✅ ${sender} message added successfully`);
    }

    formatMessage(content) {
        if (!content) return '';
        
        let formatted = content
            // Headers with proper hierarchy
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            // Bold text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic text
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Code blocks
            .replace(/```(\w+)?\n?([\s\S]*?)```/g, (match, lang, code) => {
                const language = lang ? ` data-language="${lang}"` : '';
                return `<pre><code${language}>${this.escapeHtml(code.trim())}</code></pre>`;
            })
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Links with proper accessibility
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" title="$1 (opens in new tab)">$1 <i class="fas fa-external-link-alt" style="font-size: 0.8em; margin-left: 4px;"></i></a>')
            // Line breaks
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');

        // Handle lists
        formatted = this.formatLists(formatted);
        
        // Wrap in paragraphs if needed
        if (!formatted.startsWith('<h') && !formatted.startsWith('<p') && !formatted.startsWith('<ul') && !formatted.startsWith('<ol')) {
            formatted = '<p>' + formatted + '</p>';
        }

        return formatted;
    }

    formatLists(content) {
        // Unordered lists
        let lines = content.split('\n');
        let inList = false;
        let result = [];
        
        for (let line of lines) {
            if (line.match(/^[•\-\*]\s+/)) {
                if (!inList) {
                    result.push('<ul>');
                    inList = true;
                }
                result.push('<li>' + line.replace(/^[•\-\*]\s+/, '') + '</li>');
            } else if (line.match(/^\d+\.\s+/)) {
                if (!inList) {
                    result.push('<ol>');
                    inList = true;
                }
                result.push('<li>' + line.replace(/^\d+\.\s+/, '') + '</li>');
            } else {
                if (inList) {
                    result.push('</ul>');
                    inList = false;
                }
                result.push(line);
            }
        }
        
        if (inList) {
            result.push('</ul>');
        }
        
        return result.join('\n');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showTypingIndicator() {
        this.isTyping = true;
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.classList.add('show');
        }
        this.scrollToBottom();
        console.log('⏳ Typing indicator shown');
    }

    hideTypingIndicator() {
        this.isTyping = false;
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.classList.remove('show');
        }
        console.log('✅ Typing indicator hidden');
    }

    scrollToBottom() {
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            setTimeout(() => {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }, 100);
        }
    }

    hideWelcomeSection() {
        const welcomeSection = document.getElementById('welcomeSection');
        if (welcomeSection) {
            welcomeSection.style.display = 'none';
        }
    }

    clearChatHistory() {
        const chatMessages = document.getElementById('chatMessages');
        const welcomeSection = document.getElementById('welcomeSection');
        
        if (chatMessages) {
            chatMessages.innerHTML = '';
        }
        
        if (welcomeSection) {
            welcomeSection.style.display = 'block';
        }
        
        this.messageHistory = [];
        this.showToast('Chat history cleared', 'success');
        console.log('🧹 Chat history cleared');
    }

    getSelectedSources() {
        const sources = [];
        const confluenceCheck = document.getElementById('confluence');
        const jiraCheck = document.getElementById('jira');

        if (confluenceCheck && confluenceCheck.checked) {
            sources.push('confluence');
        }
        if (jiraCheck && jiraCheck.checked) {
            sources.push('jira');
        }

        // If both are selected, add 'both' for backend processing
        if (sources.length === 2) {
            sources.push('both');
        }

        return sources;
    }

    updateSelectedSources() {
        const sourcesList = document.getElementById('sourcesList');
        if (!sourcesList) return;

        const sources = this.getSelectedSources();
        let displayText = '';

        if (sources.includes('confluence') && sources.includes('jira')) {
            displayText = 'Confluence + Jira';
        } else if (sources.includes('confluence')) {
            displayText = 'Confluence Only';
        } else if (sources.includes('jira')) {
            displayText = 'Jira Only';
        } else {
            displayText = 'No Sources Selected';
        }

        sourcesList.textContent = displayText;
        this.updateInputPlaceholder();
    }

    updateInputPlaceholder() {
        const messageInput = document.getElementById('messageInput');
        if (!messageInput) return;

        const sources = this.getSelectedSources();
        let placeholder = '';

        if (sources.includes('confluence') && sources.includes('jira')) {
            placeholder = 'Ask me anything about CME\'s documentation and tickets...';
        } else if (sources.includes('confluence')) {
            placeholder = 'Ask me about CME\'s documentation and procedures...';
        } else if (sources.includes('jira')) {
            placeholder = 'Ask me about tickets, issues, and project tracking...';
        } else {
            placeholder = 'Please select at least one data source first...';
        }

        messageInput.placeholder = placeholder;
    }

    handleExampleQuery(event) {
        const query = event.currentTarget.getAttribute('data-query');
        if (query) {
            const messageInput = document.getElementById('messageInput');
            if (messageInput) {
                messageInput.value = query;
                this.handleInputChange({ target: messageInput });
                messageInput.focus();
            }
        }
    }

    async checkSystemStatus() {
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = statusIndicator?.querySelector('.status-text');

        try {
            console.log('🔍 Checking system status...');
            const response = await fetch(`${this.apiBase}/status`);
            const data = await response.json();

            let status = 'error';
            let text = 'System Error';

            if (response.ok) {
                if (data.confluence && data.jira) {
                    status = 'connected';
                    text = `Connected (${data.confluence_pages} pages loaded)`;
                } else if (data.confluence || data.jira) {
                    status = 'degraded';
                    text = 'Partially Connected';
                } else {
                    status = 'error';
                    text = 'Connection Failed';
                }
            }

            if (statusIndicator) {
                statusIndicator.className = `status-indicator ${status}`;
            }
            if (statusText) {
                statusText.textContent = text;
            }

            console.log(`📊 System status: ${status} - ${text}`);

        } catch (error) {
            console.error('❌ Status check failed:', error);
            if (statusIndicator) {
                statusIndicator.className = 'status-indicator error';
            }
            if (statusText) {
                statusText.textContent = 'Connection Error';
            }
        }

        // Check status every 30 seconds
        setTimeout(() => this.checkSystemStatus(), 30000);
    }

    initializeParticles() {
        const particlesContainer = document.getElementById('particles');
        if (!particlesContainer) return;

        // Create initial particles
        for (let i = 0; i < this.maxParticles; i++) {
            setTimeout(() => this.createParticle(), i * 200);
        }

        // Continue creating particles
        setInterval(() => {
            if (this.particleCount < this.maxParticles) {
                this.createParticle();
            }
        }, 1000);
    }

    createParticle() {
        const particlesContainer = document.getElementById('particles');
        if (!particlesContainer) return;

        const particle = document.createElement('div');
        const type = Math.floor(Math.random() * 3) + 1;
        
        particle.className = `particle particle-${type}`;
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDuration = (8 + Math.random() * 8) + 's';
        particle.style.animationDelay = Math.random() * 2 + 's';

        particlesContainer.appendChild(particle);
        this.particleCount++;

        // Remove particle after animation
        setTimeout(() => {
            if (particle.parentNode) {
                particle.parentNode.removeChild(particle);
                this.particleCount--;
            }
        }, 16000);
    }

    initializeAnimations() {
        // Add stagger animation to example queries
        const exampleQueries = document.querySelectorAll('.example-query');
        exampleQueries.forEach((query, index) => {
            query.classList.add('stagger-item');
            query.style.animationDelay = `${0.1 + index * 0.1}s`;
        });

        // Add morphing shapes
        this.createMorphingShapes();
        
        // Setup intersection observer for scroll animations
        this.setupScrollAnimations();
    }

    createMorphingShapes() {
        const backgroundPattern = document.querySelector('.background-pattern');
        if (!backgroundPattern) return;

        const shapesContainer = document.createElement('div');
        shapesContainer.className = 'morphing-shapes';

        // Create 3 morphing shapes
        for (let i = 1; i <= 3; i++) {
            const shape = document.createElement('div');
            shape.className = 'morphing-shape';
            shapesContainer.appendChild(shape);
        }

        backgroundPattern.appendChild(shapesContainer);
    }

    setupScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, observerOptions);

        // Observe elements for scroll animations
        document.querySelectorAll('.fade-in-viewport').forEach(el => {
            observer.observe(el);
        });
    }

    handleMouseMove(event) {
        // Parallax effect for floating elements
        const floatingElements = document.querySelectorAll('.floating-element');
        const { clientX, clientY } = event;
        const { innerWidth, innerHeight } = window;

        const xPercent = (clientX / innerWidth) * 100;
        const yPercent = (clientY / innerHeight) * 100;

        floatingElements.forEach((element, index) => {
            const speed = 0.05 + (index * 0.02);
            const x = (xPercent - 50) * speed;
            const y = (yPercent - 50) * speed;
            
            element.style.transform = `translate(${x}px, ${y}px)`;
        });
    }

    handleResize() {
        // Adjust particle count based on screen size
        const width = window.innerWidth;
        
        if (width < 768) {
            this.maxParticles = 10;
        } else if (width < 1024) {
            this.maxParticles = 15;
        } else {
            this.maxParticles = 20;
        }

        // Update input placeholder
        this.updateInputPlaceholder();
    }

    handleScroll() {
        // Parallax scrolling effect
        const scrolled = window.pageYOffset;
        const parallaxElements = document.querySelectorAll('.parallax-element');
        
        parallaxElements.forEach((element, index) => {
            const speed = 0.1 + (index * 0.05);
            const yPos = -(scrolled * speed);
            element.style.transform = `translateY(${yPos}px)`;
        });
    }

    showToast(message, type = 'info', duration = 4000) {
        const toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = this.getToastIcon(type);
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="${icon}"></i>
                <span>${message}</span>
            </div>
        `;

        toastContainer.appendChild(toast);

        // Auto remove
        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s ease-in forwards';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, duration);

        // Manual close on click
        toast.addEventListener('click', () => {
            toast.style.animation = 'slideOutRight 0.3s ease-in forwards';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        });
    }

    getToastIcon(type) {
        switch (type) {
            case 'success': return 'fas fa-check-circle';
            case 'error': return 'fas fa-exclamation-circle';
            case 'warning': return 'fas fa-exclamation-triangle';
            default: return 'fas fa-info-circle';
        }
    }

    toggleHelp() {
        const tooltip = document.querySelector('.help-tooltip');
        if (tooltip) {
            const isVisible = tooltip.style.opacity === '1';
            tooltip.style.opacity = isVisible ? '0' : '1';
            tooltip.style.visibility = isVisible ? 'hidden' : 'visible';
            tooltip.style.transform = isVisible ? 'translateY(10px)' : 'translateY(0)';
        }
    }

    setupSourceTracking() {
        // Track source changes for analytics
        const sourceCheckboxes = document.querySelectorAll('.source-option input[type="checkbox"]');
        sourceCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                const sources = this.getSelectedSources();
                console.log('📊 Sources updated:', sources);
                
                // Update UI feedback
                this.updateSourceFeedback(sources);
            });
        });

        // Initialize with default state
        this.updateSelectedSources();
    }

    updateSourceFeedback(sources) {
        // Visual feedback for source selection
        const sourceOptions = document.querySelectorAll('.source-option');
        
        sourceOptions.forEach(option => {
            const checkbox = option.querySelector('input[type="checkbox"]');
            const label = option.querySelector('label');
            
            if (checkbox && label) {
                if (checkbox.checked) {
                    label.classList.add('interactive-element');
                    option.classList.add('breathing');
                } else {
                    label.classList.remove('interactive-element');
                    option.classList.remove('breathing');
                }
            }
        });

        // Show warning if no sources selected
        if (sources.length === 0) {
            this.showToast('Please select at least one data source', 'warning', 3000);
        }
    }

    setupAccessibility() {
        // Enhanced keyboard navigation
        const focusableElements = document.querySelectorAll(
            'button, input, textarea, [tabindex]:not([tabindex="-1"])'
        );

        focusableElements.forEach(element => {
            element.classList.add('focus-ring');
            
            // Add ripple effect to buttons
            if (element.tagName === 'BUTTON') {
                element.classList.add('ripple');
            }
        });

        // Screen reader announcements
        this.setupScreenReaderSupport();
        
        // High contrast mode detection
        this.detectHighContrast();
    }

    setupScreenReaderSupport() {
        // Create announcer for screen readers
        const announcer = document.createElement('div');
        announcer.setAttribute('aria-live', 'polite');
        announcer.setAttribute('aria-atomic', 'true');
        announcer.className = 'sr-only';
        announcer.style.cssText = `
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            padding: 0 !important;
            margin: -1px !important;
            overflow: hidden !important;
            clip: rect(0, 0, 0, 0) !important;
            white-space: nowrap !important;
            border: 0 !important;
        `;
        document.body.appendChild(announcer);

        this.announcer = announcer;
    }

    detectHighContrast() {
        // Detect high contrast mode
        if (window.matchMedia('(prefers-contrast: high)').matches) {
            document.body.classList.add('high-contrast-mode');
            console.log('🎨 High contrast mode detected');
        }

        // Listen for changes
        window.matchMedia('(prefers-contrast: high)').addEventListener('change', (e) => {
            if (e.matches) {
                document.body.classList.add('high-contrast-mode');
            } else {
                document.body.classList.remove('high-contrast-mode');
            }
        });
    }

    // Performance monitoring
    measurePerformance(name, fn) {
        const start = performance.now();
        const result = fn();
        const end = performance.now();
        console.log(`⚡ ${name} took ${(end - start).toFixed(2)} milliseconds`);
        return result;
    }

    // Error handling and recovery
    handleError(error, context = '') {
        console.error(`❌ Error in ${context}:`, error);
        
        // Log to analytics if available
        if (window.gtag) {
            window.gtag('event', 'exception', {
                description: `${context}: ${error.message}`,
                fatal: false
            });
        }

        // Show user-friendly error
        this.showToast(
            'Something went wrong. Please try again or refresh the page.',
            'error',
            5000
        );
    }

    // Cleanup and destroy
    destroy() {
        // Remove event listeners
        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('scroll', this.handleScroll);
        document.removeEventListener('mousemove', this.handleMouseMove);

        // Clear particles
        const particlesContainer = document.getElementById('particles');
        if (particlesContainer) {
            particlesContainer.innerHTML = '';
        }

        // Clear any running intervals/timeouts
        this.particleCount = 0;

        console.log('🧹 UnifiedAssistant cleaned up');
    }
}

// Enhanced slideOutRight animation keyframes
const slideOutRightKeyframes = `
@keyframes slideOutRight {
    from {
        transform: translateX(0);
        opacity: 1;
    }
    to {
        transform: translateX(400px);
        opacity: 0;
    }
}
`;

// Add the keyframes to the document
const style = document.createElement('style');
style.textContent = slideOutRightKeyframes;
document.head.appendChild(style);

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎯 DOM Content Loaded');
    
    // Check for modern browser features
    if (!window.fetch || !window.Promise) {
        alert('This application requires a modern browser. Please update your browser to continue.');
        return;
    }

    // Initialize the assistant
    console.log('🚀 Initializing CME Unified Knowledge Assistant...');
    window.assistant = new UnifiedAssistant();

    // Service worker registration for offline support (optional)
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js').catch(err => {
            console.log('🔧 Service worker registration failed:', err);
        });
    }

    // Add loading class to body initially
    document.body.classList.add('loading');
    
    // Remove loading class after initialization
    setTimeout(() => {
        document.body.classList.remove('loading');
        document.body.classList.add('loaded');
        console.log('✅ Application fully loaded and ready');
    }, 1000);

    // Global error handler
    window.addEventListener('error', (event) => {
        if (window.assistant) {
            window.assistant.handleError(event.error, 'Global Error');
        }
    });

    // Unhandled promise rejection handler
    window.addEventListener('unhandledrejection', (event) => {
        if (window.assistant) {
            window.assistant.handleError(event.reason, 'Unhandled Promise Rejection');
        }
    });

    console.log('🎉 CME Unified Knowledge Assistant ready!');
});

// Expose assistant to global scope for debugging
window.CMEAssistant = UnifiedAssistant;
