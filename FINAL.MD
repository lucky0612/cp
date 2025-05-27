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
import queue

# Web framework
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# HTTP requests
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# AI/ML
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UnifiedAssistant")

# Configuration
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
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "ATATTxEfGFdPemKZpWezVd58xSZ5242xlMJMRvelfVdjcrmm7G3G0KQeaBdBRw1aATINeOAwHPNmwy8qUQ2oL4kyJRWjQhIKygnc55R7O.lw.3wl1EkopOz1lFRKCYPEhR0A6ZHZNZwMCRjrYuj1Blh5r5legJFD3dGhlcwhBz0mJ1KXZE-reZDUWM=851BA963")

# Performance settings
MAX_WORKERS = 5
CACHE_SIZE = 128

class ContentExtractor:
    """Extract and process content from Confluence HTML."""
    
    def __init__(self, base_url, page_id=None, page_title=None):
        self.base_url = base_url
        self.page_id = page_id
        self.page_title = page_title
    
    def extract_content_from_html(self, html_content):
        """Extract text, tables, and images from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        metadata = {
            "id": self.page_id,
            "title": self.page_title,
            "url": f"{self.base_url}/pages/viewpage.action?pageId={self.page_id}" if self.page_id else None,
            "type": "page",
            "labels": []
        }
        
        sections = []
        
        # Extract text
        text_content = self._extract_text_content(soup)
        if text_content:
            sections.append({"type": "text", "content": text_content})
        
        # Extract tables
        tables = self._extract_tables(soup)
        if tables:
            sections.append({"type": "tables", "content": tables})
        
        return {
            "metadata": metadata,
            "sections": sections,
            "raw_html": html_content
        }
    
    def _extract_text_content(self, soup):
        """Extract and format text content."""
        text_content = []
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract headings
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(heading.name[1])
            text = heading.get_text(strip=True)
            if text:
                text_content.append({"type": "heading", "level": level, "text": text})
        
        # Extract paragraphs
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text:
                text_content.append({"type": "paragraph", "text": text})
        
        return text_content
    
    def _extract_tables(self, soup):
        """Extract tables with improved handling."""
        tables = []
        
        for table in soup.find_all('table'):
            table_data = []
            headers = []
            
            # Extract headers
            thead = table.find('thead')
            if thead:
                for th in thead.find_all('th'):
                    headers.append(th.get_text(strip=True))
            
            # Extract rows
            tbody = table.find('tbody') or table
            for tr in tbody.find_all('tr'):
                row_data = []
                for td in tr.find_all(['td', 'th']):
                    row_data.append(td.get_text(strip=True))
                if row_data:
                    table_data.append(row_data)
            
            if table_data:
                tables.append({
                    "headers": headers,
                    "rows": table_data
                })
        
        return tables
    
    def format_for_context(self, extracted_content):
        """Format extracted content for AI context."""
        formatted = []
        
        metadata = extracted_content.get('metadata', {})
        if metadata.get('title'):
            formatted.append(f"## {metadata['title']}")
        if metadata.get('url'):
            formatted.append(f"URL: {metadata['url']}")
        formatted.append("")
        
        for section in extracted_content.get('sections', []):
            section_type = section.get('type')
            content = section.get('content', [])
            
            if section_type == 'text':
                for item in content:
                    if item['type'] == 'heading':
                        formatted.append("#" * item['level'] + " " + item['text'])
                    elif item['type'] == 'paragraph':
                        formatted.append(item['text'])
                    formatted.append("")
            
            elif section_type == 'tables':
                formatted.append("### Tables")
                for table in content:
                    if table['headers']:
                        formatted.append(" | ".join(table['headers']))
                        formatted.append("-" * 50)
                    for row in table['rows']:
                        formatted.append(" | ".join(row))
                    formatted.append("")
        
        return "\n".join(formatted)

class ConfluenceClient:
    """Client for Confluence REST API operations."""
    
    def __init__(self, base_url, username, api_token):
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json"
        })
        self.cache = {}
        self.cache_lock = threading.Lock()
    
    def test_connection(self):
        """Test connection to Confluence."""
        try:
            response = self.session.get(
                f"{self.base_url}/rest/api/content",
                params={"limit": 1},
                timeout=30,
                verify=False
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_all_pages_in_space(self, space_key, batch_size=50):
        """Get all pages in a space."""
        all_pages = []
        start = 0
        
        while True:
            try:
                params = {
                    "spaceKey": space_key,
                    "expand": "history",
                    "limit": batch_size,
                    "start": start
                }
                
                response = self.session.get(
                    f"{self.base_url}/rest/api/content",
                    params=params,
                    timeout=30,
                    verify=False
                )
                
                if response.status_code != 200:
                    break
                
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    break
                
                all_pages.extend(results)
                
                if len(results) < batch_size:
                    break
                
                start += batch_size
                time.sleep(0.2)
                
            except Exception:
                break
        
        return all_pages
    
    def get_page_content(self, page_id):
        """Get content of a page."""
        try:
            cache_key = f"page_content_{page_id}"
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]
            
            response = self.session.get(
                f"{self.base_url}/rest/api/content/{page_id}",
                params={"expand": "body.storage"},
                timeout=30,
                verify=False
            )
            
            if response.status_code == 200:
                page_data = response.json()
                
                metadata = {
                    "id": page_data.get("id"),
                    "title": page_data.get("title"),
                    "url": f"{self.base_url}/pages/viewpage.action?pageId={page_data.get('id')}",
                    "space": page_data.get("space", {}).get("key", "")
                }
                
                body = page_data.get("body", {})
                storage = body.get("storage", {})
                html_content = storage.get("value", "")
                
                extractor = ContentExtractor(
                    base_url=self.base_url,
                    page_id=page_data.get("id"),
                    page_title=page_data.get("title")
                )
                extracted_content = extractor.extract_content_from_html(html_content)
                formatted_content = extractor.format_for_context(extracted_content)
                
                result = {
                    "metadata": metadata,
                    "content": formatted_content
                }
                
                with self.cache_lock:
                    self.cache[cache_key] = result
                
                return result
            
            return None
                
        except Exception:
            return None

class JiraClient:
    """Safe Jira client with error handling."""
    
    def __init__(self):
        self.base_url = JIRA_URL.rstrip('/')
        self.auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self._metadata = None
    
    def safe_get(self, obj, key, default=None):
        """Ultra-safe getter that never fails."""
        try:
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return default
        except:
            return default
    
    def make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make safe HTTP request."""
        url = f"{self.base_url}/rest/api/3/{endpoint}"
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                params=params or {},
                auth=self.auth,
                verify=False,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            return None
                
        except Exception:
            return None
    
    def test_connection(self):
        """Test connection to Jira."""
        try:
            result = self.make_request('myself')
            return result is not None
        except:
            return False
    
    def get_jira_metadata(self) -> Dict:
        """Get Jira metadata for AI context."""
        if self._metadata:
            return self._metadata
        
        metadata = {
            'projects': [],
            'users': [],
            'statuses': [],
            'priorities': [],
            'issue_types': []
        }
        
        try:
            # Get projects
            projects = self.make_request('project') or []
            for project in projects:
                if project:
                    metadata['projects'].append({
                        'key': self.safe_get(project, 'key', ''),
                        'name': self.safe_get(project, 'name', '')
                    })
            
            # Get statuses
            statuses = self.make_request('status') or []
            for status in statuses:
                if status:
                    metadata['statuses'].append(self.safe_get(status, 'name', ''))
            
            self._metadata = metadata
            
        except Exception:
            pass
        
        return metadata
    
    def execute_jql(self, jql: str, max_results: int = 50) -> List[Dict]:
        """Execute JQL query safely."""
        if not jql or not jql.strip():
            return []
        
        try:
            params = {
                "jql": jql,
                "maxResults": max_results,
                "fields": "*all"
            }
            
            result = self.make_request('search', params)
            if result:
                issues = self.safe_get(result, 'issues', [])
                return [issue for issue in issues if issue]
            
            return []
            
        except Exception:
            return []

class AIAssistant:
    """AI-powered assistant for both Confluence and Jira."""
    
    def __init__(self):
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
    
    def analyze_query_intent(self, user_query: str, selected_sources: List[str]) -> Dict:
        """Analyze user query to understand intent and required sources."""
        
        prompt = f"""Analyze this user query and determine what information sources are needed.

USER QUERY: "{user_query}"
SELECTED SOURCES: {selected_sources}

Available sources:
- confluence: Documentation, how-to guides, procedures, knowledge base
- jira: Tickets, issues, bugs, tasks, project tracking

Determine:
1. What type of information is needed from each source?
2. Should we search both sources or just specific ones?
3. How should results be combined?

Return a JSON response with:
{{
    "use_confluence": true/false,
    "use_jira": true/false,
    "confluence_intent": "what to search for in confluence",
    "jira_intent": "what to search for in jira", 
    "combination_strategy": "how to combine results"
}}
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(temperature=0.1)
            )
            
            response_text = response.text if hasattr(response, 'text') else response.candidates[0].text
            
            # Try to extract JSON
            import json
            try:
                return json.loads(response_text)
            except:
                # Fallback analysis
                query_lower = user_query.lower()
                return {
                    "use_confluence": "confluence" in selected_sources or "both" in selected_sources,
                    "use_jira": "jira" in selected_sources or "both" in selected_sources,
                    "confluence_intent": user_query,
                    "jira_intent": user_query,
                    "combination_strategy": "merge_results"
                }
                
        except Exception:
            # Fallback to simple logic
            query_lower = user_query.lower()
            return {
                "use_confluence": "confluence" in selected_sources or "both" in selected_sources,
                "use_jira": "jira" in selected_sources or "both" in selected_sources,
                "confluence_intent": user_query,
                "jira_intent": user_query,
                "combination_strategy": "merge_results"
            }
    
    def build_jql_queries(self, user_query: str, metadata: Dict) -> List[str]:
        """Build JQL queries for Jira search."""
        
        prompt = f"""Create JQL queries for this search request.

USER QUERY: "{user_query}"

AVAILABLE PROJECTS: {[p['key'] for p in metadata.get('projects', [])]}
AVAILABLE STATUSES: {metadata.get('statuses', [])}

Create 3 different JQL queries from broad to specific:
1. Broad search
2. Specific search  
3. Fallback search

Return only the JQL queries, one per line:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(temperature=0.1)
            )
            
            response_text = response.text if hasattr(response, 'text') else response.candidates[0].text
            
            # Parse JQL queries
            jql_queries = []
            for line in response_text.split('\n'):
                line = line.strip()
                if line and ('~' in line or '=' in line or 'ORDER BY' in line):
                    jql_queries.append(line)
            
            if jql_queries:
                return jql_queries[:3]
            else:
                # Fallback queries
                words = [w for w in user_query.lower().split() if len(w) > 2]
                if words:
                    return [
                        f'text ~ "{" ".join(words[:3])}" ORDER BY updated DESC',
                        f'summary ~ "{words[0]}" ORDER BY updated DESC',
                        'updated >= -30d ORDER BY updated DESC'
                    ]
                return ['updated >= -7d ORDER BY updated DESC']
                
        except Exception:
            # Simple fallback
            words = [w for w in user_query.lower().split() if len(w) > 2]
            if words:
                return [f'text ~ "{words[0]}" ORDER BY updated DESC']
            return ['updated >= -7d ORDER BY updated DESC']
    
    def generate_unified_response(self, user_query: str, confluence_data: str, jira_data: List[Dict], selected_sources: List[str]) -> str:
        """Generate unified response combining both sources."""
        
        # Format Jira data
        jira_summary = ""
        if jira_data:
            jira_summary = f"JIRA RESULTS ({len(jira_data)} tickets found):\n"
            for i, issue in enumerate(jira_data[:10], 1):
                if issue:
                    key = issue.get('key', 'Unknown')
                    fields = issue.get('fields') or {}
                    summary = fields.get('summary', 'No summary')[:100]
                    status = (fields.get('status') or {}).get('name', 'Unknown')
                    jira_summary += f"{i}. {key}: {summary} (Status: {status})\n"
        
        prompt = f"""You are a unified knowledge assistant. Provide a comprehensive response combining information from multiple sources.

USER QUERY: "{user_query}"
SELECTED SOURCES: {selected_sources}

CONFLUENCE INFORMATION:
{confluence_data if confluence_data else "No Confluence data available"}

JIRA INFORMATION:
{jira_summary if jira_summary else "No Jira data available"}

Provide a comprehensive response that:
1. Directly answers the user's question
2. Combines information from both sources intelligently
3. Uses clear formatting with sections and bullet points
4. Includes relevant ticket links for Jira items: [TICKET-KEY](https://cmegroup.atlassian.net/browse/TICKET-KEY)
5. Is professional but conversational
6. Highlights key insights and actionable information

Make the response comprehensive and well-structured:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=4000
                )
            )
            
            return response.text if hasattr(response, 'text') else response.candidates[0].text
            
        except Exception as e:
            return f"I found some information but had trouble formatting the response. Here's what I found:\n\n{confluence_data}\n\n{jira_summary}"

class UnifiedKnowledgeAssistant:
    """Main unified assistant that combines Confluence and Jira."""
    
    def __init__(self):
        self.confluence = ConfluenceClient(CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN)
        self.jira = JiraClient()
        self.ai = AIAssistant()
        self.space_pages = {}
        self.page_content_cache = {}
        
        # Initialize data
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize Confluence space data."""
        try:
            if self.confluence.test_connection():
                logger.info("Loading Confluence pages...")
                pages = self.confluence.get_all_pages_in_space(CONFLUENCE_SPACE)
                if pages:
                    self.space_pages[CONFLUENCE_SPACE] = pages
                    logger.info(f"Loaded {len(pages)} Confluence pages")
        except Exception as e:
            logger.error(f"Error initializing Confluence: {e}")
    
    def search_confluence(self, query: str) -> str:
        """Search Confluence for relevant content."""
        if not self.space_pages:
            return ""
        
        try:
            # Get all pages
            all_pages = []
            for space_key, pages in self.space_pages.items():
                all_pages.extend(pages)
            
            # Score pages by title relevance
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            candidates = []
            
            for page in all_pages:
                title = page.get("title", "").lower()
                score = sum(1 for word in query_words if word in title)
                if score > 0:
                    candidates.append((page, score))
            
            # Sort and get top candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[:5]
            
            if not top_candidates:
                # Get recent pages as fallback
                recent_pages = sorted(all_pages, 
                                    key=lambda p: p.get("history", {}).get("lastUpdated", {}).get("when", "2000-01-01"),
                                    reverse=True)[:5]
                top_candidates = [(p, 0) for p in recent_pages]
            
            # Fetch content for top candidates
            relevant_content = []
            for page, score in top_candidates:
                page_content = self.confluence.get_page_content(page["id"])
                if page_content:
                    content = page_content.get("content", "")
                    if content:
                        relevant_content.append(content)
            
            return "\n\n".join(relevant_content[:3])  # Limit to top 3 results
            
        except Exception as e:
            logger.error(f"Error searching Confluence: {e}")
            return ""
    
    def search_jira(self, query: str) -> List[Dict]:
        """Search Jira for relevant tickets."""
        try:
            # Get Jira metadata
            metadata = self.jira.get_jira_metadata()
            
            # Build JQL queries
            jql_queries = self.ai.build_jql_queries(query, metadata)
            
            # Execute queries and collect results
            all_results = []
            for jql in jql_queries:
                results = self.jira.execute_jql(jql, max_results=20)
                all_results.extend(results)
            
            # Remove duplicates
            unique_results = []
            seen_keys = set()
            for issue in all_results:
                if issue:
                    key = issue.get('key')
                    if key and key not in seen_keys:
                        seen_keys.add(key)
                        unique_results.append(issue)
            
            return unique_results[:20]  # Limit to 20 results
            
        except Exception as e:
            logger.error(f"Error searching Jira: {e}")
            return []
    
    def process_query(self, user_query: str, selected_sources: List[str]) -> str:
        """Process user query and return unified response."""
        start_time = datetime.now()
        logger.info(f"Processing query: '{user_query}' with sources: {selected_sources}")
        
        try:
            # Analyze query intent
            intent = self.ai.analyze_query_intent(user_query, selected_sources)
            
            confluence_data = ""
            jira_data = []
            
            # Search Confluence if needed
            if intent.get("use_confluence", False):
                confluence_data = self.search_confluence(intent.get("confluence_intent", user_query))
            
            # Search Jira if needed
            if intent.get("use_jira", False):
                jira_data = self.search_jira(intent.get("jira_intent", user_query))
            
            # Generate unified response
            response = self.ai.generate_unified_response(
                user_query, confluence_data, jira_data, selected_sources
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Query completed in {elapsed:.2f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I encountered an error while processing your query: {str(e)}. Please try rephrasing your question."

# Flask Application
app = Flask(__name__)
CORS(app)

# Initialize the assistant
assistant = UnifiedKnowledgeAssistant()

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    try:
        data = request.get_json()
        user_query = data.get('message', '').strip()
        selected_sources = data.get('sources', ['both'])
        
        if not user_query:
            return jsonify({'error': 'No message provided'}), 400
        
        # Handle basic greetings
        query_lower = user_query.lower()
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            return jsonify({
                'response': """👋 **Hello! I'm your CME Unified Knowledge Assistant!**

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
            })
        
        if any(word in query_lower for word in ['bye', 'goodbye', 'thanks']):
            return jsonify({
                'response': "👋 **Goodbye!** Feel free to come back anytime for insights across CME's knowledge systems! 🚀"
            })
        
        # Process the query
        response = assistant.process_query(user_query, selected_sources)
        
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/status')
def status():
    """Check system status."""
    try:
        confluence_status = assistant.confluence.test_connection()
        jira_status = assistant.jira.test_connection()
        
        return jsonify({
            'confluence': confluence_status,
            'jira': jira_status,
            'confluence_pages': len(assistant.space_pages.get(CONFLUENCE_SPACE, [])),
            'status': 'healthy' if confluence_status or jira_status else 'degraded'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting CME Unified Knowledge Assistant...")
    print(f"🔗 Confluence: {CONFLUENCE_URL}")
    print(f"🎫 Jira: {JIRA_URL}")
    print("🤖 AI: Gemini 2.0 Flash")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

























<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CME Unified Knowledge Assistant</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='animations.css') }}">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
    <!-- Background Pattern -->
    <div class="background-pattern">
        <div class="cme-logo-pattern"></div>
        <div class="gradient-overlay"></div>
    </div>

    <!-- Status Indicator -->
    <div class="status-indicator" id="statusIndicator">
        <div class="status-dot"></div>
        <span class="status-text">Connecting...</span>
    </div>

    <!-- Main Container -->
    <div class="main-container">
        <!-- Header -->
        <header class="header">
            <div class="header-content">
                <div class="logo-section">
                    <div class="cme-logo">
                        <i class="fas fa-chart-line"></i>
                        <span class="logo-text">CME</span>
                    </div>
                    <div class="title-section">
                        <h1 class="main-title">Unified Knowledge Assistant</h1>
                        <p class="subtitle">AI-Powered Confluence & Jira Intelligence</p>
                    </div>
                </div>
                
                <div class="header-controls">
                    <div class="source-selector">
                        <label class="selector-label">Data Sources:</label>
                        <div class="source-options">
                            <div class="source-option" data-source="confluence">
                                <input type="checkbox" id="confluence" checked>
                                <label for="confluence">
                                    <i class="fas fa-book"></i>
                                    <span>Confluence</span>
                                </label>
                            </div>
                            <div class="source-option" data-source="jira">
                                <input type="checkbox" id="jira" checked>
                                <label for="jira">
                                    <i class="fas fa-ticket-alt"></i>
                                    <span>Jira</span>
                                </label>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </header>

        <!-- Chat Container -->
        <div class="chat-container">
            <div class="chat-header">
                <div class="chat-title">
                    <i class="fas fa-robot"></i>
                    <span>AI Assistant</span>
                </div>
                <div class="chat-controls">
                    <button class="clear-chat" id="clearChat">
                        <i class="fas fa-trash"></i>
                        Clear Chat
                    </button>
                </div>
            </div>

            <!-- Welcome Message -->
            <div class="welcome-section" id="welcomeSection">
                <div class="welcome-content">
                    <div class="welcome-icon">
                        <i class="fas fa-brain"></i>
                    </div>
                    <h2>Welcome to CME Knowledge Hub</h2>
                    <p>Your intelligent assistant for Confluence documentation and Jira tickets</p>
                    
                    <div class="example-queries">
                        <h3>Try asking:</h3>
                        <div class="query-examples">
                            <div class="example-query" data-query="Show me rollout restart tickets and how to solve them">
                                <i class="fas fa-sync-alt"></i>
                                "Show me rollout restart tickets and how to solve them"
                            </div>
                            <div class="example-query" data-query="BAMPS project documentation">
                                <i class="fas fa-folder"></i>
                                "BAMPS project documentation"
                            </div>
                            <div class="example-query" data-query="Recent high priority issues">
                                <i class="fas fa-exclamation-triangle"></i>
                                "Recent high priority issues"
                            </div>
                            <div class="example-query" data-query="How to handle system maintenance">
                                <i class="fas fa-tools"></i>
                                "How to handle system maintenance"
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Chat Messages -->
            <div class="chat-messages" id="chatMessages">
                <!-- Messages will be dynamically added here -->
            </div>

            <!-- Typing Indicator -->
            <div class="typing-indicator" id="typingIndicator">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span class="typing-text">AI is thinking...</span>
            </div>
        </div>

        <!-- Input Section -->
        <div class="input-section">
            <div class="input-container">
                <div class="input-wrapper">
                    <textarea 
                        id="messageInput" 
                        placeholder="Ask me anything about CME's knowledge base..."
                        rows="1"
                        maxlength="2000"
                    ></textarea>
                    <div class="input-actions">
                        <div class="character-count">
                            <span id="charCount">0</span>/2000
                        </div>
                        <button class="send-button" id="sendButton" disabled>
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                </div>
                
                <div class="input-hints">
                    <div class="hint-item">
                        <i class="fas fa-lightbulb"></i>
                        <span>Pro tip: Be specific for better results</span>
                    </div>
                    <div class="selected-sources" id="selectedSources">
                        <i class="fas fa-database"></i>
                        <span>Searching: <span id="sourcesList">Confluence + Jira</span></span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-content">
            <div class="loading-spinner"></div>
            <div class="loading-text">
                <h3>Processing your query...</h3>
                <p>Searching across CME's knowledge systems</p>
            </div>
        </div>
    </div>

    <!-- Toast Notifications -->
    <div class="toast-container" id="toastContainer">
        <!-- Toasts will be dynamically added here -->
    </div>

    <!-- Floating Help Button -->
    <div class="floating-help" id="floatingHelp">
        <i class="fas fa-question-circle"></i>
        <div class="help-tooltip">
            <div class="help-content">
                <h4>Quick Help</h4>
                <ul>
                    <li><strong>Select Sources:</strong> Choose Confluence, Jira, or both</li>
                    <li><strong>Natural Language:</strong> Ask questions naturally</li>
                    <li><strong>Examples:</strong> Click example queries to try them</li>
                    <li><strong>Specific Queries:</strong> More specific = better results</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Particles Background -->
    <div class="particles" id="particles"></div>

    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>





















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
    opacity: 0;
    transform: translateY(20px);
    animation: fadeInUp 0.5s ease-out forwards;
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































/* Advanced Animations & Visual Effects */

/* Particle System */
.particles {
    overflow: hidden;
}

.particle {
    position: absolute;
    border-radius: 50%;
    pointer-events: none;
    opacity: 0;
}

.particle-1 {
    width: 4px;
    height: 4px;
    background: linear-gradient(45deg, #3b82f6, #60a5fa);
    animation: particleFloat1 8s infinite ease-in-out;
}

.particle-2 {
    width: 6px;
    height: 6px;
    background: linear-gradient(45deg, #1e3a8a, #3b82f6);
    animation: particleFloat2 12s infinite ease-in-out;
}

.particle-3 {
    width: 3px;
    height: 3px;
    background: linear-gradient(45deg, #60a5fa, #93c5fd);
    animation: particleFloat3 10s infinite ease-in-out;
}

/* Particle Animations */
@keyframes particleFloat1 {
    0% {
        transform: translateY(100vh) translateX(-50px) rotate(0deg);
        opacity: 0;
    }
    10% {
        opacity: 0.8;
    }
    90% {
        opacity: 0.8;
    }
    100% {
        transform: translateY(-100px) translateX(50px) rotate(360deg);
        opacity: 0;
    }
}

@keyframes particleFloat2 {
    0% {
        transform: translateY(100vh) translateX(30px) rotate(0deg) scale(0.5);
        opacity: 0;
    }
    15% {
        opacity: 0.6;
    }
    85% {
        opacity: 0.6;
    }
    100% {
        transform: translateY(-100px) translateX(-30px) rotate(-360deg) scale(1.2);
        opacity: 0;
    }
}

@keyframes particleFloat3 {
    0% {
        transform: translateY(100vh) translateX(0px) rotate(0deg) scale(0.8);
        opacity: 0;
    }
    20% {
        opacity: 0.4;
    }
    80% {
        opacity: 0.4;
    }
    100% {
        transform: translateY(-100px) translateX(20px) rotate(180deg) scale(1.5);
        opacity: 0;
    }
}

/* Enhanced Header Animations */
.header {
    position: relative;
    overflow: hidden;
}

.header::after {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(
        45deg,
        transparent,
        rgba(255, 255, 255, 0.1),
        transparent
    );
    animation: headerShine 6s infinite;
    pointer-events: none;
}

@keyframes headerShine {
    0% {
        transform: translateX(-100%) translateY(-100%) rotate(45deg);
    }
    50% {
        transform: translateX(100%) translateY(100%) rotate(45deg);
    }
    100% {
        transform: translateX(-100%) translateY(-100%) rotate(45deg);
    }
}

/* Logo Animations */
.cme-logo {
    position: relative;
    overflow: hidden;
}

.cme-logo::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.3),
        transparent
    );
    animation: logoGlow 4s infinite;
}

@keyframes logoGlow {
    0% {
        left: -100%;
    }
    100% {
        left: 100%;
    }
}

.cme-logo i {
    animation: logoIconPulse 3s infinite ease-in-out;
}

@keyframes logoIconPulse {
    0%, 100% {
        transform: scale(1);
        filter: drop-shadow(0 0 5px rgba(255, 255, 255, 0.5));
    }
    50% {
        transform: scale(1.1);
        filter: drop-shadow(0 0 15px rgba(255, 255, 255, 0.8));
    }
}

/* Title Gradient Animation */
.main-title {
    background: linear-gradient(45deg, #ffffff, #dbeafe, #60a5fa, #3b82f6);
    background-size: 300% 300%;
    animation: gradientShift 4s ease-in-out infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

@keyframes gradientShift {
    0%, 100% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
}

/* Source Selector Enhanced Animations */
.source-option {
    position: relative;
    overflow: hidden;
}

.source-option::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.2),
        transparent
    );
    transition: left 0.5s ease;
    border-radius: 12px;
}

.source-option:hover::before {
    left: 100%;
}

.source-option label {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.source-option:hover label {
    background: rgba(255, 255, 255, 0.25);
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.source-option input[type="checkbox"]:checked + label {
    animation: checkboxSelected 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

@keyframes checkboxSelected {
    0% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.1) rotate(2deg);
    }
    100% {
        transform: scale(1.02) translateY(-2px);
    }
}

/* Chat Container Entrance */
.chat-container {
    animation: chatContainerEntrance 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

@keyframes chatContainerEntrance {
    0% {
        opacity: 0;
        transform: translateY(30px) scale(0.98);
    }
    100% {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

/* Message Animations Enhanced */
.message {
    animation: messageSlideIn 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

.message.user {
    animation: messageSlideInUser 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

@keyframes messageSlideIn {
    0% {
        opacity: 0;
        transform: translateX(-30px) translateY(20px) scale(0.95);
    }
    100% {
        opacity: 1;
        transform: translateX(0) translateY(0) scale(1);
    }
}

@keyframes messageSlideInUser {
    0% {
        opacity: 0;
        transform: translateX(30px) translateY(20px) scale(0.95);
    }
    100% {
        opacity: 1;
        transform: translateX(0) translateY(0) scale(1);
    }
}

/* Message Bubble Hover Effects */
.message-bubble {
    position: relative;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.message-bubble:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.message.assistant .message-bubble::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    border-radius: 18px;
    background: linear-gradient(45deg, transparent, rgba(59, 130, 246, 0.05), transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
    pointer-events: none;
}

.message.assistant .message-bubble:hover::before {
    opacity: 1;
}

/* Welcome Section Animations */
.welcome-section {
    animation: welcomeFadeIn 1s ease-out;
}

@keyframes welcomeFadeIn {
    0% {
        opacity: 0;
        transform: translateY(20px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

.welcome-icon {
    animation: welcomeIconFloat 3s ease-in-out infinite;
}

@keyframes welcomeIconFloat {
    0%, 100% {
        transform: translateY(0px) rotate(0deg);
    }
    50% {
        transform: translateY(-10px) rotate(5deg);
    }
}

.welcome-icon::before {
    content: '';
    position: absolute;
    top: -20px;
    left: -20px;
    right: -20px;
    bottom: -20px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.3) 0%, transparent 70%);
    border-radius: 50%;
    animation: iconGlow 2s ease-in-out infinite alternate;
}

@keyframes iconGlow {
    0% {
        opacity: 0.5;
        transform: scale(0.8);
    }
    100% {
        opacity: 1;
        transform: scale(1.2);
    }
}

/* Example Query Animations */
.example-query {
    position: relative;
    overflow: hidden;
}

.example-query::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(59, 130, 246, 0.1),
        transparent
    );
    transition: left 0.6s ease;
    border-radius: 15px;
}

.example-query:hover::before {
    left: 100%;
}

.example-query:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 15px 35px rgba(59, 130, 246, 0.2);
}

.example-query i {
    transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

.example-query:hover i {
    transform: scale(1.2) rotate(10deg);
    color: #1e3a8a;
}

/* Typing Indicator Enhanced */
.typing-dots span {
    animation: typingPulse 1.4s infinite ease-in-out;
    transform-origin: center;
}

.typing-dots span:nth-child(1) { animation-delay: -0.32s; }
.typing-dots span:nth-child(2) { animation-delay: -0.16s; }
.typing-dots span:nth-child(3) { animation-delay: 0s; }

@keyframes typingPulse {
    0%, 80%, 100% {
        transform: scale(0.8);
        opacity: 0.5;
    }
    40% {
        transform: scale(1.2);
        opacity: 1;
    }
}

/* Input Section Animations */
.input-wrapper {
    position: relative;
}

.input-wrapper::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    width: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #60a5fa);
    transition: all 0.3s ease;
    transform: translateX(-50%);
    border-radius: 1px;
}

.input-wrapper:focus-within::after {
    width: 100%;
}

.send-button {
    position: relative;
    overflow: hidden;
}

.send-button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    transition: all 0.3s ease;
    transform: translate(-50%, -50%);
}

.send-button:not(:disabled):active::before {
    width: 100px;
    height: 100px;
}

/* Loading Spinner Enhanced */
.loading-spinner {
    position: relative;
}

.loading-spinner::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 40px;
    height: 40px;
    border: 2px solid transparent;
    border-top: 2px solid #60a5fa;
    border-radius: 50%;
    animation: spinInner 0.8s linear infinite reverse;
    transform: translate(-50%, -50%);
}

@keyframes spinInner {
    to {
        transform: translate(-50%, -50%) rotate(-360deg);
    }
}

/* Status Indicator Pulse */
.status-indicator.connected {
    animation: statusPulse 2s infinite;
}

@keyframes statusPulse {
    0%, 100% {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 0 0 0 rgba(16, 185, 129, 0.4);
    }
    50% {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 0 0 8px rgba(16, 185, 129, 0.1);
    }
}

/* Toast Animations */
.toast {
    position: relative;
    overflow: hidden;
}

.toast::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.3),
        transparent
    );
    animation: toastShine 2s ease-in-out;
}

@keyframes toastShine {
    0% {
        left: -100%;
    }
    100% {
        left: 100%;
    }
}

/* Floating Help Pulse */
.floating-help {
    animation: helpPulse 3s infinite ease-in-out;
}

@keyframes helpPulse {
    0%, 100% {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 0 0 0 rgba(59, 130, 246, 0.5);
    }
    50% {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 0 0 10px rgba(59, 130, 246, 0.2);
    }
}

/* Background Pattern Animation */
.cme-logo-pattern {
    animation: patternShift 20s linear infinite;
}

@keyframes patternShift {
    0% {
        background-position: 0 0, 0 0, 0 0;
    }
    100% {
        background-position: 100px 100px, 80px 80px, 60px 60px;
    }
}

/* Scroll-triggered Animations */
.fade-in-viewport {
    opacity: 0;
    transform: translateY(30px);
    transition: all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

.fade-in-viewport.visible {
    opacity: 1;
    transform: translateY(0);
}

/* Text Typing Animation */
.typing-text {
    overflow: hidden;
    border-right: 2px solid #3b82f6;
    white-space: nowrap;
    margin: 0 auto;
    animation: typing 3s steps(40, end), blink-caret 0.75s step-end infinite;
}

@keyframes typing {
    from {
        width: 0;
    }
    to {
        width: 100%;
    }
}

@keyframes blink-caret {
    from, to {
        border-color: transparent;
    }
    50% {
        border-color: #3b82f6;
    }
}

/* Morphing Shapes Background */
.morphing-shapes {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: -1;
    opacity: 0.05;
}

.morphing-shape {
    position: absolute;
    border-radius: 50%;
    background: linear-gradient(45deg, #3b82f6, #60a5fa);
    animation: morphShape 15s infinite ease-in-out;
}

.morphing-shape:nth-child(1) {
    top: 10%;
    left: 10%;
    width: 200px;
    height: 200px;
    animation-delay: 0s;
}

.morphing-shape:nth-child(2) {
    top: 60%;
    right: 10%;
    width: 150px;
    height: 150px;
    animation-delay: 5s;
}

.morphing-shape:nth-child(3) {
    bottom: 20%;
    left: 30%;
    width: 100px;
    height: 100px;
    animation-delay: 10s;
}

@keyframes morphShape {
    0%, 100% {
        transform: rotate(0deg) scale(1);
        border-radius: 50%;
    }
    25% {
        transform: rotate(90deg) scale(1.2);
        border-radius: 30% 70% 70% 30% / 30% 30% 70% 70%;
    }
    50% {
        transform: rotate(180deg) scale(0.8);
        border-radius: 20% 80% 20% 80% / 80% 20% 80% 20%;
    }
    75% {
        transform: rotate(270deg) scale(1.1);
        border-radius: 70% 30% 30% 70% / 70% 70% 30% 30%;
    }
}

/* Advanced Hover Effects */
.interactive-element {
    position: relative;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.interactive-element::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, rgba(59, 130, 246, 0.1), rgba(147, 197, 253, 0.1));
    opacity: 0;
    transition: opacity 0.3s ease;
    border-radius: inherit;
    z-index: -1;
}

.interactive-element:hover::before {
    opacity: 1;
}

/* Ripple Effect */
.ripple {
    position: relative;
    overflow: hidden;
}

.ripple::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.5);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.ripple:active::after {
    width: 300px;
    height: 300px;
}

/* Glass Morphism Effect */
.glass-morphism {
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}

/* Floating Elements */
.floating-element {
    animation: floatGentle 6s ease-in-out infinite;
}

.floating-element:nth-child(odd) {
    animation-delay: 3s;
    animation-direction: reverse;
}

@keyframes floatGentle {
    0%, 100% {
        transform: translateY(0px) rotate(0deg);
    }
    33% {
        transform: translateY(-10px) rotate(1deg);
    }
    66% {
        transform: translateY(5px) rotate(-1deg);
    }
}

/* Parallax Scrolling Effect */
.parallax-element {
    transition: transform 0.1s ease-out;
}

/* Text Shimmer Effect */
.text-shimmer {
    background: linear-gradient(90deg, 
        #3b82f6 0%, 
        #60a5fa 50%, 
        #93c5fd 75%, 
        #3b82f6 100%
    );
    background-size: 400% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s ease-in-out infinite;
}

@keyframes shimmer {
    0% {
        background-position: -400% 0;
    }
    100% {
        background-position: 400% 0;
    }
}

/* Card Flip Animation */
.flip-card {
    perspective: 1000px;
}

.flip-card-inner {
    position: relative;
    width: 100%;
    height: 100%;
    text-align: center;
    transition: transform 0.8s;
    transform-style: preserve-3d;
}

.flip-card:hover .flip-card-inner {
    transform: rotateY(180deg);
}

.flip-card-front, .flip-card-back {
    position: absolute;
    width: 100%;
    height: 100%;
    backface-visibility: hidden;
    border-radius: inherit;
}

.flip-card-back {
    transform: rotateY(180deg);
}

/* Breathing Animation */
.breathing {
    animation: breathe 4s ease-in-out infinite;
}

@keyframes breathe {
    0%, 100% {
        transform: scale(1);
        opacity: 1;
    }
    50% {
        transform: scale(1.05);
        opacity: 0.8;
    }
}

/* Matrix Rain Effect (Subtle) */
.matrix-rain {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: -1;
    opacity: 0.02;
}

.matrix-char {
    position: absolute;
    color: #3b82f6;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    animation: matrixFall linear infinite;
}

@keyframes matrixFall {
    0% {
        transform: translateY(-100vh);
        opacity: 1;
    }
    100% {
        transform: translateY(100vh);
        opacity: 0;
    }
}

/* Gradient Border Animation */
.gradient-border {
    position: relative;
    background: linear-gradient(white, white) padding-box,
                linear-gradient(45deg, #3b82f6, #60a5fa, #93c5fd) border-box;
    border: 2px solid transparent;
    border-radius: 15px;
}

.gradient-border::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, #3b82f6, #60a5fa, #93c5fd, #3b82f6);
    background-size: 400% 400%;
    border-radius: inherit;
    z-index: -1;
    animation: gradientBorderShift 4s ease infinite;
}

@keyframes gradientBorderShift {
    0%, 100% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
}

/* Staggered Animation */
.stagger-item {
    opacity: 0;
    transform: translateY(30px);
    animation: staggerFadeIn 0.6s ease forwards;
}

.stagger-item:nth-child(1) { animation-delay: 0.1s; }
.stagger-item:nth-child(2) { animation-delay: 0.2s; }
.stagger-item:nth-child(3) { animation-delay: 0.3s; }
.stagger-item:nth-child(4) { animation-delay: 0.4s; }
.stagger-item:nth-child(5) { animation-delay: 0.5s; }

@keyframes staggerFadeIn {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Magnetic Effect */
.magnetic {
    transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

.magnetic:hover {
    transform: scale(1.05);
}

/* Glitch Effect (Subtle) */
.glitch {
    position: relative;
    color: #3b82f6;
    animation: glitch 3s infinite;
}

.glitch::before,
.glitch::after {
    content: attr(data-text);
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0.8;
}

.glitch::before {
    animation: glitchBefore 3s infinite;
    color: #60a5fa;
    z-index: -1;
}

.glitch::after {
    animation: glitchAfter 3s infinite;
    color: #93c5fd;
    z-index: -2;
}

@keyframes glitch {
    0%, 74%, 76%, 100% {
        transform: translate(0);
    }
    75% {
        transform: translate(-2px);
    }
}

@keyframes glitchBefore {
    0%, 74%, 76%, 100% {
        transform: translate(0);
    }
    75% {
        transform: translate(2px, -1px);
    }
}

@keyframes glitchAfter {
    0%, 74%, 76%, 100% {
        transform: translate(0);
    }
    75% {
        transform: translate(-1px, 2px);
    }
}

/* Liquid Animation */
.liquid {
    position: relative;
    overflow: hidden;
}

.liquid::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, 
        transparent, 
        rgba(59, 130, 246, 0.4), 
        transparent
    );
    animation: liquidWave 2s ease-in-out infinite;
}

@keyframes liquidWave {
    0% {
        left: -100%;
        transform: skewX(-15deg);
    }
    100% {
        left: 100%;
        transform: skewX(-15deg);
    }
}

/* Neon Glow Effect */
.neon-glow {
    color: #3b82f6;
    text-shadow: 
        0 0 5px #3b82f6,
        0 0 10px #3b82f6,
        0 0 15px #3b82f6,
        0 0 20px #60a5fa;
    animation: neonFlicker 2s infinite alternate;
}

@keyframes neonFlicker {
    0%, 50%, 100% {
        opacity: 1;
        text-shadow: 
            0 0 5px #3b82f6,
            0 0 10px #3b82f6,
            0 0 15px #3b82f6,
            0 0 20px #60a5fa;
    }
    25%, 75% {
        opacity: 0.8;
        text-shadow: 
            0 0 2px #3b82f6,
            0 0 5px #3b82f6,
            0 0 8px #3b82f6,
            0 0 12px #60a5fa;
    }
}

/* 3D Transform Effects */
.transform-3d {
    perspective: 1000px;
    transform-style: preserve-3d;
}

.rotate-3d {
    animation: rotate3D 10s infinite linear;
}

@keyframes rotate3D {
    0% {
        transform: rotateX(0deg) rotateY(0deg);
    }
    25% {
        transform: rotateX(90deg) rotateY(0deg);
    }
    50% {
        transform: rotateX(90deg) rotateY(90deg);
    }
    75% {
        transform: rotateX(0deg) rotateY(90deg);
    }
    100% {
        transform: rotateX(0deg) rotateY(0deg);
    }
}

/* Performance Optimizations */
.will-change-transform {
    will-change: transform;
}

.will-change-opacity {
    will-change: opacity;
}

.gpu-accelerated {
    transform: translateZ(0);
    backface-visibility: hidden;
}

/* Reduced Motion Overrides */
@media (prefers-reduced-motion: reduce) {
    .morphing-shape,
    .particle,
    .floating-element,
    .breathing,
    .matrix-char,
    .liquid::before,
    .neon-glow,
    .rotate-3d {
        animation: none !important;
    }
    
    .interactive-element:hover,
    .magnetic:hover {
        transform: none !important;
    }
    
    .typing-text {
        animation: none !important;
        border-right: none !important;
    }
    
    * {
        transition-duration: 0.1s !important;
    }
}

/* High Performance Mode */
@media (prefers-reduced-motion: no-preference) and (min-width: 1024px) {
    .enhanced-animations .particle {
        animation-duration: 6s;
    }
    
    .enhanced-animations .morphing-shape {
        animation-duration: 12s;
    }
    
    .enhanced-animations .floating-element {
        animation-duration: 4s;
    }
}

/* Accessibility Enhancements */
@media (prefers-contrast: high) {
    .text-shimmer,
    .neon-glow {
        background: none !important;
        color: #000000 !important;
        text-shadow: none !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    .morphing-shapes,
    .matrix-rain,
    .particles {
        display: none !important;
    }
}

/* Focus Animations */
.focus-ring {
    position: relative;
}

.focus-ring:focus-visible::after {
    content: '';
    position: absolute;
    top: -4px;
    left: -4px;
    right: -4px;
    bottom: -4px;
    border: 2px solid #3b82f6;
    border-radius: inherit;
    animation: focusPulse 2s infinite;
}

@keyframes focusPulse {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.5;
        transform: scale(1.05);
    }
}























// CME Unified Knowledge Assistant - Frontend JavaScript
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
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();

        if (!message || this.isTyping) return;

        // Get selected sources
        const selectedSources = this.getSelectedSources();
        if (selectedSources.length === 0) {
            this.showToast('Please select at least one data source', 'warning');
            return;
        }

        // Clear input and hide welcome
        messageInput.value = '';
        this.handleInputChange({ target: messageInput });
        this.hideWelcomeSection();

        // Add user message
        this.addMessage(message, 'user');
        
        // Show typing indicator
        this.showTypingIndicator();

        try {
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

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Hide typing indicator
            this.hideTypingIndicator();

            // Add assistant response
            this.addMessage(data.response, 'assistant');
            
            // Store in history
            this.messageHistory.push({
                user: message,
                assistant: data.response,
                sources: selectedSources,
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            this.addMessage(
                `I apologize, but I'm having trouble connecting to the knowledge systems right now. Please try again in a moment.\n\nError: ${error.message}`,
                'assistant'
            );
            this.showToast('Connection error - please try again', 'error');
        }
    }

    addMessage(content, sender) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = sender === 'user' ? 'U' : '🤖';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';
        
        // Format message content
        messageBubble.innerHTML = this.formatMessage(content);

        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = new Date().toLocaleTimeString();

        messageContent.appendChild(messageBubble);
        messageContent.appendChild(messageTime);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);

        chatMessages.appendChild(messageDiv);

        // Scroll to bottom
        this.scrollToBottom();

        // Add entrance animation
        setTimeout(() => {
            messageDiv.classList.add('fade-in');
        }, 100);
    }

    formatMessage(content) {
        // Convert markdown-like formatting to HTML
        let formatted = content
            // Bold text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Code blocks
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Links
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
            // Line breaks
            .replace(/\n/g, '<br>');

        // Handle lists
        formatted = formatted.replace(/^[•\-\*]\s+(.+)$/gm, '<li>$1</li>');
        formatted = formatted.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

        // Handle numbered lists
        formatted = formatted.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

        return formatted;
    }

    showTypingIndicator() {
        this.isTyping = true;
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.classList.add('show');
        }
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.isTyping = false;
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.classList.remove('show');
        }
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

        } catch (error) {
            console.error('Status check failed:', error);
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

        // Update morphing shapes based on mouse position
        const shapes = document.querySelectorAll('.morphing-shape');
        shapes.forEach((shape, index) => {
            const speed = 0.02 + (index * 0.01);
            const x = (xPercent - 50) * speed;
            const y = (yPercent - 50) * speed;
            
            const currentTransform = shape.style.transform || '';
            shape.style.transform = `${currentTransform} translate(${x}px, ${y}px)`;
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
            tooltip.style.opacity = tooltip.style.opacity === '1' ? '0' : '1';
            tooltip.style.visibility = tooltip.style.visibility === 'visible' ? 'hidden' : 'visible';
        }
    }

    setupSourceTracking() {
        // Track source changes for analytics
        const sourceCheckboxes = document.querySelectorAll('.source-option input[type="checkbox"]');
        sourceCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                const sources = this.getSelectedSources();
                console.log('Sources updated:', sources);
                
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

        // Announce status changes
        const originalCheckSystemStatus = this.checkSystemStatus;
        this.checkSystemStatus = async function() {
            await originalCheckSystemStatus.call(this);
            const statusText = document.querySelector('.status-text')?.textContent;
            if (statusText && this.announcer) {
                this.announcer.textContent = `System status: ${statusText}`;
            }
        };
    }

    detectHighContrast() {
        // Detect high contrast mode
        if (window.matchMedia('(prefers-contrast: high)').matches) {
            document.body.classList.add('high-contrast-mode');
            console.log('High contrast mode detected');
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

    // Enhanced message formatting with better accessibility
    formatMessage(content) {
        let formatted = content
            // Headers with proper hierarchy
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            // Bold text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic text
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Code blocks with language detection
            .replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
                const language = lang ? ` data-language="${lang}"` : '';
                return `<pre><code${language}>${this.escapeHtml(code.trim())}</code></pre>`;
            })
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Links with proper accessibility
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" aria-label="$1 (opens in new tab)">$1 <i class="fas fa-external-link-alt" aria-hidden="true"></i></a>')
            // Blockquotes
            .replace(/^> (.*$)/gm, '<blockquote>$1</blockquote>')
            // Line breaks
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');

        // Wrap in paragraphs if not already wrapped
        if (!formatted.startsWith('<')) {
            formatted = '<p>' + formatted + '</p>';
        }

        // Handle lists with proper structure
        formatted = this.formatLists(formatted);

        // Handle tables
        formatted = this.formatTables(formatted);

        return formatted;
    }

    formatLists(content) {
        // Unordered lists
        content = content.replace(/^[•\-\*] (.+)$/gm, '<li>$1</li>');
        content = content.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');

        // Ordered lists
        content = content.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
        
        return content;
    }

    formatTables(content) {
        // Simple table detection and formatting
        const lines = content.split('\n');
        let inTable = false;
        let tableRows = [];
        let result = [];

        lines.forEach(line => {
            if (line.includes('|') && line.split('|').length > 2) {
                if (!inTable) {
                    inTable = true;
                    tableRows = [];
                }
                tableRows.push(line);
            } else {
                if (inTable) {
                    result.push(this.buildTable(tableRows));
                    inTable = false;
                    tableRows = [];
                }
                result.push(line);
            }
        });

        if (inTable && tableRows.length > 0) {
            result.push(this.buildTable(tableRows));
        }

        return result.join('\n');
    }

    buildTable(rows) {
        if (rows.length === 0) return '';

        let table = '<table role="table"><thead><tr>';
        
        // Header row
        const headerCells = rows[0].split('|').map(cell => cell.trim()).filter(cell => cell);
        headerCells.forEach(cell => {
            table += `<th scope="col">${cell}</th>`;
        });
        table += '</tr></thead><tbody>';

        // Data rows (skip header and separator if exists)
        const dataRows = rows.slice(rows[1] && rows[1].includes('---') ? 2 : 1);
        dataRows.forEach(row => {
            const cells = row.split('|').map(cell => cell.trim()).filter(cell => cell);
            if (cells.length > 0) {
                table += '<tr>';
                cells.forEach(cell => {
                    table += `<td>${cell}</td>`;
                });
                table += '</tr>';
            }
        });

        table += '</tbody></table>';
        return table;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Performance monitoring
    measurePerformance(name, fn) {
        const start = performance.now();
        const result = fn();
        const end = performance.now();
        console.log(`${name} took ${end - start} milliseconds`);
        return result;
    }

    // Error handling and recovery
    handleError(error, context = '') {
        console.error(`Error in ${context}:`, error);
        
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
    // Check for modern browser features
    if (!window.fetch || !window.Promise) {
        alert('This application requires a modern browser. Please update your browser to continue.');
        return;
    }

    // Initialize the assistant
    window.assistant = new UnifiedAssistant();

    // Service worker registration for offline support (optional)
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js').catch(err => {
            console.log('Service worker registration failed:', err);
        });
    }

    // Add loading class to body initially
    document.body.classList.add('loading');
    
    // Remove loading class after initialization
    setTimeout(() => {
        document.body.classList.remove('loading');
        document.body.classList.add('loaded');
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
