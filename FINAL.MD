#!/usr/bin/env python3
import logging
import os
import sys
import json
import re
import time
import concurrent.futures
from datetime import datetime
from functools import lru_cache
import threading
import queue

# Web framework
from flask import Flask, render_template, request, jsonify
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

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")

# Confluence Configuration
CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL", "https://cmegroup.atlassian.net")
CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME", "lakshya.vijay@cmegroup.com")
CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN", "")
CONFLUENCE_SPACE = os.environ.get("CONFLUENCE_SPACE", "RE")
CONFLUENCE_ALL_SPACES = os.environ.get("CONFLUENCE_ALL_SPACES", "false").lower() == "true"

# Jira Configuration
JIRA_URL = os.environ.get("JIRA_URL", "https://cmegroup.atlassian.net/")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL", "lakshya.vijay@cmegroup.com")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_assistant.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("UnifiedAssistant")

# Performance settings
MAX_WORKERS = 5
CACHE_SIZE = 128
PAGE_CACHE_FILE = "page_cache.json"

class ContentExtractor:
    """Extract and process content from Confluence HTML - FULL VERSION."""
    
    def __init__(self, base_url, page_id=None, page_title=None):
        self.base_url = base_url
        self.page_id = page_id
        self.page_title = page_title
    
    def extract_content_from_html(self, html_content):
        """Extract text, tables, and images from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract basic metadata
        metadata = {
            "id": self.page_id,
            "title": self.page_title,
            "url": f"{self.base_url}/pages/viewpage.action?pageId={self.page_id}" if self.page_id else None,
            "type": "page",
            "labels": []
        }
        
        # Extract content sections
        sections = []
        
        # Extract text
        text_content = self._extract_text_content(soup)
        if text_content:
            sections.append({"type": "text", "content": text_content})
        
        # Extract tables
        tables = self._extract_tables(soup)
        if tables:
            sections.append({"type": "tables", "content": tables})
        
        # Extract images
        images = self._extract_images(soup)
        if images:
            sections.append({"type": "images", "content": images})
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(soup)
        if code_blocks:
            sections.append({"type": "code_blocks", "content": code_blocks})
        
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
        
        # Extract lists
        for list_elem in soup.find_all(['ul', 'ol']):
            list_items = []
            for li in list_elem.find_all('li'):
                text = li.get_text(strip=True)
                if text:
                    list_items.append(text)
            if list_items:
                list_type = "unordered" if list_elem.name == 'ul' else "ordered"
                text_content.append({"type": "list", "list_type": list_type, "items": list_items})
        
        return text_content
    
    def _extract_tables(self, soup):
        """Extract tables with improved handling."""
        tables = []
        
        for table in soup.find_all('table'):
            table_data = []
            
            # Extract headers
            headers = []
            thead = table.find('thead')
            if thead:
                for th in thead.find_all('th'):
                    headers.append(th.get_text(strip=True))
            elif table.find('tr'):
                # Check first row for headers
                first_row = table.find('tr')
                cells = first_row.find_all(['th', 'td'])
                if cells and all(cell.name == 'th' for cell in cells):
                    headers = [cell.get_text(strip=True) for cell in cells]
            
            # Extract rows
            tbody = table.find('tbody') or table
            for tr in tbody.find_all('tr'):
                row_data = []
                for td in tr.find_all(['td', 'th']):
                    cell_content = self._extract_cell_content(td)
                    row_data.append(cell_content)
                if row_data and not all(cell.name == 'th' for cell in tr.find_all(['td', 'th'])):
                    table_data.append(row_data)
            
            if table_data:
                formatted_table = self._format_table(headers, table_data)
                tables.append({
                    "headers": headers,
                    "rows": table_data,
                    "formatted": formatted_table
                })
        
        return tables
    
    def _extract_cell_content(self, cell):
        """Extract content from table cell including nested elements."""
        nested_table = cell.find('table')
        if nested_table:
            return "[Nested Table]"
        
        img = cell.find('img')
        if img:
            alt = img.get('alt', 'Image')
            return f"[Image: {alt}]" if alt else "[Image]"
        
        code = cell.find('code')
        if code:
            return f"[Code: {code.get_text(strip=True)}]"
        
        return cell.get_text(strip=True)
    
    def _format_table(self, headers, rows):
        """Format table for better readability."""
        if not rows:
            return ""
        
        col_widths = []
        if headers:
            col_widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row[:len(headers)]):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        else:
            if rows:
                col_widths = [0] * len(rows[0])
                for row in rows:
                    for i, cell in enumerate(row):
                        col_widths[i] = max(col_widths[i], len(str(cell)))
        
        formatted = []
        
        if headers:
            header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
            separator = "-|-".join("-" * w for w in col_widths)
            formatted.append(header_row)
            formatted.append(separator)
        
        for row in rows:
            formatted_row = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row[:len(col_widths)]))
            formatted.append(formatted_row)
        
        return "\n".join(formatted)
    
    def _extract_images(self, soup):
        """Extract images with improved metadata."""
        images = []
        
        for img in soup.find_all('img'):
            img_data = {
                "src": img.get('src', ''),
                "alt": img.get('alt', ''),
                "title": img.get('title', ''),
                "width": img.get('width', ''),
                "height": img.get('height', '')
            }
            
            parent = img.find_parent(['figure', 'div', 'p'])
            if parent:
                figcaption = parent.find('figcaption')
                if figcaption:
                    img_data['caption'] = figcaption.get_text(strip=True)
                
                prev_elem = parent.find_previous_sibling(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if prev_elem and len(prev_elem.get_text(strip=True)) < 200:
                    img_data['context'] = prev_elem.get_text(strip=True)
            
            images.append(img_data)
        
        return images
    
    def _extract_code_blocks(self, soup):
        """Extract code blocks with language detection."""
        code_blocks = []
        
        for pre in soup.find_all('pre'):
            code = pre.find('code')
            if code:
                code_class = code.get('class', [])
                lang = ""
                for cls in code_class:
                    if cls.startswith('language-'):
                        lang = cls.replace('language-', '')
                        break
                
                code_content = code.get_text(strip=True)
                if code_content:
                    code_blocks.append({
                        "language": lang,
                        "content": code_content
                    })
            else:
                code_content = pre.get_text(strip=True)
                if code_content:
                    code_blocks.append({
                        "language": "",
                        "content": code_content
                    })
        
        for code in soup.find_all('code'):
            if not code.find_parent('pre'):
                code_content = code.get_text(strip=True)
                if code_content:
                    code_blocks.append({
                        "language": "inline",
                        "content": code_content
                    })
        
        return code_blocks
    
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
                    elif item['type'] == 'list':
                        for list_item in item['items']:
                            prefix = "•" if item['list_type'] == 'unordered' else "1."
                            formatted.append(f"{prefix} {list_item}")
                    formatted.append("")
            
            elif section_type == 'tables':
                formatted.append("### Tables")
                for table in content:
                    formatted.append(table['formatted'])
                    formatted.append("")
            
            elif section_type == 'code_blocks':
                formatted.append("### Code Examples")
                for code in content:
                    lang = code.get('language', '')
                    formatted.append(f"```{lang}")
                    formatted.append(code['content'])
                    formatted.append("```")
                    formatted.append("")
            
            elif section_type == 'images':
                formatted.append("### Images")
                for img in content:
                    desc = img.get('alt') or img.get('title') or 'Image'
                    formatted.append(f"[IMAGE: {desc}]")
                    if img.get('caption'):
                        formatted.append(f"Caption: {img['caption']}")
                    if img.get('context'):
                        formatted.append(f"Context: {img['context']}")
                    formatted.append("")
        
        return "\n".join(formatted)

class ConfluenceClient:
    """Client for Confluence REST API operations - FULL VERSION."""
    
    def __init__(self, base_url, username, api_token):
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "RE-AI-Python-Agent"
        })
        self.cache = {}
        self.cache_lock = threading.Lock()
        logger.info(f"Initialized Confluence client for {self.base_url}")
    
    def test_connection(self):
        """Test connection to Confluence."""
        try:
            response = self.session.get(
                f"{self.base_url}/rest/api/content",
                params={"limit": 1},
                timeout=30,
                verify=False
            )
            if response.status_code == 200:
                logger.info("Connection to Confluence successful!")
                return True
            else:
                logger.error(f"Connection failed with status code: {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    @lru_cache(maxsize=CACHE_SIZE)
    def get_cached_request(self, url, params_str):
        """Cached version of GET requests."""
        try:
            params = json.loads(params_str)
            response = self.session.get(
                url,
                params=params,
                timeout=30,
                verify=False
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error in cached request to {url}: {str(e)}")
            return None
    
    def get_all_spaces(self):
        """Get all accessible spaces in Confluence."""
        spaces = []
        start = 0
        limit = 100
        
        while True:
            try:
                url = f"{self.base_url}/rest/api/space"
                params = {
                    "start": start,
                    "limit": limit,
                    "expand": "description"
                }
                
                response = self.session.get(url, params=params, timeout=30, verify=False)
                response.raise_for_status()
                
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    break
                
                for space in results:
                    spaces.append({
                        "key": space.get("key"),
                        "name": space.get("name"),
                        "type": space.get("type"),
                        "id": space.get("id")
                    })
                
                if "size" in data and data["size"] < limit:
                    break
                
                start += limit
                
            except Exception as e:
                logger.error(f"Error fetching spaces: {str(e)}")
                break
        
        logger.info(f"Found {len(spaces)} accessible spaces")
        return spaces
    
    def get_all_pages_in_space(self, space_key, batch_size=100):
        """Get all pages in a space using efficient pagination."""
        all_pages = []
        start = 0
        has_more = True
        
        logger.info(f"Fetching all pages from space: {space_key}")
        
        while has_more:
            try:
                params = {
                    "spaceKey": space_key,
                    "expand": "history",
                    "limit": batch_size,
                    "start": start
                }
                
                params_str = json.dumps(params, sort_keys=True)
                response_data = self.get_cached_request(f"{self.base_url}/rest/api/content", params_str)
                
                if not response_data:
                    break
                
                results = response_data.get("results", [])
                all_pages.extend(results)
                
                if "size" in response_data and "limit" in response_data:
                    if response_data["size"] < response_data["limit"]:
                        has_more = False
                    else:
                        start += batch_size
                else:
                    has_more = False
                
                logger.info(f"Fetched {len(results)} pages, total so far: {len(all_pages)}")
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error fetching pages: {str(e)}")
                break
        
        logger.info(f"Successfully fetched {len(all_pages)} pages from space {space_key}")
        return all_pages
    
    def get_page_content(self, page_id, expand=None):
        """Get content of a page."""
        try:
            cache_key = f"page_content_{page_id}_{expand}"
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]
            
            url = f"{self.base_url}/rest/api/content/{page_id}"
            params = {}
            if expand:
                params["expand"] = expand
            
            response = self.session.get(url, params=params, timeout=30, verify=False)
            
            if response.status_code == 200:
                page_data = response.json()
                
                metadata = {
                    "id": page_data.get("id"),
                    "title": page_data.get("title"),
                    "type": page_data.get("type"),
                    "url": f"{self.base_url}/pages/viewpage.action?pageId={page_data.get('id')}",
                    "space": page_data.get("space", {}).get("key", ""),
                    "labels": [label.get("name") for label in page_data.get("metadata", {}).get("labels", {}).get("results", [])]
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
                    "content": formatted_content,
                    "raw_html": html_content
                }
                
                with self.cache_lock:
                    self.cache[cache_key] = result
                
                return result
            else:
                logger.warning(f"Failed to get content for page {page_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None

class SafeJiraClient:
    """Ultra-safe Jira client with bulletproof error handling - FULL VERSION."""
    
    def __init__(self):
        self.base_url = JIRA_URL.rstrip('/')
        self.auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self._metadata = None
        logger.info("Jira client initialized")
    
    def safe_get(self, obj, key, default=None):
        """Ultra-safe getter that never fails."""
        try:
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            if hasattr(obj, key):
                return getattr(obj, key, default)
            return default
        except:
            return default
    
    def make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make safe HTTP request."""
        url = f"{self.base_url}/rest/api/3/{endpoint}"
        
        try:
            logger.info(f"Making request to: {endpoint}")
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
            else:
                logger.warning(f"Request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None
    
    def test_connection(self):
        """Test connection to Jira."""
        try:
            result = self.make_request('myself')
            if result:
                logger.info("Connection to Jira successful!")
                return True
            else:
                logger.error("Jira connection failed")
                return False
        except Exception as e:
            logger.error(f"Jira connection test failed: {e}")
            return False
    
    def get_jira_metadata(self) -> Dict:
        """Get Jira metadata for AI context."""
        if self._metadata:
            return self._metadata
        
        logger.info("Loading Jira metadata...")
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
            
            # Get sample users
            users = self.make_request('users/search', {'maxResults': 50}) or []
            for user in users:
                if user:
                    metadata['users'].append({
                        'displayName': self.safe_get(user, 'displayName', ''),
                        'emailAddress': self.safe_get(user, 'emailAddress', '')
                    })
            
            # Get statuses
            statuses = self.make_request('status') or []
            for status in statuses:
                if status:
                    metadata['statuses'].append(self.safe_get(status, 'name', ''))
            
            # Get priorities
            priorities = self.make_request('priority') or []
            for priority in priorities:
                if priority:
                    metadata['priorities'].append(self.safe_get(priority, 'name', ''))
            
            # Get issue types
            issue_types = self.make_request('issuetype') or []
            for issue_type in issue_types:
                if issue_type:
                    metadata['issue_types'].append(self.safe_get(issue_type, 'name', ''))
            
            self._metadata = metadata
            logger.info(f"Loaded metadata: {len(metadata['projects'])} projects, {len(metadata['users'])} users")
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
        
        return metadata
    
    def execute_jql(self, jql: str, max_results: int = 100) -> List[Dict]:
        """Execute JQL query safely."""
        if not jql or not jql.strip():
            return []
        
        try:
            logger.info(f"Executing JQL: {jql}")
            params = {
                "jql": jql,
                "maxResults": max_results,
                "fields": "*all"
            }
            
            result = self.make_request('search', params)
            if result:
                issues = self.safe_get(result, 'issues', [])
                logger.info(f"JQL returned {len(issues)} issues")
                return [issue for issue in issues if issue]
            
            return []
            
        except Exception as e:
            logger.error(f"JQL execution error: {e}")
            return []

class AIJiraAnalyzer:
    """AI-powered Jira query analyzer and response generator - FULL VERSION."""
    
    def __init__(self):
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
        logger.info("AI analyzer initialized")
    
    def analyze_query_and_build_jql(self, user_query: str, metadata: Dict) -> List[str]:
        """Let AI analyze the query and build perfect JQL."""
        
        context = f"""
AVAILABLE JIRA PROJECTS: {[p['key'] + ' (' + p['name'] + ')' for p in metadata.get('projects', [])]}
AVAILABLE STATUSES: {metadata.get('statuses', [])}
AVAILABLE PRIORITIES: {metadata.get('priorities', [])}
AVAILABLE ISSUE TYPES: {metadata.get('issue_types', [])}
SAMPLE USERS: {[u['displayName'] + ' (' + u['emailAddress'] + ')' for u in metadata.get('users', [])[:10]]}
"""

        prompt = f"""You are a Jira JQL expert. Analyze this user query and create 3-5 different JQL queries that will find what they're looking for.

USER QUERY: "{user_query}"

JIRA CONTEXT:
{context}

IMPORTANT JQL GUIDELINES:
1. Use text search: text ~ "term" (searches all fields)
2. Use summary search: summary ~ "term" 
3. Use description search: description ~ "term"
4. Use comment search: comment ~ "term"
5. For users, try both displayName and email patterns
6. For dates, use formats like: created >= -7d, updated >= "2024-01-01"
7. Always end with: ORDER BY updated DESC
8. Use AND, OR operators appropriately
9. For project search: project = "KEY" or project in ("KEY1", "KEY2")
10. For assignee: assignee ~ "name" or assignee = "email"

EXAMPLES:
- "rollout restart issues" → text ~ "rollout restart" OR summary ~ "rollout restart"
- "tickets assigned to john" → assignee ~ "john" OR assignee ~ "John"
- "high priority bugs" → priority = "High" AND issuetype = "Bug"
- "bamps project tickets" → project = "BAMPS"
- "recent issues" → updated >= -7d

CREATE MULTIPLE JQL QUERIES:
1. Main search query (most specific)
2. Broader search query (less specific) 
3. Fallback search query (very broad)
4. Alternative search approach
5. Recent items query (if no specific criteria)

OUTPUT FORMAT - Return only JQL queries, one per line:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000,
                )
            )
            
            if hasattr(response, 'text'):
                jql_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                jql_text = response.candidates[0].text
            else:
                return self._fallback_jql(user_query)
            
            # Parse JQL queries
            jql_queries = []
            for line in jql_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    # Clean up the line
                    line = line.replace('```', '').replace('sql', '').replace('jql', '')
                    line = line.strip('- ').strip('* ').strip('1. ').strip('2. ').strip('3. ').strip('4. ').strip('5. ')
                    
                    if len(line) > 10 and ('=' in line or '~' in line or 'ORDER BY' in line):
                        jql_queries.append(line.strip())
            
            if jql_queries:
                logger.info(f"AI generated {len(jql_queries)} JQL queries")
                return jql_queries[:5]  # Limit to 5 queries
            else:
                return self._fallback_jql(user_query)
                
        except Exception as e:
            logger.error(f"AI JQL generation error: {e}")
            return self._fallback_jql(user_query)
    
    def _fallback_jql(self, user_query: str) -> List[str]:
        """Simple fallback JQL if AI fails."""
        words = [w for w in user_query.lower().split() if len(w) > 2]
        
        queries = []
        if words:
            main_terms = ' '.join(words[:3])
            queries.append(f'text ~ "{main_terms}" ORDER BY updated DESC')
            
            for word in words[:2]:
                queries.append(f'summary ~ "{word}" ORDER BY updated DESC')
        
        queries.append('updated >= -30d ORDER BY updated DESC')
        
        return queries

class GeminiAssistant:
    """Class for interacting with Gemini models via Vertex AI - FULL VERSION."""
    
    def __init__(self):
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
        logger.info(f"Initialized Gemini Assistant with model: {MODEL_NAME}")
    
    def generate_response(self, prompt, RE_context=None):
        """Generate response from Gemini based on prompt and RE context."""
        logger.info(f"Generating response for prompt: {prompt}")
        
        try:
            system_prompt = """
You are the friendly CME Unified Knowledge Assistant, an expert on both Confluence documentation and Jira ticket management.

Your personality:
- Conversational and approachable - use a casual, helpful tone while maintaining workplace professionalism
- Explain technical concepts in plain language, as if speaking to a colleague
- Use simple analogies and examples to clarify complex ideas
- Add occasional light humor where appropriate to make the conversation engaging
- Be concise but thorough - focus on answering the question directly first, then add helpful context

Your expertise:
- Deep knowledge of Confluence documentation, procedures, and knowledge base content
- Understanding of Jira tickets, issues, project tracking, and workflow management
- Ability to correlate information between documentation and actual implementation
- Expert in interpreting both structured data (tables, code) and unstructured content (text, procedures)

When answering:
1. Directly address the user's question first
2. Provide practical, actionable information when possible
3. Format tables and structured data clearly to enhance readability
4. Use bullet points or numbered lists for steps or multiple items
5. Reference specific examples from the documentation when available
6. Include relevant ticket links when discussing Jira issues
7. Acknowledge any limitations in the available information

Remember to maintain a balance between being friendly and professional - you're a helpful colleague, not a formal technical document.
"""
            
            full_prompt = system_prompt + "\n\n"
            
            if RE_context:
                source_pages = []
                
                for line in RE_context.split('\n'):
                    if line.startswith("URL:"):
                        source_pages.append(line.replace("URL:", "").strip())
                
                if len(RE_context) > 50000:
                    logger.warning(f"Context too large ({len(RE_context)} chars), trimming...")
                    paragraphs = RE_context.split('\n\n')
                    trimmed_context = ""
                    for para in paragraphs:
                        if len(trimmed_context) + len(para) + 2 < 50000:
                            trimmed_context += para + "\n\n"
                        else:
                            break
                    RE_context = trimmed_context
                
                if source_pages:
                    full_prompt += f"CONTEXT INFORMATION:\n{RE_context}\n\n"
                    full_prompt += f"SOURCE PAGES: {', '.join(source_pages[:5])}\n\n"
                else:
                    full_prompt += f"CONTEXT INFORMATION:\n{RE_context}\n\n"
            
            full_prompt += f"USER QUESTION: {prompt}\n\nResponse:"
            
            generation_config = GenerationConfig(
                temperature=0.3,
                top_p=0.95,
            )
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            if response.candidates and response.candidates[0].text:
                response_text = response.candidates[0].text.strip()
                logger.info(f"Successfully generated response ({len(response_text)} chars)")
                
                if RE_context and source_pages:
                    response_text += "\n\n---\n**Sources:**\n"
                    for i, url in enumerate(source_pages[:5], 1):
                        response_text += f"{i}. {url}\n"
                
                return response_text
            else:
                logger.warning("No response generated from Gemini")
                return "I couldn't find a specific answer to that question in our documentation. Could you try rephrasing, or maybe I can help you find what you're looking for?"
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I ran into a technical issue while looking that up. Let me know if you'd like to try a different question or approach."

class REAssistant:
    """Main class that coordinates between Confluence and Jira - FULL VERSION."""
    
    def __init__(self, confluence_url, confluence_username, confluence_api_token, space_key=None, all_spaces=False):
        self.confluence = ConfluenceClient(confluence_url, confluence_username, confluence_api_token)
        self.jira = SafeJiraClient()
        self.gemini = GeminiAssistant()
        self.jira_analyzer = AIJiraAnalyzer()
        self.space_key = space_key
        self.all_spaces = all_spaces
        self.space_pages = {}
        self.page_content_cache = {}
        logger.info(f"Initialized RE Assistant targeting {'all spaces' if all_spaces else f'space: {space_key}'}")
    
    def initialize(self):
        """Initialize by testing connections and gathering initial content."""
        confluence_ok = self.confluence.test_connection()
        jira_ok = self.jira.test_connection()
        
        if not confluence_ok and not jira_ok:
            logger.error("Failed to connect to both Confluence and Jira. Check credentials and URLs.")
            return False
        
        if confluence_ok:
            logger.info("Loading Confluence space content...")
            self.load_space_content()
        
        return True
    
    def load_space_content(self):
        """Load metadata for all pages in the specified space(s)."""
        cache_file = f"{'all_spaces' if self.all_spaces else self.space_key}_pages_cache.json"
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    if cached_data.get('timestamp', 0) > time.time() - 86400:  # 24 hours
                        self.space_pages = cached_data.get('pages', {})
                        logger.info(f"Loaded {sum(len(pages) for pages in self.space_pages.values())} pages from cache")
                        return
            except Exception as e:
                logger.warning(f"Error reading cache file: {str(e)}")
        
        if self.all_spaces:
            spaces = self.confluence.get_all_spaces()
            logger.info(f"Found {len(spaces)} accessible spaces")
            
            for space in spaces:
                space_key = space['key']
                logger.info(f"Fetching pages from space: {space_key} ({space['name']})")
                
                pages = self.confluence.get_all_pages_in_space(space_key)
                if pages:
                    self.space_pages[space_key] = pages
                    logger.info(f"Loaded {len(pages)} pages from space {space_key}")
                
                time.sleep(0.5)
        else:
            if not self.space_key:
                logger.error("No space key specified. Please provide a space key.")
                return
            
            pages = self.confluence.get_all_pages_in_space(self.space_key)
            if pages:
                self.space_pages[self.space_key] = pages
                logger.info(f"Loaded {len(pages)} pages from space {self.space_key}")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'pages': self.space_pages
                }, f)
            logger.info(f"Cached {sum(len(pages) for pages in self.space_pages.values())} pages to {cache_file}")
        except Exception as e:
            logger.warning(f"Error writing cache file: {str(e)}")
    
    def fetch_page_content_batch(self, pages):
        """Fetch content for a batch of pages in parallel."""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_page = {
                executor.submit(self.confluence.get_page_content, page["id"], "body.storage,metadata.labels"): page 
                for page in pages
            }
            
            for future in concurrent.futures.as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    content = future.result()
                    if content:
                        results[page["id"]] = content
                except Exception as e:
                    logger.error(f"Error in page content fetching task for {page['id']}: {str(e)}")
        
        return results
    
    def extract_relevant_content(self, query):
        """Extract content from pages that is most relevant to the query."""
        if not self.space_pages:
            return "No pages found in the specified Confluence space(s)."
        
        all_pages = []
        for space_key, pages in self.space_pages.items():
            for page in pages:
                page_with_space = page.copy()
                page_with_space['space_key'] = space_key
                all_pages.append(page_with_space)
        
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        domain_terms = {
            "RE", "SRE", "LOGS", "MAINTENANCE", "API", "REST", "DATABASE",
            "VIEW", "TABLE", "FIELD", "ENDPOINT", "MAPPING", "SCHEMA", "BAMPS",
            "ROLLOUT", "RESTART", "SYSTEM", "ERROR", "ISSUE", "BUG", "TICKET"
        }
        
        if len(query_words) < 3:
            query_words.update(domain_terms)
        
        candidates = []
        for page in all_pages:
            title = page.get("title", "").lower()
            title_score = sum(1 for word in query_words if word in title)
            
            if title_score > 0 or any(term in title for term in domain_terms):
                candidates.append((page, title_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:20]
        
        logger.info(f"Selected {len(top_candidates)} candidate pages for detailed analysis")
        
        if not top_candidates:
            recent_pages = sorted(all_pages, 
                                key=lambda p: p.get("history", {}).get("lastUpdated", {}).get("when", "2000-01-01"),
                                reverse=True)[:10]
            top_candidates = [(p, 0) for p in recent_pages]
        
        page_dict = {p[0]["id"]: p[0] for p in top_candidates}
        candidate_pages = list(page_dict.values())
        
        page_contents = self.fetch_page_content_batch(candidate_pages)
        
        scored_pages = []
        for page_id, content in page_contents.items():
            page = page_dict[page_id]
            page_text = content.get("content", "").lower()
            title = page.get("title", "").lower()
            
            score = 0
            
            for word in query_words:
                word_count = page_text.count(word)
                score += word_count * 0.1
            
            if word in title:
                score += 5
            
            query_lower = query.lower()
            if query_lower in page_text:
                score += 50
            else:
                for phrase_len in range(2, 5):
                    if len(query_words) >= phrase_len:
                        query_phrases = [" ".join(query_lower.split()[i:i+phrase_len]) 
                                       for i in range(len(query_lower.split()) - phrase_len + 1)]
                        for phrase in query_phrases:
                            if len(phrase.split()) >= 2 and phrase in page_text:
                                score += 3 * page_text.count(phrase)
            
            if "TABLE" in page_text:
                score += 1.5
            if "```" in page_text:
                score += 1.3
            
            code_terms = ["code", "example", "implementation", "syntax", "usage"]
            if any(term in query_lower for term in code_terms) and "```" in page_text:
                score += 5
            
            image_terms = ["image", "diagram", "screenshot", "picture"]
            if any(term in query_lower for term in image_terms) and "[IMAGE" in page_text:
                score += 3
            
            scored_pages.append((page, content, score))
        
        scored_pages.sort(key=lambda x: x[2], reverse=True)
        top_pages = scored_pages[:6]
        
        logger.info(f"Selected {len(top_pages)} most relevant pages")
        
        if not top_pages:
            return "I couldn't find any relevant information in the Confluence space(s)."
        
        relevant_content = []
        source_urls = []
        
        for page, content, score in top_pages:
            page_content = content.get("content", "")
            page_url = content.get("metadata", {}).get("url", "")
            page_title = page.get("title", "")
            space_key = page.get("space_key", "")
            
            if page_url:
                source_urls.append(page_url)
            
            sections = re.split(r'#+\s+', page_content)
            
            if len(sections) <= 2:
                sections = page_content.split('\n\n')
            
            section_scores = []
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                
                section_lower = section.lower()
                section_score = 0
                
                for word in query_words:
                    freq = section_lower.count(word)
                    section_score += freq * 0.5
                
                if query_lower in section_lower:
                    section_score += 10
                
                for phrase_len in range(2, 5):
                    if len(query_words) >= phrase_len:
                        query_phrases = [" ".join(query_lower.split()[i:i+phrase_len]) 
                                       for i in range(len(query_lower.split()) - phrase_len + 1)]
                        for phrase in query_phrases:
                            if phrase in section_lower:
                                section_score += 2
                
                if "TABLE" in section:
                    section_score *= 1.5
                if "```" in section:
                    section_score *= 1.3
                
                section_scores.append((i, section, section_score))
            
            section_scores.sort(key=lambda x: x[2], reverse=True)
            top_sections = section_scores[:3]
            
            if top_sections:
                ordered_sections = sorted(top_sections, key=lambda x: x[0])
                
                if ordered_sections:
                    content_block = f"--- FROM: {page_title} (Space: {space_key}) ---\n\n"
                    
                    first_section = ordered_sections[0][1].strip()
                    if not first_section.startswith('#'):
                        content_block += f"# {page_title}\n\n"
                    
                    for _, section, _ in ordered_sections:
                        cleaned_section = re.sub(r'\n{3,}', '\n\n', section.strip())
                        content_block += cleaned_section + "\n\n"
                    
                    content_block += f"Source: {page_url}\n"
                    relevant_content.append(content_block)
        
        combined_content = "\n\n".join(relevant_content)
        
        if source_urls:
            combined_content = f"SOURCES: {', '.join(source_urls[:5])}\n\n" + combined_content
        
        return combined_content
    
    def search_jira_tickets(self, question):
        """Search Jira tickets using AI-powered JQL generation."""
        logger.info(f"Searching Jira for: {question}")
        
        try:
            metadata = self.jira.get_jira_metadata()
            jql_queries = self.jira_analyzer.analyze_query_and_build_jql(question, metadata)
            
            all_results = []
            for jql in jql_queries:
                results = self.jira.execute_jql(jql, max_results=50)
                all_results.extend(results)
            
            unique_results = []
            seen_keys = set()
            for issue in all_results:
                if issue:
                    key = issue.get('key')
                    if key and key not in seen_keys:
                        seen_keys.add(key)
                        unique_results.append(issue)
            
            logger.info(f"Found {len(unique_results)} unique Jira issues")
            return unique_results
            
        except Exception as e:
            logger.error(f"Error searching Jira: {e}")
            return []
    
    def answer_question(self, question, selected_sources):
        """Answer a question using Confluence content and/or Jira tickets."""
        logger.info(f"Processing question: {question} with sources: {selected_sources}")
        
        confluence_content = ""
        jira_content = ""
        
        # Search Confluence if selected
        if "confluence" in selected_sources or "both" in selected_sources:
            try:
                confluence_content = self.extract_relevant_content(question)
                logger.info(f"Confluence content extracted: {len(confluence_content)} chars")
            except Exception as e:
                logger.error(f"Error extracting Confluence content: {e}")
        
        # Search Jira if selected
        if "jira" in selected_sources or "both" in selected_sources:
            try:
                jira_results = self.search_jira_tickets(question)
                if jira_results:
                    jira_summary = f"JIRA RESULTS ({len(jira_results)} tickets found):\n"
                    for i, issue in enumerate(jira_results[:10], 1):
                        key = issue.get('key', 'Unknown')
                        fields = issue.get('fields', {})
                        summary = fields.get('summary', 'No summary')[:100]
                        status = (fields.get('status') or {}).get('name', 'Unknown')
                        priority = (fields.get('priority') or {}).get('name', 'Unknown')
                        assignee = (fields.get('assignee') or {}).get('displayName', 'Unassigned')
                        
                        jira_summary += f"{i}. [{key}](https://cmegroup.atlassian.net/browse/{key}) - {summary}\n"
                        jira_summary += f"   Status: {status} | Priority: {priority} | Assignee: {assignee}\n\n"
                    
                    jira_content = jira_summary
                    logger.info(f"Jira content prepared: {len(jira_content)} chars")
            except Exception as e:
                logger.error(f"Error searching Jira: {e}")
        
        # Combine content for AI
        combined_context = ""
        if confluence_content and jira_content:
            combined_context = f"CONFLUENCE DOCUMENTATION:\n{confluence_content}\n\n{jira_content}"
        elif confluence_content:
            combined_context = f"CONFLUENCE DOCUMENTATION:\n{confluence_content}"
        elif jira_content:
            combined_context = jira_content
        
        # Generate AI response
        if combined_context:
            response = self.gemini.generate_response(question, combined_context)
        else:
            response = "I couldn't find relevant information in the selected sources. Please try a different query or check your source selection."
        
        return response

# Flask Application
app = Flask(__name__)
CORS(app)

# Initialize the assistant
logger.info("🚀 Starting CME Unified Knowledge Assistant...")
assistant = REAssistant(
    CONFLUENCE_URL, 
    CONFLUENCE_USERNAME, 
    CONFLUENCE_API_TOKEN, 
    space_key=CONFLUENCE_SPACE,
    all_spaces=CONFLUENCE_ALL_SPACES
)

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests with detailed logging."""
    request_start = time.time()
    logger.info("=" * 60)
    logger.info("📨 NEW CHAT REQUEST RECEIVED")
    logger.info("=" * 60)
    
    try:
        data = request.get_json()
        if not data:
            logger.warning("❌ No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
        
        user_query = data.get('message', '').strip()
        selected_sources = data.get('sources', [])
        
        logger.info(f"📝 USER QUERY: '{user_query}'")
        logger.info(f"📊 SELECTED SOURCES: {selected_sources}")
        
        if not user_query:
            logger.warning("❌ Empty message received")
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
            
            logger.info("✅ RESPONDED WITH GREETING")
            return jsonify({'response': response})
        
        if any(word in query_lower for word in ['bye', 'goodbye', 'thanks']):
            response = "👋 **Goodbye!** Feel free to come back anytime for insights across CME's knowledge systems! 🚀"
            logger.info("✅ RESPONDED WITH GOODBYE")
            return jsonify({'response': response})
        
        # Process the actual query
        logger.info("🔄 PROCESSING QUERY WITH ASSISTANT...")
        response = assistant.answer_question(user_query, selected_sources)
        
        request_time = time.time() - request_start
        logger.info(f"✅ CHAT REQUEST COMPLETED in {request_time:.2f} seconds")
        logger.info(f"📝 RESPONSE LENGTH: {len(response)} characters")
        logger.info("=" * 60)
        
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"❌ ERROR IN CHAT ENDPOINT: {e}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/status')
def status():
    """Check system status."""
    logger.info("📊 STATUS CHECK REQUESTED")
    try:
        confluence_status = assistant.confluence.test_connection() if assistant.confluence else False
        jira_status = assistant.jira.test_connection() if assistant.jira else False
        confluence_pages = sum(len(pages) for pages in assistant.space_pages.values())
        
        status_data = {
            'confluence': confluence_status,
            'jira': jira_status,
            'confluence_pages': confluence_pages,
            'status': 'healthy' if confluence_status or jira_status else 'degraded'
        }
        
        logger.info(f"📊 STATUS: {status_data}")
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f"❌ ERROR IN STATUS ENDPOINT: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 ERROR: {request.url}")
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 ERROR: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting CME Unified Knowledge Assistant...")
    print(f"🔗 Confluence: {CONFLUENCE_URL}")
    print(f"🎫 Jira: {JIRA_URL}")
    print(f"🤖 AI Model: {MODEL_NAME}")
    print(f"📊 Confluence Token: {'✅ SET' if CONFLUENCE_API_TOKEN else '❌ MISSING'}")
    print(f"📊 Jira Token: {'✅ SET' if JIRA_API_TOKEN else '❌ MISSING'}")
    print("=" * 50)
    
    # Initialize the assistant
    if not assistant.initialize():
        print("❌ Failed to initialize assistant. Check your configuration.")
        sys.exit(1)
    
    print("✅ Assistant initialized successfully!")
    print("🌐 Starting Flask server...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
