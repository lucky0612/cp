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
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import threading
import queue
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-pro-preview-05-06")
CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL", "https://cmegroup.atlassian.net")
CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME", "lalshya.vijay@cmegroup.com")
CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN", "")
CONFLUENCE_SPACE = os.environ.get("CONFLUENCE_SPACE", "RE")
CONFLUENCE_ALL_SPACES = os.environ.get("CONFLUENCE_ALL_SPACES", "false").lower() == "true"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("confluence_assistant.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("REAssistant")

# Performance settings
MAX_WORKERS = 5
CACHE_SIZE = 128
PAGE_CACHE_FILE = "page_cache.json"

class ContentExtractor:
    """Extract and process content from Confluence HTML."""
    
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
        
        # Extract iframes (embedded content)
        iframes = self._extract_iframes(soup)
        if iframes:
            sections.append({"type": "iframes", "content": iframes})
        
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
                    # Handle cells with nested content
                    cell_content = self._extract_cell_content(td)
                    row_data.append(cell_content)
                if row_data and not all(cell.name == 'th' for cell in tr.find_all(['td', 'th'])):
                    table_data.append(row_data)
            
            if table_data:
                # Format table
                formatted_table = self._format_table(headers, table_data)
                tables.append({
                    "headers": headers,
                    "rows": table_data,
                    "formatted": formatted_table
                })
        
        return tables
    
    def _extract_cell_content(self, cell):
        """Extract content from table cell including nested elements."""
        # Check for nested tables
        nested_table = cell.find('table')
        if nested_table:
            return "[Nested Table]"
        
        # Check for images
        img = cell.find('img')
        if img:
            src = img.get('src', '')
            alt = img.get('alt', 'Image')
            return f"[Image: {alt}]" if alt else "[Image]"
        
        # Check for code
        code = cell.find('code')
        if code:
            return f"[Code: {code.get_text(strip=True)}]"
        
        # Default text extraction
        return cell.get_text(strip=True)
    
    def _format_table(self, headers, rows):
        """Format table for better readability."""
        if not rows:
            return ""
        
        # Calculate column widths
        col_widths = []
        if headers:
            col_widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row[:len(headers)]):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        else:
            # No headers, calculate from rows
            if rows:
                col_widths = [0] * len(rows[0])
                for row in rows:
                    for i, cell in enumerate(row):
                        col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Format table
        formatted = []
        
        # Add headers if available
        if headers:
            header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
            separator = "-|-".join("-" * w for w in col_widths)
            formatted.append(header_row)
            formatted.append(separator)
        
        # Add rows
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
            
            # Get parent context
            parent = img.find_parent(['figure', 'div', 'p'])
            if parent:
                # Check for caption
                figcaption = parent.find('figcaption')
                if figcaption:
                    img_data['caption'] = figcaption.get_text(strip=True)
                
                # Get surrounding text
                prev_elem = parent.find_previous_sibling(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if prev_elem and len(prev_elem.get_text(strip=True)) < 200:
                    img_data['context'] = prev_elem.get_text(strip=True)
            
            images.append(img_data)
        
        return images
    
    def _extract_code_blocks(self, soup):
        """Extract code blocks with language detection."""
        code_blocks = []
        
        # Find pre tags with code
        for pre in soup.find_all('pre'):
            code = pre.find('code')
            if code:
                # Try to detect language
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
                # Just pre tag without code tag
                code_content = pre.get_text(strip=True)
                if code_content:
                    code_blocks.append({
                        "language": "",
                        "content": code_content
                    })
        
        # Find inline code
        for code in soup.find_all('code'):
            if not code.find_parent('pre'):
                code_content = code.get_text(strip=True)
                if code_content:
                    code_blocks.append({
                        "language": "inline",
                        "content": code_content
                    })
        
        return code_blocks
    
    def _extract_iframes(self, soup):
        """Extract iframe content with better handling."""
        iframes = []
        
        for iframe in soup.find_all('iframe'):
            iframe_data = {
                "src": iframe.get('src', ''),
                "title": iframe.get('title', ''),
                "width": iframe.get('width', ''),
                "height": iframe.get('height', ''),
                "type": "iframe"
            }
            
            # Try to identify iframe content type
            src = iframe_data['src']
            if 'youtube.com' in src or 'youtu.be' in src:
                iframe_data['content_type'] = 'video'
                iframe_data['description'] = 'YouTube video'
            elif 'vimeo.com' in src:
                iframe_data['content_type'] = 'video'
                iframe_data['description'] = 'Vimeo video'
            elif 'docs.google.com' in src:
                iframe_data['content_type'] = 'document'
                iframe_data['description'] = 'Google Document'
            elif any(ext in src for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']):
                iframe_data['content_type'] = 'document'
                iframe_data['description'] = 'Embedded document'
            else:
                iframe_data['content_type'] = 'unknown'
                iframe_data['description'] = 'Embedded content'
            
            iframes.append(iframe_data)
        
        return iframes
    
    def format_for_context(self, extracted_content):
        """Format extracted content for AI context."""
        formatted = []
        
        # Add metadata
        metadata = extracted_content.get('metadata', {})
        if metadata.get('title'):
            formatted.append(f"## {metadata['title']}")
        if metadata.get('url'):
            formatted.append(f"URL: {metadata['url']}")
        formatted.append("")
        
        # Process sections
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
            
            elif section_type == 'iframes':
                formatted.append("### Embedded Content")
                for iframe in content:
                    formatted.append(f"[{iframe['description']}]")
                    if iframe.get('title'):
                        formatted.append(f"Title: {iframe['title']}")
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
                
                # Check if there are more results
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
                
                # Extract metadata
                metadata = {
                    "id": page_data.get("id"),
                    "title": page_data.get("title"),
                    "type": page_data.get("type"),
                    "url": f"{self.base_url}/pages/viewpage.action?pageId={page_data.get('id')}",
                    "space": page_data.get("space", {}).get("key", ""),
                    "labels": [label.get("name") for label in page_data.get("metadata", {}).get("labels", {}).get("results", [])]
                }
                
                # Get raw content
                body = page_data.get("body", {})
                storage = body.get("storage", {})
                html_content = storage.get("value", "")
                
                # Process with ContentExtractor
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
                
                # Cache result
                with self.cache_lock:
                    self.cache[cache_key] = result
                
                return result
            else:
                logger.warning(f"Failed to get content for page {page_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None
    
    def get_content_by_id(self, content_id, expand=None):
        """Get content by ID with expansion parameters."""
        try:
            cache_key = f"content_{content_id}_{expand}"
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]
            
            params = {}
            if expand:
                params["expand"] = expand
            
            params_str = json.dumps(params, sort_keys=True)
            response_text = self.get_cached_request(f"{self.base_url}/rest/api/content/{content_id}", params_str)
            
            if not response_text:
                logger.warning(f"Empty response for content ID: {content_id}")
                return None
            
            content = json.loads(response_text)
            logger.info(f"Successfully retrieved content: {content.get('title', 'Unknown title')}")
            
            # Cache result
            with self.cache_lock:
                self.cache[cache_key] = content
            
            return content
            
        except Exception as e:
            logger.error(f"Error getting content by ID {content_id}: {str(e)}")
            return None

class GeminiAssistant:
    """Class for interacting with Gemini models via Vertex AI."""
    
    def __init__(self):
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
        logger.info(f"Initialized Gemini Assistant with model: {MODEL_NAME}")
    
    def generate_response(self, prompt, RE_context=None):
        """Generate response from Gemini based on prompt and RE context."""
        logger.info(f"Generating response for prompt: {prompt}")
        
        try:
            # Create system prompt
            system_prompt = """
You are the friendly RE Assistant, an expert on mapping database views to REST APIs.

Your personality:
- Conversational and approachable - use a casual, helpful tone while maintaining workplace professionalism
- Explain technical concepts in plain language, as if speaking to a colleague
- Use simple analogies and examples to clarify complex ideas
- Add occasional light humor where appropriate to make the conversation engaging
- Be concise but thorough - focus on answering the question directly first, then add helpful context

Your expertise:
- Deep knowledge of the RE database system, its views, and corresponding API endpoints
- Understanding database-to-API mapping patterns and best practices
- Awareness of how applications integrate with RE's REST APIs
- Expert in interpreting table structures, field mappings, and API parameters

When answering:
1. Directly address the user's question first
2. Provide practical, actionable information when possible
3. Format tables and structured data clearly to enhance readability
4. Use bullet points or numbered lists for steps or multiple items
5. Reference specific examples from the documentation when available
6. Acknowledge any limitations in the available information

Remember to maintain a balance between being friendly and professional - you're a helpful colleague, not a formal technical document.
"""
            
            # Craft full prompt with context
            full_prompt = system_prompt + "\n\n"
            
            if RE_context:
                # Add source tracking
                source_pages = []
                
                # Extract page URLs from context
                for line in RE_context.split('\n'):
                    if line.startswith("URL:"):
                        source_pages.append(line.replace("URL:", "").strip())
                
                # Check context size
                if len(RE_context) > 50000:
                    logger.warning(f"Context too large ({len(RE_context)} chars), trimming...")
                    # Split context into paragraphs and trim
                    paragraphs = RE_context.split('\n\n')
                    trimmed_context = ""
                    for para in paragraphs:
                        if len(trimmed_context) + len(para) + 2 < 50000:
                            trimmed_context += para + "\n\n"
                        else:
                            break
                    RE_context = trimmed_context
                
                # Add source info to prompt
                if source_pages:
                    full_prompt += f"CONTEXT INFORMATION:\n{RE_context}\n\n"
                    full_prompt += f"SOURCE PAGES: {', '.join(source_pages[:5])}\n\n"
                else:
                    full_prompt += f"CONTEXT INFORMATION:\n{RE_context}\n\n"
            
            full_prompt += f"USER QUESTION: {prompt}\n\nResponse:"
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=0.3,
                top_p=0.95,
            )
            
            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            if response.candidates and response.candidates[0].text:
                response_text = response.candidates[0].text.strip()
                logger.info(f"Successfully generated response ({len(response_text)} chars)")
                
                # Add source links if available
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
    """Main class that coordinates between Confluence and Gemini."""
    
    def __init__(self, confluence_url, confluence_username, confluence_api_token, space_key=None, all_spaces=False):
        self.confluence = ConfluenceClient(confluence_url, confluence_username, confluence_api_token)
        self.gemini = GeminiAssistant()
        self.space_key = space_key
        self.all_spaces = all_spaces
        self.space_pages = {}
        self.page_content_cache = {}
        logger.info(f"Initialized RE Assistant targeting {'all spaces' if all_spaces else f'space: {space_key}'}")
    
    def initialize(self):
        """Initialize by testing connections and gathering initial content."""
        if not self.confluence.test_connection():
            logger.error("Failed to connect to Confluence. Check credentials and URL.")
            return False
        
        logger.info("Loading space content...")
        self.load_space_content()
        return True
    
    def load_space_content(self):
        """Load metadata for all pages in the specified space(s)."""
        cache_file = f"{'all_spaces' if self.all_spaces else self.space_key}_pages_cache.json"
        
        # Try to load from cache
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
        
        # Fetch fresh data
        if self.all_spaces:
            # Get all accessible spaces
            spaces = self.confluence.get_all_spaces()
            logger.info(f"Found {len(spaces)} accessible spaces")
            
            # Fetch pages from each space
            for space in spaces:
                space_key = space['key']
                logger.info(f"Fetching pages from space: {space_key} ({space['name']})")
                
                pages = self.confluence.get_all_pages_in_space(space_key)
                if pages:
                    self.space_pages[space_key] = pages
                    logger.info(f"Loaded {len(pages)} pages from space {space_key}")
                
                # Brief pause to avoid rate limiting
                time.sleep(0.5)
        else:
            # Single space mode
            if not self.space_key:
                logger.error("No space key specified. Please provide a space key.")
                return
            
            pages = self.confluence.get_all_pages_in_space(self.space_key)
            if pages:
                self.space_pages[self.space_key] = pages
                logger.info(f"Loaded {len(pages)} pages from space {self.space_key}")
        
        # Cache the results
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'pages': self.space_pages
                }, f)
            logger.info(f"Cached {sum(len(pages) for pages in self.space_pages.values())} pages to {cache_file}")
        except Exception as e:
            logger.warning(f"Error writing cache file: {str(e)}")
    
    def fetch_page_content(self, page_id, page_title):
        """Helper to fetch page content with error handling."""
        try:
            # Check cache first
            if page_id in self.page_content_cache:
                return self.page_content_cache[page_id]
            
            page_content = self.confluence.get_page_content(page_id, expand="body.storage,metadata.labels")
            if page_content:
                # Cache the content
                self.page_content_cache[page_id] = page_content
                return page_content
            else:
                logger.warning(f"Failed to get content for page {page_title} ({page_id})")
                return None
        except Exception as e:
            logger.error(f"Error fetching content for page {page_title} ({page_id}): {str(e)}")
            return None
    
    def fetch_page_content_batch(self, pages):
        """Fetch content for a batch of pages in parallel."""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_page = {
                executor.submit(self.fetch_page_content, page["id"], page["title"]): page 
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
        
        # Combine all pages from all spaces
        all_pages = []
        for space_key, pages in self.space_pages.items():
            # Add space info to each page
            for page in pages:
                page_with_space = page.copy()
                page_with_space['space_key'] = space_key
                all_pages.append(page_with_space)
        
        # Step 1: Score pages based on title and perform initial filtering
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Add domain-specific terms if query is short
        domain_terms = {
            "RE", "SRE", "LOGS", "MAINTENANCE", "API", "REST", "DATABASE",
            "VIEW", "TABLE", "FIELD", "ENDPOINT", "MAPPING", "SCHEMA"
        }
        
        if len(query_words) < 3:
            query_words.update(domain_terms)
        
        # Initial scoring based on title
        candidates = []
        for page in all_pages:
            title = page.get("title", "").lower()
            # Quick score based on title matches
            title_score = sum(1 for word in query_words if word in title)
            
            # Bonus for exact phrase matches
            if title_score > 0 or any(term in title for term in domain_terms):
                candidates.append((page, title_score))
        
        # Sort by initial score and take top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:20]
        
        logger.info(f"Selected {len(top_candidates)} candidate pages for detailed analysis")
        
        # Fetch content for top candidates in parallel
        if not top_candidates:
            # If no candidates, take most recently updated pages
            recent_pages = sorted(all_pages, 
                                key=lambda p: p.get("history", {}).get("lastUpdated", {}).get("when", "2000-01-01"),
                                reverse=True)[:10]
            top_candidates = [(p, 0) for p in recent_pages]
        
        page_dict = {p[0]["id"]: p[0] for p in top_candidates}
        candidate_pages = list(page_dict.values())
        
        # Fetch content for candidates
        page_contents = self.fetch_page_content_batch(candidate_pages)
        
        # Step 2: Detailed relevance scoring with content
        scored_pages = []
        for page_id, content in page_contents.items():
            page = page_dict[page_id]
            page_text = content.get("content", "").lower()
            title = page.get("title", "").lower()
            
            # Calculate relevance score
            score = 0
            
            # Score based on query word frequency
            for word in query_words:
                word_count = page_text.count(word)
                score += word_count * 0.1
            
            # Higher score for words in title
            if word in title:
                score += 5
            
            # Bonus for exact phrase matches (2-4 words)
            query_lower = query.lower()
            if query_lower in page_text:
                score += 50
            else:
                # Check for partial phrase matches
                for phrase_len in range(2, 5):
                    if len(query_words) >= phrase_len:
                        query_phrases = [" ".join(query_lower.split()[i:i+phrase_len]) 
                                       for i in range(len(query_lower.split()) - phrase_len + 1)]
                        for phrase in query_phrases:
                            if len(phrase.split()) >= 2 and phrase in page_text:
                                score += 3 * page_text.count(phrase)
            
            # Bonus for specific content types
            if "TABLE" in page_text:
                score += 1.5
            if "```" in page_text:
                score += 1.3
            
            # Check for implementation-specific terms
            code_terms = ["code", "example", "implementation", "syntax", "usage"]
            if any(term in query_lower for term in code_terms) and "```" in page_text:
                score += 5
            
            # Bonus for relevant image descriptions
            image_terms = ["image", "diagram", "screenshot", "picture"]
            if any(term in query_lower for term in image_terms) and "[IMAGE" in page_text:
                score += 3
            
            scored_pages.append((page, content, score))
        
        # Sort by score and get top results
        scored_pages.sort(key=lambda x: x[2], reverse=True)
        top_pages = scored_pages[:6]  # Take top 6 pages
        
        logger.info(f"Selected {len(top_pages)} most relevant pages")
        
        if not top_pages:
            return "I couldn't find any relevant information in the Confluence space(s)."
        
        # Step 3: Extract relevant sections from top pages
        relevant_content = []
        source_urls = []
        
        for page, content, score in top_pages:
            page_content = content.get("content", "")
            page_url = content.get("metadata", {}).get("url", "")
            page_title = page.get("title", "")
            space_key = page.get("space_key", "")
            
            if page_url:
                source_urls.append(page_url)
            
            # Split content into sections for targeted extraction
            sections = re.split(r'#+\s+', page_content)
            
            if len(sections) <= 2:
                # If not many sections, use paragraphs
                sections = page_content.split('\n\n')
            
            # Score each section
            section_scores = []
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                
                section_lower = section.lower()
                section_score = 0
                
                # Score based on query terms
                for word in query_words:
                    freq = section_lower.count(word)
                    section_score += freq * 0.5
                
                # Extra points for exact phrase matches
                if query_lower in section_lower:
                    section_score += 10
                
                # Check for phrases
                for phrase_len in range(2, 5):
                    if len(query_words) >= phrase_len:
                        query_phrases = [" ".join(query_lower.split()[i:i+phrase_len]) 
                                       for i in range(len(query_lower.split()) - phrase_len + 1)]
                        for phrase in query_phrases:
                            if phrase in section_lower:
                                section_score += 2
                
                # Special handling for tables and code
                if "TABLE" in section:
                    section_score *= 1.5
                if "```" in section:
                    section_score *= 1.3
                
                section_scores.append((i, section, section_score))
            
            # Get top scoring sections (up to 3 from each page)
            section_scores.sort(key=lambda x: x[2], reverse=True)
            top_sections = section_scores[:3]
            
            # Order sections by their original position
            if top_sections:
                ordered_sections = sorted(top_sections, key=lambda x: x[0])
                
                # Format content block
                if ordered_sections:
                    content_block = f"--- FROM: {page_title} (Space: {space_key}) ---\n\n"
                    
                    # Add page title as heading if first section doesn't start with it
                    first_section = ordered_sections[0][1].strip()
                    if not first_section.startswith('#'):
                        content_block += f"# {page_title}\n\n"
                    
                    # Add sections
                    for _, section, _ in ordered_sections:
                        # Clean up section
                        cleaned_section = re.sub(r'\n{3,}', '\n\n', section.strip())
                        content_block += cleaned_section + "\n\n"
                    
                    content_block += f"Source: {page_url}\n"
                    relevant_content.append(content_block)
        
        # Combine relevant content
        combined_content = "\n\n".join(relevant_content)
        
        # Add source tracking
        if source_urls:
            combined_content = f"SOURCES: {', '.join(source_urls[:5])}\n\n" + combined_content
        
        return combined_content
    
    def answer_question(self, question):
        """Answer a question using Confluence content and Gemini."""
        logger.info(f"Processing question: {question}")
        
        # Extract relevant content based on the question
        relevant_content = self.extract_relevant_content(question)
        
        # Generate response using Gemini
        response = self.gemini.generate_response(question, relevant_content)
        
        return response

def main():
    """Main entry point for the RE Assistant."""
    logger.info("Starting RE Assistant...")
    
    # Check for required environment variables
    if not CONFLUENCE_USERNAME or not CONFLUENCE_API_TOKEN or not CONFLUENCE_URL:
        logger.error("Missing Confluence credentials. Please set CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN, and CONFLUENCE_URL environment variables.")
        print("Error: Missing Confluence credentials. Please set the required environment variables.")
        return
    
    print("\nInitializing RE Assistant...")
    print("Connecting to Confluence and loading knowledge base...")
    
    # Initialize the assistant
    assistant = REAssistant(
        CONFLUENCE_URL, 
        CONFLUENCE_USERNAME, 
        CONFLUENCE_API_TOKEN, 
        space_key=CONFLUENCE_SPACE,
        all_spaces=CONFLUENCE_ALL_SPACES
    )
    
    if not assistant.initialize():
        logger.error("Failed to initialize RE Assistant.")
        print("Error: Failed to initialize. Please check the logs for details.")
        return
    
    print(f"\n====== RE Assistant ======")
    print(f"I've loaded information from {sum(len(pages) for pages in assistant.space_pages.values())} pages in the {len(assistant.space_pages)} space(s).")
    print("I can help you understand anything about RE!")
    print("What would you like to know about SRE?")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            user_input = input("\nQuestion: ").strip()
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Thanks for using the RE Assistant. Have a great day!")
                break
            
            if not user_input:
                continue
            
            print("\nLooking that up for you...")
            start_time = time.time()
            answer = assistant.answer_question(user_input)
            end_time = time.time()
            
            print(f"\nAnswer (found in {end_time - start_time:.2f} seconds):")
            print("------")
            print(answer)
            print("------")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Feel free to come back if you have more questions.")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            print(f"Sorry, I ran into an issue: {str(e)}. Let's try a different question.")

if __name__ == "__main__":
    main()

















#!/usr/bin/env python3
import os
import json
import pickle
import time
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
from requests.auth import HTTPBasicAuth
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, GenerativeConfig, Part, Content
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jira_chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("JiraChatbot")

# Configuration (Environment Variables or Config File)
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")

# Jira Configuration
JIRA_URL = os.environ.get("JIRA_URL", "https://cmegroup.atlassian.net/")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL", "lakshya.vijay@cmegroup.com")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "ATATTxEfGFdPemKZpWezVd58xSZ5242xlMJMRvelfVdjcrmm7G3G0KQeaBdBRw1aATINeOAwHPNmwy8qUQ2oL4kyJRWjQhIKygnc55R7O.lw.3wl1EkopOz1lFRKCYPEhR0A6ZHZNZwMCRjrYuj1Blh5r5legJFD3dGhlcwhBz0mJ1KXZE-reZDUWM=851BA963")

# Cache configuration
CACHE_DIR = "jira_cache"
CACHE_EXPIRY = 24  # Cache expiry in hours

class JiraClient:
    """Enhanced Jira client with better querying capabilities"""
    
    def __init__(self, base_url: str, email: str, api_token: str):
        self.base_url = base_url.rstrip('/')
        self.auth = HTTPBasicAuth(email, api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.cache = JiraCache()
    
    def make_request(self, method: str, endpoint: str, params: Dict = None, 
                     data: Dict = None, cache_key: str = None, force_refresh: bool = False) -> Dict:
        """Make HTTP request with caching support"""
        # Check cache if applicable
        if method.lower() == 'get' and cache_key and not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data
        
        url = f"{self.base_url}/rest/api/3/{endpoint}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data,
                auth=self.auth,
                verify=False  # Disable SSL verification as requested
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache the result if applicable
            if method.lower() == 'get' and cache_key:
                self.cache.set(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"API request error: {e}")
            return {}
    
    def search_issues_advanced(self, query_params: Dict[str, Any]) -> List[Dict]:
        """Advanced issue search with multiple criteria"""
        jql_parts = []
        
        # Project filter
        if query_params.get('project'):
            jql_parts.append(f"project = '{query_params['project']}'")
        
        # Text search in summary, description, or comments
        if query_params.get('text'):
            text_query = query_params['text']
            jql_parts.append(f'(summary ~ "{text_query}" OR description ~ "{text_query}" OR comment ~ "{text_query}")')
        
        # Status filter
        if query_params.get('status'):
            statuses = query_params['status'] if isinstance(query_params['status'], list) else [query_params['status']]
            status_query = " OR ".join([f'status = "{s}"' for s in statuses])
            jql_parts.append(f"({status_query})")
        
        # Assignee filter
        if query_params.get('assignee'):
            jql_parts.append(f"assignee = '{query_params['assignee']}'")
        
        # Reporter filter
        if query_params.get('reporter'):
            jql_parts.append(f"reporter = '{query_params['reporter']}'")
        
        # Labels filter
        if query_params.get('labels'):
            labels = query_params['labels'] if isinstance(query_params['labels'], list) else [query_params['labels']]
            for label in labels:
                jql_parts.append(f"labels = '{label}'")
        
        # Issue type filter
        if query_params.get('issuetype'):
            jql_parts.append(f"issuetype = '{query_params['issuetype']}'")
        
        # Date filters
        if query_params.get('created_after'):
            jql_parts.append(f"created >= '{query_params['created_after']}'")
        
        if query_params.get('created_before'):
            jql_parts.append(f"created <= '{query_params['created_before']}'")
        
        if query_params.get('updated_after'):
            jql_parts.append(f"updated >= '{query_params['updated_after']}'")
        
        if query_params.get('updated_before'):
            jql_parts.append(f"updated <= '{query_params['updated_before']}'")
        
        # Priority filter
        if query_params.get('priority'):
            priorities = query_params['priority'] if isinstance(query_params['priority'], list) else [query_params['priority']]
            priority_query = " OR ".join([f'priority = "{p}"' for p in priorities])
            jql_parts.append(f"({priority_query})")
        
        # Component filter
        if query_params.get('component'):
            jql_parts.append(f"component = '{query_params['component']}'")
        
        # Fix Version filter
        if query_params.get('fixVersion'):
            jql_parts.append(f"fixVersion = '{query_params['fixVersion']}'")
        
        # Build JQL
        jql = " AND ".join(jql_parts) if jql_parts else "ORDER BY created DESC"
        
        # Add ordering
        if not "ORDER BY" in jql:
            jql += " ORDER BY updated DESC"
        
        logger.info(f"Executing JQL: {jql}")
        
        params = {
            "jql": jql,
            "maxResults": query_params.get('max_results', 50),
            "fields": query_params.get('fields', ['summary', 'status', 'assignee', 'reporter', 
                                                  'created', 'updated', 'priority', 'issuetype', 
                                                  'labels', 'comment', 'description'])
        }
        
        cache_key = f"search_{jql}_{params['maxResults']}"
        result = self.make_request('GET', 'search', params=params, cache_key=cache_key)
        
        return result.get('issues', [])
    
    def get_issue(self, issue_key: str) -> Dict:
        """Get detailed information about a specific issue"""
        cache_key = f"issue_{issue_key}"
        return self.make_request('GET', f'issue/{issue_key}', cache_key=cache_key)
    
    def get_issue_comments(self, issue_key: str) -> List[Dict]:
        """Get all comments for an issue"""
        cache_key = f"comments_{issue_key}"
        result = self.make_request('GET', f'issue/{issue_key}/comment', cache_key=cache_key)
        return result.get('comments', [])
    
    def get_issue_attachments(self, issue_key: str) -> List[Dict]:
        """Get all attachments for an issue"""
        cache_key = f"attachments_{issue_key}"
        issue = self.get_issue(issue_key)
        return issue.get('fields', {}).get('attachment', [])
    
    def download_attachment(self, attachment_url: str, filename: str = None) -> Optional[str]:
        """Download an attachment file"""
        if not filename:
            filename = attachment_url.split('/')[-1]
        
        cache_key = f"attachment_data_{filename}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return cached_data
        
        try:
            response = requests.get(
                attachment_url,
                auth=self.auth,
                verify=False
            )
            response.raise_for_status()
            
            # For small files, cache the content
            if len(response.content) < 1024 * 1024:  # 1MB
                self.cache.set(cache_key, response.content)
            
            return response.content
        except Exception as e:
            logger.error(f"Error downloading attachment: {e}")
            return None

class JiraCache:
    """Enhanced caching system for Jira data"""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key"""
        safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', key)
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data if it exists and is not expired"""
        cache_path = self._get_cache_path(key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cache is expired
                cache_time = cached_data.get('timestamp', 0)
                if time.time() - cache_time < CACHE_EXPIRY * 3600:
                    logger.info(f"Cache hit for key: {key}")
                    return cached_data.get('data')
                else:
                    logger.info(f"Cache expired for key: {key}")
                    os.remove(cache_path)
            except Exception as e:
                logger.error(f"Error reading cache: {e}")
        
        return None
    
    def set(self, key: str, data: Any):
        """Cache data with timestamp"""
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'timestamp': time.time(),
                    'data': data
                }, f)
            logger.info(f"Cached data for key: {key}")
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
    
    def clear(self):
        """Clear all cached data"""
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error removing cache file: {e}")
        logger.info("Cache cleared")

class QueryParser:
    """Parse natural language queries to extract Jira query parameters"""
    
    def __init__(self):
        self.date_patterns = {
            'today': 0,
            'yesterday': -1,
            'last week': -7,
            'last month': -30,
            'last year': -365
        }
        
        self.status_keywords = {
            'open': ['Open', 'To Do', 'In Progress'],
            'closed': ['Closed', 'Done', 'Resolved'],
            'in progress': ['In Progress'],
            'todo': ['To Do'],
            'done': ['Done'],
            'resolved': ['Resolved']
        }
        
        self.priority_keywords = {
            'high': ['High', 'Highest'],
            'medium': ['Medium'],
            'low': ['Low', 'Lowest'],
            'critical': ['Highest'],
            'blocker': ['Blocker']
        }
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query to extract parameters"""
        query_lower = query.lower()
        params = {}
        
        # Extract issue keys (e.g., PROJ-123)
        issue_keys = re.findall(r'[A-Z]+-\d+', query.upper())
        if issue_keys:
            params['issue_keys'] = issue_keys
            return params
        
        # Extract text search terms
        text_terms = self._extract_text_terms(query)
        if text_terms:
            params['text'] = text_terms
        
        # Extract status
        for keyword, statuses in self.status_keywords.items():
            if keyword in query_lower:
                params['status'] = statuses
                break
        
        # Extract priority
        for keyword, priorities in self.priority_keywords.items():
            if keyword in query_lower:
                params['priority'] = priorities
                break
        
        # Extract dates
        date_params = self._extract_date_params(query_lower)
        params.update(date_params)
        
        # Extract assignee/reporter
        assignee_match = re.search(r'assigned to (\w+)', query_lower)
        if assignee_match:
            params['assignee'] = assignee_match.group(1)
        
        reporter_match = re.search(r'reported by (\w+)', query_lower)
        if reporter_match:
            params['reporter'] = reporter_match.group(1)
        
        # Extract labels
        label_match = re.search(r'label[s]? (\w+)', query_lower)
        if label_match:
            params['labels'] = [label_match.group(1)]
        
        # Extract issue type
        if 'bug' in query_lower:
            params['issuetype'] = 'Bug'
        elif 'task' in query_lower:
            params['issuetype'] = 'Task'
        elif 'story' in query_lower:
            params['issuetype'] = 'Story'
        
        # Extract component
        component_match = re.search(r'component (\w+)', query_lower)
        if component_match:
            params['component'] = component_match.group(1)
        
        return params
    
    def _extract_text_terms(self, query: str) -> str:
        """Extract relevant text search terms from query"""
        # Remove common words and Jira-specific terms
        stop_words = {
            'show', 'get', 'find', 'list', 'tickets', 'issues', 'jira', 'me', 
            'all', 'the', 'with', 'for', 'about', 'related', 'to', 'in', 'on',
            'ticket', 'issue', 'please', 'can', 'you', 'i', 'need', 'want'
        }
        
        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        if quoted_phrases:
            return ' '.join(quoted_phrases)
        
        # Extract key terms
        words = query.lower().split()
        relevant_words = []
        
        for word in words:
            if (len(word) > 2 and 
                word not in stop_words and 
                not word.startswith(('show', 'get', 'find', 'list'))):
                relevant_words.append(word)
        
        # Look for specific patterns
        patterns = [
            r'rollout[\s-]?restart',
            r'dr[\s-]?(?:related|issue|ticket)',
            r'database[\s-]?(?:issue|problem|error)',
            r'deployment[\s-]?(?:issue|problem|error)',
            r'performance[\s-]?(?:issue|problem)',
            r'security[\s-]?(?:issue|vulnerability)',
            r'production[\s-]?(?:issue|problem|error)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                relevant_words.append(match.group(0).replace(' ', '-'))
        
        return ' '.join(relevant_words) if relevant_words else ''
    
    def _extract_date_params(self, query: str) -> Dict[str, str]:
        """Extract date-related parameters from query"""
        params = {}
        today = datetime.now()
        
        # Check for relative dates
        for keyword, days_offset in self.date_patterns.items():
            if keyword in query:
                date = today + timedelta(days=days_offset)
                if 'created' in query:
                    params['created_after'] = date.strftime('%Y-%m-%d')
                elif 'updated' in query:
                    params['updated_after'] = date.strftime('%Y-%m-%d')
                else:
                    # Default to created date
                    params['created_after'] = date.strftime('%Y-%m-%d')
        
        # Check for specific date patterns
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
        if date_match:
            date_str = date_match.group(1)
            if 'before' in query:
                params['created_before'] = date_str
            elif 'after' in query:
                params['created_after'] = date_str
            else:
                params['created_after'] = date_str
        
        # Check for month/year patterns
        month_year_match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', query)
        if month_year_match:
            month_name = month_year_match.group(1)
            year = month_year_match.group(2)
            month_num = datetime.strptime(month_name, '%B').month
            params['created_after'] = f"{year}-{month_num:02d}-01"
            params['created_before'] = f"{year}-{month_num:02d}-31"
        
        return params

class GeminiChat:
    """Enhanced Gemini chat integration with better context handling"""
    
    def __init__(self, project_id: str, location: str, model_name: str):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
        
        # System prompt with enhanced capabilities
        self.system_prompt = """You are JiraGPT, an intelligent assistant designed to help answer questions about Jira projects, tickets, and tasks.

Key capabilities:
1. Provide clear, concise, and accurate answers about Jira tickets, projects, and workflows
2. Format responses appropriately based on the query (tables, bullet points, paragraphs)
3. Include relevant links to Jira pages when applicable
4. Ask clarifying questions when needed to provide more accurate answers
5. Understand both technical and non-technical questions
6. Analyze tables and images from Jira tickets
7. Be capable of explaining Jira concepts in simple terms
8. Stay professional but friendly in your tone

When you don't know the answer:
- Clearly state that you don't have enough information
- Ask specific questions to help narrow down what the user is looking for
- Suggest alternative approaches or places to look for the information

When providing links, always use the format [Title](URL) for better readability.

When a user asks about specific tickets:
- Summarize key information like status, assignee, priority, and description
- Highlight recent updates or comments
- Include a direct link to the ticket

Always respond in a structured way that makes information easy to scan and understand."""
        
        # Conversation history
        self.conversation_history = []
        self.max_history_length = 10
    
    def generate_response(self, user_query: str, jira_context: Dict = None, 
                         image_data: List = None) -> str:
        """Generate a response using Gemini based on the user query and Jira context"""
        try:
            # Prepare context
            context = f"{self.system_prompt}\n\n"
            
            if jira_context:
                context += "Here is relevant information from Jira to help answer the question:\n\n"
                context += self._format_jira_context(jira_context)
                context += "\n\n"
            
            # Combine context and user query
            full_prompt = f"{context}User question: {user_query}\n\nYour response:"
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=0.2,  # Lower temperature for more factual responses
                top_p=0.95,
                max_output_tokens=8192,
            )
            
            # Generate response with or without images
            if image_data:
                # Handle multimodal content (text + images)
                content_parts = [Part.from_text(full_prompt)]
                
                for img in image_data:
                    if img.get('mime_type') and img.get('data'):
                        image_part = Part.from_data(
                            data=base64.b64decode(img['data']),
                            mime_type=img['mime_type']
                        )
                        content_parts.append(image_part)
                
                content = Content(
                    role="user",
                    parts=content_parts
                )
                
                response = self.model.generate_content(
                    content,
                    generation_config=generation_config,
                )
            else:
                # Text only response
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                )
            
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].text
            else:
                return "I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while processing your question. Please try again. Technical details: {str(e)}"
    
    def _format_jira_context(self, jira_context: Dict) -> str:
        """Format Jira context for better understanding by Gemini"""
        formatted_context = ""
        
        if jira_context.get('issues'):
            formatted_context += f"Found {len(jira_context['issues'])} relevant Jira tickets:\n\n"
            
            for idx, issue in enumerate(jira_context['issues'], 1):
                fields = issue.get('fields', {})
                key = issue.get('key', 'Unknown')
                
                formatted_context += f"**Ticket {idx}: {key}**\n"
                formatted_context += f"- Summary: {fields.get('summary', 'No summary')}\n"
                formatted_context += f"- Status: {fields.get('status', {}).get('name', 'Unknown')}\n"
                formatted_context += f"- Priority: {fields.get('priority', {}).get('name', 'Unknown')}\n"
                formatted_context += f"- Type: {fields.get('issuetype', {}).get('name', 'Unknown')}\n"
                formatted_context += f"- Assignee: {fields.get('assignee', {}).get('displayName', 'Unassigned')}\n"
                formatted_context += f"- Reporter: {fields.get('reporter', {}).get('displayName', 'Unknown')}\n"
                formatted_context += f"- Created: {fields.get('created', 'Unknown')}\n"
                formatted_context += f"- Updated: {fields.get('updated', 'Unknown')}\n"
                
                if fields.get('description'):
                    formatted_context += f"- Description: {fields['description'][:200]}...\n"
                
                if fields.get('labels'):
                    formatted_context += f"- Labels: {', '.join(fields['labels'])}\n"
                
                # Add recent comments
                if jira_context.get('comments', {}).get(key):
                    comments = jira_context['comments'][key]
                    if comments:
                        formatted_context += f"- Latest Comment: {comments[-1].get('body', '')[:200]}...\n"
                
                formatted_context += f"- Link: {JIRA_URL}browse/{key}\n\n"
        
        if jira_context.get('projects'):
            formatted_context += "\nAvailable projects:\n"
            for project in jira_context['projects']:
                formatted_context += f"- {project.get('key')}: {project.get('name')}\n"
        
        return formatted_context
    
    def add_to_history(self, user_query: str, bot_response: str):
        """Add an exchange to the conversation history"""
        self.conversation_history.append({
            "user": user_query,
            "bot": bot_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only the most recent exchanges
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

class JiraChatbot:
    """Main class that integrates Jira client with Gemini AI for a chatbot experience"""
    
    def __init__(self):
        self.jira_client = JiraClient(JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN)
        self.gemini_chat = GeminiChat(PROJECT_ID, REGION, MODEL_NAME)
        self.query_parser = QueryParser()
        self.conversation_history = []
        self.max_history_length = 10
    
    def get_relevant_jira_data(self, query: str) -> Tuple[Dict, List]:
        """Get relevant Jira data based on the user query"""
        logger.info(f"Processing query: {query}")
        
        # Parse the query
        query_params = self.query_parser.parse_query(query)
        logger.info(f"Parsed query parameters: {query_params}")
        
        relevant_data = {}
        image_data = []
        
        # If specific issue keys are mentioned, get those issues
        if query_params.get('issue_keys'):
            issues_data = []
            comments_data = {}
            
            for issue_key in query_params['issue_keys']:
                issue = self.jira_client.get_issue(issue_key)
                if issue:
                    issues_data.append(issue)
                    
                    # Get comments for the issue
                    comments = self.jira_client.get_issue_comments(issue_key)
                    if comments:
                        comments_data[issue_key] = comments
                    
                    # Get attachments if needed
                    attachments = self.jira_client.get_issue_attachments(issue_key)
                    if attachments and "attachment" in query.lower():
                        for attachment in attachments:
                            if attachment.get('mimeType', '').startswith('image/'):
                                attachment_data = self.jira_client.download_attachment(
                                    attachment['content']
                                )
                                if attachment_data:
                                    image_data.append({
                                        'mime_type': attachment['mimeType'],
                                        'data': base64.b64encode(attachment_data).decode('utf-8')
                                    })
            
            if issues_data:
                relevant_data['issues'] = issues_data
                relevant_data['comments'] = comments_data
        
        # Otherwise, search for issues based on extracted parameters
        else:
            # Add default project if not specified
            if not query_params.get('project') and not query_params.get('text'):
                # If no specific search criteria, return empty
                return relevant_data, image_data
            
            # Search for issues
            search_results = self.jira_client.search_issues_advanced(query_params)
            
            if search_results:
                relevant_data['issues'] = search_results
                
                # Get comments for top issues if the query seems to ask for details
                if any(word in query.lower() for word in ['detail', 'comment', 'update', 'latest']):
                    comments_data = {}
                    for issue in search_results[:5]:  # Limit to top 5 issues
                        issue_key = issue.get('key')
                        comments = self.jira_client.get_issue_comments(issue_key)
                        if comments:
                            comments_data[issue_key] = comments
                    
                    if comments_data:
                        relevant_data['comments'] = comments_data
        
        # Get project information if needed
        if "project" in query.lower() and not query_params.get('issue_keys'):
            projects = self.jira_client.make_request('GET', 'project')
            if projects:
                relevant_data['projects'] = projects
        
        return relevant_data, image_data
    
    def process_query(self, query: str) -> str:
        """Process a user query and return a response"""
        start_time = time.time()
        
        try:
            # Check if it's a query to clear the cache
            if query.lower().strip() in ["clear cache", "refresh cache", "reset cache"]:
                self.jira_client.cache.clear()
                response = "Cache cleared. I'll fetch fresh data for your next queries."
                self.add_to_history(query, response)
                return response
            
            # Check if it's an irrelevant query
            irrelevant_patterns = [
                r'hello|hi|hey',
                r'thank|thanks',
                r'bye|goodbye',
                r'how are you',
                r'what can you do',
                r'help me'
            ]
            
            for pattern in irrelevant_patterns:
                if re.match(f"^({pattern})$", query.lower().strip()):
                    # Handle common greetings and small talk
                    if pattern == r'hello|hi|hey':
                        response = "Hello! I'm JiraGPT, your Jira assistant. How can I help you with your Jira queries today?"
                    elif pattern == r'thank|thanks':
                        response = "You're welcome! Feel free to ask if you need any more help with Jira."
                    elif pattern == r'bye|goodbye':
                        response = "Goodbye! Feel free to come back when you need help with Jira."
                    elif pattern == r'how are you':
                        response = "I'm functioning well, thank you! How can I help you with Jira today?"
                    elif pattern == r'what can you do':
                        response = """I can help you with various Jira-related tasks, such as:
• Finding information about specific tickets (e.g., 'Tell me about RDRF-2606')
• Searching for tickets with specific criteria (e.g., 'Show me high priority bugs')
• Explaining Jira concepts and workflows
• Analyzing ticket data, including images and tables
• Providing links to relevant Jira pages
• How can I assist you today?"""
                    elif pattern == r'help me':
                        response = """I'd be happy to help! To get started, try asking specific questions about your Jira tickets, projects, or workflows. For example:
• "Tell me about ticket RDRF-2606"
• "What are the open issues in my RDS SPHERE project?"
• "Show me high priority bugs created this week"
• "How do I create alerts in Jira?"
The more specific your question, the better I can assist you!"""
                    
                    self.add_to_history(query, response)
                    return response
            
            # Get relevant data from Jira for the query
            relevant_data, image_data = self.get_relevant_jira_data(query)
            
            # Check if we need to ask a clarifying question
            if not relevant_data or (not relevant_data.get("issues") and not relevant_data.get("user")):
                # Simplified logic for asking clarifying questions
                if "ticket" in query.lower() or "issue" in query.lower():
                    response = "I need more information to help you. Could you provide the specific ticket ID (e.g., RDRF-2606)?"
                elif "project" in query.lower():
                    response = "I need more information to help you. Could you provide the specific project key or name?"
                else:
                    # Generate a response with limited context
                    history_context = []
                    if len(self.conversation_history) >= 3:
                        history_context = self.conversation_history[-3:]
                    
                    response = self.gemini_chat.generate_response(query, relevant_data, image_data)
            else:
                # Add conversation history to the context
                relevant_data["conversation_history"] = self.conversation_history[-3:] if self.conversation_history else []
                
                # Generate response using Gemini
                response = self.gemini_chat.generate_response(query, relevant_data, image_data)
            
            # Add the exchange to conversation history
            self.add_to_history(query, response)
            
            # Log processing time
            elapsed_time = time.time() - start_time
            logger.info(f"Query processed in {elapsed_time:.2f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I encountered an error while processing your question. Please try again or rephrase your question. Technical details: {str(e)}"
    
    def add_to_history(self, query: str, response: str):
        """Add an exchange to the conversation history"""
        self.conversation_history.append({
            "user": query,
            "bot": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only the most recent exchanges
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def extract_jira_issue_keys(self, text: str) -> List[str]:
        """Extract Jira issue keys from text using regex"""
        pattern = r'[A-Z]+-\d+'
        return re.findall(pattern, text)
    
    def run_chatbot(self):
        """Run the chatbot in an interactive loop"""
        print("\n===== JiraGPT Chatbot =====")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("Type 'clear cache' to refresh Jira data.")
        
        chatbot = JiraChatbot()
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nChatbot: Goodbye! Have a great day.")
                break
            
            response = chatbot.process_query(user_input)
            print(f"\nChatbot: {response}")

def run_chatbot():
    """Run the chatbot in an interactive loop"""
    print("\n===== JiraGPT Chatbot =====")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'clear cache' to refresh Jira data.")
    
    chatbot = JiraChatbot()
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nChatbot: Goodbye! Have a great day.")
            break
        
        response = chatbot.process_query(user_input)
        print(f"\nChatbot: {response}")

if __name__ == "__main__":
    run_chatbot()
