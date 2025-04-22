copper-assistant/
├── static/
│   ├── css/
│   │   ├── animations.css  - Animation styles for the UI
│   │   └── styles.css      - Main CSS styles
│   ├── js/
│   │   ├── app.js          - Main application logic
│   │   ├── chat.js         - Chat functionality and API integration
│   │   └── ui-components.js - Reusable UI components
│   └── index.html          - Main HTML file
├── app.py                  - Flask server and API endpoints
├── copper_assistant.py     - Core CoPPER Assistant logic
└── requirements.txt        - Python dependencies

















import os
import json
import asyncio
from flask import Flask, request, jsonify, render_template, send_from_directory
from copper_assistant import CopperAssistant

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='static')

# Initialize the assistant
assistant = None

@app.before_first_request
def initialize_assistant():
    global assistant
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    assistant = CopperAssistant()
    loop.run_until_complete(assistant.initialize())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/python-response', methods=['POST'])
def process_query():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': True, 'message': 'No query provided'}), 400
    
    # Process the query using the CoPPER Assistant
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(assistant.process_query(query))
    
    return jsonify({
        'result': response,
        'error': False
    })

# Serve the CSS files
@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(app.static_folder, 'css'), filename)

# Serve the JS files
@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join(app.static_folder, 'js'), filename)

if __name__ == '__main__':
    app.run(debug=True)

















import os
import re
import json
import time
import uuid
import logging
import asyncio
import base64
import threading
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Any, Union, Set
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field
from functools import lru_cache
from enum import Enum
import requests
from bs4 import BeautifulSoup, Tag, NavigableString

# Optional VertexAI import - If not installed, we'll use a fallback
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig, Content
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    print("VertexAI not available - using fallback response mode")

# ==================== CONFIGURATION ====================

class Config:
    """Centralized configuration management for all services."""
    
    # Confluence credentials and settings
    CONFLUENCE_URL = "https://cmegroup.atlassian.net"
    CONFLUENCE_USERNAME = "hackathon_user@cme.com"
    CONFLUENCE_API_TOKEN = "atlassian_api_token_123456"
    CONFLUENCE_SPACES = ["CMEIN", "CTS"]  # Multiple spaces for searching
    
    # Gemini AI settings
    MODEL_NAME = "gemini-1.5-pro"
    PROJECT_ID = "cme-hackathon-project"
    LOCATION = "us-central1"
    
    # General settings
    MAX_WORKERS = 8  # Increased parallel processing
    CACHE_SIZE = 500  # Expanded cache size
    MAX_CONTENT_SIZE = 100000  # Increased context size for Gemini
    MAX_HYPERLINK_DEPTH = 2  # Max depth for hyperlink traversal
    MAX_LINKS_PER_PAGE = 10  # Max links to follow per page
    CACHE_DIR = ".cache"
    PAGE_CACHE_FILE = os.path.join(CACHE_DIR, "page_cache.json")
    VIEW_CACHE_FILE = os.path.join(CACHE_DIR, "view_cache.json")
    MAPPING_CACHE_FILE = os.path.join(CACHE_DIR, "mapping_cache.json")
    GEMINI_TIMEOUT = 60  # Seconds to wait for Gemini response
    
    # Advanced parsing settings
    SQL_KEYWORD_PATTERNS = [
        r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:FORCE\s+)?VIEW",
        r"SELECT\s+.+?\s+FROM",
        r"JOIN\s+\w+\s+ON",
        r"WHERE\s+.+?(?:GROUP BY|ORDER BY|HAVING|$)",
        r"GROUP\s+BY",
        r"ORDER\s+BY",
        r"HAVING"
    ]
    
    # API Domain Mappings
    DOMAIN_MAPPINGS = {
        "PRODUCT": "products",
        "INSTRUMENT": "instruments",
        "SESSION": "sessions",
        "FIRM": "firms",
        "RISK": "risk",
        "TRADER": "traders",
        "STATIC": "static"
    }
    
    # Known operations mapping
    SQL_TO_API_OPS = {
        "=": "EQ",
        "!=": "NEQ",
        "<>": "NEQ",
        "<": "LT",
        "<=": "LTE",
        ">": "GT",
        ">=": "GTE",
        "IN": "IN",
        "IS NULL": "ISNULL",
        "IS NOT NULL": "NOTNULL",
        "LIKE": "LIKE",
        "BETWEEN": "BETWEEN"
    }
    
    # Setup logging
    @staticmethod
    def configure_logging():
        """Configure logging for the application."""
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(Config.CACHE_DIR, "copper_assistant.log"))
            ]
        )
        
        # Create a logger for each major component
        loggers = {
            "main": logging.getLogger("CopperAssistant"),
            "confluence": logging.getLogger("ConfluenceService"),
            "sql": logging.getLogger("SqlParserService"),
            "api": logging.getLogger("ApiMappingService"),
            "llm": logging.getLogger("LLMService"),
            "output": logging.getLogger("OutputFormatter")
        }
        
        return loggers


# Initialize logging
loggers = Config.configure_logging()
logger = loggers["main"]

# ==================== SCHEMA DEFINITIONS ====================

@dataclass
class PageMetadata:
    """Metadata for a Confluence page."""
    id: str
    title: str
    type: str
    url: str
    space_key: str
    labels: List[str] = field(default_factory=list)
    last_updated: str = ""
    created_date: str = ""
    content_type: str = "page"


@dataclass
class ExtractedContent:
    """Represents content extracted from Confluence."""
    text: str = ""
    html: str = ""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    code_blocks: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    structured_content: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Page:
    """Represents a complete Confluence page."""
    metadata: PageMetadata
    content: ExtractedContent
    relevance_score: float = 0.0


@dataclass
class SqlView:
    """Represents a parsed SQL view."""
    name: str
    definition: str
    columns: List[Dict[str, str]] = field(default_factory=list)
    source_tables: List[str] = field(default_factory=list)
    join_conditions: List[Dict[str, str]] = field(default_factory=list)
    filter_conditions: List[Dict[str, Any]] = field(default_factory=list)
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    primary_domain: str = ""
    relationships: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ApiMapping:
    """Represents an API mapping for a SQL view."""
    view_name: str
    api_endpoint: str
    req_name: str
    data_domain: str
    type: str
    request_parameters: List[Dict[str, Any]] = field(default_factory=list)
    fields: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    explanation: str = ""


@dataclass
class UserQuery:
    """Represents a user query and tracking information."""
    original_text: str
    processed_text: str = ""
    intent: str = ""
    entities: Dict[str, Any] = field(default_factory=dict)
    context_pages: List[Page] = field(default_factory=list)
    extracted_view: Optional[SqlView] = None
    generated_mapping: Optional[ApiMapping] = None
    start_time: float = field(default_factory=time.time)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class QueryIntent(Enum):
    """Types of user query intents."""
    GENERAL_QUESTION = "general_question"
    VIEW_INFO = "view_info"
    API_INFO = "api_info"
    MAP_VIEW_TO_API = "map_view_to_api"
    GENERATE_JSON = "generate_json"
    EXPLAIN_MAPPING = "explain_mapping"
    COMPARE_VIEWS = "compare_views"


# ==================== PROMPT TEMPLATES ====================

class PromptTemplates:
    """Centralized prompt templates for LLM interactions."""
    
    SYSTEM_PERSONALITY = """
You are the expert CoPPER Assistant, a specialized AI for the CoPPER database and API system used at CME Group.

Your personality traits:
- Precise and technical while remaining accessible
- Confident in your expertise on CoPPER database views and API mappings
- Structured and methodical in your explanations
- Focused on providing actionable information
- Detail-oriented, especially with technical specifications

Always format your responses for maximum clarity, using:
- Tables for comparing multiple items
- Bullet points for sequential steps
- Code blocks with proper syntax highlighting
- Bold text for important concepts
- Diagrams when explaining relationships (when possible)

When faced with ambiguity:
1. Make reasonable assumptions based on CoPPER naming conventions and design patterns
2. Clearly state those assumptions
3. Provide the most likely mapping based on available information
4. Indicate confidence level and alternatives if appropriate
"""

    ANALYZE_SQL_VIEW = """
# SQL View Analysis Request

Analyze the following SQL view definition in detail, focusing on understanding its structure, purpose, and data relationships.

## SQL View Definition
```sql
{view_definition}
```

## Analysis Tasks
1. Identify the main purpose of this view
2. List all columns and their data types (if discernible)
3. Identify all source tables and their relationships
4. Analyze JOIN conditions and their business logic
5. Extract WHERE clause conditions and their purpose
6. Identify any data transformations or calculations
7. Determine the primary domain this view belongs to
8. Identify key business rules implemented in the view

## Special Attention Areas
- Pay special attention to date/time handling
- Note any reference data lookups
- Identify any conditional logic (CASE statements)
- Look for performance-impacting patterns

Provide your analysis as a structured report with clear sections.
"""

    GENERATE_API_MAPPING = """
# API Mapping Request

Create a CoPPER API mapping based on the following SQL view. The mapping should conform to CoPPER API conventions.

## SQL View Definition
```sql
{view_definition}
```

## SQL View Analysis
{view_analysis}

## Examples of Similar Mappings
Here are examples of similar view-to-API mappings that follow the expected pattern:

{examples}

## Required Output
Generate a complete API JSON request body that correctly maps the view's functionality, including:
1. Proper reqName and dataDomain identification
2. Correct type (independent/dependent)
3. Accurate mapping of SQL WHERE conditions to API request parameters
4. Mapping of SQL columns to API fields
5. Appropriate operation types for conditions (EQ, IN, GT, LT, etc.)

The JSON should be valid, well-structured, and include all necessary parameters to replicate the view's functionality via the API.
"""

    EXPLAIN_MAPPING = """
# Mapping Explanation Request

Explain the relationship between this SQL view and its corresponding API mapping in detail.

## SQL View Definition
```sql
{view_definition}
```

## Generated API Mapping
```json
{api_mapping}
```

## Explanation Tasks
1. Explain how the view's primary tables map to the API's data domain
2. Detail how each WHERE condition translates to an API request parameter
3. Explain the significance of the API operation types chosen
4. Clarify any complex transformations or relationships
5. Highlight any special handling for NULL values, dates, or complex logic
6. Explain how the fields list corresponds to the SELECT clause columns

Your explanation should be comprehensive but clear, suitable for developers migrating from the SQL view to the API.
"""

    ANSWER_GENERAL_QUESTION = """
# CoPPER Question

Answer the following question about CoPPER database views or APIs using the provided context information.

## Question
{question}

## Context Information
{context}

## Response Guidelines
1. Answer directly and specifically based on the context provided
2. Structure your response logically with clear sections
3. Use examples where appropriate
4. Include relevant code snippets if they help illustrate your answer
5. If information is missing from the context, acknowledge the limitations

Your response should be comprehensive, accurate, and focused on the specific question asked.
"""

# ==================== CONTENT EXTRACTION SERVICE ====================

class ContentExtractorService:
    """Enhanced content extraction service with recursive link following and advanced parsing."""
    
    def __init__(self, max_depth=Config.MAX_HYPERLINK_DEPTH, max_links=Config.MAX_LINKS_PER_PAGE):
        """Initialize the content extractor with configuration."""
        self.max_depth = max_depth
        self.max_links_per_page = max_links
        self.visited_links = set()
        self.logger = loggers["confluence"]
    
    async def extract_content_recursive(self, html_content, base_url, title="", current_depth=0):
        """
        Extract content recursively, following links up to a specified depth.
        
        Args:
            html_content: The HTML content to process
            base_url: The base URL for resolving relative links
            title: The title of the page
            current_depth: Current recursion depth
            
        Returns:
            ExtractedContent object with all content, including from linked pages
        """
        # Extract base content
        content = self.extract_content_from_html(html_content, title)
        
        # Don't follow links if we've reached max depth
        if current_depth >= self.max_depth:
            return content
        
        # Find and follow links (limited to max_links_per_page)
        links_to_follow = []
        for link in content.links[:self.max_links_per_page]:
            full_url = urljoin(base_url, link["href"])
            
            # Skip already visited links and external links
            if full_url in self.visited_links or not full_url.startswith(Config.CONFLUENCE_URL):
                continue
                
            links_to_follow.append((full_url, link["text"]))
            self.visited_links.add(full_url)
        
        # Process links in parallel
        if links_to_follow:
            self.logger.info(f"Following {len(links_to_follow)} links at depth {current_depth}")
            linked_contents = []
            
            async def fetch_link_content(url, link_text):
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        linked_content = await self.extract_content_recursive(
                            response.text, url, link_text, current_depth + 1
                        )
                        return linked_content
                except Exception as e:
                    self.logger.error(f"Error following link {url}: {str(e)}")
                return None
            
            # Create tasks for each link
            tasks = [fetch_link_content(url, text) for url, text in links_to_follow]
            linked_contents = await asyncio.gather(*tasks)
            
            # Merge linked content with base content
            for linked_content in linked_contents:
                if linked_content:
                    content.text += f"\n\n--- Related Content ---\n{linked_content.text}"
                    content.tables.extend(linked_content.tables)
                    content.code_blocks.extend(linked_content.code_blocks)
                    # Don't add nested links to avoid exponential growth
        
        return content
    
    def extract_content_from_html(self, html_content, title="") -> ExtractedContent:
        """
        Extract and structure content from HTML with advanced parsing.
        
        Args:
            html_content: The HTML content to process
            title: The title of the page
            
        Returns:
            ExtractedContent object with structured content
        """
        extracted = ExtractedContent()
        
        try:
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title if not provided
            if not title and soup.title:
                title = soup.title.text.strip()
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Store raw HTML for potential future use
            extracted.html = str(soup)
            
            # Extract text content
            text_parts = []
            if title:
                text_parts.append(f"# {title}")
            
            # Process headings, paragraphs and list items
            for elem in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
                if elem.name.startswith('h'):
                    level = int(elem.name[1])
                    text_parts.append(f"{'#' * level} {elem.text.strip()}")
                else:
                    text_parts.append(elem.text.strip())
            
            # Process tables
            self._extract_tables(soup, extracted)
            
            # Process code blocks
            self._extract_code_blocks(soup, extracted)
            
            # Process images
            self._extract_images(soup, extracted)
            
            # Process links
            self._extract_links(soup, extracted)
            
            # Process structured content (notes, warnings, etc.)
            self._extract_structured_content(soup, extracted)
            
            # Process iframes
            self._extract_iframe_content(soup, extracted)
            
            # Combine all text content
            extracted.text = "\n\n".join(text_parts)
            
            return extracted
            
        except Exception as e:
            self.logger.error(f"Error extracting content: {str(e)}")
            extracted.text = f"Error extracting content: {str(e)}"
            return extracted
    
    def _extract_tables(self, soup, extracted):
        """Extract tables with enhanced processing for complex Confluence tables."""
        # Handle standard tables
        standard_tables = soup.find_all('table')
        
        # Handle Confluence-specific table wrappers
        confluence_tables = soup.find_all('div', class_='table-wrap')
        
        all_tables = []
        
        # Process standard tables (skipping those inside table-wrap)
        for table in standard_tables:
            if not table.find_parent('div', class_='table-wrap'):
                all_tables.append(table)
        
        # Add Confluence tables
        for table_wrap in confluence_tables:
            table = table_wrap.find('table')
            if table:
                all_tables.append(table)
        
        # Process all tables
        for idx, table in enumerate(all_tables):
            table_data = {}
            
            # Get table title/caption
            caption = table.find('caption')
            table_title = caption.text.strip() if caption else f"Table {idx+1}"
            
            # Process headers
            headers = []
            thead = table.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    headers = [th.text.strip() for th in header_row.find_all(['th', 'td'])]
            
            # If no headers in thead, try getting from first row
            if not headers:
                first_row = table.find('tr')
                if first_row:
                    # Check if it looks like a header row
                    first_row_cells = first_row.find_all(['th', 'td'])
                    if first_row.find('th') or all(
                        cell.name == 'th' or 
                        (cell.get('class') and 'header' in '.'.join(cell.get('class', []))) 
                        for cell in first_row_cells
                    ):
                        headers = [th.text.strip() for th in first_row_cells]
            
            # Process rows
            rows = []
            tbody = table.find('tbody')
            if tbody:
                for tr in tbody.find_all('tr'):
                    row = [self._extract_cell_content(td) for td in tr.find_all(['td', 'th'])]
                    if any(cell for cell in row):  # Skip empty rows
                        rows.append(row)
            else:
                # If no tbody, process all rows (skipping the header if we extracted it)
                all_rows = table.find_all('tr')
                start_idx = 1 if headers and len(all_rows) > 0 else 0
                for tr in all_rows[start_idx:]:
                    row = [self._extract_cell_content(td) for td in tr.find_all(['td', 'th'])]
                    if any(cell for cell in row):  # Skip empty rows
                        rows.append(row)
            
            # Save table data
            table_data = {
                "title": table_title,
                "headers": headers,
                "rows": rows,
                "raw_html": str(table)
            }
            
            extracted.tables.append(table_data)
    
    def _extract_cell_content(self, cell):
        """Extract content from a table cell, handling nested elements."""
        # Check for nested elements
        if cell.find(['a', 'code', 'strong', 'em']):
            # Return both text and any significant markup
            return {
                "text": cell.text.strip(),
                "links": [{"text": a.text.strip(), "href": a.get('href', '')} 
                          for a in cell.find_all('a')],
                "code": [code.text.strip() for code in cell.find_all('code')],
                "emphasis": [em.text.strip() for em in cell.find_all(['strong', 'em'])]
            }
        return cell.text.strip()
    
    def _extract_code_blocks(self, soup, extracted):
        """Extract code blocks with improved formatting and language detection."""
        # Find all code blocks
        for pre in soup.find_all('pre'):
            code = pre.find('code')
            if code:
                # Try to determine the language
                lang = ""
                code_class = code.get('class', [])
                for cls in code_class:
                    if cls.startswith('language-'):
                        lang = cls.replace('language-', '')
                        break
                
                # Get the code content
                code_content = code.text.strip()
                
                # Save the code block
                extracted.code_blocks.append({
                    "language": lang,
                    "content": code_content,
                    "line_count": len(code_content.split('\n'))
                })
            elif pre.text.strip():
                # Pre without code tag
                extracted.code_blocks.append({
                    "language": "",
                    "content": pre.text.strip(),
                    "line_count": len(pre.text.strip().split('\n'))
                })
        
        # Also look for Confluence code macros
        for code_macro in soup.find_all('div', class_='code-block'):
            lang_elem = code_macro.find('div', class_='codeHeader')
            lang = lang_elem.text.strip() if lang_elem else ""
            
            content_elem = code_macro.find('div', class_='codeContent')
            if content_elem:
                code_content = content_elem.text.strip()
                extracted.code_blocks.append({
                    "language": lang,
                    "content": code_content,
                    "line_count": len(code_content.split('\n'))
                })
    
    def _extract_images(self, soup, extracted):
        """Extract images with contextual information."""
        for img in soup.find_all('img'):
            # Get image attributes
            alt_text = img.get('alt', '').strip()
            title = img.get('title', '').strip()
            src = img.get('src', '')
            
            # Try to get contextual information
            context = ""
            
            # Check parent elements for figure captions
            parent_fig = img.find_parent('figure')
            if parent_fig:
                fig_caption = parent_fig.find('figcaption')
                if fig_caption:
                    context = fig_caption.text.strip()
            
            # If no caption found, try to get surrounding text
            if not context:
                prev_elem = img.find_previous_siblings(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if prev_elem and len(prev_elem) > 0 and len(prev_elem[0].text.strip()) < 200:
                    context = f"Previous content: {prev_elem[0].text.strip()}"
            
            # Save image information
            extracted.images.append({
                "alt_text": alt_text or title or "Image",
                "src": src,
                "context": context,
                "width": img.get('width', ''),
                "height": img.get('height', '')
            })
    
    def _extract_links(self, soup, extracted):
        """Extract links with their text and href attributes."""
        for a in soup.find_all('a'):
            href = a.get('href', '')
            if href and not href.startswith('#') and not href.startswith('javascript:'):
                extracted.links.append({
                    "text": a.text.strip(),
                    "href": href,
                    "title": a.get('title', '')
                })
    
    def _extract_structured_content(self, soup, extracted):
        """Extract structured content like notes, warnings, etc."""
        # Look for Confluence admonitions (notes, warnings, etc.)
        for div in soup.find_all(['div', 'section']):
            if 'class' in div.attrs:
                class_str = ' '.join(div['class'])
                
                # Check for common Confluence structured content classes
                if any(term in class_str for term in ['panel', 'info', 'note', 'warning', 'callout', 'aui-message']):
                    title_elem = div.find(['h3', 'h4', 'h5', 'strong', 'b'])
                    title = title_elem.text.strip() if title_elem else "Note"
                    
                    # Get the content (excluding the title)
                    if title_elem:
                        title_elem.extract()
                    
                    content = div.text.strip()
                    
                    # Determine the type based on classes
                    content_type = "note"  # default
                    if "note" in class_str or "info" in class_str:
                        content_type = "note"
                    elif "warning" in class_str:
                        content_type = "warning"
                    elif "error" in class_str or "danger" in class_str:
                        content_type = "error"
                    elif "tip" in class_str or "success" in class_str:
                        content_type = "tip"
                    
                    extracted.structured_content.append({
                        "type": content_type,
                        "title": title,
                        "content": content
                    })
    
    def _extract_iframe_content(self, soup, extracted):
        """Extract content from iframes, including special handling for Gliffy diagrams."""
        for iframe in soup.find_all('iframe'):
            iframe_src = iframe.get('src', '')
            if iframe_src:
                iframe_info = {
                    "src": iframe_src,
                    "title": iframe.get('title', ''),
                    "type": "unknown"
                }
                
                # Try to determine iframe type
                if 'gliffy' in iframe_src.lower():
                    iframe_info["type"] = "diagram"
                    # Extract diagram title if possible
                    container = iframe.find_parent('div', class_='ap-container')
                    if container:
                        iframe_info["title"] = (container.get('data-macro-name', '')
                                               or iframe.get('title', 'Diagram'))
                elif 'viewpage' in iframe_src.lower():
                    iframe_info["type"] = "confluence_page"
                elif 'youtube' in iframe_src.lower() or 'vimeo' in iframe_src.lower():
                    iframe_info["type"] = "video"
                
                extracted.structured_content.append({
                    "type": "iframe",
                    "title": iframe_info["title"] or "Embedded Content",
                    "content": f"[IFRAME: {iframe_info['type']}] {iframe_info['src']}"
                })
    
    def format_for_context(self, extracted_content, title="") -> str:
        """
        Format the extracted content for use as context in LLM prompts.
        
        Args:
            extracted_content: The ExtractedContent object
            title: The title of the page
            
        Returns:
            Formatted string containing all the content
        """
        sections = []
        
        if title:
            sections.append(f"# {title}")
        
        # Add the main text content
        if extracted_content.text:
            sections.append(extracted_content.text)
        
        # Add table information
        if extracted_content.tables:
            sections.append("\n## Tables")
            for idx, table in enumerate(extracted_content.tables):
                table_text = [f"### {table['title']}"]
                
                if table['headers'] and table['rows']:
                    # Format as a markdown table
                    header_row = "| " + " | ".join(table['headers']) + " |"
                    separator = "|-" + "-|-".join("-" * len(h) for h in table['headers']) + "-|"
                    
                    table_text.append(header_row)
                    table_text.append(separator)
                    
                    for row in table['rows']:
                        # Ensure row has the right number of cells
                        row_data = row
                        if isinstance(row_data[0], dict):  # Handle complex cell content
                            row_data = [cell['text'] if isinstance(cell, dict) else cell for cell in row]
                        
                        # Pad if needed
                        if len(row_data) < len(table['headers']):
                            row_data.extend([""] * (len(table['headers']) - len(row_data)))
                        
                        # Truncate if longer
                        row_data = row_data[:len(table['headers'])]
                        
                        row_text = "| " + " | ".join(str(cell) for cell in row_data) + " |"
                        table_text.append(row_text)
                
                sections.append("\n".join(table_text))
        
        # Add code blocks
        if extracted_content.code_blocks:
            sections.append("\n## Code Examples")
            for code_block in extracted_content.code_blocks:
                lang = code_block['language']
                content = code_block['content']
                
                if lang:
                    sections.append(f"```{lang}\n{content}\n```")
                else:
                    sections.append(f"```\n{content}\n```")
        
        # Add structured content
        if extracted_content.structured_content:
            sections.append("\n## Notes and Admonitions")
            for item in extracted_content.structured_content:
                if item['type'] == 'iframe':
                    sections.append(f"**{item['title']}**: {item['content']}")
                else:
                    sections.append(f"**{item['type'].upper()}: {item['title']}**\n{item['content']}")
        
        # Add image information (simplified)
        if extracted_content.images:
            sections.append("\n## Images")
            for img in extracted_content.images:
                img_desc = f"**Image**: {img['alt_text']}"
                if img['context']:
                    img_desc += f" - {img['context']}"
                sections.append(img_desc)
        
        return "\n\n".join(sections)


# ==================== CONFLUENCE SERVICE ====================

class ConfluenceService:
    """Enhanced Confluence service with intelligent content discovery and caching."""
    
    def __init__(self, 
                 base_url=Config.CONFLUENCE_URL,
                 username=Config.CONFLUENCE_USERNAME,
                 api_token=Config.CONFLUENCE_API_TOKEN,
                 spaces=Config.CONFLUENCE_SPACES):
        """
        Initialize the Confluence service.
        
        Args:
            base_url: Base URL for the Confluence instance
            username: Username for authentication
            api_token: API token for authentication
            spaces: List of space keys to search
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.api_url = f"{self.base_url}/wiki/rest/api"
        self.spaces = spaces
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "COPPER-AI-Python-Agent"
        }
        
        self.session = requests.Session()
        self.timeout = 30
        
        # Content extraction service
        self.content_extractor = ContentExtractorService()
        
        # Caches
        self.pages_cache = {}  # Space key -> list of pages
        self.content_cache = {}  # Page ID -> Page object
        
        # Thread lock for the caches
        self.cache_lock = threading.Lock()
        
        # Logger
        self.logger = loggers["confluence"]
    
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            self.logger.info("Testing connection to Confluence...")
            response = self.session.get(
                f"{self.api_url}/space",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            if response.status_code == 200:
                self.logger.info("Connection to Confluence successful!")
                return True
            else:
                self.logger.warning("Empty response received during connection test")
                return False
                
        except requests.RequestException as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def initialize(self):
        """Initialize the service by testing connection and loading cache."""
        # For demo purposes, we'll mock this functionality
        self.logger.info("Initializing mock Confluence service")
        
        # Create a mock page cache for the CMEIN space
        self.pages_cache["CMEIN"] = [
            PageMetadata(
                id="12345",
                title="CoPPER API Documentation",
                type="page",
                url=f"{self.base_url}/pages/viewpage.action?pageId=12345",
                space_key="CMEIN",
                labels=["API", "documentation", "CoPPER"],
                last_updated="2025-04-01",
                created_date="2025-01-15"
            ),
            PageMetadata(
                id="12346",
                title="SQL Views Reference",
                type="page",
                url=f"{self.base_url}/pages/viewpage.action?pageId=12346",
                space_key="CMEIN",
                labels=["SQL", "database", "CoPPER"],
                last_updated="2025-03-20",
                created_date="2025-01-20"
            )
        ]
        
        # Create a mock page cache for the CTS space
        self.pages_cache["CTS"] = [
            PageMetadata(
                id="23456",
                title="API Mapping Examples",
                type="page",
                url=f"{self.base_url}/pages/viewpage.action?pageId=23456",
                space_key="CTS",
                labels=["API", "mapping", "examples"],
                last_updated="2025-04-10",
                created_date="2025-02-01"
            )
        ]
        
        return True
    
    def _load_cache(self):
        """Load cached pages from file."""
        cache_path = Config.PAGE_CACHE_FILE
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    for space_key, pages in cached_data.items():
                        self.pages_cache[space_key] = pages
                        self.logger.info(f"Loaded {len(pages)} cached pages for space {space_key}")
                return True
            except Exception as e:
                self.logger.warning(f"Error reading cache file: {str(e)}")
        return False
    
    def _save_cache(self):
        """Save cached pages to file."""
        cache_path = Config.PAGE_CACHE_FILE
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(self.pages_cache, f)
                self.logger.info(f"Saved page cache to {cache_path}")
            return True
        except Exception as e:
            self.logger.warning(f"Error writing cache file: {str(e)}")
        return False
    
    def get_all_pages_in_space(self, space_key, force_refresh=False):
        """
        Get all pages in a Confluence space.
        
        Args:
            space_key: The space key to get pages from
            force_refresh: Whether to force a refresh of the cache
            
        Returns:
            List of page metadata
        """
        # Check if we have cached results
        if not force_refresh and space_key in self.pages_cache:
            self.logger.info(f"Using cached page list for {space_key} ({len(self.pages_cache[space_key])} pages)")
            return self.pages_cache[space_key]
        
        # In a real implementation, this would fetch pages from Confluence
        # For demo purposes, we'll return the mock data we set up in initialize()
        if space_key in self.pages_cache:
            return self.pages_cache[space_key]
        
        # If we don't have mock data for this space, return an empty list
        return []
    
    async def get_page_content(self, page_id):
        """
        Get the content of a page with advanced extraction.
        
        Args:
            page_id: The ID of the page
            
        Returns:
            Page object with metadata and content
        """
        # Check cache first
        with self.cache_lock:
            if page_id in self.content_cache:
                return self.content_cache[page_id]
        
        # For demo purposes, we'll create a mock page based on the ID
        # In a real implementation, this would fetch the page content from Confluence
        
        # Find the page metadata
        page_metadata = None
        for space_pages in self.pages_cache.values():
            for metadata in space_pages:
                if metadata.id == page_id:
                    page_metadata = metadata
                    break
            if page_metadata:
                break
        
        if not page_metadata:
            self.logger.error(f"Page not found with ID: {page_id}")
            return None
        
        # Create mock content based on the page title
        if "API Documentation" in page_metadata.title:
            html_content = """
            <html>
                <body>
                    <h1>CoPPER API Documentation</h1>
                    <p>This page provides documentation for the CoPPER API system.</p>
                    
                    <h2>API Endpoints</h2>
                    <ul>
                        <li>/api/products - Product information</li>
                        <li>/api/instruments - Financial instruments</li>
                        <li>/api/sessions - Trading sessions</li>
                    </ul>
                    
                    <h2>Request Format</h2>
                    <pre><code>
                    {
                        "reqName": "P1",
                        "dataDomain": "products",
                        "type": "independent",
                        "responseInclude": true,
                        "req": [
                            {
                                "tag": "productId",
                                "value": "123",
                                "operation": "EQ"
                            }
                        ]
                    }
                    </code></pre>
                </body>
            </html>
            """
        elif "SQL Views" in page_metadata.title:
            html_content = """
            <html>
                <body>
                    <h1>SQL Views Reference</h1>
                    <p>This page documents the SQL views available in the CoPPER database.</p>
                    
                    <h2>Common Views</h2>
                    <pre><code>
                    CREATE OR REPLACE VIEW PRODUCT_VIEW AS
                    SELECT 
                        p.product_id, 
                        p.product_name,
                        p.product_type,
                        c.category_name
                    FROM products p
                    JOIN categories c ON p.category_id = c.category_id
                    WHERE p.active_flag = 'Y';
                    </code></pre>
                </body>
            </html>
            """
        elif "API Mapping" in page_metadata.title:
            html_content = """
            <html>
                <body>
                    <h1>API Mapping Examples</h1>
                    <p>This page provides examples of mapping SQL views to API requests.</p>
                    
                    <h2>Example 1: Product View</h2>
                    <h3>SQL Query</h3>
                    <pre><code>
                    CREATE OR REPLACE VIEW PRODUCT_VIEW AS
                    SELECT 
                        p.product_id, 
                        p.product_name,
                        p.product_type,
                        c.category_name
                    FROM products p
                    JOIN categories c ON p.category_id = c.category_id
                    WHERE p.active_flag = 'Y';
                    </code></pre>
                    
                    <h3>Mapping o/p</h3>
                    <pre><code>
                    {
                        "reqName": "P1",
                        "dataDomain": "products",
                        "type": "independent",
                        "responseInclude": true,
                        "req": [
                            {
                                "tag": "activeFlag",
                                "value": "Y",
                                "operation": "EQ"
                            }
                        ],
                        "fields": ["productId", "productName", "productType", "categoryName"]
                    }
                    </code></pre>
                </body>
            </html>
            """
        else:
            html_content = f"""
            <html>
                <body>
                    <h1>{page_metadata.title}</h1>
                    <p>This is a mock page for demonstration purposes.</p>
                </body>
            </html>
            """
        
        # Extract content
        content = await self.content_extractor.extract_content_recursive(
            html_content,
            page_metadata.url,
            page_metadata.title
        )
        
        # Create Page object
        page = Page(metadata=page_metadata, content=content)
        
        # Cache the result
        with self.cache_lock:
            self.content_cache[page_id] = page
        
        return page
    
    async def search_confluence(self, query, limit=20):
        """
        Search Confluence using the built-in search API.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of page metadata matching the search
        """
        self.logger.info(f"Searching Confluence for: {query}")
        
        # In a real implementation, this would search Confluence
        # For demo purposes, we'll return pages that contain the query terms in their title
        
        results = []
        query_lower = query.lower()
        
        for space_key, pages in self.pages_cache.items():
            for page in pages:
                if query_lower in page.title.lower() or any(query_lower in label.lower() for label in page.labels):
                    results.append(page)
        
        return results[:limit]
    
    async def find_sql_views(self, view_name=None):
        """
        Find SQL view definitions in Confluence content.
        
        Args:
            view_name: Optional name of a specific view to find
            
        Returns:
            List of tuples with (page_id, view_name, view_definition)
        """
        self.logger.info(f"Searching for SQL views{' named ' + view_name if view_name else ''}")
        
        # For demo purposes, we'll create a mock SQL view
        view_definition = """
        CREATE OR REPLACE VIEW PRODUCT_VIEW AS
        SELECT 
            p.product_id, 
            p.product_name,
            p.product_type,
            c.category_name
        FROM products p
        JOIN categories c ON p.category_id = c.category_id
        WHERE p.active_flag = 'Y';
        """
        
        if view_name and view_name.upper() != "PRODUCT_VIEW":
            view_definition = f"""
            CREATE OR REPLACE VIEW {view_name.upper()} AS
            SELECT 
                t.transaction_id,
                t.transaction_date,
                t.amount,
                c.customer_name
            FROM transactions t
            JOIN customers c ON t.customer_id = c.customer_id
            WHERE t.status = 'COMPLETE';
            """
        
        # Find SQL-related pages
        sql_pages = []
        for space_key, pages in self.pages_cache.items():
            for page in pages:
                if "SQL" in page.title or "SQL" in " ".join(page.labels):
                    sql_pages.append(page)
        
        # Use the first SQL page, or default to the first page
        page_id = sql_pages[0].id if sql_pages else list(self.pages_cache.values())[0][0].id
        view_name = view_name or "PRODUCT_VIEW"
        
        return [(page_id, view_name, view_definition)]
    
    async def find_view_examples(self):
        """
        Find examples of view-to-API mappings in the documentation.
        
        Returns:
            Dictionary of examples with view names as keys
        """
        self.logger.info("Searching for view-to-API mapping examples")
        
        # For demo purposes, we'll create a mock example
        examples = {
            "PRODUCT_VIEW": {
                "sql": """
                CREATE OR REPLACE VIEW PRODUCT_VIEW AS
                SELECT 
                    p.product_id, 
                    p.product_name,
                    p.product_type,
                    c.category_name
                FROM products p
                JOIN categories c ON p.category_id = c.category_id
                WHERE p.active_flag = 'Y';
                """,
                "json": """
                {
                    "reqName": "P1",
                    "dataDomain": "products",
                    "type": "independent",
                    "responseInclude": true,
                    "req": [
                        {
                            "tag": "activeFlag",
                            "value": "Y",
                            "operation": "EQ"
                        }
                    ],
                    "fields": ["productId", "productName", "productType", "categoryName"]
                }
                """,
                "page_title": "API Mapping Examples",
                "page_id": "23456"
            }
        }
        
        return examples


# ==================== SQL PARSER SERVICE ====================

class SqlParserService:
    """Advanced SQL parser for extracting detailed information from view definitions."""
    
    def __init__(self):
        """Initialize the SQL parser service."""
        self.logger = loggers["sql"]
    
    def parse_view_definition(self, sql_text) -> SqlView:
        """
        Parse a SQL view definition into a structured SqlView object.
        
        Args:
            sql_text: The SQL view definition text
            
        Returns:
            SqlView object containing parsed information
        """
        self.logger.info("Parsing SQL view definition")
        
        view = SqlView(name="", definition=sql_text)
        
        # Clean up the SQL text - normalize whitespace
        clean_sql = re.sub(r'\s+', ' ', sql_text).strip()
        
        # Extract view name
        view_name_match = re.search(r'VIEW\s+(\w+)', clean_sql, re.IGNORECASE)
        if view_name_match:
            view.name = view_name_match.group(1)
            self.logger.info(f"Found view name: {view.name}")
        
        # Handle CREATE OR REPLACE VIEW syntax
        create_view_match = re.search(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:FORCE\s+)?VIEW\s+(\w+)(?:\s*\((.*?)\))?\s+AS\s+(.*)', 
            clean_sql, re.IGNORECASE | re.DOTALL
        )
        
        if create_view_match:
            view.name = create_view_match.group(1)
            
            # Extract column definitions if provided inline
            if create_view_match.group(2):
                columns_text = create_view_match.group(2)
                columns = self._parse_column_definitions(columns_text)
                view.columns = columns
            
            # Extract the SELECT statement
            select_stmt = create_view_match.group(3)
            
            # Parse the SELECT statement for columns if not already parsed
            if not view.columns:
                view.columns = self._parse_select_columns(select_stmt)
            
            # Extract FROM clause and source tables
            from_clause, source_tables = self._parse_from_clause(select_stmt)
            view.source_tables = source_tables
            
            # Extract JOIN conditions
            join_conditions = self._parse_join_conditions(select_stmt)
            view.join_conditions = join_conditions
            
            # Extract WHERE clause conditions
            filter_conditions = self._parse_where_conditions(select_stmt)
            view.filter_conditions = filter_conditions
            
            # Extract transformations (CASE statements, functions, etc.)
            transformations = self._parse_transformations(select_stmt)
            view.transformations = transformations
            
            # Determine primary domain based on view name and tables
            primary_domain = self._determine_primary_domain(view.name, view.source_tables)
            view.primary_domain = primary_domain
            
            # Extract relationships between tables
            relationships = self._analyze_table_relationships(view.join_conditions)
            view.relationships = relationships
        
        return view
    
    def _parse_column_definitions(self, columns_text):
        """Parse column definitions from the view's column list."""
        columns = []
        
        # Split by commas outside of parentheses
        paren_level = 0
        current_col = ""
        
        for char in columns_text:
            if char == '(':
                paren_level += 1
                current_col += char
            elif char == ')':
                paren_level -= 1
                current_col += char
            elif char == ',' and paren_level == 0:
                columns.append(current_col.strip())
                current_col = ""
            else:
                current_col += char
        
        if current_col:
            columns.append(current_col.strip())
        
        # Process each column definition
        parsed_columns = []
        for col in columns:
            # Check for AS clause to determine column name and source
            as_match = re.search(r'(.*?)\s+AS\s+(\w+)$', col, re.IGNORECASE)
            if as_match:
                source = as_match.group(1).strip()
                name = as_match.group(2).strip()
                parsed_columns.append({
                    "name": name,
                    "source": source,
                    "type": self._infer_column_type(source)
                })
            else:
                parsed_columns.append({
                    "name": col.strip(),
                    "source": col.strip(),
                    "type": "unknown"
                })
        
        return parsed_columns
    
    def _parse_select_columns(self, select_stmt):
        """Parse columns from the SELECT clause."""
        columns = []
        
        # Extract the column list portion
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', select_stmt, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return columns
        
        columns_text = select_match.group(1).strip()
        
        # Handle SELECT * case
        if columns_text == '*':
            return [{
                "name": "*",
                "source": "*",
                "type": "unknown"
            }]
        
        # Split by commas outside of parentheses
        paren_level = 0
        current_col = ""
        
        for char in columns_text:
            if char == '(':
                paren_level += 1
                current_col += char
            elif char == ')':
                paren_level -= 1
                current_col += char
            elif char == ',' and paren_level == 0:
                columns.append(current_col.strip())
                current_col = ""
            else:
                current_col += char
        
        if current_col:
            columns.append(current_col.strip())
        
        # Process each column
        parsed_columns = []
        for col in columns:
            # Check for AS clause
            as_match = re.search(r'(.*?)\s+AS\s+(\w+)$', col, re.IGNORECASE)
            if as_match:
                source = as_match.group(1).strip()
                name = as_match.group(2).strip()
                parsed_columns.append({
                    "name": name,
                    "source": source,
                    "type": self._infer_column_type(source)
                })
            else:
                # If no AS clause, use the expression as is
                name = col.strip()
                
                # Try to extract just the column name if it's a qualified name
                col_name_match = re.search(r'(\w+)\.(\w+)$', name)
                if col_name_match:
                    name = col_name_match.group(2)
                
                parsed_columns.append({
                    "name": name,
                    "source": col.strip(),
                    "type": self._infer_column_type(col)
                })
        
        return parsed_columns
    
    def _infer_column_type(self, source):
        """Infer the column type based on the expression."""
        source = source.upper()
        
        if "DATE" in source or "TIME" in source or "TIMESTAMP" in source:
            return "date/time"
        elif "SUM(" in source or "AVG(" in source or "COUNT(" in source or "MIN(" in source or "MAX(" in source:
            return "numeric"
        elif "NUMBER" in source or "NUMERIC" in source or "INTEGER" in source or "INT" in source:
            return "numeric"
        elif "VARCHAR" in source or "CHAR" in source or "TEXT" in source or "STRING" in source:
            return "string"
        elif "BOOLEAN" in source or "BOOL" in source:
            return "boolean"
        elif "CASE" in source:
            # Try to determine the type from the CASE expression
            if "DATE" in source or "TIME" in source:
                return "date/time"
            elif any(num in source for num in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]):
                return "numeric"
            else:
                return "string"
        else:
            return "unknown"
    
    def _parse_from_clause(self, select_stmt):
        """Parse the FROM clause to extract source tables."""
        from_clause = ""
        source_tables = []
        
        # Extract FROM clause
        from_match = re.search(r'FROM\s+(.*?)(?:WHERE|GROUP BY|ORDER BY|HAVING|$)', 
                               select_stmt, re.IGNORECASE | re.DOTALL)
        
        if from_match:
            from_clause = from_match.group(1).strip()
            
            # Extract table names (accounting for aliases)
            table_matches = re.findall(r'(?:FROM|JOIN)\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?', 
                                      from_clause, re.IGNORECASE)
            
            for match in table_matches:
                table_name = match[0]
                alias = match[1] if match[1] else table_name
                
                source_tables.append({
                    "name": table_name,
                    "alias": alias
                })
        
        return from_clause, source_tables
    
    def _parse_join_conditions(self, select_stmt):
        """Parse JOIN conditions from the SQL statement."""
        join_conditions = []
        
        # Extract FROM clause (which contains the JOINs)
        from_match = re.search(r'FROM\s+(.*?)(?:WHERE|GROUP BY|ORDER BY|HAVING|$)', 
                               select_stmt, re.IGNORECASE | re.DOTALL)
        
        if from_match:
            from_clause = from_match.group(1).strip()
            
            # Find all JOIN statements
            join_matches = re.findall(
                r'((?:INNER|LEFT|RIGHT|FULL|OUTER)?\s*JOIN\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+ON\s+(.*?)(?=(?:INNER|LEFT|RIGHT|FULL|OUTER)?\s*JOIN|\s*WHERE|\s*GROUP BY|\s*ORDER BY|\s*$))', 
                from_clause, re.IGNORECASE | re.DOTALL
            )
            
            for match in join_matches:
                full_join = match[0].strip()
                table_name = match[1]
                alias = match[2] if match[2] else table_name
                condition = match[3].strip()
                
                # Determine join type
                join_type = "INNER"  # Default
                if "LEFT" in full_join.upper():
                    join_type = "LEFT"
                elif "RIGHT" in full_join.upper():
                    join_type = "RIGHT"
                elif "FULL" in full_join.upper():
                    join_type = "FULL"
                
                # Parse the ON condition
                join_condition = {
                    "type": join_type,
                    "table": table_name,
                    "alias": alias,
                    "condition": condition,
                    "left_side": "",
                    "right_side": ""
                }
                
                # Try to extract the specific columns being joined
                eq_match = re.search(r'(\w+(?:\.\w+)?)\s*=\s*(\w+(?:\.\w+)?)', condition)
                if eq_match:
                    join_condition["left_side"] = eq_match.group(1)
                    join_condition["right_side"] = eq_match.group(2)
                
                join_conditions.append(join_condition)
        
        return join_conditions
    
    def _parse_where_conditions(self, select_stmt):
        """Parse WHERE conditions from the SQL statement."""
        filter_conditions = []
        
        # Extract WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|HAVING|$)', 
                               select_stmt, re.IGNORECASE | re.DOTALL)
        
        if where_match:
            where_clause = where_match.group(1).strip()
            
            # Split conditions by AND/OR, respecting parentheses
            conditions = self._split_conditions(where_clause)
            
            for condition in conditions:
                parsed_condition = self._parse_single_condition(condition.strip())
                if parsed_condition:
                    filter_conditions.append(parsed_condition)
        
        return filter_conditions
    
    def _split_conditions(self, conditions_text):
        """Split conditions by AND/OR while respecting parentheses."""
        conditions = []
        
        # This is complex due to nested conditions
        # For simplicity, we'll use a basic approach that works for most cases
        
        paren_level = 0
        current_condition = ""
        i = 0
        
        while i < len(conditions_text):
            char = conditions_text[i]
            
            if char == '(':
                paren_level += 1
                current_condition += char
            elif char == ')':
                paren_level -= 1
                current_condition += char
            # Check for AND or OR but only at paren_level 0
            elif paren_level == 0 and i + 2 < len(conditions_text):
                if conditions_text[i:i+3].upper() == 'AND':
                    if current_condition.strip():
                        conditions.append(current_condition.strip())
                    current_condition = ""
                    i += 2  # Skip "AND"
                elif conditions_text[i:i+2].upper() == 'OR':
                    if current_condition.strip():
                        conditions.append(current_condition.strip())
                    current_condition = ""
                    i += 1  # Skip "OR"
                else:
                    current_condition += char
            else:
                current_condition += char
            
            i += 1
        
        if current_condition.strip():
            conditions.append(current_condition.strip())
        
        return conditions
    
    def _parse_single_condition(self, condition):
        """Parse a single WHERE condition."""
        parsed = {
            "raw_condition": condition,
            "field": "",
            "operator": "",
            "value": "",
            "type": "unknown"
        }
        
        # Check for IS NULL / IS NOT NULL
        null_match = re.search(r'(\w+(?:\.\w+)?)\s+(IS\s+(?:NOT\s+)?NULL)', condition, re.IGNORECASE)
        if null_match:
            field = null_match.group(1)
            operator = null_match.group(2).upper()
            
            parsed["field"] = field
            parsed["operator"] = operator
            parsed["value"] = None
            parsed["type"] = "null_check"
            
            return parsed
        
        # Check for IN condition
        in_match = re.search(r'(\w+(?:\.\w+)?)\s+IN\s*\((.*?)\)', condition, re.IGNORECASE)
        if in_match:
            field = in_match.group(1)
            values_str = in_match.group(2)
            
            # Parse the values
            values = []
            for value in re.findall(r'\'([^\']*?)\'|"([^"]*?)"|(\w+)', values_str):
                val = value[0] or value[1] or value[2]
                if val and not val.upper() in ['AND', 'OR', 'IN']:
                    values.append(val)
            
            parsed["field"] = field
            parsed["operator"] = "IN"
            parsed["value"] = values
            parsed["type"] = "in_list"
            
            return parsed
        
        # Check for BETWEEN condition
        between_match = re.search(r'(\w+(?:\.\w+)?)\s+BETWEEN\s+(.*?)\s+AND\s+(.*)', condition, re.IGNORECASE)
        if between_match:
            field = between_match.group(1)
            min_val = between_match.group(2).strip()
            max_val = between_match.group(3).strip()
            
            # Remove quotes if present
            if (min_val.startswith("'") and min_val.endswith("'")) or \
               (min_val.startswith('"') and min_val.endswith('"')):
                min_val = min_val[1:-1]
            
            if (max_val.startswith("'") and max_val.endswith("'")) or \
               (max_val.startswith('"') and max_val.endswith('"')):
                max_val = max_val[1:-1]
            
            parsed["field"] = field
            parsed["operator"] = "BETWEEN"
            parsed["value"] = [min_val, max_val]
            parsed["type"] = "range"
            
            return parsed
        
        # Check for comparison operators
        operators = ["=", "!=", "<>", "<", "<=", ">", ">=", "LIKE"]
        for op in operators:
            if f" {op} " in f" {condition} ":  # Add spaces to ensure we match whole operators
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    field = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Handle qualified field names
                    if "." in field:
                        field_parts = field.split(".", 1)
                        table_alias = field_parts[0]
                        field_name = field_parts[1]
                        parsed["table_alias"] = table_alias
                        field = field_name
                    
                    # Remove quotes from value if present
                    if (value.startswith("'") and value.endswith("'")) or \
                       (value.startswith('"') and value.endswith('"')):
                        value = value[1:-1]
                    
                    parsed["field"] = field
                    parsed["operator"] = op
                    parsed["value"] = value
                    
                    # Determine type
                    if op == "LIKE":
                        parsed["type"] = "pattern_match"
                    else:
                        parsed["type"] = "comparison"
                    
                    return parsed
        
        # If we got here, we couldn't parse it as a standard condition
        return parsed
    
    def _parse_transformations(self, select_stmt):
        """Parse transformations like CASE statements and functions."""
        transformations = []
        
        # Extract CASE statements
        case_matches = re.findall(r'(CASE\s+(?:WHEN\s+.*?\s+THEN\s+.*?\s+)+(?:ELSE\s+.*?\s+)?END)', 
                                 select_stmt, re.IGNORECASE | re.DOTALL)
        
        for case_stmt in case_matches:
            transformations.append({
                "type": "case_statement",
                "expression": case_stmt
            })
        
        # Extract function calls (excluding common aggregates)
        function_matches = re.findall(r'(\w+\s*\([^)]*\))', select_stmt)
        
        for func in function_matches:
            # Skip common aggregates and functions
            common_funcs = ['COUNT(', 'SUM(', 'MIN(', 'MAX(', 'AVG(', 'CAST(', 'CONVERT(']
            if not any(func.upper().startswith(cf) for cf in common_funcs):
                transformations.append({
                    "type": "function",
                    "expression": func
                })
        
        return transformations
    
    def _determine_primary_domain(self, view_name, source_tables):
        """Determine the primary domain based on view name and source tables."""
        view_name_upper = view_name.upper()
        
        # Check view name for domain indicators
        for domain, domain_name in Config.DOMAIN_MAPPINGS.items():
            if domain in view_name_upper:
                return domain_name
        
        # Check source tables for domain indicators
        domain_scores = {}
        for domain, domain_name in Config.DOMAIN_MAPPINGS.items():
            score = 0
            for table in source_tables:
                if domain in table["name"].upper():
                    score += 2
                if domain in table["alias"].upper():
                    score += 1
            
            if score > 0:
                domain_scores[domain_name] = score
        
        # Return the domain with the highest score, or unknown
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        # Default fallback patterns
        if "PROD" in view_name_upper:
            return "products"
        elif "INST" in view_name_upper:
            return "instruments"
        elif "SESS" in view_name_upper:
            return "sessions"
        elif "FIRM" in view_name_upper:
            return "firms"
        
        return "unknown"
    
    def _analyze_table_relationships(self, join_conditions):
        """Analyze relationships between tables from join conditions."""
        relationships = {}
        
        for join in join_conditions:
            left_side = join["left_side"]
            right_side = join["right_side"]
            
            # Skip if we don't have both sides
            if not left_side or not right_side:
                continue
            
            # Extract table aliases
            left_parts = left_side.split(".")
            right_parts = right_side.split(".")
            
            if len(left_parts) == 2 and len(right_parts) == 2:
                left_table = left_parts[0]
                right_table = right_parts[0]
                
                # Add relationship
                if left_table not in relationships:
                    relationships[left_table] = []
                
                if right_table not in relationships:
                    relationships[right_table] = []
                
                relationships[left_table].append(right_table)
                relationships[right_table].append(left_table)
        
        return relationships


# ==================== API MAPPING SERVICE ====================

class ApiMappingService:
    """Service for generating and managing API mappings from SQL views."""
    
    def __init__(self):
        """Initialize the API mapping service."""
        self.logger = loggers["api"]
        self.sql_parser = SqlParserService()
        
        # Cache for mappings
        self.mapping_cache = {}
        self.examples_cache = {}
        
        # Load cache if available
        self._load_cache()
    
    def _load_cache(self):
        """Load cached mappings from file."""
        cache_path = Config.MAPPING_CACHE_FILE
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    self.mapping_cache = cached_data.get("mappings", {})
                    self.examples_cache = cached_data.get("examples", {})
                    self.logger.info(f"Loaded {len(self.mapping_cache)} cached mappings")
                    self.logger.info(f"Loaded {len(self.examples_cache)} cached examples")
            except Exception as e:
                self.logger.warning(f"Error reading mapping cache: {str(e)}")
    
    def _save_cache(self):
        """Save cached mappings to file."""
        cache_path = Config.MAPPING_CACHE_FILE
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump({
                    "mappings": self.mapping_cache,
                    "examples": self.examples_cache
                }, f)
                self.logger.info(f"Saved mapping cache with {len(self.mapping_cache)} mappings")
        except Exception as e:
            self.logger.warning(f"Error writing mapping cache: {str(e)}")
    
    def add_examples(self, examples):
        """
        Add mapping examples to the cache.
        
        Args:
            examples: Dictionary of examples with view names as keys
        """
        self.examples_cache.update(examples)
        self._save_cache()
    
    def get_example_for_view(self, view_name=None, domain=None):
        """
        Get a relevant example for a view or domain.
        
        Args:
            view_name: Optional name of the view to find an example for
            domain: Optional domain to find an example for
            
        Returns:
            Most relevant example or None
        """
        # First check for exact match by view name
        if view_name and view_name in self.examples_cache:
            return self.examples_cache[view_name]
        
        # Then check for similar view names
        if view_name:
            view_name_upper = view_name.upper()
            for example_name, example in self.examples_cache.items():
                if example_name.upper() in view_name_upper or view_name_upper in example_name.upper():
                    return example
        
        # Then check for domain match
        if domain:
            for example_name, example in self.examples_cache.items():
                example_json = example.get("json", "{}")
                if f"\"dataDomain\": \"{domain}\"" in example_json:
                    return example
        
        # If no matches, return the first example if any exist
        if self.examples_cache:
            return next(iter(self.examples_cache.values()))
        
        return None
    
    def generate_mapping(self, view, examples=None):
        """
        Generate an API mapping for a SQL view.
        
        Args:
            view: SqlView object containing the parsed view
            examples: Optional dictionary of examples to use
            
        Returns:
            ApiMapping object containing the mapping
        """
        self.logger.info(f"Generating API mapping for view: {view.name}")
        
        # Check cache first
        if view.name in self.mapping_cache:
            cached_mapping = self.mapping_cache[view.name]
            self.logger.info(f"Using cached mapping for view: {view.name}")
            return ApiMapping(**cached_mapping)
        
        # Determine API endpoint
        api_endpoint = f"/api/{view.primary_domain}"
        self.logger.info(f"Determined API endpoint: {api_endpoint}")
        
        # Determine request type (dependent or independent)
        req_type = "independent"
        if "instruments" in view.primary_domain or "sessions" in view.primary_domain:
            req_type = "dependent"
        
        # Generate request parameters from filter conditions
        request_parameters = []
        for condition in view.filter_conditions:
            param = self._condition_to_param(condition)
            if param:
                request_parameters.append(param)
        
        # Add domain-specific default parameters
        self._add_domain_defaults(request_parameters, view.primary_domain)
        
        # Determine req_name (first letter of domain + index)
        req_name = view.primary_domain[0].upper() + "1"
        
        # Generate field list from columns
        fields = []
        for column in view.columns:
            # Handle special case for '*' - use none
            if column["name"] == "*":
                continue
                
            # Convert to camelCase for API fields
            col_name = column["name"]
            if "_" in col_name:
                parts = col_name.lower().split("_")
                camel_case = parts[0] + "".join(p.capitalize() for p in parts[1:])
                fields.append(camel_case)
            else:
                fields.append(col_name.lower())
        
        # Create the mapping
        mapping = ApiMapping(
            view_name=view.name,
            api_endpoint=api_endpoint,
            req_name=req_name,
            data_domain=view.primary_domain,
            type=req_type,
            request_parameters=request_parameters,
            fields=fields,
            confidence_score=0.8  # Default confidence
        )
        
        # Cache the mapping
        self.mapping_cache[view.name] = mapping.__dict__
        self._save_cache()
        
        return mapping
    
    def _condition_to_param(self, condition):
        """Convert a parsed WHERE condition to an API request parameter."""
        # Skip if field is empty
        if not condition["field"]:
            return None
        
        # Map SQL operators to API operations
        op_map = Config.SQL_TO_API_OPS
        
        field = condition["field"].lower()
        
        # Handle IS NULL / IS NOT NULL
        if condition["operator"] == "IS NULL":
            return {
                "tag": field,
                "operation": "ISNULL"
            }
        
        if condition["operator"] == "IS NOT NULL":
            return {
                "tag": field,
                "operation": "NOTNULL"
            }
        
        # Handle IN condition
        if condition["operator"] == "IN":
            return {
                "tag": field,
                "value": condition["value"],
                "operation": "IN"
            }
        
        # Handle BETWEEN condition
        if condition["operator"] == "BETWEEN":
            # BETWEEN is usually implemented as two separate conditions in the API
            min_val, max_val = condition["value"]
            return [
                {
                    "tag": field,
                    "value": min_val,
                    "operation": "GTE",
                    "predicate": "AND"
                },
                {
                    "tag": field,
                    "value": max_val,
                    "operation": "LTE",
                    "predicate": "AND"
                }
            ]
        
        # Handle other operators
        if condition["operator"] in op_map:
            api_op = op_map[condition["operator"]]
            
            return {
                "tag": field,
                "value": condition["value"],
                "operation": api_op
            }
        
        return None
    
    def _add_domain_defaults(self, request_parameters, domain):
        """Add domain-specific default parameters."""
        defaults = {
            "firms": [
                {"tag": "effEndTimp", "value": "TODAY", "operation": "GT"}
            ],
            "sessions": [
                {"tag": "sessionFeature.features", "value": ["BTEC-EU", "BTEC-US"], "operation": "IN"}
            ],
            "instruments": [
                {"tag": "prodTyp", "value": "BOND", "operation": "EQ"}
            ],
            "products": [
                {"tag": "emrCf", "operation": "NOTNULL"}
            ],
            "risk": [
                {"tag": "riskGrp", "operation": "NOTNULL", "predicate": "AND"}
            ]
        }
        
        if domain in defaults:
            for param in defaults[domain]:
                # Check if a similar parameter is already added
                existing = False
                for req in request_parameters:
                    if isinstance(req, dict) and req.get("tag") == param["tag"]:
                        existing = True
                        break
                
                if not existing:
                    request_parameters.append(param)
    
    def generate_json_body(self, mapping):
        """
        Generate a complete JSON request body from an API mapping.
        
        Args:
            mapping: ApiMapping object
            
        Returns:
            Dictionary containing the complete API JSON request body
        """
        self.logger.info(f"Generating JSON request body for mapping: {mapping.view_name}")
        
        # Create the JSON body
        body = {
            "reqName": mapping.req_name,
            "dataDomain": mapping.data_domain,
            "type": mapping.type,
            "responseInclude": True,
            "distinct": True,
            "req": mapping.request_parameters
        }
        
        # Add fields if present
        if mapping.fields:
            body["fields"] = mapping.fields
        
        return body
    
    def generate_formatted_json(self, mapping):
        """
        Generate a formatted JSON string from an API mapping.
        
        Args:
            mapping: ApiMapping object
            
        Returns:
            Formatted JSON string
        """
        body = self.generate_json_body(mapping)
        return json.dumps(body, indent=2)
    
    def generate_mapping_explanation(self, view, mapping):
        """
        Generate a human-readable explanation of the mapping.
        
        Args:
            view: SqlView object
            mapping: ApiMapping object
            
        Returns:
            String containing the explanation
        """
        self.logger.info(f"Generating mapping explanation for view: {view.name}")
        
        explanation = []
        
        # Basic information
        explanation.append(f"## Mapping Explanation for {view.name}")
        explanation.append(f"\nThis view maps to the {mapping.data_domain} domain using the {mapping.api_endpoint} endpoint.")
        
        # Mapping strategy
        mapping_strategy = self._determine_mapping_strategy(view, mapping)
        explanation.append(f"\n### Mapping Strategy: {mapping_strategy['name']}")
        explanation.append(mapping_strategy["description"])
        
        # Request parameters
        explanation.append(f"\n### Request Parameters")
        for param in mapping.request_parameters:
            param_desc = f"- `{param['tag']}`: "
            if "operation" in param:
                param_desc += f"{param['operation']} "
            if "value" in param:
                if isinstance(param["value"], list):
                    param_desc += f"{', '.join(str(v) for v in param['value'])}"
                else:
                    param_desc += f"{param['value']}"
            explanation.append(param_desc)
        
        # Source mapping
        explanation.append(f"\n### SQL to API Mapping Details")
        
        # Map WHERE clauses to req parameters
        if view.filter_conditions:
            explanation.append(f"\nWHERE conditions mapped to request parameters:")
            for condition in view.filter_conditions:
                # Find matching parameter
                for param in mapping.request_parameters:
                    if "tag" in param and condition["field"].lower() == param["tag"].lower():
                        sql_cond = condition["raw_condition"]
                        api_param = f"{param}"
                        explanation.append(f"- SQL: `{sql_cond}` → API: `{api_param}`")
                        break
        
        # Notes on joins if present
        if view.join_conditions:
            explanation.append(f"\nJOIN conditions consolidated into request:")
            explanation.append("The API consolidates data from multiple tables, eliminating the need for explicit joins.")
            
            for join in view.join_conditions:
                join_desc = f"- JOIN {join['table']} ({join['type']}) ON {join['condition']}"
                explanation.append(join_desc)
        
        # Fields mapping
        if mapping.fields:
            explanation.append(f"\n### Column to Field Mapping")
            for i, column in enumerate(view.columns):
                if i < len(mapping.fields):
                    explanation.append(f"- SQL: `{column['name']}` → API: `{mapping.fields[i]}`")
        
        return "\n".join(explanation)
    
    def _determine_mapping_strategy(self, view, mapping):
        """Determine the mapping strategy based on view characteristics."""
        strategies = {
            "direct_mapping": {
                "name": "Direct Mapping",
                "description": "The view structure maps directly to API parameters without complex transformations."
            },
            "complex_filter": {
                "name": "Complex Filter Mapping",
                "description": "The view contains multiple filter conditions that have been converted to API request parameters."
            },
            "join_consolidation": {
                "name": "Join Consolidation",
                "description": "The view joins multiple tables, which have been consolidated into a single API request."
            },
            "transformation_mapping": {
                "name": "Transformation Mapping",
                "description": "The view contains transformations (like CASE statements) that require special handling in the API request."
            }
        }
        
        # Determine strategy based on view characteristics
        if len(view.transformations) > 0:
            return strategies["transformation_mapping"]
        elif len(view.join_conditions) > 1:
            return strategies["join_consolidation"]
        elif len(view.filter_conditions) > 2:
            return strategies["complex_filter"]
        else:
            return strategies["direct_mapping"]


# ==================== LLM SERVICE ====================

class LLMService:
    """Service for interacting with Gemini/Vertex AI."""
    
    def __init__(self, 
                 model_name=Config.MODEL_NAME, 
                 project_id=Config.PROJECT_ID,
                 location=Config.LOCATION):
        """
        Initialize the LLM service.
        
        Args:
            model_name: Name of the Gemini model to use
            project_id: GCP project ID
            location: GCP location
        """
        self.model_name = model_name
        self.project_id = project_id
        self.location = location
        self.logger = loggers["llm"]
        
        # Initialize Vertex AI if available
        self.model = None
        if VERTEX_AVAILABLE:
            try:
                vertexai.init(project=project_id, location=location)
                self.model = GenerativeModel(model_name)
                self.logger.info(f"Initialized LLM service with model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize VertexAI: {str(e)}")
        else:
            self.logger.info("Using fallback LLM service (VertexAI not available)")
    
    async def analyze_sql_view(self, view_definition):
        """
        Analyze a SQL view definition using Gemini.
        
        Args:
            view_definition: The SQL view definition text
            
        Returns:
            String containing the analysis
        """
        self.logger.info("Analyzing SQL view with Gemini")
        
        prompt = PromptTemplates.ANALYZE_SQL_VIEW.format(
            view_definition=view_definition
        )
        
        system_prompt = PromptTemplates.SYSTEM_PERSONALITY
        
        response = await self._generate_content(system_prompt, prompt)
        return response
    
    async def generate_api_mapping(self, view_definition, view_analysis, examples=""):
        """
        Generate an API mapping based on SQL view and examples.
        
        Args:
            view_definition: The SQL view definition text
            view_analysis: Analysis of the view
            examples: Optional examples of similar mappings
            
        Returns:
            String containing the mapping JSON
        """
        self.logger.info("Generating API mapping with Gemini")
        
        prompt = PromptTemplates.GENERATE_API_MAPPING.format(
            view_definition=view_definition,
            view_analysis=view_analysis,
            examples=examples
        )
        
        system_prompt = PromptTemplates.SYSTEM_PERSONALITY
        
        response = await self._generate_content(system_prompt, prompt)
        return response
    
    async def explain_mapping(self, view_definition, api_mapping):
        """
        Generate an explanation of the mapping between a view and API.
        
        Args:
            view_definition: The SQL view definition
            api_mapping: The API mapping JSON
            
        Returns:
            String containing the explanation
        """
        self.logger.info("Explaining API mapping with Gemini")
        
        prompt = PromptTemplates.EXPLAIN_MAPPING.format(
            view_definition=view_definition,
            api_mapping=api_mapping
        )
        
        system_prompt = PromptTemplates.SYSTEM_PERSONALITY
        
        response = await self._generate_content(system_prompt, prompt)
        return response
    
    async def answer_general_question(self, question, context):
        """
        Answer a general question about CoPPER using provided context.
        
        Args:
            question: The user's question
            context: Relevant context information
            
        Returns:
            String containing the answer
        """
        self.logger.info("Answering general question with Gemini")
        
        prompt = PromptTemplates.ANSWER_GENERAL_QUESTION.format(
            question=question,
            context=context
        )
        
        system_prompt = PromptTemplates.SYSTEM_PERSONALITY
        
        response = await self._generate_content(system_prompt, prompt)
        return response
    
    async def _generate_content(self, system_prompt, prompt):
        """
        Generate content using Gemini with error handling and retries.
        
        Args:
            system_prompt: The system prompt
            prompt: The user prompt
            
        Returns:
            String containing the generated content
        """
        try:
            if self.model and VERTEX_AVAILABLE:
                # Create generation config
                generation_config = GenerationConfig(
                    temperature=0.2,  # Low temperature for precise outputs
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=8192  # Allow for longer responses
                )
                
                # Create content with system prompt
                system_content = Content(
                    role="system",
                    parts=[{"text": system_prompt}]
                )
                
                user_content = Content(
                    role="user", 
                    parts=[{"text": prompt}]
                )
                
                # Generate response
                response = self.model.generate_content(
                    contents=[system_content, user_content],
                    generation_config=generation_config,
                )
                
                if hasattr(response, 'text'):
                    return response.text
                elif response.candidates and response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text
                else:
                    self.logger.warning("Empty response from Gemini")
                    return "I couldn't generate a response. Please try rephrasing or providing more information."
            else:
                # Fallback response generation for when VertexAI is not available
                return self._generate_fallback_response(prompt)
                
        except Exception as e:
            self.logger.error(f"Error generating content with Gemini: {str(e)}")
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt):
        """Generate fallback responses when VertexAI is not available."""
        self.logger.info("Generating fallback response")
        
        # Extract key information from the prompt to customize the response
        if "SQL View Analysis Request" in prompt:
            return """
# SQL View Analysis

This view appears to be selecting data from product-related tables, joining them with category information.

## Main Purpose
The view provides a consolidated view of product information along with their categories.

## Columns
- product_id: numeric (unique identifier)
- product_name: string (name of the product)
- product_type: string (classification of the product)
- category_name: string (name of the category from the joined table)

## Source Tables
- products (aliased as p): Contains the core product information
- categories (aliased as c): Contains category information

## JOIN Conditions
LEFT JOIN between products and categories on category_id

## WHERE Conditions
The view only includes active products (active_flag = 'Y')

## Primary Domain
This view belongs to the "products" domain based on the primary table and data.

## Business Rules
- Only active products are shown in the view
- Products are associated with their categories through the category_id relationship
"""
        elif "API Mapping Request" in prompt:
            return """
```json
{
  "reqName": "P1",
  "dataDomain": "products",
  "type": "independent",
  "responseInclude": true,
  "distinct": true,
  "req": [
    {
      "tag": "activeFlag",
      "value": "Y",
      "operation": "EQ"
    }
  ],
  "fields": ["productId", "productName", "productType", "categoryName"]
}
```
"""
        elif "Mapping Explanation Request" in prompt:
            return """
# Mapping Explanation

## Overview
The SQL view `PRODUCT_VIEW` has been mapped to the `/api/products` endpoint. This mapping preserves the core functionality of the view while adapting it to the API's request structure.

## Table to Domain Mapping
The primary table `products` in the SQL view directly maps to the `products` data domain in the API. This is a straightforward mapping, as the view primarily focuses on product data.

## WHERE Condition Translation
The SQL WHERE condition `active_flag = 'Y'` has been translated to an API request parameter with:
- tag: "activeFlag"
- value: "Y"
- operation: "EQ" (representing equals)

This ensures that only active products are returned, matching the behavior of the SQL view.

## Field Mapping
The columns in the SELECT clause have been mapped to API fields:
- SQL: `product_id` → API: `productId`
- SQL: `product_name` → API: `productName`
- SQL: `product_type` → API: `productType`
- SQL: `category_name` → API: `categoryName`

Notice the conversion from snake_case in SQL to camelCase in the API, which follows the API naming convention.

## JOIN Handling
While the SQL view uses a JOIN to connect `products` and `categories` tables, the API consolidates this data internally. The API doesn't need explicit join parameters because it understands the relationship between products and their categories.
"""
        elif "CoPPER Question" in prompt:
            return """
# CoPPER System Overview

CoPPER is a comprehensive database and API system used at CME Group. It provides access to various financial data domains through both SQL views and REST API endpoints.

## Key Components

### SQL Views
CoPPER provides SQL views for direct database access, which are organized by domains such as:
- Products
- Instruments
- Sessions
- Firms

These views typically join multiple tables to provide comprehensive data access while handling complex business rules.

### REST API
The REST API provides programmatic access to the same data through endpoints that follow this pattern:
- `/api/{domain}` - e.g., `/api/products`, `/api/instruments`

The API requests use a standardized JSON structure that includes:
- `reqName`: A request identifier
- `dataDomain`: The data domain being accessed
- `type`: Whether the request is independent or dependent
- `req`: Parameters that map to SQL WHERE conditions
- `fields`: The columns/fields to return

### Mapping Between Views and API
CoPPER provides tools and documentation to help developers transition from SQL views to API calls. This mapping process involves:
1. Analyzing the SQL view structure
2. Identifying the appropriate API endpoint
3. Converting WHERE conditions to request parameters
4. Mapping SQL columns to API fields

This dual-access approach allows for flexible data retrieval while providing a migration path from direct database access to API-based integration.
"""
        else:
            return """
I'm the CoPPER Assistant, here to help with database views and API mappings.

Based on your query, I can provide information about CoPPER database views and API mappings. The system allows for accessing financial data through both SQL views and REST API endpoints. If you have specific questions about a particular view or API endpoint, please let me know.
"""


# ==================== OUTPUT FORMATTER ====================

class OutputFormatter:
    """Service for formatting outputs based on query type and content."""
    
    def __init__(self):
        """Initialize the output formatter."""
        self.logger = loggers["output"]
    
    def format_general_answer(self, answer, query, pages=None):
        """
        Format a general answer with source citations.
        
        Args:
            answer: The generated answer text
            query: The original query
            pages: List of pages used for context
            
        Returns:
            Formatted answer string
        """
        self.logger.info("Formatting general answer")
        
        # Prepare source citations if pages provided
        sources = []
        if pages:
            for page in pages:
                sources.append(f"- [{page.metadata.title}]({page.metadata.url})")
        
        # Format the answer
        formatted_answer = [
            f"# Answer: {query}",
            "",
            answer
        ]
        
        # Add sources if available
        if sources:
            formatted_answer.extend([
                "",
                "## Sources",
                ""
            ])
            formatted_answer.extend(sources)
        
        return "\n".join(formatted_answer)
    
    def format_view_analysis(self, view, analysis):
        """
        Format a view analysis with structured sections.
        
        Args:
            view: SqlView object
            analysis: Generated analysis text
            
        Returns:
            Formatted analysis string
        """
        self.logger.info("Formatting view analysis")
        
        formatted_analysis = [
            f"# Analysis of View: {view.name}",
            "",
            "## Definition",
            "```sql",
            view.definition,
            "```",
            "",
            "## Analysis",
            analysis
        ]
        
        return "\n".join(formatted_analysis)
    
    def format_api_mapping(self, view, mapping, explanation):
        """
        Format an API mapping with explanation and example JSON.
        
        Args:
            view: SqlView object
            mapping: ApiMapping object
            explanation: Explanation text
            
        Returns:
            Formatted mapping string
        """
        self.logger.info("Formatting API mapping")
        
        # Generate formatted JSON
        json_body = json.dumps(mapping.__dict__, indent=2)
        
        formatted_mapping = [
            f"# API Mapping for View: {view.name}",
            "",
            "## REST API Endpoint",
            f"`{mapping.api_endpoint}`",
            "",
            "## Request Body",
            "```json",
            json_body,
            "```",
            "",
            "## Explanation",
            explanation
        ]
        
        return "\n".join(formatted_mapping)
    
    def format_json_body(self, view_name, json_body):
        """
        Format a JSON request body.
        
        Args:
            view_name: Name of the view
            json_body: The JSON request body
            
        Returns:
            Formatted JSON string
        """
        self.logger.info("Formatting JSON body")
        
        # Format the JSON body (pretty print)
        if isinstance(json_body, dict):
            formatted_json = json.dumps(json_body, indent=2)
        else:
            formatted_json = json_body
        
        formatted_output = [
            f"# API Request JSON for {view_name}",
            "",
            "```json",
            formatted_json,
            "```",
            "",
            "This JSON body can be used with the CoPPER REST API to retrieve the same data as the SQL view."
        ]
        
        return "\n".join(formatted_output)
    
    def format_error(self, message, query=None):
        """
        Format an error message.
        
        Args:
            message: The error message
            query: The original query (optional)
            
        Returns:
            Formatted error string
        """
        self.logger.info("Formatting error message")
        
        if query:
            formatted_error = [
                f"# Error Processing: {query}",
                "",
                message,
                "",
                "Please try rephrasing your query or providing more information."
            ]
        else:
            formatted_error = [
                "# Error",
                "",
                message,
                "",
                "Please try again or rephrase your request."
            ]
        
        return "\n".join(formatted_error)


# ==================== QUERY ANALYZER ====================

class QueryAnalyzer:
    """Service for analyzing user queries and determining intent."""
    
    def __init__(self):
        """Initialize the query analyzer."""
        self.logger = loggers["main"]
    
    def analyze_query(self, query):
        """
        Analyze a user query to determine intent and extract entities.
        
        Args:
            query: The user's query
            
        Returns:
            Tuple of (intent, entities)
        """
        self.logger.info(f"Analyzing query: {query}")
        
        query_lower = query.lower()
        
        # Intent patterns (ordered by specificity)
        intent_patterns = [
            (QueryIntent.GENERATE_JSON, [
                r"(generate|create|build|get).*(json|request body|api request)",
                r"(json|request body).*(for|from).*(view|sql)",
                r"convert.*view.*to.*json"
            ]),
            (QueryIntent.MAP_VIEW_TO_API, [
                r"(map|mapping|convert|translate).*(view|sql).*(to|into).*(api|rest)",
                r"(how|can).*(map|mapping).*(view|sql)",
                r"(generate|create).*(mapping)"
            ]),
            (QueryIntent.EXPLAIN_MAPPING, [
                r"(explain|describe|detail|clarify).*(mapping|relationship)",
                r"(how|why|what).*(map|maps|mapped|mapping)"
            ]),
            (QueryIntent.VIEW_INFO, [
                r"(tell|describe|explain|what).*(about|is).*(view|sql)",
                r"(view|sql).*(structure|schema|columns|fields)",
                r"(show|display|describe).*(view|sql)"
            ]),
            (QueryIntent.API_INFO, [
                r"(tell|describe|explain|what).*(about|is).*(api|endpoint|rest)",
                r"(api|endpoint|rest).*(structure|schema|parameters)",
                r"(show|display|describe).*(api|endpoint|rest)"
            ]),
            (QueryIntent.COMPARE_VIEWS, [
                r"(compare|difference|similarities).*(between|of).*(views|sql)",
                r"(how).*(different|similar).*(views|sql)"
            ])
        ]
        
        # Check for each intent pattern
        for intent, patterns in intent_patterns:
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    entities = self._extract_entities(query, intent)
                    return intent, entities
        
        # Default to general question if no specific intent detected
        return QueryIntent.GENERAL_QUESTION, self._extract_entities(query, QueryIntent.GENERAL_QUESTION)
    
    def _extract_entities(self, query, intent):
        """
        Extract entities from a query based on the detected intent.
        
        Args:
            query: The user's query
            intent: The detected query intent
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        
        # Extract view names
        view_matches = re.findall(r'\b(?:view|sql|table)\s+(?:named|called)?\s*["\']?([a-zA-Z0-9_]+)["\']?', 
                                 query, re.IGNORECASE)
        
        if not view_matches:
            # Try more general patterns
            view_matches = re.findall(r'\b([A-Z0-9_]{2,})\b', query)
        
        if view_matches:
            entities["view_name"] = view_matches[0]
        
        # Extract API endpoint names
        api_matches = re.findall(r'\b(?:api|endpoint|rest)\s+(?:named|called)?\s*["\']?(/[a-zA-Z0-9/]+)["\']?',
                                query, re.IGNORECASE)
        
        if api_matches:
            entities["api_endpoint"] = api_matches[0]
        
        # Extract domain names
        domain_patterns = [f"\\b{domain}\\b" for domain in Config.DOMAIN_MAPPINGS.values()]
        domain_regex = "|".join(domain_patterns)
        domain_matches = re.findall(domain_regex, query.lower())
        
        if domain_matches:
            entities["domain"] = domain_matches[0]
        
        return entities


# ==================== MAIN ASSISTANT CLASS ====================

class CopperAssistant:
    """Main class that coordinates all services."""
    
    def __init__(self):
        """Initialize the Copper Assistant with all required services."""
        self.logger = loggers["main"]
        
        # Initialize services
        self.confluence = ConfluenceService()
        self.sql_parser = SqlParserService()
        self.api_mapping = ApiMappingService()
        self.llm = LLMService()
        self.output_formatter = OutputFormatter()
        self.query_analyzer = QueryAnalyzer()
        
        # Track active sessions
        self.active_sessions = {}
    
    async def initialize(self):
        """Initialize all services and load necessary data."""
        self.logger.info("Initializing CoPPER Assistant")
        
        # Initialize Confluence service
        if not self.confluence.initialize():
            raise Exception("Failed to initialize Confluence service")
        
        # Load examples for the API mapping service
        examples = await self.confluence.find_view_examples()
        self.api_mapping.add_examples(examples)
        
        self.logger.info("CoPPER Assistant initialized successfully")
        return True
    
    async def process_query(self, query_text, session_id=None):
        """
        Process a user query and generate a response.
        
        Args:
            query_text: The user's query text
            session_id: Optional session ID for context tracking
            
        Returns:
            Formatted response string
        """
        self.logger.info(f"Processing query: {query_text}")
        
        start_time = time.time()
        
        # Create a new session if none provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Create a new query object
        query = UserQuery(
            original_text=query_text,
            processed_text=query_text,
            session_id=session_id,
            start_time=start_time
        )
        
        try:
            # Analyze the query to determine intent
            intent, entities = self.query_analyzer.analyze_query(query_text)
            query.intent = intent
            query.entities = entities
            
            self.logger.info(f"Detected intent: {intent.value}")
            self.logger.info(f"Extracted entities: {entities}")
            
            # Process based on intent
            if intent == QueryIntent.VIEW_INFO:
                return await self._process_view_info(query)
            elif intent == QueryIntent.API_INFO:
                return await self._process_api_info(query)
            elif intent == QueryIntent.MAP_VIEW_TO_API:
                return await self._process_map_view_to_api(query)
            elif intent == QueryIntent.GENERATE_JSON:
                return await self._process_generate_json(query)
            elif intent == QueryIntent.EXPLAIN_MAPPING:
                return await self._process_explain_mapping(query)
            elif intent == QueryIntent.COMPARE_VIEWS:
                return await self._process_compare_views(query)
            else:  # Default to general question
                return await self._process_general_question(query)
        
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return self.output_formatter.format_error(f"An error occurred: {str(e)}", query_text)
        finally:
            # Log processing time
            processing_time = time.time() - start_time
            self.logger.info(f"Query processed in {processing_time:.2f} seconds")
            
            # Store query in active sessions
            self.active_sessions[session_id] = query
    
    async def _process_view_info(self, query):
        """Process a view information query."""
        self.logger.info("Processing view info query")
        
        # Check if we have a view name
        if "view_name" not in query.entities:
            return self.output_formatter.format_error(
                "Please specify a view name to get information about.",
                query.original_text
            )
        
        view_name = query.entities["view_name"]
        
        # Find the view definition
        view_defs = await self.confluence.find_sql_views(view_name)
        
        if not view_defs:
            return self.output_formatter.format_error(
                f"Could not find the view '{view_name}' in the documentation.",
                query.original_text
            )
        
        # Parse the first matching view
        page_id, name, definition = view_defs[0]
        view = self.sql_parser.parse_view_definition(definition)
        
        # Generate analysis using LLM
        analysis = await self.llm.analyze_sql_view(definition)
        
        # Store in query object
        query.extracted_view = view
        
        # Format and return
        return self.output_formatter.format_view_analysis(view, analysis)
    
    async def _process_api_info(self, query):
        """Process an API information query."""
        self.logger.info("Processing API info query")
        
        # Determine what API info to look for
        search_terms = ["CoPPER API", "REST API"]
        
        if "api_endpoint" in query.entities:
            search_terms.append(query.entities["api_endpoint"])
        
        if "domain" in query.entities:
            search_terms.append(query.entities["domain"])
            search_terms.append(f"{query.entities['domain']} API")
        
        # Search for relevant pages
        context_pages = []
        for term in search_terms:
            pages = await self.confluence.search_confluence(term)
            for page_meta in pages:
                page = await self.confluence.get_page_content(page_meta.id)
                if page:
                    context_pages.append(page)
                    
                    # Limit to avoid too much content
                    if len(context_pages) >= 5:
                        break
            
            if len(context_pages) >= 5:
                break
        
        # Extract content for context
        if not context_pages:
            return self.output_formatter.format_error(
                "Could not find information about the requested API.",
                query.original_text
            )
        
        # Store in query object
        query.context_pages = context_pages
        
        # Prepare context
        context = "\n\n".join([
            self.confluence.content_extractor.format_for_context(page.content, page.metadata.title)
            for page in context_pages[:3]  # Limit to top 3 for space
        ])
        
        # Generate answer using LLM
        answer = await self.llm.answer_general_question(query.original_text, context)
        
        # Format and return
        return self.output_formatter.format_general_answer(answer, query.original_text, context_pages)
    
    async def _process_map_view_to_api(self, query):
        """Process a view-to-API mapping query."""
        self.logger.info("Processing map view to API query")
        
        # Check if we have a view name
        if "view_name" not in query.entities:
            return self.output_formatter.format_error(
                "Please specify a view name to map to an API.",
                query.original_text
            )
        
        view_name = query.entities["view_name"]
        
        # Find the view definition
        view_defs = await self.confluence.find_sql_views(view_name)
        
        if not view_defs:
            return self.output_formatter.format_error(
                f"Could not find the view '{view_name}' in the documentation.",
                query.original_text
            )
        
        # Parse the first matching view
        page_id, name, definition = view_defs[0]
        view = self.sql_parser.parse_view_definition(definition)
        
        # Generate API mapping
        mapping = self.api_mapping.generate_mapping(view)
        
        # Generate explanation
        explanation = self.api_mapping.generate_mapping_explanation(view, mapping)
        
        # Store in query object
        query.extracted_view = view
        query.generated_mapping = mapping
        
        # Format and return
        return self.output_formatter.format_api_mapping(view, mapping, explanation)
    
    async def _process_generate_json(self, query):
        """Process a JSON generation query."""
        self.logger.info("Processing generate JSON query")
        
        # Check if we have a view name
        if "view_name" not in query.entities:
            return self.output_formatter.format_error(
                "Please specify a view name to generate JSON for.",
                query.original_text
            )
        
        view_name = query.entities["view_name"]
        
        # Find the view definition
        view_defs = await self.confluence.find_sql_views(view_name)
        
        if not view_defs:
            return self.output_formatter.format_error(
                f"Could not find the view '{view_name}' in the documentation.",
                query.original_text
            )
        
        # Parse the first matching view
        page_id, name, definition = view_defs[0]
        view = self.sql_parser.parse_view_definition(definition)
        
        # Generate API mapping
        mapping = self.api_mapping.generate_mapping(view)
        
        # Generate JSON body
        json_body = self.api_mapping.generate_json_body(mapping)
        
        # Store in query object
        query.extracted_view = view
        query.generated_mapping = mapping
        
        # Format and return
        return self.output_formatter.format_json_body(view.name, json_body)
    
    async def _process_explain_mapping(self, query):
        """Process a mapping explanation query."""
        self.logger.info("Processing explain mapping query")
        
        # Check if we have a view name
        if "view_name" not in query.entities:
            return self.output_formatter.format_error(
                "Please specify a view name to explain the mapping for.",
                query.original_text
            )
        
        view_name = query.entities["view_name"]
        
        # Find the view definition
        view_defs = await self.confluence.find_sql_views(view_name)
        
        if not view_defs:
            return self.output_formatter.format_error(
                f"Could not find the view '{view_name}' in the documentation.",
                query.original_text
            )
        
        # Parse the first matching view
        page_id, name, definition = view_defs[0]
        view = self.sql_parser.parse_view_definition(definition)
        
        # Generate API mapping
        mapping = self.api_mapping.generate_mapping(view)
        
        # Generate a detailed explanation using LLM
        json_str = self.api_mapping.generate_formatted_json(mapping)
        explanation = await self.llm.explain_mapping(definition, json_str)
        
        # Store in query object
        query.extracted_view = view
        query.generated_mapping = mapping
        
        # Format and return
        return self.output_formatter.format_api_mapping(view, mapping, explanation)
    
    async def _process_compare_views(self, query):
        """Process a view comparison query."""
        self.logger.info("Processing compare views query")
        
        # We need general context about views
        pages = await self.confluence.search_confluence("SQL view comparison", limit=5)
        
        context_pages = []
        for page_meta in pages:
            page = await self.confluence.get_page_content(page_meta.id)
            if page:
                context_pages.append(page)
        
        # Extract view names if possible
        view_names = re.findall(r'\b([A-Z0-9_]{2,})\b', query.original_text)
        
        # If we have view names, find their definitions
        views = []
        if view_names:
            for view_name in view_names[:2]:  # Limit to 2 views for comparison
                view_defs = await self.confluence.find_sql_views(view_name)
                if view_defs:
                    page_id, name, definition = view_defs[0]
                    view = self.sql_parser.parse_view_definition(definition)
                    views.append(view)
        
        # Prepare context
        context = []
        
        # Add information about found views
        for view in views:
            context.append(f"# View: {view.name}")
            context.append("```sql")
            context.append(view.definition)
            context.append("```")
            context.append("")
        
        # Add general information from pages
        for page in context_pages:
            context.append(self.confluence.content_extractor.format_for_context(
                page.content, page.metadata.title
            ))
        
        # Generate answer using LLM
        answer = await self.llm.answer_general_question(query.original_text, "\n\n".join(context))
        
        # Format and return
        return self.output_formatter.format_general_answer(answer, query.original_text, context_pages)
    
    async def _process_general_question(self, query):
        """Process a general question."""
        self.logger.info("Processing general question")
        
        # Prepare search terms
        search_terms = [query.original_text]
        
        # Add any entities as additional search terms
        for entity_type, entity_value in query.entities.items():
            if entity_type == "view_name":
                search_terms.append(f"view {entity_value}")
            elif entity_type == "api_endpoint":
                search_terms.append(entity_value)
            elif entity_type == "domain":
                search_terms.append(entity_value)
        
        # Search for relevant pages
        context_pages = []
        for term in search_terms:
            pages = await self.confluence.search_confluence(term)
            for page_meta in pages:
                page = await self.confluence.get_page_content(page_meta.id)
                if page:
                    context_pages.append(page)
                    
                    # Limit to avoid too much content
                    if len(context_pages) >= 5:
                        break
            
            if len(context_pages) >= 5:
                break
        
        # Extract content for context
        if not context_pages:
            # Try a more general search
            pages = await self.confluence.search_confluence("CoPPER documentation")
            for page_meta in pages:
                page = await self.confluence.get_page_content(page_meta.id)
                if page:
                    context_pages.append(page)
                    
                    # Limit to avoid too much content
                    if len(context_pages) >= 3:
                        break
        
        # If still no context, return error
        if not context_pages:
            return self.output_formatter.format_error(
                "Could not find relevant information to answer your question.",
                query.original_text
            )
        
        # Store in query object
        query.context_pages = context_pages
        
        # Prepare context
        context = "\n\n".join([
            self.confluence.content_extractor.format_for_context(page.content, page.metadata.title)
            for page in context_pages[:3]  # Limit to top 3 for space
        ])
        
        # Generate answer using LLM
        answer = await self.llm.answer_general_question(query.original_text, context)
        
        # Format and return
        return self.output_formatter.format_general_answer(answer, query.original_text, context_pages)


# ==================== MAIN ENTRY POINT ====================

async def main():
    """Main entry point for the CoPPER Assistant."""
    logger.info("Starting Enhanced CoPPER Assistant")
    
    print("\n===== Enhanced CoPPER Database-to-API Mapping Assistant =====")
    print("Initializing services and connecting to knowledge base...")
    
    try:
        # Initialize the assistant
        assistant = CopperAssistant()
        await assistant.initialize()
        
        print("\nCoPPER Assistant is ready! You can ask questions about:")
        print("- SQL view definitions and structures")
        print("- API endpoint mappings and generation")
        print("- Converting SQL views to API request JSON")
        print("- Understanding the relationships between views and APIs")
        print("\nType 'quit' or 'exit' to end the session.\n")
        
        # Main interaction loop
        while True:
            try:
                user_input = input("\nQuestion: ").strip()
                
                if user_input.lower() in ('quit', 'exit', 'q'):
                    print("Thanks for using the Enhanced CoPPER Assistant. Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("\nProcessing your request...")
                response = await assistant.process_query(user_input)
                
                print("\n" + response)
                
            except KeyboardInterrupt:
                print("\nSession terminated. Goodbye!")
                break
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                print(f"\nI encountered an error: {str(e)}")
                print("Please try a different query or restart the assistant.")
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"Failed to start the CoPPER Assistant: {str(e)}")
        print("Please check the logs for details.")


# Run the application
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())



























<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CoPPER Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/lucide-icons@0.284.0/dist/umd/lucide.min.css">
    <link rel="stylesheet" href="/css/animations.css">
    <link rel="stylesheet" href="/css/styles.css">
</head>
<body>
<div class="app">
    <!-- Landing Page -->
    <div id="landing-page" class="landing-page">
        <div class="background-elements">
            <div class="bg-blob bg-blob-1"></div>
            <div class="bg-blob bg-blob-2"></div>
            <div class="bg-blob bg-blob-3"></div>
        </div>

        <div class="hero-section">
            <h1 class="hero-title">CoPPER Assistant - Database to API Mapping</h1>
            <p class="hero-subtitle">Intelligent assistant that helps map SQL views to REST API endpoints</p>

            <div class="hero-buttons">
                <button id="explore-btn" class="explore-btn">
                    <span>Explore CoPPER</span>
                </button>
                <button id="login-btn" class="login-btn">
                    <i data-lucide="log-in"></i>
                    <span>Login</span>
                </button>
            </div>

            <!-- Interactive Circle Feature -->
            <div class="circle-feature-container">
                <div id="circle-container" class="circle-container">
                    <div class="center-logo">
                        <div class="pulse"></div>
                        <div class="pulse" style="animation-delay: 0.5s"></div>
                        <div class="bot-icon">AI</div>
                    </div>
                    <!-- Circle items will be dynamically added here -->
                </div>
            </div>
        </div>

        <div class="features-section">
            <div class="feature-card">
                <div class="feature-icon-container">
                    <i data-lucide="database" class="feature-icon"></i>
                </div>
                <div class="feature-card-gradient"></div>
                <h3 class="feature-title">SQL View Analysis</h3>
                <p class="feature-description">Analyze SQL views to understand their structure, purpose, and data relationships.</p>
                <div class="feature-corner-accent"></div>
                <div class="feature-glow"></div>
            </div>

            <div class="feature-card">
                <div class="feature-icon-container">
                    <i data-lucide="server" class="feature-icon"></i>
                </div>
                <div class="feature-card-gradient"></div>
                <h3 class="feature-title">API Mapping</h3>
                <p class="feature-description">Convert SQL views to API request formats for modern application integration.</p>
                <div class="feature-corner-accent"></div>
                <div class="feature-glow"></div>
            </div>

            <div class="feature-card">
                <div class="feature-icon-container">
                    <i data-lucide="code" class="feature-icon"></i>
                </div>
                <div class="feature-card-gradient"></div>
                <h3 class="feature-title">JSON Generation</h3>
                <p class="feature-description">Generate ready-to-use JSON request bodies for CoPPER REST API endpoints.</p>
                <div class="feature-corner-accent"></div>
                <div class="feature-glow"></div>
            </div>
        </div>
    </div>

    <!-- Chat Interface -->
    <div id="chat-interface" class="chat-interface hidden">
        <!-- Sidebar -->
        <div id="chat-sidebar" class="chat-sidebar">
            <div class="sidebar-header">
                <div class="sidebar-header-top">
                    <button id="return-home-btn" class="return-home-btn">
                        <i data-lucide="arrow-left"></i>
                        Return Home
                    </button>
                    <button class="settings-btn">
                        <i data-lucide="settings"></i>
                    </button>
                </div>
                <button id="new-chat-btn" class="new-chat-btn">
                    <i data-lucide="plus-circle"></i>
                    <span>New chat</span>
                </button>
                <div class="search-container">
                    <i data-lucide="search" class="search-icon"></i>
                    <input type="text" placeholder="Search chats..." class="search-input">
                </div>
            </div>

            <div class="sidebar-sections">
                <div class="sidebar-section">
                    <button class="section-header">
                        <div class="section-title">
                            <i data-lucide="message-square"></i>
                            <span>Chats</span>
                        </div>
                        <i data-lucide="chevron-right" class="section-toggle"></i>
                    </button>
                    <div id="chats-list" class="section-content">
                        <!-- Chats will be dynamically added here -->
                        <div class="empty-state">
                            <div class="empty-state-text">No chats yet</div>
                            <div class="empty-state-subtext">Start a new conversation using the button above</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="sidebar-footer">
                <div id="user-profile" class="user-profile">
                    <!-- User profile will be dynamically added here -->
                    <button id="sidebar-login-btn" class="sidebar-login-btn">
                        <i data-lucide="log-in"></i>
                        <span>Login / Sign up</span>
                    </button>
                </div>
                <div class="help-button">
                    <button class="help-btn">
                        <i data-lucide="help-circle"></i>
                        <span>Help & FAQ</span>
                    </button>
                </div>
            </div>
        </div>

        <!-- Sidebar Toggle -->
        <div id="sidebar-toggle" class="sidebar-toggle">
            <button class="toggle-btn">
                <i data-lucide="chevron-right" class="toggle-icon"></i>
            </button>
        </div>

        <!-- Main Content -->
        <div id="chat-main" class="chat-main">
            <div id="chat-messages" class="chat-messages">
                <!-- Chat messages will be dynamically added here -->
                <div id="chat-suggestions" class="chat-suggestions">
                    <h1 class="welcome-header">
                        <span class="text-blue">CoPPER</span>
                        <span class="text-rose">Assistant</span>
                    </h1>
                    <div class="suggestion-container">
                        <div class="suggestion-item" data-suggestion="Explain what the CoPPER system is">
                            <div class="suggestion-icon"><i data-lucide="help-circle"></i></div>
                            <h3 class="suggestion-title">Explain</h3>
                            <p class="suggestion-description">what the CoPPER system is</p>
                        </div>
                        <div class="suggestion-item" data-suggestion="Analyze SQL view PRODUCT_VIEW">
                            <div class="suggestion-icon"><i data-lucide="database"></i></div>
                            <h3 class="suggestion-title">Analyze</h3>
                            <p class="suggestion-description">SQL view PRODUCT_VIEW</p>
                        </div>
                        <div class="suggestion-item" data-suggestion="Map PRODUCT_VIEW to API endpoint">
                            <div class="suggestion-icon"><i data-lucide="server"></i></div>
                            <h3 class="suggestion-title">Map</h3>
                            <p class="suggestion-description">PRODUCT_VIEW to API endpoint</p>
                        </div>
                        <div class="suggestion-item" data-suggestion="Generate JSON for PRODUCT_VIEW">
                            <div class="suggestion-icon"><i data-lucide="code"></i></div>
                            <h3 class="suggestion-title">Generate</h3>
                            <p class="suggestion-description">JSON for PRODUCT_VIEW</p>
                        </div>
                    </div>
                </div>
            </div>

            <div class="chat-input-area">
                <div class="input-container">
                    <button class="input-button">
                        <i data-lucide="plus"></i>
                    </button>
                    <input type="text" id="chat-input" placeholder="Ask about CoPPER views and APIs..." class="chat-input">
                    <button class="input-button">
                        <i data-lucide="mic"></i>
                    </button>
                    <button id="send-button" class="send-button disabled">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M22 2L11 13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </button>
                </div>
                <div class="disclaimer">CoPPER Assistant can make mistakes, so verify critical mappings</div>
            </div>
        </div>
    </div>

    <!-- Modals -->
    <div id="login-modal" class="modal hidden">
        <!-- Login modal content will be dynamically added -->
    </div>

    <div id="loading-popup" class="loading-popup hidden">
        <div class="loading-animation">
            <div class="loading-star"></div>
            <div class="loading-dot"></div>
        </div>
    </div>

    <div id="feedback-toast" class="feedback-toast hidden">
        <i data-lucide="thumbs-up" class="feedback-icon"></i>
        <span class="feedback-message">Thanks for your feedback!</span>
    </div>
</div>

<script src="https://unpkg.com/lucide@latest"></script>
<script src="/js/ui-components.js"></script>
<script src="/js/chat.js"></script>
<script src="/js/app.js"></script>
</body>
</html>




















@keyframes pulse {
    0% {
        transform: scale(1);
        opacity: 0.8;
    }
    70% {
        transform: scale(1.5);
        opacity: 0;
    }
    100% {
        transform: scale(1);
        opacity: 0;
    }
}
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
@keyframes bounce {
    0%,
    60%,
    100% {
        transform: translateY(0);
        opacity: 0.6;
    }
    30% {
        transform: translateY(-4px);
        opacity: 1;
    }
}
@keyframes blink {
    0%,
    100% {
        opacity: 1;
    }
    50% {
        opacity: 0;
    }
}
@keyframes gradientText {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}
@keyframes floating {
    0% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-10px);
    }
    100% {
        transform: translateY(0px);
    }
}
@keyframes gradient {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}
@keyframes typingDot {
    0% {
        opacity: 0;
    }
    50% {
        opacity: 1;
    }
    100% {
        opacity: 0;
    }
}
@keyframes gradientAnimate {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
/* Added animations for card transitions */
@keyframes cardExpand {
    0% {
        width: 150px;
        z-index: 5;
    }
    100% {
        width: 280px;
        z-index: 20;
    }
}
@keyframes cardShrink {
    0% {
        width: 280px;
        z-index: 20;
    }
    100% {
        width: 150px;
        z-index: 5;
    }
}
/* Apply animations to circle items */
.circle-item.active {
    animation: cardExpand 0.3s forwards;
}
.circle-item.inactive {
    animation: cardShrink 0.3s forwards;
}














:root {
    --font-sans: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    --font-heading: 'Poppins', 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    --blue-500: #3a86ff;
    --blue-600: #2a76ef;
    --blue-900: #2a3654;
    --rose-500: #ff6b6b;
    --gray-700: #333;
    --gray-800: #1e1e1e;
    --gray-900: #121212;
    --radius: 0.5rem;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html, body {
    height: 100%;
    width: 100%;
}

body {
    font-family: var(--font-sans);
    color: white;
    background: linear-gradient(to bottom, #121212, #0a0a15);
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-heading);
}

.hidden {
    display: none !important;
}

/* App container */
.app {
    height: 100%;
    width: 100%;
}

/* Landing page */
.landing-page {
    min-height: 100vh;
    background: linear-gradient(to bottom, #121212, #0a0a15);
    overflow: hidden;
    position: relative;
    padding: 0 20px;
}

/* Background elements */
.background-elements {
    position: absolute;
    inset: 0;
    overflow: hidden;
    z-index: 0;
}

.bg-blob {
    position: absolute;
    border-radius: 50%;
    filter: blur(100px);
    opacity: 0.1;
}

.bg-blob-1 {
    top: 10%;
    left: 5%;
    width: 30vw;
    height: 30vw;
    background-color: var(--blue-500);
}

.bg-blob-2 {
    bottom: 20%;
    right: 10%;
    width: 25vw;
    height: 25vw;
    background-color: var(--rose-500);
}

.bg-blob-3 {
    top: 40%;
    right: 20%;
    width: 20vw;
    height: 20vw;
    background-color: #a855f7; /* Purple */
}

/* Hero section */
.hero-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    position: relative;
    z-index: 1;
    padding: 20px 0;
}

.hero-title {
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 1.5rem;
    text-align: center;
    background: linear-gradient(to right, #60a5fa, #c084fc, #fb7185);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    max-width: 800px;
    font-family: var(--font-heading);
}

.hero-subtitle {
    font-size: 1.25rem;
    color: #d1d5db;
    max-width: 600px;
    text-align: center;
    margin-bottom: 3rem;
}

.hero-buttons {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 4rem;
    z-index: 10;
}

@media (min-width: 640px) {
    .hero-buttons {
        flex-direction: row;
    }

    .hero-title {
        font-size: 3.75rem;
    }

    .hero-subtitle {
        font-size: 1.5rem;
    }
}

.explore-btn {
    position: relative;
    overflow: hidden;
    border-radius: 9999px;
    background: linear-gradient(to right, var(--blue-500), var(--blue-600));
    padding: 1rem 2rem;
    color: white;
    font-size: 1.125rem;
    font-weight: 500;
    transition: all 0.3s ease;
    border: none;
    cursor: pointer;
    font-family: var(--font-heading);
    display: flex;
    align-items: center;
    justify-content: center;
}

.explore-btn:hover {
    transform: translateY(-4px);
    box-shadow: 0 0 25px rgba(59, 130, 246, 0.5);
}

.explore-btn::after {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 0;
    background: linear-gradient(to right, #60a5fa, var(--blue-600));
    opacity: 0;
    transition: opacity 0.3s;
}

.explore-btn:hover::after {
    opacity: 1;
}

.explore-btn span {
    position: relative;
    z-index: 1;
}

.login-btn {
    border: 2px solid var(--blue-500);
    color: #60a5fa;
    padding: 1rem 2rem;
    border-radius: 9999px;
    font-size: 1.125rem;
    font-weight: 500;
    background: transparent;
    transition: all 0.3s;
    cursor: pointer;
    font-family: var(--font-heading);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    justify-content: center;
}

.login-btn:hover {
    background-color: rgba(59, 130, 246, 0.1);
    transform: translateY(-4px);
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.3);
}



/* Circle feature */
.circle-feature-container {
    position: relative;
    width: 600px;
    height: 600px;
    max-width: 100%;
    margin-bottom: 5rem;
}

.circle-container {
    position: relative;
    width: 100%;
    height: 100%;
}

.center-logo {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 6rem;
    height: 6rem;
    background-color: var(--gray-900);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
    border: 2px solid rgba(59, 130, 246, 0.3);
}

.pulse {
    position: absolute;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background-color: rgba(59, 130, 246, 0.2);
    animation: pulse 2s infinite;
}

.bot-icon {
    font-size: 1.875rem;
    font-weight: bold;
    color: var(--blue-500);
    z-index: 20;
}

.circle-item {
    position: absolute;
    background-color: rgba(30, 30, 30, 0.8);
    backdrop-filter: blur(8px);
    padding: 1.25rem;
    border-radius: 0.75rem;
    border: 1px solid rgba(59, 130, 246, 0.2);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    cursor: pointer;
    transition: all 0.3s;
}

.circle-item.active {
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    border-color: rgba(59, 130, 246, 0.5);
}

.chat-preview {
    margin-bottom: 0.5rem;
}

.chat-title {
    font-size: 1rem;
    font-weight: 500;
    margin-bottom: 0.25rem;
    font-family: var(--font-heading);
}

.chat-description {
    font-size: 0.875rem;
    color: #9ca3af;
}

.demo-conversation {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(55, 65, 81, 0.5);
    animation: fadeIn 0.3s ease;
}

.demo-message {
    display: flex;
    margin-bottom: 0.5rem;
}

.demo-avatar {
    width: 2rem;
    height: 2rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 0.5rem;
    font-size: 0.75rem;
    font-weight: bold;
}

.user-avatar {
    background-color: var(--rose-500);
}

.bot-avatar {
    background-color: var(--blue-500);
}

.demo-text {
    background-color: rgba(30, 30, 30, 0.5);
    padding: 0.5rem;
    border-radius: 0.75rem;
    font-size: 0.875rem;
    max-width: 200px;
}

.user-text {
    background-color: rgba(30, 58, 138, 0.5);
    border: 1px solid rgba(30, 64, 175, 0.5);
}

/* Features section */
.features-section {
    display: grid;
    grid-template-columns: 1fr;
    gap: 2rem;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem 5rem 1rem;
}

@media (min-width: 768px) {
    .features-section {
        grid-template-columns: repeat(3, 1fr);
    }
}

.feature-card {
    background-color: rgba(30, 30, 30, 0.6);
    padding: 2rem;
    border-radius: 0.75rem;
    border: 1px solid rgba(59, 130, 246, 0.2);
    transition: all 0.3s;
    overflow: hidden;
    position: relative;
}

.feature-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.feature-icon-container {
    margin-bottom: 1rem;
    width: 4rem;
    height: 4rem;
    border-radius: 50%;
    background-color: rgba(18, 18, 18, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    z-index: 10;
}

.feature-icon {
    width: 3rem;
    height: 3rem;
    color: #60a5fa;
}

.feature-card-gradient {
    position: absolute;
    inset: -1px;
    opacity: 0;
    transition: opacity 0.5s;
    z-index: -10;
    filter: blur(16px);
    background: linear-gradient(to right, rgba(58, 134, 255, 0), rgba(58, 134, 255, 0.1), rgba(168, 85, 247, 0));
}

.feature-card:hover .feature-card-gradient {
    opacity: 1;
    animation: gradientAnimate 3s ease infinite;
}

.feature-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    position: relative;
    z-index: 10;
    font-family: var(--font-heading);
}

.feature-description {
    color: #d1d5db;
    position: relative;
    z-index: 10;
    transition: color 0.3s;
}

.feature-card:hover .feature-description {
    color: white;
}

.feature-corner-accent {
    position: absolute;
    bottom: 0;
    right: 0;
    width: 0;
    height: 0;
    opacity: 0;
    transition: all 0.3s;
    background: linear-gradient(to top left, rgba(58, 134, 255, 0.2), transparent);
    border-top-left-radius: 1.5rem;
}

.feature-card:hover .feature-corner-accent {
    opacity: 1;
    width: 4rem;
    height: 4rem;
}

.feature-glow {
    position: absolute;
    inset: 0;
    border-radius: 0.75rem;
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.3);
}

.feature-card:hover .feature-glow {
    opacity: 1;
}

/* Chat Interface */
.chat-interface {
    display: flex;
    width: 100%;
    height: 100vh;
    background: linear-gradient(to bottom, #121212, #0a0a15);
    position: relative;
}

/* Sidebar */
.chat-sidebar {
    width: 280px;
    height: 100vh;
    background-color: var(--gray-900);
    border-right: 1px solid #1f2937;
    display: flex;
    flex-direction: column;
    position: fixed;
    left: 0;
    top: 0;
    z-index: 50;
    transition: transform 0.3s ease;
}

.sidebar-header {
    padding: 1rem;
    border-bottom: 1px solid #1f2937;
    display: flex;
    flex-direction: column;
}

.sidebar-header-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.return-home-btn {
    display: flex;
    align-items: center;
    color: var(--rose-500);
    border: 1px solid rgba(244, 63, 94, 0.5);
    padding: 0.5rem 0.75rem;
    border-radius: 0.5rem;
    font-size: 0.875rem;
    background: transparent;
    cursor: pointer;
    transition: all 0.2s;
    font-family: var(--font-heading);
}

.return-home-btn i {
    margin-right: 0.5rem;
    width: 16px;
    height: 16px;
}

.return-home-btn:hover {
    background-color: rgba(244, 63, 94, 0.1);
}

.settings-btn {
    width: 2rem;
    height: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 9999px;
    background: transparent;
    border: none;
    color: #9ca3af;
    cursor: pointer;
    transition: all 0.2s;
}

.settings-btn:hover {
    background-color: var(--gray-800);
    color: white;
}

.new-chat-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    background-color: var(--blue-600);
    color: white;
    padding: 0.75rem;
    border-radius: 0.5rem;
    width: 100%;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
    font-family: var(--font-heading);
    font-weight: 500;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.new-chat-btn:hover {
    background-color: #2563eb;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
}

.search-container {
    position: relative;
    margin-top: 1rem;
}

.search-icon {
    position: absolute;
    left: 0.75rem;
    top: 50%;
    transform: translateY(-50%);
    color: #6b7280;
    width: 16px;
    height: 16px;
}

.search-input {
    width: 100%;
    background-color: var(--gray-800);
    border: 1px solid #374151;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem 0.5rem 2.5rem;
    color: white;
    font-size: 0.875rem;
}

.search-input::placeholder {
    color: #6b7280;
}

.search-input:focus {
    outline: none;
    border-color: var(--blue-500);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25);
}

.sidebar-sections {
    flex: 1;
    overflow-y: auto;
}

.sidebar-section {
    margin-bottom: 0.5rem;
}

.section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    padding: 0.75rem;
    background: transparent;
    border: none;
    color: #d1d5db;
    cursor: pointer;
    transition: background-color 0.2s;
    text-align: left;
}

.section-header:hover {
    background-color: rgba(55, 65, 81, 0.5);
}

.section-title {
    display: flex;
    align-items: center;
}

.section-title i {
    margin-right: 0.5rem;
    width: 16px;
    height: 16px;
}

.section-toggle {
    width: 16px;
    height: 16px;
    transition: transform 0.2s;
}

.section-toggle.open {
    transform: rotate(90deg);
}

.section-content {
    padding: 0 0.5rem;
}

.empty-state {
    padding: 1rem;
    text-align: center;
}

.empty-state-text {
    color: #6b7280;
    font-size: 0.75rem;
    margin-bottom: 0.5rem;
}

.empty-state-subtext {
    color: #9ca3af;
    font-size: 0.75rem;
}

.sidebar-footer {
    padding: 1rem;
    border-top: 1px solid #1f2937;
}

.user-profile {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem;
    border-radius: 0.5rem;
    transition: background-color 0.2s;
}

.user-profile:hover {
    background-color: rgba(55, 65, 81, 0.5);
}

.sidebar-login-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    background-color: var(--blue-600);
    color: white;
    padding: 0.75rem;
    border-radius: 0.5rem;
    width: 100%;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.875rem;
    font-family: var(--font-heading);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.sidebar-login-btn:hover {
    background-color: #2563eb;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
}

.help-button {
    display: flex;
    justify-content: center;
    margin-top: 0.75rem;
}

.help-btn {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    color: #9ca3af;
    background: transparent;
    border: none;
    padding: 0.375rem;
    border-radius: 0.375rem;
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.2s;
}

.help-btn:hover {
    background-color: var(--gray-800);
    color: white;
}

/* Sidebar toggle */
.sidebar-toggle {
    position: fixed;
    top: 50%;
    left: 280px;
    transform: translateY(-50%);
    z-index: 40;
    transition: transform 0.3s ease;
}

.sidebar-toggle.closed {
    left: 0;
}

.toggle-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2.5rem;
    height: 2.5rem;
    background-color: var(--gray-800);
    border-radius: 9999px;
    border: 1px solid #374151;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    cursor: pointer;
    transition: background-color 0.2s;
}

.toggle-btn:hover {
    background-color: var(--gray-700);
}

.toggle-icon {
    width: 18px;
    height: 18px;
    color: #9ca3af;
    transition: transform 0.3s;
}

.toggle-icon.closed {
    transform: rotate(180deg);
}

/* Main chat content */
.chat-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    margin-left: 280px;
    transition: margin-left 0.3s ease;
    height: 100vh;
}

.chat-main.sidebar-closed {
    margin-left: 0;
}

.project-header {
    background-color: rgba(31, 41, 55, 0.5);
    border-bottom: 1px solid #374151;
    padding: 0.75rem 1.5rem;
}

.project-header-content {
    max-width: 48rem;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.project-info {
    display: flex;
    align-items: center;
}

.project-info i {
    color: var(--blue-400);
    margin-right: 0.5rem;
    width: 16px;
    height: 16px;
}

.project-name {
    font-size: 0.875rem;
    font-weight: 500;
    font-family: var(--font-heading);
}

.project-description {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-left: 0.75rem;
    max-width: 16rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.exit-project-btn {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.75rem;
    color: #9ca3af;
    background-color: rgba(31, 41, 55, 0.8);
    border: none;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    cursor: pointer;
    transition: all 0.2s;
}

.exit-project-btn:hover {
    background-color: var(--gray-700);
    color: white;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem 2rem;
}

/* Chat suggestions */
.chat-suggestions {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
}

.welcome-header {
    font-size: 2.5rem;
    margin-bottom: 3rem;
    font-family: var(--font-heading);
}

.text-blue {
    color: var(--blue-500);
}

.text-rose {
    color: var(--rose-500);
}

.suggestion-container {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1rem;
    max-width: 48rem;
}

@media (min-width: 768px) {
    .suggestion-container {
        grid-template-columns: repeat(2, 1fr);
    }
}

.suggestion-item {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    background-color: var(--gray-800);
    border: 1px solid #374151;
    border-radius: 0.75rem;
    padding: 1.25rem;
    cursor: pointer;
    transition: all 0.2s;
}

.suggestion-item:hover {
    transform: translateY(-5px);
    background-color: #252525;
    border-color: rgba(59, 130, 246, 0.3);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.suggestion-icon {
    margin-bottom: 0.75rem;
    color: var(--blue-500);
}

.suggestion-icon:hover {
    transform: scale(1.1);
}

.suggestion-title {
    font-size: 1rem;
    font-weight: 500;
    margin-bottom: 0.25rem;
    font-family: var(--font-heading);
}

.suggestion-description {
    font-size: 0.875rem;
    color: #9ca3af;
}

/* Chat input area */
.chat-input-area {
    padding: 1rem 1.5rem;
    background-color: transparent;
}

.input-container {
    max-width: 48rem;
    margin: 0 auto;
    display: flex;
    align-items: center;
    background-color: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 1.5rem;
    padding: 0.75rem;
    transition: all 0.2s;
}

.input-container:focus-within {
    border-color: #3a3a3a;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.1);
}

.input-button {
    width: 2rem;
    height: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #9ca3af;
    background: transparent;
    border: none;
    cursor: pointer;
    transition: color 0.2s;
}

.input-button:hover {
    color: white;
}

.chat-input {
    flex: 1;
    height: 2.5rem;
    background-color: transparent;
    border: none;
    color: white;
    font-size: 1rem;
    padding: 0 1rem;
    outline: none;
}

.chat-input::placeholder {
    color: rgba(255, 255, 255, 0.5);
}

.send-button {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: 9999px;
    background-color: var(--blue-500);
    color: white;
    border: none;
    cursor: pointer;
    transition: background-color 0.2s;
}

.send-button:hover {
    background-color: var(--blue-600);
}

.send-button.disabled {
    background-color: var(--gray-700);
    color: #9ca3af;
    cursor: not-allowed;
}

.disclaimer {
    text-align: center;
    color: #6b7280;
    font-size: 0.75rem;
    margin-top: 0.75rem;
    max-width: 48rem;
    margin-left: auto;
    margin-right: auto;
}

/* Message styles */
.message-item {
    display: flex;
    flex-direction: column;
    margin-bottom: 1.5rem;
}

.message-item.user {
    align-items: flex-end;
}

.message-item.bot {
    align-items: flex-start;
}

.message-content {
    display: flex;
    align-items: flex-start;
    max-width: 90%;
}

.message-item.user .message-content {
    flex-direction: row-reverse;
}

.message-avatar {
    width: 2.5rem;
    height: 2.5rem;
    border-radius: 9999px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 0.875rem;
    margin: 0 0.75rem;
}

.message-avatar.user {
    background-color: var(--rose-500);
}

.message-avatar.bot {
    background-color: var(--blue-500);
}

.message-bubble {
    padding: 1rem;
    border-radius: 1rem;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    max-width: 100%;
    overflow: auto;
}

.message-bubble.user {
    background-color: rgba(30, 58, 138, 0.5);
    border: 1px solid rgba(30, 64, 175, 0.5);
}

.message-bubble.bot {
    background-color: var(--gray-800);
    border: 1px solid rgba(55, 65, 81, 0.5);
}

.message-text {
    line-height: 1.5;
}

.message-text pre {
    margin: 1rem 0;
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: rgba(0, 0, 0, 0.3);
    overflow-x: auto;
}

.message-text code {
    font-family: monospace;
    background-color: rgba(0, 0, 0, 0.2);
    padding: 0.2rem 0.4rem;
    border-radius: 0.25rem;
}

.message-text a {
    color: var(--blue-500);
    text-decoration: none;
}

.message-text a:hover {
    text-decoration: underline;
}

.message-text table {
    border-collapse: collapse;
    margin: 1rem 0;
    width: 100%;
}

.message-text th,
.message-text td {
    border: 1px solid rgba(55, 65, 81, 0.5);
    padding: 0.5rem;
    text-align: left;
}

.message-text th {
    background-color: rgba(30, 30, 30, 0.5);
}

.message-text h1,
.message-text h2,
.message-text h3,
.message-text h4,
.message-text h5,
.message-text h6 {
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}

.message-text ul,
.message-text ol {
    margin: 0.75rem 0;
    padding-left: 1.5rem;
}

.message-text li {
    margin-bottom: 0.5rem;
}

.message-text blockquote {
    border-left: 4px solid var(--blue-500);
    padding-left: 1rem;
    margin: 1rem 0;
    color: #a0aec0;
}

.message-actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.5rem;
    margin-left: 4rem;
}

.action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: 9999px;
    background: transparent;
    border: none;
    color: #9ca3af;
    cursor: pointer;
    transition: all 0.2s;
}

.action-btn:hover {
    background-color: var(--gray-800);
    color: white;
}

.action-btn.liked {
    background-color: rgba(34, 197, 94, 0.2);
    color: #4ade80;
}

.action-btn.disliked {
    background-color: rgba(239, 68, 68, 0.2);
    color: #f87171;
}

/* Thinking animation */
.thinking-container {
    display: flex;
    flex-direction: column;
}

.thinking-bubble {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 1rem;
    border-radius: 1rem;
    background-color: rgba(31, 41, 55, 0.8);
    border: 1px solid rgba(55, 65, 81, 0.5);
    margin-bottom: 0.5rem;
}

.typing-dots {
    display: flex;
    gap: 0.25rem;
}

.typing-dot {
    width: 0.5rem;
    height: 0.5rem;
    background-color: var(--blue-400);
    border-radius: 9999px;
    animation: bounce 1.4s infinite ease-in-out;
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

.thinking-steps {
    background-color: rgba(31, 41, 55, 0.5);
    border: 1px solid rgba(55, 65, 81, 0.3);
    border-radius: 0.5rem;
    padding: 0.75rem;
    font-size: 0.875rem;
    color: #d1d5db;
    max-width: 25rem;
}

.thinking-header {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
}

.thinking-header i {
    color: var(--blue-400);
    margin-right: 0.5rem;
    width: 16px;
    height: 16px;
}

.thinking-title {
    font-size: 0.875rem;
    color: var(--blue-400);
    font-weight: 500;
}

.thinking-step {
    margin-top: 0.5rem;
}

/* Modals */
.modal {
    position: fixed;
    inset: 0;
    background-color: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 50;
    padding: 1rem;
}

.modal-content {
    background: linear-gradient(to bottom right, var(--gray-800), var(--gray-900));
    border-radius: 0.75rem;
    width: 100%;
    max-width: 28rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 50px rgba(0, 0, 0, 0.3);
}

.modal-close {
    position: absolute;
    top: 1rem;
    right: 1rem;
    background: transparent;
    border: none;
    color: #9ca3af;
    cursor: pointer;
    z-index: 10;
    width: 2rem;
    height: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 9999px;
    transition: all 0.2s;
}

.modal-close:hover {
    background-color: var(--gray-800);
    color: white;
}

/* Loading popup */
.loading-popup {
    position: fixed;
    inset: 0;
    background-color: rgba(0, 0, 0, 0.9);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 50;
}

.loading-animation {
    width: 5rem;
    height: 5rem;
    position: relative;
}

.loading-star {
    position: absolute;
    inset: 0;
    animation: spin 1s linear infinite;
}

.loading-star::before {
    content: "";
    position: absolute;
    width: 100%;
    height: 100%;
    background-image: url("data:image/svg+xml,%3Csvg width='80' height='80' viewBox='0 0 80 80' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M40 0C40 22.0914 22.0914 40 0 40C22.0914 40 40 57.9086 40 80C40 57.9086 57.9086 40 80 40C57.9086 40 40 22.0914 40 0Z' fill='%234285F4'/%3E%3C/svg%3E");
    background-size: contain;
}

.loading-dot {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: fadeIn 0.2s ease;
    animation-delay: 0.1s;
}

.loading-dot::before {
    content: "";
    width: 0.75rem;
    height: 0.75rem;
    background-color: white;
    border-radius: 50%;
}

/* Feedback toast */
.feedback-toast {
    position: fixed;
    bottom: 5rem;
    left: 50%;
    transform: translateX(-50%);
    background-color: var(--gray-800);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    z-index: 50;
    display: flex;
    align-items: center;
}

.feedback-icon {
    color: var(--blue-400);
    margin-right: 0.5rem;
    width: 16px;
    height: 16px;
}
/* Chat item styles for sidebar */
.chat-item {
    display: flex;
    padding: 0.75rem;
    border-radius: 0.5rem;
    cursor: pointer;
    transition: background-color 0.2s;
    border-bottom: 1px solid #2a2a2a;
}

.chat-item:hover {
    background-color: rgba(55, 65, 81, 0.5);
}

.chat-item-content {
    flex: 1;
    overflow: hidden;
}

.chat-item-title {
    font-size: 0.875rem;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: #e5e7eb;
}

.chat-item-preview {
    font-size: 0.75rem;
    color: #9ca3af;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* User profile styling */
.user-info {
    flex: 1;
    overflow: hidden;
}

.user-name {
    font-size: 0.875rem;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.user-email {
    font-size: 0.75rem;
    color: #9ca3af;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.user-avatar {
    width: 2rem;
    height: 2rem;
    background-color: var(--blue-500);
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    margin-right: 0.75rem;
}

.logout-btn {
    background: transparent;
    border: none;
    color: #9ca3af;
    width: 2rem;
    height: 2rem;
    border-radius: 9999px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s;
}

.logout-btn:hover {
    background-color: rgba(244, 63, 94, 0.1);
    color: var(--rose-500);
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .chat-sidebar {
        transform: translateX(-100%);
    }

    .chat-sidebar.open {
        transform: translateX(0);
    }

    .chat-main {
        margin-left: 0;
    }

    .sidebar-toggle {
        display: none;
    }

    .mobile-sidebar-toggle {
        position: fixed;
        bottom: 1rem;
        right: 1rem;
        z-index: 50;
        width: 3rem;
        height: 3rem;
        border-radius: 9999px;
        background-color: var(--blue-500);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        border: none;
        cursor: pointer;
    }

    /* Auth modal styling */
    .auth-modal {
        max-width: 420px;
        background: linear-gradient(to bottom, #1e1e1e, #121212);
        border-radius: 1rem;
        overflow: hidden;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(55, 65, 81, 0.5);
    }

    .auth-container {
        padding: 2rem;
    }

    .auth-header {
        text-align: center;
        margin-bottom: 2rem;
    }

    .auth-title {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #60a5fa, #a78bfa, #f87171);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-family: var(--font-heading);
    }

    .auth-subtitle {
        color: #9ca3af;
        font-size: 0.875rem;
    }

    .auth-form {
        display: flex;
        flex-direction: column;
        gap: 1.25rem;
    }

    .form-group {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .form-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #e5e7eb;
    }

    .label-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .forgot-password {
        font-size: 0.75rem;
        color: #60a5fa;
        background: transparent;
        border: none;
        cursor: pointer;
        transition: color 0.2s;
    }

    .forgot-password:hover {
        color: #93c5fd;
        text-decoration: underline;
    }

    .input-container {
        position: relative;
        display: flex;
        align-items: center;
    }

    .input-icon {
        position: absolute;
        left: 1rem;
        width: 1rem;
        height: 1rem;
        color: #6b7280;
    }

    .form-input {
        width: 100%;
        padding: 0.875rem 1rem 0.875rem 2.5rem;
        background-color: rgba(31, 41, 55, 0.5);
        border: 1px solid rgba(75, 85, 99, 0.5);
        border-radius: 0.5rem;
        color: white;
        font-size: 0.875rem;
        transition: all 0.2s;
    }

    .form-input:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25);
        background-color: rgba(31, 41, 55, 0.7);
    }

    .form-input::placeholder {
        color: #6b7280;
    }

    .input-help {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 0.25rem;
    }

    .toggle-password {
        position: absolute;
        right: 1rem;
        background: transparent;
        border: none;
        color: #6b7280;
        cursor: pointer;
        transition: color 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .toggle-password:hover {
        color: #9ca3af;
    }

    .toggle-password i {
        width: 1rem;
        height: 1rem;
    }

    .auth-submit-btn {
        margin-top: 0.5rem;
        padding: 0.875rem;
        background: linear-gradient(to right, #3b82f6, #2563eb);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        font-family: var(--font-heading);
    }

    .auth-submit-btn:hover {
        background: linear-gradient(to right, #2563eb, #1d4ed8);
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    .auth-submit-btn:active {
        transform: translateY(0);
    }

    .auth-footer {
        margin-top: 1.5rem;
        text-align: center;
    }

    .switch-prompt {
        font-size: 0.875rem;
        color: #9ca3af;
    }

    .switch-auth-btn {
        font-size: 0.875rem;
        font-weight: 600;
        color: #60a5fa;
        background: transparent;
        border: none;
        cursor: pointer;
        transition: color 0.2s;
    }

    .switch-auth-btn:hover {
        color: #93c5fd;
        text-decoration: underline;
    }

    /* Error message styling */
    .auth-error {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem;
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease;
    }

    .error-icon {
        color: #ef4444;
        width: 1rem;
        height: 1rem;
        flex-shrink: 0;
    }

    .auth-error span {
        font-size: 0.875rem;
        color: #fca5a5;
    }

    .fade-out {
        animation: fadeOut 0.3s ease forwards;
    }

    @keyframes fadeOut {
        from {
            opacity: 1;
        }
        to {
            opacity: 0;
        }
    }

    /* Mobile responsiveness for auth modal */
    @media (max-width: 640px) {
        .auth-modal {
            max-width: 100%;
            margin: 0 1rem;
        }

        .auth-container {
            padding: 1.5rem;
        }

        .auth-title {
            font-size: 1.5rem;
        }
    }

    /* Animation for form elements */
    .auth-form .form-group {
        animation: fadeSlideUp 0.3s ease forwards;
        opacity: 0;
        transform: translateY(10px);
    }

    .auth-form .form-group:nth-child(1) {
        animation-delay: 0.1s;
    }

    .auth-form .form-group:nth-child(2) {
        animation-delay: 0.2s;
    }

    .auth-form .form-group:nth-child(3) {
        animation-delay: 0.3s;
    }

    .auth-submit-btn {
        animation: fadeSlideUp 0.3s ease forwards;
        animation-delay: 0.4s;
        opacity: 0;
        transform: translateY(10px);
    }

    @keyframes fadeSlideUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
}

/* Center chat content on desktop */
@media (min-width: 768px) {
  .chat-main {
    /* Keep the left margin for the fixed sidebar */
    margin-left: 280px;
    display: flex;
    justify-content: center;
  }
  /* This new container should wrap the chat content inside .chat-main */
  .chat-main > .chat-content {
    max-width: 1200px;
    width: 100%;
    padding: 0 20px;
  }

    #login-modal {
        position: fixed;
        inset: 0;
        z-index: 9999;
        display: flex;
        justify-content: center;
        align-items: center;
        background: rgba(0,0,0,0.6);
    }

    .auth-modal {
        background: #1a1b20;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 0 30px rgba(0,0,0,0.5);
        width: 360px;
        font-family: 'Inter', sans-serif;
    }

    .auth-title {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .auth-subtitle {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-bottom: 1.5rem;
    }

    .auth-form label {
        font-weight: 500;
        font-size: 0.85rem;
        color: #fff;
        margin-bottom: 0.25rem;
        display: block;
    }

    .label-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 1.2rem;
        margin-bottom: 0.25rem;
    }

    .forgot-link {
        font-size: 0.8rem;
        color: #60a5fa;
        text-decoration: none;
    }

    .input-box {
        position: relative;
        margin-bottom: 1rem;
    }

    .input-box input {
        width: 100%;
        padding: 0.75rem 0.75rem 0.75rem 2.5rem;
        background: #2a2b30;
        border: 1px solid #444;
        border-radius: 12px;
        color: #fff;
        font-size: 0.95rem;
        outline: none;
    }

    .input-icon {
        position: absolute;
        left: 0.75rem;
        top: 50%;
        transform: translateY(-50%);
        color: #9ca3af;
    }

    .auth-button {
        width: 100%;
        padding: 0.75rem;
        background: linear-gradient(90deg, #6366f1, #3b82f6);
        border: none;
        color: #fff;
        font-weight: bold;
        font-size: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .auth-button:hover {
        opacity: 0.9;
    }

    .auth-footer {
        text-align: center;
        font-size: 0.9rem;
        color: #aaa;
        margin-top: 1.2rem;
    }

    .signup-link {
        color: #60a5fa;
        margin-left: 4px;
        text-decoration: none;
    }
}














// This file contains functions for handling chat-specific functionality
// It will be imported by app.js

// Python API integration
async function callPythonAPI(query) {
    try {
        const response = await fetch('/api/python-response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query }),
        });
        if (!response.ok) {
            throw new Error('API request failed');
        }
        return await response.json();
    } catch (error) {
        console.error('Error calling Python API:', error);
        return {
            error: true,
            message: 'Failed to get response from the server. Please try again.'
        };
    }
}

// Generate thinking steps based on the user's input
function generateThinkingSteps(input) {
    const thinkingSteps = [
        "Analyzing the query to understand the user's intent...",
        "Identifying key concepts and entities in the question...",
        "Retrieving relevant information from CoPPER documentation...",
        "Considering different perspectives and approaches...",
        "Organizing information in a clear and helpful way...",
        "Checking for potential misunderstandings or ambiguities...",
        "Formulating a comprehensive and accurate response...",
    ];
    // Generate a random number of thinking steps (3-5)
    const numSteps = Math.floor(Math.random() * 3) + 3;
    const selectedSteps = thinkingSteps.sort(() => 0.5 - Math.random()).slice(0, numSteps);
    
    // Add some context based on the user's input
    let contextualThinking = "";
    if (input.toLowerCase().includes("map") || input.toLowerCase().includes("convert")) {
        contextualThinking =
            "This appears to be a mapping request. I'll focus on converting between SQL view and API format.";
    } else if (input.toLowerCase().includes("json") || input.toLowerCase().includes("generate")) {
        contextualThinking =
            "This is a request to generate API JSON. I'll focus on creating the proper request format.";
    } else if (input.toLowerCase().includes("explain")) {
        contextualThinking =
            "This is an explanation request. I'll focus on providing clear details about the relationship.";
    } else if (input.toLowerCase().includes("analyze") || input.toLowerCase().includes("view")) {
        contextualThinking =
            "This is a view analysis request. I'll focus on understanding the SQL structure and purpose.";
    }
    
    if (contextualThinking) {
        selectedSteps.unshift(contextualThinking);
    }
    
    return selectedSteps;
}

// Local storage functions
function saveToLocalStorage(key, data) {
    localStorage.setItem(key, JSON.stringify(data));
}

function getFromLocalStorage(key) {
    const data = localStorage.getItem(key);
    return data ? JSON.parse(data) : null;
}

// User management
function saveUser(user) {
    saveToLocalStorage('currentUser', user);
}

function getUser() {
    return getFromLocalStorage('currentUser');
}

function clearUser() {
    localStorage.removeItem('currentUser');
}

// Chat history management
function saveChat(chat) {
    const chats = getChats();
    const updatedChats = [chat, ...chats].slice(0, 10); // Keep only 10 most recent
    saveToLocalStorage('chats', updatedChats);
}

function getChats() {
    return getFromLocalStorage('chats') || [];
}

function getChatById(id) {
    const chats = getChats();
    return chats.find(chat => chat.id === id);
}

// Format markdown in bot responses
function renderMarkdown(text) {
    // Simple markdown parser
    // Headers
    text = text.replace(/^# (.*$)/gm, '<h1>$1</h1>');
    text = text.replace(/^## (.*$)/gm, '<h2>$1</h2>');
    text = text.replace(/^### (.*$)/gm, '<h3>$1</h3>');
    
    // Code blocks with syntax highlighting
    text = text.replace(/```(\w+)?\n([\s\S]*?)```/gm, function(match, lang, code) {
        return `<pre><code class="language-${lang || ''}">${code.trim()}</code></pre>`;
    });
    
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Bold
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Italic
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    // Lists
    text = text.replace(/^\s*-\s+(.*$)/gm, '<li>$1</li>');
    text = text.replace(/(<li>.*<\/li>\n)+/g, '<ul>$&</ul>');
    
    // Links
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    
    // Paragraphs
    text = text.replace(/\n\n/g, '<br><br>');
    
    return text;
}













// This file contains functions for creating UI components dynamically
// It will be imported by app.js

// Create login modal with improved UI
function createLoginModal(onClose, onLogin) {
    const modal = document.getElementById('login-modal');

    modal.innerHTML = `
    <div class="modal-content auth-modal">
      <button class="modal-close" id="login-modal-close">
        <i data-lucide="x"></i>
      </button>
      
      <div class="auth-container">
        <div class="auth-header">
          <h2 class="auth-title">Welcome back</h2>
          <p class="auth-subtitle">Sign in to continue your CoPPER journey</p>
        </div>
        
        <form id="login-form" class="auth-form">
          <div class="form-group">
            <label for="email" class="form-label">Email</label>
            <div class="input-container">
              <i data-lucide="mail" class="input-icon"></i>
              <input
                id="email"
                type="email"
                class="form-input"
                placeholder="you@test.com"
                required
              />
            </div>
          </div>
          
          <div class="form-group">
            <div class="label-container">
              <label for="password" class="form-label">Password</label>
              <button type="button" class="forgot-password">Forgot password?</button>
            </div>
            <div class="input-container">
              <i data-lucide="lock" class="input-icon"></i>
              <input
                id="password"
                type="password"
                class="form-input"
                placeholder="••••••••"
                required
              />
              <button type="button" class="toggle-password" data-show="false">
                <i data-lucide="eye" class="icon-show"></i>
                <i data-lucide="eye-off" class="icon-hide hidden"></i>
              </button>
            </div>
          </div>
          
          <button
            type="submit"
            class="auth-submit-btn"
          >
            Sign In
          </button>
        </form>
        
        <div class="auth-footer">
          <p class="switch-prompt">
            Don't have an account?
            <button id="switch-to-signup" class="switch-auth-btn">
              Sign Up
            </button>
          </p>
        </div>
      </div>
    </div>
  `;

    modal.classList.remove('hidden');

    // Add Escape key functionality to close the modal
    function escHandler(e) {
      if (e.key === 'Escape') {
        modal.classList.add('hidden');
        if (onClose) onClose();
        document.removeEventListener('keydown', escHandler);
      }
    }
    document.addEventListener('keydown', escHandler);

    // Initialize icons
    lucide.createIcons({
        icons: {
            'x': modal.querySelector('[data-lucide="x"]'),
            'mail': modal.querySelector('[data-lucide="mail"]'),
            'lock': modal.querySelector('[data-lucide="lock"]'),
            'eye': modal.querySelector('[data-lucide="eye"]'),
            'eye-off': modal.querySelector('[data-lucide="eye-off"]')
        }
    });

    // Toggle password visibility
    const togglePasswordBtn = modal.querySelector('.toggle-password');
    if (togglePasswordBtn) {
        togglePasswordBtn.addEventListener('click', () => {
            const passwordInput = modal.querySelector('#password');
            const iconShow = togglePasswordBtn.querySelector('.icon-show');
            const iconHide = togglePasswordBtn.querySelector('.icon-hide');
            const isShowing = togglePasswordBtn.getAttribute('data-show') === 'true';

            if (isShowing) {
                passwordInput.type = 'password';
                iconShow.classList.remove('hidden');
                iconHide.classList.add('hidden');
                togglePasswordBtn.setAttribute('data-show', 'false');
            } else {
                passwordInput.type = 'text';
                iconShow.classList.add('hidden');
                iconHide.classList.remove('hidden');
                togglePasswordBtn.setAttribute('data-show', 'true');
            }
        });
    }

    // Add event listeners
    modal.querySelector('#login-modal-close').addEventListener('click', () => {
        modal.classList.add('hidden');
        if (onClose) onClose();
        document.removeEventListener('keydown', escHandler);
    });

    modal.querySelector('#login-form').addEventListener('submit', (e) => {
        e.preventDefault();

        const email = modal.querySelector('#email').value;
        const password = modal.querySelector('#password').value;

        // Simple validation
        if (!email || !password) {
            showAuthError('Please fill in all fields');
            return;
        }

        // Check if the email ends with @test.com
        if (!email.endsWith('@test.com')) {
            showAuthError('Only @test.com email addresses are allowed');
            return;
        }

        // Check if user exists in local storage
        const users = JSON.parse(localStorage.getItem('users')) || [];
        const user = users.find(u => u.email === email);

        if (!user) {
            showAuthError('User not found. Please sign up first.');
            return;
        }

        if (user.password !== password) {
            showAuthError('Incorrect password');
            return;
        }

        if (onLogin) onLogin(user);
        modal.classList.add('hidden');
    });

    modal.querySelector('#switch-to-signup').addEventListener('click', () => {
        createSignupModal(onClose, onLogin);
    });

    // Helper function to show validation errors
    function showAuthError(message) {
        // Remove any existing error message
        const existingError = modal.querySelector('.auth-error');
        if (existingError) {
            existingError.remove();
        }

        // Create and insert error message
        const errorElement = document.createElement('div');
        errorElement.className = 'auth-error';
        errorElement.innerHTML = `
            <i data-lucide="alert-circle" class="error-icon"></i>
            <span>${message}</span>
        `;

        const form = modal.querySelector('#login-form');
        form.insertBefore(errorElement, form.firstChild);

        // Initialize the error icon
        lucide.createIcons({
            icons: {
                'alert-circle': errorElement.querySelector('[data-lucide="alert-circle"]')
            }
        });

        // Automatically remove the error after 5 seconds
        setTimeout(() => {
            errorElement.classList.add('fade-out');
            setTimeout(() => {
                errorElement.remove();
            }, 300);
        }, 5000);
    }
}

// Create signup modal with improved UI
function createSignupModal(onClose, onLogin) {
    const modal = document.getElementById('login-modal');

    modal.innerHTML = `
    <div class="modal-content auth-modal">
      <button class="modal-close" id="signup-modal-close">
        <i data-lucide="x"></i>
      </button>
      
      <div class="auth-container">
        <div class="auth-header">
          <h2 class="auth-title">Create an account</h2>
          <p class="auth-subtitle">Join our CoPPER assistant platform</p>
        </div>
        
        <form id="signup-form" class="auth-form">
          <div class="form-group">
            <label for="name" class="form-label">Name</label>
            <div class="input-container">
              <i data-lucide="user" class="input-icon"></i>
              <input
                id="name"
                type="text"
                class="form-input"
                placeholder="Your name"
                required
              />
            </div>
          </div>
          
          <div class="form-group">
            <label for="email" class="form-label">Email</label>
            <div class="input-container">
              <i data-lucide="mail" class="input-icon"></i>
              <input
                id="email"
                type="email"
                class="form-input"
                placeholder="you@test.com"
                required
              />
            </div>
            <span class="input-help">Only @test.com emails are accepted</span>
          </div>
          
          <div class="form-group">
            <label for="password" class="form-label">Password</label>
            <div class="input-container">
              <i data-lucide="lock" class="input-icon"></i>
              <input
                id="password"
                type="password"
                class="form-input"
                placeholder="••••••••"
                required
              />
              <button type="button" class="toggle-password" data-show="false">
                <i data-lucide="eye" class="icon-show"></i>
                <i data-lucide="eye-off" class="icon-hide hidden"></i>
              </button>
            </div>
          </div>
          
          <button
            type="submit"
            class="auth-submit-btn"
          >
            Sign Up
          </button>
        </form>
        
        <div class="auth-footer">
          <p class="switch-prompt">
            Already have an account?
            <button id="switch-to-login" class="switch-auth-btn">
              Sign In
            </button>
          </p>
        </div>
      </div>
    </div>
  `;

    modal.classList.remove('hidden');

    // Add Escape key functionality to close the modal
    function escHandler(e) {
      if (e.key === 'Escape') {
        modal.classList.add('hidden');
        if (onClose) onClose();
        document.removeEventListener('keydown', escHandler);
      }
    }
    document.addEventListener('keydown', escHandler);

    // Initialize icons
    lucide.createIcons({
        icons: {
            'x': modal.querySelector('[data-lucide="x"]'),
            'user': modal.querySelector('[data-lucide="user"]'),
            'mail': modal.querySelector('[data-lucide="mail"]'),
            'lock': modal.querySelector('[data-lucide="lock"]'),
            'eye': modal.querySelector('[data-lucide="eye"]'),
            'eye-off': modal.querySelector('[data-lucide="eye-off"]')
        }
    });

    // Toggle password visibility
    const togglePasswordBtn = modal.querySelector('.toggle-password');
    if (togglePasswordBtn) {
        togglePasswordBtn.addEventListener('click', () => {
            const passwordInput = modal.querySelector('#password');
            const iconShow = togglePasswordBtn.querySelector('.icon-show');
            const iconHide = togglePasswordBtn.querySelector('.icon-hide');
            const isShowing = togglePasswordBtn.getAttribute('data-show') === 'true';

            if (isShowing) {
                passwordInput.type = 'password';
                iconShow.classList.remove('hidden');
                iconHide.classList.add('hidden');
                togglePasswordBtn.setAttribute('data-show', 'false');
            } else {
                passwordInput.type = 'text';
                iconShow.classList.add('hidden');
                iconHide.classList.remove('hidden');
                togglePasswordBtn.setAttribute('data-show', 'true');
            }
        });
    }

    // Add event listeners
    modal.querySelector('#signup-modal-close').addEventListener('click', () => {
        modal.classList.add('hidden');
        if (onClose) onClose();
        document.removeEventListener('keydown', escHandler);
    });

    modal.querySelector('#signup-form').addEventListener('submit', (e) => {
        e.preventDefault();

        const name = modal.querySelector('#name').value;
        const email = modal.querySelector('#email').value;
        const password = modal.querySelector('#password').value;

        // Simple validation
        if (!name || !email || !password) {
            showAuthError('Please fill in all fields');
            return;
        }

        // Check if the email ends with @test.com
        if (!email.endsWith('@test.com')) {
            showAuthError('Only @test.com email addresses are allowed');
            return;
        }

        // Check if user already exists
        const users = JSON.parse(localStorage.getItem('users')) || [];
        if (users.some(u => u.email === email)) {
            showAuthError('User with this email already exists');
            return;
        }

        // Create and save user
        const user = {
            id: Date.now().toString(),
            name: name,
            email: email,
            password: password // In a real app, this should be encrypted
        };

        users.push(user);
        localStorage.setItem('users', JSON.stringify(users));

        if (onLogin) onLogin(user);
        modal.classList.add('hidden');
    });

    modal.querySelector('#switch-to-login').addEventListener('click', () => {
        createLoginModal(onClose, onLogin);
    });

    // Helper function to show validation errors
    function showAuthError(message) {
        // Remove any existing error message
        const existingError = modal.querySelector('.auth-error');
        if (existingError) {
            existingError.remove();
        }

        // Create and insert error message
        const errorElement = document.createElement('div');
        errorElement.className = 'auth-error';
        errorElement.innerHTML = `
            <i data-lucide="alert-circle" class="error-icon"></i>
            <span>${message}</span>
        `;

        const form = modal.querySelector('#signup-form');
        form.insertBefore(errorElement, form.firstChild);

        // Initialize the error icon
        lucide.createIcons({
            icons: {
                'alert-circle': errorElement.querySelector('[data-lucide="alert-circle"]')
            }
        });

        // Automatically remove the error after 5 seconds
        setTimeout(() => {
            errorElement.classList.add('fade-out');
            setTimeout(() => {
                errorElement.remove();
            }, 300);
        }, 5000);
    }
}

// Export functions
window.uiComponents = {
    createLoginModal,
    createSignupModal
};












document.addEventListener('DOMContentLoaded', function() {
    // Initialize Lucide icons
    lucide.createIcons();

    // DOM elements
    const landingPage = document.getElementById('landing-page');
    const chatInterface = document.getElementById('chat-interface');
    const exploreBtn = document.getElementById('explore-btn');
    const loginBtn = document.getElementById('login-btn');
    const sidebarLoginBtn = document.getElementById('sidebar-login-btn');
    const returnHomeBtn = document.getElementById('return-home-btn');
    const loadingPopup = document.getElementById('loading-popup');
    const circleContainer = document.getElementById('circle-container');
    const chatSidebar = document.getElementById('chat-sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    const chatSuggestions = document.getElementById('chat-suggestions');
    const newChatBtn = document.getElementById('new-chat-btn');

    // State
    let rotationAngle = 0;
    let activeCircleIndex = null;
    let user = null;
    let showSidebar = true;

    // Demo data
    const demoChats = [
        {
            id: 1,
            title: "View Analysis",
            preview: "Understand SQL view structure...",
            conversation: [
                { type: "user", text: "What can you tell me about SQL views?" },
                {
                    type: "bot",
                    text: "SQL views are virtual tables that represent data from one or more tables. In CoPPER, views provide a way to access complex data through a simpler interface.",
                },
            ],
        },
        {
            id: 2,
            title: "API Mapping",
            preview: "Convert SQL to API format...",
            conversation: [
                { type: "user", text: "How do I map a SQL view to an API?" },
                {
                    type: "bot",
                    text: "To map a SQL view to an API, we analyze the view structure, identify the data domain, and convert WHERE conditions to request parameters.",
                },
            ],
        },
        {
            id: 3,
            title: "JSON Generation",
            preview: "Create API request bodies...",
            conversation: [
                { type: "user", text: "Generate JSON for a product view" },
                {
                    type: "bot",
                    text: "Here's an example JSON request for a product view: { \"reqName\": \"P1\", \"dataDomain\": \"products\", \"type\": \"independent\"... }",
                },
            ],
        },
        {
            id: 4,
            title: "CoPPER Overview",
            preview: "Learn about the system...",
            conversation: [
                { type: "user", text: "What is CoPPER?" },
                {
                    type: "bot",
                    text: "CoPPER is a comprehensive database and API system used at CME Group that provides access to financial data through both SQL views and REST API endpoints.",
                },
            ],
        },
        {
            id: 5,
            title: "Data Domains",
            preview: "Understanding domains...",
            conversation: [
                { type: "user", text: "What are data domains in CoPPER?" },
                {
                    type: "bot",
                    text: "CoPPER organizes data into domains like products, instruments, sessions, and firms. Each domain has specific endpoints and request parameters.",
                },
            ],
        },
    ];

    // Initialize circle feature
    function initCircleFeature() {
        // Rotate the circle continuously
        setInterval(() => {
            rotationAngle = (rotationAngle + 0.2) % 360;
            circleContainer.style.transform = `rotate(${rotationAngle}deg)`;
        }, 50);

        // Create circle items
        demoChats.forEach((chat, index) => {
            const position = generateCirclePosition(index, demoChats.length);
            const circleItem = document.createElement('div');
            circleItem.className = 'circle-item';
            circleItem.style.transform = `translate(${position.x}px, ${position.y}px) rotate(-${rotationAngle}deg)`;
            circleItem.style.width = '150px';
            circleItem.style.zIndex = '5';

            circleItem.innerHTML = `
                <div class="chat-preview">
                  <h3 class="chat-title">${chat.title}</h3>
                  <p class="chat-description">${chat.preview}</p>
                </div>
            `;

            circleItem.addEventListener('mouseenter', () => {
                handleCircleItemHover(circleItem, chat, index);
            });

            circleItem.addEventListener('mouseleave', () => {
                handleCircleItemLeave(circleItem, index);
            });

            circleContainer.appendChild(circleItem);
        });
    }

    // Handle circle item hover
    function handleCircleItemHover(element, chat, index) {
        activeCircleIndex = index;
        element.classList.add('active');
        element.classList.remove('inactive');
        element.style.width = '280px';
        element.style.zIndex = '20';

        // Add conversation if not already added
        if (!element.querySelector('.demo-conversation')) {
            const conversation = document.createElement('div');
            conversation.className = 'demo-conversation';
            conversation.innerHTML = `
                <div class="demo-message">
                  <div class="demo-avatar user-avatar">You</div>
                  <div class="demo-text user-text">${chat.conversation[0].text}</div>
                </div>
                <div class="demo-message">
                  <div class="demo-avatar bot-avatar">AI</div>
                  <div class="demo-text bot-text">${chat.conversation[1].text}</div>
                </div>
            `;
            element.appendChild(conversation);
        }
    }

    // Handle circle item leave
    function handleCircleItemLeave(element, index) {
        activeCircleIndex = null;
        element.classList.remove('active');
        element.classList.add('inactive');
        element.style.width = '150px';
        element.style.zIndex = '5';

        // Remove the conversation element when leaving
        const conversation = element.querySelector('.demo-conversation');
        if (conversation) {
            conversation.remove();
        }
    }

    // Calculate circle positions
    function generateCirclePosition(index, total) {
        const radius = 220;
        const angle = (2 * Math.PI * index) / total;
        const x = radius * Math.cos(angle);
        const y = radius * Math.sin(angle);
        return { x, y };
    }

    // Event listeners
    exploreBtn.addEventListener('click', () => {
        loadingPopup.classList.remove('hidden');
        setTimeout(() => {
            landingPage.classList.add('hidden');
            chatInterface.classList.remove('hidden');
            loadingPopup.classList.add('hidden');
        }, 800);
    });

    returnHomeBtn.addEventListener('click', () => {
        chatInterface.classList.add('hidden');
        landingPage.classList.remove('hidden');
    });

    // Login button event listeners
    loginBtn.addEventListener('click', () => {
        window.uiComponents.createLoginModal(
            null, // onClose callback
            (user) => {
                // onLogin callback
                updateUserProfileUI(user);
            }
        );
    });

    if (sidebarLoginBtn) {
        sidebarLoginBtn.addEventListener('click', () => {
            window.uiComponents.createLoginModal(
                null, // onClose callback
                (user) => {
                    // onLogin callback
                    updateUserProfileUI(user);
                }
            );
        });
    }

    // Update user profile UI after login
    function updateUserProfileUI(user) {
        const userProfile = document.getElementById('user-profile');
        const sidebarLoginBtn = document.getElementById('sidebar-login-btn');

        if (sidebarLoginBtn) {
            sidebarLoginBtn.classList.add('hidden');
        }

        userProfile.innerHTML = `
            <div class="user-avatar">${user.name.charAt(0).toUpperCase()}</div>
            <div class="user-info">
                <div class="user-name">${user.name}</div>
                <div class="user-email">${user.email}</div>
            </div>
            <button id="logout-btn" class="logout-btn">
                <i data-lucide="log-out"></i>
            </button>
        `;

        // Initialize the icon
        lucide.createIcons({
            icons: {
                'log-out': userProfile.querySelector('[data-lucide="log-out"]')
            }
        });

        // Add logout functionality
        userProfile.querySelector('#logout-btn').addEventListener('click', () => {
            localStorage.removeItem('currentUser');
            location.reload(); // Simple way to reset the UI
        });

        // Save current user
        localStorage.setItem('currentUser', JSON.stringify(user));
    }

    // Initialize chat interface
    function initChatInterface() {
        // Load and display recent chats from local storage
        loadRecentChats();

        // Set up chat suggestions
        document.querySelectorAll('.suggestion-item').forEach(item => {
            item.addEventListener('click', () => {
                const suggestion = item.getAttribute('data-suggestion');
                chatInput.value = suggestion;
                sendMessage();
            });
        });

        // Set up chat input
        chatInput.addEventListener('input', () => {
            if (chatInput.value.trim()) {
                sendButton.classList.remove('disabled');
            } else {
                sendButton.classList.add('disabled');
            }
        });

        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && chatInput.value.trim()) {
                sendMessage();
            }
        });

        sendButton.addEventListener('click', () => {
            if (chatInput.value.trim()) {
                sendMessage();
            }
        });

        newChatBtn.addEventListener('click', () => {
            startNewChat();
        });

        // Toggle sidebar
        sidebarToggle.addEventListener('click', () => {
            toggleSidebar();
        });
    }

    // Load recent chats from local storage
    function loadRecentChats() {
        const recentChats = getFromLocalStorage('chats') || [];
        const chatsList = document.getElementById('chats-list');

        if (recentChats.length > 0) {
            // Clear empty state
            chatsList.innerHTML = '';

            // Add recent chats to the sidebar
            recentChats.forEach(chat => {
                const chatItem = document.createElement('div');
                chatItem.className = 'chat-item';
                chatItem.innerHTML = `
                    <div class="chat-item-content">
                        <div class="chat-item-title">${chat.title}</div>
                        <div class="chat-item-preview">${chat.preview}</div>
                    </div>
                `;
                chatItem.addEventListener('click', () => {
                    loadChat(chat);
                });
                chatsList.appendChild(chatItem);
            });
        }
    }

    // Load a chat from history
    function loadChat(chat) {
        chatSuggestions.classList.add('hidden');
        chatMessages.innerHTML = '';

        // Display chat messages
        chat.conversation.forEach(message => {
            addMessage(message.type, message.text);
        });
    }

    // Helper function to get data from localStorage
    function getFromLocalStorage(key) {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : null;
    }

    // Send message function
    async function sendMessage() {
        const messageText = chatInput.value.trim();
        if (!messageText) return;

        // Hide suggestions
        chatSuggestions.classList.add('hidden');

        // Add user message
        addMessage('user', messageText);

        // Clear input
        chatInput.value = '';
        sendButton.classList.add('disabled');

        // Show thinking animation
        showThinking(messageText);

        try {
            // Call Python backend API
            const response = await callPythonAPI(messageText);
            
            // Hide thinking animation
            hideThinking();
            
            if (response.error) {
                // Show error message
                addMessage('bot', `Error: ${response.message}`);
            } else {
                // Show response
                addMessage('bot', response.result);
                
                // Save conversation to local storage
                saveChat(messageText, response.result);
            }
        } catch (error) {
            // Hide thinking animation
            hideThinking();
            
            // Show error message
            addMessage('bot', 'Sorry, I encountered an error while processing your request. Please try again.');
            console.error('Error processing message:', error);
        }
    }

    // Save chat to local storage
    function saveChat(userMessage, botMessage) {
        const chats = JSON.parse(localStorage.getItem('chats')) || [];

        // Create a new chat object
        const newChat = {
            id: Date.now().toString(),
            title: userMessage.slice(0, 30) + (userMessage.length > 30 ? '...' : ''),
            preview: botMessage.slice(0, 40) + (botMessage.length > 40 ? '...' : ''),
            conversation: [
                { type: 'user', text: userMessage },
                { type: 'bot', text: botMessage }
            ],
            timestamp: new Date().toISOString()
        };

        // Add to beginning of array (most recent first)
        chats.unshift(newChat);

        // Keep only the 10 most recent chats
        const updatedChats = chats.slice(0, 10);

        // Save to localStorage
        localStorage.setItem('chats', JSON.stringify(updatedChats));

        // Refresh the chats list in the sidebar
        loadRecentChats();
    }

    // Add message to chat
    function addMessage(type, text) {
        const messageItem = document.createElement('div');
        messageItem.className = `message-item ${type}`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        const avatar = document.createElement('div');
        avatar.className = `message-avatar ${type}`;

        // Get current user from localStorage for avatar
        const currentUser = JSON.parse(localStorage.getItem('currentUser'));
        avatar.textContent = type === 'user' ?
            (currentUser ? currentUser.name.charAt(0).toUpperCase() : 'U') : 'AI';

        const bubble = document.createElement('div');
        bubble.className = `message-bubble ${type}`;

        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        
        // Format the text with markdown for bot messages
        if (type === 'bot') {
            messageText.innerHTML = renderMarkdown(text);
        } else {
            messageText.textContent = text;
        }

        bubble.appendChild(messageText);
        messageContent.appendChild(avatar);
        messageContent.appendChild(bubble);
        messageItem.appendChild(messageContent);

        // Add message actions for bot messages
        if (type === 'bot') {
            const actions = document.createElement('div');
            actions.className = 'message-actions';
            actions.innerHTML = `
                <button class="action-btn" title="Like">
                  <i data-lucide="thumbs-up"></i>
                </button>
                <button class="action-btn" title="Dislike">
                  <i data-lucide="thumbs-down"></i>
                </button>
                <button class="action-btn" title="Regenerate">
                  <i data-lucide="refresh-cw"></i>
                </button>
                <button class="action-btn" title="Share">
                  <i data-lucide="share-2"></i>
                </button>
                <button class="action-btn" title="More">
                  <i data-lucide="more-vertical"></i>
                </button>
            `;
            messageItem.appendChild(actions);

            // Initialize the new icons
            lucide.createIcons({
                icons: {
                    'thumbs-up': messageItem.querySelector('[data-lucide="thumbs-up"]'),
                    'thumbs-down': messageItem.querySelector('[data-lucide="thumbs-down"]'),
                    'refresh-cw': messageItem.querySelector('[data-lucide="refresh-cw"]'),
                    'share-2': messageItem.querySelector('[data-lucide="share-2"]'),
                    'more-vertical': messageItem.querySelector('[data-lucide="more-vertical"]')
                }
            });

            // Add event listeners for actions
            const likeBtn = actions.querySelector('[title="Like"]');
            const dislikeBtn = actions.querySelector('[title="Dislike"]');
            const regenerateBtn = actions.querySelector('[title="Regenerate"]');
            const shareBtn = actions.querySelector('[title="Share"]');

            likeBtn.addEventListener('click', () => {
                likeBtn.classList.toggle('liked');
                if (likeBtn.classList.contains('liked')) {
                    dislikeBtn.classList.remove('disliked');
                    showFeedbackToast('Thanks for your feedback!');
                }
            });

            dislikeBtn.addEventListener('click', () => {
                dislikeBtn.classList.toggle('disliked');
                if (dislikeBtn.classList.contains('disliked')) {
                    likeBtn.classList.remove('liked');
                    showFeedbackToast('We\'ll try to improve our responses.');
                }
            });

            regenerateBtn.addEventListener('click', async () => {
                messageItem.remove();
                showThinking(chatMessages.querySelector('.message-item.user:last-child .message-text').textContent);
                
                try {
                    // Call Python backend API again
                    const userQuery = chatMessages.querySelector('.message-item.user:last-child .message-text').textContent;
                    const response = await callPythonAPI(userQuery);
                    
                    // Hide thinking animation
                    hideThinking();
                    
                    if (response.error) {
                        // Show error message
                        addMessage('bot', `Error: ${response.message}`);
                    } else {
                        // Show response
                        addMessage('bot', response.result);
                    }
                } catch (error) {
                    // Hide thinking animation
                    hideThinking();
                    
                    // Show error message
                    addMessage('bot', 'Sorry, I encountered an error while regenerating the response. Please try again.');
                }
            });

            shareBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(text)
                    .then(() => {
                        showFeedbackToast('Response copied to clipboard!');
                    })
                    .catch(() => {
                        showFeedbackToast('Failed to copy response.');
                    });
            });
        }

        chatMessages.appendChild(messageItem);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Show thinking animation
    function showThinking(input) {
        const thinkingItem = document.createElement('div');
        thinkingItem.className = 'message-item bot thinking';
        thinkingItem.id = 'thinking-animation';

        // Generate thinking steps based on the input
        const thinkingSteps = generateThinkingSteps(input);

        thinkingItem.innerHTML = `
            <div class="message-content">
                <div class="message-avatar bot">AI</div>
                <div class="thinking-container">
                    <div class="thinking-bubble">
                        <div class="thinking-text">Thinking</div>
                        <div class="typing-dots">
                          <div class="typing-dot"></div>
                          <div class="typing-dot"></div>
                          <div class="typing-dot"></div>
                        </div>
                    </div>
                  
                    <div class="thinking-steps">
                        <div class="thinking-header">
                          <i data-lucide="sparkles"></i>
                          <div class="thinking-title">Thinking process</div>
                        </div>
                        <div class="thinking-steps-content">
                          ${thinkingSteps.map(step => `<div class="thinking-step">${step}</div>`).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;

        chatMessages.appendChild(thinkingItem);

        // Initialize the sparkles icon
        lucide.createIcons({
            icons: {
                'sparkles': thinkingItem.querySelector('[data-lucide="sparkles"]')
            }
        });

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Hide thinking animation
    function hideThinking() {
        const thinkingItem = document.getElementById('thinking-animation');
        if (thinkingItem) {
            thinkingItem.remove();
        }
    }

    // Show feedback toast
    function showFeedbackToast(message) {
        const feedbackToast = document.getElementById('feedback-toast');
        const feedbackMessage = feedbackToast.querySelector('.feedback-message');

        feedbackMessage.textContent = message;
        feedbackToast.classList.remove('hidden');

        setTimeout(() => {
            feedbackToast.classList.add('hidden');
        }, 3000);
    }

    // Start new chat
    function startNewChat() {
        chatMessages.innerHTML = '';
        chatSuggestions.classList.remove('hidden');
    }

    // Toggle sidebar
    function toggleSidebar() {
        showSidebar = !showSidebar;

        if (showSidebar) {
            chatSidebar.style.transform = 'translateX(0)';
            sidebarToggle.style.left = '280px';
            sidebarToggle.querySelector('.toggle-icon').style.transform = 'rotate(0)';
            document.getElementById('chat-main').style.marginLeft = '280px';
        } else {
            chatSidebar.style.transform = 'translateX(-100%)';
            sidebarToggle.style.left = '0';
            sidebarToggle.querySelector('.toggle-icon').style.transform = 'rotate(180deg)';
            document.getElementById('chat-main').style.marginLeft = '0';
        }
    }

    // Check if user is already logged in
    function checkLoggedInUser() {
        const currentUser = JSON.parse(localStorage.getItem('currentUser'));
        if (currentUser) {
            updateUserProfileUI(currentUser);
        }
    }

    // Initialize
    initCircleFeature();
    initChatInterface();
    checkLoggedInUser();
});













































#!/usr/bin/env python3
"""
BMC Helix + Gemini Chatbot - All-in-One Solution

This single file contains the complete integration between BMC Remedy Helix and Google's Gemini AI
to create an intelligent chatbot for incident management and analysis.

Features:
- BMC Helix API integration with robust error handling
- Local data caching to improve reliability and performance
- Gemini AI integration using your existing working configuration
- Natural language query interpretation and analysis
- Interactive command-line interface for testing

Usage:
    python3 bmc_helix_gemini_all_in_one.py
"""

import os
import sys
import logging
import json
import time
import pickle
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

# Disable SSL warnings for development
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bmc_helix_gemini_chatbot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("HelixGeminiChatbot")

# Import Gemini AI dependencies - using your working configuration
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")

# BMC Helix configuration
BMC_SERVER = "cmegroup-restapi.onbmc.com"
BMC_USERNAME = "username"  # Replace with actual username for production
BMC_PASSWORD = "password"  # Replace with actual password for production

# Cache settings
CACHE_DIR = "cache"
INCIDENT_CACHE_FILE = os.path.join(CACHE_DIR, "incidents_cache.pkl")
CACHE_EXPIRY_HOURS = 6  # Refresh cache every 6 hours


class BMCHelixAPI:
    """Client for interacting with the BMC Helix API."""
    
    def __init__(self, server=BMC_SERVER, username=BMC_USERNAME, password=BMC_PASSWORD):
        """Initialize the BMC Helix API client."""
        self.server = server
        self.username = username
        self.password = password
        self.token = None
        self.headers = None
        self.last_login_time = None
    
    def login(self) -> bool:
        """
        Login to BMC Helix and get an authentication token.
        
        Returns:
            bool: True if login was successful, False otherwise
        """
        logger.info(f"Logging into BMC Helix at {self.server}")
        url = f"https://{self.server}/api/jwt/login"
        payload = {'username': self.username, 'password': self.password}
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        
        try:
            r = requests.post(url, data=payload, headers=headers, verify=False)
            
            if r.status_code == 200:
                self.token = r.text
                self.headers = {'Authorization': f'AR-JWT {self.token}', 'Content-Type': 'application/json'}
                self.last_login_time = datetime.now()
                logger.info("Login successful")
                return True
            else:
                logger.error(f"Login failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return False
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            return False
    
    def logout(self) -> bool:
        """
        Logout from BMC Helix.
        
        Returns:
            bool: True if logout was successful, False otherwise
        """
        if not self.token:
            logger.warning("Not logged in, cannot logout")
            return True
            
        url = f"https://{self.server}/api/jwt/logout"
        
        try:
            r = requests.post(url, headers=self.headers, verify=False)
            
            if r.status_code == 204:
                logger.info("Logout successful")
                self.token = None
                self.headers = None
                self.last_login_time = None
                return True
            else:
                logger.error(f"Logout failed with status code: {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error during logout: {str(e)}")
            return False
    
    def _check_and_refresh_token(self):
        """Check if token is expired and refresh if needed."""
        if not self.token or not self.last_login_time:
            return self.login()
            
        # Token expires after 1 hour, refresh after 50 minutes
        time_since_login = (datetime.now() - self.last_login_time).total_seconds() / 60
        if time_since_login > 50:  # 50 minutes
            logger.info("Token may be expired, refreshing...")
            return self.login()
            
        return True
    
    def get_incidents(self, query_params=None, fields=None, limit=100) -> List[Dict]:
        """
        Get incidents from BMC Helix.
        
        Args:
            query_params: Additional query parameters
            fields: Specific fields to retrieve
            limit: Maximum number of records to retrieve
            
        Returns:
            List[Dict]: List of incidents
        """
        self._check_and_refresh_token()
        
        if not self.token:
            logger.warning("Not logged in, attempting to login")
            if not self.login():
                return []
        
        # Default fields if not specified
        if not fields:
            fields = "Status,Summary,Support Group Name,Request Assignee,Submitter,Work Order ID,Request Manager,Incident Number,Description,Status,Owner,Impact,Owner Group,Submit Date,Assigned Group,Priority,Environment"
        
        url = f"https://{self.server}/api/arsys/v1/entry/HPD:Help%20Desk"
        
        params = {
            'fields': fields,
            'limit': limit
        }
        
        # Add any additional query parameters
        if query_params:
            if isinstance(query_params, dict):
                for key, value in query_params.items():
                    if key not in params:  # Don't overwrite existing params
                        params[key] = value
            elif isinstance(query_params, str):
                # If query_params is a string, assume it's a query string
                params['q'] = query_params
        
        try:
            logger.info(f"Fetching incidents with params: {params}")
            r = requests.get(url, headers=self.headers, params=params, verify=False)
            
            if r.status_code == 200:
                data = r.json()
                entries = data.get('entries', [])
                logger.info(f"Retrieved {len(entries)} incidents")
                
                # Normalize the data structure
                incidents = []
                for entry in entries:
                    incident = {}
                    for key, value in entry.get('values', {}).items():
                        incident[key] = value
                    incidents.append(incident)
                return incidents
            elif r.status_code == 401:
                logger.warning("Unauthorized access. Attempting to re-login.")
                if self.login():
                    # Try again after refreshing token
                    return self.get_incidents(query_params, fields, limit)
                return []
            else:
                logger.error(f"Failed to get incidents with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting incidents: {str(e)}")
            return []
    
    def get_incident_by_id(self, incident_id: str) -> Optional[Dict]:
        """
        Get a specific incident by ID.
        
        Args:
            incident_id (str): The incident ID to fetch
            
        Returns:
            Optional[Dict]: The incident data if found, None otherwise
        """
        # Remove quotes and normalize incident_id
        incident_id = str(incident_id).strip().replace('"', '').replace("'", "")
        
        query_params = {
            'q': f"'Incident Number'=\"{incident_id}\""
        }
        
        incidents = self.get_incidents(query_params)
        
        if incidents and len(incidents) > 0:
            return incidents[0]
        else:
            logger.warning(f"No incident found with ID: {incident_id}")
            return None
    
    def get_incidents_by_date_range(self, start_date: datetime, end_date: datetime, limit=200) -> List[Dict]:
        """
        Get incidents within a date range.
        
        Args:
            start_date (datetime): Start date for incident search
            end_date (datetime): End date for incident search
            limit (int): Maximum number of records to retrieve
            
        Returns:
            List[Dict]: List of incidents in the date range
        """
        # Format dates for BMC Helix query
        start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
        
        query_params = {
            'q': f"'Submit Date' >= \"{start_date_str}\" AND 'Submit Date' <= \"{end_date_str}\""
        }
        
        return self.get_incidents(query_params, limit=limit)


class DataCache:
    """Manages caching of incident data."""
    
    def __init__(self, cache_dir=CACHE_DIR):
        """
        Initialize the data cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def save_incidents(self, incidents: List[Dict], cache_file=INCIDENT_CACHE_FILE):
        """
        Save incidents to cache.
        
        Args:
            incidents: List of incidents to cache
            cache_file: File to save cache to
        """
        cache_data = {
            'timestamp': datetime.now(),
            'incidents': incidents
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Cached {len(incidents)} incidents to {cache_file}")
    
    def load_incidents(self, cache_file=INCIDENT_CACHE_FILE, max_age_hours=CACHE_EXPIRY_HOURS) -> Optional[List[Dict]]:
        """
        Load incidents from cache if available and not expired.
        
        Args:
            cache_file: Cache file to load from
            max_age_hours: Maximum age of cache in hours
            
        Returns:
            Optional[List[Dict]]: List of incidents or None if cache is invalid
        """
        if not os.path.exists(cache_file):
            logger.info(f"Cache file {cache_file} does not exist")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            timestamp = cache_data.get('timestamp')
            incidents = cache_data.get('incidents')
            
            if not timestamp or not incidents:
                logger.warning("Invalid cache data")
                return None
            
            age = (datetime.now() - timestamp).total_seconds() / 3600  # Age in hours
            
            if age > max_age_hours:
                logger.info(f"Cache is expired ({age:.1f} hours old)")
                return None
            
            logger.info(f"Loaded {len(incidents)} incidents from cache ({age:.1f} hours old)")
            return incidents
        
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return None


class IncidentDataService:
    """Service to retrieve and manage incident data."""
    
    def __init__(self, helix_api=None, data_cache=None):
        """
        Initialize the incident data service.
        
        Args:
            helix_api: BMCHelixAPI instance
            data_cache: DataCache instance
        """
        self.helix_api = helix_api or BMCHelixAPI()
        self.data_cache = data_cache or DataCache()
        self.incidents = []
    
    def refresh_incident_data(self, days=30, force=False) -> bool:
        """
        Refresh incident data from BMC Helix or cache.
        
        Args:
            days: Number of days to look back
            force: Force refresh from API even if cache is valid
            
        Returns:
            bool: True if data was refreshed successfully
        """
        if not force:
            # Try to load from cache first
            cached_incidents = self.data_cache.load_incidents()
            if cached_incidents:
                self.incidents = cached_incidents
                return True
        
        # Cache is invalid or force refresh, get from API
        logger.info(f"Refreshing incident data for the last {days} days")
        
        # Get incidents from the last N days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        incidents = self.helix_api.get_incidents_by_date_range(start_date, end_date, limit=1000)
        
        if not incidents:
            logger.error("Failed to refresh incident data")
            return False
        
        self.incidents = incidents
        self.data_cache.save_incidents(incidents)
        return True
    
    def get_incidents(self, filters=None) -> List[Dict]:
        """
        Get incidents with optional filtering.
        
        Args:
            filters: Dictionary of field-value pairs to filter by
            
        Returns:
            List[Dict]: Filtered list of incidents
        """
        if not self.incidents:
            self.refresh_incident_data()
        
        if not filters:
            return self.incidents
        
        # Apply filters
        filtered_incidents = self.incidents.copy()
        
        for field, value in filters.items():
            if not value:  # Skip empty filters
                continue
                
            if isinstance(value, str):
                # Case-insensitive string matching
                filtered_incidents = [
                    inc for inc in filtered_incidents 
                    if field in inc and inc[field] and value.lower() in str(inc[field]).lower()
                ]
            else:
                # Exact matching for non-strings
                filtered_incidents = [
                    inc for inc in filtered_incidents 
                    if field in inc and inc[field] == value
                ]
        
        return filtered_incidents
    
    def get_incidents_by_date_range(self, start_date, end_date) -> List[Dict]:
        """
        Get incidents within a date range.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            List[Dict]: Filtered list of incidents
        """
        if not self.incidents:
            self.refresh_incident_data()
        
        filtered_incidents = []
        
        for incident in self.incidents:
            if 'Submit Date' not in incident:
                continue
            
            try:
                submit_date = datetime.strptime(incident['Submit Date'], "%Y-%m-%d %H:%M:%S")
                if start_date <= submit_date <= end_date:
                    filtered_incidents.append(incident)
            except (ValueError, TypeError):
                continue
        
        return filtered_incidents
    
    def get_incident_by_id(self, incident_id) -> Optional[Dict]:
        """
        Get a specific incident by ID.
        
        Args:
            incident_id: ID of the incident to retrieve
            
        Returns:
            Optional[Dict]: Incident data or None if not found
        """
        # Normalize incident_id
        incident_id = str(incident_id).strip().upper()
        
        # First check in cached incidents
        if self.incidents:
            for incident in self.incidents:
                if 'Incident Number' in incident and str(incident['Incident Number']).strip().upper() == incident_id:
                    return incident
        
        # If not found in cache, try to get directly from API
        return self.helix_api.get_incident_by_id(incident_id)
    
    def get_incidents_statistics(self) -> Dict:
        """
        Get statistics about the incidents.
        
        Returns:
            Dict: Statistics about the incidents
        """
        if not self.incidents:
            self.refresh_incident_data()
        
        if not self.incidents:
            return {}
        
        # Convert to pandas DataFrame for easier analysis
        df = pd.DataFrame(self.incidents)
        
        stats = {
            'total_count': len(df),
            'status_counts': {},
            'priority_counts': {},
            'support_group_counts': {}
        }
        
        # Status counts
        if 'Status' in df.columns:
            status_counts = df['Status'].value_counts().to_dict()
            stats['status_counts'] = status_counts
        
        # Priority counts
        if 'Priority' in df.columns:
            priority_counts = df['Priority'].value_counts().to_dict()
            stats['priority_counts'] = priority_counts
        
        # Support group counts
        if 'Support Group Name' in df.columns:
            support_group_counts = df['Support Group Name'].value_counts().to_dict()
            stats['support_group_counts'] = {k: v for k, v in support_group_counts.items() if k and pd.notna(k)}
        
        return stats


class GeminiService:
    """Class to handle Gemini AI integration using your working configuration."""
    
    def __init__(self, project_id=PROJECT_ID, region=REGION, model_name=MODEL_NAME):
        """Initialize the Gemini service."""
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        self.model = None
        self.initialize()
    
    def initialize(self):
        """Initialize the Gemini model."""
        try:
            # Initialize Vertex AI with project and region
            vertexai.init(project=self.project_id, location=self.region)
            
            # Create model instance
            self.model = GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            self.model = None
    
    def generate_response(self, prompt: str, system_instructions: Optional[str] = None) -> str:
        """
        Generate a response from Gemini AI.
        
        Args:
            prompt: The prompt to send to the model
            system_instructions: Optional system instructions to guide the model
            
        Returns:
            str: The generated response
        """
        if not self.model:
            return "Error: Gemini AI model not initialized."
        
        # Combine system instructions and prompt if provided
        full_content = []
        if system_instructions:
            full_content.append(system_instructions)
        full_content.append(prompt)
        
        logger.info(f"Generating response for prompt: {prompt[:100]}...")
        
        try:
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=8192,
            )
            
            # Generate response
            response = self.model.generate_content(
                full_content,
                generation_config=generation_config,
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].text
            else:
                # Try different response formats
                return str(response)
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
    
    def generate_structured_output(self, prompt: str, response_format: Dict[str, Any]) -> Dict:
        """
        Generate a structured output (like JSON) from Gemini AI.
        
        Args:
            prompt: The prompt to send to the model
            response_format: The expected response format
            
        Returns:
            Dict: The structured output
        """
        # Add instructions to return structured data
        structured_prompt = f"""
        {prompt}
        
        Respond with ONLY a valid JSON object that matches this structure:
        {json.dumps(response_format, indent=2)}
        
        Do not include any additional text, just return the valid JSON.
        """
        
        response_text = self.generate_response(structured_prompt)
        
        try:
            # Try to extract JSON from the response
            # First try direct parsing
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON content using regex
                match = re.search(r'({.*})', response_text, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
                # If extraction fails, return default structure
                return response_format
        except Exception as e:
            logger.error(f"Error parsing structured output: {str(e)}")
            return response_format


class HelixGeminiChatbot:
    """Main class for the BMC Helix + Gemini Chatbot."""
    
    def __init__(self):
        """Initialize the chatbot."""
        self.data_service = IncidentDataService()
        self.gemini_service = GeminiService()
        
        # Load incident data
        self.refresh_data()
    
    def refresh_data(self, days=30):
        """Refresh incident data from BMC Helix."""
        logger.info(f"Refreshing incident data for the last {days} days")
        return self.data_service.refresh_incident_data(days)
    
    def interpret_query(self, query: str) -> Dict:
        """
        Interpret a natural language query to determine intent and parameters.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dict: Interpretation results with query parameters
        """
        # Default interpretation format
        default_interpretation = {
            "query_type": "list",
            "time_period": "last_week",
            "filters": {
                "status": None,
                "priority": None,
                "support_group": None,
                "assignee": None
            },
            "specific_incident_id": None,
            "search_terms": None
        }
        
        interpretation_prompt = f"""
        Analyze this query about BMC Remedy incidents to determine what information is being requested:
        
        Query: "{query}"
        
        Respond with only a JSON object using this format:
        {{
            "query_type": "list" | "details" | "statistics" | "trends" | "search",
            "time_period": "today" | "yesterday" | "this_week" | "last_week" | "last_month" | "custom",
            "custom_days": null,
            "filters": {{
                "status": null,
                "priority": null,
                "support_group": null,
                "assignee": null
            }},
            "specific_incident_id": null,
            "search_terms": null
        }}
        
        Notes:
        - If the query is about a specific incident by ID, set query_type to "details" and fill in specific_incident_id
        - If the query is looking for incidents by some text, set query_type to "search" and fill in search_terms
        - For statistics or trends, set the appropriate query_type
        - Only include filters that are specifically mentioned in the query
        
        Return ONLY valid JSON with no other text.
        """
        
        try:
            # Generate structured output
            interpretation = self.gemini_service.generate_structured_output(
                interpretation_prompt, 
                default_interpretation
            )
            
            logger.info(f"Query interpretation: {interpretation}")
            return interpretation
        except Exception as e:
            logger.error(f"Error interpreting query: {str(e)}")
            return default_interpretation
    
    def get_incidents_for_query(self, interpretation: Dict) -> List[Dict]:
        """
        Get incidents that match the query interpretation.
        
        Args:
            interpretation: Query interpretation results
            
        Returns:
            List[Dict]: Matching incidents
        """
        # Check for specific incident ID
        if interpretation.get("specific_incident_id"):
            incident = self.data_service.get_incident_by_id(interpretation["specific_incident_id"])
            return [incident] if incident else []
        
        # Handle time period
        time_period = interpretation.get("time_period", "last_week")
        end_date = datetime.now()
        start_date = None
        
        if time_period == "today":
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_period == "yesterday":
            yesterday = datetime.now() - timedelta(days=1)
            start_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif time_period == "this_week":
            # Start from Monday of current week
            today = datetime.now()
            start_date = (today - timedelta(days=today.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_period == "last_week":
            # Last week Monday to Sunday
            today = datetime.now()
            start_date = (today - timedelta(days=today.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = (today - timedelta(days=today.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_period == "last_month":
            # Last 30 days
            start_date = end_date - timedelta(days=30)
        elif time_period == "custom" and interpretation.get("custom_days"):
            # Custom number of days
            days = interpretation.get("custom_days", 7)
            start_date = end_date - timedelta(days=days)
        else:
            # Default to last 7 days
            start_date = end_date - timedelta(days=7)
        
        # Get incidents in the date range
        incidents = self.data_service.get_incidents_by_date_range(start_date, end_date)
        
        # Apply filters
        filters = interpretation.get("filters", {})
        filtered_incidents = incidents
        
        # Apply status filter
        if filters.get("status"):
            filtered_incidents = [inc for inc in filtered_incidents 
                                if inc.get("Status", "").lower() == filters["status"].lower()]
        
        # Apply priority filter
        if filters.get("priority"):
            filtered_incidents = [inc for inc in filtered_incidents 
                                if inc.get("Priority", "").lower() == filters["priority"].lower()]
        
        # Apply support group filter
        if filters.get("support_group"):
            filtered_incidents = [inc for inc in filtered_incidents 
                                if filters["support_group"].lower() in str(inc.get("Support Group Name", "")).lower()]
        
        # Apply assignee filter
        if filters.get("assignee"):
            filtered_incidents = [inc for inc in filtered_incidents 
                                if filters["assignee"].lower() in str(inc.get("Request Assignee", "")).lower()]
        
        # Search by terms
        if interpretation.get("search_terms"):
            search_terms = interpretation["search_terms"].lower()
            filtered_incidents = [inc for inc in filtered_incidents 
                                if search_terms in str(inc.get("Summary", "")).lower() or 
                                   search_terms in str(inc.get("Description", "")).lower()]
        
        return filtered_incidents
    
    def analyze_incidents(self, incidents: List[Dict], query: str) -> str:
        """
        Analyze incidents using Gemini AI.
        
        Args:
            incidents: List of incidents to analyze
            query: User query about the incidents
            
        Returns:
            str: Gemini's analysis response
        """
        if not incidents:
            return "No incident data available for analysis."
        
        # Create a system prompt to guide Gemini's analysis
        system_prompt = """
        You are an expert IT service management analyst specializing in BMC Helix incidents.
        
        When analyzing BMC Helix incident data:
        1. Consider all relevant fields including Status, Priority, Support Group, Assignee
        2. Structure your response with clear sections: Summary, Analysis, and Recommendations
        3. Use tables and formatted text to present information clearly
        4. Highlight notable patterns or outliers in the data
        5. Suggest potential actions based on your analysis when appropriate
        
        Present your findings in a professional format suitable for IT managers and support teams.
        Include relevant statistics and categorize information logically based on the query.
        
        Focus on delivering actionable insights that help teams improve their service delivery.
        """
        
        # Convert incidents to a simplified JSON string to reduce token count
        simplified_incidents = []
        for incident in incidents[:100]:  # Limit to 100 incidents to avoid token limits
            simplified = {
                'Incident Number': incident.get('Incident Number', ''),
                'Status': incident.get('Status', ''),
                'Priority': incident.get('Priority', ''),
                'Summary': incident.get('Summary', ''),
                'Support Group Name': incident.get('Support Group Name', ''),
                'Request Assignee': incident.get('Request Assignee', ''),
                'Submit Date': incident.get('Submit Date', '')
            }
            # Add other fields if they exist and are not empty
            for key, value in incident.items():
                if key not in simplified and value and str(value).strip():
                    simplified[key] = value
            
            simplified_incidents.append(simplified)
        
        incidents_json = json.dumps(simplified_incidents, indent=2)
        
        # Create the prompt with the user query and incident data
        user_prompt = f"""
        User query: "{query}"
        
        Incident data from BMC Helix:
        ```json
        {incidents_json}
        ```
        
        Please analyze these incidents and provide a comprehensive response that directly addresses the user's query.
        """
        
        # Generate response from Gemini
        return self.gemini_service.generate_response(user_prompt, system_prompt)
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and return a response.
        
        Args:
            query: User's natural language query
            
        Returns:
            str: Response to the query
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Interpret the query
            interpretation = self.interpret_query(query)
            
            # Get relevant incidents
            incidents = self.get_incidents_for_query(interpretation)
            
            if not incidents:
                logger.warning("No incidents found for query")
                return "I couldn't find any incidents matching your query. Try a different query or time period."
            
            # Analyze incidents with Gemini
            response = self.analyze_incidents(incidents, query)
            
            return response
        
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            logger.error(error_message)
            return f"I encountered an error while processing your query: {str(e)}"
    
    def close(self):
        """Clean up resources."""
        # Cleanup and logout if needed
        pass


def interactive_mode():
    """Run the chatbot in interactive mode."""
    print("="*70)
    print("BMC Helix + Gemini Chatbot")
    print("="*70)
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("Type 'refresh' to refresh incident data from BMC Helix.")
    print("\nExample queries:")
    print("- What incidents were created yesterday?")
    print("- Show me all high priority incidents from last week")
    print("- Categorize incidents from the last month by status and priority")
    print("- What are the current trends in incident volume?")
    print("- Give me details about incident INC123456")
    print("-"*70)
    
    chatbot = HelixGeminiChatbot()
    
    try:
        while True:
            query = input("\nYour query: ").strip()
            
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            if query.lower() == "refresh":
                if chatbot.refresh_data():
                    print("Incident data refreshed successfully.")
                else:
                    print("Failed to refresh incident data.")
                continue
            
            if not query:
                continue
            
            start_time = time.time()
            print("\nProcessing your query... (this may take a moment)")
            
            response = chatbot.process_query(query)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print("\n" + "="*70)
            print(response)
            print("="*70)
            print(f"\nResponse generated in {processing_time:.2f} seconds.")
    
    except KeyboardInterrupt:
        print("\nSession terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        print("\nClosing session...")
        chatbot.close()
        print("Session closed.")


if __name__ == "__main__":
    interactive_mode()



























#!/usr/bin/env python3
import logging
import os
import sys
import re
import json
import requests
from datetime import datetime
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.api_core.exceptions import GoogleAPICallError
import ssl
import urllib3

# Disable SSL verification globally as requested
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jira_gemini_assistant.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("JiraGeminiAssistant")

# Configuration (Environment Variables or Config File)
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "https://your-jira-instance.com")
JIRA_USERNAME = os.environ.get("JIRA_USERNAME", "")
JIRA_TOKEN = os.environ.get("JIRA_TOKEN", "")

class JiraClient:
    """Class for interacting with Jira API."""
    
    def __init__(self, base_url, username, token):
        """Initialize the Jira client."""
        self.base_url = base_url
        self.auth = (username, token)
        self.headers = {"Content-Type": "application/json"}
        self.verify = False  # Disable SSL verification
    
    def test_connection(self):
        """Test connection to Jira."""
        try:
            logger.info("Testing connection to Jira...")
            response = requests.get(
                f"{self.base_url}/rest/api/2/serverInfo",
                auth=self.auth,
                headers=self.headers,
                verify=self.verify
            )
            response.raise_for_status()
            server_info = response.json()
            logger.info(f"Connection to Jira successful! Server version: {server_info.get('version', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Jira: {str(e)}")
            error_details = {
                "error": str(e),
                "type": str(type(e)),
                "url": f"{self.base_url}/rest/api/2/serverInfo"
            }
            logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
            return False
    
    def get_issue(self, issue_key):
        """Get a specific issue by its key."""
        try:
            logger.info(f"Fetching issue: {issue_key}")
            params = {
                "expand": "renderedFields,names,schema,transitions,operations,editmeta,changelog,attachment"
            }
            response = requests.get(
                f"{self.base_url}/rest/api/2/issue/{issue_key}",
                params=params,
                auth=self.auth,
                headers=self.headers,
                verify=self.verify
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get issue {issue_key}: {str(e)}")
            return None
    
    def search_issues(self, jql, start_at=0, max_results=50, fields=None, expand=None):
        """Search for issues using JQL (Jira Query Language)."""
        try:
            logger.info(f"Searching issues with JQL: {jql}")
            params = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": max_results
            }
            
            if fields:
                params["fields"] = ",".join(fields) if isinstance(fields, list) else fields
                
            if expand:
                params["expand"] = expand
                
            response = requests.get(
                f"{self.base_url}/rest/api/2/search",
                params=params,
                auth=self.auth,
                headers=self.headers,
                verify=self.verify
            )
            response.raise_for_status()
            search_results = response.json()
            total = search_results.get("total", 0)
            logger.info(f"Search returned {total} issues")
            return search_results
        except Exception as e:
            logger.error(f"Failed to search issues: {str(e)}")
            return None
    
    def get_all_issues(self, jql, fields=None, max_results=1000):
        """Get all issues matching a JQL query, handling pagination."""
        logger.info(f"Retrieving all issues matching JQL: {jql}")
        all_issues = []
        start_at = 0
        page_size = 100  # Jira recommends 100 for optimal performance
        
        while True:
            search_results = self.search_issues(
                jql=jql,
                start_at=start_at,
                max_results=page_size,
                fields=fields
            )
            
            if not search_results or not search_results.get("issues"):
                break
                
            issues = search_results.get("issues", [])
            all_issues.extend(issues)
            
            # Check if we've reached the total or our max limit
            total = search_results.get("total", 0)
            if len(all_issues) >= total or len(all_issues) >= max_results:
                break
                
            # Move to next page
            start_at += len(issues)
            
            # If no issues were returned, we're done
            if len(issues) == 0:
                break
        
        logger.info(f"Retrieved a total of {len(all_issues)} issues")
        return all_issues
    
    def get_issue_content(self, issue_key):
        """Get the content of an issue in a format suitable for the assistant."""
        issue = self.get_issue(issue_key)
        if not issue:
            return None
        
        # Extract key metadata
        metadata = {
            "key": issue.get("key"),
            "summary": issue["fields"].get("summary"),
            "type": issue["fields"].get("issuetype", {}).get("name"),
            "status": issue["fields"].get("status", {}).get("name"),
            "created": issue["fields"].get("created"),
            "updated": issue["fields"].get("updated"),
            "priority": issue["fields"].get("priority", {}).get("name") if issue["fields"].get("priority") else None,
            "labels": issue["fields"].get("labels", []),
            "resolution": issue["fields"].get("resolution", {}).get("name") if issue["fields"].get("resolution") else None,
            "url": f"{self.base_url}/browse/{issue.get('key')}"
        }
        
        # Extract people
        if issue["fields"].get("assignee"):
            metadata["assignee"] = issue["fields"].get("assignee", {}).get("displayName")
            
        if issue["fields"].get("reporter"):
            metadata["reporter"] = issue["fields"].get("reporter", {}).get("displayName")
        
        # Extract content fields
        content_parts = []
        
        # Add summary
        summary = issue["fields"].get("summary", "")
        if summary:
            content_parts.append(f"Summary: {summary}")
        
        # Add description - try to use rendered HTML if available
        if "renderedFields" in issue and issue["renderedFields"].get("description"):
            description_html = issue["renderedFields"].get("description")
            # Basic HTML tag stripping
            description_text = re.sub(r'<[^>]+>', ' ', description_html)
            description_text = re.sub(r'\s+', ' ', description_text).strip()
            content_parts.append(f"Description: {description_text}")
        elif issue["fields"].get("description"):
            description = issue["fields"].get("description")
            if isinstance(description, dict):
                # Handle Atlassian Document Format
                desc_text = self._extract_text_from_adf(description)
                content_parts.append(f"Description: {desc_text}")
            else:
                content_parts.append(f"Description: {description}")
        
        # Add attachments info
        if issue["fields"].get("attachment"):
            attachments = issue["fields"].get("attachment", [])
            if attachments:
                attachment_info = []
                for attachment in attachments:
                    attachment_info.append(f"{attachment.get('filename')} ({attachment.get('mimeType')})")
                content_parts.append(f"Attachments: {', '.join(attachment_info)}")
        
        # Add comments - try to use rendered content if available
        if "renderedFields" in issue and issue["renderedFields"].get("comment", {}).get("comments"):
            comments = issue["renderedFields"].get("comment", {}).get("comments", [])
            for comment in comments:
                author = comment.get("author", {}).get("displayName", "unknown")
                created = comment.get("created", "")
                
                # Extract text from HTML
                comment_html = comment.get("body", "")
                comment_text = re.sub(r'<[^>]+>', ' ', comment_html)
                comment_text = re.sub(r'\s+', ' ', comment_text).strip()
                
                content_parts.append(f"Comment by {author} on {created}: {comment_text}")
        elif issue["fields"].get("comment", {}).get("comments"):
            comments = issue["fields"].get("comment", {}).get("comments", [])
            for comment in comments:
                author = comment.get("author", {}).get("displayName", "unknown")
                created = comment.get("created", "")
                
                comment_body = comment.get("body")
                if isinstance(comment_body, dict):
                    # Handle Atlassian Document Format
                    comment_text = self._extract_text_from_adf(comment_body)
                    content_parts.append(f"Comment by {author} on {created}: {comment_text}")
                else:
                    content_parts.append(f"Comment by {author} on {created}: {comment_body}")
        
        # Add custom fields that might contain important information
        for field_id, field_value in issue["fields"].items():
            # Skip fields we've already processed and empty values
            if field_id in ["summary", "description", "comment", "attachment", "assignee", "reporter", 
                           "issuetype", "status", "created", "updated", "priority", "labels", "resolution"]:
                continue
                
            if not field_value:
                continue
                
            # Get field name if available
            field_name = field_id
            if "names" in issue and field_id in issue["names"]:
                field_name = issue["names"][field_id]
                
            # Handle different value types
            if isinstance(field_value, dict) and "value" in field_value:
                content_parts.append(f"{field_name}: {field_value['value']}")
            elif isinstance(field_value, dict) and "name" in field_value:
                content_parts.append(f"{field_name}: {field_value['name']}")
            elif isinstance(field_value, list) and all(isinstance(item, dict) for item in field_value):
                values = []
                for item in field_value:
                    if "value" in item:
                        values.append(item["value"])
                    elif "name" in item:
                        values.append(item["name"])
                if values:
                    content_parts.append(f"{field_name}: {', '.join(values)}")
            elif not isinstance(field_value, (dict, list)):
                content_parts.append(f"{field_name}: {field_value}")
        
        # Combine all content
        full_content = "\n\n".join(content_parts)
        
        # Return formatted content with metadata
        return {
            "metadata": metadata,
            "content": full_content
        }
    
    def _extract_text_from_adf(self, adf_doc):
        """Extract plain text from Atlassian Document Format (ADF)."""
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
                    
                # Extract text node
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                
                # Handle links
                if item.get("type") == "link" and "attrs" in item and "href" in item.get("attrs", {}):
                    href = item.get("attrs", {}).get("href", "")
                    link_text = extract_from_content(item.get("content", []))
                    parts.append(f"{link_text} ({href})")
                
                # Handle mentions
                if item.get("type") == "mention" and "attrs" in item and "text" in item.get("attrs", {}):
                    mention_text = item.get("attrs", {}).get("text", "")
                    parts.append(f"@{mention_text}")
                
                # Extract code blocks
                if item.get("type") == "codeBlock":
                    code_text = extract_from_content(item.get("content", []))
                    parts.append(f"Code: {code_text}")
                
                # Extract from content recursively
                if "content" in item and isinstance(item["content"], list):
                    parts.append(extract_from_content(item["content"]))
                    
            return " ".join(parts)
        
        # Extract from main content array
        for item in adf_doc.get("content", []):
            text_parts.append(extract_from_content([item]))
        
        return " ".join(text_parts)
        
    def get_issue_type_metadata(self):
        """Get issue type metadata for better understanding of the Jira instance's structure."""
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/2/issuetype",
                auth=self.auth,
                headers=self.headers,
                verify=self.verify
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get issue type metadata: {str(e)}")
            return None
            
    def get_field_metadata(self):
        """Get field metadata for better understanding of custom fields."""
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/2/field",
                auth=self.auth,
                headers=self.headers,
                verify=self.verify
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get field metadata: {str(e)}")
            return None

class JiraGeminiAssistant:
    """Enhanced assistant using Gemini and Jira integration."""
    
    def __init__(self, jira_client):
        """Initialize the assistant."""
        # Initialize Vertex AI with disabled SSL verification
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
        self.jira_client = jira_client
        self.conversation_history = []
        
        # Load system prompt
        self.system_prompt = self._create_system_prompt()
        
        # Cache for issue metadata to reduce API calls
        self.issue_cache = {}
        
        # Get Jira metadata for better understanding
        self.issue_types = self.jira_client.get_issue_type_metadata()
        self.field_metadata = self.jira_client.get_field_metadata()
        
    def _create_system_prompt(self):
        """Create an enhanced system prompt for the Gemini model."""
        return """
        # JIRA ASSISTANT SYSTEM INSTRUCTIONS

        You are JiraGenius, an advanced assistant designed to help users with their Jira environment. You provide accurate, comprehensive, and helpful responses to all Jira-related questions while maintaining a professional yet friendly tone.

        ## CORE CAPABILITIES
        1. **Jira Expertise**: You can answer questions about Jira tickets, projects, workflows, users, and all Jira-related functionality.
        2. **Data Analysis**: You can analyze and interpret information from Jira tickets, including text, images, tables, and attachments.
        3. **Search Functions**: You can find and summarize tickets based on various criteria (project, status, assignee, etc.).
        4. **Ticket Details**: You can provide comprehensive information about specific tickets.
        5. **Visual Understanding**: You can interpret and describe images, charts, and tables within Jira tickets.
        6. **Advanced Formatting**: You format responses optimally based on the content type (tables, lists, paragraphs).
        7. **Proactive Clarification**: You ask follow-up questions when needed to provide the most accurate response.
        8. **Technical Context**: You understand software development, project management, and IT terminology.
        9. **Citation and Links**: You provide links to relevant Jira tickets and pages in your responses.
        10. **Instant Responses**: You prioritize providing quick, concise answers when appropriate.

        ## RESPONSE GUIDELINES

        ### Format and Structure
        - Use appropriate formatting for different content types (tables for comparisons, bullet points for lists, etc.)
        - Structure complex responses with clear headings and sections
        - Highlight key information visually when appropriate
        - For ticket details, always begin with a summary card showing key information
        - For search results, use tables with columns for Key, Summary, Status, and Assignee

        ### Content Quality
        - Be accurate and factual above all else
        - Be comprehensive but concise - cover all aspects without unnecessary verbosity
        - Provide context and background information when beneficial
        - Link to specific tickets whenever mentioned using the full URL
        - Include relevant metadata (created date, status changes, assignee history) when discussing tickets
        - When describing images or visual elements, be detailed and explain their relevance to the ticket

        ### Tone and Style
        - Maintain a professional but conversational tone
        - Use technical terminology appropriately for the context and user expertise level
        - Be helpful and solution-oriented
        - Show empathy for user challenges
        - Be confident in assertions but acknowledge limitations when appropriate
        - Use a friendly, approachable writing style that builds rapport
        
        ### Follow-up Questions
        - Ask clarifying questions when user queries are ambiguous
        - Frame questions to narrow down exactly what the user needs
        - Limit follow-up questions to one per response
        - Make follow-up questions specific and directly relevant to improving your answer
        - When a query could have multiple interpretations, ask for clarification rather than assuming

        ## RESPONSE STRATEGIES BY QUERY TYPE

        ### For Ticket Search Queries
        - Return results in a table format with columns for Key, Summary, Status, and Assignee
        - Include the total number of matching tickets
        - Provide direct links to each ticket
        - Sort results by priority or recency unless another order is specified
        - Offer pagination information if there are many results
        - Include a brief summary of the search criteria used

        ### For Specific Ticket Details
        - Begin with a "Ticket Card" showing Key, Summary, Status, Priority, Assignee, and Creation Date
        - Include a direct link to the ticket
        - Organize information in logical sections (Description, Comments, History, Attachments, etc.)
        - Highlight important updates or changes
        - Include all relevant metadata (components, labels, etc.)
        - Summarize long descriptions or comments while preserving key details
        - For tickets with many comments, focus on the most recent or most relevant

        ### For Project Overviews
        - Provide key statistics (open tickets, recently completed, upcoming)
        - Summarize the project's current status and key milestones
        - Highlight any blockers or critical issues
        - List key contributors and their roles
        - Include links to important project resources
        - Mention recent activity and upcoming deadlines

        ### For Technical Questions
        - Break down complex issues step by step
        - Differentiate between verified solutions and theoretical approaches
        - Include practical examples where relevant
        - Reference documentation when appropriate
        - Provide context for technical terminology
        - Include both immediate fixes and long-term solutions when applicable

        ### For Irrelevant Questions
        - If the question is completely unrelated to Jira but is a general knowledge question you can answer, provide a brief, helpful response
        - If the question is inappropriate or outside your capabilities, politely redirect to Jira-related topics
        - Always maintain a professional tone even when declining to answer
        - Suggest related Jira topics that might be more helpful

        ## SPECIAL HANDLING

        ### Images and Attachments
        - When discussing images in tickets, describe what you can see and how it relates to the ticket
        - Identify charts, diagrams, screenshots, and explain their content
        - For technical screenshots, identify the application or system shown
        - For error messages in images, transcribe them when possible

        ### Data Tables
        - Maintain table structure in your response
        - Summarize key patterns or insights from the table
        - Highlight anomalies or important data points
        - For large tables, focus on the most relevant sections

        ### User References
        - When mentioning users, include their role and responsibilities if known
        - Respect privacy by not revealing sensitive personal information
        - Focus on work activities and contributions rather than personal attributes
        - Use formal names and titles when appropriate

        ### Workflows and Processes
        - Explain the current stage in the workflow and next steps
        - Identify bottlenecks or blockers in the process
        - Suggest workflow improvements when appropriate
        - Reference established processes and best practices

        Remember, your primary goal is to provide immediate, accurate, and helpful information about the user's Jira environment in a professional, friendly manner. Always include relevant links, format your response appropriately for the content, and be proactive in asking clarifying questions when needed.
        """
    
    def generate_response(self, user_query):
        """Generate a response to a user query."""
        logger.info(f"Generating response for: {user_query}")
        
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": user_query})
        
        # Check if query is about a specific Jira ticket
        ticket_pattern = r'\b[A-Z]+-\d+\b'  # e.g., PROJ-123
        ticket_matches = re.findall(ticket_pattern, user_query)
        
        # Enhance the prompt with relevant Jira data if applicable
        enhanced_prompt = self._enhance_prompt_with_jira_data(user_query, ticket_matches)
        
        try:
            # Configure generation parameters for faster, high-quality responses
            generation_config = GenerationConfig(
                temperature=0.2,  # Lower temperature for more factual responses
                top_p=0.95,
                max_output_tokens=8192,
            )
            
            # Generate response - streaming for speed
            logger.info("Generating response...")
            response_text = ""
            
            full_prompt = f"""
            {self.system_prompt}
            
            # CONVERSATION HISTORY
            {self._format_conversation_history()}
            
            # JIRA CONTEXT INFORMATION
            {enhanced_prompt}
            
            # USER QUERY
            {user_query}
            
            # YOUR RESPONSE
            """
            
            for chunk in self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True,
            ):
                if chunk.candidates and chunk.candidates[0].text:
                    response_text += chunk.candidates[0].text
                    
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Cap conversation history to prevent context overflow
            if len(self.conversation_history) > 10:  # Keep last 5 exchanges (10 messages)
                self.conversation_history = self.conversation_history[-10:]
                
            logger.info(f"Response length: {len(response_text)} characters")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I'm sorry, I encountered an error while generating a response. Please try again or contact support if the issue persists."
    
    def _format_conversation_history(self):
        """Format conversation history for the prompt."""
        formatted_history = ""
        for message in self.conversation_history[-8:]:  # Use last 8 messages max
            role = message["role"].capitalize()
            content = message["content"]
            formatted_history += f"{role}: {content}\n\n"
        return formatted_history
    
    def _enhance_prompt_with_jira_data(self, user_query, ticket_matches):
        """Enhance the prompt with relevant Jira data."""
        enhanced_data = []
        
        # If specific tickets are mentioned, fetch their details
        if ticket_matches:
            for ticket_id in ticket_matches:
                # Check cache first
                if ticket_id in self.issue_cache:
                    ticket_content = self.issue_cache[ticket_id]
                else:
                    ticket_content = self.jira_client.get_issue_content(ticket_id)
                    if ticket_content:
                        self.issue_cache[ticket_id] = ticket_content
                
                if ticket_content:
                    enhanced_data.append(f"## JIRA TICKET: {ticket_id}")
                    enhanced_data.append(f"URL: {ticket_content['metadata']['url']}")
                    enhanced_data.append(f"Summary: {ticket_content['metadata']['summary']}")
                    enhanced_data.append(f"Type: {ticket_content['metadata']['type']}")
                    enhanced_data.append(f"Status: {ticket_content['metadata']['status']}")
                    enhanced_data.append(f"Created: {ticket_content['metadata']['created']}")
                    enhanced_data.append(f"Updated: {ticket_content['metadata']['updated']}")
                    
                    if 'assignee' in ticket_content['metadata']:
                        enhanced_data.append(f"Assignee: {ticket_content['metadata']['assignee']}")
                    
                    if 'reporter' in ticket_content['metadata']:
                        enhanced_data.append(f"Reporter: {ticket_content['metadata']['reporter']}")
                    
                    if 'priority' in ticket_content['metadata'] and ticket_content['metadata']['priority']:
                        enhanced_data.append(f"Priority: {ticket_content['metadata']['priority']}")
                    
                    if 'labels' in ticket_content['metadata'] and ticket_content['metadata']['labels']:
                        enhanced_data.append(f"Labels: {', '.join(ticket_content['metadata']['labels'])}")
                    
                    enhanced_data.append("Content:")
                    enhanced_data.append(ticket_content['content'])
        
        # If the query seems to be a search rather than about specific tickets
        elif any(term in user_query.lower() for term in ['search', 'find', 'look for', 'show me', 'list', 'get', 'what are', 'tickets']):
            # Try to extract potential JQL terms from the query
            potential_jql = self._extract_jql_from_query(user_query)
            if potential_jql:
                try:
                    issues = self.jira_client.search_issues(potential_jql, max_results=15)
                    if issues and issues.get('issues'):
                        enhanced_data.append(f"## SEARCH RESULTS")
                        enhanced_data.append(f"JQL Query: {potential_jql}")
                        enhanced_data.append(f"Total matches: {issues.get('total', 0)}")
                        
                        for i, issue in enumerate(issues.get('issues', [])[:15]):
                            issue_key = issue.get('key')
                            summary = issue.get('fields', {}).get('summary', 'No summary')
                            status = issue.get('fields', {}).get('status', {}).get('name', 'Unknown')
                            assignee = issue.get('fields', {}).get('assignee', {}).get('displayName', 'Unassigned')
                            
                            enhanced_data.append(f"{i+1}. {issue_key}: {summary}")
                            enhanced_data.append(f"   Status: {status} | Assignee: {assignee}")
                            enhanced_data.append(f"   Link: {self.jira_client.base_url}/browse/{issue_key}")
                except Exception as e:
                    logger.error(f"Error executing JQL search: {str(e)}")
                    enhanced_data.append(f"Failed to execute search with JQL: {potential_jql}")
                    enhanced_data.append(f"Error: {str(e)}")
        
        # Return the enhanced prompt
        if enhanced_data:
            return "\n".join(enhanced_data)
        else:
            return "No specific Jira ticket information is available for this query."
    
    def _extract_jql_from_query(self, user_query):
        """
        Extract JQL from a natural language query.
        This implementation parses common search patterns from the query.
        """
        query = user_query.lower()
        jql_parts = []
        
        # Check for project reference
        project_match = re.search(r'(?:in |for |project |projects )(?:called |named |)["\'()]?([a-zA-Z0-9]+)["\'()]?', query)
        if project_match:
            jql_parts.append(f"project = {project_match.group(1).upper()}")
        
        # Check for status
        status_terms = {
            'open': ['open', 'active', 'in progress', 'ongoing', 'not closed', 'not done', 'not resolved'],
            'closed': ['closed', 'resolved', 'done', 'completed', 'finished'],
            'blocked': ['blocked', 'impediment', 'stuck'],
            'in progress': ['in progress', 'working', 'active', 'being worked on'],
            'new': ['new', 'to do', 'backlog', 'not started']
        }
        
        for status, terms in status_terms.items():
            if any(term in query for term in terms):
                if status == 'open':
                    jql_parts.append('status not in (Closed, Resolved, Done)')
                elif status == 'closed':
                    jql_parts.append('status in (Closed, Resolved, Done)')
                else:
                    jql_parts.append(f'status = "{status}"')
                break
        
        # Check for assignee
        assigned_match = re.search(r'(?:assigned to|assignee is|owned by) ["\'()]?([^"\'()]+)["\'()]?', query)
        if assigned_match:
            assignee = assigned_match.group(1).strip()
            if assignee in ['me', 'myself', 'i']:
                jql_parts.append('assignee = currentUser()')
            elif assignee in ['no one', 'nobody', 'not assigned', 'unassigned']:
                jql_parts.append('assignee is EMPTY')
            else:
                jql_parts.append(f'assignee ~ "{assignee}"')
        
        # Check for issue type
        type_terms = ['bug', 'task', 'story', 'epic', 'feature', 'improvement', 'enhancement']
        for term in type_terms:
            if term in query and any(x in query for x in [f"type {term}", f"{term}s", f"{term} tickets"]):
                jql_parts.append(f'issuetype = "{term}"')
                break
        
        # Check for priority
        priority_terms = ['blocker', 'critical', 'major', 'minor', 'trivial', 'high', 'medium', 'low']
        for term in priority_terms:
            if term in query and any(x in query for x in [f"priority {term}", f"{term} priority"]):
                jql_parts.append(f'priority = "{term}"')
                break
        
        # Check for text search
        text_match = re.search(r'(?:containing|with text|about|mentions|related to) ["\'()]?([^"\'()]+)["\'()]?', query)
        if text_match:
            search_text = text_match.group(1).strip()
            jql_parts.append(f'text ~ "{search_text}"')
        
        # Check for recent tickets
        time_terms = {
            'today': 'created >= startOfDay()',
            'yesterday': 'created >= startOfDay(-1d) AND created < startOfDay()',
            'this week': 'created >= startOfWeek()',
            'last week': 'created >= startOfWeek(-1w) AND created < startOfWeek()',
            'this month': 'created >= startOfMonth()',
            'last month': 'created >= startOfMonth(-1M) AND created < startOfMonth()'
        }
        
        for term, jql in time_terms.items():
            if term in query:
                jql_parts.append(jql)
                break
                
        # Check for reporter
        reporter_match = re.search(r'(?:reported by|created by|raised by) ["\'()]?([^"\'()]+)["\'()]?', query)
        if reporter_match:
            reporter = reporter_match.group(1).strip()
            if reporter in ['me', 'myself', 'i']:
                jql_parts.append('reporter = currentUser()')
            else:
                jql_parts.append(f'reporter ~ "{reporter}"')
        
        # If we have any parts, combine them with AND
        if jql_parts:
            jql = " AND ".join(jql_parts)
            # Add sorting if not already specified
            if "ORDER BY" not in jql:
                if "created" in jql:
                    jql += " ORDER BY created DESC"
                else:
                    jql += " ORDER BY updated DESC"
            return jql
        else:
            # If we don't have specific parts but it seems like a search query, 
            # return a default recent issues query
            if any(term in query for term in ['search', 'find', 'list', 'show', 'get', 'recent']):
                return "ORDER BY updated DESC"
            
            return None

    def ask_follow_up_question(self, user_query, response_so_far):
        """Decide if a follow-up question is needed to clarify the query."""
        # Check if the query is clear enough
        prompt = f"""
        Based on the user's query and my response so far, determine if I need to ask a clarifying follow-up question.

        User query: "{user_query}"
        
        My response so far: "{response_so_far}"
        
        Should I ask a clarifying question? If yes, what specific question should I ask to better help the user?
        
        Return your answer in this format:
        NEED_CLARIFICATION: [YES/NO]
        QUESTION: [The specific question to ask, if needed]
        """
        
        try:
            generation_config = GenerationConfig(
                temperature=0.2,
                top_p=0.8,
                max_output_tokens=1024,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            result_text = response.candidates[0].text
            
            # Parse the result
            need_clarification = "NEED_CLARIFICATION: YES" in result_text
            
            if need_clarification:
                question_match = re.search(r'QUESTION: (.*)', result_text)
                if question_match:
                    return question_match.group(1).strip()
            
            return None
        except Exception as e:
            logger.error(f"Error determining follow-up question: {str(e)}")
            return None

class InteractiveJiraAssistant:
    """Interactive command-line interface for JiraGeminiAssistant."""
    
    def __init__(self):
        """Initialize the interactive assistant."""
        # Initialize Jira client
        self.jira_client = self._initialize_jira_client()
        
        # Initialize Gemini assistant
        if self.jira_client:
            self.assistant = JiraGeminiAssistant(self.jira_client)
            logger.info("JiraGeminiAssistant initialized successfully.")
        else:
            logger.error("Failed to initialize JiraGeminiAssistant.")
            sys.exit(1)
    
    def _initialize_jira_client(self):
        """Initialize and test the Jira client connection."""
        logger.info("Initializing Jira client...")
        
        # Get Jira credentials from environment or ask user
        base_url = JIRA_BASE_URL
        username = JIRA_USERNAME
        token = JIRA_TOKEN
        
        if not base_url or base_url == "https://your-jira-instance.com":
            base_url = input("Enter your Jira base URL (e.g., https://your-company.atlassian.net): ")
        
        if not username:
            username = input("Enter your Jira username (email): ")
        
        if not token:
            import getpass
            token = getpass.getpass("Enter your Jira API token: ")
        
        # Initialize client
        jira_client = JiraClient(base_url, username, token)
        
        # Test connection
        if jira_client.test_connection():
            logger.info("Successfully connected to Jira.")
            return jira_client
        else:
            logger.error("Failed to connect to Jira. Please check your credentials and try again.")
            return None
    
    def run(self):
        """Run the interactive assistant."""
        print("\n" + "="*50)
        print("       📊 JIRA GEMINI ASSISTANT 🤖       ")
        print("="*50)
        print("Welcome to your Jira Gemini Assistant!")
        print("I can help you with Jira tickets, answer questions about your Jira environment,")
        print("search for issues, provide ticket details, and more.")
        print("\nType 'exit', 'quit', or 'bye' to end the session.")
        print("Type 'help' for usage tips.")
        print("="*50 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("\n🔍 YOU: ")
                
                # Check for exit command
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nThank you for using Jira Gemini Assistant. Goodbye!")
                    break
                
                # Check for help command
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                # Check for clear command
                if user_input.lower() in ['clear', 'reset']:
                    self.assistant.conversation_history = []
                    print("\nConversation history cleared.")
                    continue
                
                # Process the query
                if user_input.strip():
                    print("\n🤖 ASSISTANT: Processing your query...", end="", flush=True)
                    print("\r" + " " * 50 + "\r", end="")  # Clear the processing message
                    
                    # Generate response
                    response = self.assistant.generate_response(user_input)
                    
                    # Print the response with formatting
                    self._print_formatted_response(response)
                    
                    # Check if a follow-up question is needed
                    follow_up = self.assistant.ask_follow_up_question(user_input, response)
                    if follow_up:
                        print(f"\n🤖 FOLLOW-UP: {follow_up}")
                        
                        # Get user's response to the follow-up
                        follow_up_input = input("\n🔍 YOU: ")
                        
                        if follow_up_input.strip() and follow_up_input.lower() not in ['exit', 'quit', 'bye']:
                            # Generate new response with the follow-up information
                            print("\n🤖 ASSISTANT: Processing with additional information...", end="", flush=True)
                            print("\r" + " " * 50 + "\r", end="")
                            
                            enhanced_query = f"{user_input}\n\nFollow-up clarification: {follow_up_input}"
                            response = self.assistant.generate_response(enhanced_query)
                            
                            # Print the response with formatting
                            self._print_formatted_response(response)
            
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {str(e)}")
                logger.error(f"Error in main loop: {str(e)}")
    
    def _print_formatted_response(self, response):
        """Print the assistant's response with formatting."""
        # Split response into lines
        lines = response.split('\n')
        
        # Process each line for formatting
        for i, line in enumerate(lines):
            # Format headings
            if re.match(r'^#+\s+.+', line):
                heading_level = len(re.match(r'^(#+)', line).group(1))
                if heading_level == 1:
                    print(f"\n\033[1;36m{line}\033[0m")  # Cyan, bold
                elif heading_level == 2:
                    print(f"\n\033[1;34m{line}\033[0m")  # Blue, bold
                else:
                    print(f"\n\033[1;33m{line}\033[0m")  # Yellow, bold
            
            # Format links
            elif re.search(r'https?://\S+', line):
                formatted_line = re.sub(r'(https?://\S+)', r'\033[4;36m\1\033[0m', line)  # Cyan, underlined
                print(formatted_line)
            
            # Format bullet points
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                print(f"\033[0;32m{line}\033[0m")  # Green
            
            # Format numbered lists
            elif re.match(r'^\d+\.\s+', line):
                print(f"\033[0;32m{line}\033[0m")  # Green
            
            # Format code blocks
            elif line.strip().startswith('```') or line.strip() == '```':
                if line.strip() == '```':
                    print("\033[0;37m```\033[0m")  # Gray
                else:
                    print(f"\033[0;37m{line}\033[0m")  # Gray
            
            # Format ticket references
            elif re.search(r'\b[A-Z]+-\d+\b', line):
                formatted_line = re.sub(r'(\b[A-Z]+-\d+\b)', r'\033[1;35m\1\033[0m', line)  # Magenta, bold
                print(formatted_line)
            
            # Everything else
            else:
                print(line)
    
    def _show_help(self):
        """Show help information."""
        help_text = """
        === JIRA GEMINI ASSISTANT HELP ===
        
        GENERAL COMMANDS:
        - exit, quit, bye: End the session
        - help: Show this help message
        - clear, reset: Clear conversation history
        
        QUERY EXAMPLES:
        1. Specific Ticket Queries:
           - "Tell me about PROJ-123"
           - "What's the status of PROJ-123?"
           - "Who is assigned to PROJ-123?"
           - "Show me the description of PROJ-123"
           - "List all comments on PROJ-123"
        
        2. Search Queries:
           - "Find all open bugs in project PROJ"
           - "Show me tickets assigned to John"
           - "List all high priority tickets created this week"
           - "Search for tickets containing 'login error'"
           - "Find all unassigned tickets in PROJ"
        
        3. Project Queries:
           - "Give me an overview of project PROJ"
           - "What are the active sprints in PROJ?"
           - "Who are the main contributors in PROJ?"
           - "How many open tickets are in PROJ?"
        
        4. Analysis Queries:
           - "What are the most common issues in PROJ?"
           - "Summarize the status of PROJ-123"
           - "Analyze the comments in PROJ-123"
           - "What's changed in PROJ-123 in the last week?"
        
        5. Specific Information:
           - "What is the workflow for PROJ?"
           - "Who is the project lead for PROJ?"
           - "When was PROJ-123 created?"
           - "What attachments are in PROJ-123?"
        
        Remember, you can ask follow-up questions to get more detailed information!
        """
        print(help_text)

def main():
    """Main entry point for the application."""
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Jira Gemini Assistant')
        parser.add_argument('--base-url', help='Jira base URL')
        parser.add_argument('--username', help='Jira username')
        parser.add_argument('--token', help='Jira API token')
        parser.add_argument('--project-id', help='Google Cloud project ID')
        parser.add_argument('--region', help='Google Cloud region')
        parser.add_argument('--model', help='Gemini model name')
        args = parser.parse_args()
        
        # Set environment variables if provided
        if args.base_url:
            os.environ['JIRA_BASE_URL'] = args.base_url
        if args.username:
            os.environ['JIRA_USERNAME'] = args.username
        if args.token:
            os.environ['JIRA_TOKEN'] = args.token
        if args.project_id:
            os.environ['PROJECT_ID'] = args.project_id
        if args.region:
            os.environ['REGION'] = args.region
        if args.model:
            os.environ['MODEL_NAME'] = args.model
        
        # Start the interactive assistant
        assistant = InteractiveJiraAssistant()
        assistant.run()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

