import os
import re
import json
import time
import logging
import threading
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Any, Union
from functools import lru_cache
import requests
from bs4 import BeautifulSoup, Tag
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
==================== CONFIGURATION ====================
Configure logging
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
handlers=[
logging.StreamHandler(),
logging.FileHandler("copper_assistant.log")
]
)
logger = logging.getLogger("CopperAssistant")
Constants
DEFAULT_BATCH_SIZE = 100
MAX_CONTENT_SIZE = 50000  # Maximum content size for Gemini input
MAX_WORKERS = 4  # Maximum workers for concurrent processing
CACHE_SIZE = 1000  # Maximum number of entries in LRU cache
CACHE_DIR = ".cache"
if not os.path.exists(CACHE_DIR):
os.makedirs(CACHE_DIR, exist_ok=True)
PAGE_CACHE_FILE = os.path.join(CACHE_DIR, "page_cache.json")
MODEL_NAME = "gemini-1.5-pro"
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "default-project")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
==================== CONTENT EXTRACTION ====================
class ContentExtractor:
"""Enhanced content extraction with support for iframes, complex tables, and images."""
@staticmethod
def extract_content_from_html(html_content, title=""):
    """
    Extract and structure content from HTML, handling complex elements.
    
    Args:
        html_content: The HTML content to process
        title: The title of the page
    
    Returns:
        Dict containing structured content including text, tables, images, and code blocks
    """
    try:
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Track all extracted content
        text_content = []
        tables = []
        images = []
        code_blocks = []
        structured_content = []
        
        # Extract title if not provided
        if not title and soup.title:
            title = soup.title.text.strip()
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Process paragraphs and list items
        for p in soup.find_all(['p', 'li']):
            if p.text.strip():
                text_content.append(p.text.strip())
        
        # Extract tables with enhanced processing
        ContentExtractor._extract_tables(soup, tables)
        
        # Extract images with context
        ContentExtractor._extract_images(soup, images)
        
        # Extract code blocks with improved formatting
        ContentExtractor._extract_code_blocks(soup, code_blocks)
        
        # Extract content from iframes
        ContentExtractor._extract_iframe_content(soup, text_content, tables, images)
        
        # Extract any important structured content
        ContentExtractor._extract_structured_content(soup, structured_content)
        
        return {
            "text": "\n\n".join(text_content),
            "tables": tables,
            "images": images,
            "code_blocks": code_blocks,
            "structured_content": structured_content
        }
    
    except Exception as e:
        logger.error(f"Error extracting content: {str(e)}")
        # Return minimal structure with error message
        return {
            "text": f"Error extracting content: {str(e)}",
            "tables": [],
            "images": [],
            "code_blocks": [],
            "structured_content": []
        }

@staticmethod
def _extract_tables(soup, tables):
    """Extract tables with enhanced processing for complex Confluence tables."""
    # Handle standard tables
    standard_tables = soup.find_all('table')
    
    # Handle Confluence-specific table wrappers
    confluence_tables = soup.find_all('div', class_='table-wrap')
    
    all_tables = []
    # Process standard tables
    for table in standard_tables:
        # Skip if table is inside a already-processed table-wrap
        if table.find_parent('div', class_='table-wrap'):
            continue
        all_tables.append(table)
    
    # Add Confluence tables
    for table_wrap in confluence_tables:
        table = table_wrap.find('table')
        if table:
            all_tables.append(table)
    
    # Process all tables
    for table in all_tables:
        # Extract table data
        table_data = []
        
        # Get table title/caption if available
        caption = table.find('caption')
        table_title = caption.text.strip() if caption else f"Table {len(tables)+1}"
        
        # Get headers
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
                # Check if it looks like a header row (has th elements or all cells look like headers)
                first_row_cells = first_row.find_all(['th', 'td'])
                if first_row.find('th') or all(cell.name == 'th' and cell.get('class') and 'header' in '.'.join(cell.get('class', [])) for cell in first_row_cells):
                    headers = [th.text.strip() for th in first_row_cells]
        
        # Process rows
        rows = []
        tbody = table.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                row = [td.text.strip() for td in tr.find_all(['td', 'th'])]
                if any(cell for cell in row):  # Skip empty rows
                    rows.append(row)
        else:
            # If no tbody, process all rows (skipping the header if we extracted it)
            all_rows = table.find_all('tr')
            start_idx = 1 if headers and len(all_rows) > 0 else 0
            for tr in all_rows[start_idx:]:
                row = [td.text.strip() for td in tr.find_all(['td', 'th'])]
                if any(cell for cell in row):  # Skip empty rows
                    rows.append(row)
        
        # Convert table to text representation with improved formatting
        table_text = [f"[TABLE: {table_title}]"]
        
        # Format with consistent column widths for better readability
        if headers and rows:
            # Calculate column widths
            col_widths = [max(len(str(h)), 3) for h in headers]
            for row in rows:
                for i, cell in enumerate(row[:len(col_widths)]):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
            
            # Format header row
            header_row = "| " + " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
            separator = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
            table_text.append(header_row)
            table_text.append(separator)
            
            # Format data rows
            for row in rows:
                # Pad row if needed to match header length
                if len(row) < len(headers):
                    row.extend([""] * (len(headers) - len(row)))
                # Truncate if longer than headers
                row = row[:len(headers)]
                row_text = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
                table_text.append(row_text)
        elif rows:  # Table with no headers
            # Calculate column widths
            max_cols = max(len(row) for row in rows)
            col_widths = [0] * max_cols
            for row in rows:
                for i, cell in enumerate(row[:max_cols]):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
            
            # Format rows
            for row in rows:
                # Pad row if needed to match max columns
                if len(row) < max_cols:
                    row.extend([""] * (max_cols - len(row)))
                row_text = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
                table_text.append(row_text)
        
        if len(table_text) > 1:  # Only add tables with content
            tables.append("\n".join(table_text))

@staticmethod
def _extract_images(soup, images):
    """Extract images with improved context."""
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
            if prev_elem and len(str(prev_elem).strip()) < 200:  # Only short contexts
                if hasattr(prev_elem, 'text'):
                    context = f"Previous content: {prev_elem.text.strip()}"
        
        # Construct meaningful image description
        desc = alt_text or title or "Image"
        if context:
            desc += f" - {context}"
        
        images.append(f"[IMAGE: {desc}]")

@staticmethod
def _extract_code_blocks(soup, code_blocks):
    """Extract code blocks with improved formatting."""
    for pre in soup.find_all('pre'):
        code = pre.find('code')
        if code:
            # Check for any language specification
            code_class = code.get('class', [])
            lang = ""
            for cls in code_class:
                if cls.startswith('language-'):
                    lang = cls.replace('language-', '')
                    break
            
            code_content = code.text.strip()
            if lang:
                code_blocks.append(f"```{lang}\n{code_content}\n```")
            else:
                code_blocks.append(f"```\n{code_content}\n```")
        else:
            # Pre without code tag
            code_blocks.append(f"```\n{pre.text.strip()}\n```")

@staticmethod
def _extract_iframe_content(soup, text_content, tables, images):
    """Extract content from iframes."""
    for iframe in soup.find_all('iframe'):
        iframe_src = iframe.get('src', '')
        if iframe_src:
            # Add reference to iframe content
            text_content.append(f"[IFRAME CONTENT: {iframe_src}]")
            
            # For Gliffy diagrams, extract available metadata
            if 'gliffy' in iframe_src.lower():
                # Extract diagram title if possible
                diagram_title = iframe.get('title', '') 
                parent_div = iframe.find_parent('div', class_='ap-container')
                if parent_div and not diagram_title:
                    diagram_title = parent_div.get('data-macro-name', '')
                if diagram_title:
                    images.append(f"[DIAGRAM: {diagram_title}]")

@staticmethod
def _extract_structured_content(soup, structured_content):
    """Extract any important structured content."""
    for div in soup.find_all(['div', 'section']):
        if 'class' in div.attrs:
            # Look for common Confluence structured content classes
            class_str = ' '.join(div['class'])
            if any(term in class_str for term in ['panel', 'info', 'note', 'warning', 'callout', 'aui-message']):
                title_elem = div.find(['h3', 'h4', 'h5', 'strong', 'b'])
                title = title_elem.text.strip() if title_elem else "Note"
                content = div.text.strip()
                structured_content.append(f"--- {title} ---\n{content}")

@staticmethod
def format_for_context(extracted_content, title=""):
    """
    Format the extracted content for use as context.
    
    Args:
        extracted_content: The dictionary of extracted content
        title: The title of the page
        
    Returns:
        Formatted string containing all the content
    """
    sections = []
    
    if title:
        sections.append(f"# {title}")
    
    if extracted_content.get("text"):
        sections.append(extracted_content["text"])
    
    if extracted_content.get("tables"):
        for table in extracted_content["tables"]:
            sections.append(f"\n{table}")
    
    if extracted_content.get("code_blocks"):
        sections.append("\nCode Examples:")
        sections.extend(extracted_content["code_blocks"])
    
    if extracted_content.get("structured_content"):
        sections.append("\nImportant Notes:")
        sections.extend(extracted_content["structured_content"])
    
    if extracted_content.get("images"):
        sections.append("\nImage Information:")
        sections.extend(extracted_content["images"])
    
    return "\n\n".join(sections)
==================== CONFLUENCE INTEGRATION ====================
class ConfluenceClient:
"""Client for Confluence REST API operations with comprehensive error handling and caching."""
def __init__(self, base_url, username, api_token):
    """
    Initialize the Confluence client with authentication details.
    
    Args:
        base_url: The base URL of the Confluence instance (e.g., https://mycompany.atlassian.net)
        username: The username for authentication
        api_token: The API token for authentication
    """
    self.base_url = base_url.rstrip('/')
    self.auth = (username, api_token)
    self.api_url = f"{self.base_url}/wiki/rest/api"
    self.headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "COPPER-AI-Python-Agent"
    }
    self.session = requests.Session()
    # Set up a default request timeout
    self.timeout = 30
    # Cache for API responses
    self.cache = {}
    # Thread lock for the cache
    self.cache_lock = threading.Lock()
    
    logger.info(f"Initialized Confluence client for {self.base_url}")

def test_connection(self):
    """Test the connection to Confluence API."""
    try:
        logger.info("Testing connection to Confluence...")
        response = self.session.get(
            f"{self.api_url}/space",
            auth=self.auth,
            headers=self.headers,
            params={"limit": 1},
            timeout=self.timeout,
            verify=False  # Using verify=False as specified for your environment
        )
        response.raise_for_status()
        
        if response.status_code == 200:
            logger.info("Connection to Confluence successful!")
            return True
        else:
            logger.warning("Empty response received during connection test")
            return False
            
    except requests.RequestException as e:
        logger.error(f"Connection test failed: {str(e)}")
        return False

@lru_cache(maxsize=CACHE_SIZE)
def get_cached_request(self, url, params_str):
    """Cached version of GET requests to reduce API calls."""
    try:
        params = json.loads(params_str)
        response = self.session.get(
            url,
            auth=self.auth,
            headers=self.headers,
            params=params,
            timeout=self.timeout,
            verify=False  # Using verify=False as specified for your environment
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Error in cached request to {url}: {str(e)}")
        return None

def get_all_pages_in_space(self, space_key, batch_size=100):
    """
    Get all pages in a Confluence space using efficient pagination.
    
    Args:
        space_key: The space key to get all pages from
        batch_size: Number of results per request (max 100)
        
    Returns:
        List of page objects with basic information
    """
    logger.info(f"Fetching all pages from space: {space_key}")
    
    all_pages = []
    start = 0
    has_more = True
    
    # Check if we have cached results
    cache_path = os.path.join(CACHE_DIR, f"pages_{space_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
                if cached_data.get('space_key') == space_key:
                    logger.info(f"Using cached page list from {cache_path}")
                    return cached_data.get('pages', [])
        except Exception as e:
            logger.warning(f"Error reading cache file: {str(e)}")
    
    # If no cache, fetch all pages
    while has_more:
        logger.info(f"Fetching pages batch from start={start}")
        
        try:
            params = {
                "spaceKey": space_key,
                "expand": "history",  # Include basic history info to get last updated date
                "limit": batch_size,
                "start": start
            }
            
            # Convert params to string for cache key
            params_str = json.dumps(params, sort_keys=True)
            
            # Try to get from cache first
            response_text = self.get_cached_request(f"{self.api_url}/content", params_str)
            
            if not response_text:
                logger.warning(f"Empty response when fetching pages at start={start}")
                break
            
            response_data = json.loads(response_text)
            
            results = response_data.get("results", [])
            all_pages.extend(results)
            
            # Check if there are more pages
            if "size" in response_data and "limit" in response_data:
                if response_data["size"] < response_data["limit"]:
                    has_more = False
                else:
                    start += batch_size
            else:
                has_more = False
            
            logger.info(f"Fetched {len(results)} pages, total so far: {len(all_pages)}")
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error fetching pages: {str(e)}")
            break
    
    # Cache the results
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({'space_key': space_key, 'pages': all_pages}, f)
            logger.info(f"Cached {len(all_pages)} pages to {cache_path}")
    except Exception as e:
        logger.warning(f"Error writing cache file: {str(e)}")
    
    logger.info(f"Successfully fetched {len(all_pages)} pages from space {space_key}")
    return all_pages

def get_page_content(self, page_id, expand="body.storage,metadata.labels"):
    """
    Get the content of a page in a suitable format for NLP.
    This extracts and processes the content to be more suitable for embeddings.
    
    Args:
        page_id: The ID of the page
        expand: What to expand in the API request
        
    Returns:
        Dict containing the processed content
    """
    try:
        # Use cached version if available
        cache_key = f"page_content_{page_id}"
        with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        page = self.get_content_by_id(page_id, expand=f"body.storage,metadata.labels")
        if not page:
            return None
        
        # Extract basic metadata
        metadata = {
            "id": page.get("id"),
            "title": page.get("title"),
            "type": page.get("type"),
            "url": f"{self.base_url}/pages/viewpage.action?pageId={page.get('id')}",
            "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
        }
        
        # Get raw content
        html_content = page.get("body", {}).get("storage", {}).get("value", "")
        
        # Process with our advanced content extractor
        extracted_content = ContentExtractor.extract_content_from_html(html_content, page.get("title", ""))
        formatted_content = ContentExtractor.format_for_context(extracted_content, page.get("title", ""))
        
        result = {
            "metadata": metadata,
            "content": formatted_content,
            "raw_html": html_content
        }
        
        # Cache the result
        with self.cache_lock:
            self.cache[cache_key] = result
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing page content: {str(e)}")
        return None

def get_content_by_id(self, content_id, expand=None):
    """
    Get content by ID with optional expansion parameters.
    
    Args:
        content_id: The ID of the content to retrieve
        expand: Optional expansion parameters
        
    Returns:
        Content object or None if not found/error
    """
    params = {}
    if expand:
        params["expand"] = expand
    
    # Convert params to string for cache key
    params_str = json.dumps(params, sort_keys=True)
    
    try:
        # Try to get from cache first
        response_text = self.get_cached_request(f"{self.api_url}/content/{content_id}", params_str)
        
        if not response_text:
            return None
        
        content = json.loads(response_text)
        logger.info(f"Successfully retrieved content: {content.get('title', 'Unknown title')}")
        return content
        
    except Exception as e:
        logger.error(f"Error getting content by ID {content_id}: {str(e)}")
        return None

def fetch_page_content_batch(self, pages):
    """Fetch content for a batch of pages in parallel."""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_page = {
            executor.submit(self.get_page_content, page["id"]): page["id"]
            for page in pages
        }
        
        for future in concurrent.futures.as_completed(future_to_page):
            page_id = future_to_page[future]
            try:
                content = future.result()
                if content:
                    results[page_id] = content
            except Exception as e:
                logger.error(f"Error processing page {page_id}: {str(e)}")
    
    return results
==================== SQL VIEW PARSER ====================
class SqlViewParser:
"""
Advanced SQL View parser for extracting detailed information from view definitions.
Handles complex SQL syntax and extracts structural information.
"""
@staticmethod
def parse_view_definition(sql_text):
    """
    Parse a SQL view definition into a structured format.
    
    Args:
        sql_text: The SQL view definition text
        
    Returns:
        Dict containing parsed view information
    """
    view_info = {
        "view_name": "",
        "columns": [],
        "select_clause": "",
        "from_clause": "",
        "where_clause": "",
        "join_conditions": [],
        "filter_conditions": [],
        "case_statements": [],
        "transformations": [],
        "source_tables": []
    }
    
    # Extract view name
    view_name_match = re.search(r'VIEW\s+(\w+)', sql_text, re.IGNORECASE)
    if view_name_match:
        view_info["view_name"] = view_name_match.group(1)
    
    # Handle CREATE OR REPLACE VIEW syntax
    create_view_match = re.search(r'CREATE\s+OR\s+REPLACE\s+(?:FORCE\s+)?VIEW\s+(\w+)(?:\s*\((.*?)\))?\s+AS\s+(.*)', 
                                sql_text, re.IGNORECASE | re.DOTALL)
    
    if create_view_match:
        view_info["view_name"] = create_view_match.group(1)
        
        # Extract column definitions if provided inline
        if create_view_match.group(2):
            columns_text = create_view_match.group(2)
            # Split by commas that are not inside parentheses
            columns = []
            
            # Simple column extraction (can be improved with regex)
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
            
            view_info["columns"] = columns
        
        # Extract the SELECT statement
        select_stmt = create_view_match.group(3)
        view_info["select_clause"] = select_stmt
        
        # Extract FROM clause
        from_match = re.search(r'FROM\s+(.*?)(?:WHERE|GROUP BY|ORDER BY|$)', select_stmt, re.IGNORECASE | re.DOTALL)
        if from_match:
            view_info["from_clause"] = from_match.group(1).strip()
            
            # Extract source tables from FROM and JOIN clauses
            table_matches = re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', from_match.group(1), re.IGNORECASE)
            source_tables = []
            for match in table_matches:
                if match[0]:
                    source_tables.append(match[0])
                elif match[1]:
                    source_tables.append(match[1])
            
            view_info["source_tables"] = source_tables
            
            # Extract JOIN conditions
            join_matches = re.findall(r'((?:INNER|LEFT|RIGHT|FULL|OUTER)?\s*JOIN\s+\w+\s+.*?ON\s+[^;]+?)(?=(?:INNER|LEFT|RIGHT|FULL|OUTER)?\s*JOIN|\s*WHERE|\s*GROUP BY|\s*ORDER BY|$)', 
                                     from_match.group(1), re.IGNORECASE | re.DOTALL)
            
            view_info["join_conditions"] = [j.strip() for j in join_matches]
        
        # Extract WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|$)', select_stmt, re.IGNORECASE | re.DOTALL)
        if where_match:
            view_info["where_clause"] = where_match.group(1).strip()
            
            # Parse individual filter conditions
            # This is complex due to nested conditions - a simplified approach
            conditions_text = where_match.group(1)
            
            # Handle nested parentheses in conditions
            parsed_conditions = SqlViewParser._parse_conditions(conditions_text)
            view_info["filter_conditions"] = parsed_conditions
        
        # Extract CASE statements
        case_matches = re.findall(r'(CASE\s+.*?END)', sql_text, re.IGNORECASE | re.DOTALL)
        view_info["case_statements"] = case_matches
        
        # Extract other transformations/functions
        function_matches = re.findall(r'(\w+\s*\([^)]*\))', sql_text)
        view_info["transformations"] = [f for f in function_matches if not any(
            common in f.upper() for common in ['COUNT(', 'SUM(', 'MIN(', 'MAX(', 'AVG('])]
    
    return view_info

@staticmethod
def _parse_conditions(conditions_text):
    """Parse complex WHERE conditions with nested ANDs and ORs."""
    # Simple parsing for demo - in production would use a proper SQL parser
    # This handles basic AND/OR separation but not nested parentheses properly
    
    # Split on top-level AND/OR
    conditions = []
    
    # Handle simple case first
    if 'AND' not in conditions_text.upper() and 'OR' not in conditions_text.upper():
        return [conditions_text.strip()]
    
    # Otherwise, split by AND/OR preserving parenthesized expressions
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
==================== API MAPPING GENERATOR ====================
class ApiMappingGenerator:
"""
Generator for CoPPER API JSON request bodies based on view definitions.
Creates properly structured JSON for querying CoPPER APIs.
"""
@staticmethod
def generate_api_json(view_info, primary_domain="data"):
    """
    Generate API JSON request body based on view information.
    
    Args:
        view_info: Dict with parsed view information
        primary_domain: Primary domain for the API
        
    Returns:
        Dict containing the API JSON request body
    """
    # Start with base request structure
    request_body = {
        "reqName": primary_domain,
        "dataDomain": primary_domain,
        "type": "independent",
        "responseInclude": True,
        "req": []
    }
    
    # For dependent domains, adjust the type
    if primary_domain in ["session", "instruments"]:
        dependent_domains = {
            "session": {
                "reqName": "session",
                "dataDomain": "sessions",
                "type": "dependent", 
                "distinct": True
            },
            "instruments": {
                "reqName": "instruments",
                "dataDomain": "instruments",
                "type": "dependent",
                "distinct": True
            }
        }
        
        domain_info = dependent_domains.get(primary_domain)
        if domain_info:
            for key, value in domain_info.items():
                request_body[key] = value
    
    # Process WHERE conditions to generate request parameters
    if view_info.get("filter_conditions"):
        for condition in view_info["filter_conditions"]:
            param = ApiMappingGenerator._condition_to_param(condition)
            if param:
                request_body["req"].append(param)
    
    # Add domain-specific default parameters
    ApiMappingGenerator._add_domain_defaults(request_body, primary_domain)
    
    # Add fields based on view columns
    field_list = []
    for column in view_info.get("columns", []):
        # Extract just the column name without table prefixes or aliases
        if " AS " in column.upper():
            # Column has an alias
            col_name = column.split(" AS ", 1)[1].strip()
        else:
            # No alias, just extract the base name
            col_parts = column.split(".")
            col_name = col_parts[-1].strip()
        
        # Convert to camelCase for API fields
        if "_" in col_name:
            parts = col_name.lower().split("_")
            camel_case = parts[0] + "".join(p.capitalize() for p in parts[1:])
            field_list.append(camel_case)
        else:
            field_list.append(col_name.lower())
    
    if field_list:
        request_body["fields"] = field_list
    
    return request_body

@staticmethod
def _condition_to_param(condition):
    """Convert a SQL WHERE condition to an API request parameter."""
    # Map of common SQL operators to API operations
    op_map = {
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
        "LIKE": "LIKE"
    }
    
    condition = condition.strip()
    
    # Handle IS NULL / IS NOT NULL
    if "IS NULL" in condition.upper():
        field = condition.split("IS NULL", 1)[0].strip()
        field = field.split(".")[-1].lower()  # Get just the column name
        return {
            "tag": field,
            "operation": "ISNULL"
        }
    
    if "IS NOT NULL" in condition.upper():
        field = condition.split("IS NOT NULL", 1)[0].strip()
        field = field.split(".")[-1].lower()  # Get just the column name
        return {
            "tag": field,
            "operation": "NOTNULL"
        }
    
    # Handle IN conditions
    in_match = re.search(r'(\w+(?:\.\w+)?)\s+IN\s*\((.*?)\)', condition, re.IGNORECASE | re.DOTALL)
    if in_match:
        field = in_match.group(1).split(".")[-1].lower()
        values_text = in_match.group(2)
        values = []
        
        # Parse the values, handling quoted strings
        for val in re.findall(r'\'([^\']*?)\'|"([^"]*?)"|(\w+)', values_text):
            value = val[0] or val[1] or val[2]
            if value and not value.upper() in ['AND', 'OR', 'IN']:
                values.append(value)
        
        return {
            "tag": field,
            "value": values,
            "operation": "IN"
        }
    
    # Handle other operators
    for op in op_map.keys():
        if f" {op} " in f" {condition} ":  # Add spaces to ensure we match whole operators
            parts = condition.split(op, 1)
            if len(parts) == 2:
                field = parts[0].strip()
                field = field.split(".")[-1].lower()  # Get just the column name
                
                value = parts[1].strip()
                # Remove quotes if present
                if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
                    value = value[1:-1]
                
                return {
                    "tag": field,
                    "value": value,
                    "operation": op_map[op]
                }
    
    return None

@staticmethod
def _add_domain_defaults(request_body, domain):
    """Add domain-specific default parameters."""
    defaults = {
        "firm": [
            {"tag": "effEndTimp", "value": "TODAY", "operation": "GT"}
        ],
        "session": [
            {"tag": "sessionFeature.features", "value": ["BTEC-EU", "BTEC-US"], "operation": "IN"}
        ],
        "instruments": [
            {"tag": "prodTyp", "value": "BOND", "operation": "EQ"}
        ],
        "product": [
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
            for req in request_body["req"]:
                if req.get("tag") == param["tag"]:
                    existing = True
                    break
            
            if not existing:
                request_body["req"].append(param)
==================== DOMAIN CLASSIFIER ====================
class DomainClassifier:
"""Classifier for determining the primary domain for a view or query."""
@staticmethod
def classify_view(view_info):
    """
    Classify a view to determine its primary domain.
    
    Args:
        view_info: Dict with parsed view information
        
    Returns:
        String containing the primary domain
    """
    view_name = view_info.get("view_name", "").upper()
    source_tables = view_info.get("source_tables", [])
    filter_conditions = view_info.get("filter_conditions", [])
    
    # Define domain indicators
    domain_indicators = {
        "firm": ["FIRM", "TRADER", "INSTITUTION"],
        "session": ["SESSION", "TRADER", "GRID"],
        "instruments": ["INSTRUMENT", "ERSA", "IPALM"],
        "product": ["PRODUCT", "CORE"],
        "risk": ["RISK", "BTEC"],
        "static": ["STATIC", "REF"]
    }
    
    # Score each domain based on matches
    domain_scores = {}
    for domain, indicators in domain_indicators.items():
        score = 0
        
        # Check view name
        for indicator in indicators:
            if indicator in view_name:
                score += 5
        
        # Check source tables
        for table in source_tables:
            for indicator in indicators:
                if indicator in table.upper():
                    score += 2
        
        # Check filter conditions
        for condition in filter_conditions:
            for indicator in indicators:
                if indicator in condition.upper():
                    score += 1
        
        domain_scores[domain] = score
    
    # Return domain with highest score, default to "data"
    if domain_scores:
        return max(domain_scores.items(), key=lambda x: x[1])[0] if max(domain_scores.values()) > 0 else "data"
    return "data"

@staticmethod
def classify_query(query):
    """
    Classify a user query to determine relevant domains.
    
    Args:
        query: User's query string
        
    Returns:
        List of relevant domains
    """
    query_lower = query.lower()
    
    # Define domain keywords
    domain_keywords = {
        "firm": ["firm", "trader", "institution", "company", "organization"],
        "session": ["session", "trade", "trading", "grid"],
        "instruments": ["instrument", "bond", "security", "product"],
        "product": ["product", "core", "item"],
        "risk": ["risk", "btec", "group", "exposure"],
        "static": ["static", "reference", "lookup"]
    }
    
    # Score domains based on keyword matches
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in query_lower:
                score += 1
        domain_scores[domain] = score
    
    # Return domains with scores > 0, sorted by score
    relevant_domains = [domain for domain, score in domain_scores.items() if score > 0]
    return sorted(relevant_domains, key=lambda x: domain_scores[x], reverse=True)
==================== GEMINI INTEGRATION ====================
class GeminiAssistant:
"""Class for interacting with Gemini models via Vertex AI."""
def __init__(self):
    """Initialize Vertex AI and Gemini model."""
    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    self.model = GenerativeModel(MODEL_NAME)
    logger.info(f"Initialized Gemini Assistant with model: {MODEL_NAME}")

def generate_response(self, prompt, copper_context=None, system_prompt=None):
    """
    Generate a response from Gemini based on the prompt and CoPPER context.
    
    Args:
        prompt: The user's question or prompt
        copper_context: Context information about CoPPER (from Confluence)
        system_prompt: Optional system prompt to guide the response
        
    Returns:
        The generated response
    """
    logger.info(f"Generating response for prompt: {prompt}")
    
    try:
        # Default system prompt if none provided
        if not system_prompt:
            system_prompt = """
You are the friendly CoPPER Assistant, an expert on mapping database views to REST APIs.
Your personality:

Conversational and approachable - use a casual, helpful tone while maintaining workplace professionalism
Explain technical concepts in plain language, as if speaking to a colleague
Use simple analogies and examples to clarify complex ideas
Add occasional light humor where appropriate to make the conversation engaging
Be concise but thorough - focus on answering the question directly first, then add helpful context

Your expertise:

Deep knowledge of the CoPPER database system, its views, and corresponding API endpoints
Understanding database-to-API mapping patterns and best practices
Awareness of how applications integrate with CoPPER's REST APIs
Expert in interpreting table structures, field mappings, and API parameters

When answering:

Directly address the user's question first
Provide practical, actionable information when possible
Format tables and structured data clearly to enhance readability
Use bullet points or numbered lists for steps or multiple items
Reference specific examples from the documentation when available
Acknowledge any limitations in the available information

Remember to maintain a balance between being friendly and professional - you're a helpful colleague, not a formal technical document.
"""
        # Trim context if it's too large
        if copper_context:
            if len(copper_context) > MAX_CONTENT_SIZE:
                logger.warning(f"Context too large ({len(copper_context)} chars), trimming...")
                # Try to trim at paragraph boundaries
                paragraphs = copper_context.split("\n\n")
                trimmed_content = ""
                for para in paragraphs:
                    if len(trimmed_content) + len(para) + 2 < MAX_CONTENT_SIZE:
                        trimmed_content += para + "\n\n"
                    else:
                        break
                copper_context = trimmed_content
                logger.info(f"Trimmed context to {len(copper_context)} chars")
            
            full_prompt = f"{system_prompt}\n\nCONTEXT INFORMATION:\n{copper_context}\n\nUSER QUESTION: {prompt}\n\nResponse:"
        else:
            full_prompt = f"{system_prompt}\n\nUSER QUESTION: {prompt}\n\nResponse:"
        
        # Configure generation parameters
        generation_config = GenerationConfig(
            temperature=0.3,  # Lower temperature for more factual responses
            top_p=0.95,
        )
        
        # Generate the response
        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config,
        )
        
        if response.candidates and response.candidates[0].text:
            response_text = response.candidates[0].text.strip()
            logger.info(f"Successfully generated response ({len(response_text)} chars)")
            return response_text
        else:
            logger.warning("No response generated from Gemini")
            return "I couldn't find a specific answer to that question in our documentation. Could you try rephrasing, or maybe I can help you find the right documentation to look at?"
            
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"I ran into a technical issue while looking that up. Let me know if you'd like to try a different question or approach."

def analyze_view(self, view_def):
    """
    Use Gemini to analyze a view definition and provide insights.
    
    Args:
        view_def: SQL view definition string
        
    Returns:
        String containing analysis of the view
    """
    system_prompt = """
You are a database view analysis expert. Your task is to analyze a SQL view definition and extract meaningful insights.
Focus on:

The overall purpose of the view
Key columns and their significance
Join relationships and their business meaning
Important filtering conditions
Data transformations and calculations
Potential use cases for this view

Provide your analysis in a structured format with clear sections for:

Overview & Purpose
Key Columns
Relationships
Filtering Logic
Business Rules
Typical Use Cases

Be concise but thorough, highlighting what's most important about this view.
"""
    prompt = f"Please analyze this SQL view definition:\n\n```sql\n{view_def}\n```"
    
    return self.generate_response(prompt, system_prompt=system_prompt)

def generate_api_mapping_explanation(self, view_def, api_json):
    """
    Generate an explanation of how a view maps to an API request.
    
    Args:
        view_def: SQL view definition string
        api_json: The generated API JSON structure
        
    Returns:
        String containing explanation of the mapping
    """
    system_prompt = """
You are an expert in mapping SQL database views to REST API requests. Your task is to explain how a SQL view definition translates to a specific API request format.
Focus on:

How WHERE conditions in SQL map to request parameters
How source tables and joins map to domain selection
How columns map to field selection
How default values and assumptions are made

Provide your explanation in a clear, tutorial-like manner that helps the user understand:

The overall mapping strategy
Why specific parameters were included
How to modify the API request for different scenarios
Any special handling or edge cases

Use concrete examples from the provided view and API request.
"""
    # Convert API JSON to string representation
    api_json_str = json.dumps(api_json, indent=2)
    
    prompt = f"""
Please explain how this SQL view definition:
sql{view_def}
Maps to this API request JSON:
json{api_json_str}
Provide a detailed explanation of the mapping logic and how someone could modify this for similar views.
"""
    return self.generate_response(prompt, system_prompt=system_prompt)
==================== COPPER ASSISTANT ====================
class CopperAssistant:
"""Main class that coordinates between Confluence and Gemini."""
def __init__(self, confluence_url, confluence_username, confluence_api_token, space_key=None):
    """
    Initialize the Copper Assistant.
    
    Args:
        confluence_url: The base URL of the Confluence instance
        confluence_username: The username for Confluence authentication
        confluence_api_token: The API token for Confluence authentication
        space_key: The space key to target (or 'all' for all spaces)
    """
    self.confluence = ConfluenceClient(confluence_url, confluence_username, confluence_api_token)
    self.gemini = GeminiAssistant()
    self.space_key = space_key
    self.space_pages = []
    self.page_content_cache = {}  # Cache for page content to avoid re-fetching
    
    logger.info(f"Initialized Copper Assistant targeting space: {space_key or 'all spaces'}")

def initialize(self):
    """Initialize by testing connections and gathering initial space content."""
    if not self.confluence.test_connection():
        logger.error("Failed to connect to Confluence. Check credentials and URL.")
        return False
    
    logger.info("Loading space content...")
    self.load_space_content()
    return True

def load_space_content(self):
    """Load metadata for all pages in the specified space."""
    if not self.space_key:
        logger.error("No space key specified. Please provide a space key.")
        return
    
    self.space_pages = self.confluence.get_all_pages_in_space(self.space_key)
    logger.info(f"Loaded metadata for {len(self.space_pages)} pages from space {self.space_key}")

def extract_relevant_content(self, query):
    """
    Extract relevant content based on the user's query.
    
    Args:
        query: The user's question or query
        
    Returns:
        String containing the relevant information
    """
    # Check if we have pages loaded
    if not self.space_pages:
        logger.error("No space pages loaded. Call load_space_content() first.")
        return "I couldn't find any relevant information in our documentation."
    
    # First, filter pages based on title and metadata relevance
    candidate_pages = []
    
    # Simple relevance filtering (could be enhanced with embeddings)
    query_lower = query.lower()
    query_words = query_lower.split()
    
    # Look for view definition requests
    view_match = re.search(r'view\s+(\w+)', query_lower)
    if view_match:
        view_name = view_match.group(1)
        logger.info(f"Looking for specific view: {view_name}")
        
        # First, search for exact matches in the title
        for page in self.space_pages:
            if view_name.lower() in page.get("title", "").lower():
                candidate_pages.append(page)
        
        if not candidate_pages:
            # If no exact matches, search for pages with "view" in the title
            for page in self.space_pages:
                if "view" in page.get("title", "").lower():
                    candidate_pages.append(page)
    else:
        # For general queries, score based on query term presence
        page_scores = []
        for page in self.space_pages:
            page_score = 0
            page_title = page.get("title", "").lower()
            
            # Score based on words in title
            for word in query_words:
                if word in page_title:
                    page_score += 2
            
            # Higher score for multi-word matches
            for i in range(len(query_words)-1):
                if i + 1 < len(query_words):
                    two_words = f"{query_words[i]} {query_words[i+1]}"
                    if two_words in page_title:
                        page_score += 3
            
            if page_score > 0:
                page_scores.append((page, page_score))
        
        # Sort by score and take top 20
        page_scores.sort(key=lambda x: x[1], reverse=True)
        candidate_pages = [page for page, _ in page_scores[:20]]
    
    # If no candidates found, take 10 most recently updated pages
    if not candidate_pages:
        recent_pages = sorted(self.space_pages, 
                             key=lambda p: p.get("history", {}).get("lastUpdated", "2000-01-01"),
                             reverse=True)[:10]
        candidate_pages = recent_pages
    
    # Create dictionary for easy lookup
    page_dict = {p["id"]: p for p in candidate_pages}
    
    # Fetch content for candidates
    page_contents = self._fetch_page_content_batch(candidate_pages)
    
    # Step 2: Detailed relevance scoring with content
    scored_pages = []
    for page_id, content in page_contents.items():
        page = page_dict[page_id]
        page_text = content["content"].lower()
        title = page["title"].lower()
        
        # Calculate a relevance score
        score = 0
        
        # Score based on query word frequency
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                word_count = page_text.count(word)
                score += word_count * 0.1  # Base score per occurrence
                
                # Higher score for words in title
                if word in title:
                    score += 5
        
        # Bonus for exact phrase matches
        if query_lower in page_text:
            score += 50  # Huge bonus for exact match
        else:
            # Check phrases
            query_phrases = []
            for phrase_len in range(2, 5):
                if len(query_words) >= phrase_len:
                    for i in range(len(query_words) - phrase_len + 1):
                        phrase = " ".join(query_words[i:i+phrase_len])
                        query_phrases.append(phrase)
            
            for phrase in query_phrases:
                if phrase in page_text:
                    score += 3 * page_text.count(phrase)
        
        # Bonus for tables if query suggests data interest
        table_terms = {"table", "column", "field", "value", "schema", "mapping"}
        if any(term in query_lower for term in table_terms) and "TABLE:" in content["content"]:
            table_count = content["content"].count("TABLE:")
            score += table_count * 7  # Bonus for each table
        
        # Bonus for code examples if query suggests implementation interest
        code_terms = {"code", "example", "implementation", "syntax", "usage"}
        if any(term in query_lower for term in code_terms) and "```" in content["content"]:
            code_count = content["content"].count("```") // 2  # Each block has opening and closing
            score += code_count * 5  # Bonus for each code block
        
        # Check for relevant image descriptions
        image_terms = {"image", "diagram", "screenshot", "picture"}
        if any(term in query_lower for term in image_terms) and "[IMAGE:" in content["content"]:
            image_count = content["content"].count("[IMAGE:")
            score += image_count * 3  # Bonus for each image
        
        scored_pages.append((page, content, score))
    
    # Sort by score and take top results
    scored_pages.sort(key=lambda x: x[2], reverse=True)
    top_pages = scored_pages[:6]  # Take top 6 most relevant pages
    
    logger.info(f"Selected {len(top_pages)} most relevant pages")
    
    if not top_pages:
        return "I couldn't find any relevant information in the Confluence space."
    
    # Step 3: Extract relevant sections from top pages
    relevant_content = []
    
    for page, content, score in top_pages:
        page_content = content["content"]
        page_url = content["metadata"]["url"]
        
        # Split content into sections for more targeted extraction
        sections = re.split(r"#{1,6}\s", page_content)
        
        # If not many headings, use paragraphs
        if len(sections) <= 3:
            sections = page_content.split("\n\n")
        
        # Score each section
        section_scores = []
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            section_lower = section.lower()
            section_score = 0
            
            # Score based on query terms
            for word in query_words:
                if len(word) > 2:  # Skip very short words
                    freq = section_lower.count(word)
                    section_score += freq * 0.3
            
            # Extra points for exact phrase matches
            if query_lower in section_lower:
                section_score += 10
            else:
                # Check phrases
                for phrase_len in range(2, 5):
                    if len(query_words) >= phrase_len:
                        for i in range(len(query_words) - phrase_len + 1):
                            phrase = " ".join(query_words[i:i+phrase_len])
                            if phrase in section_lower:
                                section_score += 3
            
            # Special handling for tables and code
            if "TABLE:" in section:
                section_score *= 1.5  # Tables are usually highly relevant
            if "```" in section:
                section_score *= 1.3  # Code examples are valuable
            
            section_scores.append((i, section, section_score))
        
        # Get top scoring sections (up to 3 from each page)
        section_scores.sort(key=lambda s: s[2], reverse=True)
        top_sections = section_scores[:3]
        
        # Order sections by their original position in the document
        ordered_sections = sorted(top_sections, key=lambda x: x[0])
        
        if ordered_sections:
            content_block = f"--- FROM: {page['title']} ---\n\n"
            
            # If first section doesn't start with a heading, add the page title as heading
            first_section = ordered_sections[0][1].strip()
            if not first_section.startswith("#"):
                content_block += f"# {page['title']}\n\n"
            
            for _, section, _ in ordered_sections:
                # Clean up the section
                cleaned_section = re.sub(r'\s{2,}', '\n\n', section.strip())
                content_block += cleaned_section + "\n\n"
            
            content_block += f"Source: {page_url}\n"
            relevant_content.append(content_block)
    
    # Combine relevant content from all pages
    return "\n\n" + "\n\n".join(relevant_content)

def _fetch_page_content_batch(self, pages):
    """Fetch content for a batch of pages."""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_page = {}
        for page in pages:
            page_id = page["id"]
            # Check cache first
            if page_id in self.page_content_cache:
                results[page_id] = self.page_content_cache[page_id]
            else:
                future_to_page[executor.submit(self.confluence.get_page_content, page_id)] = page_id
        
        for future in concurrent.futures.as_completed(future_to_page):
            page_id = future_to_page[future]
            try:
                content = future.result()
                if content:
                    results[page_id] = content
                    # Cache the result
                    self.page_content_cache[page_id] = content
            except Exception as e:
                logger.error(f"Error fetching content for page {page_id}: {str(e)}")
    
    return results

def find_view_definition(self, view_name):
    """
    Find the SQL definition for a specified view.
    
    Args:
        view_name: The name of the view to search for
        
    Returns:
        String containing the view definition or None if not found
    """
    logger.info(f"Searching for view definition: {view_name}")
    
    # Normalize view name for comparison (handle VW_ prefix, case insensitivity)
    normalized_name = view_name.lower()
    if not normalized_name.startswith("vw_"):
        normalized_name_alt = f"vw_{normalized_name}"
    else:
        normalized_name_alt = normalized_name[3:]  # Remove vw_ prefix
    
    # Search page titles and content for view definition
    for page in self.space_pages:
        page_title = page.get("title", "").lower()
        if normalized_name in page_title or normalized_name_alt in page_title:
            # Likely match, fetch the content
            content = self.confluence.get_page_content(page["id"])
            if content:
                # Look for view definition in the content
                view_def_match = re.search(
                    r'CREATE\s+OR\s+REPLACE\s+(?:FORCE\s+)?VIEW\s+(\w+)(?:\s*\(.*?\))?\s+AS[\s\S]*?;',
                    content["content"],
                    re.IGNORECASE | re.DOTALL
                )
                if view_def_match:
                    return view_def_match.group(0)
    
    # If not found in titles, search the content of pages with "VIEW" in the title
    view_pages = [p for p in self.space_pages if "VIEW" in p.get("title", "").upper()]
    page_contents = self._fetch_page_content_batch(view_pages[:20])  # Limit to 20 pages for efficiency
    
    for page_id, content in page_contents.items():
        # Search for "CREATE OR REPLACE VIEW view_name" pattern
        view_def_patterns = [
            rf'CREATE\s+OR\s+REPLACE\s+(?:FORCE\s+)?VIEW\s+{re.escape(normalized_name)}(?:\s*\(.*?\))?\s+AS[\s\S]*?;',
            rf'CREATE\s+OR\s+REPLACE\s+(?:FORCE\s+)?VIEW\s+{re.escape(normalized_name_alt)}(?:\s*\(.*?\))?\s+AS[\s\S]*?;',
            rf'VIEW\s+NAME[:\s]+{re.escape(normalized_name)}[\s\S]*?CREATE\s+OR\s+REPLACE\s+(?:FORCE\s+)?VIEW[\s\S]*?;',
            rf'VIEW\s+NAME[:\s]+{re.escape(normalized_name_alt)}[\s\S]*?CREATE\s+OR\s+REPLACE\s+(?:FORCE\s+)?VIEW[\s\S]*?;'
        ]
        
        for pattern in view_def_patterns:
            view_def_match = re.search(pattern, content["content"], re.IGNORECASE | re.DOTALL)
            if view_def_match:
                return view_def_match.group(0)
    
    # Not found
    return None

def generate_api_mapping(self, view_name):
    """
    Generate API mapping for a specified view.
    
    Args:
        view_name: The name of the view to map
        
    Returns:
        Dict containing the mapping details and explanation
    """
    logger.info(f"Generating API mapping for view: {view_name}")
    
    # Find the view definition
    view_def = self.find_view_definition(view_name)
    
    if not view_def:
        return {
            "success": False,
            "message": f"Could not find the definition for view {view_name} in the documentation."
        }
    
    # Parse the view definition
    parsed_view = SqlViewParser.parse_view_definition(view_def)
    
    # Determine the primary domain
    primary_domain = DomainClassifier.classify_view(parsed_view)
    
    # Generate the API JSON
    api_json = ApiMappingGenerator.generate_api_json(parsed_view, primary_domain)
    
    # Generate explanation
    explanation = self.gemini.generate_api_mapping_explanation(view_def, api_json)
    
    return {
        "success": True,
        "view_name": view_name,
        "view_definition": view_def,
        "parsed_structure": parsed_view,
        "primary_domain": primary_domain,
        "api_json": api_json,
        "explanation": explanation
    }

def answer_question(self, question):
    """
    Answer a question using Confluence content and Gemini.
    
    Args:
        question: The user's question
        
    Returns:
        The generated answer
    """
    logger.info(f"Processing question: {question}")
    
    # Check if this is a view mapping request
    view_mapping_pattern = r'(?:generate|create|provide|what is).*(?:mapping|json|api request|api body).*(?:for|of)\s+(?:view|the view)?\s*(\w+)'
    view_match = re.search(view_mapping_pattern, question.lower())
    
    if view_match:
        view_name = view_match.group(1)
        logger.info(f"Identified view mapping request for view: {view_name}")
        
        # Generate the API mapping
        mapping_result = self.generate_api_mapping(view_name)
        
        if mapping_result["success"]:
            # Return a formatted response with the mapping and explanation
            api_json_str = json.dumps(mapping_result["api_json"], indent=2)
            
            response = f"""
API Mapping for {mapping_result['view_name']}
I've generated the API request mapping for the view {mapping_result['view_name']}. Here's the JSON request body:
json{api_json_str}
Explanation
{mapping_result['explanation']}
You can use this JSON as the request body when calling the CoPPER API endpoint for the {mapping_result['primary_domain']} domain.
"""
return response
else:
return mapping_result["message"]
    # Check if this is a view analysis request
    view_analysis_pattern = r'(?:analyze|explain|describe|what does).*(?:view|the view)?\s*(\w+)'
    analysis_match = re.search(view_analysis_pattern, question.lower())
    
    if analysis_match:
        view_name = analysis_match.group(1)
        logger.info(f"Identified view analysis request for view: {view_name}")
        
        # Find the view definition
        view_def = self.find_view_definition(view_name)
        
        if view_def:
            # Analyze the view
            analysis = self.gemini.analyze_view(view_def)
            
            response = f"""
Analysis of View {view_name}
{analysis}
If you'd like to see the API mapping for this view, just ask!
"""
return response
else:
return f"I couldn't find the definition for view {view_name} in the documentation."
    # For other types of questions, extract relevant content and generate response
    relevant_content = self.extract_relevant_content(question)
    
    # Generate response using Gemini
    response = self.gemini.generate_response(question, relevant_content)
    
    return response
==================== MAIN ENTRY POINT ====================
def main():
"""Main entry point for the CoPPER Assistant."""
logger.info("Starting COPPER Assistant")
# Check for required environment variables or use defaults for demo
confluence_username = os.environ.get("CONFLUENCE_USERNAME")
confluence_api_token = os.environ.get("CONFLUENCE_API_TOKEN")
confluence_url = os.environ.get("CONFLUENCE_URL")
confluence_space = os.environ.get("CONFLUENCE_SPACE", "COPPER")

if not confluence_username or not confluence_api_token or not confluence_url:
    print("WARNING: Missing Confluence credentials in environment variables.")
    print("For a real deployment, please set CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN, and CONFLUENCE_URL.")
    
    # Ask for credentials if not in environment
    print("\nEnter Confluence credentials:")
    confluence_username = confluence_username or input("Username/Email: ")
    confluence_api_token = confluence_api_token or input("API Token: ")
    confluence_url = confluence_url or input("Confluence URL: ")

print("\nInitializing COPPER Assistant...")
print("Connecting to Confluence and loading knowledge base...")

# Initialize the assistant
assistant = CopperAssistant(
    confluence_url,
    confluence_username,
    confluence_api_token,
    space_key=confluence_space
)

if not assistant.initialize():
    logger.error("Failed to initialize COPPER Assistant. Please check the logs for details.")
    print("Error: Failed to initialize. Please check the logs for details.")
    return

print(f"\n===== COPPER Database-to-API Mapping Assistant =====")
print(f"Loaded information from {len(assistant.space_pages)} pages in the {assistant.space_key} space.")
print("I can answer questions about CoPPER database views and how to map them to REST APIs.")
print("What would you like to know about CoPPER views or APIs?")
print("Type 'quit' or 'exit' to end the session.\n")

while True:
    try:
        user_input = input("\nQuestion: ").strip()
        
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Thanks for using the COPPER Assistant. Have a great day!")
            break
        
        if not user_input:
            continue
        
        print("\nLooking that up for you...")
        start_time = time.time()
        answer = assistant.answer_question(user_input)
        end_time = time.time()
        
        print(f"\nAnswer (found in {end_time - start_time:.2f} seconds):")
        print("-------")
        print(answer)
        print("-------")
        
    except KeyboardInterrupt:
        print("\nGoodbye! Feel free to come back if you have more questions.")
        break
        
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        print(f"Sorry, I ran into an issue: {str(e)}. Let's try a different question.")
Execute the main function if the script is run directly
if name == "main":
main()
























"""
CoPPER Assistant - CME BLR Hackathon 2025

A comprehensive solution for automating the mapping between CoPPER database
views and REST API endpoints. This integrated solution uses Gemini/Vertex AI
to provide intelligent assistance for CoPPER documentation, with a focus on
view-to-API mapping.

Key features:
- Advanced Confluence content extraction (including iframes, tables, images)
- SQL view definition parsing and analysis
- Automated generation of API request JSON bodies
- Natural language explanations of mappings and CoPPER concepts
- Smart content relevance scoring for accurate information retrieval

Author: AI Assistant + Human Collaboration
Date: April 2025
"""

import os
import re
import json
import time
import logging
import threading
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Any, Union
from functools import lru_cache
import requests
from bs4 import BeautifulSoup, Tag
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# ==================== CONFIGURATION ====================

# Confluence credentials and settings (hardcoded for hackathon)
CONFLUENCE_URL = "https://cmegroup.atlassian.net"
CONFLUENCE_USERNAME = "hackathon_user@cme.com"
CONFLUENCE_API_TOKEN = "atlassian_api_token_123456"
CONFLUENCE_SPACE = "CMEIN"

# Gemini AI settings
MODEL_NAME = "gemini-1.5-pro"
PROJECT_ID = "cme-hackathon-project"
LOCATION = "us-central1"

# General settings
MAX_WORKERS = 4
CACHE_SIZE = 200
MAX_CONTENT_SIZE = 50000  # Maximum content size for Gemini input
CACHE_DIR = ".cache"
PAGE_CACHE_FILE = os.path.join(CACHE_DIR, "page_cache.json")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("copper_assistant.log")
    ]
)

logger = logging.getLogger("CopperAssistant")

# System prompts for Gemini
SYSTEM_PROMPTS = {
    "general": """
You are the friendly CoPPER Assistant, an expert on mapping database views to REST APIs.

Your personality:
- Conversational and approachable - use a casual, helpful tone while maintaining workplace professionalism
- Explain technical concepts in plain language, as if speaking to a colleague
- Use simple analogies and examples to clarify complex ideas
- Add occasional light humor where appropriate to make the conversation engaging
- Be concise but thorough - focus on answering the question directly first, then add helpful context

Your expertise:
- Deep knowledge of the CoPPER database system, its views, and corresponding API endpoints
- Understanding database-to-API mapping patterns and best practices
- Awareness of how applications integrate with CoPPER's REST APIs
- Expert in interpreting table structures, field mappings, and API parameters

When answering:
1. Directly address the user's question first
2. Provide practical, actionable information when possible
3. Format tables and structured data clearly to enhance readability
4. Use bullet points or numbered lists for steps or multiple items
5. Reference specific examples from the documentation when available
6. Acknowledge any limitations in the available information

Remember to maintain a balance between being friendly and professional - you're a helpful colleague, not a formal technical document.
""",
    "view_analysis": """
You are a database view analysis expert. Your task is to analyze CoPPER SQL view definitions and extract meaningful information about their structure, purpose, and usage patterns.

Focus on:
1. Identifying the key columns and their significance
2. Understanding join relationships between tables
3. Recognizing filtering conditions and their business logic
4. Detecting data transformations and calculations
5. Inferring the overall purpose of the view

Provide your analysis in a structured format that highlights:
- Primary entities and their relationships
- Key business rules implemented in the view
- Potential use cases for this data
- Any performance considerations
""",
    "api_mapping": """
You are an API mapping specialist. Your task is to convert CoPPER database view definitions into equivalent REST API request JSON bodies.

When creating mappings:
1. Establish proper endpoint identification based on view name
2. Map view columns to API fields correctly
3. Convert SQL WHERE conditions to API request parameters
4. Transform SQL joins into proper API relationship structures
5. Handle special cases like NULL checks, type conversions, and date formatting

Your output should be valid JSON formatted for CoPPER APIs, with each request parameter properly structured with:
- Appropriate tag name
- Correct operation type (EQ, IN, GT, LT, etc.)
- Proper value formatting
- Correct predicate relationships (AND, OR)

Include explanatory comments for complex mappings to explain your reasoning.
""",
    "qa": """
You are a CoPPER documentation expert who specializes in answering user questions. Your answers should be:

1. Direct and to the point, addressing the specific question asked
2. Based on factual information from the CoPPER documentation
3. Formatted for clarity with examples where appropriate
4. Educational, helping the user understand underlying concepts
5. Free of speculation - if information is not available, acknowledge this fact

For technical questions, include code examples, relevant API calls, or SQL snippets as appropriate.
For conceptual questions, use analogies and clear explanations.
"""
}

# ==================== CONTENT EXTRACTOR ====================

class ContentExtractor:
    """Enhanced content extraction with support for iframes, complex tables, and images."""
    
    @staticmethod
    def extract_content_from_html(html_content, title=""):
        """
        Extract and structure content from HTML, handling complex elements.
        
        Args:
            html_content: The HTML content to process
            title: The title of the page
        
        Returns:
            Dict containing structured content including text, tables, images, and code blocks
        """
        try:
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Track all extracted content
            text_content = []
            tables = []
            images = []
            code_blocks = []
            structured_content = []
            
            # Extract title if not provided
            if not title and soup.title:
                title = soup.title.text.strip()
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Process paragraphs and list items
            for p in soup.find_all(['p', 'li']):
                if p.text.strip():
                    text_content.append(p.text.strip())
            
            # Extract tables with enhanced processing
            ContentExtractor._extract_tables(soup, tables)
            
            # Extract images with context
            ContentExtractor._extract_images(soup, images)
            
            # Extract code blocks with improved formatting
            ContentExtractor._extract_code_blocks(soup, code_blocks)
            
            # Extract content from iframes
            ContentExtractor._extract_iframe_content(soup, text_content, tables, images)
            
            # Extract any important structured content
            ContentExtractor._extract_structured_content(soup, structured_content)
            
            return {
                "text": "\n\n".join(text_content),
                "tables": tables,
                "images": images,
                "code_blocks": code_blocks,
                "structured_content": structured_content
            }
        
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            # Return minimal structure with error message
            return {
                "text": f"Error extracting content: {str(e)}",
                "tables": [],
                "images": [],
                "code_blocks": [],
                "structured_content": []
            }
    
    @staticmethod
    def _extract_tables(soup, tables):
        """Extract tables with enhanced processing for complex Confluence tables."""
        # Handle standard tables
        standard_tables = soup.find_all('table')
        
        # Handle Confluence-specific table wrappers
        confluence_tables = soup.find_all('div', class_='table-wrap')
        
        all_tables = []
        # Process standard tables
        for table in standard_tables:
            # Skip if table is inside a already-processed table-wrap
            if table.find_parent('div', class_='table-wrap'):
                continue
            all_tables.append(table)
        
        # Add Confluence tables
        for table_wrap in confluence_tables:
            table = table_wrap.find('table')
            if table:
                all_tables.append(table)
        
        # Process all tables
        for table in all_tables:
            # Extract table data
            table_data = []
            
            # Get table title/caption if available
            caption = table.find('caption')
            table_title = caption.text.strip() if caption else f"Table {len(tables)+1}"
            
            # Get headers
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
                    # Check if it looks like a header row (has th elements or all cells look like headers)
                    first_row_cells = first_row.find_all(['th', 'td'])
                    if first_row.find('th') or all(cell.name == 'th' and cell.get('class') and 'header' in '.'.join(cell.get('class', [])) for cell in first_row_cells):
                        headers = [th.text.strip() for th in first_row_cells]
            
            # Process rows
            rows = []
            tbody = table.find('tbody')
            if tbody:
                for tr in tbody.find_all('tr'):
                    row = [td.text.strip() for td in tr.find_all(['td', 'th'])]
                    if any(cell for cell in row):  # Skip empty rows
                        rows.append(row)
            else:
                # If no tbody, process all rows (skipping the header if we extracted it)
                all_rows = table.find_all('tr')
                start_idx = 1 if headers and len(all_rows) > 0 else 0
                for tr in all_rows[start_idx:]:
                    row = [td.text.strip() for td in tr.find_all(['td', 'th'])]
                    if any(cell for cell in row):  # Skip empty rows
                        rows.append(row)
            
            # Convert table to text representation with improved formatting
            table_text = ["[TABLE: {table_title}]"]
            
            # Format with consistent column widths for better readability
            if headers and rows:
                # Calculate column widths
                col_widths = [len(h) for h in headers]
                for row in rows:
                    for i, cell in enumerate(row[:len(col_widths)]):
                        col_widths[i] = max(col_widths[i], len(cell))
                
                # Format header row
                header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
                separator = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
                table_text.append(header_row)
                table_text.append(separator)
                
                # Format data rows
                for row in rows:
                    # Pad row if needed to match header length
                    if len(row) < len(headers):
                        row.extend([""] * (len(headers) - len(row)))
                    # Truncate if longer than headers
                    row = row[:len(headers)]
                    row_text = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
                    table_text.append(row_text)
            elif rows:  # Table with no headers
                # Calculate column widths
                max_cols = max(len(row) for row in rows)
                col_widths = [0] * max_cols
                for row in rows:
                    for i, cell in enumerate(row[:max_cols]):
                        col_widths[i] = max(col_widths[i], len(cell))
                
                # Format rows
                for row in rows:
                    # Pad row if needed to match max columns
                    if len(row) < max_cols:
                        row.extend([""] * (max_cols - len(row)))
                    row_text = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
                    table_text.append(row_text)
            
            if len(table_text) > 1:  # Only add tables with content
                tables.append("\n".join(table_text))
    
    @staticmethod
    def _extract_images(soup, images):
        """Extract images with improved context."""
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
                if prev_elem and len(prev_elem) > 0 and len(prev_elem[0].text.strip()) < 200:  # Only short contexts
                    context = f"Previous content: {prev_elem[0].text.strip()}"
            
            # Construct meaningful image description
            desc = alt_text or title or "Image"
            if context:
                desc += f" - {context}"
            
            images.append(f"[IMAGE: {desc}]")
    
    @staticmethod
    def _extract_code_blocks(soup, code_blocks):
        """Extract code blocks with improved formatting."""
        for pre in soup.find_all('pre'):
            code = pre.find('code')
            if code:
                # Check for any language specification
                code_class = code.get('class', [])
                lang = ""
                for cls in code_class:
                    if cls.startswith('language-'):
                        lang = cls.replace('language-', '')
                        break
                
                code_content = code.text.strip()
                if lang:
                    code_blocks.append(f"```{lang}\n{code_content}\n```")
                else:
                    code_blocks.append(f"```\n{code_content}\n```")
            else:
                # Pre without code tag
                code_blocks.append(f"```\n{pre.text.strip()}\n```")
    
    @staticmethod
    def _extract_iframe_content(soup, text_content, tables, images):
        """Extract content from iframes."""
        for iframe in soup.find_all('iframe'):
            iframe_src = iframe.get('src', '')
            if iframe_src:
                # Add reference to iframe content
                text_content.append(f"[IFRAME CONTENT: {iframe_src}]")
                
                # For Gliffy diagrams, extract available metadata
                if 'gliffy' in iframe_src.lower():
                    # Extract diagram title if possible
                    diagram_title = iframe.get('title', '') or iframe.find_parent('div', class_='ap-container').get('data-macro-name', '') if iframe.find_parent('div', class_='ap-container') else ''
                    if diagram_title:
                        images.append(f"[DIAGRAM: {diagram_title}]")
    
    @staticmethod
    def _extract_structured_content(soup, structured_content):
        """Extract any important structured content."""
        for div in soup.find_all(['div', 'section']):
            if 'class' in div.attrs:
                # Look for common Confluence structured content classes
                class_str = ' '.join(div['class'])
                if any(term in class_str for term in ['panel', 'info', 'note', 'warning', 'callout', 'aui-message']):
                    title_elem = div.find(['h3', 'h4', 'h5', 'strong', 'b'])
                    title = title_elem.text.strip() if title_elem else "Note"
                    content = div.text.strip()
                    structured_content.append(f"--- {title} ---\n{content}")

    @staticmethod
    def format_for_context(extracted_content, title=""):
        """
        Format the extracted content for use as context.
        
        Args:
            extracted_content: The dictionary of extracted content
            title: The title of the page
            
        Returns:
            Formatted string containing all the content
        """
        sections = []
        
        if title:
            sections.append(f"# {title}")
        
        if extracted_content.get("text"):
            sections.append(extracted_content["text"])
        
        if extracted_content.get("tables"):
            for table in extracted_content["tables"]:
                sections.append(f"\n{table}")
        
        if extracted_content.get("code_blocks"):
            sections.append("\nCode Examples:")
            sections.extend(extracted_content["code_blocks"])
        
        if extracted_content.get("structured_content"):
            sections.append("\nImportant Notes:")
            sections.extend(extracted_content["structured_content"])
        
        if extracted_content.get("images"):
            sections.append("\nImage Information:")
            sections.extend(extracted_content["images"])
        
        return "\n\n".join(sections)


# ==================== CONFLUENCE INTEGRATION ====================

class ConfluenceClient:
    """Client for Confluence REST API operations with comprehensive error handling and caching."""
    
    def __init__(self, base_url, username, api_token):
        """
        Initialize the Confluence client with authentication details.
        
        Args:
            base_url: The base URL of the Confluence instance (e.g., https://mycompany.atlassian.net)
            username: The username for authentication
            api_token: The API token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.api_url = f"{self.base_url}/wiki/rest/api"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "COPPER-AI-Python-Agent"
        }
        self.session = requests.Session()
        # Set up a default request timeout
        self.timeout = 30
        # Cache for API responses
        self.cache = {}
        # Thread lock for the cache
        self.cache_lock = threading.Lock()
        
        logger.info(f"Initialized Confluence client for {self.base_url}")
    
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            response = self.session.get(
                f"{self.api_url}/space",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                timeout=self.timeout,
                verify=False  # Using verify=False as requested
            )
            response.raise_for_status()
            
            if response.status_code == 200:
                logger.info("Connection to Confluence successful!")
                return True
            else:
                logger.warning("Empty response received during connection test")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    @lru_cache(maxsize=CACHE_SIZE)
    def get_cached_request(self, url, params_str):
        """Cached version of GET requests to reduce API calls."""
        try:
            params = json.loads(params_str)
            response = self.session.get(
                url,
                auth=self.auth,
                headers=self.headers,
                params=params,
                timeout=self.timeout,
                verify=False  # Using verify=False as requested
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error in cached request to {url}: {str(e)}")
            return None
    
    def get_all_pages_in_space(self, space_key, batch_size=100):
        """
        Get all pages in a Confluence space using efficient pagination.
        
        Args:
            space_key: The space key to get all pages from
            batch_size: Number of results per request (max 100)
            
        Returns:
            List of page objects with basic information
        """
        logger.info(f"Fetching all pages from space: {space_key}")
        
        all_pages = []
        start = 0
        has_more = True
        
        # Check if we have cached results
        cache_path = PAGE_CACHE_FILE
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    if cached_data.get('space_key') == space_key:
                        logger.info(f"Using cached page list from {cache_path}")
                        return cached_data.get('pages', [])
            except Exception as e:
                logger.warning(f"Error reading cache file: {str(e)}")
        
        # If no cache, fetch all pages
        while has_more:
            logger.info(f"Fetching pages batch from start={start}")
            
            try:
                params = {
                    "spaceKey": space_key,
                    "expand": "history",  # Include basic history info to get last updated date
                    "limit": batch_size,
                    "start": start
                }
                
                # Convert params to string for cache key
                params_str = json.dumps(params, sort_keys=True)
                
                # Try to get from cache first
                response_text = self.get_cached_request(f"{self.api_url}/content", params_str)
                
                if not response_text:
                    logger.warning(f"Empty response when fetching pages at start={start}")
                    break
                
                response_data = json.loads(response_text)
                
                results = response_data.get("results", [])
                all_pages.extend(results)
                
                # Check if there are more pages
                if "size" in response_data and "limit" in response_data:
                    if response_data["size"] < response_data["limit"]:
                        has_more = False
                    else:
                        start += batch_size
                else:
                    has_more = False
                
                logger.info(f"Fetched {len(results)} pages, total so far: {len(all_pages)}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching pages: {str(e)}")
                break
        
        # Cache the results
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump({'space_key': space_key, 'pages': all_pages}, f)
                logger.info(f"Cached {len(all_pages)} pages to {cache_path}")
        except Exception as e:
            logger.warning(f"Error writing cache file: {str(e)}")
        
        logger.info(f"Successfully fetched {len(all_pages)} pages from space {space_key}")
        return all_pages
    
    def get_page_content(self, page_id, expand="body.storage,metadata.labels"):
        """
        Get the content of a page in a suitable format for NLP.
        This extracts and processes the content to be more suitable for embeddings.
        
        Args:
            page_id: The ID of the page
            expand: What to expand in the API request
            
        Returns:
            Dict containing the processed content
        """
        try:
            # Use cached version if available
            cache_key = f"page_content_{page_id}"
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]
            
            page = self.get_content_by_id(page_id, expand=f"body.storage,metadata.labels")
            if not page:
                return None
            
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "url": f"{self.base_url}/pages/viewpage.action?pageId={page.get('id')}",
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
            }
            
            # Get raw content
            html_content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process with our advanced content extractor
            extracted_content = ContentExtractor.extract_content_from_html(html_content, page.get("title", ""))
            formatted_content = ContentExtractor.format_for_context(extracted_content, page.get("title", ""))
            
            result = {
                "metadata": metadata,
                "content": formatted_content,
                "raw_html": html_content
            }
            
            # Cache the result
            with self.cache_lock:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None
    
    def get_content_by_id(self, content_id, expand=None):
        """
        Get content by ID with optional expansion parameters.
        
        Args:
            content_id: The ID of the content to retrieve
            expand: Optional expansion parameters
            
        Returns:
            Content object or None if not found/error
        """
        params = {}
        if expand:
            params["expand"] = expand
        
        # Convert params to string for cache key
        params_str = json.dumps(params, sort_keys=True)
        
        try:
            # Try to get from cache first
            response_text = self.get_cached_request(f"{self.api_url}/content/{content_id}", params_str)
            
            if not response_text:
                return None
            
            content = json.loads(response_text)
            logger.info(f"Successfully retrieved content: {content.get('title', 'Unknown title')}")
            return content
            
        except Exception as e:
            logger.error(f"Error getting content by ID {content_id}: {str(e)}")
            return None
    
    def fetch_page_content_batch(self, pages):
        """Fetch content for a batch of pages in parallel."""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_page = {
                executor.submit(self.get_page_content, page["id"]): page["id"]
                for page in pages
            }
            
            for future in concurrent.futures.as_completed(future_to_page):
                page_id = future_to_page[future]
                try:
                    content = future.result()
                    if content:
                        results[page_id] = content
                except Exception as e:
                    logger.error(f"Error processing page {page_id}: {str(e)}")
        
        return results


# ==================== SQL VIEW PARSER ====================

class SqlViewParser:
    """
    Advanced SQL View parser for extracting detailed information from view definitions.
    Handles complex SQL syntax and extracts structural information.
    """
    
    @staticmethod
    def parse_view_definition(sql_text):
        """
        Parse a SQL view definition into a structured format.
        
        Args:
            sql_text: The SQL view definition text
            
        Returns:
            Dict containing parsed view information
        """
        view_info = {
            "view_name": "",
            "columns": [],
            "select_clause": "",
            "from_clause": "",
            "where_clause": "",
            "join_conditions": [],
            "filter_conditions": [],
            "case_statements": [],
            "transformations": [],
            "source_tables": []
        }
        
        # Extract view name
        view_name_match = re.search(r'VIEW\s+(\w+)', sql_text, re.IGNORECASE)
        if view_name_match:
            view_info["view_name"] = view_name_match.group(1)
        
        # Handle CREATE OR REPLACE VIEW syntax
        create_view_match = re.search(r'CREATE\s+OR\s+REPLACE\s+(?:FORCE\s+)?VIEW\s+(\w+)(?:\s*\((.*?)\))?\s+AS\s+(.*)', 
                                    sql_text, re.IGNORECASE | re.DOTALL)
        
        if create_view_match:
            view_info["view_name"] = create_view_match.group(1)
            
            # Extract column definitions if provided inline
            if create_view_match.group(2):
                columns_text = create_view_match.group(2)
                # Split by commas that are not inside parentheses
                columns = []
                
                # Simple column extraction (can be improved with regex)
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
                
                view_info["columns"] = columns
            
            # Extract the SELECT statement
            select_stmt = create_view_match.group(3)
            view_info["select_clause"] = select_stmt
            
            # Extract FROM clause
            from_match = re.search(r'FROM\s+(.*?)(?:WHERE|GROUP BY|ORDER BY|$)', select_stmt, re.IGNORECASE | re.DOTALL)
            if from_match:
                view_info["from_clause"] = from_match.group(1).strip()
                
                # Extract source tables from FROM and JOIN clauses
                table_matches = re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', from_match.group(1), re.IGNORECASE)
                source_tables = []
                for match in table_matches:
                    if match[0]:
                        source_tables.append(match[0])
                    elif match[1]:
                        source_tables.append(match[1])
                
                view_info["source_tables"] = source_tables
                
                # Extract JOIN conditions
                join_matches = re.findall(r'((?:INNER|LEFT|RIGHT|FULL|OUTER)?\s*JOIN\s+\w+\s+.*?ON\s+[^;]+?)(?=(?:INNER|LEFT|RIGHT|FULL|OUTER)?\s*JOIN|\s*WHERE|\s*GROUP BY|\s*ORDER BY|$)', 
                                         from_match.group(1), re.IGNORECASE | re.DOTALL)
                
                view_info["join_conditions"] = [j.strip() for j in join_matches]
            
            # Extract WHERE clause
            where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|$)', select_stmt, re.IGNORECASE | re.DOTALL)
            if where_match:
                view_info["where_clause"] = where_match.group(1).strip()
                
                # Parse individual filter conditions
                # This is complex due to nested conditions - a simplified approach
                conditions_text = where_match.group(1)
                
                # Handle nested parentheses in conditions
                parsed_conditions = SqlViewParser._parse_conditions(conditions_text)
                view_info["filter_conditions"] = parsed_conditions
            
            # Extract CASE statements
            case_matches = re.findall(r'(CASE\s+.*?END)', sql_text, re.IGNORECASE | re.DOTALL)
            view_info["case_statements"] = case_matches
            
            # Extract other transformations/functions
            function_matches = re.findall(r'(\w+\s*\([^)]*\))', sql_text)
            view_info["transformations"] = [f for f in function_matches if not any(
                common in f.upper() for common in ['COUNT(', 'SUM(', 'MIN(', 'MAX(', 'AVG('])]
        
        return view_info
    
    @staticmethod
    def _parse_conditions(conditions_text):
        """Parse complex WHERE conditions with nested ANDs and ORs."""
        # Simple parsing for demo - in production would use a proper SQL parser
        # This handles basic AND/OR separation but not nested parentheses properly
        
        # Split on top-level AND/OR
        conditions = []
        
        # Handle simple case first
        if 'AND' not in conditions_text.upper() and 'OR' not in conditions_text.upper():
            return [conditions_text.strip()]
        
        # Otherwise, split by AND/OR preserving parenthesized expressions
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


# ==================== API MAPPING CLASSIFIER ====================

class ApiMappingClassifier:
    """
    Classifier for determining appropriate API mapping patterns based on view characteristics.
    Uses pattern recognition to classify views and select mapping strategies.
    """
    
    @staticmethod
    def classify_view(view_info):
        """
        Classify a view based on its structure and content.
        
        Args:
            view_info: Dict containing parsed view information
            
        Returns:
            Dict with classification information
        """
        classification = {
            "primary_domain": "",
            "api_endpoint": "",
            "query_params": [],
            "response_fields": [],
            "mapping_strategy": "",
            "confidence": 0.0
        }
        
        view_name = view_info.get("view_name", "").upper()
        source_tables = view_info.get("source_tables", [])
        filter_conditions = view_info.get("filter_conditions", [])
        
        # Determine primary domain based on view name and tables
        domain_indicators = {
            "FIRM": ["FIRM", "TRADER", "INSTITUTION"],
            "SESSION": ["SESSION", "TRADER", "GRID"],
            "INSTRUMENT": ["INSTRUMENT", "ERSA", "IPALM"],
            "PRODUCT": ["PRODUCT", "CORE"],
            "RISK": ["RISK", "BTEC"],
            "STATIC": ["STATIC", "REF"]
        }
        
        # Score each domain based on matches in view name and source tables
        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            score = 0
            # Check view name
            for indicator in indicators:
                if indicator in view_name:
                    score += 5
            
            # Check source tables
            for table in source_tables:
                for indicator in indicators:
                    if indicator in table.upper():
                        score += 2
            
            # Check filter conditions
            for condition in filter_conditions:
                for indicator in indicators:
                    if indicator in condition.upper():
                        score += 1
            
            domain_scores[domain] = score
        
        # Select domain with highest score
        if domain_scores:
            top_domain = max(domain_scores.items(), key=lambda x: x[1])
            if top_domain[1] > 0:
                classification["primary_domain"] = top_domain[0].lower()
                classification["confidence"] = min(1.0, top_domain[1] / 10.0)  # Normalize to 0-1
        
        # If no clear domain, try to infer from view name
        if not classification["primary_domain"] and "_" in view_name:
            parts = view_name.split("_")
            if len(parts) > 1:
                prefix = parts[1].lower()
                if prefix in [d.lower() for d in domain_indicators.keys()]:
                    classification["primary_domain"] = prefix
                    classification["confidence"] = 0.5
        
        # Determine API endpoint based on domain
        if classification["primary_domain"]:
            classification["api_endpoint"] = f"/api/{classification['primary_domain'].lower()}"
            
            # Determine mapping strategy based on view characteristics
            if len(filter_conditions) > 3:
                classification["mapping_strategy"] = "complex_filter"
            elif len(view_info.get("join_conditions", [])) > 2:
                classification["mapping_strategy"] = "join_consolidation"
            elif len(view_info.get("case_statements", [])) > 0:
                classification["mapping_strategy"] = "transformation_mapping"
            else:
                classification["mapping_strategy"] = "direct_mapping"
        
        return classification


# ==================== API JSON GENERATOR ====================

class ApiJsonGenerator:
    """
    Generator for CoPPER API JSON request bodies based on view definitions.
    Creates properly structured JSON for querying CoPPER APIs.
    """
    
    @staticmethod
    def generate_api_json(view_info, classification):
        """
        Generate API JSON request body based on view information and classification.
        
        Args:
            view_info: Dict with parsed view information
            classification: Dict with view classification information
            
        Returns:
            Dict containing the API JSON request body
        """
        # Start with base request structure
        request_body = {
            "reqName": classification["primary_domain"] or "data",
            "dataDomain": classification["primary_domain"] or "data",
            "type": "independent",
            "responseInclude": True,
            "req": []
        }
        
        # For dependent domains, adjust the type
        if classification["primary_domain"] in ["session", "instruments"]:
            dependent_domains = {
                "session": {
                    "reqName": "session",
                    "dataDomain": "sessions",
                    "type": "dependent", 
                    "distinct": True
                },
                "instruments": {
                    "reqName": "instruments",
                    "dataDomain": "instruments",
                    "type": "dependent",
                    "distinct": True
                }
            }
            
            domain_info = dependent_domains.get(classification["primary_domain"])
            if domain_info:
                for key, value in domain_info.items():
                    request_body[key] = value
        
        # Process WHERE conditions to generate request parameters
        if view_info.get("filter_conditions"):
            for condition in view_info["filter_conditions"]:
                param = ApiJsonGenerator._condition_to_param(condition)
                if param:
                    request_body["req"].append(param)
        
        # Add domain-specific default parameters
        ApiJsonGenerator._add_domain_defaults(request_body, classification["primary_domain"])
        
        # Add fields based on view columns
        field_list = []
        for column in view_info.get("columns", []):
            # Extract just the column name without table prefixes or aliases
            if " AS " in column.upper():
                # Column has an alias
                col_name = column.split(" AS ", 1)[1].strip()
            else:
                # No alias, just extract the base name
                col_parts = column.split(".")
                col_name = col_parts[-1].strip()
            
            # Convert to camelCase for API fields
            if "_" in col_name:
                parts = col_name.lower().split("_")
                camel_case = parts[0] + "".join(p.capitalize() for p in parts[1:])
                field_list.append(camel_case)
            else:
                field_list.append(col_name.lower())
        
        if field_list:
            request_body["fields"] = field_list
        
        return request_body
    
    @staticmethod
    def _condition_to_param(condition):
        """Convert a SQL WHERE condition to an API request parameter."""
        # Map of common SQL operators to API operations
        op_map = {
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
            "LIKE": "LIKE"
        }
        
        condition = condition.strip()
        
        # Handle IS NULL / IS NOT NULL
        if "IS NULL" in condition.upper():
            field = condition.split("IS NULL", 1)[0].strip()
            field = field.split(".")[-1].lower()  # Get just the column name
            return {
                "tag": field,
                "operation": "ISNULL"
            }
        
        if "IS NOT NULL" in condition.upper():
            field = condition.split("IS NOT NULL", 1)[0].strip()
            field = field.split(".")[-1].lower()  # Get just the column name
            return {
                "tag": field,
                "operation": "NOTNULL"
            }
        
        # Handle IN conditions
        in_match = re.search(r'(\w+(?:\.\w+)?)\s+IN\s*\((.*?)\)', condition, re.IGNORECASE | re.DOTALL)
        if in_match:
            field = in_match.group(1).split(".")[-1].lower()
            values_text = in_match.group(2)
            values = []
            
            # Parse the values, handling quoted strings
            for val in re.findall(r'\'([^\']*?)\'|"([^"]*?)"|(\w+)', values_text):
                value = val[0] or val[1] or val[2]
                if value and not value.upper() in ['AND', 'OR', 'IN']:
                    values.append(value)
            
            return {
                "tag": field,
                "value": values,
                "operation": "IN"
            }
        
        # Handle other operators
        for op in op_map.keys():
            if f" {op} " in f" {condition} ":  # Add spaces to ensure we match whole operators
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    field = parts[0].strip()
                    field = field.split(".")[-1].lower()  # Get just the column name
                    
                    value = parts[1].strip()
                    # Remove quotes if present
                    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
                        value = value[1:-1]
                    
                    return {
                        "tag": field,
                        "value": value,
                        "operation": op_map[op]
                    }
        
        return None
    
    @staticmethod
    def _add_domain_defaults(request_body, domain):
        """Add domain-specific default parameters."""
        defaults = {
            "firm": [
                {"tag": "effEndTimp", "value": "TODAY", "operation": "GT"}
            ],
            "session": [
                {"tag": "sessionFeature.features", "value": ["BTEC-EU", "BTEC-US"], "operation": "IN"}
            ],
            "instruments": [
                {"tag": "prodTyp", "value": "BOND", "operation": "EQ"}
            ],
            "product": [
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
                for req in request_body["req"]:
                    if req.get("tag") == param["tag"]:
                        existing = True
                        break
                
                if not existing:
                    request_body["req"].append(param)


# ==================== RELEVANCE SERVICE ====================

class RelevanceService:
    """
    Smart content relevance scoring and search service.
    Finds the most relevant information for user queries.
    """
    
    @staticmethod
    def extract_relevant_content(query, page_contents):
        """
        Extract the most relevant content for a query from a collection of pages.
        
        Args:
            query: The user's question or query
            page_contents: Dict of page contents keyed by page_id
            
        Returns:
            String containing the relevant information
        """
        logger.info(f"Processing question: {query}")
        
        # Step 1: Initial candidate selection
        candidate_pages = []
        
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Look for view definition requests
        view_match = re.search(r'view\s+(\w+)', query_lower)
        if view_match:
            view_name = view_match.group(1)
            logger.info(f"Looking for specific view: {view_name}")
            
            # First, search for exact matches in the title
            for page_id, page in page_contents.items():
                if view_name.lower() in page["metadata"]["title"].lower():
                    candidate_pages.append(page)
            
            if not candidate_pages:
                # If no exact matches, search for pages with "view" in the title
                for page_id, page in page_contents.items():
                    if "view" in page["metadata"]["title"].lower():
                        candidate_pages.append(page)
        else:
            # For general queries, score based on query term presence
            page_scores = []
            for page_id, page in page_contents.items():
                page_score = 0
                page_title = page["metadata"]["title"].lower()
                
                # Score based on words in title
                for word in query_words:
                    if word in page_title:
                        page_score += 2
                
                # Higher score for multi-word matches
                for i in range(len(query_words)-1):
                    if i+1 < len(query_words):
                        two_words = f"{query_words[i]} {query_words[i+1]}"
                        if two_words in page_title:
                            page_score += 3
                
                if page_score > 0:
                    page_scores.append((page, page_score))
            
            # Sort by score and take top 20
            page_scores.sort(key=lambda x: x[1], reverse=True)
            candidate_pages = [page for page, _ in page_scores[:20]]
        
        # If no candidates found, take 10 most recently updated pages
        if not candidate_pages:
            recent_pages = sorted(
                [page for page_id, page in page_contents.items()], 
                key=lambda p: p["metadata"].get("history", {}).get("lastUpdated", "2000-01-01"),
                reverse=True)[:10]
            candidate_pages = recent_pages
        
        # Step 2: Detailed relevance scoring with content
        scored_pages = []
        for page in candidate_pages:
            page_text = page["content"].lower()
            title = page["metadata"]["title"].lower()
            
            # Calculate a relevance score
            score = 0
            
            # Score based on query word frequency
            for word in query_words:
                word_count = page_text.count(word)
                score += word_count * 0.1  # Base score per occurrence
                
                # Higher score for words in title
                if word in title:
                    score += 5
            
            # Bonus for exact phrase matches
            if query_lower in page_text:
                score += 50  # Huge bonus for exact match
            else:
                # Look for phrases (2-4 words)
                for phrase_len in range(2, 5):
                    if len(query_words) >= phrase_len:
                        for i in range(len(query_words) - phrase_len + 1):
                            phrase = " ".join(query_words[i:i+phrase_len])
                            if phrase in page_text:
                                score += 3 * page_text.count(phrase)
            
            # Bonus for tables if query suggests data interest
            table_terms = {"table", "column", "field", "value", "schema", "mapping"}
            if any(term in query_lower for term in table_terms) and "TABLE:" in page["content"]:
                table_count = page["content"].count("TABLE:")
                score += table_count * 7  # Bonus for each table
            
            # Bonus for code examples if query suggests implementation interest
            code_terms = {"code", "example", "implementation", "syntax", "usage"}
            if any(term in query_lower for term in code_terms) and "```" in page["content"]:
                code_count = page["content"].count("```") // 2  # Each block has opening and closing
                score += code_count * 5  # Bonus for each code block
            
            # Check for relevant image descriptions
            image_terms = {"image", "diagram", "screenshot", "picture"}
            if any(term in query_lower for term in image_terms) and "[IMAGE:" in page["content"]:
                image_count = page["content"].count("[IMAGE:")
                score += image_count * 3  # Bonus for each image
            
            scored_pages.append((page, score))
        
        # Sort by score and take top results
        scored_pages.sort(key=lambda x: x[1], reverse=True)
        top_pages = scored_pages[:5]  # Take top 5 most relevant pages
        
        logger.info(f"Selected {len(top_pages)} most relevant pages")
        
        if not top_pages:
            return "I couldn't find any relevant information in the Confluence space."
        
        # Step 3: Extract relevant sections from top pages
        relevant_content = []
        
        for page, score in top_pages:
            page_content = page["content"]
            page_url = page["metadata"]["url"]
            
            # Split content into sections for more targeted extraction
            sections = re.split(r"#{1,6}\s", page_content)
            
            # If not many headings, use paragraphs
            if len(sections) <= 3:
                sections = page_content.split("\n\n")
            
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
                    section_score += freq * 0.3
                
                # Extra points for exact phrase matches
                if query_lower in section_lower:
                    section_score += 10
                else:
                    # Check phrases
                    for phrase_len in range(2, 5):
                        if len(query_words) >= phrase_len:
                            for j in range(len(query_words) - phrase_len + 1):
                                phrase = " ".join(query_words[j:j+phrase_len])
                                if phrase in section_lower:
                                    section_score += 3
                
                # Special handling for tables and code
                if "TABLE:" in section:
                    section_score *= 1.5  # Tables are usually highly relevant
                if "```" in section:
                    section_score *= 1.3  # Code examples are valuable
                
                section_scores.append((i, section, section_score))
            
            # Get top scoring sections (up to 3 from each page)
            section_scores.sort(key=lambda s: s[2], reverse=True)
            top_sections = section_scores[:3]
            
            # Order sections by their original position in the document
            ordered_sections = sorted(top_sections, key=lambda x: x[0])
            
            if ordered_sections:
                content_block = f"--- FROM: {page['metadata']['title']} ---\n\n"
                
                # If first section doesn't start with a heading, add the page title as heading
                first_section = ordered_sections[0][1].strip()
                if not first_section.startswith("#"):
                    content_block += f"# {page['metadata']['title']}\n\n"
                
                for _, section, _ in ordered_sections:
                    # Clean up the section
                    cleaned_section = re.sub(r'\s{2,}', '\n\n', section.strip())
                    content_block += cleaned_section + "\n\n"
                
                content_block += f"Source: {page_url}\n"
                relevant_content.append(content_block)
        
        # Combine relevant content from all pages
        return "\n\n" + "\n\n".join(relevant_content)


# ==================== GEMINI INTEGRATION ====================

class GeminiAssistant:
    """Class for interacting with Gemini models via Vertex AI."""
    
    def __init__(self):
        """Initialize Vertex AI and Gemini model."""
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        self.model = GenerativeModel(MODEL_NAME)
        logger.info(f"Initialized Gemini Assistant with model: {MODEL_NAME}")
    
    def generate_response(self, prompt, copper_context=None):
        """
        Generate a response from Gemini based on the prompt and CoPPER context.
        
        Args:
            prompt: The user's question or prompt
            copper_context: Context information about CoPPER (from Confluence)
            
        Returns:
            The generated response
        """
        logger.info(f"Generating response for prompt: {prompt}")
        
        try:
            # Create a system prompt that instructs Gemini on how to use the context
            system_prompt = SYSTEM_PROMPTS["general"]
            
            # Trim context if it's too large
            if copper_context:
                if len(copper_context) > MAX_CONTENT_SIZE:
                    logger.warning(f"Context too large ({len(copper_context)} chars), trimming...")
                    # Try to trim at paragraph boundaries
                    paragraphs = copper_context.split("\n\n")
                    trimmed_content = ""
                    for para in paragraphs:
                        if len(trimmed_content) + len(para) + 2 < MAX_CONTENT_SIZE:
                            trimmed_content += para + "\n\n"
                        else:
                            break
                    copper_context = trimmed_content
                    logger.info(f"Trimmed context to {len(copper_context)} chars")
                
                full_prompt = f"{system_prompt}\n\nCONTEXT INFORMATION:\n{copper_context}\n\nUSER QUESTION: {prompt}\n\nResponse:"
            else:
                full_prompt = f"{system_prompt}\n\nUSER QUESTION: {prompt}\n\nResponse:"
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=0.3,  # Lower temperature for more factual responses
                top_p=0.95,
            )
            
            # Generate the response
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
            
            if response.candidates and response.candidates[0].text:
                response_text = response.candidates[0].text.strip()
                logger.info(f"Successfully generated response ({len(response_text)} chars)")
                return response_text
            else:
                logger.warning("No response generated from Gemini")
                return "I couldn't find a specific answer to that question in our documentation. Could you try rephrasing, or maybe I can help you find the right documentation to look at?"
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I ran into a technical issue while looking that up. Let me know if you'd like to try a different question or approach."


# ==================== MAIN ASSISTANT CLASS ====================

class CopperAssistant:
    """Main class that coordinates between Confluence and Gemini."""
    
    def __init__(self, confluence_url, confluence_username, confluence_api_token, space_key=None):
        """
        Initialize the Copper Assistant.
        
        Args:
            confluence_url: The base URL of the Confluence instance
            confluence_username: The username for Confluence authentication
            confluence_api_token: The API token for Confluence authentication
            space_key: The space key to target (or 'all' for all spaces)
        """
        self.confluence = ConfluenceClient(confluence_url, confluence_username, confluence_api_token)
        self.gemini = GeminiAssistant()
        self.space_key = space_key
        self.space_pages = []
        self.page_content_cache = {}  # Cache for page content to avoid re-fetching
        
        logger.info(f"Initialized Copper Assistant targeting space: {space_key or 'all spaces'}")
    
    def initialize(self):
        """Initialize by testing connections and gathering initial space content."""
        if not self.confluence.test_connection():
            logger.error("Failed to connect to Confluence. Check credentials and URL.")
            return False
        
        logger.info("Loading space content...")
        self.load_space_content()
        return True
    
    def load_space_content(self):
        """Load metadata for all pages in the specified space."""
        if not self.space_key:
            logger.error("No space key specified. Please provide a space key.")
            return
        
        self.space_pages = self.confluence.get_all_pages_in_space(self.space_key)
        logger.info(f"Loaded metadata for {len(self.space_pages)} pages from space {self.space_key}")
    
    def extract_relevant_content(self, query):
        """
        Extract relevant content based on the user's query.
        
        Args:
            query: The user's question or query
            
        Returns:
            String containing the relevant information
        """
        # Check if we have pages loaded
        if not self.space_pages:
            logger.error("No space pages loaded. Call load_space_content() first.")
            return "I couldn't find any relevant information in our documentation."
        
        # First, fetch content for all pages if not already cached
        if not self.page_content_cache:
            logger.info("Building page content cache...")
            # Start with a small batch of recent pages to speed up initial responses
            recent_pages = sorted(
                self.space_pages, 
                key=lambda p: p.get("history", {}).get("lastUpdated", "2000-01-01"),
                reverse=True)[:50]  # Start with 50 most recent pages
            
            self._fetch_page_content_batch(recent_pages)
            logger.info(f"Cached content for {len(self.page_content_cache)} pages")
        
        # Use the relevance service to find relevant content
        page_contents = self.page_content_cache
        return RelevanceService.extract_relevant_content(query, page_contents)
    
    def _fetch_page_content_batch(self, pages):
        """Fetch content for a batch of pages."""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_page = {}
            for page in pages:
                page_id = page["id"]
                # Check cache first
                if page_id in self.page_content_cache:
                    results[page_id] = self.page_content_cache[page_id]
                else:
                    future_to_page[executor.submit(self.confluence.get_page_content, page_id)] = page_id
            
            for future in concurrent.futures.as_completed(future_to_page):
                page_id = future_to_page[future]
                try:
                    content = future.result()
                    if content:
                        results[page_id] = content
                        # Cache the result
                        self.page_content_cache[page_id] = content
                except Exception as e:
                    logger.error(f"Error fetching content for page {page_id}: {str(e)}")
        
        return results
    
    def answer_question(self, question):
        """
        Answer a question using Confluence content and Gemini.
        
        Args:
            question: The user's question
            
        Returns:
            The generated answer
        """
        logger.info(f"Processing question: {question}")
        
        # Extract relevant content based on the question
        relevant_content = self.extract_relevant_content(question)
        
        # Generate response using Gemini
        response = self.gemini.generate_response(question, relevant_content)
        
        return response
    
    def process_view_mapping_request(self, view_name):
        """
        Process a request to map a specific view to an API.
        
        Args:
            view_name: The name of the view to map
            
        Returns:
            Dict containing the mapping details
        """
        logger.info(f"Processing view mapping request for: {view_name}")
        
        # Search for view definition in Confluence
        view_def = self._find_view_definition(view_name)
        
        if not view_def:
            return {
                "success": False,
                "message": f"Could not find the definition for view {view_name} in the documentation."
            }
        
        # Parse the view definition
        parsed_view = SqlViewParser.parse_view_definition(view_def)
        
        # Classify the view
        classification = ApiMappingClassifier.classify_view(parsed_view)
        
        # Generate the API JSON
        api_json = ApiJsonGenerator.generate_api_json(parsed_view, classification)
        
        return {
            "success": True,
            "view_name": view_name,
            "api_endpoint": classification["api_endpoint"],
            "api_request_body": api_json,
            "mapping_confidence": classification["confidence"],
            "mapping_explanation": self._generate_mapping_explanation(parsed_view, classification, api_json)
        }
    
    def _find_view_definition(self, view_name):
        """Find a view definition in the Confluence content."""
        # Search for the view definition in cached page content
        for page_id, page in self.page_content_cache.items():
            # Look for CREATE OR REPLACE VIEW statements with this view name
            content = page["content"]
            view_pattern = rf"CREATE\s+OR\s+REPLACE\s+(?:FORCE\s+)?VIEW\s+{view_name}"
            if re.search(view_pattern, content, re.IGNORECASE):
                # Extract the full view definition
                match = re.search(rf"(CREATE\s+OR\s+REPLACE\s+(?:FORCE\s+)?VIEW\s+{view_name}.*?;)", 
                                 content, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(1)
        
        # If not found, try expanding the search to view names containing the search term
        for page_id, page in self.page_content_cache.items():
            content = page["content"]
            # Look for any view with a similar name
            view_pattern = r"CREATE\s+OR\s+REPLACE\s+(?:FORCE\s+)?VIEW\s+\w*" + view_name + r"\w*"
            match = re.search(view_pattern, content, re.IGNORECASE)
            if match:
                # Get the view name
                view_name_match = re.search(r"VIEW\s+(\w+)", match.group(0), re.IGNORECASE)
                if view_name_match:
                    actual_view_name = view_name_match.group(1)
                    # Extract the full view definition
                    full_match = re.search(rf"(CREATE\s+OR\s+REPLACE\s+(?:FORCE\s+)?VIEW\s+{actual_view_name}.*?;)", 
                                         content, re.IGNORECASE | re.DOTALL)
                    if full_match:
                        return full_match.group(1)
        
        return None
    
    def _generate_mapping_explanation(self, parsed_view, classification, api_json):
        """Generate a human-readable explanation of the view-to-API mapping."""
        explanation = []
        
        # Basic information
        explanation.append(f"## Mapping Explanation for {parsed_view['view_name']}")
        explanation.append(f"\nThis view has been mapped to the {classification['primary_domain']} domain with {classification['confidence']*100:.0f}% confidence.")
        
        # API endpoint
        explanation.append(f"\n### API Endpoint")
        explanation.append(f"The appropriate endpoint for this view is: `{classification['api_endpoint']}`")
        
        # Mapping strategy
        explanation.append(f"\n### Mapping Strategy: {classification['mapping_strategy']}")
        
        strategy_explanations = {
            "direct_mapping": "The view structure maps directly to API parameters without complex transformations.",
            "complex_filter": "The view contains multiple filter conditions that have been converted to API request parameters.",
            "join_consolidation": "The view joins multiple tables, which have been consolidated into a single API request.",
            "transformation_mapping": "The view contains transformations (like CASE statements) that require special handling in the API request."
        }
        
        explanation.append(strategy_explanations.get(classification["mapping_strategy"], ""))
        
        # Request parameters
        explanation.append(f"\n### Request Parameters")
        for param in api_json.get("req", []):
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
        if parsed_view.get("filter_conditions"):
            explanation.append(f"\nWHERE conditions mapped to request parameters:")
            for condition in parsed_view["filter_conditions"]:
                # Find matching parameter
                for param in api_json.get("req", []):
                    if condition.lower().find(param["tag"].lower()) >= 0:
                        explanation.append(f"- SQL: `{condition}` → API: `{param}`")
                        break
        
        # Notes on joins if present
        if parsed_view.get("join_conditions"):
            explanation.append(f"\nJOIN conditions consolidated into request:")
            explanation.append("The API consolidates data from multiple tables, eliminating the need for explicit joins.")
            
        return "\n".join(explanation)
    
    def generate_api_request_body(self, view_name):
        """
        Generate a complete API request body for a view.
        
        Args:
            view_name: The name of the view
            
        Returns:
            Dict containing the API request body or error information
        """
        result = self.process_view_mapping_request(view_name)
        
        if result["success"]:
            return result["api_request_body"]
        else:
            return {"error": result["message"]}


# ==================== MAIN ENTRY POINT ====================

def main():
    """Main entry point for the CoPPER Assistant."""
    logger.info("Starting COPPER Assistant")
    
    # Using hardcoded credentials for the hackathon
    confluence_url = CONFLUENCE_URL
    confluence_username = CONFLUENCE_USERNAME
    confluence_api_token = CONFLUENCE_API_TOKEN
    confluence_space = CONFLUENCE_SPACE
    
    print("\nInitializing COPPER Assistant...")
    print("Connecting to Confluence and loading knowledge base...")
    
    # Initialize the assistant
    assistant = CopperAssistant(
        confluence_url, 
        confluence_username, 
        confluence_api_token,
        space_key=confluence_space
    )
    
    if not assistant.initialize():
        logger.error("Failed to initialize COPPER Assistant. Please check the logs for details.")
        print("Error: Failed to initialize. Please check the logs for details.")
        return
    
    print(f"\n===== COPPER Database-to-API Mapping Assistant =====")
    print(f"Loaded information from {len(assistant.space_pages)} pages in the {assistant.space_key} space.")
    print("I can answer questions about CoPPER database views and how to map them to REST APIs.")
    print("What would you like to know about CoPPER views or APIs?")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            user_input = input("\nQuestion: ").strip()
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Thanks for using the COPPER Assistant. Have a great day!")
                break
            
            if not user_input:
                continue
            
            print("\nLooking that up for you...")
            start_time = time.time()
            answer = assistant.answer_question(user_input)
            end_time = time.time()
            
            print(f"\nAnswer (found in {end_time - start_time:.2f} seconds):")
            print("-------")
            print(answer)
            print("-------")
            
        except KeyboardInterrupt:
            print("\nGoodbye! Feel free to come back if you have more questions.")
            break
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            print(f"Sorry, I ran into an issue: {str(e)}. Let's try a different question.")


# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()
