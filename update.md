app/config.py
"""
Configuration settings for the Enterprise Knowledge Hub.
"""
import os
import ssl
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Security settings: Disable SSL verification in development if set
if os.getenv("PYTHONHTTPSVERIFY") == "0":
    ssl._create_default_https_context = ssl._create_unverified_context

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

# Feature flags
USE_SPACY = os.getenv("USE_SPACY", "True").lower() == "true"
USE_TRANSFORMERS = os.getenv("USE_TRANSFORMERS", "True").lower() == "true"
USE_FAISS = os.getenv("USE_FAISS", "True").lower() == "true"

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
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
import pytesseract
import docx
from pdfminer.high_level import extract_text

from app.api.confluence import confluence_client
from app.api.remedy import remedy_client
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, CACHE_DIR, USE_SPACY

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

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)

# Initialize NLTK resources
STOPWORDS = set(stopwords.words('english'))

# Check if spaCy is available and initialize if requested
SPACY_AVAILABLE = False
nlp = None
if USE_SPACY:
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            SPACY_AVAILABLE = True
            logger.info("Using spaCy with en_core_web_sm model")
        except:
            # Try with blank model if language model is not available
            nlp = spacy.blank("en")
            SPACY_AVAILABLE = True
            logger.info("Using spaCy with blank English model")
    except ImportError:
        logger.info("spaCy not available, using NLTK alternatives")

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

class KnowledgeGraph:
    """Knowledge graph for entity and relationship information."""
    
    def __init__(self):
        """Initialize the knowledge graph."""
        self.entities = {}
        self.relationships = {}
    
    def add_entity(self, entity: str, entity_type: str = None, source_doc: str = None):
        """Add an entity to the knowledge graph."""
        if entity not in self.entities:
            self.entities[entity] = {
                "type": entity_type,
                "mentions": [],
                "relationships": []
            }
        
        if source_doc and source_doc not in self.entities[entity]["mentions"]:
            self.entities[entity]["mentions"].append(source_doc)
    
    def add_relationship(self, entity1: str, relation: str, entity2: str, source_doc: str = None):
        """Add a relationship between entities to the knowledge graph."""
        # Add entities if they don't exist
        if entity1 not in self.entities:
            self.add_entity(entity1, source_doc=source_doc)
        
        if entity2 not in self.entities:
            self.add_entity(entity2, source_doc=source_doc)
        
        # Create the relationship
        rel_key = f"{entity1}:{relation}:{entity2}"
        
        if rel_key not in self.relationships:
            self.relationships[rel_key] = {
                "entity1": entity1,
                "relation": relation,
                "entity2": entity2,
                "mentions": []
            }
        
        if source_doc and source_doc not in self.relationships[rel_key]["mentions"]:
            self.relationships[rel_key]["mentions"].append(source_doc)
        
        # Add relationship to entity
        if rel_key not in self.entities[entity1]["relationships"]:
            self.entities[entity1]["relationships"].append(rel_key)
        
        if rel_key not in self.entities[entity2]["relationships"]:
            self.entities[entity2]["relationships"].append(rel_key)
    
    def extract_entities_and_relationships(self, text: str, doc_id: str = None):
        """Extract entities and relationships from text."""
        if SPACY_AVAILABLE and nlp:
            self._extract_with_spacy(text, doc_id)
        else:
            self._extract_with_nltk(text, doc_id)
    
    def _extract_with_spacy(self, text: str, doc_id: str = None):
        """Extract entities and relationships using spaCy."""
        doc = nlp(text)
        
        # Extract entities
        for ent in doc.ents:
            self.add_entity(ent.text, ent.label_, doc_id)
        
        # Extract simple relationships based on dependency parsing
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                    # Find the subject
                    subj = token.text
                    verb = token.head.text
                    
                    # Find objects
                    for child in token.head.children:
                        if child.dep_ in ("dobj", "pobj"):
                            obj = child.text
                            self.add_relationship(subj, verb, obj, doc_id)
    
    def _extract_with_nltk(self, text: str, doc_id: str = None):
        """Extract entities and relationships using NLTK."""
        # Tokenize and tag sentences
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            
            # Extract named entities
            chunks = ne_chunk(tagged)
            
            entities = []
            current_entity = []
            current_type = None
            
            # Extract entities from chunks
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    # Named entity
                    entity_type = chunk.label()
                    entity_text = ' '.join([c[0] for c in chunk])
                    entities.append((entity_text, entity_type))
                    self.add_entity(entity_text, entity_type, doc_id)
            
            # Extract simple relationship patterns (subject-verb-object)
            # This is a simplified approach compared to dependency parsing
            # Format: [(word, POS), (word, POS), ...]
            for i in range(len(tagged) - 2):
                if tagged[i][1].startswith('N') and tagged[i+1][1].startswith('V') and tagged[i+2][1].startswith('N'):
                    subj = tagged[i][0]
                    verb = tagged[i+1][0]
                    obj = tagged[i+2][0]
                    self.add_relationship(subj, verb, obj, doc_id)

class DocumentProcessor:
    """Processor for extracting text from various document types and sources."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.documents = {}  # Cache of processed documents
        self.chunks = []  # All document chunks
        self.processed_sources = set()  # Track processed sources
        self.cache_dir = os.path.join(CACHE_DIR, "processed")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.knowledge_graph = KnowledgeGraph()
    
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
                
                # Extract entities for knowledge graph
                self.knowledge_graph.extract_entities_and_relationships(text_content, doc_id)
                
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
                
                # Extract entities for knowledge graph
                self.knowledge_graph.extract_entities_and_relationships(content, doc_id)
                
                # Add to documents dictionary
                self.documents[doc_id] = doc
                processed_docs.append(doc)
                
                logger.info(f"Processed Remedy incident: {incident_id}")
                
            except Exception as e:
                incident_id = incident.get('values', {}).get('Incident Number', 'unknown')
                logger.error(f"Error processing Remedy incident {incident_id}: {str(e)}")
        
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
            
        # Extract potential title from the first line
        lines = doc.content.split('\n')
        title = lines[0] if lines else ""
        
        # Use adaptive chunking based on available NLP library
        chunks = self._create_semantic_chunks(doc.content, doc.doc_id, doc.source)
        
        # Store chunks in the document
        doc.chunks = chunks
        
        # Add chunks to global chunk list
        self.chunks.extend(chunks)
        
        logger.debug(f"Created {len(chunks)} chunks for document {doc.doc_id}")
    
    def _create_semantic_chunks(self, text: str, doc_id: str, source: str,
                              min_chunk_size: int = 100, 
                              max_chunk_size: int = 500) -> List[DocumentChunk]:
        """
        Chunk text into semantically coherent pieces.
        
        Args:
            text: Text to chunk
            doc_id: Document ID
            source: Source system (confluence, remedy)
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            
        Returns:
            List of document chunks
        """
        if not text:
            return []
        
        # Extract potential title
        lines = text.split('\n')
        title = None
        if lines and len(lines[0].strip()) < 200 and not lines[0].strip().endswith('.'):
            title = lines[0].strip()
            text = '\n'.join(lines[1:])
        
        chunks = []
        chunk_id = 0
        
        # Add title as a special chunk if present
        if title:
            chunks.append(DocumentChunk(
                text=title,
                doc_id=doc_id,
                chunk_id=chunk_id,
                source=source,
                metadata={"position": "title", "is_title": True}
            ))
            chunk_id += 1
        
        # Split text into paragraphs
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        # Check if we should use advanced semantic chunking with spaCy
        if SPACY_AVAILABLE and nlp:
            # Use spaCy for more semantically aware chunking
            return self._spacy_semantic_chunks(text, doc_id, source, title, chunk_id)
        else:
            # Use simpler NLTK-based chunking as fallback
            return self._nltk_semantic_chunks(text, doc_id, source, title, chunk_id)
    
    def _spacy_semantic_chunks(self, text: str, doc_id: str, source: str, 
                             title=None, start_chunk_id: int = 0) -> List[DocumentChunk]:
        """
        Create chunks with spaCy for better semantic coherence.
        
        Args:
            text: Text to chunk
            doc_id: Document ID
            source: Source system
            title: Document title if available
            start_chunk_id: Starting chunk ID
            
        Returns:
            List of document chunks
        """
        chunks = []
        chunk_id = start_chunk_id
        
        # Add title as first chunk if it exists and wasn't already added
        if title and chunk_id == 0:
            chunks.append(DocumentChunk(
                text=title,
                doc_id=doc_id,
                chunk_id=chunk_id,
                source=source,
                metadata={"position": "title", "is_title": True}
            ))
            chunk_id += 1
        
        # Process with spaCy
        doc = nlp(text)
        
        # First, identify all section headings
        headings = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            # Check if this looks like a heading (short, no ending period, etc.)
            if (len(sent_text) < 100 and 
                not sent_text.endswith('.') and 
                any(token.is_upper for token in sent) and
                len(sent_text.split()) < 10):
                headings.append((sent.start_char, sent.end_char, sent_text))
        
        # Split by headings if found
        if headings:
            for i, (start, end, heading_text) in enumerate(headings):
                # Add heading as a chunk
                chunks.append(DocumentChunk(
                    text=heading_text,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    source=source,
                    metadata={"position": chunk_id, "is_heading": True}
                ))
                chunk_id += 1
                
                # Process content between this heading and the next
                if i < len(headings) - 1:
                    next_start = headings[i+1][0]
                    section_text = text[end:next_start]
                else:
                    section_text = text[end:]
                
                # Skip if section is empty
                if not section_text.strip():
                    continue
                
                # Create chunks from the section
                section_chunks = self._create_content_chunks(
                    section_text, doc_id, source, chunk_id
                )
                chunks.extend(section_chunks)
                chunk_id += len(section_chunks)
        else:
            # No headings found, split by semantic coherence (paragraphs, sentences)
            content_chunks = self._create_content_chunks(text, doc_id, source, chunk_id)
            chunks.extend(content_chunks)
            chunk_id += len(content_chunks)
        
        return chunks
    
    def _nltk_semantic_chunks(self, text: str, doc_id: str, source: str, 
                             title=None, start_chunk_id: int = 0) -> List[DocumentChunk]:
        """
        Create chunks with NLTK for reasonable coherence.
        
        Args:
            text: Text to chunk
            doc_id: Document ID
            source: Source system
            title: Document title if available
            start_chunk_id: Starting chunk ID
            
        Returns:
            List of document chunks
        """
        chunks = []
        chunk_id = start_chunk_id
        
        # Add title as first chunk if it exists and wasn't already added
        if title and chunk_id == 0:
            chunks.append(DocumentChunk(
                text=title,
                doc_id=doc_id,
                chunk_id=chunk_id,
                source=source,
                metadata={"position": "title", "is_title": True}
            ))
            chunk_id += 1
        
        # Simple heading detection with regex
        heading_pattern = re.compile(r'^(#+|\d+\.|\*+|\d+\)|\w+\))\s+(.+)$', re.MULTILINE)
        headings = list(heading_pattern.finditer(text))
        
        if headings:
            # Process by headings
            for i, match in enumerate(headings):
                heading_text = match.group(2).strip()
                start_pos = match.end()
                
                # Add heading as a chunk
                chunks.append(DocumentChunk(
                    text=heading_text,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    source=source,
                    metadata={"position": chunk_id, "is_heading": True}
                ))
                chunk_id += 1
                
                # Process content between this heading and the next
                if i < len(headings) - 1:
                    next_start = headings[i+1].start()
                    section_text = text[start_pos:next_start]
                else:
                    section_text = text[start_pos:]
                
                # Skip if section is empty
                if not section_text.strip():
                    continue
                
                # Create chunks from the section
                section_chunks = self._create_content_chunks(
                    section_text, doc_id, source, chunk_id
                )
                chunks.extend(section_chunks)
                chunk_id += len(section_chunks)
        else:
            # No headings found, split by paragraphs and sentences
            content_chunks = self._create_content_chunks(text, doc_id, source, chunk_id)
            chunks.extend(content_chunks)
        
        return chunks
    
    def _create_content_chunks(self, text: str, doc_id: str, source: str, 
                             start_chunk_id: int) -> List[DocumentChunk]:
        """
        Split content text into appropriately sized chunks.
        
        Args:
            text: Text to chunk
            doc_id: Document ID
            source: Source system
            start_chunk_id: Starting chunk ID
            
        Returns:
            List of document chunks
        """
        chunks = []
        chunk_id = start_chunk_id
        
        # Split into paragraphs
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        current_chunk = []
        current_size = 0
        max_size = CHUNK_SIZE
        
        for para in paragraphs:
            para_size = len(para)
            
            # If paragraph is too large, split by sentences
            if para_size > max_size:
                # Process current chunk first if not empty
                if current_chunk:
                    chunk_text = "\n".join(current_chunk)
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        source=source,
                        metadata={"position": chunk_id}
                    ))
                    chunk_id += 1
                    current_chunk = []
                    current_size = 0
                
                # Split paragraph into sentences
                sentences = sent_tokenize(para)
                
                # Process sentences
                sentence_chunk = []
                sentence_size = 0
                
                for sentence in sentences:
                    sent_size = len(sentence)
                    
                    if sentence_size + sent_size <= max_size:
                        sentence_chunk.append(sentence)
                        sentence_size += sent_size
                    else:
                        # Add current sentence chunk
                        if sentence_chunk:
                            chunk_text = " ".join(sentence_chunk)
                            chunks.append(DocumentChunk(
                                text=chunk_text,
                                doc_id=doc_id,
                                chunk_id=chunk_id,
                                source=source,
                                metadata={"position": chunk_id}
                            ))
                            chunk_id += 1
                        
                        # Start new chunk with current sentence
                        sentence_chunk = [sentence]
                        sentence_size = sent_size
                
                # Add remaining sentence chunk
                if sentence_chunk:
                    chunk_text = " ".join(sentence_chunk)
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        source=source,
                        metadata={"position": chunk_id}
                    ))
                    chunk_id += 1
            
            # Regular paragraph processing
            elif current_size + para_size <= max_size:
                current_chunk.append(para)
                current_size += para_size
            else:
                # Add current chunk
                if current_chunk:
                    chunk_text = "\n".join(current_chunk)
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        source=source,
                        metadata={"position": chunk_id}
                    ))
                    chunk_id += 1
                
                # Start new chunk
                current_chunk = [para]
                current_size = para_size
        
        # Add final chunk if not empty
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append(DocumentChunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_id=chunk_id,
                source=source,
                metadata={"position": chunk_id}
            ))
        
        return chunks
    
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
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from app.config import USE_SPACY

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

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

# Initialize resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Check if spaCy is available and initialize if requested
SPACY_AVAILABLE = False
nlp = None
if USE_SPACY:
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            SPACY_AVAILABLE = True
            logger.info("Using spaCy with en_core_web_sm model")
        except:
            # Try with blank model if language model is not available
            nlp = spacy.blank("en")
            SPACY_AVAILABLE = True
            logger.info("Using spaCy with blank English model")
    except ImportError:
        logger.info("spaCy not available, using NLTK alternatives")

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
        
        # Process with spaCy if available, otherwise use NLTK
        if SPACY_AVAILABLE and nlp:
            # SpaCy-based processing
            keywords, entities, intent = self._process_with_spacy(processed_query)
        else:
            # NLTK-based processing
            keywords = self._extract_keywords_nltk(processed_query)
            entities = self._extract_entities_nltk(processed_query)
            intent = self._determine_intent(processed_query)
        
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
        
        # Remove punctuation (except for question marks which might be important)
        query = query.translate(str.maketrans('', '', string.punctuation.replace('?', '').replace('-', '')))
        
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
    
    def _process_with_spacy(self, query: str) -> Tuple[List[str], Dict[str, List[str]], str]:
        """
        Process query using spaCy for better NLP understanding.
        
        Args:
            query: The processed query
            
        Returns:
            Tuple of (keywords, entities, intent)
        """
        # Parse query with spaCy
        doc = nlp(query)
        
        # Extract keywords
        keywords = []
        for token in doc:
            if not token.is_stop and not token.is_punct and token.pos_ in ('NOUN', 'PROPN', 'VERB', 'ADJ'):
                # Lemmatize the token
                keyword = token.lemma_
                if keyword not in keywords:
                    keywords.append(keyword)
        
        # Extract entities
        entities = {}
        for ent in doc.ents:
            entity_type = ent.label_
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(ent.text)
        
        # Determine intent
        intent = self._determine_intent(query)
        
        return keywords, entities, intent
    
    def _extract_keywords_nltk(self, query: str) -> List[str]:
        """
        Extract important keywords from the query using NLTK.
        
        Args:
            query: The processed query
            
        Returns:
            List of extracted keywords
        """
        # Tokenize
        tokens = nltk.word_tokenize(query)
        
        # POS tag for improved keyword extraction
        pos_tags = nltk.pos_tag(tokens)
        
        # Keep nouns, verbs, adjectives
        keywords = []
        for token, pos in pos_tags:
            if token.lower() not in stop_words and len(token) > 2:
                if pos.startswith('N') or pos.startswith('V') or pos.startswith('J'):
                    lemma = lemmatizer.lemmatize(token.lower())
                    if lemma not in keywords:
                        keywords.append(lemma)
        
        return keywords
    
    def _extract_entities_nltk(self, query: str) -> Dict[str, List[str]]:
        """
        Extract named entities from the query using NLTK.
        
        Args:
            query: The processed query
            
        Returns:
            Dictionary of entity types and values
        """
        tokens = nltk.word_tokenize(query)
        pos_tags = nltk.pos_tag(tokens)
        
        # Use NLTK chunking for named entity recognition
        chunks = ne_chunk(pos_tags)
        
        entities = {}
        current_entity = []
        current_type = None
        
        # Extract named entities from chunks
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entity_type = chunk.label()
                entity_text = ' '.join([c[0] for c in chunk])
                
                if entity_type not in entities:
                    entities[entity_type] = []
                    
                entities[entity_type].append(entity_text)
        
        return entities
    
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
    
    def _expand_query(self, query: str, keywords: List[str]) -> str:
        """
        Expand the query with synonyms or related terms for better retrieval.
        
        Args:
            query: The processed query
            keywords: Extracted keywords
            
        Returns:
            Expanded query
        """
        # Use WordNet for synonym expansion
        expanded_terms = set()
        
        if keywords:
            # Take top 3 keywords for expansion
            for keyword in keywords[:3]:
                # Get synonyms from WordNet
                synonyms = set()
                for syn in nltk.corpus.wordnet.synsets(keyword):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().lower().replace('_', ' ')
                        # Only add if different from original and not already in expanded_terms
                        if synonym != keyword and synonym not in expanded_terms:
                            synonyms.add(synonym)
                
                # Add top 2 synonyms
                expanded_terms.update(list(synonyms)[:2])
        
        # Create expanded query if new terms found
        if expanded_terms:
            expanded_query = query + " " + " ".join(expanded_terms)
            return expanded_query
        
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









app/core/summarizer.py
"""
Summarizer module for generating concise summaries from retrieved content.
"""
import re
import logging
import string
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.core.retriever import SearchResult
from app.config import USE_SPACY

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
lemmatizer = WordNetLemmatizer()

# Check if spaCy is available
SPACY_AVAILABLE = False
nlp = None
if USE_SPACY:
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            SPACY_AVAILABLE = True
            logger.info("Using spaCy in Summarizer")
        except:
            nlp = spacy.blank("en")
            SPACY_AVAILABLE = True
            logger.info("Using spaCy blank model in Summarizer")
    except ImportError:
        logger.info("spaCy not available in Summarizer, using NLTK alternatives")

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
            if SPACY_AVAILABLE and nlp:
                summary_text = self._spacy_summarization(query, docs, max_length)
            else:
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
    
    def _spacy_summarization(self, query: str, docs: List[str], max_length: int) -> str:
        """
        Generate a summary using spaCy for better linguistic understanding.
        
        Args:
            query: Original search query
            docs: List of document texts
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        # Parse query
        query_doc = nlp(query)
        
        # Split documents into sentences
        all_sentences = []
        doc_sentences = []
        
        for doc in docs:
            doc_obj = nlp(doc)
            sentences = [sent.text.strip() for sent in doc_obj.sents if len(sent.text.strip()) > 20]
            doc_sentences.append(sentences)
            all_sentences.extend(sentences)
        
        # Remove duplicates
        unique_sentences = []
        seen = set()
        
        for sentence in all_sentences:
            # Normalize for comparison (lowercase, remove extra spaces)
            norm_sent = re.sub(r'\s+', ' ', sentence.lower()).strip()
            if norm_sent not in seen and len(norm_sent) > 20:
                seen.add(norm_sent)
                unique_sentences.append(sentence)
        
        # Score sentences with multiple features
        sentence_scores = {}
        
        for sentence in unique_sentences:
            # Parse sentence
            sent_doc = nlp(sentence)
            
            # Base score from similarity to query
            query_similarity = query_doc.similarity(sent_doc)
            
            # Additional features
            features = {
                'query_similarity': query_similarity,
                'sentence_length': min(1.0, len(sentence) / 200),  # Prefer medium-length sentences
                'entity_count': len(sent_doc.ents) * 0.1,  # Bonus for sentences with entities
                'has_numbers': any(token.is_digit for token in sent_doc) * 0.1,  # Bonus for factual sentences
            }
            
            # Calculate weighted score
            score = (
                features['query_similarity'] * 0.6 +
                (1 - abs(features['sentence_length'] - 0.5)) * 0.2 +  # Prefer medium-length sentences
                features['entity_count'] * 0.1 +
                features['has_numbers'] * 0.1
            )
            
            sentence_scores[sentence] = score
        
        # Rank sentences
        ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top sentences
        selected_sentences = []
        current_length = 0
        
        for sentence, _ in ranked_sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= max_length:
                selected_sentences.append(sentence)
                current_length += sentence_length
            else:
                break
        
        # Reorder sentences to match original document order for better coherence
        ordered_sentences = self._reorder_sentences(selected_sentences, doc_sentences)
        
        # Combine into coherent summary
        summary = " ".join(ordered_sentences)
        
        return summary
    
    def _extractive_summarization(self, query: str, docs: List[str], max_length: int) -> str:
        """
        Generate an extractive summary from multiple documents using NLTK.
        
        Args:
            query: Original search query
            docs: List of document texts
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        # Split documents into sentences
        all_sentences = []
        doc_sentences = []
        
        for doc in docs:
            sentences = sent_tokenize(doc)
            valid_sentences = [s for s in sentences if len(s.strip()) > 20]
            doc_sentences.append(valid_sentences)
            all_sentences.extend(valid_sentences)
        
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
        
        # 4. Contains key entities (simple detection)
        entity_scores = {}
        query_tokens = set(word_tokenize(query.lower()))
        for sentence in unique_sentences:
            sentence_tokens = set(word_tokenize(sentence.lower()))
            overlap = len(query_tokens.intersection(sentence_tokens))
            entity_scores[sentence] = min(1.0, overlap * 0.2)
        
        # Combine scores: TF-IDF (50%) + Position (20%) + Length (15%) + Entities (15%)
        for i, sentence in enumerate(unique_sentences):
            sentence_scores[sentence] = (
                tfidf_scores.get(sentence, 0) * 0.5 +
                position_scores.get(sentence, 0) * 0.2 +
                length_scores.get(sentence, 0) * 0.15 +
                entity_scores.get(sentence, 0) * 0.15
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
        
        # Reorder sentences for better coherence
        ordered_sentences = self._reorder_sentences(selected_sentences, doc_sentences)
        
        if not ordered_sentences:
            # Fallback if no sentences were selected
            return ranked_sentences[0][0] if ranked_sentences else "No relevant information could be extracted."
        
        # Join sentences into a coherent summary
        summary = ' '.join(ordered_sentences)
        
        return summary
    
    def _reorder_sentences(self, selected_sentences: List[str], 
                          doc_sentences: List[List[str]]) -> List[str]:
        """
        Reorder selected sentences to match original document order for better coherence.
        
        Args:
            selected_sentences: Selected sentences for the summary
            doc_sentences: Sentences grouped by document
            
        Returns:
            Reordered sentences
        """
        # Create a map of sentence to its original position
        sentence_positions = {}
        global_pos = 0
        
        for doc_idx, sentences in enumerate(doc_sentences):
            for sent_idx, sent in enumerate(sentences):
                # Create a position key that preserves document and sentence order
                position_key = (doc_idx, sent_idx)
                sentence_positions[sent] = position_key
                global_pos += 1
        
        # Sort selected sentences by their original positions
        ordered_sentences = []
        for sent in selected_sentences:
            for doc_sent, position in sentence_positions.items():
                if self._sentence_similarity(sent, doc_sent) > 0.8:
                    ordered_sentences.append((sent, position))
                    break
        
        # Sort by original position
        ordered_sentences.sort(key=lambda x: x[1])
        
        # Return just the sentences
        return [sent for sent, _ in ordered_sentences]
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """
        Calculate similarity between two sentences.
        
        Args:
            sent1: First sentence
            sent2: Second sentence
            
        Returns:
            Similarity score (0-1)
        """
        # Normalize sentences
        sent1 = re.sub(r'\s+', ' ', sent1.lower()).strip()
        sent2 = re.sub(r'\s+', ' ', sent2.lower()).strip()
        
        # Check if they're very similar initially
        if sent1 == sent2:
            return 1.0
            
        # Tokenize
        tokens1 = set(word_tokenize(sent1))
        tokens2 = set(word_tokenize(sent2))
        
        # Remove stopwords
        tokens1 = {token for token in tokens1 if token not in stop_words}
        tokens2 = {token for token in tokens2 if token not in stop_words}
        
        # Calculate Jaccard similarity
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
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

# Create a singleton instance
summarizer = Summarizer()
