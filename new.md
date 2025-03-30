enterprise_rag/
├── app.py                      # Main Flask application entry point
├── config.py                   # Configuration management
├── requirements.txt            # Package dependencies
├── .env.example                # Example environment variables
├── static/                     # Static assets
│   ├── css/
│   │   └── styles.css          # Custom styling
│   ├── js/
│   │   └── main.js             # Frontend JavaScript
│   └── img/
├── templates/                  # HTML templates
│   ├── index.html              # Main chat interface
│   ├── base.html               # Base template
│   └── components/             # Reusable UI components
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── logger.py               # Logging setup
│   └── cache.py                # Caching mechanisms
├── data_sources/               # Data source integrations
│   ├── __init__.py
│   ├── confluence.py           # Confluence client and processing
│   ├── remedy.py               # Remedy client and processing
│   ├── jira.py                 # JIRA client and processing
│   └── base.py                 # Base data source interface
├── rag/                        # RAG pipeline components
│   ├── __init__.py
│   ├── query_processor.py      # Query analysis and processing
│   ├── document_processor.py   # Document formatting and normalization
│   ├── chunking.py             # Text chunking strategies
│   ├── embedding.py            # Embedding generation and storage
│   ├── vector_search.py        # Vector similarity search
│   └── context_builder.py      # Context building for generation
└── llm/                        # LLM integration
    ├── __init__.py
    ├── gemini_client.py        # Google Gemini integration
    └── prompt_templates.py     # System prompts














#!/usr/bin/env python3
import os
import logging
from flask import Flask, render_template, request, jsonify
from config import Config
from utils.logger import setup_logging
from utils.cache import setup_cache
from data_sources.confluence import ConfluenceClient
from data_sources.remedy import RemedyClient
from data_sources.jira import JiraClient
from rag.query_processor import QueryProcessor
from llm.gemini_client import GeminiClient

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Setup logging
logger = setup_logging()

# Initialize cache
cache = setup_cache(app)

# Initialize clients based on config
try:
    # Initialize Confluence client
    confluence_client = ConfluenceClient(
        base_url=app.config['CONFLUENCE_URL'],
        username=app.config['CONFLUENCE_USERNAME'],
        password=app.config['CONFLUENCE_PASSWORD'],
        ssl_verify=app.config.get('CONFLUENCE_SSL_VERIFY', False)
    )
    
    # Initialize Remedy client
    remedy_client = RemedyClient(
        server_url=app.config['REMEDY_URL'],
        username=app.config['REMEDY_USERNAME'],
        password=app.config['REMEDY_PASSWORD'],
        ssl_verify=app.config.get('REMEDY_SSL_VERIFY', False)
    )
    
    # Initialize JIRA client
    jira_client = JiraClient(
        base_url=app.config['JIRA_URL'],
        username=app.config['JIRA_USERNAME'],
        password=app.config['JIRA_PASSWORD'],
        ssl_verify=app.config.get('JIRA_SSL_VERIFY', False)
    )
    
    # Initialize Gemini client
    gemini_client = GeminiClient(
        project_id=app.config['GEMINI_PROJECT_ID'],
        region=app.config['GEMINI_REGION'],
        model_name=app.config['GEMINI_MODEL_NAME']
    )
    
    # Initialize the query processor with all data sources
    query_processor = QueryProcessor(
        confluence_client=confluence_client,
        remedy_client=remedy_client,
        jira_client=jira_client,
        gemini_client=gemini_client,
        cache=cache
    )

    logger.info("All clients initialized successfully")
    
except Exception as e:
    logger.error(f"Error initializing clients: {e}")
    raise

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat messages and return responses."""
    try:
        data = request.json
        query = data.get('message', '')
        selected_sources = data.get('sources', ['confluence', 'remedy', 'jira'])
        
        if not query:
            return jsonify({'error': 'No message provided'}), 400
        
        # Process the query and get a response
        response = query_processor.process_query(
            query=query,
            sources=selected_sources
        )
        
        return jsonify({
            'response': response.get('answer', ''),
            'sources': response.get('sources', []),
            'processing_time': response.get('processing_time', 0)
        })
        
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request',
            'details': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = {
        'status': 'ok',
        'confluence': confluence_client.test_connection(),
        'remedy': remedy_client.test_connection(),
        'jira': jira_client.test_connection(),
        'gemini': gemini_client.test_connection()
    }
    
    all_healthy = all(status.values())
    return jsonify(status), 200 if all_healthy else 503

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)















import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class Config:
    """Configuration class for the application."""
    
    # Flask config
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-dev-key-change-in-production')
    
    # Caching config
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'SimpleCache')
    CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_TIMEOUT', 300))
    
    # Confluence config
    CONFLUENCE_URL = os.environ.get('CONFLUENCE_URL', '')
    CONFLUENCE_USERNAME = os.environ.get('CONFLUENCE_USERNAME', '')
    CONFLUENCE_PASSWORD = os.environ.get('CONFLUENCE_PASSWORD', '')
    CONFLUENCE_SSL_VERIFY = os.environ.get('CONFLUENCE_SSL_VERIFY', 'False').lower() == 'true'
    
    # Remedy config
    REMEDY_URL = os.environ.get('REMEDY_URL', '')
    REMEDY_USERNAME = os.environ.get('REMEDY_USERNAME', '')
    REMEDY_PASSWORD = os.environ.get('REMEDY_PASSWORD', '')
    REMEDY_SSL_VERIFY = os.environ.get('REMEDY_SSL_VERIFY', 'False').lower() == 'true'
    
    # JIRA config
    JIRA_URL = os.environ.get('JIRA_URL', '')
    JIRA_USERNAME = os.environ.get('JIRA_USERNAME', '')
    JIRA_PASSWORD = os.environ.get('JIRA_PASSWORD', '')
    JIRA_SSL_VERIFY = os.environ.get('JIRA_SSL_VERIFY', 'False').lower() == 'true'
    
    # Gemini config
    GEMINI_PROJECT_ID = os.environ.get('GEMINI_PROJECT_ID', 'prj-dv-cws-4363')
    GEMINI_REGION = os.environ.get('GEMINI_REGION', 'us-central1')
    GEMINI_MODEL_NAME = os.environ.get('GEMINI_MODEL_NAME', 'gemini-2.0-flash-001')
    
    # RAG config
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    VECTOR_DB_PATH = os.environ.get('VECTOR_DB_PATH', './data/vector_db')
    
    # Chunking config
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 200))
    
    # Response generation config
    TEMPERATURE = float(os.environ.get('TEMPERATURE', 0.7))
    MAX_OUTPUT_TOKENS = int(os.environ.get('MAX_OUTPUT_TOKENS', 8192))

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = False
    TESTING = True
    # Use in-memory caching for testing
    CACHE_TYPE = 'SimpleCache'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    # Use Redis caching in production for better performance
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'RedisCache')
    CACHE_REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# Select the appropriate configuration based on environment
config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig
}

# Get the current config
app_config = config_by_name.get(os.environ.get('FLASK_ENV', 'development'), DevelopmentConfig)




















import logging
import os
import time
from typing import Dict, List, Optional, Union, Any

# Import Vertex AI packages
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.api_core.exceptions import GoogleAPICallError

logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for interacting with Google's Gemini model via Vertex AI."""
    
    def __init__(
        self, 
        project_id: str, 
        location: str = "us-central1", 
        model_name: str = "gemini-2.0-flash-001"
    ):
        """
        Initialize the Gemini client.
        
        Args:
            project_id: The GCP project ID
            location: The region for Vertex AI
            model_name: The Gemini model to use
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.model = None
        
        try:
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            
            # Create model instance
            self.model = GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise

    def test_connection(self) -> bool:
        """
        Test the connection to the Gemini API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Simple query to test the connection
            response = self.model.generate_content(
                "Hello, this is a test.", 
                generation_config=GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=10,
                )
            )
            return True if response else False
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def generate_response(
        self, 
        query: str, 
        context: str, 
        system_instructions: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 8192,
        streaming: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a response based on the query and retrieved context.
        
        Args:
            query: The user's query
            context: Retrieved context from data sources
            system_instructions: Optional system instructions
            temperature: Temperature for generation (0.0-1.0)
            max_output_tokens: Maximum output tokens
            streaming: Whether to stream the response
            
        Returns:
            Dict containing the response and metadata
        """
        start_time = time.time()
        logger.info(f"Generating response for query: {query[:100]}...")
        
        # Construct the prompt with context and query
        prompt = self._construct_prompt(query, context, system_instructions)
        
        # Configure generation parameters
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=0.95,
            max_output_tokens=max_output_tokens,
        )
        
        try:
            if streaming:
                # Streaming response (for real-time UI updates)
                response_text = ""
                for chunk in self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    stream=True,
                ):
                    if chunk.candidates and chunk.candidates[0].text:
                        response_text += chunk.candidates[0].text
                        # Could yield chunks here for streaming to the client
                
                processed_response = self._process_response(response_text)
            else:
                # Non-streaming response
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                
                if response.candidates and response.candidates[0].text:
                    processed_response = self._process_response(response.candidates[0].text)
                else:
                    processed_response = {"answer": "I couldn't generate a response. Please try again."}
            
            # Add metadata
            processed_response["processing_time"] = time.time() - start_time
            
            return processed_response
            
        except GoogleAPICallError as e:
            logger.error(f"Gemini API error: {str(e)}")
            return {
                "answer": "Sorry, I encountered an error while generating a response. Please try again later.",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}")
            return {
                "answer": "Sorry, something went wrong. Please try again.",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _construct_prompt(
        self, 
        query: str, 
        context: str,
        system_instructions: Optional[str] = None
    ) -> str:
        """
        Construct the prompt for the Gemini model.
        
        Args:
            query: The user's query
            context: Retrieved context from data sources
            system_instructions: Optional system instructions
            
        Returns:
            Formatted prompt string
        """
        # Default system instructions if none provided
        if not system_instructions:
            system_instructions = """
            You are a professional corporate assistant with expertise in technical and business matters.
            Your role is to provide helpful, accurate, and concise information based on the context provided.
            Always maintain a professional tone while being friendly and approachable.
            When responding to technical questions, be clear and precise.
            If information is not available in the context, acknowledge this and provide general guidance if possible.
            Format your responses in a clear and structured way, using markdown for better readability.
            When appropriate, include source references from the context.
            """
        
        # Construct the full prompt
        prompt = f"""
        {system_instructions}

        CONTEXT:
        {context}

        USER QUERY:
        {query}

        Please respond to the user query based on the provided context. 
        Be professional, helpful, and comprehensive.
        """
        
        return prompt
    
    def _process_response(self, response_text: str) -> Dict[str, Any]:
        """
        Process the raw response from Gemini.
        
        Args:
            response_text: Raw text response from Gemini
            
        Returns:
            Processed response dictionary
        """
        # Basic processing for now - can be extended for more complex parsing
        return {
            "answer": response_text.strip()
        }













import logging
import time
from typing import Dict, List, Any, Optional, Union
import re

from data_sources.confluence import ConfluenceClient
from data_sources.remedy import RemedyClient
from data_sources.jira import JiraClient
from llm.gemini_client import GeminiClient
from rag.chunking import ChunkingEngine
from rag.embedding import EmbeddingEngine
from rag.vector_search import VectorSearch
from rag.context_builder import ContextBuilder

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Main query processing pipeline for the RAG system.
    Handles query analysis, retrieval, and response generation.
    """
    
    def __init__(
        self,
        confluence_client: ConfluenceClient,
        remedy_client: RemedyClient,
        jira_client: JiraClient,
        gemini_client: GeminiClient,
        cache: Any = None
    ):
        """
        Initialize the query processor with required clients.
        
        Args:
            confluence_client: Client for Confluence
            remedy_client: Client for Remedy
            jira_client: Client for JIRA
            gemini_client: Client for Gemini
            cache: Optional cache instance
        """
        self.confluence_client = confluence_client
        self.remedy_client = remedy_client
        self.jira_client = jira_client
        self.gemini_client = gemini_client
        self.cache = cache
        
        # Initialize RAG components
        self.chunking_engine = ChunkingEngine()
        self.embedding_engine = EmbeddingEngine()
        self.vector_search = VectorSearch(self.embedding_engine)
        self.context_builder = ContextBuilder()
        
        logger.info("Query processor initialized")
    
    def process_query(
        self,
        query: str,
        sources: List[str] = ["confluence", "remedy", "jira"],
        max_results: int = 10,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            query: The user's query
            sources: List of data sources to query
            max_results: Maximum number of results to retrieve
            use_cache: Whether to use caching
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        logger.info(f"Processing query: {query}")
        
        # Check cache first if enabled
        if use_cache and self.cache:
            cache_key = f"query:{query}:sources:{','.join(sorted(sources))}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: {query}")
                cached_result["cached"] = True
                return cached_result
        
        try:
            # 1. Analyze query to understand intent and determine sources
            analyzed_query = self._analyze_query(query)
            
            # 2. Select appropriate data sources based on analysis
            active_sources = self._select_data_sources(analyzed_query, sources)
            
            # 3. Retrieve relevant information from selected sources
            retrieved_data = self._retrieve_from_sources(analyzed_query, active_sources, max_results)
            
            # 4. Process retrieved documents
            processed_documents = self._process_documents(retrieved_data)
            
            # 5. Generate embeddings and perform vector search if needed
            relevant_chunks = self._retrieve_relevant_chunks(analyzed_query, processed_documents)
            
            # 6. Build context from relevant chunks
            context = self.context_builder.build_context(relevant_chunks, analyzed_query)
            
            # 7. Generate response using Gemini
            response = self.gemini_client.generate_response(
                query=query,
                context=context,
                system_instructions=self._get_system_instructions(analyzed_query, active_sources)
            )
            
            # 8. Add metadata about sources
            response["sources"] = self._format_sources(relevant_chunks)
            response["processing_time"] = time.time() - start_time
            
            # Cache the result if caching is enabled
            if use_cache and self.cache:
                self.cache.set(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": "I encountered an error while processing your query. Please try again.",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to understand intent and extract key information.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with query analysis
        """
        # For now, a simple rule-based analysis
        # In a full implementation, this could use a classifier or LLM
        analysis = {
            "original_query": query,
            "normalized_query": query.lower(),
            "detected_entities": [],
            "query_type": "general",
            "suggested_sources": []
        }
        
        # Check for data source specific keywords
        if re.search(r'\b(confluence|page|document|documentation|wiki)\b', query.lower()):
            analysis["suggested_sources"].append("confluence")
        
        if re.search(r'\b(remedy|incident|ticket|issue|problem|service desk)\b', query.lower()):
            analysis["suggested_sources"].append("remedy")
            analysis["query_type"] = "incident"
        
        if re.search(r'\b(jira|task|story|epic|sprint|backlog|project)\b', query.lower()):
            analysis["suggested_sources"].append("jira")
            analysis["query_type"] = "task"
        
        # Extract potential entity references
        # Ticket numbers
        ticket_matches = re.findall(r'\b(INC\d+|TASK\d+|PROJ\d+)\b', query)
        for match in ticket_matches:
            analysis["detected_entities"].append({"type": "ticket", "value": match})
        
        # Dates
        date_matches = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', query)
        for match in date_matches:
            analysis["detected_entities"].append({"type": "date", "value": match})
        
        logger.debug(f"Query analysis: {analysis}")
        return analysis
    
    def _select_data_sources(
        self, 
        analyzed_query: Dict[str, Any],
        user_selected_sources: List[str]
    ) -> List[str]:
        """
        Select appropriate data sources based on query analysis and user preferences.
        
        Args:
            analyzed_query: Query analysis results
            user_selected_sources: Sources selected by the user
            
        Returns:
            List of selected data source names
        """
        # Start with user selected sources
        selected_sources = user_selected_sources.copy()
        
        # If no sources suggested by analysis, use all selected sources
        if not analyzed_query["suggested_sources"]:
            return selected_sources
        
        # Otherwise, prioritize the intersection of user selected and suggested sources
        prioritized_sources = [s for s in selected_sources if s in analyzed_query["suggested_sources"]]
        
        # If there are prioritized sources, use them, otherwise fall back to user selected
        return prioritized_sources if prioritized_sources else selected_sources
    
    def _retrieve_from_sources(
        self,
        analyzed_query: Dict[str, Any],
        sources: List[str],
        max_results: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant information from selected data sources.
        
        Args:
            analyzed_query: Query analysis results
            sources: List of data sources to query
            max_results: Maximum number of results to retrieve
            
        Returns:
            Dictionary with retrieved data grouped by source
        """
        retrieved_data = {}
        
        # Process entity-based queries separately
        if analyzed_query["detected_entities"]:
            for entity in analyzed_query["detected_entities"]:
                if entity["type"] == "ticket" and "remedy" in sources:
                    # Direct ticket lookup in Remedy
                    if entity["value"].startswith("INC"):
                        incident = self.remedy_client.get_incident(entity["value"])
                        if incident:
                            retrieved_data.setdefault("remedy", []).append(incident)
                
                if entity["type"] == "ticket" and "jira" in sources:
                    # Direct issue lookup in JIRA
                    if entity["value"].startswith("TASK") or entity["value"].startswith("PROJ"):
                        issue = self.jira_client.get_issue(entity["value"])
                        if issue:
                            retrieved_data.setdefault("jira", []).append(issue)
        
        # For each source, retrieve relevant information
        query_text = analyzed_query["normalized_query"]
        
        if "confluence" in sources:
            confluence_results = self.confluence_client.search_content(
                cql=query_text, 
                limit=max_results
            )
            if confluence_results:
                retrieved_data["confluence"] = confluence_results
        
        if "remedy" in sources and "remedy" not in retrieved_data:
            # Search for incidents based on the query
            if analyzed_query["query_type"] == "incident":
                remedy_results = self.remedy_client.search_incidents(query_text, limit=max_results)
                if remedy_results:
                    retrieved_data["remedy"] = remedy_results
            else:
                # For general queries, get recent or popular incidents
                remedy_results = self.remedy_client.get_incidents_by_status("Open", limit=max_results)
                if remedy_results:
                    retrieved_data["remedy"] = remedy_results
        
        if "jira" in sources and "jira" not in retrieved_data:
            # Search for JIRA issues
            jira_results = self.jira_client.search_issues(query_text, limit=max_results)
            if jira_results:
                retrieved_data["jira"] = jira_results
        
        return retrieved_data
    
    def _process_documents(
        self,
        retrieved_data: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Process retrieved documents into a normalized format.
        
        Args:
            retrieved_data: Retrieved data grouped by source
            
        Returns:
            List of processed documents
        """
        processed_documents = []
        
        # Process Confluence content
        if "confluence" in retrieved_data:
            for content in retrieved_data["confluence"]:
                # For each Confluence page, get full content
                page_id = content.get("id")
                if page_id:
                    page_content = self.confluence_client.get_page_content(page_id)
                    if page_content:
                        processed_documents.append({
                            "source": "confluence",
                            "source_id": page_id,
                            "title": content.get("title", "Untitled Page"),
                            "url": content.get("_links", {}).get("webui", ""),
                            "content": page_content.get("content", ""),
                            "metadata": {
                                "space": content.get("space", {}).get("name", ""),
                                "last_updated": content.get("history", {}).get("lastUpdated", ""),
                                "author": content.get("history", {}).get("createdBy", {}).get("displayName", "")
                            }
                        })
        
        # Process Remedy incidents
        if "remedy" in retrieved_data:
            for incident in retrieved_data["remedy"]:
                processed_incident = self.remedy_client.process_incident_for_rag(incident)
                if processed_incident:
                    processed_documents.append({
                        "source": "remedy",
                        "source_id": processed_incident["metadata"].get("incident_number", ""),
                        "title": processed_incident["metadata"].get("summary", "Untitled Incident"),
                        "url": f"remedyurl/incident/{processed_incident['metadata'].get('incident_number', '')}",
                        "content": processed_incident["content"],
                        "metadata": processed_incident["metadata"]
                    })
        
        # Process JIRA issues
        if "jira" in retrieved_data:
            for issue in retrieved_data["jira"]:
                issue_key = issue.get("key", "")
                if issue_key:
                    # Get detailed issue content
                    issue_content = self.jira_client.get_issue_content(issue_key)
                    if issue_content:
                        processed_documents.append({
                            "source": "jira",
                            "source_id": issue_key,
                            "title": issue.get("fields", {}).get("summary", "Untitled Issue"),
                            "url": f"{self.jira_client.base_url}/browse/{issue_key}",
                            "content": issue_content.get("content", ""),
                            "metadata": {
                                "status": issue.get("fields", {}).get("status", {}).get("name", ""),
                                "assignee": issue.get("fields", {}).get("assignee", {}).get("displayName", ""),
                                "priority": issue.get("fields", {}).get("priority", {}).get("name", ""),
                                "project": issue.get("fields", {}).get("project", {}).get("name", "")
                            }
                        })
        
        return processed_documents
    
    def _retrieve_relevant_chunks(
        self,
        analyzed_query: Dict[str, Any],
        processed_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk documents and retrieve the most relevant chunks.
        
        Args:
            analyzed_query: Query analysis results
            processed_documents: List of processed documents
            
        Returns:
            List of relevant document chunks
        """
        all_chunks = []
        
        # Chunk each document
        for document in processed_documents:
            chunks = self.chunking_engine.chunk_document(document)
            all_chunks.extend(chunks)
        
        # If no chunks, return empty list
        if not all_chunks:
            return []
        
        # Use vector search to find relevant chunks
        query_text = analyzed_query["normalized_query"]
        relevant_chunks = self.vector_search.search(query_text, all_chunks, top_k=10)
        
        return relevant_chunks
    
    def _get_system_instructions(
        self,
        analyzed_query: Dict[str, Any],
        active_sources: List[str]
    ) -> str:
        """
        Get appropriate system instructions based on query and sources.
        
        Args:
            analyzed_query: Query analysis results
            active_sources: Active data sources
            
        Returns:
            System instructions for the LLM
        """
        # Start with common instructions
        instructions = """
        You are a professional corporate assistant with expertise in technical and business matters.
        Your role is to provide helpful, accurate, and concise information based on the context provided.
        Always maintain a professional tone while being friendly and approachable.
        
        When responding to technical questions, be clear and precise.
        If information is not available in the context, acknowledge this and provide general guidance if possible.
        Format your responses in a clear and structured way, using markdown for better readability.
        
        When appropriate, include source references from the context.
        """
        
        # Add source-specific instructions
        if "remedy" in active_sources:
            instructions += """
            
            When discussing IT incidents or tickets:
            - Clearly state the incident number, status, and priority
            - Summarize the issue and any resolution steps
            - Mention the assigned group or individual if available
            - Use technical terminology appropriately but explain complex concepts
            """
        
        if "jira" in active_sources:
            instructions += """
            
            When discussing tasks or projects:
            - Clearly reference the issue key (e.g., TASK-123)
            - Highlight the current status and any blockers
            - Mention assignees and deadlines if available
            - Organize information logically by priority or timeline
            """
        
        if "confluence" in active_sources:
            instructions += """
            
            When sharing information from documentation:
            - Cite the specific page or section
            - Maintain the structure and formatting of technical content
            - Accurately represent diagrams and tables in text form
            - Preserve the context of the information
            """
        
        return instructions
    
    def _format_sources(self, relevant_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format source references for the response.
        
        Args:
            relevant_chunks: The relevant chunks used for the response
            
        Returns:
            Formatted source references
        """
        sources = []
        seen_sources = set()
        
        for chunk in relevant_chunks:
            source_id = f"{chunk['source']}:{chunk['source_id']}"
            # Deduplicate sources
            if source_id not in seen_sources:
                sources.append({
                    "name": chunk.get("title", "Untitled"),
                    "url": chunk.get("url", ""),
                    "source_type": chunk.get("source", "unknown"),
                    "source_id": chunk.get("source_id", ""),
                    "relevance": chunk.get("relevance", 0)
                })
                seen_sources.add(source_id)
        
        # Sort by relevance
        sources.sort(key=lambda x: x["relevance"], reverse=True)
        
        return sources













import logging
import re
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import html2text

logger = logging.getLogger(__name__)

class ChunkingEngine:
    """
    Engine for chunking documents into smaller pieces for embedding and retrieval.
    Implements various chunking strategies including sliding window and semantic boundary detection.
    """
    
    def __init__(
        self,
        default_chunk_size: int = 1000,
        default_chunk_overlap: int = 200,
        respect_semantic_boundaries: bool = True
    ):
        """
        Initialize the chunking engine.
        
        Args:
            default_chunk_size: Default maximum chunk size in characters
            default_chunk_overlap: Default overlap between chunks in characters
            respect_semantic_boundaries: Whether to respect semantic boundaries when chunking
        """
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.respect_semantic_boundaries = respect_semantic_boundaries
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.ignore_tables = False
        
        logger.info("Chunking engine initialized")
    
    def chunk_document(
        self,
        document: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            document: Document to chunk
            chunk_size: Optional custom chunk size
            chunk_overlap: Optional custom chunk overlap
            
        Returns:
            List of document chunks with metadata
        """
        # Use default values if not provided
        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_chunk_overlap
        
        content = document.get("content", "")
        source = document.get("source", "unknown")
        
        # Skip empty documents
        if not content:
            logger.warning(f"Empty document from source {source}, skipping chunking")
            return []
        
        # Process content based on source and format
        if source == "confluence" and self._is_html(content):
            chunks = self._chunk_html_content(content, chunk_size, chunk_overlap)
        elif source == "remedy":
            chunks = self._chunk_with_semantic_boundaries(content, chunk_size, chunk_overlap)
        elif source == "jira":
            chunks = self._chunk_with_semantic_boundaries(content, chunk_size, chunk_overlap)
        else:
            # Default chunking for plain text
            chunks = self._chunk_with_semantic_boundaries(content, chunk_size, chunk_overlap)
        
        # Add document metadata to each chunk
        result_chunks = []
        for i, chunk_text in enumerate(chunks):
            result_chunks.append({
                "source": document.get("source", "unknown"),
                "source_id": document.get("source_id", ""),
                "title": document.get("title", "Untitled"),
                "url": document.get("url", ""),
                "chunk_id": f"{document.get('source_id', '')}_chunk_{i}",
                "content": chunk_text,
                "metadata": document.get("metadata", {}),
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
        
        logger.debug(f"Document from {source} chunked into {len(result_chunks)} chunks")
        return result_chunks
    
    def _is_html(self, content: str) -> bool:
        """
        Check if content is HTML.
        
        Args:
            content: Content to check
            
        Returns:
            True if content is HTML, False otherwise
        """
        return bool(re.search(r'<[^>]+>', content))
    
    def _chunk_html_content(
        self,
        content: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """
        Chunk HTML content respecting semantic boundaries.
        
        Args:
            content: HTML content to chunk
            chunk_size: Maximum chunk size
            chunk_overlap: Chunk overlap
            
        Returns:
            List of content chunks
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract text with some structure preserved
            markdown_text = self.html_converter.handle(str(soup))
            
            # Chunk the markdown text
            return self._chunk_with_semantic_boundaries(markdown_text, chunk_size, chunk_overlap)
            
        except Exception as e:
            logger.error(f"Error chunking HTML content: {str(e)}")
            # Fall back to simple chunking
            return self._simple_chunk(content, chunk_size, chunk_overlap)
    
    def _chunk_with_semantic_boundaries(
        self,
        content: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """
        Chunk content respecting semantic boundaries like paragraphs and headers.
        
        Args:
            content: Content to chunk
            chunk_size: Maximum chunk size
            chunk_overlap: Chunk overlap
            
        Returns:
            List of content chunks
        """
        if not self.respect_semantic_boundaries:
            return self._simple_chunk(content, chunk_size, chunk_overlap)
        
        # Split content into semantic units (paragraphs, sections, etc.)
        units = []
        
        # Split on markdown-style headers
        header_split = re.split(r'(#+\s+.+?\n)', content)
        
        for item in header_split:
            # Further split paragraphs
            if item.startswith('#'):
                # Headers are kept as their own units
                units.append(item)
            else:
                # Split paragraphs
                paragraphs = re.split(r'(\n\s*\n)', item)
                units.extend([p for p in paragraphs if p.strip()])
        
        # Combine units into chunks
        chunks = []
        current_chunk = ""
        
        for unit in units:
            # If adding this unit would exceed the chunk size and we already have content,
            # save the current chunk and start a new one
            if len(current_chunk) + len(unit) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                
                # Start new chunk with overlap from the end of the previous chunk
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + unit
                else:
                    current_chunk = unit
            else:
                # Add unit to current chunk
                current_chunk += unit
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Handle the case where a single unit is larger than the chunk size
        if not chunks:
            return self._simple_chunk(content, chunk_size, chunk_overlap)
        
        return chunks
    
    def _simple_chunk(
        self,
        content: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """
        Simple chunking implementation using sliding window.
        
        Args:
            content: Content to chunk
            chunk_size: Maximum chunk size
            chunk_overlap: Chunk overlap
            
        Returns:
            List of content chunks
        """
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            # Calculate end position for this chunk
            end = start + chunk_size
            
            # If we're not at the end of the content, try to find a good break point
            if end < len(content):
                # Look for natural break points: newline, period, comma, space
                for break_char in ['\n', '.', ',', ' ']:
                    break_pos = content.rfind(break_char, start, end)
                    if break_pos != -1:
                        end = break_pos + 1  # Include the break character
                        break
            
            # Extract the chunk
            chunk = content[start:end]
            chunks.append(chunk)
            
            # Calculate the next start position with overlap
            start = end - chunk_overlap
            if start < 0:
                start = 0
            
            # Avoid infinite loops if we couldn't make progress
            if start >= end:
                start = end
        
        return chunks














import logging
import os
import numpy as np
from typing import Dict, List, Any, Optional, Union
import pickle
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """
    Engine for generating and managing embeddings for document chunks.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        use_gpu: bool = False
    ):
        """
        Initialize the embedding engine.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            cache_dir: Directory to cache embeddings
            use_gpu: Whether to use GPU for embedding generation
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "data", "embedding_cache")
        self.use_gpu = use_gpu
        self.model = None
        self.embedding_cache = {}
        
        # Create cache directory if it doesn't exist
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Load cache if available
        self._load_cache()
        
        logger.info(f"Embedding engine initialized with model {model_name}")
    
    def _load_model(self):
        """Load the embedding model if not already loaded."""
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                
                # Configure device
                if self.use_gpu:
                    self.model = self.model.to('cuda')
                    logger.info("Using GPU for embeddings")
                else:
                    self.model = self.model.to('cpu')
                    logger.info("Using CPU for embeddings")
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                raise
    
    def _load_cache(self):
        """Load embedding cache from disk if available."""
        if not self.cache_dir:
            return
        
        cache_file = os.path.join(self.cache_dir, f"{self.model_name.replace('/', '_')}_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} embeddings from cache")
            except Exception as e:
                logger.error(f"Error loading embedding cache: {str(e)}")
                self.embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        if not self.cache_dir:
            return
        
        cache_file = os.path.join(self.cache_dir, f"{self.model_name.replace('/', '_')}_cache.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving embedding cache: {str(e)}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache first
        text_hash = hash(text)
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Load model if not loaded
        self._load_model()
        
        # Generate embedding
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            # Cache the embedding
            self.embedding_cache[text_hash] = embedding
            
            # Save cache periodically
            if len(self.embedding_cache) % 100 == 0:
                self._save_cache()
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector as fallback
            return np.zeros(384)  # Default dimension for all-MiniLM-L6-v2
    
    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Check which texts are not in cache
        text_hashes = [hash(text) for text in texts]
        uncached_indices = [i for i, text_hash in enumerate(text_hashes) if text_hash not in self.embedding_cache]
        
        if not uncached_indices:
            # All embeddings are in cache
            return [self.embedding_cache[text_hash] for text_hash in text_hashes]
        
        # Load model if not loaded
        self._load_model()
        
        # Get uncached texts
        uncached_texts = [texts[i] for i in uncached_indices]
        
        try:
            # Generate embeddings for uncached texts
            uncached_embeddings = self.model.encode(uncached_texts, show_progress_bar=False)
            
            # Normalize embeddings
            uncached_embeddings = [emb / np.linalg.norm(emb) for emb in uncached_embeddings]
            
            # Update cache
            for i, embedding in zip(uncached_indices, uncached_embeddings):
                self.embedding_cache[text_hashes[i]] = embedding
            
            # Save cache if there are many new embeddings
            if len(uncached_indices) > 10:
                self._save_cache()
            
            # Return all embeddings in the same order as the input texts
            return [self.embedding_cache[text_hash] for text_hash in text_hashes]
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            # Fall back to individual embedding generation
            return [self.get_embedding(text) for text in texts]

















import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from .embedding import EmbeddingEngine

logger = logging.getLogger(__name__)

class VectorSearch:
    """
    Vector similarity search for finding relevant document chunks.
    Supports both semantic search (embeddings) and lexical search (BM25/TF-IDF).
    """
    
    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        use_hybrid_search: bool = True,
        hybrid_alpha: float = 0.7  # Weight for semantic search vs lexical search
    ):
        """
        Initialize the vector search.
        
        Args:
            embedding_engine: Embedding engine to use
            use_hybrid_search: Whether to use hybrid search (semantic + lexical)
            hybrid_alpha: Weight for semantic search vs lexical search (0-1)
        """
        self.embedding_engine = embedding_engine
        self.use_hybrid_search = use_hybrid_search
        self.hybrid_alpha = hybrid_alpha
        self.tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        
        logger.info("Vector search initialized")
    
    def search(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search for the most relevant chunks for a query.
        
        Args:
            query: The search query
            chunks: List of document chunks to search
            top_k: Number of top results to return
            min_score: Minimum relevance score threshold
            
        Returns:
            List of relevant chunks with relevance scores
        """
        if not chunks:
            logger.warning("No chunks provided for search")
            return []
        
        if not query:
            logger.warning("Empty query provided for search")
            return []
        
        try:
            if self.use_hybrid_search:
                # Hybrid search combining semantic and lexical approaches
                return self._hybrid_search(query, chunks, top_k, min_score)
            else:
                # Semantic search only
                return self._semantic_search(query, chunks, top_k, min_score)
                
        except Exception as e:
            logger.error(f"Error during vector search: {str(e)}")
            # Fall back to basic search if there's an error
            return self._basic_search(query, chunks, top_k)
    
    def _semantic_search(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        min_score: float
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: The search query
            chunks: List of document chunks to search
            top_k: Number of top results to return
            min_score: Minimum relevance score threshold
            
        Returns:
            List of relevant chunks with relevance scores
        """
        # Get query embedding
        query_embedding = self.embedding_engine.get_embedding(query)
        
        # Extract content from chunks
        texts = [chunk.get("content", "") for chunk in chunks]
        
        # Get chunk embeddings
        chunk_embeddings = self.embedding_engine.get_embeddings(texts)
        
        # Convert embeddings to numpy arrays
        query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
        chunk_embeddings = np.array(chunk_embeddings).astype('float32')
        
        # Use FAISS for efficient similarity search
        dimension = query_embedding.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        if len(chunks) > 0:
            index.add(chunk_embeddings)
            
            # Search
            scores, indices = index.search(query_embedding, min(top_k, len(chunks)))
            
            # Get results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(chunks) and scores[0][i] >= min_score:
                    chunk = chunks[idx].copy()
                    chunk["relevance"] = float(scores[0][i])
                    results.append(chunk)
            
            return results
        else:
            return []
    
    def _lexical_search(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        min_score: float
    ) -> List[Dict[str, Any]]:
        """
        Perform lexical search using TF-IDF.
        
        Args:
            query: The search query
            chunks: List of document chunks to search
            top_k: Number of top results to return
            min_score: Minimum relevance score threshold
            
        Returns:
            List of relevant chunks with relevance scores
        """
        # Extract content from chunks
        texts = [chunk.get("content", "") for chunk in chunks]
        
        # Fit vectorizer on all texts including the query
        all_texts = texts + [query]
        try:
            self.tfidf_vectorizer.fit(all_texts)
            
            # Get document vectors
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            
            # Get query vector
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = (query_vector @ tfidf_matrix.T).toarray()[0]
            
            # Get top results
            results = []
            for idx in np.argsort(-similarities)[:top_k]:
                if similarities[idx] >= min_score:
                    chunk = chunks[idx].copy()
                    chunk["relevance"] = float(similarities[idx])
                    results.append(chunk)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during lexical search: {str(e)}")
            return self._basic_search(query, chunks, top_k)
    
    def _hybrid_search(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        min_score: float
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and lexical approaches.
        
        Args:
            query: The search query
            chunks: List of document chunks to search
            top_k: Number of top results to return
            min_score: Minimum relevance score threshold
            
        Returns:
            List of relevant chunks with relevance scores
        """
        try:
            # Get semantic search results
            semantic_results = self._semantic_search(query, chunks, top_k * 2, min_score / 2)
            
            # Get lexical search results
            lexical_results = self._lexical_search(query, chunks, top_k * 2, min_score / 2)
            
            # Combine results
            combined_results = {}
            
            # Add semantic results with weighted scores
            for result in semantic_results:
                chunk_id = result["chunk_id"]
                combined_results[chunk_id] = {
                    "chunk": result,
                    "score": result["relevance"] * self.hybrid_alpha
                }
            
            # Add lexical results with weighted scores
            for result in lexical_results:
                chunk_id = result["chunk_id"]
                if chunk_id in combined_results:
                    # Add lexical score to existing entry
                    combined_results[chunk_id]["score"] += result["relevance"] * (1 - self.hybrid_alpha)
                else:
                    # Create new entry
                    combined_results[chunk_id] = {
                        "chunk": result,
                        "score": result["relevance"] * (1 - self.hybrid_alpha)
                    }
            
            # Convert to list and sort by score
            result_list = []
            for chunk_id, data in combined_results.items():
                result = data["chunk"].copy()
                result["relevance"] = data["score"]
                result_list.append(result)
            
            # Sort by relevance score
            result_list.sort(key=lambda x: x["relevance"], reverse=True)
            
            # Filter by minimum score and limit to top_k
            return [r for r in result_list[:top_k] if r["relevance"] >= min_score]
            
        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            return self._basic_search(query, chunks, top_k)
    
    def _basic_search(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Perform basic keyword search as a fallback.
        
        Args:
            query: The search query
            chunks: List of document chunks to search
            top_k: Number of top results to return
            
        Returns:
            List of relevant chunks
        """
        # Simple keyword matching as a fallback
        query_terms = query.lower().split()
        results = []
        
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            # Count matches
            matches = sum(1 for term in query_terms if term in content)
            if matches > 0:
                result = chunk.copy()
                result["relevance"] = matches / len(query_terms)
                results.append(result)
        
        # Sort by relevance score
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Return top_k results
        return results[:top_k]



















import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Builder for constructing context from retrieved chunks for LLM prompting.
    """
    
    def __init__(
        self,
        max_context_size: int = 16000,
        add_source_metadata: bool = True,
        format_for_readability: bool = True
    ):
        """
        Initialize the context builder.
        
        Args:
            max_context_size: Maximum context size in characters
            add_source_metadata: Whether to add source metadata to the context
            format_for_readability: Whether to format the context for readability
        """
        self.max_context_size = max_context_size
        self.add_source_metadata = add_source_metadata
        self.format_for_readability = format_for_readability
        
        logger.info("Context builder initialized")
    
    def build_context(
        self,
        chunks: List[Dict[str, Any]],
        analyzed_query: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build context from retrieved chunks.
        
        Args:
            chunks: List of relevant chunks
            analyzed_query: Optional query analysis results
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found."
        
        # Sort chunks by relevance
        sorted_chunks = sorted(chunks, key=lambda x: x.get("relevance", 0), reverse=True)
        
        # Group chunks by source to improve readability
        source_groups = {}
        for chunk in sorted_chunks:
            source = chunk.get("source", "unknown")
            source_id = chunk.get("source_id", "")
            key = f"{source}:{source_id}"
            
            if key not in source_groups:
                source_groups[key] = {
                    "source": source,
                    "source_id": source_id,
                    "title": chunk.get("title", ""),
                    "url": chunk.get("url", ""),
                    "chunks": []
                }
            
            source_groups[key]["chunks"].append(chunk)
        
        # Build context for each source
        context_parts = []
        current_size = 0
        
        for group_key, group in source_groups.items():
            source_context = self._build_source_context(group)
            
            # Check if adding this source would exceed the maximum context size
            if current_size + len(source_context) > self.max_context_size:
                # If we already have some context, stop adding more
                if context_parts:
                    break
                
                # If we don't have any context yet, include at least one chunk
                source_context = self._truncate_context(source_context, self.max_context_size)
            
            context_parts.append(source_context)
            current_size += len(source_context)
        
        # Combine context parts
        combined_context = "\n\n".join(context_parts)
        
        # Apply final formatting
        if self.format_for_readability:
            combined_context = self._format_for_readability(combined_context)
        
        logger.debug(f"Built context of size {len(combined_context)} from {len(chunks)} chunks")
        return combined_context
    
    def _build_source_context(self, source_group: Dict[str, Any]) -> str:
        """
        Build context for a specific source.
        
        Args:
            source_group: Group of chunks from the same source
            
        Returns:
            Formatted source context
        """
        source = source_group["source"]
        title = source_group["title"]
        chunks = source_group["chunks"]
        
        # Start with source header
        if self.add_source_metadata:
            context = f"SOURCE: {source.upper()} - {title}\n"
            if source_group["url"]:
                context += f"URL: {source_group['url']}\n"
        else:
            context = ""
        
        # Add chunks content
        chunk_texts = []
        for chunk in chunks:
            chunk_text = chunk.get("content", "").strip()
            if chunk_text:
                chunk_texts.append(chunk_text)
        
        # Join chunk texts, avoiding duplicates
        unique_texts = []
        for text in chunk_texts:
            # Check if this text is very similar to already included texts
            is_duplicate = False
            for included_text in unique_texts:
                if self._text_similarity(text, included_text) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_texts.append(text)
        
        context += "\n\n".join(unique_texts)
        
        return context
    
    def _truncate_context(self, context: str, max_size: int) -> str:
        """
        Truncate context to maximum size, preserving semantic boundaries.
        
        Args:
            context: Context to truncate
            max_size: Maximum size
            
        Returns:
            Truncated context
        """
        if len(context) <= max_size:
            return context
        
        # Try to truncate at paragraph boundary
        truncated = context[:max_size]
        last_para = truncated.rfind("\n\n")
        if last_para != -1 and last_para > max_size * 0.8:
            return context[:last_para] + "\n\n[Content truncated due to length...]"
        
        # Try to truncate at sentence boundary
        last_sentence = re.finditer(r'[.!?]\s+', truncated)
        last_pos = None
        for match in last_sentence:
            last_pos = match.end()
        
        if last_pos and last_pos > max_size * 0.8:
            return context[:last_pos] + "\n\n[Content truncated due to length...]"
        
        # Fall back to simple truncation
        return truncated + "..."
    
    def _format_for_readability(self, context: str) -> str:
        """
        Format context for better readability.
        
        Args:
            context: Context to format
            
        Returns:
            Formatted context
        """
        # Clean up excessive whitespace
        context = re.sub(r'\n{3,}', '\n\n', context)
        
        # Make sure source headers stand out
        context = re.sub(r'SOURCE: ([^\n]+)', r'### SOURCE: \1', context)
        
        # Ensure proper spacing around sections
        context = re.sub(r'(#+\s+[^\n]+)\n(?!\n)', r'\1\n\n', context)
        
        return context
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # For short texts, check if one is contained in the other
        if len(text1) < 100 or len(text2) < 100:
            if text1 in text2 or text2 in text1:
                return 0.9
        
        # For longer texts, use a simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)















{% extends "base.html" %}

{% block title %}Enterprise Knowledge Bot{% endblock %}

{% block content %}
<div class="chat-container">
    <div class="chat-header">
        <h1>Enterprise Knowledge Bot</h1>
        <div class="data-source-toggle">
            <span>Data Sources:</span>
            <div class="toggle-group">
                <label class="toggle">
                    <input type="checkbox" id="confluence-toggle" checked>
                    <span class="toggle-label">Confluence</span>
                </label>
                <label class="toggle">
                    <input type="checkbox" id="remedy-toggle" checked>
                    <span class="toggle-label">Remedy</span>
                </label>
                <label class="toggle">
                    <input type="checkbox" id="jira-toggle" checked>
                    <span class="toggle-label">JIRA</span>
                </label>
            </div>
        </div>
    </div>
    
    <div class="messages-container" id="messages-container">
        <div class="message system-message">
            <div class="message-content">
                <p>👋 Hello! I'm your Enterprise Knowledge Assistant. I can help you with information from Confluence, Remedy tickets, and JIRA issues. What would you like to know?</p>
            </div>
        </div>
        <!-- Messages will be added here dynamically -->
    </div>
    
    <div class="chat-footer">
        <form id="chat-form">
            <div class="input-container">
                <textarea id="user-input" placeholder="Ask a question..." rows="1"></textarea>
                <button type="submit" id="send-button">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-send">
                        <line x1="22" y1="2" x2="11" y2="13"></line>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                    </svg>
                </button>
            </div>
        </form>
        <div class="typing-indicator hidden" id="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    </div>
</div>

<div class="sources-panel" id="sources-panel">
    <div class="sources-header">
        <h3>Sources</h3>
        <button id="close-sources">×</button>
    </div>
    <div class="sources-content" id="sources-content">
        <!-- Sources will be added here dynamically -->
        <p class="no-sources">No sources to display.</p>
    </div>
</div>

<template id="message-template">
    <div class="message">
        <div class="message-content">
            <p></p>
        </div>
        <div class="message-footer">
            <div class="message-time"></div>
            <div class="message-actions">
                <button class="view-sources-btn hidden">View Sources</button>
            </div>
        </div>
    </div>
</template>

<template id="source-item-template">
    <div class="source-item">
        <div class="source-icon"></div>
        <div class="source-details">
            <h4 class="source-title"></h4>
            <p class="source-type"></p>
        </div>
        <a class="source-link" target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-external-link">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                <polyline points="15 3 21 3 21 9"></polyline>
                <line x1="10" y1="14" x2="21" y2="3"></line>
            </svg>
        </a>
    </div>
</template>
{% endblock %}

{% block scripts %}
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.5/dist/purify.min.js"></script>
<script>
    document.addEventListener('DOMContentLoaded', () => {
        const messagesContainer = document.getElementById('messages-container');
        const chatForm = document.getElementById('chat-form');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');
        const typingIndicator = document.getElementById('typing-indicator');
        
        const confluenceToggle = document.getElementById('confluence-toggle');
        const remedyToggle = document.getElementById('remedy-toggle');
        const jiraToggle = document.getElementById('jira-toggle');
        
        const sourcesPanel = document.getElementById('sources-panel');
        const closeSourcesBtn = document.getElementById('close-sources');
        const sourcesContent = document.getElementById('sources-content');
        
        let currentSources = [];
        
        // Handle textarea auto-resize
        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = (userInput.scrollHeight) + 'px';
            
            // Limit to 5 rows max
            const lineHeight = parseInt(window.getComputedStyle(userInput).lineHeight);
            const maxHeight = lineHeight * 5;
            if (userInput.scrollHeight > maxHeight) {
                userInput.style.height = maxHeight + 'px';
                userInput.style.overflowY = 'auto';
            } else {
                userInput.style.overflowY = 'hidden';
            }
        });
        
        // Handle form submission
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const message = userInput.value.trim();
            if (!message) return;
            
            // Disable input while processing
            userInput.disabled = true;
            sendButton.disabled = true;
            
            // Add user message to UI
            addMessage(message, 'user');
            
            // Clear input
            userInput.value = '';
            userInput.style.height = 'auto';
            
            // Show typing indicator
            typingIndicator.classList.remove('hidden');
            
            // Get selected data sources
            const selectedSources = [];
            if (confluenceToggle.checked) selectedSources.push('confluence');
            if (remedyToggle.checked) selectedSources.push('remedy');
            if (jiraToggle.checked) selectedSources.push('jira');
            
            try {
                // Send request to API
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: message,
                        sources: selectedSources
                    })
                });
                
                const data = await response.json();
                
                // Hide typing indicator
                typingIndicator.classList.add('hidden');
                
                if (data.error) {
                    // Handle error
                    addMessage(`Error: ${data.error}`, 'system', true);
                } else {
                    // Add response to UI
                    const messageElement = addMessage(data.response, 'assistant', false, data.processing_time);
                    
                    // Store sources for this message
                    if (data.sources && data.sources.length > 0) {
                        currentSources = data.sources;
                        
                        // Add view sources button
                        const footer = messageElement.querySelector('.message-footer');
                        const viewSourcesBtn = footer.querySelector('.view-sources-btn');
                        viewSourcesBtn.classList.remove('hidden');
                        viewSourcesBtn.addEventListener('click', () => {
                            showSources(currentSources);
                        });
                    }
                }
            } catch (error) {
                console.error('Error:', error);
                addMessage('Sorry, there was an error processing your request. Please try again.', 'system', true);
                typingIndicator.classList.add('hidden');
            } finally {
                // Re-enable input
                userInput.disabled = false;
                sendButton.disabled = false;
                userInput.focus();
            }
        });
        
        // Function to add a message to the UI
        function addMessage(content, role, isError = false, processingTime = null) {
            const template = document.getElementById('message-template');
            const messageElement = template.content.cloneNode(true).querySelector('.message');
            
            // Add appropriate class based on role
            messageElement.classList.add(`${role}-message`);
            if (isError) messageElement.classList.add('error-message');
            
            const messageContent = messageElement.querySelector('.message-content p');
            
            // For assistant messages, render markdown and sanitize
            if (role === 'assistant') {
                const sanitizedHtml = DOMPurify.sanitize(marked.parse(content));
                messageContent.innerHTML = sanitizedHtml;
            } else {
                messageContent.textContent = content;
            }
            
            // Add timestamp
            const messageTime = messageElement.querySelector('.message-time');
            const now = new Date();
            messageTime.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            
            // Add processing time if available
            if (processingTime !== null) {
                const processingTimeSpan = document.createElement('span');
                processingTimeSpan.classList.add('processing-time');
                processingTimeSpan.textContent = ` (${processingTime.toFixed(2)}s)`;
                messageTime.appendChild(processingTimeSpan);
            }
            
            // Add message to container
            messagesContainer.appendChild(messageElement);
            
            // Scroll to bottom
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            return messageElement;
        }
        
        // Function to show sources panel
        function showSources(sources) {
            // Clear previous sources
            sourcesContent.innerHTML = '';
            
            if (!sources || sources.length === 0) {
                const noSourcesElement = document.createElement('p');
                noSourcesElement.classList.add('no-sources');
                noSourcesElement.textContent = 'No sources to display.';
                sourcesContent.appendChild(noSourcesElement);
            } else {
                const template = document.getElementById('source-item-template');
                
                sources.forEach(source => {
                    const sourceElement = template.content.cloneNode(true).querySelector('.source-item');
                    
                    // Set source icon based on type
                    const sourceIcon = sourceElement.querySelector('.source-icon');
                    switch (source.source_type) {
                        case 'confluence':
                            sourceIcon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>';
                            sourceIcon.classList.add('confluence-icon');
                            break;
                        case 'remedy':
                            sourceIcon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path><polyline points="13 2 13 9 20 9"></polyline></svg>';
                            sourceIcon.classList.add('remedy-icon');
                            break;
                        case 'jira':
                            sourceIcon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5V3m0 18v-2M5 12H3m18 0h-2M6 6l1.5 1.5M6 18l1.5-1.5M18 6l-1.5 1.5m1.5 10.5l-1.5-1.5"/></svg>';
                            sourceIcon.classList.add('jira-icon');
                            break;
                        default:
                            sourceIcon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';
                    }
                    
                    // Set title and type
                    sourceElement.querySelector('.source-title').textContent = source.name;
                    sourceElement.querySelector('.source-type').textContent = 
                        `${source.source_type.charAt(0).toUpperCase() + source.source_type.slice(1)} ${source.source_id}`;
                    
                    // Set link if available
                    const sourceLink = sourceElement.querySelector('.source-link');
                    if (source.url) {
                        sourceLink.href = source.url;
                    } else {
                        sourceLink.style.display = 'none';
                    }
                    
                    sourcesContent.appendChild(sourceElement);
                });
            }
            
            // Show panel
            sourcesPanel.classList.add('active');
        }
        
        // Close sources panel
        closeSourcesBtn.addEventListener('click', () => {
            sourcesPanel.classList.remove('active');
        });
        
        // Handle Escape key to close sources panel
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && sourcesPanel.classList.contains('active')) {
                sourcesPanel.classList.remove('active');
            }
        });
    });
</script>
{% endblock %}
















<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Enterprise Knowledge Bot{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    {% block head %}{% endblock %}
</head>
<body>
    <div class="app-container">
        {% block content %}{% endblock %}
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <script>
        // Initialize highlight.js
        document.addEventListener('DOMContentLoaded', (event) => {
            document.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightBlock(block);
            });
        });
    </script>
    {% block scripts %}{% endblock %}
</body>
</html>












:root {
    /* Color palette */
    --primary-color: #4f46e5;
    --primary-light: #818cf8;
    --primary-dark: #3730a3;
    --secondary-color: #06b6d4;
    --secondary-light: #22d3ee;
    --secondary-dark: #0e7490;
    --gray-50: #f9fafb;
    --gray-100: #f3f4f6;
    --gray-200: #e5e7eb;
    --gray-300: #d1d5db;
    --gray-400: #9ca3af;
    --gray-500: #6b7280;
    --gray-600: #4b5563;
    --gray-700: #374151;
    --gray-800: #1f2937;
    --gray-900: #111827;
    
    /* Source colors */
    --confluence-color: #0052cc;
    --remedy-color: #6554c0;
    --jira-color: #0052cc;
    
    /* UI elements */
    --body-bg: #f9fafb;
    --chat-bg: white;
    --header-bg: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    --user-message-bg: var(--gray-100);
    --assistant-message-bg: white;
    --system-message-bg: rgba(79, 70, 229, 0.1);
    --error-message-bg: rgba(220, 38, 38, 0.1);
    
    /* Spacing */
    --container-padding: 1.5rem;
    --message-spacing: 1rem;
    --message-padding: 1rem;
    --border-radius: 0.75rem;
    
    /* Typography */
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    --font-size-small: 0.875rem;
    --font-size-base: 1rem;
    --font-size-large: 1.125rem;
    --font-size-xl: 1.25rem;
    --font-size-2xl: 1.5rem;
    
    /* Transitions */
    --transition-fast: 150ms ease;
    --transition-normal: 250ms ease;
    
    /* Shadows */
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: var(--font-family);
    background-color: var(--body-bg);
    color: var(--gray-800);
    line-height: 1.5;
    height: 100vh;
    overflow: hidden;
}

.app-container {
    height: 100vh;
    max-width: 100%;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
}

.chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    height: 100%;
    background-color: var(--chat-bg);
    box-shadow: var(--shadow-md);
    max-width: 1200px;
    margin: 0 auto;
    width: 100%;
}

/* Header */
.chat-header {
    background: var(--header-bg);
    color: white;
    padding: var(--container-padding);
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-top-left-radius: 0.5rem;
    border-top-right-radius: 0.5rem;
}

.chat-header h1 {
    font-size: var(--font-size-2xl);
    font-weight: 600;
    margin: 0;
}

.data-source-toggle {
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: var(--font-size-small);
}

.toggle-group {
    display: flex;
    gap: 0.5rem;
}

.toggle {
    display: inline-flex;
    align-items: center;
    background-color: rgba(255, 255, 255, 0.2);
    border-radius: 9999px;
    padding: 0.25rem 0.75rem;
    cursor: pointer;
    transition: all var(--transition-fast);
}

.toggle:hover {
    background-color: rgba(255, 255, 255, 0.3);
}

.toggle input {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
}

.toggle-label {
    font-weight: 500;
}

.toggle input:checked + .toggle-label {
    color: white;
}

.toggle input:not(:checked) + .toggle-label {
    color: rgba(255, 255, 255, 0.7);
}

/* Messages container */
.messages-container {
    flex: 1;
    overflow-y: auto;
    padding: var(--container-padding);
    display: flex;
    flex-direction: column;
    gap: var(--message-spacing);
}

.message {
    display: flex;
    flex-direction: column;
    max-width: 90%;
    border-radius: var(--border-radius);
    animation: message-appear 0.3s ease-out;
}

@keyframes message-appear {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.user-message {
    align-self: flex-end;
    background-color: var(--user-message-bg);
    border-bottom-right-radius: 0;
}

.assistant-message {
    align-self: flex-start;
    background-color: var(--assistant-message-bg);
    border: 1px solid var(--gray-200);
    border-bottom-left-radius: 0;
    box-shadow: var(--shadow-sm);
}

.system-message {
    align-self: center;
    background-color: var(--system-message-bg);
    border: 1px solid rgba(79, 70, 229, 0.2);
    width: fit-content;
    max-width: 80%;
}

.error-message {
    background-color: var(--error-message-bg);
    border: 1px solid rgba(220, 38, 38, 0.2);
}

.message-content {
    padding: var(--message-padding);
}

.message-content p {
    margin-bottom: 0.5rem;
}

.message-content p:last-child {
    margin-bottom: 0;
}

.message-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.25rem var(--message-padding);
    border-top: 1px solid var(--gray-200);
    font-size: var(--font-size-small);
    color: var(--gray-500);
}

.message-time .processing-time {
    color: var(--gray-400);
    font-size: 0.75rem;
}

.message-actions {
    display: flex;
    gap: 0.5rem;
}

.view-sources-btn {
    background: none;
    border: none;
    color: var(--primary-color);
    font-size: var(--font-size-small);
    cursor: pointer;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    transition: background-color var(--transition-fast);
}

.view-sources-btn:hover {
    background-color: rgba(79, 70, 229, 0.1);
}

.hidden {
    display: none;
}

/* Message content styling */
.assistant-message .message-content {
    line-height: 1.6;
}

.assistant-message .message-content h1,
.assistant-message .message-content h2,
.assistant-message .message-content h3,
.assistant-message .message-content h4,
.assistant-message .message-content h5,
.assistant-message .message-content h6 {
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
    font-weight: 600;
}

.assistant-message .message-content h1 {
    font-size: 1.5rem;
    border-bottom: 1px solid var(--gray-200);
    padding-bottom: 0.5rem;
}

.assistant-message .message-content h2 {
    font-size: 1.25rem;
}

.assistant-message .message-content h3 {
    font-size: 1.125rem;
}

.assistant-message .message-content p {
    margin-bottom: 1rem;
}

.assistant-message .message-content ul,
.assistant-message .message-content ol {
    margin-bottom: 1rem;
    padding-left: 1.5rem;
}

.assistant-message .message-content li {
    margin-bottom: 0.5rem;
}

.assistant-message .message-content pre {
    background-color: var(--gray-50);
    border: 1px solid var(--gray-200);
    border-radius: 0.25rem;
    padding: 0.75rem;
    margin-bottom: 1rem;
    overflow-x: auto;
}

.assistant-message .message-content code {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 0.875rem;
    padding: 0.15rem 0.25rem;
    border-radius: 0.25rem;
    background-color: var(--gray-100);
}

.assistant-message .message-content pre code {
    background-color: transparent;
    padding: 0;
    border-radius: 0;
}

.assistant-message .message-content table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1rem;
}

.assistant-message .message-content th,
.assistant-message .message-content td {
    padding: 0.5rem;
    border: 1px solid var(--gray-300);
    text-align: left;
}

.assistant-message .message-content th {
    background-color: var(--gray-100);
    font-weight: 600;
}

/* Footer */
.chat-footer {
    padding: var(--container-padding);
    border-top: 1px solid var(--gray-200);
    background-color: white;
}

.input-container {
    display: flex;
    gap: 0.5rem;
    background-color: white;
    border: 1px solid var(--gray-300);
    border-radius: var(--border-radius);
    padding: 0.75rem;
    transition: border-color var(--transition-fast);
}

.input-container:focus-within {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.25);
}

#user-input {
    flex: 1;
    border: none;
    outline: none;
    font-family: var(--font-family);
    font-size: var(--font-size-base);
    resize: none;
    max-height: 150px;
    overflow-y: auto;
}

#send-button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    border-radius: 0.5rem;
    width: 2.5rem;
    height: 2.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: background-color var(--transition-fast);
}

#send-button:hover {
    background-color: var(--primary-dark);
}

#send-button:disabled {
    background-color: var(--gray-400);
    cursor: not-allowed;
}

/* Typing indicator */
.typing-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 0.5rem;
    height: 2rem;
}

.typing-indicator span {
    height: 0.5rem;
    width: 0.5rem;
    background-color: var(--gray-400);
    border-radius: 50%;
    display: inline-block;
    margin: 0 0.15rem;
    animation: typing 1.5s infinite ease-in-out;
}

.typing-indicator span:nth-child(1) {
    animation-delay: 0s;
}

.typing-indicator span:nth-child(2) {
    animation-delay: 0.3s;
}

.typing-indicator span:nth-child(3) {
    animation-delay: 0.6s;
}

@keyframes typing {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-0.5rem);
    }
}

/* Sources panel */
.sources-panel {
    position: fixed;
    top: 0;
    right: -400px;
    width: 400px;
    height: 100vh;
    background-color: white;
    box-shadow: -5px 0 15px rgba(0, 0, 0, 0.1);
    transition: right var(--transition-normal);
    z-index: 1000;
    display: flex;
    flex-direction: column;
}

.sources-panel.active {
    right: 0;
}

.sources-header {
    padding: 1rem;
    background-color: var(--gray-800);
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.sources-header h3 {
    font-size: var(--font-size-xl);
    font-weight: 600;
    margin: 0;
}

#close-sources {
    background: none;
    border: none;
    color: white;
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0.25rem;
    line-height: 1;
}

.sources-content {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
}

.no-sources {
    color: var(--gray-500);
    text-align: center;
    margin-top: 2rem;
}

.source-item {
    display: flex;
    align-items: flex-start;
    padding: 1rem;
    border: 1px solid var(--gray-200);
    border-radius: var(--border-radius);
    margin-bottom: 1rem;
    background-color: var(--gray-50);
    transition: transform var(--transition-fast), box-shadow var(--transition-fast);
}

.source-item:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.source-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2.5rem;
    height: 2.5rem;
    background-color: white;
    border-radius: 0.5rem;
    margin-right: 1rem;
    box-shadow: var(--shadow-sm);
}

.confluence-icon {
    color: var(--confluence-color);
}

.remedy-icon {
    color: var(--remedy-color);
}

.jira-icon {
    color: var(--jira-color);
}

.source-details {
    flex: 1;
}

.source-title {
    font-weight: 600;
    margin-bottom: 0.25rem;
    font-size: var(--font-size-base);
}

.source-type {
    color: var(--gray-500);
    font-size: var(--font-size-small);
}

.source-link {
    color: var(--primary-color);
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: 0.25rem;
    transition: background-color var(--transition-fast);
}

.source-link:hover {
    background-color: rgba(79, 70, 229, 0.1);
}

/* Responsive styles */
@media (max-width: 768px) {
    .chat-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .message {
        max-width: 100%;
    }
    
    .sources-panel {
        width: 100%;
        right: -100%;
    }
}














import logging
import json
import time
import re
from typing import Dict, List, Any, Optional, Union
import requests
from bs4 import BeautifulSoup
import html2text

logger = logging.getLogger(__name__)

class HTMLFilter:
    """Filter for converting HTML to plain text."""
    def __init__(self, data):
        self.text = data + " "

class ConfluenceClient:
    """
    Client for Confluence REST API operations with comprehensive error handling and
    advanced querying.
    """
    
    def __init__(
        self, 
        base_url: str, 
        username: str = None, 
        password: str = None, 
        ssl_verify: bool = False
    ):
        """
        Initialize the Confluence client with authentication details and SSL options.
        
        Args:
            base_url: The base URL of the Confluence instance (e.g., https://company.atlassian.net)
            username: The username for authentication
            password: The API token for authentication
            ssl_verify: Whether to verify SSL certificates (set to False to disable verification)
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, password) if username and password else None
        self.ssl_verify = ssl_verify
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Atlassian-Token": "no-check"
        }
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.ignore_tables = False
        
        logger.info(f"Initialized Confluence client for {self.base_url}")
    
    def test_connection(self) -> bool:
        """
        Test the connection to Confluence API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            logger.info("Testing connection to Confluence...")
            response = requests.get(
                f"{self.base_url}/rest/api/serverInfo",
                auth=self.auth,
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                server_info = response.json()
                logger.info(f"Connection to Confluence successful! Server version: {server_info.get('version', 'unknown')}")
                return True
            else:
                logger.error(f"Failed to connect to Confluence. Status code: {response.status_code}")
                if hasattr(response, 'text'):
                    logger.error(f"Response content: {response.text}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_content_by_id(self, content_id: str, expand: str = None) -> Optional[Dict[str, Any]]:
        """
        Get a specific content by its ID with optional expansion parameters.
        
        Args:
            content_id: The ID of the content to retrieve
            expand: Comma-separated list of properties to expand
            
        Returns:
            dict: Content data or None if not found/error
        """
        try:
            logger.info(f"Fetching content with ID: {content_id}")
            params = {}
            if expand:
                params["expand"] = expand
            
            response = requests.get(
                f"{self.base_url}/rest/api/content/{content_id}",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                content = response.json()
                logger.info(f"Successfully retrieved content: {content.get('title', 'unknown title')}")
                return content
            else:
                logger.error(f"Failed to get content ID {content_id}. Status code: {response.status_code}")
                if hasattr(response, 'text'):
                    logger.error(f"Response content: {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error retrieving content: {str(e)}")
            return None
    
    def search_content(
        self, 
        cql: str = None, 
        title: str = None, 
        content_type: str = "page", 
        limit: int = 10,
        expand: str = None,
        start: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search for content using CQL or specific parameters.
        
        Args:
            cql: Confluence Query Language string
            title: Title to search for
            content_type: Content type to search for (default: page)
            limit: Maximum number of results to return
            expand: Comma-separated list of properties to expand
            start: Starting index for pagination
            
        Returns:
            list: List of content items or empty list if none found/error
        """
        try:
            logger.info(f"Searching for content with CQL: {cql}")
            params = {}
            
            # Build CQL if not provided
            if not cql:
                query_parts = []
                if title:
                    # Escape special characters in title
                    safe_title = title.replace('"', '\\"')
                    query_parts.append(f'title ~ "{safe_title}"')
                
                if query_parts:
                    params["cql"] = " AND ".join(query_parts)
                else:
                    # Default query if nothing specified
                    params["cql"] = f'type = "{content_type}"'
            else:
                params["cql"] = cql
            
            if content_type:
                # Add content type filter if not in CQL
                if "type =" not in params["cql"]:
                    params["cql"] += f' AND type = "{content_type}"'
            
            # Add other parameters
            params["limit"] = limit
            params["start"] = start
            if expand:
                params["expand"] = expand
            
            logger.info(f"Searching with params: {params}")
            response = requests.get(
                f"{self.base_url}/rest/api/content/search",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                search_results = response.json()
                results = search_results.get("results", [])
                logger.info(f"Search returned {len(results)} results")
                return results
            else:
                logger.error(f"Failed to search content. Status code: {response.status_code}")
                if hasattr(response, 'text'):
                    logger.error(f"Response content: {response.text}")
                return []
                
        except requests.RequestException as e:
            logger.error(f"Error searching content: {str(e)}")
            return []
    
    def get_page_content(self, page_id: str, expand: str = "body.storage,metadata.labels") -> Optional[Dict[str, Any]]:
        """
        Get the content of a page in a suitable format for RAG.
        This extracts and processes the content to be more suitable for embeddings.
        
        Args:
            page_id: The ID of the page
        
        Returns:
            dict: Processed page content with metadata
        """
        try:
            page = self.get_content_by_id(page_id, expand=expand)
            if not page:
                return None
            
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "space": page.get("_expandable", {}).get("space", "") if "_expandable" in page else None,
                "created": page.get("created"),
                "updated": page.get("updated"),
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])] if "metadata" in page and "labels" in page["metadata"] else None,
            }
            
            # Get raw HTML content
            content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process content
            if content:
                html_filter = HTMLFilter(content)
                soup = BeautifulSoup(content, 'html.parser')
                
                # Clean up and extract text
                plain_text = self.html_converter.handle(str(soup))
                
                return {
                    "id": page_id,
                    "metadata": metadata,
                    "content": plain_text,
                    "raw_html": content
                }
            else:
                return {
                    "id": page_id,
                    "metadata": metadata,
                    "content": "",
                    "raw_html": ""
                }
                
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None
    
    def get_all_content(self, content_type: str = "page", limit: int = 25, expand: None = None) -> List[Dict[str, Any]]:
        """
        Retrieve all content of specified type with pagination handling.
        
        Args:
            content_type: Type of content to retrieve (default: page)
            limit: Maximum number of results per request
            expand: Properties to expand in results
        
        Returns:
            list: List of content items
        """
        all_content = []
        start = 0
        
        logger.info(f"Retrieving all {content_type} content")
        
        while True:
            # Get batch of spaces
            spaces = self.search_content(
                cql=f'type="{content_type}"',
                limit=limit,
                start=start,
                expand=expand
            )
            
            if not spaces:
                break
            
            all_content.extend(spaces)
            
            # Check if there are more pages
            if len(spaces) < limit:
                break
            
            # Get next page
            start += limit
        
        # Check the "_links" for a "next" link
        links = spaces.get("_links", {})
        if not links.get("next"):
            break
        
        logger.info(f"Retrieved a total of {len(all_content)} {content_type}s")
        return all_content
    
    def get_spaces(self, limit: int = 25, start: int = 0) -> List[Dict[str, Any]]:
        """
        Get all spaces the user has access to.
        
        Args:
            limit: Maximum number of results per request
            start: Starting index for pagination
            
        Returns:
            list: List of spaces
        """
        try:
            logger.info("Fetching spaces...")
            response = requests.get(
                f"{self.base_url}/rest/api/space",
                auth=self.auth,
                headers=self.headers,
                params={
                    "limit": limit,
                    "start": start
                },
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                result = response.json()
                spaces = result.get("results", [])
                logger.info(f"Successfully retrieved {len(spaces)} spaces")
                return spaces
            else:
                logger.error(f"Failed to get spaces. Status code: {response.status_code}")
                if hasattr(response, 'text'):
                    logger.error(f"Response content: {response.text}")
                return []
                
        except requests.RequestException as e:
            logger.error(f"Error retrieving spaces: {str(e)}")
            return []
    
    def extract_text_from_html(self, html: str) -> str:
        """
        Extract plain text from HTML content.
        
        Args:
            html: HTML content
            
        Returns:
            str: Extracted plain text
        """
        if not html:
            return ""
        
        try:
            # Use BeautifulSoup to parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract text from soup
            text = soup.get_text(separator="\n", strip=True)
            
            # Clean up newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {str(e)}")
            return html
    
    def process_for_embeddings(self, content: Dict[str, Any]) -> str:
        """
        Process content into a format suitable for generating embeddings.
        
        Args:
            content: Content data
            
        Returns:
            str: Processed text for embeddings
        """
        if not content:
            return ""
        
        processed_text = f"Title: {content.get('title', '')}\n"
        
        if "type" in content:
            processed_text += f"Type: {content.get('type', '')}\n"
        
        if "space" in content and isinstance(content["space"], dict):
            processed_text += f"Space: {content['space'].get('name', '')}\n"
        
        html_content = content.get("body", {}).get("storage", {}).get("value", "")
        if html_content:
            extracted_text = self.extract_text_from_html(html_content)
            processed_text += f"\nContent:\n{extracted_text}"
        
        return processed_text














import logging
import json
import time
from datetime import datetime, timedelta
import re
from typing import Dict, List, Any, Optional, Union
from urllib.parse import quote
import requests

logger = logging.getLogger(__name__)

class RemedyClient:
    """
    Client for BMC Remedy REST API operations with comprehensive error handling and
    advanced querying.
    """
    
    def __init__(
        self, 
        server_url: str, 
        username: str = None, 
        password: str = None, 
        ssl_verify: bool = False
    ):
        """
        Initialize the Remedy client with server and authentication details.
        
        Args:
            server_url: The base URL of the Remedy server (e.g., https://company-restapi.onbmc.com)
            username: Username for authentication (will prompt if None)
            password: Password for authentication (will prompt if None)
            ssl_verify: Whether to verify SSL certificates (set to False to disable verification)
        """
        self.server_url = server_url.rstrip('/')
        self.username = username
        self.password = password
        self.token = None
        self.token_type = "AR-JWT"
        
        # Handle SSL verification
        if ssl_verify is False:
            # Disable SSL warnings if verification is disabled
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL certificate verification is disabled.")
        
        self.ssl_verify = ssl_verify
        
        logger.info(f"Initialized Remedy client for {self.server_url}")
    
    def login(self) -> tuple:
        """
        Log in to Remedy and get authentication token.
        
        Returns:
            tuple: (returnVal, token) where returnVal is 1 on success, -1 on failure
        """
        if not self.username:
            self.username = input("Enter Username: ")
        
        if not self.password:
            import getpass
            self.password = getpass.getpass(prompt="Enter Password: ")
        
        logger.info(f"Attempting to login as {self.username}")
        url = f"{self.server_url}/api/jwt/login"
        payload = {"username": self.username, "password": self.password}
        headers = {"content-type": "application/x-www-form-urlencoded"}
        
        try:
            r = requests.post(url, data=payload, headers=headers, verify=self.ssl_verify)
            if r.status_code == 200:
                self.token = r.text
                logger.info("Login successful")
                return 1, self.token
            else:
                logger.error(f"Login failed with status code: {r.status_code}")
                print(f"Failure...")
                print(f"Status Code: {r.status_code}")
                return -1, r.text
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return -1, str(e)
    
    def test_connection(self) -> bool:
        """
        Test connection to the Remedy API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        # If we don't have a token yet, try to login
        if not self.token:
            status, _ = self.login()
            if status != 1:
                return False
        
        try:
            # Try a basic API call to test the connection
            url = f"{self.server_url}/api/arsys/v1/entry/HPD:Help Desk?limit=1"
            headers = {"Authorization": f"{self.token_type} {self.token}"}
            
            response = requests.get(url, headers=headers, verify=self.ssl_verify)
            
            if response.status_code == 200:
                logger.info("Remedy connection test successful")
                return True
            else:
                logger.error(f"Remedy connection test failed with status code: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Remedy connection test error: {str(e)}")
            return False
        finally:
            # Logout to clean up
            self.logout()
    
    def logout(self) -> bool:
        """
        Log out from Remedy and invalidate the token.
        
        Returns:
            bool: True on success, False on failure
        """
        if not self.token:
            logger.warning("Cannot logout: No active token")
            return False
        
        logger.info("Logging out and invalidating token")
        url = f"{self.server_url}/api/jwt/logout"
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        try:
            r = requests.post(url, headers=headers, verify=self.ssl_verify)
            if r.status_code == 204 or r.status_code == 200:
                logger.info("Logout successful")
                self.token = None
                return True
            else:
                logger.error(f"Logout failed with status code: {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return False
    
    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific incident by its ID.
        
        Args:
            incident_id: The Incident Number (e.g., INC000001482087)
            
        Returns:
            dict: Incident data or None if not found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return None
        
        logger.info(f"Fetching incident: {incident_id}")
        
        # Create qualified query
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incident data
        result = self.query_form("HPD:Help Desk", qualified_query, fields)
        
        if result and "entries" in result and len(result["entries"]) > 0:
            logger.info(f"Successfully retrieved incident: {incident_id}")
            return result["entries"][0]
        else:
            logger.error(f"Incident not found or error: {incident_id}")
            return None
    
    def get_incidents_by_date(
        self, 
        date: str, 
        status: Optional[str] = None, 
        owner_group: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all incidents submitted on a specific date.
        
        Args:
            date: The submission date in YYYY-MM-DD format
            status: Optional status filter (e.g., "Closed", "Open")
            owner_group: Optional owner group filter
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return []
        
        logger.info(f"Fetching incidents for date: {date}")
        
        # Parse the date and create date range (entire day)
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            start_datetime = date_obj.strftime("%Y-%m-%d 00:00:00.000")
            end_datetime = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00.000")
            
            # Create qualified query
            query_parts = [f"'Submit Date' >= \"{start_datetime}\" AND 'Submit Date' < \"{end_datetime}\""]
            
            # Add status filter if provided
            if status:
                query_parts.append(f"'Status'=\"{status}\"")
            
            # Add owner group filter if provided
            if owner_group:
                query_parts.append(f"'Owner Group'=\"{owner_group}\"")
            
            qualified_query = " AND ".join(query_parts)
            
            # Fields to retrieve
            fields = [
                "Assignee", "Incident Number", "Description", "Status", "Owner",
                "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
                "Priority", "Environment", "Summary", "Support Group Name",
                "Request Assignee", "Work Order ID", "Request Manager"
            ]
            
            # Get the incidents
            result = self.query_form("HPD:Help Desk", qualified_query, fields)
            
            if result and "entries" in result:
                logger.info(f"Retrieved {len(result['entries'])} incidents for date {date}")
                return result["entries"]
            else:
                logger.warning(f"No incidents found for date {date} or error occurred")
                return []
                
        except ValueError:
            logger.error(f"Invalid date format: {date}. Use YYYY-MM-DD.")
            return []
    
    def get_incidents_by_status(self, status: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get incidents by their status.
        
        Args:
            status: The status to filter by (e.g., "Open", "Closed", "Resolved")
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return []
        
        logger.info(f"Fetching incidents with status: {status}")
        
        # Create qualified query
        qualified_query = f"'Status'=\"{status}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents with status {status}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found with status {status} or error occurred")
            return []
    
    def search_incidents(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search for incidents using the query text.
        
        Args:
            query: The search query
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return []
        
        logger.info(f"Searching incidents with query: {query}")
        
        # Create qualified query that matches in Summary, Description, or Notes
        query_terms = query.split()
        query_conditions = []
        
        for term in query_terms:
            # Escape quotes
            term = term.replace('"', '\\"')
            condition = f"('Summary' LIKE \"%{term}%\" OR 'Description' LIKE \"%{term}%\")"
            query_conditions.append(condition)
        
        qualified_query = " AND ".join(query_conditions)
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        
        if result and "entries" in result:
            logger.info(f"Search returned {len(result['entries'])} incidents")
            return result["entries"]
        else:
            logger.warning(f"No incidents found for query {query} or error occurred")
            return []
    
    def query_form(
        self, 
        form_name: str, 
        qualified_query: Optional[str] = None, 
        fields: Optional[List[str]] = None, 
        limit: int = 100
    ) -> Optional[Dict[str, Any]]:
        """
        Query a Remedy form with optional filters and field selection.
        
        Args:
            form_name: The name of the form to query (e.g., "HPD:Help Desk")
            qualified_query: Optional qualified query string for filtering
            fields: Optional list of fields to retrieve
            limit: Maximum number of records to retrieve
            
        Returns:
            dict: Query result or None if error
        """
        if not self.token:
            logger.error("No authentication token. Please login first.")
            return None
        
        logger.info(f"Querying form: {form_name}")
        
        # Build URL
        url = f"{self.server_url}/api/arsys/v1/entry/{form_name}"
        
        # Build headers
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        # Build query parameters
        params = {}
        if qualified_query:
            params["q"] = qualified_query
        if fields:
            params["fields"] = ",".join(fields)
        if limit:
            params["limit"] = limit
        
        # Make the request
        try:
            r = requests.get(url, headers=headers, params=params, verify=self.ssl_verify)
            if r.status_code == 200:
                result = r.json()
                logger.info(f"Successfully queried form {form_name} and got {len(result.get('entries', []))} results")
                return result
            else:
                logger.error(f"Query failed with status code: {r.status_code}")
                logger.error(f"Headers: {r.headers}")
                logger.error(f"Response: {r.text}")
                return None
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return None
    
    def process_incident_for_rag(self, incident: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process an incident into a format suitable for RAG indexing.
        
        Args:
            incident: Raw incident data from Remedy API
            
        Returns:
            dict: Processed incident with metadata and content
        """
        if not incident or "values" not in incident:
            return None
        
        values = incident.get("values", {})
        
        # Extract metadata
        metadata = {
            "incident_number": values.get("Incident Number"),
            "status": values.get("Status"),
            "priority": values.get("Priority"),
            "impact": values.get("Impact"),
            "assignee": values.get("Assignee"),
            "owner": values.get("Owner"),
            "owner_group": values.get("Owner Group"),
            "assigned_group": values.get("Assigned Group"),
            "submitter": values.get("Submitter"),
            "submit_date": values.get("Submit Date"),
            "summary": values.get("Summary")
        }
        
        # Build content for embedding
        content_parts = []
        if values.get("Summary"):
            content_parts.append(f"Summary: {values.get('Summary')}")
        if values.get("Description"):
            content_parts.append(f"Description: {values.get('Description')}")
        if values.get("Status"):
            content_parts.append(f"Status: {values.get('Status')}")
        if values.get("Priority"):
            content_parts.append(f"Priority: {values.get('Priority')}")
        if values.get("Impact"):
            content_parts.append(f"Impact: {values.get('Impact')}")
        if values.get("Assignee"):
            content_parts.append(f"Assigned to: {values.get('Assignee')}")
        if values.get("Owner Group"):
            content_parts.append(f"Owner Group: {values.get('Owner Group')}")
        
        # Combine content parts into a single text
        content = "\n".join(content_parts)
        
        return {
            "metadata": metadata,
            "content": content,
            "raw_data": values
        }



















import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
import requests

logger = logging.getLogger(__name__)

class JiraClient:
    """
    Client for JIRA REST API operations with comprehensive error handling.
    """
    
    def __init__(
        self, 
        base_url: str, 
        username: str = None, 
        password: str = None, 
        ssl_verify: bool = False
    ):
        """
        Initialize the JIRA client with authentication details.
        
        Args:
            base_url: The base URL of the JIRA instance (e.g., https://company.atlassian.net)
            username: The username for authentication
            password: The API token for authentication
            ssl_verify: Whether to verify SSL certificates (set to False to disable verification)
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, password) if username and password else None
        self.ssl_verify = ssl_verify
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized JIRA client for {self.base_url}")
    
    def test_connection(self) -> bool:
        """
        Test the connection to JIRA API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            logger.info("Testing connection to JIRA...")
            response = requests.get(
                f"{self.base_url}/rest/api/2/serverInfo",
                auth=self.auth,
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                server_info = response.json()
                logger.info(f"Connection to JIRA successful! Server version: {server_info.get('version', 'unknown')}")
                return True
            else:
                logger.error(f"Failed to connect to JIRA. Status code: {response.status_code}")
                if hasattr(response, 'text'):
                    logger.error(f"Response content: {response.text}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_issue(self, issue_key: str, fields: str = None, expand: str = None) -> Optional[Dict[str, Any]]:
        """
        Get a specific issue by its key.
        
        Args:
            issue_key: The issue key (e.g., DEMO-1)
            fields: Comma-separated string of field names to include
            expand: Comma-separated string of sections to expand
            
        Returns:
            dict: Issue data or None if not found/error
        """
        try:
            logger.info(f"Fetching issue: {issue_key}")
            
            params = {}
            if fields:
                params["fields"] = fields
            if expand:
                params["expand"] = expand
            
            response = requests.get(
                f"{self.base_url}/rest/api/2/issue/{issue_key}",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                issue = response.json()
                logger.info(f"Successfully retrieved issue: {issue_key}")
                return issue
            else:
                logger.error(f"Failed to get issue {issue_key}. Status code: {response.status_code}")
                error_details = response.json() if hasattr(response, 'json') else {}
                logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Failed to get issue {issue_key}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None
    
    def search_issues(
        self, 
        jql: str = None, 
        fields: List[str] = None, 
        max_results: int = 50, 
        start_at: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search for issues using JQL (Jira Query Language).
        
        Args:
            jql: JQL search string
            fields: List of fields to include in results
            max_results: Maximum number of results to return
            start_at: Index of the first result to return
            
        Returns:
            list: List of issues or empty list if none found/error
        """
        try:
            logger.info(f"Searching issues with JQL: {jql}")
            
            # Build query parameters
            params = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": max_results,
            }
            
            if fields:
                params["fields"] = ",".join(fields)
            
            response = requests.get(
                f"{self.base_url}/rest/api/2/search",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                search_results = response.json()
                total = search_results.get("total", 0)
                issues = search_results.get("issues", [])
                logger.info(f"Search returned {len(issues)} issues (total: {total})")
                return issues
            else:
                logger.error(f"Failed to search issues: {str(e)}")
                error_details = response.json() if hasattr(response, 'json') else {}
                logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching issues: {str(e)}")
            return []
    
    def get_all_issues(self, jql: str, fields: List[str] = None, batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Get all issues matching a JQL query, handling pagination.
        
        Args:
            jql: JQL search string
            fields: List of fields to include in results
            batch_size: Number of issues to retrieve per request
            
        Returns:
            list: List of all matching issues
        """
        all_issues = []
        start_at = 0
        
        logger.info(f"Retrieving all issues matching: {jql}")
        
        while True:
            # Get batch of issues
            issues = self.search_issues(jql, fields, batch_size, start_at)
            
            if not issues:
                break
            
            all_issues.extend(issues)
            
            # Check if we've reached the total or end of results
            if len(issues) < batch_size:
                break
            
            # Next page
            start_at += len(issues)
        
        logger.info(f"Retrieved a total of {len(all_issues)} issues")
        return all_issues
    
    def get_issue_content(self, issue_key: str) -> Dict[str, Any]:
        """
        Get the content of an issue in a format suitable for RAG.
        
        Args:
            issue_key: The issue key (e.g., DEMO-1)
            
        Returns:
            dict: Issue content data
        """
        try:
            # Get issue with all relevant fields
            fields = "summary,description,issuetype,status,created,updated,assignee,reporter,priority,labels,components,fixVersions,resolution,comment"
            issue = self.get_issue(issue_key, fields=fields)
            
            if not issue:
                return {
                    "content": "",
                    "title": f"Issue {issue_key} not found",
                    "metadata": {}
                }
            
            fields = issue.get("fields", {})
            
            # Extract key metadata
            metadata = {
                "key": issue_key,
                "type": fields.get("issuetype", {}).get("name", ""),
                "status": fields.get("status", {}).get("name", ""),
                "created": fields.get("created", ""),
                "updated": fields.get("updated", ""),
                "assignee": fields.get("assignee", {}).get("displayName", "Unassigned") if fields.get("assignee") else "Unassigned",
                "reporter": fields.get("reporter", {}).get("displayName", "Unknown") if fields.get("reporter") else "Unknown",
                "priority": fields.get("priority", {}).get("name", "") if fields.get("priority") else "",
                "labels": fields.get("labels", []),
                "resolution": fields.get("resolution", {}).get("name", "Unresolved") if fields.get("resolution") else "Unresolved"
            }
            
            # Extract content parts
            content_parts = []
            
            # Add summary
            summary = fields.get("summary", "")
            content_parts.append(f"Summary: {summary}")
            
            # Add description
            description = fields.get("description", "")
            if description:
                content_parts.append(f"Description: {description}")
            
            # Add comments
            comments = fields.get("comment", {}).get("comments", [])
            for comment in comments:
                author = comment.get("author", {}).get("displayName", "Unknown")
                created = comment.get("created", "")
                
                comment_body = comment.get("body", "")
                if comment_body:
                    # Extract text from comment body if it's in Atlassian Document Format
                    if isinstance(comment_body, dict):
                        comment_text = self._extract_text_from_adf(comment_body)
                    else:
                        comment_text = comment_body
                    
                    content_parts.append(f"Comment by {author} on {created}: {comment_text}")
            
            # Combine all content
            full_content = "\n\n".join(content_parts)
            
            return {
                "content": full_content,
                "title": summary,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing issue content: {str(e)}")
            return {
                "content": f"Error retrieving issue content: {str(e)}",
                "title": f"Issue {issue_key} - Error",
                "metadata": {"key": issue_key}
            }
    
    def _extract_text_from_adf(self, adf_doc: Dict[str, Any]) -> str:
        """
        Extract plain text from Atlassian Document Format (ADF).
        
        Args:
            adf_doc: The ADF document object
            
        Returns:
            str: Extracted plain text
        """
        text_parts = []
        
        def extract_from_content(content_list):
            parts = []
            for item in content_list:
                if not isinstance(item, dict):
                    continue
                
                # Handle text nodes
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                
                # Recursively handle content arrays
                if "content" in item and isinstance(item["content"], list):
                    parts.append(extract_from_content(item["content"]))
            
            return "".join(parts)
        
        # Main content
        if "content" in adf_doc and isinstance(adf_doc["content"], list):
            text_parts.append(extract_from_content(adf_doc["content"]))
        
        return "".join(text_parts)
    
    def create_issue(
        self, 
        project_key: str, 
        issue_type: str, 
        summary: str, 
        description: str, 
        fields: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new issue.
        
        Args:
            project_key: The project key
            issue_type: The issue type name or ID
            summary: The issue summary
            description: Detailed description
            fields: Dictionary of additional fields to set
            
        Returns:
            dict: Created issue data or None if error
        """
        try:
            logger.info(f"Creating issue in project {project_key} of type {issue_type}")
            
            # Base issue data
            issue_data = {
                "fields": {
                    "project": {
                        "key": project_key
                    },
                    "summary": summary,
                    "description": description,
                    "issuetype": {
                        "name": issue_type
                    }
                }
            }
            
            # Add additional fields if provided
            if fields:
                issue_data["fields"].update(fields)
            
            response = requests.post(
                f"{self.base_url}/rest/api/2/issue",
                auth=self.auth,
                headers=self.headers,
                json=issue_data,
                verify=self.ssl_verify
            )
            
            if response.status_code == 201:
                result = response.json()
                logger.info(f"Successfully created issue: {result.get('key')}")
                return result
            else:
                logger.error(f"Failed to create issue: {str(e)}")
                error_details = response.json() if hasattr(response, 'json') else {}
                logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating issue: {str(e)}")
            return None
    
    def get_projects(self) -> List[Dict[str, Any]]:
        """
        Get all projects visible to the authenticated user.
        
        Returns:
            list: List of projects or empty list if error
        """
        try:
            logger.info("Fetching projects...")
            
            response = requests.get(
                f"{self.base_url}/rest/api/2/project",
                auth=self.auth,
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                projects = response.json()
                logger.info(f"Successfully retrieved {len(projects)} projects")
                return projects
            else:
                logger.error(f"Failed to get projects: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving projects: {str(e)}")
            return []
    
    def get_issue_types(self) -> List[Dict[str, Any]]:
        """
        Get all issue types defined in the JIRA instance.
        
        Returns:
            list: List of issue types or empty list if error
        """
        try:
            logger.info("Fetching issue types...")
            
            response = requests.get(
                f"{self.base_url}/rest/api/2/issuetype",
                auth=self.auth,
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                issue_types = response.json()
                logger.info(f"Successfully retrieved {len(issue_types)} issue types")
                return issue_types
            else:
                logger.error(f"Failed to get issue types: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving issue types: {str(e)}")
            return []















import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logging(
    log_level: int = logging.INFO,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_file: str = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_format: Format string for log entries
        log_file: Path to log file or None to use auto-generated name
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate log file name if not provided
    if not log_file:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_file = os.path.join(log_dir, f'enterprise_rag_{timestamp}.log')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Configure formatter
    formatter = logging.Formatter(log_format)
    
    # Add file handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create app logger
    logger = logging.getLogger('enterprise_rag')
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)













import os
import pickle
from typing import Any, Optional
import logging
from flask import Flask
from flask_caching import Cache

logger = logging.getLogger(__name__)

def setup_cache(app: Flask) -> Cache:
    """
    Set up caching for the application.
    
    Args:
        app: Flask application instance
    
    Returns:
        Cache instance
    """
    # Create cache config based on app config
    cache_config = {
        "CACHE_TYPE": app.config.get("CACHE_TYPE", "SimpleCache"),
        "CACHE_DEFAULT_TIMEOUT": app.config.get("CACHE_DEFAULT_TIMEOUT", 300)
    }
    
    # Configure Redis if using RedisCache
    if cache_config["CACHE_TYPE"] == "RedisCache":
        cache_config["CACHE_REDIS_URL"] = app.config.get("CACHE_REDIS_URL", "redis://localhost:6379/0")
    
    # Create and initialize cache
    cache = Cache(app, config=cache_config)
    
    logger.info(f"Cache initialized with type: {cache_config['CACHE_TYPE']}")
    return cache

class DiskCache:
    """
    Simple disk-based cache implementation for storing embeddings and other data.
    """
    
    def __init__(self, cache_dir: str = None, max_size: int = 1000):
        """
        Initialize the disk cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of items to keep in memory
        """
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "data", "cache")
        self.max_size = max_size
        self.memory_cache = {}
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        logger.info(f"Disk cache initialized at {self.cache_dir}")
    
    def _get_cache_path(self, key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            key: Cache key
        
        Returns:
            File path
        """
        # Create a safe filename from the key
        safe_key = "".join(c if c.isalnum() else "_" for c in str(key))
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            default: Default value to return if key not found
        
        Returns:
            Cached value or default
        """
        # Check memory cache first
        if key in self.memory_cache:
            logger.debug(f"Cache hit (memory) for key: {key}")
            return self.memory_cache[key]
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Add to memory cache
                self._add_to_memory_cache(key, value)
                
                logger.debug(f"Cache hit (disk) for key: {key}")
                return value
            except Exception as e:
                logger.error(f"Error loading cache item: {str(e)}")
        
        logger.debug(f"Cache miss for key: {key}")
        return default
    
    def set(self, key: str, value: Any, timeout: Optional[int] = None) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            timeout: Optional timeout in seconds (not used for disk cache)
        """
        # Add to memory cache
        self._add_to_memory_cache(key, value)
        
        # Save to disk
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logger.debug(f"Saved item to disk cache: {key}")
        except Exception as e:
            logger.error(f"Error saving cache item to disk: {str(e)}")
    
    def _add_to_memory_cache(self, key: str, value: Any) -> None:
        """
        Add an item to the memory cache, evicting items if necessary.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Check if cache is full
        if len(self.memory_cache) >= self.max_size:
            # Remove random item (first item in the dict)
            if self.memory_cache:
                self.memory_cache.pop(next(iter(self.memory_cache)))
        
        # Add to memory cache
        self.memory_cache[key] = value
    
    def delete(self, key: str) -> None:
        """
        Delete an item from the cache.
        
        Args:
            key: Cache key
        """
        # Remove from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Remove from disk cache
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                logger.debug(f"Deleted item from disk cache: {key}")
            except Exception as e:
                logger.error(f"Error deleting cache item from disk: {str(e)}")
    
    def clear(self) -> None:
        """
        Clear the entire cache.
        """
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear disk cache
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing disk cache: {str(e)}")





















# Core application packages
flask==2.3.3
flask-caching==2.0.2
python-dotenv==1.0.0
werkzeug==2.3.7
gunicorn==21.2.0

# Data source integrations
requests==2.31.0
beautifulsoup4==4.12.2
html2text==2020.1.16

# RAG components
sentence-transformers==2.2.2
faiss-cpu==1.7.4
scikit-learn==1.3.0
numpy==1.25.2
pandas==2.1.0

# Google Vertex AI / Gemini integration
google-cloud-aiplatform==1.36.4
vertexai==0.0.1
google-api-core==2.11.1

# Utility packages
python-dateutil==2.8.2
markdown==3.5
dompurify==0.1.0












# Flask application configuration
FLASK_APP=app.py
FLASK_ENV=development  # Change to 'production' for production deployment
SECRET_KEY=your-secret-key-change-in-production
PORT=5000

# Logging configuration
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Caching configuration
CACHE_TYPE=SimpleCache  # SimpleCache, RedisCache, etc.
CACHE_DEFAULT_TIMEOUT=300  # In seconds
# CACHE_REDIS_URL=redis://localhost:6379/0  # Uncomment if using RedisCache

# Confluence configuration
CONFLUENCE_URL=https://your-company.atlassian.net
CONFLUENCE_USERNAME=your-username
CONFLUENCE_PASSWORD=your-api-token
CONFLUENCE_SSL_VERIFY=False

# Remedy configuration
REMEDY_URL=https://your-company-restapi.onbmc.com
REMEDY_USERNAME=your-username
REMEDY_PASSWORD=your-password
REMEDY_SSL_VERIFY=False

# JIRA configuration
JIRA_URL=https://your-company.atlassian.net
JIRA_USERNAME=your-username
JIRA_PASSWORD=your-api-token
JIRA_SSL_VERIFY=False

# Google Gemini configuration
GEMINI_PROJECT_ID=prj-dv-cws-4363
GEMINI_REGION=us-central1
GEMINI_MODEL_NAME=gemini-2.0-flash-001

# RAG configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_DB_PATH=./data/vector_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TEMPERATURE=0.7
MAX_OUTPUT_TOKENS=8192
















# data_sources/__init__.py
from .confluence import ConfluenceClient
from .remedy import RemedyClient
from .jira_client import JiraClient

# rag/__init__.py
from .query_processor import QueryProcessor
from .chunking import ChunkingEngine
from .embedding import EmbeddingEngine
from .vector_search import VectorSearch
from .context_builder import ContextBuilder

# llm/__init__.py
from .gemini_client import GeminiClient

# utils/__init__.py
from .logger import setup_logging, get_logger
from .cache import setup_cache, DiskCache












from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

class DataSource(ABC):
    """
    Abstract base class for all data source integrations.
    Implementations should extend this class to ensure consistent functionality.
    """
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test the connection to the data source.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for content in the data source.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional source-specific parameters
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    def get_content(self, content_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get specific content by ID.
        
        Args:
            content_id: ID of the content to retrieve
            **kwargs: Additional source-specific parameters
            
        Returns:
            Content data or None if not found/error
        """
        pass
    
    @abstractmethod
    def process_for_rag(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process content into a format suitable for RAG indexing.
        
        Args:
            content: Raw content data
            
        Returns:
            Processed content with metadata and text
        """
        pass

class ConfluenceDataSource(DataSource):
    """
    Implementation of DataSource for Confluence.
    This adapter wraps the ConfluenceClient with the standard interface.
    """
    
    def __init__(self, client):
        """
        Initialize with a ConfluenceClient instance.
        
        Args:
            client: ConfluenceClient instance
        """
        self.client = client
        self.source_type = "confluence"
    
    def test_connection(self) -> bool:
        """
        Test the connection to Confluence.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        return self.client.test_connection()
    
    def search(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for content in Confluence.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional parameters (content_type, space_key, etc.)
            
        Returns:
            List of search results
        """
        content_type = kwargs.get('content_type', 'page')
        expand = kwargs.get('expand', None)
        
        # Search using CQL
        results = self.client.search_content(
            cql=query,
            content_type=content_type,
            limit=limit,
            expand=expand
        )
        
        return [{
            'id': item.get('id'),
            'title': item.get('title', 'Untitled'),
            'type': item.get('type', 'unknown'),
            'url': item.get('_links', {}).get('webui', ''),
            'source': self.source_type,
            'last_updated': item.get('history', {}).get('lastUpdated', ''),
            'space': item.get('space', {}).get('name', '') if 'space' in item else '',
            'raw_data': item
        } for item in results]
    
    def get_content(self, content_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get specific content from Confluence by ID.
        
        Args:
            content_id: ID of the content to retrieve
            **kwargs: Additional parameters (expand, etc.)
            
        Returns:
            Content data or None if not found/error
        """
        expand = kwargs.get('expand', 'body.storage,metadata.labels')
        return self.client.get_page_content(content_id, expand=expand)
    
    def process_for_rag(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Confluence content into a format suitable for RAG indexing.
        
        Args:
            content: Raw Confluence content
            
        Returns:
            Processed content with metadata and text
        """
        if isinstance(content, dict) and 'id' in content and 'raw_html' in content:
            # Content is already processed by get_page_content
            return {
                'id': content.get('id', ''),
                'title': content.get('metadata', {}).get('title', 'Untitled'),
                'content': content.get('content', ''),
                'metadata': content.get('metadata', {}),
                'source': self.source_type,
                'source_url': f"{self.client.base_url}/wiki/spaces/{content.get('metadata', {}).get('space', '')}/pages/{content.get('id', '')}"
            }
        else:
            # Content needs processing
            processed = self.client.process_for_embeddings(content)
            return {
                'id': content.get('id', ''),
                'title': content.get('title', 'Untitled'),
                'content': processed,
                'metadata': {
                    'type': content.get('type', 'unknown'),
                    'space': content.get('space', {}).get('name', '') if isinstance(content.get('space'), dict) else '',
                    'last_updated': content.get('history', {}).get('lastUpdated', '') if 'history' in content else ''
                },
                'source': self.source_type,
                'source_url': content.get('_links', {}).get('webui', '') if '_links' in content else ''
            }

class RemedyDataSource(DataSource):
    """
    Implementation of DataSource for Remedy.
    This adapter wraps the RemedyClient with the standard interface.
    """
    
    def __init__(self, client):
        """
        Initialize with a RemedyClient instance.
        
        Args:
            client: RemedyClient instance
        """
        self.client = client
        self.source_type = "remedy"
    
    def test_connection(self) -> bool:
        """
        Test the connection to Remedy.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        return self.client.test_connection()
    
    def search(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for incidents in Remedy.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional parameters (status, etc.)
            
        Returns:
            List of search results
        """
        status = kwargs.get('status', None)
        
        if status:
            # Search by status
            results = self.client.get_incidents_by_status(status, limit=limit)
        else:
            # Search by query text
            results = self.client.search_incidents(query, limit=limit)
        
        return [{
            'id': item.get('values', {}).get('Incident Number', ''),
            'title': item.get('values', {}).get('Summary', 'Untitled Incident'),
            'status': item.get('values', {}).get('Status', ''),
            'priority': item.get('values', {}).get('Priority', ''),
            'assignee': item.get('values', {}).get('Assignee', ''),
            'submit_date': item.get('values', {}).get('Submit Date', ''),
            'source': self.source_type,
            'url': f"remedyurl/incident/{item.get('values', {}).get('Incident Number', '')}",
            'raw_data': item
        } for item in results]
    
    def get_content(self, content_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get specific incident from Remedy by ID.
        
        Args:
            content_id: Incident Number to retrieve
            **kwargs: Additional parameters
            
        Returns:
            Incident data or None if not found/error
        """
        return self.client.get_incident(content_id)
    
    def process_for_rag(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Remedy incident into a format suitable for RAG indexing.
        
        Args:
            content: Raw Remedy incident data
            
        Returns:
            Processed incident with metadata and text
        """
        processed = self.client.process_incident_for_rag(content)
        if not processed:
            return None
            
        return {
            'id': processed['metadata'].get('incident_number', ''),
            'title': processed['metadata'].get('summary', 'Untitled Incident'),
            'content': processed['content'],
            'metadata': processed['metadata'],
            'source': self.source_type,
            'source_url': f"remedyurl/incident/{processed['metadata'].get('incident_number', '')}"
        }

class JiraDataSource(DataSource):
    """
    Implementation of DataSource for JIRA.
    This adapter wraps the JiraClient with the standard interface.
    """
    
    def __init__(self, client):
        """
        Initialize with a JiraClient instance.
        
        Args:
            client: JiraClient instance
        """
        self.client = client
        self.source_type = "jira"
    
    def test_connection(self) -> bool:
        """
        Test the connection to JIRA.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        return self.client.test_connection()
    
    def search(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for issues in JIRA.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional parameters (project, issuetype, etc.)
            
        Returns:
            List of search results
        """
        # Create JQL query
        jql_parts = []
        
        # Add text search if query is provided
        if query:
            jql_parts.append(f'text ~ "{query}"')
        
        # Add project filter if provided
        project = kwargs.get('project', None)
        if project:
            jql_parts.append(f'project = "{project}"')
        
        # Add issue type filter if provided
        issue_type = kwargs.get('issue_type', None)
        if issue_type:
            jql_parts.append(f'issuetype = "{issue_type}"')
        
        # Add status filter if provided
        status = kwargs.get('status', None)
        if status:
            jql_parts.append(f'status = "{status}"')
        
        # Combine JQL parts
        jql = ' AND '.join(jql_parts) if jql_parts else ''
        
        # Default fields to retrieve
        fields = kwargs.get('fields', [
            'summary', 'description', 'status', 'assignee', 
            'reporter', 'priority', 'issuetype', 'project'
        ])
        
        # Search for issues
        results = self.client.search_issues(jql, fields=fields, max_results=limit)
        
        return [{
            'id': item.get('key', ''),
            'title': item.get('fields', {}).get('summary', 'Untitled Issue'),
            'type': item.get('fields', {}).get('issuetype', {}).get('name', '') if 'fields' in item and 'issuetype' in item['fields'] else '',
            'status': item.get('fields', {}).get('status', {}).get('name', '') if 'fields' in item and 'status' in item['fields'] else '',
            'assignee': item.get('fields', {}).get('assignee', {}).get('displayName', '') if 'fields' in item and 'assignee' in item['fields'] else '',
            'priority': item.get('fields', {}).get('priority', {}).get('name', '') if 'fields' in item and 'priority' in item['fields'] else '',
            'source': self.source_type,
            'url': f"{self.client.base_url}/browse/{item.get('key', '')}",
            'raw_data': item
        } for item in results]
    
    def get_content(self, content_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get specific issue from JIRA by ID.
        
        Args:
            content_id: Issue key to retrieve
            **kwargs: Additional parameters
            
        Returns:
            Issue data or None if not found/error
        """
        return self.client.get_issue_content(content_id)
    
    def process_for_rag(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process JIRA issue into a format suitable for RAG indexing.
        
        Args:
            content: Raw JIRA issue data or processed content
            
        Returns:
            Processed issue with metadata and text
        """
        # Check if content is already processed
        if 'content' in content and 'metadata' in content:
            return {
                'id': content.get('metadata', {}).get('key', ''),
                'title': content.get('title', 'Untitled Issue'),
                'content': content.get('content', ''),
                'metadata': content.get('metadata', {}),
                'source': self.source_type,
                'source_url': f"{self.client.base_url}/browse/{content.get('metadata', {}).get('key', '')}"
            }
        
        # Get processed content
        issue_key = content.get('key', '') if 'key' in content else content.get('id', '')
        processed = self.client.get_issue_content(issue_key)
        
        return {
            'id': processed.get('metadata', {}).get('key', ''),
            'title': processed.get('title', 'Untitled Issue'),
            'content': processed.get('content', ''),
            'metadata': processed.get('metadata', {}),
            'source': self.source_type,
            'source_url': f"{self.client.base_url}/browse/{processed.get('metadata', {}).get('key', '')}"
        }

















/**
 * Enterprise RAG System - Main JavaScript
 * 
 * Handles the chat interface interactions, API calls,
 * and dynamic content updates.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const messagesContainer = document.getElementById('messages-container');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const typingIndicator = document.getElementById('typing-indicator');
    
    const confluenceToggle = document.getElementById('confluence-toggle');
    const remedyToggle = document.getElementById('remedy-toggle');
    const jiraToggle = document.getElementById('jira-toggle');
    
    const sourcesPanel = document.getElementById('sources-panel');
    const closeSourcesBtn = document.getElementById('close-sources');
    const sourcesContent = document.getElementById('sources-content');
    
    // Store the current conversation sources for reference
    let currentSources = [];
    let conversationHistory = [];
    
    // Constants
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 1000; // ms
    
    /**
     * Initialize the chat interface
     */
    function initChat() {
        // Set up event listeners
        setupEventListeners();
        
        // Auto-resize the input field
        userInput.addEventListener('input', autoResizeInput);
        
        // Enable user input
        enableUserInput();
        
        // Set focus on input
        userInput.focus();
        
        // Load conversation history if available
        loadConversationHistory();
    }
    
    /**
     * Set up event listeners for user interactions
     */
    function setupEventListeners() {
        // Handle form submission
        chatForm.addEventListener('submit', handleSubmit);
        
        // Close sources panel button
        closeSourcesBtn.addEventListener('click', () => {
            sourcesPanel.classList.remove('active');
        });
        
        // Handle Escape key to close sources panel
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && sourcesPanel.classList.contains('active')) {
                sourcesPanel.classList.remove('active');
            }
        });
        
        // Enable pressing Enter to send message (but Shift+Enter for new line)
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                chatForm.dispatchEvent(new Event('submit'));
            }
        });
        
        // Add click handlers for code block copy buttons
        messagesContainer.addEventListener('click', (e) => {
            if (e.target.classList.contains('copy-code-btn')) {
                const codeBlock = e.target.previousElementSibling;
                copyToClipboard(codeBlock.textContent);
                
                // Show "Copied!" feedback
                const originalText = e.target.textContent;
                e.target.textContent = 'Copied!';
                e.target.classList.add('copied');
                
                setTimeout(() => {
                    e.target.textContent = originalText;
                    e.target.classList.remove('copied');
                }, 2000);
            }
        });
    }
    
    /**
     * Handle form submission when user sends a message
     */
    async function handleSubmit(e) {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message) return;
        
        // Disable input while processing
        disableUserInput();
        
        // Add user message to UI
        addMessage(message, 'user');
        
        // Store in conversation history
        conversationHistory.push({ role: 'user', content: message });
        
        // Clear input
        userInput.value = '';
        userInput.style.height = 'auto';
        
        // Show typing indicator
        typingIndicator.classList.remove('hidden');
        
        // Get selected data sources
        const selectedSources = getSelectedSources();
        
        // Attempt to send the message with retries
        let attempt = 0;
        let success = false;
        
        while (attempt < MAX_RETRIES && !success) {
            try {
                // Send request to API
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: message,
                        sources: selectedSources,
                        conversation_history: conversationHistory.slice(-10) // Send last 10 messages for context
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }
                
                const data = await response.json();
                
                // Hide typing indicator
                typingIndicator.classList.add('hidden');
                
                handleApiResponse(data);
                success = true;
                
            } catch (error) {
                console.error(`Attempt ${attempt + 1} failed:`, error);
                attempt++;
                
                if (attempt >= MAX_RETRIES) {
                    // Hide typing indicator
                    typingIndicator.classList.add('hidden');
                    
                    // Show error message
                    addMessage(`I encountered an error while processing your request. Please try again later. (Error: ${error.message})`, 'system', true);
                } else {
                    // Wait before retrying
                    await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
                }
            }
        }
        
        // Re-enable input
        enableUserInput();
    }
    
    /**
     * Process and display the API response
     */
    function handleApiResponse(data) {
        if (data.error) {
            // Handle error
            addMessage(`Error: ${data.error}`, 'system', true);
        } else {
            // Add response to UI
            const messageElement = addMessage(data.response, 'assistant', false, data.processing_time);
            
            // Store in conversation history
            conversationHistory.push({ role: 'assistant', content: data.response });
            
            // Save updated conversation history
            saveConversationHistory();
            
            // Store sources for this message
            if (data.sources && data.sources.length > 0) {
                currentSources = data.sources;
                
                // Add view sources button
                const footer = messageElement.querySelector('.message-footer');
                const viewSourcesBtn = footer.querySelector('.view-sources-btn');
                viewSourcesBtn.classList.remove('hidden');
                viewSourcesBtn.addEventListener('click', () => {
                    showSources(currentSources);
                });
            }
            
            // Add copy buttons to code blocks
            addCodeBlockCopyButtons(messageElement);
        }
    }
    
    /**
     * Auto-resize textarea based on content
     */
    function autoResizeInput() {
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
        
        // Limit to 5 rows max
        const lineHeight = parseInt(window.getComputedStyle(userInput).lineHeight);
        const maxHeight = lineHeight * 5;
        if (userInput.scrollHeight > maxHeight) {
            userInput.style.height = maxHeight + 'px';
            userInput.style.overflowY = 'auto';
        } else {
            userInput.style.overflowY = 'hidden';
        }
    }
    
    /**
     * Add a message to the UI
     */
    function addMessage(content, role, isError = false, processingTime = null) {
        const template = document.getElementById('message-template');
        const messageElement = template.content.cloneNode(true).querySelector('.message');
        
        // Add appropriate class based on role
        messageElement.classList.add(`${role}-message`);
        if (isError) messageElement.classList.add('error-message');
        
        const messageContent = messageElement.querySelector('.message-content p');
        
        // For assistant messages, render markdown and sanitize
        if (role === 'assistant') {
            // Use DOMPurify to sanitize HTML from marked
            const sanitizedHtml = DOMPurify.sanitize(marked.parse(content), {
                ADD_ATTR: ['target'],
                ALLOWED_TAGS: [
                    'a', 'abbr', 'b', 'blockquote', 'br', 'code', 'div', 'em', 
                    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'i', 'img', 'li', 
                    'ol', 'p', 'pre', 'strong', 'table', 'tbody', 'td', 'th', 
                    'thead', 'tr', 'ul'
                ]
            });
            
            messageContent.innerHTML = sanitizedHtml;
            
            // Initialize syntax highlighting for code blocks
            messageElement.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightBlock(block);
            });
        } else {
            messageContent.textContent = content;
        }
        
        // Add timestamp
        const messageTime = messageElement.querySelector('.message-time');
        const now = new Date();
        messageTime.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        // Add processing time if available
        if (processingTime !== null) {
            const processingTimeSpan = document.createElement('span');
            processingTimeSpan.classList.add('processing-time');
            processingTimeSpan.textContent = ` (${processingTime.toFixed(2)}s)`;
            messageTime.appendChild(processingTimeSpan);
        }
        
        // Add message to container
        messagesContainer.appendChild(messageElement);
        
        // Scroll to bottom
        scrollToBottom();
        
        return messageElement;
    }
    
    /**
     * Add copy buttons to code blocks in a message
     */
    function addCodeBlockCopyButtons(messageElement) {
        const codeBlocks = messageElement.querySelectorAll('pre code');
        
        codeBlocks.forEach(codeBlock => {
            const pre = codeBlock.parentElement;
            
            // Skip if button already exists
            if (pre.querySelector('.copy-code-btn')) {
                return;
            }
            
            // Create button
            const copyButton = document.createElement('button');
            copyButton.className = 'copy-code-btn';
            copyButton.textContent = 'Copy';
            
            // Add button to pre element
            pre.appendChild(copyButton);
            pre.style.position = 'relative';
        });
    }
    
    /**
     * Copy text to clipboard
     */
    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            console.log('Text copied to clipboard');
        }).catch(err => {
            console.error('Could not copy text: ', err);
        });
    }
    
    /**
     * Show sources panel with provided sources
     */
    function showSources(sources) {
        // Clear previous sources
        sourcesContent.innerHTML = '';
        
        if (!sources || sources.length === 0) {
            const noSourcesElement = document.createElement('p');
            noSourcesElement.classList.add('no-sources');
            noSourcesElement.textContent = 'No sources to display.';
            sourcesContent.appendChild(noSourcesElement);
        } else {
            const template = document.getElementById('source-item-template');
            
            sources.forEach(source => {
                const sourceElement = template.content.cloneNode(true).querySelector('.source-item');
                
                // Set source icon based on type
                const sourceIcon = sourceElement.querySelector('.source-icon');
                switch (source.source_type) {
                    case 'confluence':
                        sourceIcon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>';
                        sourceIcon.classList.add('confluence-icon');
                        break;
                    case 'remedy':
                        sourceIcon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path><polyline points="13 2 13 9 20 9"></polyline></svg>';
                        sourceIcon.classList.add('remedy-icon');
                        break;
                    case 'jira':
                        sourceIcon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5V3m0 18v-2M5 12H3m18 0h-2M6 6l1.5 1.5M6 18l1.5-1.5M18 6l-1.5 1.5m1.5 10.5l-1.5-1.5"/></svg>';
                        sourceIcon.classList.add('jira-icon');
                        break;
                    default:
                        sourceIcon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';
                }
                
                // Set title and type
                sourceElement.querySelector('.source-title').textContent = source.name || 'Untitled';
                sourceElement.querySelector('.source-type').textContent = 
                    `${source.source_type.charAt(0).toUpperCase() + source.source_type.slice(1)} ${source.source_id}`;
                
                // Set link if available
                const sourceLink = sourceElement.querySelector('.source-link');
                if (source.url) {
                    sourceLink.href = source.url;
                } else {
                    sourceLink.style.display = 'none';
                }
                
                // Add relevance indicator if available
                if (source.relevance) {
                    const relevanceIndicator = document.createElement('div');
                    relevanceIndicator.className = 'relevance-indicator';
                    
                    // Scale relevance to percentage (assuming relevance is 0-1)
                    const relevancePercent = Math.min(Math.round(source.relevance * 100), 100);
                    
                    relevanceIndicator.innerHTML = `
                        <div class="relevance-bar">
                            <div class="relevance-fill" style="width: ${relevancePercent}%"></div>
                        </div>
                        <span class="relevance-label">${relevancePercent}% relevant</span>
                    `;
                    
                    sourceElement.appendChild(relevanceIndicator);
                }
                
                sourcesContent.appendChild(sourceElement);
            });
        }
        
        // Show panel
        sourcesPanel.classList.add('active');
    }
    
    /**
     * Get currently selected data sources
     */
    function getSelectedSources() {
        const selectedSources = [];
        if (confluenceToggle.checked) selectedSources.push('confluence');
        if (remedyToggle.checked) selectedSources.push('remedy');
        if (jiraToggle.checked) selectedSources.push('jira');
        return selectedSources;
    }
    
    /**
     * Disable user input during processing
     */
    function disableUserInput() {
        userInput.disabled = true;
        sendButton.disabled = true;
    }
    
    /**
     * Enable user input after processing
     */
    function enableUserInput() {
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();
    }
    
    /**
     * Scroll the message container to the bottom
     */
    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    /**
     * Save conversation history to local storage
     */
    function saveConversationHistory() {
        try {
            localStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
        } catch (e) {
            console.error('Error saving conversation history:', e);
        }
    }
    
    /**
     * Load conversation history from local storage
     */
    function loadConversationHistory() {
        try {
            const saved = localStorage.getItem('conversationHistory');
            if (saved) {
                const history = JSON.parse(saved);
                
                // Only restore history if it's a valid array and not too large
                if (Array.isArray(history) && history.length > 0 && history.length < 100) {
                    conversationHistory = history;
                    
                    // Ask user if they want to restore the conversation
                    const restoreChat = confirm('Would you like to restore your previous conversation?');
                    
                    if (restoreChat) {
                        // Display the conversation history
                        history.forEach(item => {
                            addMessage(item.content, item.role);
                        });
                    } else {
                        // Clear the history
                        conversationHistory = [];
                        localStorage.removeItem('conversationHistory');
                    }
                }
            }
        } catch (e) {
            console.error('Error loading conversation history:', e);
            // Clear potentially corrupted data
            localStorage.removeItem('conversationHistory');
        }
    }
    
    // Initialize the chat interface
    initChat();
});
















import unittest
import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config, DevelopmentConfig, TestingConfig, ProductionConfig

class TestConfig(unittest.TestCase):
    """Test the configuration classes."""
    
    def test_base_config(self):
        """Test the base configuration."""
        # Test default values
        self.assertEqual(Config.CACHE_TYPE, 'SimpleCache')
        self.assertEqual(Config.CACHE_DEFAULT_TIMEOUT, 300)
        
        # Test that sensitive values are not hardcoded
        self.assertNotEqual(Config.SECRET_KEY, 'some-hardcoded-secret-key')
    
    def test_development_config(self):
        """Test the development configuration."""
        self.assertTrue(DevelopmentConfig.DEBUG)
        self.assertFalse(DevelopmentConfig.TESTING)
    
    def test_testing_config(self):
        """Test the testing configuration."""
        self.assertFalse(TestingConfig.DEBUG)
        self.assertTrue(TestingConfig.TESTING)
        self.assertEqual(TestingConfig.CACHE_TYPE, 'SimpleCache')
    
    def test_production_config(self):
        """Test the production configuration."""
        self.assertFalse(ProductionConfig.DEBUG)
        self.assertFalse(ProductionConfig.TESTING)
        
        # Check that production has different cache config
        self.assertNotEqual(ProductionConfig.CACHE_TYPE, DevelopmentConfig.CACHE_TYPE)

if __name__ == '__main__':
    unittest.main()


















import unittest
import os
import sys
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.vector_search import VectorSearch
from rag.embedding import EmbeddingEngine

class MockEmbeddingEngine:
    """Mock embedding engine for testing."""
    
    def get_embedding(self, text):
        """Return a mock embedding."""
        # Simple mock embedding - hash the text to a number and use it to seed a random vector
        seed = hash(text) % 10000
        np.random.seed(seed)
        return np.random.random(384)  # Same dimension as the default embedding model
    
    def get_embeddings(self, texts):
        """Return mock embeddings for multiple texts."""
        return [self.get_embedding(text) for text in texts]

class TestVectorSearch(unittest.TestCase):
    """Test the vector search functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        self.embedding_engine = MockEmbeddingEngine()
        self.vector_search = VectorSearch(self.embedding_engine)
        
        # Create some test documents
        self.test_documents = [
            {
                "chunk_id": "doc1_chunk1",
                "source": "confluence",
                "source_id": "doc1",
                "title": "Python Programming",
                "content": "Python is a high-level programming language known for its readability and simplicity.",
                "url": "https://example.com/doc1"
            },
            {
                "chunk_id": "doc2_chunk1",
                "source": "confluence",
                "source_id": "doc2",
                "title": "Java Programming",
                "content": "Java is a popular programming language used for enterprise applications.",
                "url": "https://example.com/doc2"
            },
            {
                "chunk_id": "doc3_chunk1",
                "source": "jira",
                "source_id": "ISSUE-123",
                "title": "Database Connection Issue",
                "content": "Users are experiencing connection timeouts when accessing the database.",
                "url": "https://example.com/issue123"
            },
            {
                "chunk_id": "doc4_chunk1",
                "source": "remedy",
                "source_id": "INC12345",
                "title": "Email Server Down",
                "content": "The email server is not responding to requests. Users cannot send or receive emails.",
                "url": "https://example.com/incident12345"
            }
        ]
    
    def test_semantic_search(self):
        """Test semantic search functionality."""
        # Search for Python-related content
        results = self.vector_search._semantic_search(
            "Tell me about Python programming",
            self.test_documents,
            top_k=2,
            min_score=0.0  # Set min_score to 0 for testing
        )
        
        # Check that we got results
        self.assertGreater(len(results), 0)
        
        # Check that Python document is in the results
        python_doc_found = False
        for result in results:
            if "Python" in result["content"]:
                python_doc_found = True
                break
        
        self.assertTrue(python_doc_found)
    
    def test_lexical_search(self):
        """Test lexical search functionality."""
        # Search for email-related content
        results = self.vector_search._lexical_search(
            "email problems",
            self.test_documents,
            top_k=2,
            min_score=0.0  # Set min_score to 0 for testing
        )
        
        # Check that we got results
        self.assertGreater(len(results), 0)
        
        # Check that email document is in the results
        email_doc_found = False
        for result in results:
            if "email" in result["content"].lower():
                email_doc_found = True
                break
        
        self.assertTrue(email_doc_found)
    
    def test_hybrid_search(self):
        """Test hybrid search functionality."""
        # Search using hybrid approach
        results = self.vector_search.search(
            "database connection problems",
            self.test_documents,
            top_k=2,
            min_score=0.0  # Set min_score to 0 for testing
        )
        
        # Check that we got results
        self.assertGreater(len(results), 0)
        
        # Check that database document is in the results
        db_doc_found = False
        for result in results:
            if "database" in result["content"].lower():
                db_doc_found = True
                break
        
        self.assertTrue(db_doc_found)
    
    def test_relevance_scoring(self):
        """Test that results include relevance scores."""
        results = self.vector_search.search(
            "programming languages",
            self.test_documents,
            top_k=3,
            min_score=0.0
        )
        
        # Check that all results have relevance scores
        for result in results:
            self.assertIn("relevance", result)
            self.assertIsInstance(result["relevance"], float)
            
        # Check that results are ordered by relevance
        if len(results) >= 2:
            self.assertGreaterEqual(results[0]["relevance"], results[-1]["relevance"])

if __name__ == '__main__':
    unittest.main()















import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Flask app for testing
import app as flask_app

class TestApi(unittest.TestCase):
    """Test the API endpoints."""
    
    def setUp(self):
        """Set up the test environment."""
        # Configure Flask for testing
        flask_app.app.config['TESTING'] = True
        flask_app.app.config['DEBUG'] = False
        
        # Create a test client
        self.client = flask_app.app.test_client()
        
        # Mock the data sources and query processor
        self.mock_confluence_client = MagicMock()
        self.mock_remedy_client = MagicMock()
        self.mock_jira_client = MagicMock()
        self.mock_gemini_client = MagicMock()
        self.mock_query_processor = MagicMock()
        
        # Set up common test data
        self.test_query = "What is the status of the email server?"
        
        # Set up mock response data
        self.mock_response = {
            "answer": "The email server is currently experiencing issues. A ticket (INC12345) has been opened to address the problem.",
            "sources": [
                {
                    "name": "Email Server Status",
                    "url": "https://example.com/confluence/email-server",
                    "source_type": "confluence",
                    "source_id": "12345",
                    "relevance": 0.92
                },
                {
                    "name": "Email Server Down",
                    "url": "https://example.com/remedy/INC12345",
                    "source_type": "remedy",
                    "source_id": "INC12345",
                    "relevance": 0.87
                }
            ],
            "processing_time": 1.25
        }
    
    @patch('app.query_processor')
    def test_chat_endpoint(self, mock_query_processor):
        """Test the /api/chat endpoint."""
        # Configure the mock to return a predefined response
        mock_query_processor.process_query.return_value = self.mock_response
        
        # Send a POST request to the /api/chat endpoint
        response = self.client.post(
            '/api/chat',
            data=json.dumps({
                'message': self.test_query,
                'sources': ['confluence', 'remedy', 'jira']
            }),
            content_type='application/json'
        )
        
        # Check that the response is valid
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check the structure of the response
        self.assertIn('response', data)
        self.assertIn('sources', data)
        self.assertIn('processing_time', data)
        
        # Check that the mock was called with the correct parameters
        mock_query_processor.process_query.assert_called_once_with(
            query=self.test_query,
            sources=['confluence', 'remedy', 'jira']
        )
    
    @patch('app.confluence_client')
    @patch('app.remedy_client')
    @patch('app.jira_client')
    @patch('app.gemini_client')
    def test_health_check_endpoint(self, mock_gemini, mock_jira, mock_remedy, mock_confluence):
        """Test the /api/health endpoint."""
        # Configure the mocks to return predefined health status
        mock_confluence.test_connection.return_value = True
        mock_remedy.test_connection.return_value = True
        mock_jira.test_connection.return_value = False  # Simulate JIRA being down
        mock_gemini.test_connection.return_value = True
        
        # Send a GET request to the /api/health endpoint
        response = self.client.get('/api/health')
        
        # Check the response code (should be 503 since one service is down)
        self.assertEqual(response.status_code, 503)
        
        # Check the response structure
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
        self.assertTrue(data['confluence'])
        self.assertTrue(data['remedy'])
        self.assertFalse(data['jira'])
        self.assertTrue(data['gemini'])
        
        # Set all services to healthy
        mock_jira.test_connection.return_value = True
        
        # Send another GET request to the /api/health endpoint
        response = self.client.get('/api/health')
        
        # Check the response code (should be 200 since all services are up)
        self.assertEqual(response.status_code, 200)
    
    def test_missing_message(self):
        """Test sending a request with no message."""
        # Send a POST request to the /api/chat endpoint with no message
        response = self.client.post(
            '/api/chat',
            data=json.dumps({
                'sources': ['confluence', 'remedy', 'jira']
            }),
            content_type='application/json'
        )
        
        # Check that the response indicates an error
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()














    
