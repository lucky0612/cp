"""
Embedding module for the RAG application.
Simple implementation that avoids dependency issues with transformers.
"""
from typing import List, Dict, Any, Union, Optional
import os
import json
import numpy as np
import hashlib
import time

import config
from utils import get_logger, timer_decorator, generate_cache_key, ensure_directory

# Initialize logger
logger = get_logger("embedding")

class SimpleEmbedding:
    """Simple embedding implementation using deterministic hash-based vectors."""
    
    def __init__(self, dimension: int = None, cache_dir: str = None):
        """
        Initialize the simple embedding model.
        
        Args:
            dimension (int, optional): Embedding dimension. Defaults to config value.
            cache_dir (str, optional): Directory to cache embeddings. Defaults to config value.
        """
        self.dimension = dimension or config.EMBEDDING_DIMENSION
        self.cache_dir = cache_dir or os.path.join(config.CACHE_DIR, "embeddings")
        ensure_directory(self.cache_dir)
        self.logger = get_logger("simple_embedding")
        
        self.logger.info(f"Initialized simple embedding model with dimension {self.dimension}")
    
    def _get_cache_path(self, text: str) -> str:
        """
        Get the cache path for an embedding.
        
        Args:
            text (str): Text to generate cache path for
            
        Returns:
            str: Cache file path
        """
        key = generate_cache_key(text, prefix="emb")
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def _load_from_cache(self, text: str) -> Optional[List[float]]:
        """
        Load embedding from cache.
        
        Args:
            text (str): Text to load embedding for
            
        Returns:
            list or None: Embedding vector if found, None otherwise
        """
        cache_path = self._get_cache_path(text)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return data.get("embedding")
            except Exception as e:
                self.logger.error(f"Error loading embedding from cache: {e}")
        
        return None
    
    def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """
        Save embedding to cache.
        
        Args:
            text (str): Text that was embedded
            embedding (list): Embedding vector
        """
        cache_path = self._get_cache_path(text)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "embedding": embedding
                }, f)
        except Exception as e:
            self.logger.error(f"Error saving embedding to cache: {e}")
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """
        Generate a deterministic embedding based on text content.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list: Embedding vector
        """
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        # Clean and normalize text
        text = text.lower().strip()
        
        # Create a hash of the text to seed the random generator
        text_hash = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
        np.random.seed(text_hash)
        
        # Create a base random vector
        vec = np.random.rand(self.dimension).astype(np.float32)
        
        # Add semantic features based on text properties
        # Length feature
        vec[0] = min(1.0, len(text) / 1000.0)
        
        # Question feature
        if '?' in text:
            vec[1] = 0.9
            
        # Keyword features - set specific dimensions based on important words
        keywords = [
            "what", "how", "why", "when", "who", "where",  # Question words
            "is", "are", "was", "were", "will", "should",  # Verbs
            "and", "or", "not", "but", "however", "because",  # Connectors
            "important", "critical", "essential", "key",  # Significance words
            "problem", "issue", "error", "bug", "fix", "solution"  # Technical terms
        ]
        
        for i, keyword in enumerate(keywords):
            if keyword in text:
                # Use modulo to stay within dimension bounds
                pos = (i + 10) % (self.dimension - 10) + 5
                vec[pos] = 0.8
        
        # Count of sentences as a feature
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        vec[2] = min(1.0, sentence_count / 20.0)
        
        # Word count as a feature
        word_count = len(text.split())
        vec[3] = min(1.0, word_count / 200.0)
        
        # Normalize to unit length
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec.tolist()
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text with caching.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list: Embedding vector
        """
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        # Check cache first
        cached_embedding = self._load_from_cache(text)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate embedding
        embedding = self._generate_simple_embedding(text)
        
        # Save to cache
        self._save_to_cache(text, embedding)
        
        return embedding
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with caching.
        
        Args:
            texts (list): List of texts to embed
            
        Returns:
            list: List of embedding vectors
        """
        if not texts:
            return []
        
        # Check which texts need embedding
        to_embed = []
        to_embed_indices = []
        cached_embeddings = {}
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                cached_embeddings[i] = [0.0] * self.dimension
                continue
                
            cached = self._load_from_cache(text)
            if cached is not None:
                cached_embeddings[i] = cached
            else:
                to_embed.append(text)
                to_embed_indices.append(i)
        
        # Generate embeddings for texts not in cache
        new_embeddings = []
        for text in to_embed:
            embedding = self._generate_simple_embedding(text)
            new_embeddings.append(embedding)
            self._save_to_cache(text, embedding)
        
        # Combine cached and new embeddings
        all_embeddings = [None] * len(texts)
        
        # Add cached embeddings
        for idx, embedding in cached_embeddings.items():
            all_embeddings[idx] = embedding
        
        # Add new embeddings
        for i, idx in enumerate(to_embed_indices):
            all_embeddings[idx] = new_embeddings[i]
        
        return all_embeddings


class EmbeddingProcessor:
    """Process documents and generate embeddings for chunks."""
    
    def __init__(self):
        """Initialize the embedding processor with simple embedding."""
        self.embedding_model = SimpleEmbedding()
        self.logger = get_logger("embedding_processor")
    
    @timer_decorator
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process chunks and add embeddings.
        
        Args:
            chunks (list): List of document chunks
            
        Returns:
            list: List of chunks with embeddings
        """
        if not chunks:
            return []
        
        # Extract texts to embed
        texts = [chunk.get("text", "") for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.generate_embeddings(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        self.logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks
    
    @timer_decorator
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query (str): Query text
            
        Returns:
            list: Query embedding vector
        """
        return self.embedding_model.generate_embedding(query)
    
    @timer_decorator
    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple queries.
        
        Args:
            queries (list): List of query texts
            
        Returns:
            list: List of query embedding vectors
        """
        return self.embedding_model.generate_embeddings(queries)








"""
Main application module for the RAG application.
Fixed for compatibility with Flask version and templating issues.
"""
import os
import datetime
from flask import Flask, render_template, send_from_directory, abort

import config
from utils import get_logger
from modules.api import api_bp

# Initialize logger
logger = get_logger("app")

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, static_folder='static', template_folder='templates')
    
    # Configure app
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['DEBUG'] = config.DEBUG
    
    # Enable CORS for API routes only
    try:
        from flask_cors import CORS
        CORS(app, resources={r"/api/*": {"origins": "*"}})
    except ImportError:
        logger.warning("flask-cors not available. CORS not enabled.")
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Add routes with proper error handling
    @app.route('/')
    def index():
        """Render the main page."""
        try:
            return render_template('index.html')
        except Exception as e:
            logger.error(f"Error rendering index template: {e}")
            return "Error loading homepage. Please check logs.", 500
    
    @app.route('/chat')
    def chat():
        """Render the chat page."""
        try:
            return render_template('chat.html')
        except Exception as e:
            logger.error(f"Error rendering chat template: {e}")
            return "Error loading chat page. Please check logs.", 500
    
    @app.route('/favicon.ico')
    def favicon():
        """Serve favicon."""
        try:
            icon_path = os.path.join(app.root_path, 'static', 'images')
            return send_from_directory(icon_path, 'favicon.ico', mimetype='image/vnd.microsoft.icon')
        except Exception as e:
            logger.error(f"Error serving favicon: {e}")
            abort(404)
    
    # Add error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors."""
        try:
            return render_template('error.html', error_code=404, error_message="Page not found"), 404
        except Exception as err:
            logger.error(f"Error rendering error template: {err}")
            return "Page not found (404)", 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors."""
        logger.error(f"Server error: {str(e)}")
        try:
            return render_template('error.html', error_code=500, error_message="Server error"), 500
        except Exception as err:
            logger.error(f"Error rendering error template: {err}")
            return "Internal server error (500)", 500
    
    # Custom template context processor
    @app.context_processor
    def inject_now():
        """Add current datetime to template context."""
        return {'now': datetime.datetime.now}
    
    # Custom Jinja filter for date formatting
    @app.template_filter('now')
    def datetime_now(format_string):
        """Return current date formatted."""
        if format_string == 'year':
            return datetime.datetime.now().year
        return datetime.datetime.now().strftime(format_string)
    
    # Validate configuration
    if not config.validate_config():
        logger.warning("Application configuration is incomplete. Some features may not work.")
    
    logger.info(f"Application initialized in {config.DEBUG and 'DEBUG' or 'PRODUCTION'} mode")
    return app









# Core Flask
Flask>=2.0.0,<3.1.0
Flask-Cors>=3.0.0
Flask-RESTful>=0.3.9
python-dotenv>=0.20.0
gunicorn>=20.1.0

# Data processing
beautifulsoup4>=4.9.0
lxml>=4.6.0
html2text>=2020.1.16
jsonschema>=4.0.0
markdown>=3.3.0

# API and HTTP
requests>=2.25.0
requests-cache>=0.9.0
urllib3>=1.26.0,<2.1.0
PyJWT>=2.0.0

# Vector database and embeddings - simplified
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0
faiss-cpu>=1.7.0

# Text processing - simplified
nltk>=3.6.0
rank-bm25>=0.2.1

# Google AI
google-api-python-client>=2.0.0
google-auth>=2.0.0
google-auth-oauthlib>=0.5.0
google-cloud-aiplatform>=1.0.0
protobuf>=3.19.0,<4.0.0

# Utilities
tqdm>=4.50.0
pydantic>=1.8.0,<2.0.0
tenacity>=8.0.0
loguru>=0.5.0
python-slugify>=6.0.0
cachetools>=4.0.0









"""
Run script for the RAG application.
Enhanced with better error handling.
"""
import os
import sys
import traceback

# Try to import config first 
try:
    import config
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load configuration: {e}")
    print(traceback.format_exc())
    sys.exit(1)

# Then try to set up logging
try:
    from utils import get_logger
    logger = get_logger("run")
except Exception as e:
    print(f"ERROR: Failed to initialize logger: {e}")
    print(traceback.format_exc())
    print("Continuing without proper logging...")
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("run")

# Only run the app if this file is executed directly
if __name__ == '__main__':
    try:
        # Create the Flask application
        from app import create_app
        app = create_app()
        
        # Default port from config or environment
        port = int(os.environ.get("PORT", config.PORT))
        
        # Run the application
        logger.info(f"Starting application on port {port}")
        app.run(host='0.0.0.0', port=port, debug=config.DEBUG)
        
    except ImportError as e:
        logger.critical(f"Failed to import required module: {e}")
        print(f"CRITICAL ERROR: Failed to import required module: {e}")
        print(traceback.format_exc())
        sys.exit(1)
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        print(f"CRITICAL ERROR: Failed to start application: {e}")
        print(traceback.format_exc())
        sys.exit(1)











"""
Vector search module for the RAG application.
Updated to work with the SimpleEmbedding implementation.
"""
from typing import List, Dict, Any, Optional
import time

import config
from utils import get_logger, timer_decorator
from modules.processing import EmbeddingProcessor, IndexManager

# Initialize logger
logger = get_logger("vector_search")

class VectorSearchRetriever:
    """Retriever that uses vector similarity search."""
    
    def __init__(self, embedding_processor: EmbeddingProcessor = None, index_manager: IndexManager = None):
        """
        Initialize the vector search retriever.
        
        Args:
            embedding_processor (EmbeddingProcessor, optional): Embedding processor to use
            index_manager (IndexManager, optional): Index manager to use
        """
        self.embedding_processor = embedding_processor or EmbeddingProcessor()
        self.index_manager = index_manager or IndexManager()
        self.logger = get_logger("vector_search_retriever")
    
    @timer_decorator
    def search(self, query: str, k: int = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query.
        
        Args:
            query (str): Query text
            k (int, optional): Number of results to return. Defaults to config value.
            min_score (float, optional): Minimum similarity score. Defaults to 0.0.
            
        Returns:
            list: List of search results
        """
        # Use default k if not specified
        if k is None:
            k = config.NUM_RESULTS
        
        try:
            # Get query embedding
            start_time = time.time()
            query_embedding = self.embedding_processor.embed_query(query)
            embedding_time = time.time() - start_time
            
            # Search the index
            start_time = time.time()
            results = self.index_manager.search(query_embedding, k=k)
            search_time = time.time() - start_time
            
            # Filter by minimum score
            filtered_results = [result for result in results if result.get("score", 0) >= min_score]
            
            # Log search metrics
            self.logger.debug(f"Vector search metrics - Embedding: {embedding_time:.4f}s, Search: {search_time:.4f}s")
            self.logger.info(f"Vector search found {len(filtered_results)} results for query: {query[:50]}...")
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error in vector search: {str(e)}")
            return []
    
    @timer_decorator
    def multi_query_search(self, query: str, k: int = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search with multiple query variations to improve recall.
        
        Args:
            query (str): Original query text
            k (int, optional): Number of results to return per query variant. Defaults to config value.
            min_score (float, optional): Minimum similarity score. Defaults to 0.0.
            
        Returns:
            list: Merged list of search results
        """
        # Use default k if not specified
        if k is None:
            k = config.NUM_RESULTS
        
        try:
            # Generate query variations
            query_variations = self._generate_query_variations(query)
            
            # Get embeddings for all variations
            all_embeddings = self.embedding_processor.embed_queries(query_variations)
            
            # Search with each variation
            all_results = []
            for i, (variation, embedding) in enumerate(zip(query_variations, all_embeddings)):
                # Search the index
                results = self.index_manager.search(embedding, k=k)
                
                # Add query variant info to results
                for result in results:
                    result["query_variant"] = i
                    result["query_text"] = variation
                
                all_results.extend(results)
            
            # Merge results (remove duplicates, keep highest score)
            merged_results = {}
            
            for result in all_results:
                result_id = result.get("id")
                score = result.get("score", 0)
                
                if result_id not in merged_results or score > merged_results[result_id].get("score", 0):
                    merged_results[result_id] = result
            
            # Convert to list and sort by score
            results_list = list(merged_results.values())
            results_list.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Apply minimum score filter
            filtered_results = [result for result in results_list if result.get("score", 0) >= min_score]
            
            # Truncate to k results
            final_results = filtered_results[:k]
            
            self.logger.info(f"Multi-query search found {len(final_results)} results for query: {query[:50]}...")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in multi-query search: {str(e)}")
            return []
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """
        Generate variations of a query to improve retrieval.
        
        Args:
            query (str): Original query
            
        Returns:
            list: List of query variations
        """
        variations = [query]  # Original query
        
        try:
            # Add a "what is" variant if query is short and doesn't have it
            if len(query) < 100 and not query.lower().startswith("what is"):
                variations.append(f"What is {query}?")
            
            # Add a more detailed variant
            if len(query) < 100:
                variations.append(f"Explain in detail about {query}")
            
            # Add a simpler variant (first 8 words or so)
            words = query.split()
            if len(words) > 8:
                variations.append(" ".join(words[:8]))
            
            # If query is a question, add an instruction variant
            if "?" in query:
                variations.append(query.replace("?", ""))
            
            # Remove duplicates
            unique_variations = list(dict.fromkeys(variations))
            
            self.logger.debug(f"Generated {len(unique_variations)} query variations")
            return unique_variations
            
        except Exception as e:
            self.logger.error(f"Error generating query variations: {str(e)}")
            return [query]  # Return original query on error
    
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Index document chunks.
        
        Args:
            chunks (list): List of document chunks
        """
        try:
            # Generate embeddings for chunks
            chunks_with_embeddings = self.embedding_processor.process_chunks(chunks)
            
            # Index the chunks
            self.index_manager.index_chunks(chunks_with_embeddings)
            
            # Save the index
            self.index_manager.save_index()
            
        except Exception as e:
            self.logger.error(f"Error indexing chunks: {str(e)}")











<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}RAG Confluence & Remedy{% endblock %}</title>
    <link rel="shortcut icon" href="{{ url_for('static', filename='images/favicon.ico') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    {% block extra_css %}{% endblock %}
</head>
<body>
    <header>
        <div class="header-container">
            <div class="logo">
                <a href="/">
                    <img src="{{ url_for('static', filename='images/logo.png') }}" alt="Logo">
                    <span>RAG Assistant</span>
                </a>
            </div>
            <nav>
                <ul>
                    <li><a href="/" {% if request.path == '/' %}class="active"{% endif %}>Home</a></li>
                    <li><a href="/chat" {% if request.path == '/chat' %}class="active"{% endif %}>Chat</a></li>
                </ul>
            </nav>
        </div>
    </header>
    
    <main>
        <div class="container">
            {% block content %}{% endblock %}
        </div>
    </main>
    
    <footer>
        <div class="container">
            <p>&copy; 2024 RAG Confluence & Remedy Assistant</p>
        </div>
    </footer>
    
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>










"""
Test script to verify Vertex AI setup.
Run this before starting the main application to check Google Cloud connectivity.
"""
import os
import sys
import traceback

def check_vertexai():
    """Test Vertex AI initialization and model access."""
    try:
        print("Testing Vertex AI connectivity...")
        
        # Try to import required packages
        try:
            import google.cloud.aiplatform as vertexai
            from google.auth import default
            print("✅ Successfully imported Google Cloud packages")
        except ImportError as e:
            print(f"❌ Failed to import Google Cloud packages: {e}")
            return False
        
        # Try to get project ID from config
        try:
            import config
            project_id = config.PROJECT_ID
            region = config.REGION
            print(f"✅ Found configuration: Project ID = {project_id}, Region = {region}")
        except (ImportError, AttributeError) as e:
            print(f"❌ Failed to get project configuration: {e}")
            return False
        
        # Try to get credentials
        try:
            credentials, project = default()
            print(f"✅ Successfully obtained credentials for project: {project}")
        except Exception as e:
            print(f"❌ Failed to obtain Google Cloud credentials: {e}")
            print("   Make sure you have run: gcloud auth application-default login")
            return False
        
        # Try to initialize Vertex AI
        try:
            vertexai.init(project=project_id, location=region)
            print(f"✅ Successfully initialized Vertex AI in {region}")
        except Exception as e:
            print(f"❌ Failed to initialize Vertex AI: {e}")
            return False
        
        # Try to access the model
        try:
            model_name = config.MODEL_NAME
            print(f"Checking model access: {model_name}")
            from vertexai.generative_models import GenerativeModel
            
            model = GenerativeModel(model_name)
            print(f"✅ Successfully accessed model: {model_name}")
        except Exception as e:
            print(f"❌ Failed to access model {model_name}: {e}")
            print(traceback.format_exc())
            return False
        
        print("✅ Vertex AI setup looks good!")
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error during Vertex AI check: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = check_vertexai()
    if not success:
        print("\n⚠️ Vertex AI setup has issues that need to be resolved before running the application.")
        sys.exit(1)
    else:
        print("\n✅ All Vertex AI checks passed! You can now run the main application.")





        
