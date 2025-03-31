"""
Embedding utilities for Enterprise RAG System without external dependencies.
"""
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.tokenize import word_tokenize
import hashlib

import config
from utils.logger import get_logger
from utils.cache import cache_manager

# Ensure NLTK resources are downloaded locally if available
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    # If download fails, continue - we'll use simple tokenization
    pass

logger = get_logger(__name__)

class EmbeddingManager:
    """
    Handles text embedding generation without external model dependencies.
    Uses TF-IDF and SVD for creating vector representations.
    """
    
    def __init__(self, vector_dimension=None):
        """
        Initialize the embedding manager.
        
        Args:
            vector_dimension: Dimension for embedding vectors
        """
        self.vector_dimension = vector_dimension or config.VECTOR_DIMENSION
        self.tfidf_vectorizer = None
        self.svd_transformer = None
        self.is_fitted = False
        
        # Initialize the vectorizer and transformer
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the TF-IDF vectorizer and SVD transformer."""
        try:
            # Initialize TF-IDF vectorizer for basic text representation
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),
                max_features=10000
            )
            
            # Initialize SVD transformer for dimensionality reduction
            self.svd_transformer = TruncatedSVD(
                n_components=self.vector_dimension,
                random_state=42
            )
            
            logger.info(f"Initialized TF-IDF vectorizer and SVD transformer for embeddings (dimension: {self.vector_dimension})")
        except Exception as e:
            logger.error(f"Error initializing embedding components: {e}")
            raise RuntimeError(f"Failed to initialize embedding components: {str(e)}")
    
    def _fit_on_texts(self, texts):
        """
        Fit the TF-IDF vectorizer and SVD transformer on texts.
        
        Args:
            texts: List of input texts
        """
        if not texts:
            logger.warning("No texts provided for fitting embedding model")
            return
        
        try:
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Fit SVD transformer
            self.svd_transformer.fit(tfidf_matrix)
            
            self.is_fitted = True
            logger.info(f"Fitted embedding model on {len(texts)} texts")
        except Exception as e:
            logger.error(f"Error fitting embedding model: {e}")
            self.is_fitted = False
    
    def get_embedding(self, text):
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.vector_dimension)
        
        # Generate a unique cache key based on the text content
        cache_key = f"embedding:tfidf-svd:{hashlib.md5(text.encode()).hexdigest()}"
        cached_embedding = cache_manager.get(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            # Transform text to TF-IDF representation
            tfidf_vector = self.tfidf_vectorizer.transform([text])
            
            # Transform TF-IDF to lower-dimensional embedding via SVD
            embedding = self.svd_transformer.transform(tfidf_vector)[0]
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            # Cache the embedding
            cache_manager.set(cache_key, embedding)
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            
            # If model is not fitted, try to fit on this single text
            if not self.is_fitted:
                logger.info("Attempting to fit embedding model on first text")
                self._fit_on_texts([text])
                
                # Try again after fitting
                try:
                    tfidf_vector = self.tfidf_vectorizer.transform([text])
                    embedding = self.svd_transformer.transform(tfidf_vector)[0]
                    
                    # Normalize the embedding
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    return embedding
                except Exception as e2:
                    logger.error(f"Error generating embedding after fitting: {e2}")
            
            # Return zero vector in case of error
            return np.zeros(self.vector_dimension)
    
    def get_embeddings(self, texts, batch_size=32):
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for embedding generation
            
        Returns:
            List of numpy.ndarray: Embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts and check cache
        non_empty_texts = []
        cached_embeddings = {}
        indices_to_generate = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                cached_embeddings[i] = np.zeros(self.vector_dimension)
            else:
                cache_key = f"embedding:tfidf-svd:{hashlib.md5(text.encode()).hexdigest()}"
                cached_embedding = cache_manager.get(cache_key)
                
                if cached_embedding is not None:
                    cached_embeddings[i] = cached_embedding
                else:
                    non_empty_texts.append(text)
                    indices_to_generate.append(i)
        
        # If not fitted and we have texts, fit the model first
        if non_empty_texts and not self.is_fitted:
            logger.info(f"Fitting embedding model on {len(non_empty_texts)} texts")
            self._fit_on_texts(non_empty_texts)
        
        # Generate embeddings for texts not in cache
        if non_empty_texts and self.is_fitted:
            try:
                # Transform texts to TF-IDF representation
                tfidf_matrix = self.tfidf_vectorizer.transform(non_empty_texts)
                
                # Transform TF-IDF to lower-dimensional embeddings via SVD
                embeddings = self.svd_transformer.transform(tfidf_matrix)
                
                # Normalize the embeddings
                for j, embedding in enumerate(embeddings):
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embeddings[j] = embedding / norm
                
                # Cache and store the new embeddings
                for j, idx in enumerate(indices_to_generate):
                    embedding = embeddings[j]
                    text = texts[idx]
                    cache_key = f"embedding:tfidf-svd:{hashlib.md5(text.encode()).hexdigest()}"
                    cache_manager.set(cache_key, embedding)
                    cached_embeddings[idx] = embedding
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                # Use zero vectors for failed embeddings
                for idx in indices_to_generate:
                    cached_embeddings[idx] = np.zeros(self.vector_dimension)
        elif indices_to_generate:
            # If we couldn't fit the model but have indices to generate
            for idx in indices_to_generate:
                cached_embeddings[idx] = np.zeros(self.vector_dimension)
        
        # Reconstruct the ordered list of embeddings
        embeddings = [cached_embeddings[i] for i in range(len(texts))]
        return embeddings
    
    def embed_chunks(self, chunks):
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of chunks with added embeddings
        """
        if not chunks:
            return []
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.get_embeddings(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding['embedding'] = embedding
            embedded_chunks.append(chunk_with_embedding)
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return embedded_chunks
    
    def save_embeddings(self, embeddings, filepath):
        """
        Save embeddings to a file.
        
        Args:
            embeddings: List of embeddings or dict mapping IDs to embeddings
            filepath: Output file path
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved embeddings to {filepath}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def load_embeddings(self, filepath):
        """
        Load embeddings from a file.
        
        Args:
            filepath: Input file path
            
        Returns:
            Loaded embeddings
        """
        try:
            with open(filepath, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"Loaded embeddings from {filepath}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None
    
    def get_query_embedding(self, query):
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            numpy.ndarray: Query embedding vector
        """
        return self.get_embedding(query)
    
    def save_model(self, filepath):
        """
        Save the embedding model to a file.
        
        Args:
            filepath: Output file path
        """
        try:
            model_data = {
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'svd_transformer': self.svd_transformer,
                'vector_dimension': self.vector_dimension,
                'is_fitted': self.is_fitted
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved embedding model to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving embedding model: {e}")
            return False
    
    def load_model(self, filepath):
        """
        Load the embedding model from a file.
        
        Args:
            filepath: Input file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.svd_transformer = model_data['svd_transformer']
            self.vector_dimension = model_data['vector_dimension']
            self.is_fitted = model_data['is_fitted']
            
            logger.info(f"Loaded embedding model from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            return False

# Global embedding manager instance
embedding_manager = EmbeddingManager()


















"""
Retrieval components for Enterprise RAG System.
"""
import os
import pickle
import numpy as np
import faiss
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk

import config
from utils.logger import get_logger
from utils.cache import cache_manager
from rag_engine.embedding import embedding_manager

# Try to download NLTK resources if available locally
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    pass  # Handle silently, log below

logger = get_logger(__name__)

class RetrievalEngine:
    """
    Handles document retrieval using vector similarity search and TF-IDF ranking.
    Supports hybrid retrieval combining multiple ranking methods.
    """
    
    def __init__(self):
        """Initialize the retrieval engine."""
        self.vector_index = None
        self.chunks = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Try to ensure NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.warning("NLTK punkt tokenizer not found, will use simple tokenization")
    
    def initialize_vector_index(self, dimension=None):
        """
        Initialize the FAISS vector index.
        
        Args:
            dimension: Dimension of the embedding vectors
        """
        dimension = dimension or embedding_manager.vector_dimension
        
        try:
            # Create a new FAISS index for L2 similarity
            self.vector_index = faiss.IndexFlatL2(dimension)
            logger.info(f"Initialized FAISS vector index with dimension {dimension}")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            # Fallback to a simple list if FAISS fails
            logger.info("Using fallback vector storage instead of FAISS")
            self.vector_index = None
    
    def add_documents(self, chunks):
        """
        Add document chunks to the retrieval engine.
        
        Args:
            chunks: List of document chunks with embeddings
        """
        if not chunks:
            logger.warning("No chunks provided to add to retrieval engine")
            return
        
        # Store the chunks
        self.chunks = chunks
        
        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in chunks], dtype=np.float32)
        
        # Initialize vector index if needed
        if self.vector_index is None:
            self.initialize_vector_index(embeddings.shape[1])
        
        # Add embeddings to the index
        try:
            if isinstance(self.vector_index, faiss.Index):
                self.vector_index.add(embeddings)
                logger.info(f"Added {len(chunks)} chunks to the FAISS vector index")
            else:
                # We're using the fallback approach, nothing to do as chunks are already stored
                logger.info(f"Added {len(chunks)} chunks to fallback storage")
        except Exception as e:
            logger.error(f"Error adding embeddings to vector index: {e}")
        
        # Initialize TF-IDF for lexical search
        self._initialize_tfidf([chunk['text'] for chunk in chunks])
    
    def _initialize_tfidf(self, texts):
        """
        Initialize TF-IDF vectorizer for lexical search.
        
        Args:
            texts: List of document texts
        """
        try:
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),
                max_features=10000
            )
            
            # Fit and transform the texts
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info("Initialized TF-IDF vectorizer for lexical search")
        except Exception as e:
            logger.error(f"Error initializing TF-IDF vectorizer: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def save_index(self, directory):
        """
        Save the retrieval index and related data.
        
        Args:
            directory: Directory to save the index files
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save FAISS index if available
            if isinstance(self.vector_index, faiss.Index):
                index_path = os.path.join(directory, 'vector_index.faiss')
                faiss.write_index(self.vector_index, index_path)
            
            # Save chunks
            chunks_path = os.path.join(directory, 'chunks.pkl')
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            # Save TF-IDF vectorizer and matrix
            tfidf_path = os.path.join(directory, 'tfidf.pkl')
            with open(tfidf_path, 'wb') as f:
                pickle.dump((self.tfidf_vectorizer, self.tfidf_matrix), f)
            
            logger.info(f"Saved retrieval index to {directory}")
        except Exception as e:
            logger.error(f"Error saving retrieval index: {e}")
    
    def load_index(self, directory):
        """
        Load the retrieval index and related data.
        
        Args:
            directory: Directory containing the index files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load FAISS index if available
            index_path = os.path.join(directory, 'vector_index.faiss')
            if os.path.exists(index_path):
                self.vector_index = faiss.read_index(index_path)
            
            # Load chunks
            chunks_path = os.path.join(directory, 'chunks.pkl')
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            # Load TF-IDF vectorizer and matrix
            tfidf_path = os.path.join(directory, 'tfidf.pkl')
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer, self.tfidf_matrix = pickle.load(f)
            
            logger.info(f"Loaded retrieval index from {directory}")
            return True
        except Exception as e:
            logger.error(f"Error loading retrieval index: {e}")
            return False
    
    def _compute_similarity(self, query_embedding, doc_embedding):
        """
        Compute cosine similarity between embeddings.
        
        Args:
            query_embedding: Query embedding vector
            doc_embedding: Document embedding vector
            
        Returns:
            float: Similarity score
        """
        try:
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(query_embedding, doc_embedding)
            return similarity
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def vector_search(self, query_embedding, k=None):
        """
        Perform vector similarity search.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        k = k or config.TOP_K_RESULTS
        
        if not self.chunks:
            logger.warning("No chunks available for vector search")
            return []
        
        try:
            results = []
            
            # Use FAISS if available
            if isinstance(self.vector_index, faiss.Index):
                # Reshape query embedding for FAISS
                query_embedding = np.array([query_embedding], dtype=np.float32)
                
                # Perform the search
                distances, indices = self.vector_index.search(query_embedding, k)
                
                # Format the results
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    # Skip invalid indices
                    if idx < 0 or idx >= len(self.chunks):
                        continue
                    
                    # Calculate cosine similarity (1 - distance_normalized)
                    # FAISS uses L2 distance, so we convert to a similarity score
                    max_distance = np.sqrt(2)  # Max L2 distance for normalized vectors
                    similarity = 1 - (distance / max_distance)
                    
                    results.append((self.chunks[idx], float(similarity)))
            else:
                # Fallback to brute force search if FAISS is not available
                similarities = []
                
                for i, chunk in enumerate(self.chunks):
                    doc_embedding = chunk['embedding']
                    similarity = self._compute_similarity(query_embedding, doc_embedding)
                    similarities.append((i, similarity))
                
                # Sort by similarity (descending) and take top k
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_indices = similarities[:k]
                
                # Format the results
                for idx, similarity in top_indices:
                    results.append((self.chunks[idx], float(similarity)))
            
            logger.debug(f"Vector search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            return []
    
    def bm25_search(self, query, k=None):
        """
        Perform BM25-like lexical search using TF-IDF.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        k = k or config.TOP_K_RESULTS
        
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None or not self.chunks:
            logger.warning("TF-IDF not initialized or no chunks available")
            return []
        
        try:
            # Transform query to TF-IDF space
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarity scores
            similarity_scores = (self.tfidf_matrix @ query_vector.T).toarray().flatten()
            
            # Get the top k results
            top_indices = similarity_scores.argsort()[-k:][::-1]
            
            # Format the results
            results = []
            for idx in top_indices:
                if similarity_scores[idx] > 0:
                    results.append((self.chunks[idx], float(similarity_scores[idx])))
            
            logger.debug(f"BM25 search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error performing BM25 search: {e}")
            return []
    
    def hybrid_search(self, query, query_embedding=None, k=None, alpha=None):
        """
        Perform hybrid search combining vector and BM25 search.
        
        Args:
            query: Query text
            query_embedding: Query embedding vector (optional)
            k: Number of results to return
            alpha: Weight for vector search (0-1)
            
        Returns:
            List of (chunk, score) tuples
        """
        k = k or config.TOP_K_RESULTS
        alpha = alpha or config.HYBRID_ALPHA
        
        # Generate query embedding if not provided
        if query_embedding is None:
            query_embedding = embedding_manager.get_query_embedding(query)
        
        # Cache key for hybrid search
        cache_key = f"hybrid_search:{hash(query)}:{k}:{alpha}"
        cached_results = cache_manager.get(cache_key)
        if cached_results:
            return cached_results
        
        # Perform vector search
        vector_results = self.vector_search(query_embedding, k=k*2)
        
        # Perform BM25 search
        bm25_results = self.bm25_search(query, k=k*2)
        
        # Combine results using the specified alpha
        combined_scores = {}
        
        # Add vector search results
        for chunk, score in vector_results:
            chunk_id = id(chunk)  # Use object id as a unique identifier
            combined_scores[chunk_id] = {
                'chunk': chunk,
                'vector_score': score,
                'bm25_score': 0,
                'combined_score': alpha * score
            }
        
        # Add BM25 search results
        for chunk, score in bm25_results:
            chunk_id = id(chunk)
            if chunk_id in combined_scores:
                combined_scores[chunk_id]['bm25_score'] = score
                combined_scores[chunk_id]['combined_score'] += (1 - alpha) * score
            else:
                combined_scores[chunk_id] = {
                    'chunk': chunk,
                    'vector_score': 0,
                    'bm25_score': score,
                    'combined_score': (1 - alpha) * score
                }
        
        # Sort by combined score and take top k
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:k]
        
        # Format final results
        results = [(item['chunk'], item['combined_score']) for item in sorted_results]
        
        # Cache the results
        cache_manager.set(cache_key, results, timeout=3600)
        
        logger.info(f"Hybrid search returned {len(results)} results with alpha={alpha}")
        return results
    
    def retrieve(self, query, search_type='hybrid', k=None):
        """
        Retrieve documents for a query.
        
        Args:
            query: Query text
            search_type: Type of search ('vector', 'bm25', or 'hybrid')
            k: Number of results to return
            
        Returns:
            List of retrieved document chunks with scores
        """
        k = k or config.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = embedding_manager.get_query_embedding(query)
        
        # Determine which search method to use based on availability
        if search_type == 'vector' and (self.vector_index is not None or self.chunks):
            results = self.vector_search(query_embedding, k=k)
        elif search_type == 'bm25' and self.tfidf_vectorizer is not None:
            results = self.bm25_search(query, k=k)
        else:
            # Default to hybrid search when possible, fallback to available methods
            if self.vector_index is not None and self.tfidf_vectorizer is not None:
                results = self.hybrid_search(query, query_embedding, k=k)
            elif self.vector_index is not None or self.chunks:
                logger.info("TF-IDF not available, using vector search")
                results = self.vector_search(query_embedding, k=k)
            elif self.tfidf_vectorizer is not None:
                logger.info("Vector index not available, using BM25 search")
                results = self.bm25_search(query, k=k)
            else:
                logger.warning("No search methods available")
                return []
        
        # Format the results for the RAG engine
        retrieved_chunks = []
        for chunk, score in results:
            retrieved_chunk = {
                'content': chunk['text'],
                'metadata': chunk.get('metadata', {}),
                'source': chunk.get('metadata', {}).get('source', 'Unknown'),
                'source_type': chunk.get('metadata', {}).get('source_type', 'Unknown'),
                'source_link': chunk.get('metadata', {}).get('source_link', ''),
                'score': score
            }
            retrieved_chunks.append(retrieved_chunk)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query}")
        return retrieved_chunks

# Global retrieval engine instance
retrieval_engine = RetrievalEngine()




















"""
Main RAG processor for Enterprise RAG System.
"""
import time
from concurrent.futures import ThreadPoolExecutor
import os
import pickle

import config
from utils.logger import get_logger
from utils.cache import cache_manager
from rag_engine.chunking import document_chunker
from rag_engine.embedding import embedding_manager
from rag_engine.retrieval import retrieval_engine
from rag_engine.gemini_integration import gemini_manager

logger = get_logger(__name__)

class RAGProcessor:
    """
    Main RAG processor handling the end-to-end flow:
    1. Document ingestion and chunking
    2. Embedding generation
    3. Retrieval
    4. Response generation using LLM
    """
    
    def __init__(self):
        """Initialize the RAG processor."""
        self.is_initialized = False
        self.document_sources = {}
        self.cache_dir = os.path.join(config.CACHE_DIR, 'rag_processor')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def initialize(self, force=False):
        """
        Initialize the RAG processor components.
        
        Args:
            force: Force reinitialization even if already initialized
        """
        if self.is_initialized and not force:
            return
        
        try:
            # Try to load cached embeddings and index if available
            embeddings_path = os.path.join(self.cache_dir, 'embedding_model.pkl')
            index_path = os.path.join(self.cache_dir, 'retrieval_index')
            
            # Load embedding model if available
            if os.path.exists(embeddings_path) and not force:
                try:
                    embedding_manager.load_model(embeddings_path)
                    logger.info("Loaded embedding model from cache")
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding model: {e}")
            
            # Load retrieval index if available
            if os.path.exists(index_path) and not force:
                try:
                    retrieval_engine.load_index(index_path)
                    logger.info("Loaded retrieval index from cache")
                except Exception as e:
                    logger.warning(f"Failed to load cached retrieval index: {e}")
            
            self.is_initialized = True
            logger.info("RAG processor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG processor: {e}")
            self.is_initialized = False
    
    def register_document_source(self, source_id, source_manager):
        """
        Register a document source.
        
        Args:
            source_id: Unique identifier for the source
            source_manager: Source manager instance
        """
        self.document_sources[source_id] = source_manager
        logger.info(f"Registered document source: {source_id}")
    
    def process_documents(self, documents, source_type=None):
        """
        Process documents for RAG.
        
        Args:
            documents: List of documents to process
            source_type: Type of source (e.g., 'confluence', 'jira', 'remedy')
            
        Returns:
            List of processed chunks with embeddings
        """
        if not documents:
            logger.warning("No documents provided for processing")
            return []
        
        # Step 1: Chunk documents
        start_time = time.time()
        chunks = []
        
        for doc in documents:
            # Add source type to metadata
            if isinstance(doc, dict) and 'metadata' in doc and source_type:
                doc['metadata']['source_type'] = source_type
            
            # Chunk the document using semantic boundaries by default
            doc_chunks = document_chunker.chunk_document(doc, method='semantic')
            chunks.extend(doc_chunks)
        
        chunk_time = time.time() - start_time
        logger.info(f"Chunked {len(documents)} documents into {len(chunks)} chunks in {chunk_time:.2f}s")
        
        # Step 2: Generate embeddings
        start_time = time.time()
        chunks_with_embeddings = embedding_manager.embed_chunks(chunks)
        embed_time = time.time() - start_time
        logger.info(f"Generated embeddings for {len(chunks)} chunks in {embed_time:.2f}s")
        
        # Save the embedding model for future use
        embeddings_path = os.path.join(self.cache_dir, 'embedding_model.pkl')
        try:
            embedding_manager.save_model(embeddings_path)
            logger.info(f"Saved embedding model to {embeddings_path}")
        except Exception as e:
            logger.warning(f"Failed to save embedding model: {e}")
        
        return chunks_with_embeddings
    
    def index_documents(self, chunks_with_embeddings, overwrite=False):
        """
        Index document chunks for retrieval.
        
        Args:
            chunks_with_embeddings: List of chunks with embeddings
            overwrite: Whether to overwrite existing index
        """
        if not chunks_with_embeddings:
            logger.warning("No chunks provided for indexing")
            return
        
        # If overwrite, create a new index
        if overwrite or retrieval_engine.vector_index is None:
            retrieval_engine.initialize_vector_index()
        
        # Add chunks to the index
        retrieval_engine.add_documents(chunks_with_embeddings)
        
        # Save the index for future use
        index_path = os.path.join(self.cache_dir, 'retrieval_index')
        try:
            retrieval_engine.save_index(index_path)
            logger.info(f"Saved retrieval index to {index_path}")
        except Exception as e:
            logger.warning(f"Failed to save retrieval index: {e}")
    
    def retrieve_relevant_chunks(self, query, search_type='hybrid', top_k=None):
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            search_type: Type of search ('vector', 'bm25', or 'hybrid')
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks
        """
        top_k = top_k or config.TOP_K_RESULTS
        
        # Check if retrieval engine is ready
        if not retrieval_engine.chunks:
            logger.warning("Retrieval engine not initialized with documents")
            return []
        
        # Cache key for retrieval
        cache_key = f"retrieval:{hash(query)}:{search_type}:{top_k}"
        cached_results = cache_manager.get(cache_key)
        if cached_results:
            return cached_results
        
        # Retrieve chunks
        start_time = time.time()
        
        # Choose appropriate search method based on availability
        available_methods = []
        if retrieval_engine.vector_index is not None or retrieval_engine.chunks:
            available_methods.append('vector')
        if retrieval_engine.tfidf_vectorizer is not None:
            available_methods.append('bm25')
        if len(available_methods) >= 2:
            available_methods.append('hybrid')
        
        # If requested method is not available, use the best available
        if search_type not in available_methods and available_methods:
            logger.warning(f"Search type '{search_type}' not available, using '{available_methods[0]}' instead")
            search_type = available_methods[0]
        
        # Retrieve chunks
        retrieved_chunks = retrieval_engine.retrieve(query, search_type, k=top_k)
        retrieval_time = time.time() - start_time
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.2f}s using {search_type} search")
        
        # Cache the results
        cache_manager.set(cache_key, retrieved_chunks, timeout=1800)  # 30 minutes
        
        return retrieved_chunks
    
    def generate_response(self, query, context_items, conversation_history=None, stream=False):
        """
        Generate a response using the Gemini LLM.
        
        Args:
            query: User query
            context_items: Retrieved context items
            conversation_history: Optional conversation history
            stream: Whether to stream the response
            
        Returns:
            Generated response or generator
        """
        start_time = time.time()
        
        # Generate response
        response = gemini_manager.generate_response(
            query, 
            context_items, 
            conversation_history=conversation_history,
            stream=stream
        )
        
        if not stream:
            generation_time = time.time() - start_time
            logger.info(f"Generated response in {generation_time:.2f}s")
        
        return response
    
    def process_query(self, query, sources=None, search_type=None, top_k=None, stream=False):
        """
        Process a user query through the entire RAG pipeline.
        
        Args:
            query: User query
            sources: List of source types to query ('confluence', 'jira', 'remedy')
            search_type: Type of search ('vector', 'bm25', or 'hybrid')
            top_k: Number of chunks to retrieve
            stream: Whether to stream the response
            
        Returns:
            Generated response and retrieved chunks
        """
        self.initialize()
        
        # Determine best search method if not specified
        if not search_type:
            if retrieval_engine.vector_index is not None and retrieval_engine.tfidf_vectorizer is not None:
                search_type = 'hybrid'
            elif retrieval_engine.vector_index is not None or retrieval_engine.chunks:
                search_type = 'vector'
            elif retrieval_engine.tfidf_vectorizer is not None:
                search_type = 'bm25'
            else:
                search_type = 'hybrid'  # Default, will be handled later if not available
        
        # If no specific sources are provided, use all available
        if not sources:
            sources = list(self.document_sources.keys())
        elif isinstance(sources, str):
            sources = [sources]
        
        # Retrieve chunks from specified sources
        if sources and self.document_sources:
            # Check if we need to retrieve and index documents
            if not retrieval_engine.chunks:
                self._retrieve_and_index_sources(sources)
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve_relevant_chunks(query, search_type, top_k)
        
        # Generate response
        response = self.generate_response(query, retrieved_chunks, stream=stream)
        
        return {
            'response': response,
            'context': retrieved_chunks
        }
    
    def _retrieve_and_index_sources(self, sources):
        """
        Retrieve documents from sources and index them.
        
        Args:
            sources: List of source types to retrieve from
        """
        all_chunks = []
        
        # Use ThreadPoolExecutor for parallel document retrieval
        with ThreadPoolExecutor(max_workers=min(len(sources), 3)) as executor:
            future_to_source = {}
            
            # Start document retrieval for each source
            for source in sources:
                if source in self.document_sources:
                    source_manager = self.document_sources[source]
                    future = executor.submit(source_manager.get_documents)
                    future_to_source[future] = source
            
            # Process results as they complete
            for future in future_to_source:
                source = future_to_source[future]
                try:
                    documents = future.result()
                    if documents:
                        # Process and chunk documents
                        chunks = self.process_documents(documents, source_type=source)
                        all_chunks.extend(chunks)
                        logger.info(f"Retrieved and processed {len(documents)} documents from {source}")
                except Exception as e:
                    logger.error(f"Error retrieving documents from {source}: {e}")
        
        # Index all chunks
        if all_chunks:
            self.index_documents(all_chunks)
            logger.info(f"Indexed {len(all_chunks)} chunks from {len(sources)} sources")
    
    def load_index(self, directory):
        """
        Load a saved retrieval index.
        
        Args:
            directory: Directory containing the index files
            
        Returns:
            bool: True if successful, False otherwise
        """
        return retrieval_engine.load_index(directory)
    
    def save_index(self, directory):
        """
        Save the current retrieval index.
        
        Args:
            directory: Directory to save the index files
        """
        retrieval_engine.save_index(directory)
    
    def clear_cache(self):
        """Clear all cached data."""
        cache_manager.clear()
        logger.info("Cleared all cache data")

# Global RAG processor instance
rag_processor = RAGProcessor()




















"""
Confluence connector for the RAG system.
"""
import os
import time
from utils.logger import get_logger
from utils.cache import cache_manager
from utils.content_parser import content_parser
from data_sources.confluence.client import ConfluenceClient
import config

logger = get_logger(__name__)

class ConfluenceConnector:
    """
    Connector for Confluence data source.
    Handles retrieving and processing Confluence content for RAG.
    """
    
    def __init__(self, base_url=None, username=None, api_token=None, ssl_verify=True):
        """
        Initialize the Confluence connector.
        
        Args:
            base_url: The base URL of the Confluence server
            username: Username for authentication
            api_token: API token for authentication
            ssl_verify: Whether to verify SSL certificates
        """
        self.client = ConfluenceClient(base_url, username, api_token, ssl_verify)
        self.cache_dir = os.path.join(config.CACHE_DIR, 'confluence')
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_documents(self, space_key=None, max_pages=100):
        """
        Get documents from Confluence.
        
        Args:
            space_key: Optional space key to filter content
            max_pages: Maximum number of pages to retrieve
            
        Returns:
            List of processed documents
        """
        # Cache key for documents
        cache_key = f"confluence_documents:{space_key}:{max_pages}"
        cached_docs = cache_manager.get(cache_key)
        if cached_docs is not None:
            logger.info(f"Using cached Confluence documents ({len(cached_docs)})")
            return cached_docs
        
        logger.info(f"Retrieving Confluence documents (space_key={space_key}, max_pages={max_pages})")
        
        # Test the connection first
        if not self.client.test_connection():
            logger.error("Failed to connect to Confluence. Check credentials and network.")
            return []
        
        try:
            # Get all content of type 'page'
            all_pages = self.client.get_all_content(
                content_type="page",
                limit=100,  # Batch size
                expand="body.storage,metadata.labels",
                space_key=space_key
            )
            
            # Limit to max_pages
            if len(all_pages) > max_pages:
                all_pages = all_pages[:max_pages]
            
            # Process each page
            documents = []
            
            for i, page in enumerate(all_pages):
                try:
                    page_id = page.get('id')
                    if not page_id:
                        continue
                    
                    # Check if this page already has expanded content
                    if 'body' in page and 'storage' in page['body'] and page['body']['storage'].get('value'):
                        # If content is already expanded, process it directly
                        page_content = {
                            'body': {'storage': page['body']['storage']}
                        }
                    else:
                        # Otherwise, get the full page content
                        page_content = self.client.get_content(
                            page_id,
                            expand="body.storage,metadata.labels"
                        )
                    
                    if not page_content:
                        continue
                    
                    # Extract and process the content
                    parsed_content = content_parser.parse_confluence_content(page_content)
                    
                    if not parsed_content or not parsed_content.get('text'):
                        continue
                    
                    # Build the document
                    title = page.get('title', 'Untitled')
                    space = page.get('_expandable', {}).get('space', '').split('/')[-1] if '_expandable' in page and 'space' in page['_expandable'] else ""
                    
                    # Build URL for source reference
                    if space:
                        source_url = f"{self.client.base_url}/display/{space}/{page_id}"
                    else:
                        source_url = f"{self.client.base_url}/pages/viewpage.action?pageId={page_id}"
                    
                    # Extract labels if available
                    labels = []
                    if 'metadata' in page and 'labels' in page['metadata']:
                        labels = [label.get('name') for label in page['metadata']['labels'].get('results', [])]
                    
                    document = {
                        'text': parsed_content['text'],
                        'metadata': {
                            'id': page_id,
                            'title': title,
                            'source': f"Confluence: {title}",
                            'source_link': source_url,
                            'source_type': 'confluence',
                            'space': space,
                            'labels': labels,
                            'last_updated': page.get('version', {}).get('when', '')
                        }
                    }
                    
                    documents.append(document)
                    
                    # Log progress for large retrievals
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(all_pages)} Confluence pages")
                        
                except Exception as e:
                    logger.error(f"Error processing Confluence page {page.get('id')}: {str(e)}")
            
            # Cache the results
            cache_manager.set(cache_key, documents, timeout=3600 * 4)  # Cache for 4 hours
            
            logger.info(f"Retrieved {len(documents)} documents from Confluence")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving Confluence documents: {str(e)}")
            return []
    
    def search_documents(self, query, max_results=20):
        """
        Search for documents in Confluence.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of processed documents
        """
        logger.info(f"Searching Confluence for: {query}")
        
        try:
            # Use CQL for searching
            cql = f'text ~ "{query}"'
            
            # Add content type filter
            cql += ' AND type = "page"'
            
            # Search for content
            search_results = self.client.search_content(
                cql=cql,
                max_results=max_results,
                expand="body.storage,metadata.labels"
            )
            
            if not search_results or 'results' not in search_results:
                logger.info("No Confluence search results found")
                return []
            
            # Process each search result
            documents = []
            
            for result in search_results['results']:
                try:
                    # Extract and process the content
                    parsed_content = content_parser.parse_confluence_content(result)
                    
                    if not parsed_content or not parsed_content.get('text'):
                        continue
                    
                    # Build the document
                    page_id = result.get('id')
                    title = result.get('title', 'Untitled')
                    space = result.get('_expandable', {}).get('space', '').split('/')[-1] if '_expandable' in result and 'space' in result['_expandable'] else ""
                    
                    # Build URL for source reference
                    if space:
                        source_url = f"{self.client.base_url}/display/{space}/{page_id}"
                    else:
                        source_url = f"{self.client.base_url}/pages/viewpage.action?pageId={page_id}"
                    
                    # Extract labels if available
                    labels = []
                    if 'metadata' in result and 'labels' in result['metadata']:
                        labels = [label.get('name') for label in result['metadata']['labels'].get('results', [])]
                    
                    document = {
                        'text': parsed_content['text'],
                        'metadata': {
                            'id': page_id,
                            'title': title,
                            'source': f"Confluence: {title}",
                            'source_link': source_url,
                            'source_type': 'confluence',
                            'space': space,
                            'labels': labels,
                            'last_updated': result.get('version', {}).get('when', '')
                        }
                    }
                    
                    documents.append(document)
                    
                except Exception as e:
                    logger.error(f"Error processing Confluence search result {result.get('id')}: {str(e)}")
            
            logger.info(f"Retrieved {len(documents)} search results from Confluence")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching Confluence documents: {str(e)}")
            return []

# Initialize the Confluence connector with default configuration
confluence_connector = ConfluenceConnector(
    base_url=config.CONFLUENCE_URL,
    username=config.CONFLUENCE_USERNAME,
    api_token=config.CONFLUENCE_TOKEN,
    ssl_verify=True
)






















"""
Test script for the RAG system to verify the fixes work.
"""
import sys
import logging
import os
from utils.logger import get_logger
import config
from rag_engine.processor import rag_processor
from rag_engine.embedding import embedding_manager
from rag_engine.retrieval import retrieval_engine
from data_sources.confluence.connector import confluence_connector

# Configure logging to also output to console
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = get_logger(__name__)

def test_embedding():
    """Test the embedding module."""
    logger.info("==== Testing Embedding Module ====")
    
    # Test embedding generation
    test_texts = [
        "This is a test document about RAG systems.",
        "Retrieval Augmented Generation combines search with LLMs.",
        "Confluence is a collaboration tool for documentation."
    ]
    
    logger.info(f"Generating embeddings for {len(test_texts)} test texts")
    embeddings = embedding_manager.get_embeddings(test_texts)
    
    if embeddings and len(embeddings) == len(test_texts):
        logger.info(f"✅ Successfully generated {len(embeddings)} embeddings")
        logger.info(f"   Embedding dimension: {len(embeddings[0])}")
    else:
        logger.error("❌ Failed to generate embeddings")
    
    return embeddings is not None and len(embeddings) == len(test_texts)

def test_retrieval(sample_documents=None):
    """
    Test the retrieval module.
    
    Args:
        sample_documents: Optional sample documents to index
    """
    logger.info("==== Testing Retrieval Module ====")
    
    # Create some test documents if none provided
    if not sample_documents:
        sample_documents = [
            {
                'text': "RAG systems combine retrieval mechanisms with generative AI models. "
                        "They typically involve indexing documents, retrieving relevant information, "
                        "and then using an LLM to generate responses based on the retrieved context.",
                'metadata': {
                    'title': "RAG Overview",
                    'source': "Test Document 1",
                    'source_type': 'test'
                }
            },
            {
                'text': "Confluent API provides real-time data streaming capabilities. "
                        "It enables organizations to process and analyze large volumes of data as it is generated.",
                'metadata': {
                    'title': "Confluent Overview",
                    'source': "Test Document 2",
                    'source_type': 'test'
                }
            },
            {
                'text': "JIRA is a project management tool used for issue tracking and agile project management. "
                        "It helps teams plan, track, and manage software development projects.",
                'metadata': {
                    'title': "JIRA Overview",
                    'source': "Test Document 3",
                    'source_type': 'test'
                }
            }
        ]
    
    # Process documents
    logger.info(f"Processing {len(sample_documents)} sample documents")
    chunks = rag_processor.process_documents(sample_documents, source_type='test')
    
    if not chunks:
        logger.error("❌ Failed to process documents into chunks")
        return False
    
    logger.info(f"✅ Processed documents into {len(chunks)} chunks")
    
    # Index the chunks
    logger.info("Indexing chunks")
    rag_processor.index_documents(chunks, overwrite=True)
    
    # Test retrieval
    test_query = "What is RAG system?"
    logger.info(f"Testing retrieval with query: '{test_query}'")
    
    retrieved_chunks = rag_processor.retrieve_relevant_chunks(test_query)
    
    if retrieved_chunks:
        logger.info(f"✅ Successfully retrieved {len(retrieved_chunks)} chunks")
        # Show top result
        if retrieved_chunks:
            top_chunk = retrieved_chunks[0]
            logger.info(f"   Top chunk source: {top_chunk.get('source', 'Unknown')}")
            logger.info(f"   Top chunk score: {top_chunk.get('score', 0):.4f}")
            logger.info(f"   Content preview: {top_chunk.get('content', '')[:100]}...")
    else:
        logger.error("❌ Failed to retrieve chunks")
    
    return retrieved_chunks is not None and len(retrieved_chunks) > 0

def test_end_to_end():
    """Test the entire RAG pipeline end-to-end."""
    logger.info("==== Testing End-to-End RAG Pipeline ====")
    
    # Register Confluence connector
    rag_processor.register_document_source('confluence', confluence_connector)
    
    # Create some test documents
    sample_documents = [
        {
            'text': "RAG systems combine retrieval mechanisms with generative AI models. "
                    "They typically involve indexing documents, retrieving relevant information, "
                    "and then using an LLM to generate responses based on the retrieved context. "
                    "RAG is particularly useful for grounding LLM responses in factual information.",
            'metadata': {
                'title': "RAG Overview",
                'source': "Test Document",
                'source_type': 'test'
            }
        },
        {
            'text': "BMC Remedy is an IT service management tool that provides incident management, "
                    "problem management, change management, and IT asset management capabilities. "
                    "It helps organizations streamline their IT operations and service delivery processes.",
            'metadata': {
                'title': "Remedy Overview",
                'source': "Test Document",
                'source_type': 'test'
            }
        }
    ]
    
    # Process documents
    logger.info(f"Processing {len(sample_documents)} sample documents")
    chunks = rag_processor.process_documents(sample_documents, source_type='test')
    
    if not chunks:
        logger.error("❌ Failed to process documents into chunks")
        return False
    
    logger.info(f"✅ Processed documents into {len(chunks)} chunks")
    
    # Index the chunks
    logger.info("Indexing chunks")
    rag_processor.index_documents(chunks, overwrite=True)
    
    # Test query processing
    test_query = "What is Remedy and how does it relate to RAG?"
    logger.info(f"Testing query processing: '{test_query}'")
    
    result = rag_processor.process_query(test_query)
    
    if result and 'response' in result:
        logger.info(f"✅ Successfully generated response")
        logger.info(f"   Retrieved {len(result.get('context', []))} context chunks")
        logger.info(f"   Response: {result['response'][:150]}...")
    else:
        logger.error("❌ Failed to process query")
    
    return result is not None and 'response' in result

def run_tests():
    """Run all tests."""
    logger.info("Starting RAG system tests")
    
    test_results = {
        "embedding": test_embedding(),
        "retrieval": False,
        "end_to_end": False
    }
    
    # Only proceed with retrieval test if embedding test passed
    if test_results["embedding"]:
        test_results["retrieval"] = test_retrieval()
    
    # Only proceed with end-to-end test if retrieval test passed
    if test_results["retrieval"]:
        test_results["end_to_end"] = test_end_to_end()
    
    # Report results
    logger.info("==== Test Results ====")
    for test_name, result in test_results.items():
        logger.info(f"{test_name}: {'✅ Passed' if result else '❌ Failed'}")
    
    return all(test_results.values())

if __name__ == "__main__":
    success = run_tests()
    if success:
        logger.info("All tests passed successfully!")
        sys.exit(0)
    else:
        logger.error("Some tests failed. Check the logs for details.")
        sys.exit(1)








