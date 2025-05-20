#!/usr/bin/env python3
import requests
import logging
import os
import sys
import json
import re
import time
from datetime import datetime
from urllib.parse import quote
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.api_core.exceptions import GoogleAPICallError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("confluence_gemini.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ConfluenceGemini")

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "25"))

class HTMLFilter:
    """Simple HTML to text converter using regex."""
    @staticmethod
    def clean_html(html_text):
        """Remove HTML tags and normalize whitespace."""
        if not html_text:
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]*>', ' ', html_text)
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text).strip()
        # Handle common HTML entities
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        return text

class ConfluenceClient:
    """Client for Confluence REST API operations."""
    
    def __init__(self, base_url, username, api_token):
        """
        Initialize the Confluence client with authentication details.
        
        Args:
            base_url: The base URL of the Confluence instance
            username: The username for authentication
            api_token: The API token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.auth = (username, api_token)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.session = requests.Session()
        logger.info(f"Initialized Confluence client for {self.base_url}")
    
    def test_connection(self):
        """Test the connection to Confluence API."""
        try:
            logger.info("Testing connection to Confluence...")
            response = self.session.get(
                f"{self.base_url}/rest/api/content",
                auth=self.auth,
                headers=self.headers,
                params={"limit": 1},
                verify=False  # Setting SSL verify to False as requested
            )
            response.raise_for_status()
            
            if response.ok:
                logger.info("Connection successful!")
                return True
            return False
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_content_by_id(self, content_id, expand=None):
        """
        Get content by ID with optional expansion parameters.
        
        Args:
            content_id: The ID of the content to retrieve
            expand: Comma-separated list of properties to expand
        """
        try:
            params = {}
            if expand:
                params["expand"] = expand
            
            logger.info(f"Fetching content with ID: {content_id}")
            response = self.session.get(
                f"{self.base_url}/rest/api/content/{content_id}",
                auth=self.auth,
                headers=self.headers,
                params=params,
                verify=False
            )
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Error retrieving content: {str(e)}")
            return None
    
    def get_page_content(self, page_id):
        """
        Get the content of a page in a suitable format for NLP.
        
        Args:
            page_id: The ID of the page
        """
        try:
            page = self.get_content_by_id(page_id, expand="body.storage,metadata.labels")
            if not page:
                return None
            
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "url": f"{self.base_url}/wiki/spaces/{page.get('_expandable', {}).get('space', '').split('/')[-1]}/pages/{page.get('id')}",
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
            }
            
            # Get raw content
            content = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Process HTML content to plain text
            plain_text = HTMLFilter.clean_html(content)
            
            return {
                "metadata": metadata,
                "content": plain_text,
                "raw_html": content
            }
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None
    
    def search_content(self, query, content_type="page", limit=MAX_RESULTS):
        """
        Search for content using Confluence's built-in search.
        
        Args:
            query: Search query
            content_type: Type of content to search for
            limit: Maximum number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Create CQL query
            cql = f'type={content_type} AND text ~ "{query}"'
            
            logger.info(f"Searching with CQL: {cql}")
            response = self.session.get(
                f"{self.base_url}/rest/api/content/search",
                auth=self.auth,
                headers=self.headers,
                params={
                    "cql": cql,
                    "limit": limit
                },
                verify=False
            )
            response.raise_for_status()
            
            search_results = response.json()
            logger.info(f"Search returned {len(search_results.get('results', []))} results")
            
            # Get full content for each result
            detailed_results = []
            for result in search_results.get("results", []):
                content = self.get_page_content(result.get("id"))
                if content:
                    detailed_results.append(content)
            
            return detailed_results
        except Exception as e:
            logger.error(f"Error searching content: {str(e)}")
            return []

class TextProcessor:
    """Class for simple text processing without external dependencies."""
    
    @staticmethod
    def extract_keywords(text, top_n=5):
        """
        Extract potential keywords from text, using simple frequency analysis.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-z]{3,}\b', text)
        
        # Remove common stop words (simplified list)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                     'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
                     'about', 'like', 'through', 'over', 'before', 'after', 'between', 
                     'under', 'above', 'of', 'from', 'up', 'down', 'this', 'that', 'these', 
                     'those', 'it', 'its', 'he', 'she', 'we', 'they', 'them', 'their', 'has', 
                     'have', 'had', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should'}
        
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N words
        return [word for word, count in sorted_words[:top_n]]
    
    @staticmethod
    def is_multi_part_question(question):
        """
        Detect if a question has multiple parts.
        
        Args:
            question: The question text
            
        Returns:
            Boolean indicating if it's a multi-part question
        """
        # Patterns that suggest multi-part questions
        patterns = [
            r'\d+\s*\.\s+',  # Numbered list (1. 2. etc)
            r'first.*?second',
            r'part\s+\d+',
            r'multiple questions',
            r'several questions',
            r'and also',
            r';',
            r'\?.*?\?'  # Multiple question marks
        ]
        
        return any(re.search(pattern, question, re.IGNORECASE) for pattern in patterns)
    
    @staticmethod
    def split_multi_part_question(question):
        """
        Split a multi-part question into individual questions.
        
        Args:
            question: Multi-part question
            
        Returns:
            List of individual questions
        """
        # Split by question marks
        parts = re.split(r'\?\s+', question)
        
        # Make sure each part ends with a question mark
        for i, part in enumerate(parts):
            if i < len(parts) - 1 and not part.endswith('?'):
                parts[i] = part + '?'
        
        # Filter out empty parts
        parts = [part.strip() for part in parts if part.strip()]
        
        # If no good splits, try other methods
        if len(parts) <= 1:
            # Try numbered patterns
            numbered_parts = re.split(r'\d+\s*\.\s+', question)
            numbered_parts = [part.strip() for part in numbered_parts if part.strip()]
            
            if len(numbered_parts) > 1:
                parts = numbered_parts
        
        # If still no success, try other delimiters
        if len(parts) <= 1:
            for delimiter in [';', 'and also,', 'additionally,', 'moreover,', 'furthermore,']:
                if delimiter in question.lower():
                    parts = question.split(delimiter)
                    parts = [part.strip() for part in parts if part.strip()]
                    break
        
        # Default to the original question if all else fails
        if len(parts) <= 1:
            return [question]
        
        return parts

class GeminiClient:
    """Wrapper for interacting with Google's Gemini API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = GenerativeModel(MODEL_NAME)
        logger.info(f"Initialized Gemini client with model {MODEL_NAME}")
    
    def generate_response(self, prompt, system_prompt=None, temperature=0.7, max_tokens=8192):
        """
        Generate a response from Gemini.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            logger.info(f"Generating response with temperature {temperature}")
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=max_tokens,
            )
            
            # Combine prompts if system prompt is provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Generate response with streaming for better user experience
            response_text = ""
            print("Generating response", end="")
            for chunk in self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True,
            ):
                if chunk.candidates and chunk.candidates[0].text:
                    response_text += chunk.candidates[0].text
                    print(".", end="", flush=True)  # Show progress
            
            print()  # New line after progress dots
            
            logger.info(f"Response length: {len(response_text)} characters")
            return response_text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def build_system_prompt(self, context_docs=None):
        """
        Build a comprehensive system prompt for Gemini.
        
        Args:
            context_docs: List of documents to include as context
            
        Returns:
            System prompt string
        """
        system_prompt = """You are a highly knowledgeable and professional AI assistant that specializes in providing accurate information from a company's Confluence knowledge base. Your responses should be:

1. PRECISE AND ACCURATE: Always base your answers strictly on the provided Confluence documents. If the information isn't in the provided context, clearly state that you don't have that specific information.

2. PROFESSIONAL BUT FRIENDLY: Maintain a professional tone that would be appropriate in a corporate environment, while still being approachable and helpful.

3. WELL-STRUCTURED: For complex responses, use appropriate formatting with headings, bullet points, or numbered lists to improve readability. Use tables for tabular data.

4. CONTEXTUALLY AWARE: If you need clarification to provide a better answer, politely ask follow-up questions to better understand the user's needs.

5. SOURCE-TRANSPARENT: Always include references to the specific Confluence pages you used to answer the question. Include the exact page titles and URLs at the end of your response.

When responding to technical questions:
- Be precise with technical terminology
- Include code snippets when relevant
- Explain complex concepts clearly

When handling data from tables:
- Present numerical data clearly, using markdown tables if appropriate
- Explain what the data means in business terms

When images are referenced in the context:
- Clearly describe what information the image contains
- Explain the relevance of the image to the question

If the question seems ambiguous or could have multiple interpretations:
- Consider the most likely interpretation based on the available context
- If necessary, provide answers to multiple interpretations
- Politely ask for clarification

If you don't have enough information to fully answer the question:
- Clearly state what you do know based on the provided context
- Explain what additional information would be needed
- Suggest which Confluence spaces might contain the relevant information

FORMAT YOUR RESPONSES:
1. Start with a direct answer to the question
2. Follow with supporting details, explanations, or elaborations
3. For complex topics, use appropriate headings and structured formatting
4. End with source references listing the Confluence pages you used

Remember that you are assisting with company-internal information. Your responses should be helpful for employees trying to find and understand information in their company's knowledge base.
"""
        
        # Add context documents if provided
        if context_docs and len(context_docs) > 0:
            context_text = "\n\n### CONTEXT DOCUMENTS ###\n\n"
            
            for i, doc in enumerate(context_docs):
                metadata = doc.get("metadata", {})
                title = metadata.get("title", "Untitled Document")
                url = metadata.get("url", "")
                content = doc.get("content", "")
                
                # Truncate content if too long - Gemini has context limits
                if len(content) > 10000:
                    content = content[:10000] + "... [content truncated]"
                
                context_text += f"[DOCUMENT {i+1}]: {title}\n"
                context_text += f"URL: {url}\n"
                context_text += f"CONTENT: {content}\n\n"
            
            system_prompt += context_text
        
        return system_prompt

class ConfluenceGeminiBot:
    """Main class that integrates Confluence content with Gemini for answering questions."""
    
    def __init__(self, confluence_url, username, api_token):
        """
        Initialize the bot with Confluence credentials.
        
        Args:
            confluence_url: Base URL of Confluence instance
            username: Confluence username
            api_token: Confluence API token
        """
        self.confluence = ConfluenceClient(confluence_url, username, api_token)
        self.text_processor = TextProcessor()
        self.gemini = GeminiClient()
        
        # Test connection
        self.confluence.test_connection()
        logger.info("Initialized ConfluenceGeminiBot")
    
    def answer_question(self, question, temperature=0.7):
        """
        Answer a question using Confluence content and Gemini.
        
        Args:
            question: User's question
            temperature: Temperature for Gemini response generation
            
        Returns:
            Answer from Gemini
        """
        logger.info(f"Answering question: {question}")
        
        # Check if it's a multi-part question
        if self.text_processor.is_multi_part_question(question):
            logger.info("Detected multi-part question, splitting into parts")
            question_parts = self.text_processor.split_multi_part_question(question)
            
            if len(question_parts) > 1:
                logger.info(f"Split into {len(question_parts)} parts: {question_parts}")
                
                # Process each part separately
                responses = []
                for i, part in enumerate(question_parts):
                    logger.info(f"Processing question part {i+1}: {part}")
                    part_response = self._process_single_question(part, temperature)
                    responses.append(f"Part {i+1}: {part}\n\n{part_response}")
                
                # Combine responses
                combined_response = "\n\n" + "-" * 30 + "\n\n".join(responses)
                return combined_response
        
        # For single questions or if splitting failed
        return self._process_single_question(question, temperature)
    
    def _process_single_question(self, question, temperature=0.7):
        """
        Process a single question.
        
        Args:
            question: Question to answer
            temperature: Temperature for response generation
            
        Returns:
            Answer to the question
        """
        # Extract keywords for better search
        keywords = self.text_processor.extract_keywords(question)
        logger.info(f"Extracted keywords: {keywords}")
        
        # Create search query - use either keywords or original question
        search_query = " ".join(keywords) if keywords else question
        logger.info(f"Search query: {search_query}")
        
        # Search for relevant content using Confluence's built-in search
        relevant_docs = self.confluence.search_content(search_query)
        
        if not relevant_docs:
            logger.warning(f"No relevant documents found for question: {question}")
            return "I couldn't find any information in the Confluence knowledge base that answers your question. Could you rephrase your question or provide more details?"
        
        # Prepare context for Gemini
        logger.info(f"Found {len(relevant_docs)} relevant documents for the question")
        
        # Build system prompt with context
        system_prompt = self.gemini.build_system_prompt(context_docs=relevant_docs)
        
        # Generate response
        response = self.gemini.generate_response(
            prompt=question,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        return response
    
    def interactive_session(self):
        """Start an interactive Q&A session in the console."""
        print("\n=== Confluence Gemini Bot ===")
        print("Type 'exit' or 'quit' to end the session.")
        
        while True:
            try:
                question = input("\nYour question: ")
                
                if question.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                # Process the question
                print("\nSearching Confluence...")
                answer = self.answer_question(question)
                
                print("\nAnswer:")
                print("=" * 80)
                print(answer)
                print("=" * 80)
            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {str(e)}")
                print(f"An error occurred: {str(e)}")

# Main function to run the bot
def main():
    """Main function to run the application."""
    # Configuration
    confluence_url = os.environ.get("CONFLUENCE_URL", "https://your-company.atlassian.net")
    username = os.environ.get("CONFLUENCE_USERNAME", "your.email@company.com")
    api_token = os.environ.get("CONFLUENCE_API_TOKEN", "your-api-token")
    
    # Initialize the bot
    try:
        print("Initializing Confluence Gemini Bot...")
        bot = ConfluenceGeminiBot(confluence_url, username, api_token)
        
        # Start interactive session
        bot.interactive_session()
    except Exception as e:
        logger.error(f"Error initializing bot: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
